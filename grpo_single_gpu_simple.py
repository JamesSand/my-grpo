"""
Simple GRPO implementation with ref model and training model on the same GPU.
No multiprocessing, no ref server, no deepspeed.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, re, random, time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Configuration
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
beta = 0.04
all_steps = 1000
Q_batch_size = 5
num_samples_per_prompt = 8  # num_pre_Q
train_batch_size = 8
save_steps = 200
clip_param = 0.2
learning_rate = 1e-6
gradient_accumulation_steps = 4
use_bf16 = False  # Set to True if your GPU supports bf16

# Generation parameters
temperature = 0.9
max_new_tokens = 700

print(f"Using device: {device}")
print(f"Using bf16: {use_bf16}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} 
       for x, y in zip(dataset['question'], dataset['answer'])]

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

def apply_chat_template(question):
    """Apply chat template to a question"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Reward functions
from math_verify import parse, verify, ExprExtractionConfig
    
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

def reward_format(item, answer):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = answer.count("<think>") + answer.count("</think>")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else -1


def get_per_token_logps(logits, input_ids):
    """Compute per-token log probabilities"""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def generate_samples(model, prompts_text, num_samples):
    """Generate multiple samples for each prompt"""
    all_answers = []
    all_answer_ids = []
    
    for prompt in prompts_text:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate multiple samples
        for _ in range(num_samples):
            with torch.no_grad():
                output = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Extract the generated answer
            answer_ids = output[0, prompt_ids.shape[1]:]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            all_answers.append(answer_text)
            all_answer_ids.append(answer_ids.cpu().tolist())
    
    return all_answers, all_answer_ids


def compute_ref_logps(ref_model, prompt_ids, answer_ids, prompt_length):
    """Compute reference model log probabilities"""
    with torch.no_grad():
        logits = ref_model(prompt_ids).logits
        logits = logits[:, :-1, :]
        input_ids = prompt_ids[:, 1:]
        per_token_logps = get_per_token_logps(logits, input_ids)
        # Only keep the completion part
        ref_logps = per_token_logps[:, prompt_length-1:]
    return ref_logps


def GRPO_step(model, batch, prompt_length):
    """Perform one GRPO training step"""
    inputs = batch['inputs'].to(device)
    advantages = batch['rewards'].to(device).unsqueeze(1)
    ref_per_token_logps = batch['ref_logps'].to(device)
    
    # Forward pass
    logits = model(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    
    # Compute per-token log probabilities
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length-1:]
    
    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    
    # Compute mask for completion tokens
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # PPO-style clipped loss
    if 'old_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['old_logps'].to(device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # For first iteration, use on-policy (no clipping needed)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
    
    # Add KL penalty
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    # Compute final loss
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
    
    return loss


def main():
    # Load models
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    print("Loading training model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        _attn_implementation="sdpa"
    ).to(device)
    model.train()
    
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        _attn_implementation="sdpa"
    ).to(device)
    ref_model.eval()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Optional: Setup gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' and not use_bf16 else None
    
    accumulated_steps = 0
    progress = tqdm(range(1, all_steps + 1), desc="Training")
    
    for step in progress:
        # Sample questions from dataset
        sampled_items = random.sample(QAs, Q_batch_size)
        
        # Prepare prompts
        prompts_text = [apply_chat_template(item['Q']) for item in sampled_items]
        
        # Generate answers using current policy
        print(f"\nStep {step}: Generating samples...")
        model.eval()
        with torch.no_grad():
            answers, answer_ids = generate_samples(model, prompts_text, num_samples_per_prompt)
        model.train()
        
        # Compute rewards
        rewards = []
        for i, item in enumerate(sampled_items):
            for j in range(num_samples_per_prompt):
                answer = answers[i * num_samples_per_prompt + j]
                reward = reward_correct(item, answer) + reward_format(item, answer)
                rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Process each prompt's samples
        for i, prompt_text in enumerate(prompts_text):
            # Get prompt ids
            prompt_ids_single = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
            prompt_length = prompt_ids_single.shape[1]
            
            # Get current prompt's answers and rewards
            curr_answer_ids = answer_ids[i * num_samples_per_prompt:(i + 1) * num_samples_per_prompt]
            curr_rewards = rewards[i * num_samples_per_prompt:(i + 1) * num_samples_per_prompt]
            
            # Skip if no variance in rewards
            if curr_rewards.max() - curr_rewards.min() < 1e-4:
                print(f"  Prompt {i}: All rewards same, skipping")
                continue
            
            # Normalize rewards (advantage estimation)
            curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-8)
            
            # Process in mini-batches
            for batch_start in range(0, num_samples_per_prompt, train_batch_size):
                batch_end = min(batch_start + train_batch_size, num_samples_per_prompt)
                batch_answer_ids = curr_answer_ids[batch_start:batch_end]
                batch_rewards = curr_rewards[batch_start:batch_end]
                
                # Prepare batch inputs
                tensor_list = [torch.tensor(ans_ids) for ans_ids in batch_answer_ids]
                answer_ids_padded = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)
                
                # Concatenate prompt and answers
                batch_size = answer_ids_padded.shape[0]
                prompt_ids_repeated = prompt_ids_single.repeat(batch_size, 1)
                merged_ids = torch.cat([prompt_ids_repeated, answer_ids_padded], dim=1).to(device)
                
                # Compute reference logps
                ref_logps = compute_ref_logps(ref_model, merged_ids, answer_ids_padded, prompt_length)
                
                # Prepare batch
                batch = {
                    'inputs': merged_ids,
                    'rewards': batch_rewards,
                    'ref_logps': ref_logps,
                }
                
                # Training step with optional automatic mixed precision
                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        loss = GRPO_step(model, batch, prompt_length)
                    loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    loss = GRPO_step(model, batch, prompt_length)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                
                accumulated_steps += 1
                
                # Update weights after gradient accumulation
                if accumulated_steps >= gradient_accumulation_steps:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    accumulated_steps = 0
        
        # Update progress bar
        if 'loss' in locals():
            progress.set_description(f"Loss: {loss.item() * gradient_accumulation_steps:.6f}")
        
        # Print sample answer
        if step % 5 == 0 and len(answers) > 0:
            print(f"\nSample answer: {answers[0][:200]}...")
        
        # Save checkpoint
        if step % save_steps == 0:
            print(f'\nSaving model at step {step}...')
            save_name = f"./checkpoint_step_{step}"
            os.makedirs(save_name, exist_ok=True)
            model.save_pretrained(save_name)
            tokenizer.save_pretrained(save_name)
            print(f'Model saved to {save_name}')
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
