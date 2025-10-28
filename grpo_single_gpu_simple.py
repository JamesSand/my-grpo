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
import wandb

# Configuration
# model_path = "Qwen/Qwen3-0.6B"
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
beta = 0.01
all_steps = 1000
Q_batch_size = 5 # sampel 5 question for each steps
num_samples_per_prompt = 8  # num_pre_Q
train_batch_size = 8 # backward for each 8 samples
save_steps = 200
clip_param = 0.2
learning_rate = 1e-6
gradient_accumulation_steps = 4 # thise means we actually 
use_bf16 = False  # Set to True if your GPU supports bf16
use_wandb = True  # Set to True to enable wandb logging
wandb_project = "simple-grpo"  # wandb project name
wandb_run_name = None  # wandb run name, None for auto-generated

# Generation parameters
temperature = 0.9
max_new_tokens = 1024

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
    nums = re.findall(pattern, answer) # find all numbers in answer
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] #  use the last number in answer to compare with ground_truth
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else 0

def reward_format(item, answer):
    reasoning_word_list = ["since", "therefore", "thus", "because", "so", "hence", "perhaps", "let", "actually", "maybe", "assume", "suppose", "consider", "it follows that", "we have", "note that", "observe that"]
    
    lower_answer = answer.lower()
    
    reasoning_word_cnt = sum(lower_answer.count(word) for word in reasoning_word_list)
    
    if reasoning_word_cnt >= 10:
        return 1
    
    return 0
    
    # print(reasoning_word_cnt)
    
    # return 0
    
    
    # # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    # pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    # return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1


def get_per_token_logps(logits, input_ids):
    """Compute per-token log probabilities"""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def generate_samples(model, prompts_text, num_samples):
    """Generate multiple samples for each prompt and compute their log probabilities"""
    all_answers, all_answer_ids, all_old_logps = [], [], []
    debug_cnt = 0
    for prompt in prompts_text:
        print(f"gen {debug_cnt}"); debug_cnt += 1
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_length = prompt_ids.shape[1]
        
        with torch.no_grad():
            output = model.generate(
                prompt_ids.repeat(num_samples, 1),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Compute log probabilities for the generated sequences
            logits = model(output).logits
            logits = logits[:, :-1, :]
            input_ids = output[:, 1:]
            per_token_logps = get_per_token_logps(logits, input_ids)
            # Only keep the completion part
            per_token_logps = per_token_logps[:, prompt_length-1:]
        
        gen_part = output[:, prompt_ids.shape[1]:]
        for i in range(gen_part.size(0)):
            answer_ids = gen_part[i]
            all_answers.append(tokenizer.decode(answer_ids, skip_special_tokens=True))
            all_answer_ids.append(answer_ids.cpu().tolist())
            all_old_logps.append(per_token_logps[i].cpu())
    
    return all_answers, all_answer_ids, all_old_logps

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
    
    # Forward pass, calcualte pi(x)
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
    ratio = torch.exp(per_token_logps - batch['old_logps'].to(device))
    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    per_token_pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # KL penalty (per token)
    per_token_kl_loss = beta * per_token_kl
    
    # Combined loss per token
    per_token_loss = per_token_pg_loss + per_token_kl_loss
    
    # Compute final loss
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
    
    # Compute average pg_loss and kl_loss for logging
    avg_pg_loss = ((per_token_pg_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
    avg_kl_loss = ((per_token_kl_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
    
    return loss, avg_pg_loss.item(), avg_kl_loss.item()


def main():
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_path": model_path,
                "beta": beta,
                "all_steps": all_steps,
                "Q_batch_size": Q_batch_size,
                "num_samples_per_prompt": num_samples_per_prompt,
                "train_batch_size": train_batch_size,
                "clip_param": clip_param,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "use_bf16": use_bf16,
            }
        )
    
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
        # Calculate start and end index based on step
        start_idx = ((step - 1) * Q_batch_size) % len(QAs)
        end_idx = start_idx + Q_batch_size
        
        # Handle wrapping around the dataset
        if end_idx <= len(QAs):
            sampled_items = QAs[start_idx:end_idx]
        else:
            # Wrap around to the beginning
            sampled_items = QAs[start_idx:] + QAs[:end_idx - len(QAs)]
        
        # Prepare prompts
        prompts_text = [apply_chat_template(item['Q']) for item in sampled_items]
        
        # Generate answers using current policy
        print(f"\nStep {step}: Generating samples...")
        model.eval()
        with torch.no_grad():
            answers, answer_ids, old_logps = generate_samples(model, prompts_text, num_samples_per_prompt)
        model.train()
        
        # Compute rewards
        rewards = []
        correct_rewards = []
        format_rewards = []
        for i, item in enumerate(sampled_items):
            for j in range(num_samples_per_prompt):
                answer = answers[i * num_samples_per_prompt + j]
                r_correct = reward_correct(item, answer)
                r_format = reward_format(item, answer)
                reward = r_correct + r_format
                rewards.append(reward)
                correct_rewards.append(r_correct)
                format_rewards.append(r_format)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Global metrics for this step
        step_pg_losses = []
        step_kl_losses = []
        
        # Process each prompt's samples
        for i, prompt_text in enumerate(prompts_text):
            # Get prompt ids
            prompt_ids_single = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
            prompt_length = prompt_ids_single.shape[1]
            
            # Get current prompt's answers, rewards, and old logps
            curr_answer_ids = answer_ids[i * num_samples_per_prompt:(i + 1) * num_samples_per_prompt]
            curr_rewards = rewards[i * num_samples_per_prompt:(i + 1) * num_samples_per_prompt]
            curr_old_logps = old_logps[i * num_samples_per_prompt:(i + 1) * num_samples_per_prompt]
            
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
                batch_old_logps = curr_old_logps[batch_start:batch_end]
                
                # Prepare batch inputs
                tensor_list = [torch.tensor(ans_ids) for ans_ids in batch_answer_ids]
                answer_ids_padded = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)
                
                # Pad old_logps to match answer_ids_padded
                old_logps_padded = pad_sequence(batch_old_logps, batch_first=True, padding_value=0.0)
                
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
                    'old_logps': old_logps_padded,
                }
                
                # Training step with optional automatic mixed precision
                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        loss, pg_loss, kl_loss = GRPO_step(model, batch, prompt_length)
                    loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    loss, pg_loss, kl_loss = GRPO_step(model, batch, prompt_length)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                
                # Record metrics
                step_pg_losses.append(pg_loss)
                step_kl_losses.append(kl_loss)
                
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
        
        # Log to wandb
        if use_wandb and len(step_pg_losses) > 0:
            avg_correct_reward = np.mean(correct_rewards)
            avg_format_reward = np.mean(format_rewards)
            avg_pg_loss = np.mean(step_pg_losses)
            avg_kl_loss = np.mean(step_kl_losses)
            
            wandb.log({
                "reward/correct": avg_correct_reward,
                "reward/format": avg_format_reward,
                "loss/pg_loss": avg_pg_loss,
                "loss/kl_loss": avg_kl_loss,
                "loss/total": avg_pg_loss + avg_kl_loss,
                "step": step,
            }, step=step)
        
        # # Print sample answer
        # if step % 5 == 0 and len(answers) > 0:
        #     # print(f"\nSample answer: {answers[0][:200]}...")
            
        print(f"\nSample answer: {answers[0]}")
        
        # Save checkpoint
        if step % save_steps == 0:
            print(f'\nSaving model at step {step}...')
            save_name = f"./checkpoint_step_{step}"
            os.makedirs(save_name, exist_ok=True)
            model.save_pretrained(save_name)
            tokenizer.save_pretrained(save_name)
            print(f'Model saved to {save_name}')
    
    print("\nTraining completed!")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
