# 不使用 DeepSpeed 的 GRPO 训练

## 文件说明

- `grpo_vllm_one_no_deepspeed.py` - 不使用 DeepSpeed 的版本
- `grpo_vllm_one.py` - 原始使用 DeepSpeed 的版本

## 主要改动

### 1. 移除 DeepSpeed 依赖
```python
# 原版本
import deepspeed
deepspeed.init_distributed()
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, ...)

# 新版本
# 直接使用 PyTorch
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

### 2. 配置变化
```python
# 替代 ds_config 的配置
learning_rate = 1e-6
gradient_accumulation_steps = 4
use_bf16 = False  # 根据你的 GPU 设置
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 3. 训练循环修改
- 移除了 DeepSpeed 的 `engine.backward()` 和 `engine.step()`
- 使用标准的 PyTorch `loss.backward()` 和 `optimizer.step()`
- 添加了梯度累积的手动实现
- 可选的混合精度训练支持 (fp16)

### 4. 保存模型
```python
# 新版本直接使用 model.save_pretrained()
state_dict = model.state_dict()
model.save_pretrained(save_name, state_dict=state_dict)
```

## 运行方式

### 原版本 (使用 DeepSpeed)
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed grpo_vllm_one.py
```

### 新版本 (不使用 DeepSpeed)
```bash
# 单 GPU 训练
CUDA_VISIBLE_DEVICES=0 python grpo_vllm_one_no_deepspeed.py

# 或者直接运行
python grpo_vllm_one_no_deepspeed.py
```

## 优缺点对比

### 不使用 DeepSpeed 的优点：
1. **简单** - 不需要安装和配置 DeepSpeed
2. **易调试** - 标准 PyTorch 代码，更容易理解和调试
3. **兼容性好** - 不依赖特定的 DeepSpeed 版本
4. **单 GPU 友好** - 适合单 GPU 训练

### 不使用 DeepSpeed 的缺点：
1. **显存效率** - 没有 ZeRO 优化，显存占用更高
2. **不支持多 GPU** - 当前版本只支持单 GPU 训练
3. **没有 CPU Offload** - 无法将优化器状态卸载到 CPU
4. **大模型困难** - 对于特别大的模型，可能无法在单 GPU 上训练

## 配置建议

### 如果你的 GPU 支持 bf16 (如 A100, H100, GH200):
```python
use_bf16 = True
```

### 如果你的 GPU 只支持 fp16 (如 V100):
```python
use_bf16 = False
# 代码会自动使用 fp16 混合精度训练
```

### 如果显存不足:
1. 减小 `train_batch_size`
2. 增加 `gradient_accumulation_steps`
3. 减小 `num_pre_Q`
4. 或者使用原版本的 DeepSpeed

### 调整学习率和优化器:
```python
learning_rate = 1e-6  # 可以调整
# 如果需要其他优化器，修改这一行：
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

## 注意事项

1. **gen_device 设置**: 确保 `gen_device` 不与训练使用的 GPU 冲突
2. **显存监控**: 使用 `nvidia-smi` 监控显存使用
3. **梯度累积**: `gradient_accumulation_steps` 可以根据显存情况调整
4. **模型更新频率**: `gen_update_steps` 控制多久更新一次 vLLM 的模型

## 常见问题

**Q: 显存不足怎么办？**
A: 减小 batch size 或增加 gradient accumulation steps

**Q: 可以多 GPU 训练吗？**
A: 当前版本不支持。如需多 GPU，建议使用原版的 DeepSpeed 版本或实现 DDP

**Q: 训练速度比 DeepSpeed 版本慢吗？**
A: 单 GPU 训练速度相近，但无法像 DeepSpeed 那样通过多 GPU 加速

**Q: 可以恢复训练吗？**
A: 需要手动添加 checkpoint 恢复逻辑，保存和加载 optimizer 的 state_dict
