# 无条件扩散模型训练 - Mask 生成

这个项目使用 Hugging Face Diffusers 训练一个无条件扩散模型来生成细胞 mask 图像。

## 文件说明

- `setup_diffusion_env.sh`: 环境设置脚本
- `prepare_training_data.py`: 准备训练数据（从 mask 目录选择 10000 张图片）
- `train_unconditional.py`: 训练脚本
- `generate_masks.py`: 生成脚本
- `run_training.sh`: 一键运行训练的脚本

## 使用步骤

### 1. 设置环境

```bash
# 运行环境设置脚本
bash setup_diffusion_env.sh

# 激活环境
conda activate diffusion_mask
```

### 2. 配置 Accelerate（首次使用）

```bash
accelerate config
```

选择配置：
- 计算环境: This machine
- 使用多少个进程: 1（单GPU）或更多（多GPU）
- 使用 DeepSpeed: No
- 使用 FullyShardedDataParallel: No
- 混合精度: fp16

### 3. 准备训练数据并开始训练

```bash
# 一键运行（包括数据准备和训练）
bash run_training.sh
```

或者分步运行：

```bash
# 步骤 1: 准备训练数据
python prepare_training_data.py \
    --source_dir /research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/mask \
    --output_dir mask_train_10k \
    --num_samples 10000

# 步骤 2: 训练模型
accelerate launch train_unconditional.py \
    --train_data_dir=mask_train_10k \
    --resolution=256 \
    --output_dir=uncond-ddpm-mask-256 \
    --train_batch_size=16 \
    --num_epochs=50 \
    --learning_rate=1e-4 \
    --lr_warmup_steps=500 \
    --use_ema \
    --mixed_precision="fp16" \
    --save_images_epochs=5 \
    --save_model_epochs=10
```

### 4. 生成新的 mask 图像

```bash
# 使用训练好的模型生成图像
python generate_masks.py \
    --model_path uncond-ddpm-mask-256/checkpoint-epoch-49 \
    --output_dir generated_masks \
    --num_images 1000 \
    --batch_size=4 \
    --num_inference_steps=50
```

## 训练参数说明

- `--train_data_dir`: 训练数据目录
- `--resolution`: 图像分辨率（默认 256）
- `--train_batch_size`: 批次大小（根据 GPU 内存调整）
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--lr_warmup_steps`: 学习率预热步数
- `--use_ema`: 使用指数移动平均（提高生成质量）
- `--mixed_precision`: 混合精度训练（fp16 可以节省内存）
- `--save_images_epochs`: 每隔多少轮保存生成的样本图像
- `--save_model_epochs`: 每隔多少轮保存模型检查点

## 输出结构

```
uncond-ddpm-mask-256/
├── checkpoint-epoch-9/      # 第 10 轮的模型检查点
├── checkpoint-epoch-19/     # 第 20 轮的模型检查点
├── checkpoint-epoch-29/     # 第 30 轮的模型检查点
├── checkpoint-epoch-39/     # 第 40 轮的模型检查点
├── checkpoint-epoch-49/     # 第 50 轮的模型检查点（最终模型）
├── samples/
│   ├── epoch_4/            # 第 5 轮生成的样本
│   ├── epoch_9/            # 第 10 轮生成的样本
│   └── ...
└── logs/                   # TensorBoard 日志
```

## 监控训练

```bash
# 使用 TensorBoard 监控训练
tensorboard --logdir uncond-ddpm-mask-256/logs
```

## 注意事项

1. **GPU 内存**: 如果遇到 OOM 错误，可以：
   - 减小 `--train_batch_size`
   - 增加 `--gradient_accumulation_steps`
   - 使用更小的 `--resolution`

2. **训练时间**: 在单个 GPU 上训练 50 轮可能需要几个小时到一天，取决于 GPU 性能

3. **生成质量**: 
   - 前期生成的图像质量较差，随着训练进行会逐渐改善
   - 使用 `--use_ema` 可以提高生成质量
   - 可以尝试不同的 `--num_inference_steps`（更多步数 = 更好质量但更慢）

4. **数据增强**: 训练脚本已包含随机水平翻转，可以根据需要添加更多增强

## 故障排除

### 问题: ImportError: cannot import name 'DDPMPipeline'

解决方案:
```bash
pip install --upgrade diffusers
```

### 问题: CUDA out of memory

解决方案:
```bash
# 减小批次大小
--train_batch_size=8  # 或更小
```

### 问题: 训练速度太慢

解决方案:
- 确保使用了 `--mixed_precision="fp16"`
- 检查是否正确使用了 GPU
- 考虑使用多 GPU 训练
