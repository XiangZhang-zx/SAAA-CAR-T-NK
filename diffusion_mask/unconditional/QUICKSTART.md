# 快速开始指南 - 训练 Mask 生成模型

## 🚀 一键启动（推荐）

如果你已经有 conda 环境和必要的包，可以直接运行：

```bash
# 1. 进入项目目录
cd /research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset

# 2. 一键运行（包括数据准备和训练）
bash run_training.sh
```

## 📋 完整步骤

### 步骤 1: 设置环境

```bash
# 运行环境设置脚本
bash setup_diffusion_env.sh

# 激活环境
conda activate diffusion_mask
```

### 步骤 2: 配置 Accelerate（首次使用必须）

```bash
accelerate config
```

**推荐配置**：
- Compute environment: `This machine`
- Number of processes: `1` (单GPU) 或 GPU数量 (多GPU)
- Use DeepSpeed: `no`
- Use FullyShardedDataParallel: `no`
- Mixed precision: `fp16`

### 步骤 3: 开始训练

```bash
bash run_training.sh
```

这个脚本会：
1. 从 141685 张 mask 图像中随机选择 10000 张用于训练
2. 启动训练（50 轮，每 5 轮保存样本，每 10 轮保存模型）

### 步骤 4: 监控训练（可选）

在另一个终端中运行：

```bash
tensorboard --logdir uncond-ddpm-mask-256/logs
```

然后在浏览器中打开 `http://localhost:6006`

### 步骤 5: 生成新的 mask 图像

训练完成后，使用最终模型生成图像：

```bash
python generate_masks.py \
    --model_path uncond-ddpm-mask-256/checkpoint-epoch-49 \
    --output_dir generated_masks \
    --num_images 1000 \
    --batch_size=4
```

## ⚙️ 自定义训练参数

如果需要调整训练参数，编辑 `run_training.sh` 文件：

```bash
# 常用调整：
--train_batch_size=16      # GPU 内存不足时减小
--num_epochs=50            # 增加训练轮数
--resolution=256           # 改变图像分辨率
--learning_rate=1e-4       # 调整学习率
```

## 📊 预期结果

- **训练时间**: 约 4-8 小时（取决于 GPU）
- **模型大小**: 约 500MB 每个检查点
- **生成速度**: 约 1-2 秒/张（50 步推理）

## 🔍 检查训练进度

```bash
# 查看生成的样本图像
ls -lh uncond-ddpm-mask-256/samples/

# 查看保存的模型检查点
ls -lh uncond-ddpm-mask-256/checkpoint-*/
```

## ❓ 常见问题

### Q: CUDA out of memory 错误
A: 减小批次大小：`--train_batch_size=8` 或 `--train_batch_size=4`

### Q: 训练太慢
A: 确保使用了 `--mixed_precision="fp16"` 和 GPU

### Q: 生成的图像质量不好
A: 
- 训练更多轮次
- 使用 `--use_ema` 参数
- 增加推理步数：`--num_inference_steps=100`

## 📁 项目文件

```
.
├── setup_diffusion_env.sh      # 环境设置
├── prepare_training_data.py    # 数据准备
├── train_unconditional.py      # 训练脚本
├── generate_masks.py           # 生成脚本
├── run_training.sh             # 一键运行
├── README_DIFFUSION.md         # 详细文档
└── QUICKSTART.md              # 本文件
```

## 🎯 下一步

训练完成后，你可以：
1. 使用生成的 mask 训练 pix2pix 模型
2. 调整参数重新训练以提高质量
3. 生成更多样化的 mask 数据集
