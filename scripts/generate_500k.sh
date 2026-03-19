#!/bin/bash

echo "================================================================================"
echo "🚀 开始生成 500,000 个单细胞图像"
echo "================================================================================"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate Inov_env

# 切换到工作目录
cd /research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/pytorch-CycleGAN-and-pix2pix

# 输出目录
OUTPUT_DIR="./result_A"

echo "📁 输出目录: ${OUTPUT_DIR}"
echo "🎯 目标数量: 500,000"
echo "📦 模型: innov_cell_pixpix (latest)"
echo "🔧 Batch Size: 1024 (4 GPU)"
echo "================================================================================"

# 使用训练集作为数据源
TRAIN_DATAROOT="./datasets/cell_dataset/train"

# 检查训练集
if [ ! -d "$TRAIN_DATAROOT" ]; then
    echo "❌ 错误：训练集目录不存在: $TRAIN_DATAROOT"
    exit 1
fi

# 统计文件数
NUM_FILES=$(ls -1 "$TRAIN_DATAROOT" | wc -l)
echo "📊 训练集文件数: $NUM_FILES"

# 运行测试命令
echo ""
echo "🔥 开始生成..."
echo "================================================================================"

python test.py \
    --dataroot "$TRAIN_DATAROOT" \
    --name "cell_pix2pix_v200" \
    --model "test" \
    --netG "unet_256" \
    --norm "batch" \
    --direction "AtoB" \
    --num_test 500000 \
    --results_dir "$OUTPUT_DIR" \
    --batch_size 256 \
    --gpu_ids 0,1,2,3 \
    --no_dropout

echo ""
echo "================================================================================"
echo "✅ 生成完成！"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "================================================================================"

# 后处理：确保细胞完全覆盖 mask，边缘不触碰图像边界
echo ""
echo "================================================================================"
echo "🔧 开始后处理：填充 mask 边缘 + 清除图像边界"
echo "================================================================================"

RESULT_DIR="${OUTPUT_DIR}/cell_pix2pix_v200/test_latest/images"

if [ -d "$RESULT_DIR" ]; then
    python postprocess_cells.py \
        --input_dir "$RESULT_DIR" \
        --dilate_iter 3 \
        --border_size 2

    echo ""
    echo "================================================================================"
    echo "✅ 后处理完成！"
    echo "================================================================================"
else
    echo "⚠️  警告：结果目录不存在: $RESULT_DIR"
fi

