#!/bin/bash

# =============================================================================
# 快速多进程数据集准备脚本
# Fast Multi-Process Dataset Preparation Script
# =============================================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}=========================================="
echo -e "🚀 快速多进程数据集准备"
echo -e "Fast Multi-Process Dataset Preparation"
echo -e "==========================================${NC}"

# 工作目录
WORK_DIR="/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/pytorch-CycleGAN-and-pix2pix"

# 切换到工作目录
cd "$WORK_DIR"
echo -e "${GREEN}工作目录: $(pwd)${NC}"

# 检查CPU信息
CPU_COUNT=$(nproc)
echo -e "${CYAN}🖥️  CPU信息:${NC}"
echo -e "  CPU核心数: $CPU_COUNT"
echo -e "  推荐进程数: $(echo "scale=0; $CPU_COUNT * 0.8" | bc)"

# 检查数据集大小
echo -e "${YELLOW}=========================================="
echo -e "📊 数据集统计"
echo -e "==========================================${NC}"

IMAGE_COUNT=$(ls /research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/image/ 2>/dev/null | wc -l)
MASK_COUNT=$(ls /research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/mask/ 2>/dev/null | wc -l)

echo -e "${GREEN}原始数据:${NC}"
echo -e "  📷 图像文件: $IMAGE_COUNT"
echo -e "  🎭 掩码文件: $MASK_COUNT"

# 估算处理时间
TOTAL_FILES=$IMAGE_COUNT
ESTIMATED_TIME_SINGLE=$((TOTAL_FILES / 40))  # 单进程约40张/秒
ESTIMATED_TIME_MULTI=$((ESTIMATED_TIME_SINGLE / CPU_COUNT * 2))  # 多进程加速

echo -e "${BLUE}时间估算:${NC}"
echo -e "  ⏱️  单进程: ~${ESTIMATED_TIME_SINGLE}秒 ($(echo "scale=1; $ESTIMATED_TIME_SINGLE/60" | bc)分钟)"
echo -e "  🚀 多进程: ~${ESTIMATED_TIME_MULTI}秒 ($(echo "scale=1; $ESTIMATED_TIME_MULTI/60" | bc)分钟)"
echo -e "  ⚡ 加速比: ~$(echo "scale=1; $ESTIMATED_TIME_SINGLE/$ESTIMATED_TIME_MULTI" | bc)x"

echo ""

# 检查当前训练集状态
CURRENT_TRAIN_COUNT=$(ls datasets/cell_dataset/train/ 2>/dev/null | wc -l)
echo -e "${YELLOW}当前训练集: $CURRENT_TRAIN_COUNT 个文件${NC}"

if [ $CURRENT_TRAIN_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠️  训练集不为空，将被清空并重新生成${NC}"
    echo ""
    read -p "确认继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}用户取消操作${NC}"
        exit 0
    fi
fi

# 开始处理
echo -e "${GREEN}=========================================="
echo -e "🚀 开始多进程数据准备"
echo -e "Starting Multi-Process Data Preparation"
echo -e "==========================================${NC}"

# 记录开始时间
START_TIME=$(date +%s)

# 运行数据准备
echo -e "${BLUE}正在启动多进程数据准备...${NC}"
echo "1" | python cycle.py

# 记录结束时间
END_TIME=$(date +%s)
ACTUAL_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}=========================================="
echo -e "✅ 数据准备完成！"
echo -e "Data Preparation Completed!"
echo -e "==========================================${NC}"

# 统计最终结果
FINAL_TRAIN_COUNT=$(ls datasets/cell_dataset/train/ 2>/dev/null | wc -l)
FINAL_VAL_COUNT=$(ls datasets/cell_dataset/val/ 2>/dev/null | wc -l)
FINAL_TEST_COUNT=$(ls datasets/cell_dataset/test/ 2>/dev/null | wc -l)

echo -e "${CYAN}📊 最终统计:${NC}"
echo -e "  🚂 训练集: $FINAL_TRAIN_COUNT 个文件"
echo -e "  🔍 验证集: $FINAL_VAL_COUNT 个文件"
echo -e "  🧪 测试集: $FINAL_TEST_COUNT 个文件"
echo -e "  📁 总计: $((FINAL_TRAIN_COUNT + FINAL_VAL_COUNT + FINAL_TEST_COUNT)) 个文件"

echo -e "${BLUE}⏱️  性能统计:${NC}"
echo -e "  实际用时: ${ACTUAL_TIME}秒 ($(echo "scale=1; $ACTUAL_TIME/60" | bc)分钟)"
echo -e "  处理速度: $(echo "scale=1; $TOTAL_FILES/$ACTUAL_TIME" | bc) 张/秒"
echo -e "  预估准确性: $(echo "scale=1; $ESTIMATED_TIME_MULTI*100/$ACTUAL_TIME" | bc)%"

# 检查增强类型分布
echo -e "${YELLOW}🎨 增强类型分布:${NC}"
if [ $FINAL_TRAIN_COUNT -gt 0 ]; then
    echo -e "  分析训练集中的增强类型..."
    
    ELASTIC_COUNT=$(ls datasets/cell_dataset/train/ | grep "^elastic_" | wc -l)
    CONTRAST_COUNT=$(ls datasets/cell_dataset/train/ | grep "^contrast_" | wc -l)
    ROTATE_COUNT=$(ls datasets/cell_dataset/train/ | grep "^rotate" | wc -l)
    FLIP_COUNT=$(ls datasets/cell_dataset/train/ | grep "^flip_" | wc -l)
    SCALE_COUNT=$(ls datasets/cell_dataset/train/ | grep "^scale_" | wc -l)
    MORPH_COUNT=$(ls datasets/cell_dataset/train/ | grep "^morph_" | wc -l)
    ORIGINAL_COUNT=$(ls datasets/cell_dataset/train/ | grep -v -E "^(elastic_|contrast_|rotate|flip_|scale_|morph_)" | wc -l)
    
    echo -e "    📷 原始: $ORIGINAL_COUNT ($(echo "scale=1; $ORIGINAL_COUNT*100/$FINAL_TRAIN_COUNT" | bc)%)"
    echo -e "    🌊 弹性变形: $ELASTIC_COUNT ($(echo "scale=1; $ELASTIC_COUNT*100/$FINAL_TRAIN_COUNT" | bc)%)"
    echo -e "    🎨 对比度: $CONTRAST_COUNT ($(echo "scale=1; $CONTRAST_COUNT*100/$FINAL_TRAIN_COUNT" | bc)%)"
    echo -e "    🔄 旋转: $ROTATE_COUNT ($(echo "scale=1; $ROTATE_COUNT*100/$FINAL_TRAIN_COUNT" | bc)%)"
    echo -e "    🪞 翻转: $FLIP_COUNT ($(echo "scale=1; $FLIP_COUNT*100/$FINAL_TRAIN_COUNT" | bc)%)"
    echo -e "    📏 缩放: $SCALE_COUNT ($(echo "scale=1; $SCALE_COUNT*100/$FINAL_TRAIN_COUNT" | bc)%)"
    echo -e "    🔧 形态学: $MORPH_COUNT ($(echo "scale=1; $MORPH_COUNT*100/$FINAL_TRAIN_COUNT" | bc)%)"
fi

echo ""
echo -e "${GREEN}🎯 下一步建议:${NC}"
echo -e "  1. 运行 H200 优化训练: ${CYAN}./train_h200_optimized.sh${NC}"
echo -e "  2. 预期最大生成数量: ${YELLOW}$FINAL_TRAIN_COUNT 张 result_A${NC}"
echo -e "  3. H200 batch_size=128 训练时间: ${BLUE}~2-4小时${NC}"

echo -e "${PURPLE}=========================================="
echo -e "准备完成时间: $(date)"
echo -e "==========================================${NC}"
