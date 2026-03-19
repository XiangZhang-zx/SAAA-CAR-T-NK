#!/bin/bash

# =============================================================================
# 自动化创新版 Pix2Pix 训练脚本
# Automated Innovative Pix2Pix Training Script
# =============================================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo -e "创新版 Pix2Pix 自动训练脚本"
echo -e "Innovative Pix2Pix Auto Training Script"
echo -e "==========================================${NC}"

# 工作目录
WORK_DIR="/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/pytorch-CycleGAN-and-pix2pix"

# 检查目录是否存在
if [ ! -d "$WORK_DIR" ]; then
    echo -e "${RED}错误: 工作目录不存在: $WORK_DIR${NC}"
    exit 1
fi

# 切换到工作目录
cd "$WORK_DIR"
echo -e "${GREEN}切换到工作目录: $(pwd)${NC}"

# 检查cycle.py是否存在
if [ ! -f "cycle.py" ]; then
    echo -e "${RED}错误: 找不到 cycle.py 文件${NC}"
    exit 1
fi

# 创建日志文件
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
echo -e "${BLUE}日志文件: $LOG_FILE${NC}"

# 备份现有模型
if [ -d "checkpoints/innov_cell_pixpix" ]; then
    BACKUP_DIR="checkpoints/innov_cell_pixpix_backup_$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}备份现有模型到: $BACKUP_DIR${NC}"
    mv "checkpoints/innov_cell_pixpix" "$BACKUP_DIR"
fi

echo -e "${GREEN}=========================================="
echo -e "步骤 1: 开始创新版训练"
echo -e "Step 1: Starting Innovative Training"
echo -e "==========================================${NC}"

# 自动输入选项5 (创新训练)
echo "5" | python cycle.py 2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}训练失败，请检查日志: $LOG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}=========================================="
echo -e "步骤 2: 生成变体"
echo -e "Step 2: Generating Variants"
echo -e "==========================================${NC}"

# 检查模型是否存在
if [ ! -f "checkpoints/innov_cell_pixpix/latest_net_G.pth" ]; then
    echo -e "${RED}错误: 找不到训练好的模型文件${NC}"
    exit 1
fi

# 自动输入选项6 (生成变体) 和数量10
printf "6\n10\n" | python cycle.py 2>&1 | tee -a "$LOG_FILE"

# 检查生成是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}生成变体失败，请检查日志: $LOG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}=========================================="
echo -e "步骤 3: 检查结果"
echo -e "Step 3: Checking Results"
echo -e "==========================================${NC}"

# 查找结果目录
RESULT_DIRS=(
    "result_A/innov_cell_pixpix/test_latest/images"
    "cell_pix2pix_results/innovative"
    "results/innov_cell_pixpix"
)

FOUND_RESULTS=false

for dir in "${RESULT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        REAL_A_COUNT=$(find "$dir" -name "*_real_A.png" 2>/dev/null | wc -l)
        FAKE_B_COUNT=$(find "$dir" -name "*_fake_B.png" 2>/dev/null | wc -l)
        
        if [ $REAL_A_COUNT -gt 0 ] && [ $FAKE_B_COUNT -gt 0 ]; then
            echo -e "${GREEN}✓ 找到结果在: $dir${NC}"
            echo -e "  输入掩码 (*_real_A.png): $REAL_A_COUNT 个文件"
            echo -e "  生成细胞 (*_fake_B.png): $FAKE_B_COUNT 个文件"
            
            # 显示前5个文件
            echo -e "${BLUE}前5个生成的文件:${NC}"
            find "$dir" -name "*_real_A.png" | head -5 | while read file; do
                echo "  - $(basename "$file")"
            done
            
            FOUND_RESULTS=true
            break
        fi
    fi
done

if [ "$FOUND_RESULTS" = false ]; then
    echo -e "${YELLOW}警告: 未找到生成的结果文件${NC}"
fi

echo -e "${GREEN}=========================================="
echo -e "训练统计信息"
echo -e "Training Statistics"
echo -e "==========================================${NC}"

# 显示损失日志
LOSS_LOG="checkpoints/innov_cell_pixpix/loss_log.txt"
if [ -f "$LOSS_LOG" ]; then
    TOTAL_EPOCHS=$(grep -c "epoch" "$LOSS_LOG" 2>/dev/null || echo "未知")
    echo -e "训练轮数: $TOTAL_EPOCHS"
    
    echo -e "${BLUE}最后的损失值:${NC}"
    tail -5 "$LOSS_LOG"
fi

# 显示模型大小
MODEL_FILE="checkpoints/innov_cell_pixpix/latest_net_G.pth"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo -e "生成器模型大小: $MODEL_SIZE"
fi

echo -e "${GREEN}=========================================="
echo -e "完成！"
echo -e "COMPLETED!"
echo -e "==========================================${NC}"
echo -e "✓ 创新版模型训练完成"
echo -e "✓ 变体生成完成"
echo -e "✓ 结果保存在 result_A 目录"
echo -e "✓ 训练日志: $LOG_FILE"
echo ""
echo -e "${BLUE}下一步:${NC}"
echo -e "1. 检查 result_A/innov_cell_pixpix/test_latest/images/ 中的结果"
echo -e "2. *_real_A.png 是输入的掩码"
echo -e "3. *_fake_B.png 是生成的细胞图像"
echo ""
echo -e "${GREEN}训练完成时间: $(date)${NC}"
