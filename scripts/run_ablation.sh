#!/bin/bash
# SAAA 消融实验 — 在 trustai 上使用 GPU 2,3 执行
# 用法: bash run_ablation.sh [ablation_name]
#   不带参数 → 顺序跑所有 4 个消融
#   带参数 → 只跑指定的消融 (no_attention / no_perc / no_style / no_edge)

set -e

PIX_DIR="/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/pytorch-CycleGAN-and-pix2pix"
PYTHON="$PIX_DIR/Inov_env/bin/python"
# accelerate shebang 指向错误路径，改用 python -m accelerate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

cd "$PIX_DIR"

# 更新 accelerate config 为 2 GPU
cat > /tmp/ablation_accelerate.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# 通用训练参数（与 innov_cell_pixpix 完全一致，除了消融变量）
BASE_ARGS="--dataroot ./datasets/cell_dataset \
  --model pix2pix --netG unet_256 --norm batch --direction AtoB \
  --n_epochs 200 --n_epochs_decay 100 \
  --batch_size 384 --lr 0.0005 \
  --lambda_L1 100 \
  --use_gradient_checkpointing \
  --save_epoch_freq 50 --print_freq 100 \
  --display_id -1"

run_ablation() {
    local name=$1
    local extra_args=$2
    echo ""
    echo "========================================"
    echo "Starting ablation: $name"
    echo "Extra args: $extra_args"
    echo "Time: $(date)"
    echo "========================================"

    $PYTHON -m accelerate.commands.launch --config_file /tmp/ablation_accelerate.yaml \
        train_accelerate.py \
        $BASE_ARGS \
        --name "ablation_${name}" \
        $extra_args \
        2>&1 | tee "checkpoints/ablation_${name}_training.log"

    echo "Finished $name at $(date)"
}

# 定义 5 个消融实验（含 plain pix2pix baseline，Codex Review #7）
declare -A ABLATIONS
ABLATIONS[plain]="--lambda_perc 0 --lambda_style 0 --lambda_edge 0"
ABLATIONS[no_attention]="--lambda_perc 10.0 --lambda_style 1.0 --lambda_edge 5.0"
ABLATIONS[no_perc]="--use_attention --lambda_perc 0 --lambda_style 1.0 --lambda_edge 5.0"
ABLATIONS[no_style]="--use_attention --lambda_perc 10.0 --lambda_style 0 --lambda_edge 5.0"
ABLATIONS[no_edge]="--use_attention --lambda_perc 10.0 --lambda_style 1.0 --lambda_edge 0"

if [ -n "$1" ]; then
    # 只跑指定的消融
    if [ -z "${ABLATIONS[$1]}" ]; then
        echo "Unknown ablation: $1"
        echo "Available: ${!ABLATIONS[@]}"
        exit 1
    fi
    run_ablation "$1" "${ABLATIONS[$1]}"
else
    # 顺序跑所有消融
    for name in plain no_attention no_perc no_style no_edge; do
        run_ablation "$name" "${ABLATIONS[$name]}"
    done
    echo ""
    echo "All ablation experiments completed at $(date)"
fi
