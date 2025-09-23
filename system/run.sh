#!/bin/bash

# ========================================
# FedEXT 大模型尾部独立训练脚本
# 使用 clients-feature 数据快速调试全局分类器
# ========================================

# ======== 基础路径配置 ========
DATASET_DIR="../clients-feature"  # 修改为实际的 clients-feature 目录
SAVE_PATH=""                      # 可选：保存训练好的分类器 (例如 tail_resnet34.pt)

# ======== 模型与切分设置 ========
MODEL_EXPR="torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)"
ENCODER_RATIO=0.7           # 与联邦训练时保持一致，必要时可调整

# ======== 训练超参数 ========
BATCH_SIZE=64
EPOCHS=20
LEARNING_RATE=0.01
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
DEVICE="cuda"               # 根据需要改为 cpu
USE_BALANCED_TEST=1         # 1=使用平衡测试集，0=使用完整测试集
FORCE_LINEAR_PROJECTION=0  # 1=保留卷积尾部并用线性层映射输入维度

# ======== WandB 配置 ========
USE_WANDB=1                 # 1=开启日志，0=关闭
WANDB_PROJECT="fedktl"
WANDB_ENTITY="epicmo"
WANDB_RUN_NAME="resnet34-tail-debug"

# ========================================
# 参数检查与提示
# ========================================

if [ ! -d "$DATASET_DIR" ]; then
    echo "[错误] DATASET_DIR 无效：$DATASET_DIR"
    echo "请先确认 FedEXT 训练后的 clients-feature 目录路径。"
    exit 1
fi

echo "========================================="
echo "启动 FedEXT 后端分类器独立训练"
echo "特征数据路径: $DATASET_DIR"
echo "模型表达式  : $MODEL_EXPR"
echo "Encoder比例 : $ENCODER_RATIO"
echo "批次大小    : $BATCH_SIZE"
echo "总轮次      : $EPOCHS"
echo "学习率      : $LEARNING_RATE"
echo "设备        : $DEVICE"
echo "线性映射尾部 : $( [ "$FORCE_LINEAR_PROJECTION" -eq 1 ] && echo 是 || echo 否 )"
if [ "$USE_WANDB" -eq 1 ]; then
    echo "WandB 项目  : $WANDB_PROJECT/$WANDB_ENTITY"
    echo "运行名称    : $WANDB_RUN_NAME"
else
    echo "WandB       : 已关闭"
fi
echo "========================================="

# ========================================
# 构建命令
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/train_resnet_tail.py"

CMD="python -u \"$PYTHON_SCRIPT\""
CMD="$CMD --dataset-dir \"$DATASET_DIR\""
CMD="$CMD --model \"$MODEL_EXPR\""
CMD="$CMD --encoder-ratio $ENCODER_RATIO"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --momentum $MOMENTUM"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --device $DEVICE"

if [ "$USE_BALANCED_TEST" -eq 1 ]; then
    CMD="$CMD --use-balanced-test"
fi

if [ "$USE_WANDB" -eq 1 ]; then
    CMD="$CMD --use-wandb --wandb-project \"$WANDB_PROJECT\" --wandb-entity \"$WANDB_ENTITY\" --wandb-run-name \"$WANDB_RUN_NAME\""
fi

if [ -n "$SAVE_PATH" ]; then
    CMD="$CMD --save-path \"$SAVE_PATH\""
fi

if [ "$FORCE_LINEAR_PROJECTION" -eq 1 ]; then
    CMD="$CMD --force-linear-projection"
fi

# ========================================
# 执行命令
# ========================================

eval $CMD
