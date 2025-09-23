#!/bin/bash

# ========================================
# FedEXT 客户端特征数据导出脚本
# 基于已保存的客户端模型重新生成 feature-label 数据集
# ========================================

# ======== 基础路径设置 ========
RUN_DIR="./temp/Cifar10/FedEXT/base"   # 修改为实际的运行目录
OUTPUT_DIR=""                                   # 可选：覆盖默认输出目录 (默认 RUN_DIR/clients-feature)
KEEP_SPATIAL=0                                   # 1=保留卷积特征张量, 0=保存为扁平向量

# ======== 数据与模型配置 ========
DATASET="Cifar10"
NUM_CLIENTS=20
NUM_CLASSES=10
BATCH_SIZE=128
DEVICE="cuda"
SEED=0

# 如果想覆盖训练时的切分比例，设置 ENCODER_RATIO，例如 0.7；留空则沿用模型内记录
ENCODER_RATIO=""

# 可选：数据分布配置（需与训练时一致）
DISTRIBUTION_CONFIG=""

# ========================================
# 参数检查
# ========================================

if [ ! -d "$RUN_DIR" ]; then
    echo "[错误] RUN_DIR 无效：$RUN_DIR"
    echo "请填写包含 Client_x_model.pt 的运行目录。"
    exit 1
fi

if [ -n "$OUTPUT_DIR" ] && [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    echo "[错误] OUTPUT_DIR 的上级目录不存在：$OUTPUT_DIR"
    exit 1
fi

echo "========================================="
echo "启动客户端特征生成"
echo "运行目录      : $RUN_DIR"
echo "数据集        : $DATASET"
echo "客户端数量    : $NUM_CLIENTS"
echo "批次大小      : $BATCH_SIZE"
echo "设备          : $DEVICE"
if [ -n "$ENCODER_RATIO" ]; then
    echo "覆盖 Encoder 比例: $ENCODER_RATIO"
else
    echo "Encoder 比例  : 使用模型内部设置"
fi
if [ -n "$DISTRIBUTION_CONFIG" ]; then
    echo "数据分布配置  : $DISTRIBUTION_CONFIG"
fi
echo "输出目录      : ${OUTPUT_DIR:-默认 (RUN_DIR/clients-feature)}"
echo "保留卷积特征  : $( [ "$KEEP_SPATIAL" -eq 1 ] && echo 是 || echo 否 )"
echo "随机种子      : $SEED"
echo "========================================="

# ========================================
# 构建命令
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/generate_datasets.py"

CMD="python -u \"$PY_SCRIPT\""
CMD="$CMD --run-dir \"$RUN_DIR\""
CMD="$CMD --dataset \"$DATASET\""
CMD="$CMD --num-clients $NUM_CLIENTS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --device $DEVICE"
CMD="$CMD --num-classes $NUM_CLASSES"
CMD="$CMD --seed $SEED"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

if [ -n "$ENCODER_RATIO" ]; then
    CMD="$CMD --encoder-ratio $ENCODER_RATIO"
fi

if [ -n "$DISTRIBUTION_CONFIG" ]; then
    CMD="$CMD --distribution-config \"$DISTRIBUTION_CONFIG\""
fi

if [ "$KEEP_SPATIAL" -eq 1 ]; then
    CMD="$CMD --keep-spatial"
fi

# ========================================
# 执行命令
# ========================================

eval $CMD
