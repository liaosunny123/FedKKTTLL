#!/bin/bash

# ========================================
# FedKTL-Stable-Diffusion 实验配置脚本
# ========================================

# 基础实验设置
TIMES=3                    # 实验运行次数
AUTO_BREAK=1              # 自动停止 (1=True, 0=False)
DEVICE_ID=1               # GPU设备ID

# 数据集和模型配置
DATASET="Cifar10"         # 数据集 (Cifar10, MNIST, etc.)
MODEL_FAMILY="HtM10"      # 模型族 (HtM10包含10种异构模型)
IS_HOMOGENEITY_MODEL=1    # 是否使用同构模型 (1=所有客户端使用相同模型, 0=使用异构模型族)
NUM_CLASSES=10            # 分类数量
FEATURE_DIM=512           # 特征维度

# 联邦学习参数
NUM_CLIENTS=3            # 客户端总数
JOIN_RATIO=1              # 每轮参与的客户端比例 (0.0-1.0)

# 本地训练参数
LOCAL_LR=0.01            # 本地学习率
LOCAL_BATCH_SIZE=10      # 本地批次大小
LOCAL_EPOCHS=1           # 本地训练轮次

# 算法配置
#ALGORITHM="FedKTL-stable-diffusion"  # 联邦学习算法
#GENERATOR_PATH="runwayml/stable-diffusion-v1-5"  # Stable Diffusion模型路径

ALGORITHM="FedKTL-stylegan-3"  # 联邦学习算法
GENERATOR_PATH="stylegan/stylegan-3-models/stylegan3-t-afhqv2-512x512.pkl"  # Stable Diffusion模型路径

# 服务器端参数
SERVER_LR=0.1            # 服务器学习率
SERVER_BATCH_SIZE=100    # 服务器批次大小
SERVER_EPOCHS=100        # 服务器训练轮次

# 算法调优参数
LAMBDA=0.01              # 正则化系数
MU=100                   # 损失权重系数

# 使用 wandb
USE_WANDB=True

# 数据分布配置文件（可选）
# 取消注释以下行以使用特定的数据分布配置
# DISTRIBUTION_CONFIG="../configs/distribution_fix_missing.json"
DISTRIBUTION_CONFIG="../configs/distribution_missing_classes.json"
# DISTRIBUTION_CONFIG="../configs/distribution_dirichlet.json"

# ========================================
# 运行命令
# ========================================

echo "========================================="
echo "启动 FedKTL-Stable-Diffusion 实验"
echo "数据集: $DATASET"
echo "模型族: $MODEL_FAMILY"
echo "同构模型: $IS_HOMOGENEITY_MODEL"
echo "客户端数量: $NUM_CLIENTS"
echo "算法: $ALGORITHM"
if [ ! -z "$DISTRIBUTION_CONFIG" ]; then
    echo "数据分布配置: $DISTRIBUTION_CONFIG"
fi
echo "========================================="

# 设置CUDA相关环境变量
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# RTX 4090架构
export TORCH_CUDA_ARCH_LIST="8.9"

# 确认设置
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" HF_ENDPOINT=https://hf-mirror.com python -u main.py \
  -t $TIMES \
  -ab $AUTO_BREAK \
  -lr $LOCAL_LR \
  -jr $JOIN_RATIO \
  -lbs $LOCAL_BATCH_SIZE \
  -ls $LOCAL_EPOCHS \
  -nc $NUM_CLIENTS \
  -nb $NUM_CLASSES \
  -data $DATASET \
  -m $MODEL_FAMILY \
  -hm $IS_HOMOGENEITY_MODEL \
  -fd $FEATURE_DIM \
  -did $DEVICE_ID \
  -algo $ALGORITHM \
  -slr $SERVER_LR \
  -sbs $SERVER_BATCH_SIZE \
  -se $SERVER_EPOCHS \
  -lam $LAMBDA \
  -mu $MU \
  -wb $USE_WANDB \
  -GPath $GENERATOR_PATH \
  ${DISTRIBUTION_CONFIG:+-dc "$DISTRIBUTION_CONFIG"}