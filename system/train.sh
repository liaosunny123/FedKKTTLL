#!/bin/bash

# ========================================
# FedKTL-Stable-Diffusion 实验配置脚本
# ========================================

# 基础实验设置
TIMES=1                    # 实验运行次数
AUTO_BREAK=1              # 自动停止 (1=True, 0=False)
DEVICE_ID=1               # GPU设备ID

# 数据集和模型配置
DATASET="Cifar10"         # 数据集 (Cifar10, MNIST, etc.)
MODEL_FAMILY="HtM10"      # 模型族 (HtM10包含10种异构模型)
IS_HOMOGENEITY_MODEL=1    # 是否使用同构模型 (1=所有客户端使用相同模型, 0=使用异构模型族)
NUM_CLASSES=10            # 分类数量
FEATURE_DIM=512           # 特征维度

# 联邦学习参数
NUM_CLIENTS=6            # 客户端总数
JOIN_RATIO=1              # 每轮参与的客户端比例 (0.0-1.0)
GLOBAL_ROUNDS=100         # 全局训练轮次

# 本地训练参数
LOCAL_LR=0.01            # 本地学习率
LOCAL_BATCH_SIZE=10      # 本地批次大小
LOCAL_EPOCHS=5           # 本地训练轮次

# 算法配置
# 可选算法：FedAvg, FedEXT, FedKTL-stylegan-3, FedKTL-stylegan-xl, FedKTL-stable-diffusion
ALGORITHM="FedEXT"  # 联邦学习算法

# 生成器配置
# FedKTL-stable-diffusion:
#GENERATOR_PATH="runwayml/stable-diffusion-v1-5"
# FedKTL-stylegan-3:
#GENERATOR_PATH="stylegan/stylegan-3-models/stylegan3-t-afhqv2-512x512.pkl"
# FedKTL-stylegan-xl:
#GENERATOR_PATH="stylegan/stylegan-xl-models/stylegan_xl-afhqv2-512x512.pkl"
GENERATOR_PATH=""

# 服务器端参数
SERVER_LR=0.1            # 服务器学习率
SERVER_BATCH_SIZE=100    # 服务器批次大小
SERVER_EPOCHS=50        # 服务器训练轮次

# 算法调优参数
LAMBDA=0.01              # 正则化系数
MU=1                   # 损失权重系数

# FedEXT对比学习参数
CONTRASTIVE_WEIGHT=0.1   # 对比学习损失权重 (0.0=关闭对比学习, 0.1=推荐值)
CONTRASTIVE_TEMP=0.1     # 对比学习温度参数 (较小值使相似度计算更尖锐)

# ETF Classifier配置
USE_ETF=1                # 是否使用ETF分类器 (1=使用, 0=不使用)

# 全局模型配置
# FedAvg: 全局模型用于参数聚合
# FedKTL: 全局模型基于prototype训练
USE_GLOBAL_MODEL=1       # 是否使用全局模型 (1=使用, 0=不使用，仅在同构模型下有效)

# Prototype聚合配置 (FedProto-like)
# 当设置为1时，将在参数层面直接聚合prototype，绕过生成器
# 当设置为0时，使用原始的FedKTL生成器方法
IS_GLOBAL_MODEL_GENERATED_LOADER=0  # 是否使用prototype聚合 (1=使用prototype聚合, 0=使用生成器)

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
echo "全局轮次: $GLOBAL_ROUNDS"
echo "算法: $ALGORITHM"
echo "使用ETF分类器: $USE_ETF"
echo "使用全局模型: $USE_GLOBAL_MODEL"
if [[ "$ALGORITHM" == "FedKTL-"* ]]; then
    if [ "$IS_GLOBAL_MODEL_GENERATED_LOADER" == "1" ]; then
        echo "Prototype聚合模式: 是 (FedProto-like)"
    else
        echo "Prototype聚合模式: 否 (使用生成器)"
    fi
fi
if [ ! -z "$DISTRIBUTION_CONFIG" ]; then
    echo "数据分布配置: $DISTRIBUTION_CONFIG"
fi
if [ "$ALGORITHM" == "FedEXT" ]; then
    echo "对比学习权重: $CONTRASTIVE_WEIGHT"
    echo "对比学习温度: $CONTRASTIVE_TEMP"
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

# 导出环境变量供Python脚本使用
export IS_GLOBAL_MODEL_GENERATED_LOADER

# 构建运行命令
CMD="TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' HF_ENDPOINT=https://hf-mirror.com IS_GLOBAL_MODEL_GENERATED_LOADER=$IS_GLOBAL_MODEL_GENERATED_LOADER python -u main.py"
CMD="$CMD -t $TIMES"
CMD="$CMD -ab $AUTO_BREAK"
CMD="$CMD -lr $LOCAL_LR"
CMD="$CMD -jr $JOIN_RATIO"
CMD="$CMD -gr $GLOBAL_ROUNDS"
CMD="$CMD -lbs $LOCAL_BATCH_SIZE"
CMD="$CMD -ls $LOCAL_EPOCHS"
CMD="$CMD -nc $NUM_CLIENTS"
CMD="$CMD -nb $NUM_CLASSES"
CMD="$CMD -data $DATASET"
CMD="$CMD -m $MODEL_FAMILY"
CMD="$CMD -hm $IS_HOMOGENEITY_MODEL"
CMD="$CMD -fd $FEATURE_DIM"
CMD="$CMD -did $DEVICE_ID"
CMD="$CMD -algo $ALGORITHM"
CMD="$CMD -slr $SERVER_LR"
CMD="$CMD -sbs $SERVER_BATCH_SIZE"
CMD="$CMD -se $SERVER_EPOCHS"
CMD="$CMD -lam $LAMBDA"
CMD="$CMD -mu $MU"
CMD="$CMD -wb $USE_WANDB"
CMD="$CMD -etf $USE_ETF"
CMD="$CMD -gm $USE_GLOBAL_MODEL"

# 仅在使用FedKTL算法时添加生成器路径
if [[ "$ALGORITHM" == "FedKTL-"* ]]; then
    CMD="$CMD -GPath $GENERATOR_PATH"
fi

# 添加数据分布配置（如果存在）
if [ ! -z "$DISTRIBUTION_CONFIG" ]; then
    CMD="$CMD -dc $DISTRIBUTION_CONFIG"
fi

# 添加FedEXT对比学习参数（仅在使用FedEXT时）
if [ "$ALGORITHM" == "FedEXT" ]; then
    CMD="$CMD -cw $CONTRASTIVE_WEIGHT"
    CMD="$CMD -ct $CONTRASTIVE_TEMP"
fi

# 执行命令
eval $CMD