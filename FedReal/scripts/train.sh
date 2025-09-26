#!/usr/bin/env bash
# End-to-end pipeline: launch FedEXT training, export embeddings, train tail classifier.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

# ==========================
# (1) 联邦训练配置
# ==========================
ALGORITHM="FedEXT"      # 可选：FedEXT / FedAvg
DATASET="Cifar10"
NUM_CLASSES=10
NUM_CLIENTS=20
ROUNDS=20
LOCAL_EPOCHS=3
BATCH_SIZE=10
LR=0.005
MOMENTUM=0.9
SAMPLE_FRACTION=1.0
SEED=42
MODEL_NAME="resnet18"
FEATURE_DIM=512

# FedEXT 相关参数（仅在 ALGORITHM=FedEXT 时使用）
FED_EXT_ENCODER_RATIO=0.2

case "${ALGORITHM}" in
  "FedAvg")
    ENCODER_RATIO=1.0
    ;;
  "FedEXT")
    ENCODER_RATIO=${FED_EXT_ENCODER_RATIO}
    ;;
  *)
    echo "[Pipeline] 未知算法 ${ALGORITHM}，仅支持 FedEXT 或 FedAvg" >&2
    exit 1
    ;;
esac

SERVER_BIND="0.0.0.0:50052"
SERVER_ADDR="127.0.0.1:50052"
SERVER_DEVICE="cuda"
CLIENT_DEVICE="cuda"
CLIENT_GPUS="0"
SERVER_WARMUP_SEC=5
CLIENT_STAGGER_SEC=0.3

# ==========================
# (2) 特征导出配置 (system/generate_datasets.py)
# ==========================
GEN_BATCH_SIZE=128
GEN_DEVICE="cuda"
GEN_SEED=0
GEN_KEEP_SPATIAL=1

# ==========================
# (3) 尾部分类器训练配置 (system/train_resnet_tail.py)
# ==========================
TAIL_MODEL_EXPR="torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)"
TAIL_BATCH_SIZE=64
TAIL_EPOCHS=20
TAIL_LR=0.01
TAIL_MOMENTUM=0.9
TAIL_WEIGHT_DECAY=1e-4
TAIL_DEVICE="cuda"
TAIL_FORCE_LINEAR_PROJECTION=1
TAIL_USE_BALANCED_TEST=1
TAIL_USE_WANDB=1
TAIL_WANDB_PROJECT="fedktl"
TAIL_WANDB_ENTITY="epicmo"
TAIL_WANDB_RUN_NAME="resnet34-tail-debug"

# ==========================
# WandB 控制
# ==========================
USE_WANDB=1
WANDB_PROJECT="fedktl"
WANDB_ENTITY="epicmo"
WANDB_RUN_NAME_PREFIX="fedext"

# ==========================
# 目录准备
# ==========================
DATA_ROOT="${REPO_ROOT}/dataset"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${PROJECT_ROOT}/runs/${DATASET}/${ALGORITHM}/${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}-${TIMESTAMP}"
  TAIL_USE_WANDB=1
  TAIL_WANDB_PROJECT="${WANDB_PROJECT}"
  TAIL_WANDB_ENTITY="${WANDB_ENTITY}"
  TAIL_WANDB_RUN_NAME="${WANDB_RUN_NAME}-tail"
  SERVER_WARMUP_SEC=15
else
  WANDB_RUN_NAME=""
fi

echo "========================================="
echo " FedReal :: FedEXT Pipeline"
echo "-----------------------------------------"
echo " dataset            : ${DATASET}"
echo " clients            : ${NUM_CLIENTS}"
echo " rounds             : ${ROUNDS}"
echo " algorithm          : ${ALGORITHM}"
echo " encoder ratio      : ${ENCODER_RATIO}"
echo " run directory      : ${RUN_DIR}"
if [[ "${USE_WANDB}" == "1" ]]; then
  echo " wandb project       : ${WANDB_PROJECT}"
  echo " wandb run name      : ${WANDB_RUN_NAME}"
else
  echo " wandb               : disabled"
fi
echo "========================================="

pushd "${PROJECT_ROOT}" >/dev/null

LAUNCH_CMD=(python scripts/launch.py
  --num_clients "${NUM_CLIENTS}"
  --bind "${SERVER_BIND}"
  --server_addr "${SERVER_ADDR}"
  --data_root "${DATA_ROOT}"
  --dataset_name "${DATASET}"
  --num_classes "${NUM_CLASSES}"
  --rounds "${ROUNDS}"
  --local_epochs "${LOCAL_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --momentum "${MOMENTUM}"
  --sample_fraction "${SAMPLE_FRACTION}"
  --seed "${SEED}"
  --model_name "${MODEL_NAME}"
  --feature_dim "${FEATURE_DIM}"
  --encoder_ratio "${ENCODER_RATIO}"
  --algorithm "${ALGORITHM}"
  --device "${SERVER_DEVICE}"
  --client_device "${CLIENT_DEVICE}"
  --gpus "${CLIENT_GPUS}"
  --server_warmup_sec "${SERVER_WARMUP_SEC}"
  --stagger_sec "${CLIENT_STAGGER_SEC}"
  --run_dir "${RUN_DIR}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  LAUNCH_CMD+=(
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "${WANDB_RUN_NAME}"
  )
fi

"${LAUNCH_CMD[@]}"

popd >/dev/null

LAST_RUN_FILE="${SCRIPT_DIR}/.last_run_dir"
echo "${RUN_DIR}" > "${LAST_RUN_FILE}"

# ==========================
# 生成特征数据集
# ==========================

echo "\n[Pipeline] 导出客户端特征..."
GEN_CMD=(python "${REPO_ROOT}/system/generate_datasets.py"
  --run-dir "${RUN_DIR}"
  --dataset "${DATASET}"
  --num-clients "${NUM_CLIENTS}"
  --num-classes "${NUM_CLASSES}"
  --batch-size "${GEN_BATCH_SIZE}"
  --device "${GEN_DEVICE}"
  --seed "${GEN_SEED}"
  --encoder-ratio "${ENCODER_RATIO}"
)

if [[ "${GEN_KEEP_SPATIAL}" == "1" ]]; then
  GEN_CMD+=(--keep-spatial)
fi

"${GEN_CMD[@]}"

FEATURE_DIR="${RUN_DIR}/clients-feature"
if [[ ! -d "${FEATURE_DIR}" ]]; then
  echo "[Pipeline] 错误：未找到特征目录 ${FEATURE_DIR}" >&2
  exit 1
fi

# ==========================
# 训练尾部分类器
# ==========================

echo "\n[Pipeline] 训练尾部分类器..."
TAIL_CMD=(python "${REPO_ROOT}/system/train_resnet_tail.py"
  --dataset-dir "${FEATURE_DIR}"
  --model "${TAIL_MODEL_EXPR}"
  --encoder-ratio "${ENCODER_RATIO}"
  --batch-size "${TAIL_BATCH_SIZE}"
  --epochs "${TAIL_EPOCHS}"
  --learning-rate "${TAIL_LR}"
  --momentum "${TAIL_MOMENTUM}"
  --weight-decay "${TAIL_WEIGHT_DECAY}"
  --device "${TAIL_DEVICE}"
)

if [[ "${TAIL_USE_BALANCED_TEST}" == "1" ]]; then
  TAIL_CMD+=(--use-balanced-test)
fi

if [[ "${TAIL_FORCE_LINEAR_PROJECTION}" == "1" ]]; then
  TAIL_CMD+=(--force-linear-projection)
fi

if [[ "${TAIL_USE_WANDB}" == "1" ]]; then
  TAIL_CMD+=(
    --use-wandb \
    --wandb-project "${TAIL_WANDB_PROJECT}" \
    --wandb-entity "${TAIL_WANDB_ENTITY}" \
    --wandb-run-name "${TAIL_WANDB_RUN_NAME}"
  )
fi

"${TAIL_CMD[@]}"

echo "\n[Pipeline] 全流程完成。最终文件位于：${RUN_DIR}"
