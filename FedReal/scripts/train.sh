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
ROUNDS=3
LOCAL_EPOCHS=1
BATCH_SIZE=10
LR=0.005
MOMENTUM=0.9
SAMPLE_FRACTION=1.0
SEED=42
MODEL_NAME="resnet18"
FEATURE_DIM=512
MAX_MESSAGE_MB=256

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
# (2) 特征导出与尾部分类器配置（由服务端自动完成）
# ==========================
FEATURE_BATCH_SIZE=128
FEATURE_KEEP_SPATIAL=1
TAIL_BATCH_SIZE=64
TAIL_EPOCHS=10
TAIL_LR=0.01
TAIL_MOMENTUM=0.9
TAIL_WEIGHT_DECAY=1e-4
TAIL_DEVICE="cuda"

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
  --max_message_mb "${MAX_MESSAGE_MB}"
  --encoder_ratio "${ENCODER_RATIO}"
  --algorithm "${ALGORITHM}"
  --device "${SERVER_DEVICE}"
  --client_device "${CLIENT_DEVICE}"
  --gpus "${CLIENT_GPUS}"
  --server_warmup_sec "${SERVER_WARMUP_SEC}"
  --stagger_sec "${CLIENT_STAGGER_SEC}"
  --run_dir "${RUN_DIR}"
  --feature_batch_size "${FEATURE_BATCH_SIZE}"
  --tail_batch_size "${TAIL_BATCH_SIZE}"
  --tail_epochs "${TAIL_EPOCHS}"
  --tail_lr "${TAIL_LR}"
  --tail_momentum "${TAIL_MOMENTUM}"
  --tail_weight_decay "${TAIL_WEIGHT_DECAY}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  LAUNCH_CMD+=(
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "${WANDB_RUN_NAME}"
  )
fi

if [[ "${FEATURE_KEEP_SPATIAL}" == "1" ]]; then
  LAUNCH_CMD+=(--feature_keep_spatial)
fi

if [[ -n "${TAIL_DEVICE}" ]]; then
  LAUNCH_CMD+=(--tail_device "${TAIL_DEVICE}")
fi

"${LAUNCH_CMD[@]}"

popd >/dev/null

LAST_RUN_FILE="${SCRIPT_DIR}/.last_run_dir"
echo "${RUN_DIR}" > "${LAST_RUN_FILE}"

echo "\n[Pipeline] 联邦训练与服务端尾部训练正在运行。最新结果保存在：${RUN_DIR}"
