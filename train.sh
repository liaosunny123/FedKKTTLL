#!/bin/bash

# FedEXT Training Script

# Dataset and model configuration
DATASET="CIFAR10"
NUM_CLASSES=10
MODEL_FAMILY="HtFE2"  # Can be HtFE2, HtFE3, HtFE4, HtFE8, HtFE9, HtM10

# FedEXT specific settings
ALGORITHM="FedEXT"
NUM_CLIENTS=3
GLOBAL_ROUNDS=100
LOCAL_EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=0.005
FEATURE_DIM=512

# Server training settings (for final classifier)
SERVER_LEARNING_RATE=0.01
SERVER_BATCH_SIZE=100
SERVER_EPOCHS=50

# Distribution configuration (with group assignments)
DISTRIBUTION_CONFIG="configs/distribution_missing_classes.json"

# Other settings
DEVICE="cuda"
DEVICE_ID="0"
EVAL_GAP=10
SAVE_FOLDER="results/fedext_${DATASET}_${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)"

echo "Starting FedEXT training..."
echo "Algorithm: $ALGORITHM"
echo "Dataset: $DATASET"
echo "Model Family: $MODEL_FAMILY"
echo "Number of Clients: $NUM_CLIENTS"
echo "Global Rounds: $GLOBAL_ROUNDS"
echo "Distribution Config: $DISTRIBUTION_CONFIG"
echo "Save Folder: $SAVE_FOLDER"
echo "----------------------------------------"

# Run FedEXT training
python system/main.py \
    -data $DATASET \
    -nb $NUM_CLASSES \
    -m $MODEL_FAMILY \
    -algo $ALGORITHM \
    -nc $NUM_CLIENTS \
    -gr $GLOBAL_ROUNDS \
    -ls $LOCAL_EPOCHS \
    -lbs $BATCH_SIZE \
    -lr $LEARNING_RATE \
    -fd $FEATURE_DIM \
    -slr $SERVER_LEARNING_RATE \
    -sbs $SERVER_BATCH_SIZE \
    -se $SERVER_EPOCHS \
    -dev $DEVICE \
    -did $DEVICE_ID \
    -eg $EVAL_GAP \
    -sfn $SAVE_FOLDER \
    -dc $DISTRIBUTION_CONFIG \
    -t 1

echo "----------------------------------------"
echo "Training completed!"
echo "Results saved to: $SAVE_FOLDER"