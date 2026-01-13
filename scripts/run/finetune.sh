#!/bin/bash
# This script is used to finetune the model
# arguments:
#   GPU number
#   task config
#   other hydra overrides

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

GPU=$1
config=$2
ARGS=${@:3}

config="${config#configs/}" # delete prefix configs/
config="${config#task/}" # delete prefix task/
config="${config%.yaml}" # delete suffix .yaml

# Create a unique timestamp for this training run to avoid multiple output directories
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR="${GALAXEA_FM_OUTPUT_DIR}/${config}/${TIMESTAMP}"

# Pass the run directory to Hydra to ensure all processes use the same output directory
torchrun --standalone --nnodes 1 --nproc-per-node $GPU scripts/finetune.py task=$config hydra.run.dir=$RUN_DIR $ARGS