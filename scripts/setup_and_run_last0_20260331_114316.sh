#!/usr/bin/env bash
# setup_and_run_last0_20260331_114316.sh
#
# 在真机（robot@5090）上执行：clone 仓库、配置 PYTHONPATH、启动 deploy_last0.py
#
# 前提：
#   - /home/robot/weights/MyData_finetune_last0_checkpoint-13-171570/ 已存在
#   - last0 源码已在 /home/robot/last0/（或修改 LAST0_ROOT）
#   - GalaxeaVLA 的 Python 环境已安装（或用 conda/venv）
#
# 用法：
#   chmod +x setup_and_run_last0_20260331_114316.sh
#   ./setup_and_run_last0_20260331_114316.sh

set -euo pipefail

# ── 路径配置（按实际情况修改）────────────────────────────────────────────────
GALAXEA_DIR="/home/robot/project/GalaxeaVLA_HW"
LAST0_ROOT="${GALAXEA_DIR}/last0"
CKPT_DIR="/home/robot/weights/MyData_finetune_last0_checkpoint-13-171570"
CONDA_ENV="last0"

echo "[1/3] Activating conda env: $CONDA_ENV"
# 硬编码 conda 路径，避免 conda info --base 在非交互式 shell 卡住
CONDA_BASE="${CONDA_PREFIX_1:-${CONDA_PREFIX:-/home/robot/miniconda3}}"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # 尝试常见路径
    for p in /home/robot/anaconda3 /opt/conda /usr/local/miniconda3 /home/robot/miniforge3; do
        if [ -f "$p/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$p"
            break
        fi
    done
fi
echo "    conda base: $CONDA_BASE"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "    Python: $(which python)  ($(python --version 2>&1))"

echo "[2/3] Setting PYTHONPATH"
export PYTHONPATH="$LAST0_ROOT:${PYTHONPATH:-}"
echo "    PYTHONPATH=$PYTHONPATH"

echo "[3/3] Launching deploy_last0.py"
echo "    GALAXEA_DIR : $GALAXEA_DIR"
echo "    CKPT_DIR    : $CKPT_DIR"
cd "$GALAXEA_DIR"

python scripts/deploy_last0.py \
    --config-name task/real/last0_deploy \
    last0.model_dir="${CKPT_DIR}/tfmr" \
    last0.stats_path="${CKPT_DIR}/stats_data.json" \
    deploy.host=192.168.12.12 \
    deploy.port=8000 \
    deploy.use_chunk=8 \
    deploy.prompt="Pick up the rubbish on the table and throw it into the bin."
