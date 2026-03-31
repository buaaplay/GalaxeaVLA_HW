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
GITHUB_REPO_URL="https://github.com/<your-username>/GalaxeaVLA.git"
GALAXEA_DIR="/home/robot/GalaxeaVLA"
LAST0_ROOT="/home/robot/last0"                  # last0 源码根目录
CKPT_DIR="/home/robot/weights/MyData_finetune_last0_checkpoint-13-171570"
CONDA_ENV="base"                                 # 改成你的 conda env 名

# ── 1. clone（已存在则 pull）─────────────────────────────────────────────────
if [ -d "$GALAXEA_DIR/.git" ]; then
    echo "[INFO] Repo exists, pulling latest..."
    git -C "$GALAXEA_DIR" pull origin main
else
    echo "[INFO] Cloning from GitHub..."
    git clone "$GITHUB_REPO_URL" "$GALAXEA_DIR"
fi

# ── 2. 激活 conda 环境 ────────────────────────────────────────────────────────
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── 3. 设置 PYTHONPATH（让 deploy_last0.py 能 import janus / experiments）────
export PYTHONPATH="$LAST0_ROOT:$LAST0_ROOT/janus:${PYTHONPATH:-}"

# ── 4. 启动 deploy ────────────────────────────────────────────────────────────
cd "$GALAXEA_DIR"

python scripts/deploy_last0.py \
    --config-name task/real/last0_deploy \
    last0.model_dir="${CKPT_DIR}/tfmr" \
    last0.stats_path="${CKPT_DIR}/stats_data.json" \
    deploy.host=192.168.12.12 \
    deploy.port=8000 \
    deploy.use_chunk=8 \
    deploy.prompt="Pick up the rubbish on the table and throw it into the bin."
