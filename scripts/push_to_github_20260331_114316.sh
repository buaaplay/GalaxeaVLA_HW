#!/usr/bin/env bash
# push_to_github_20260331_114316.sh
#
# 把 GalaxeaVLA 的新增文件推送到 GitHub（不影响 Gitee origin）
#
# 使用前：
#   1. 在 GitHub.com 上 New repository，名字随意（如 GalaxeaVLA），
#      不要勾选 Add README / .gitignore / license，保持完全空仓库
#   2. 把下面 GITHUB_REPO_URL 改成你的仓库地址
#   3. chmod +x 本脚本后执行

set -euo pipefail

GITHUB_REPO_URL="https://github.com/buaaplay/GalaxeaVLA_HW.git"
REMOTE_NAME="github"
BRANCH="main"

cd "$(dirname "$0")/.."   # 切到 GalaxeaVLA 根目录

# ── 1. 添加 github remote（已存在则跳过）──────────────────────────────────
if git remote get-url "$REMOTE_NAME" &>/dev/null; then
    echo "[INFO] remote '$REMOTE_NAME' already exists, skipping add"
else
    git remote add "$REMOTE_NAME" "$GITHUB_REPO_URL"
    echo "[INFO] Added remote '$REMOTE_NAME' -> $GITHUB_REPO_URL"
fi

# ── 2. 暂存新增的两个文件 ─────────────────────────────────────────────────
git add scripts/deploy_last0.py
git add configs/task/real/last0_deploy.yaml

# ── 3. 如果有其他已修改文件也一起提交（可选，注释掉则只提交新文件）────────
# git add scripts/deploy.py

# ── 4. commit ────────────────────────────────────────────────────────────────
git commit -m "feat: add LaST0 deploy script and config for R1-Lite EE_POSE mode"

# ── 5. push 到 GitHub（不动 Gitee origin）────────────────────────────────────
git push "$REMOTE_NAME" HEAD:"$BRANCH"

echo ""
echo "✓ Pushed to GitHub: $GITHUB_REPO_URL  branch=$BRANCH"
echo ""
echo "On the robot machine, clone with:"
echo "  git clone $GITHUB_REPO_URL"
echo "  cd GalaxeaVLA"
echo "  # then run deploy (see below)"
