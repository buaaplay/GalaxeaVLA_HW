#!/usr/bin/env bash
# 在推理机上执行，完全同步开发机最新代码
set -euo pipefail

cd "$(dirname "$0")/.."

git stash
git pull origin main
git stash pop 2>/dev/null || true
