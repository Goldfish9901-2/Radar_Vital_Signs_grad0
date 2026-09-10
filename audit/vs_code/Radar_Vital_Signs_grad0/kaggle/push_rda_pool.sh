#!/usr/bin/env bash
# 打包代码 -> 更新 dataset goldfish9901/radar-vital-signs-code -> 推送 RDA 池筛选 kernel
# 用法: bash kaggle/push_rda_pool.sh
set -euo pipefail

# 脚本位于 <project>/Radar_Vital_Signs_grad0/kaggle/，REPO 指向包含 .venv/pyproject.toml 的项目根
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SCRIPT_DIR/../../.venv/bin/kaggle" ]; then
  REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [ -x "$SCRIPT_DIR/../.venv/bin/kaggle" ]; then
  REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# kaggle CLI 由 uv 管理：优先用 uv 创建的 venv，回退到系统 PATH
if [ -x "$REPO/.venv/bin/kaggle" ]; then
  KAGGLE="$REPO/.venv/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
  KAGGLE="$(command -v kaggle)"
else
  echo "kaggle CLI 未找到，正在用 uv 安装..." >&2
  (cd "$REPO" && uv add kaggle) >&2
  KAGGLE="$REPO/.venv/bin/kaggle"
fi
PROXY="proxychains4 -q"
# 打包的是 Radar_Vital_Signs_grad0 子目录（脚本所在项目的源码树），与 venv 根 REPO 区分开
PKG_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR=/tmp/kaggle_pkg_rda
DATASET_DIR="$PKG_DIR/dataset"
KERNEL_DIR="$PKG_DIR/kernel"
ZIP="$DATASET_DIR/radar_vital_signs_code.zip"

echo "==> 打包代码 (exclude 数据/权重/.git)"
rm -rf "$PKG_DIR"
mkdir -p "$DATASET_DIR" "$KERNEL_DIR"
cd "$PKG_SRC"
tar --exclude='.git' --exclude='Dataset' --exclude='exports' \
    --exclude='training_exports' --exclude='model_outputs' \
    --exclude='docs' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.pt' --exclude='*.npz' --exclude='*.zip' \
    -czf /tmp/repo_tar.gz .
rm -rf /tmp/radar_pkg_tmp && mkdir -p /tmp/radar_pkg_tmp/Radar_Vital_Signs_grad0
tar -xzf /tmp/repo_tar.gz -C /tmp/radar_pkg_tmp/Radar_Vital_Signs_grad0
cd /tmp/radar_pkg_tmp
python3 -m zipfile -c "$ZIP" Radar_Vital_Signs_grad0
rm -rf /tmp/radar_pkg_tmp /tmp/repo_tar.gz
echo "zip: $(du -h "$ZIP" | cut -f1)"

echo "==> dataset metadata"
cat > "$DATASET_DIR/dataset-metadata.json" <<'JSON'
{
  "id": "goldfish9901/radar-vital-signs-code",
  "title": "Radar Vital Signs Code",
  "subtitle": "Source code for mmWave radar vital signs pipeline (Kaggle runs)",
  "isPrivate": true,
  "licenses": [
    { "name": "other" }
  ]
}
JSON

echo "==> 更新 dataset (version)"
$PROXY "$KAGGLE" datasets version -p "$DATASET_DIR" -m "add RDA front-end algorithm pool (src/radar) + screening kernel" 2>&1 | tail -3

echo "==> 等待 dataset 新版本就绪（避免 kernel 挂载旧版本代码）"
sleep 60

echo "==> kernel metadata (挂载官方 PhysDrive dataset xiaoyang274/physdrive)"
cat > "$KERNEL_DIR/kernel-metadata.json" <<'JSON'
{
  "id": "goldfish9901/radar-vital-signs-rda-front-end-pool",
  "title": "Radar Vital Signs RDA Front-end Pool",
  "code_file": "run_rda_pool.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "goldfish9901/radar-vital-signs-code",
    "xiaoyang274/physdrive"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON
cp "$PKG_SRC/kaggle/run_rda_pool.py" "$KERNEL_DIR/"

echo "==> 推送 kernel"
$PROXY "$KAGGLE" kernels push -p "$KERNEL_DIR" 2>&1 | tail -5

echo "==> 查看 kernel 状态"
$PROXY "$KAGGLE" kernels status goldfish9901/radar-vital-signs-rda-front-end-pool 2>&1 | tail -3
echo "done"
