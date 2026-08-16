#!/usr/bin/env bash
# 打包代码 -> 更新 dataset goldfish9901/radar-vital-signs-code -> 推送验证 kernel
# 用法: bash kaggle/push_validate_physdrive.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="/tmp/kaggle_env/bin/kaggle"
PROXY="proxychains4 -q"
PKG_DIR=/tmp/kaggle_pkg_validate
DATASET_DIR="$PKG_DIR/dataset"
KERNEL_DIR="$PKG_DIR/kernel"
ZIP="$DATASET_DIR/radar_vital_signs_code.zip"

echo "==> 打包代码 (exclude 数据/权重/.git)"
rm -rf "$PKG_DIR"
mkdir -p "$DATASET_DIR" "$KERNEL_DIR"
cd "$REPO"
# 以 Radar_Vital_Signs_grad0/ 为前缀打包，与挂载结构一致
tar --exclude='.git' --exclude='Dataset' --exclude='exports' \
    --exclude='training_exports' --exclude='model_outputs' \
    --exclude='experiment_runs' --exclude='docs' --exclude='__pycache__' \
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
$PROXY "$KAGGLE" datasets version -p "$DATASET_DIR" -m "add PhysDrive validation scripts" 2>&1 | tail -3

echo "==> kernel metadata (挂载官方 PhysDrive dataset xiaoyang274/physdrive)"
cat > "$KERNEL_DIR/kernel-metadata.json" <<'JSON'
{
  "id": "goldfish9901/radar-vital-signs-validate-physdrive",
  "title": "Radar Vital Signs Validate PhysDrive",
  "code_file": "run_validate_physdrive.py",
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
cp "$REPO/kaggle/run_validate_physdrive.py" "$KERNEL_DIR/"

echo "==> 推送 kernel"
$PROXY "$KAGGLE" kernels push -p "$KERNEL_DIR" 2>&1 | tail -5

echo "==> 查看 kernel 状态"
$PROXY "$KAGGLE" kernels status goldfish9901/radar-vital-signs-validate-physdrive 2>&1 | tail -3
echo "done"
