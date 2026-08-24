#!/usr/bin/env bash
# Export all three trained models to ONNX for the sensing-bench comparison.
set -u

LOG=/workspace/mlruns/_bench_logs
mkdir -p "$LOG"

run() {
  local name=$1 cfg=$2 ckpt=$3
  echo "[$(date -Is)] EXPORT START  $name" | tee -a "$LOG/export.log"
  autoware-ml deploy --config-name "$cfg" --weights "$ckpt" \
    deploy.tensorrt.enabled=false >"$LOG/export_$name.log" 2>&1
  local rc=$?
  echo "[$(date -Is)] EXPORT END    $name  exit=$rc" | tee -a "$LOG/export.log"
}

run ptv3_tiny segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
  /workspace/mlruns/segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2/f0f91b5331c14c7a988aaa6f9e6d77e0/artifacts/checkpoints/best.ckpt

run litept_tiny_h32 segmentation3d/litept/voxel012_tiny_h32_122m_t4dataset_j6gen2 \
  /workspace/mlruns/segmentation3d/litept/voxel012_tiny_h32_122m_t4dataset_j6gen2/47914b21b1274161a5e10d0ca9077925/artifacts/checkpoints/best.ckpt

run litept_tiny_h16 segmentation3d/litept/voxel012_tiny_h16_122m_t4dataset_j6gen2 \
  /workspace/mlruns/segmentation3d/litept/voxel012_tiny_h16_122m_t4dataset_j6gen2/1caba33374da4ef480d3bd62578124cf/artifacts/checkpoints/best.ckpt

echo "[$(date -Is)] EXPORTS DONE" | tee -a "$LOG/export.log"
