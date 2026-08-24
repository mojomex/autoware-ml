#!/usr/bin/env bash
# Sequential apples-to-apples training: ptv3-tiny, LitePT-tiny h32, LitePT-tiny h16.
# Sequential rather than concurrent because training here is NFS-read bound at
# ~175 MB/s and that does not scale with parallel readers, so concurrency would
# split the same bandwidth three ways for no net gain.
set -u

EPOCHS=30
LOG=/workspace/mlruns/_bench_logs
mkdir -p "$LOG"

run() {
  local name=$1 cfg=$2
  echo "[$(date -Is)] START  $name  ($cfg)" | tee -a "$LOG/driver.log"
  autoware-ml train --config-name "$cfg" trainer.max_epochs=$EPOCHS \
    >"$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date -Is)] END    $name  exit=$rc" | tee -a "$LOG/driver.log"
}

echo "[$(date -Is)] driver starting, epochs=$EPOCHS" | tee -a "$LOG/driver.log"
run ptv3_tiny       segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2
run litept_tiny_h32 segmentation3d/litept/voxel012_tiny_h32_122m_t4dataset_j6gen2
run litept_tiny_h16 segmentation3d/litept/voxel012_tiny_h16_122m_t4dataset_j6gen2
echo "[$(date -Is)] ALL DONE" | tee -a "$LOG/driver.log"
