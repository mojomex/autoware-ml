"""Summarize the three-way comparison from the MLflow sqlite store.

Pure stdlib sqlite over mlruns/mlflow.db; imports nothing from autoware_ml.
"""

import sqlite3
import sys

DB = "mlruns/mlflow.db"
KEYS = [
    "val/loss",
    "val/seg3d/mIoU",
    "val/seg3d/acc",
    "val/seg3d/mIoU_0m_50m",
    "val/seg3d/mIoU_50m_90m",
    "val/seg3d/mIoU_90m_121m",
]

con = sqlite3.connect(DB)
runs = con.execute(
    "select r.run_uuid, e.name, r.start_time, r.end_time, r.status "
    "from runs r join experiments e on r.experiment_id = e.experiment_id "
    "order by r.start_time"
).fetchall()

if not runs:
    sys.exit("no runs in the store yet")

for uuid, exp, start, end, status in runs:
    steps = con.execute(
        "select max(step) from metrics where run_uuid = ? and key = 'val/loss'", (uuid,)
    ).fetchone()[0]
    print(f"\n=== {exp}")
    print(f"    run {uuid[:12]}  status {status}  last step {steps}")
    if end and start:
        print(f"    wall {(end - start) / 3.6e6:.2f} h")

    # the epoch selected by the checkpoint callback is the one with lowest val/loss
    best = con.execute(
        "select step, value from metrics where run_uuid = ? and key = 'val/loss' "
        "order by value asc limit 1",
        (uuid,),
    ).fetchone()
    if not best:
        print("    no val/loss logged yet")
        continue
    best_step, best_loss = best
    print(f"    best val/loss {best_loss:.5f} at step {best_step}")
    for k in KEYS:
        at_best = con.execute(
            "select value from metrics where run_uuid = ? and key = ? and step = ?",
            (uuid, k, best_step),
        ).fetchone()
        peak = con.execute(
            "select max(value) from metrics where run_uuid = ? and key = ?", (uuid, k)
        ).fetchone()[0]
        if at_best is None and peak is None:
            continue
        a = f"{at_best[0]:.4f}" if at_best else "   -  "
        p = f"{peak:.4f}" if peak is not None else "   -  "
        print(f"      {k:<26} at best-loss {a}   peak {p}")
