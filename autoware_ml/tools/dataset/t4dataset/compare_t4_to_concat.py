"""Reconstruct an MCAP's concatenated pointcloud from an unfiltered T4Dataset.

The T4Dataset is generated with the Nebula mask, ego crop box and ring outlier filter all
DISABLED, so it holds the full ego-motion-corrected cloud. The MCAP is a replay of the same drive
with the ring outlier filter disabled but the mask and crop enabled, so its
/sensing/lidar/concatenated/pointcloud is exactly what our two transforms should produce.

Compares two variants:
  mask_crop          - both decisions taken on the corrected coordinates T4Dataset provides
  inverse_mask_crop  - both decisions taken in pre-correction space, where the vehicle takes them

Output points stay ego-motion corrected in either case; only the keep/drop decision moves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoware_ml.tools.dataset.t4dataset.compare_nebula_voxels_to_concat import (
    _load_concat_pointclouds,
    _voxelize,
)
from autoware_ml.tools.dataset.t4dataset.nebula_order_error import (
    _corrected_to_raw_ego,
    _ego_to_lidar_points,
    _load_pose_table,
    _point_offsets_seconds,
    _quat_to_rotmat,
    _stamp_to_seconds,
)
from autoware_ml.transforms.point_cloud.filters import (
    AIP_X2_GEN2_EGO_CROP_BOXES,
    NebulaDownsampleMaskFilter,
    _nebula_azimuth_rad,
    _normalize_lidar_name,
    _round_half_up,
)

CONCAT_TOPIC = "/sensing/lidar/concatenated/pointcloud"
CHANNEL_DIM, TIME_DIM = 4, 6


def voxel_iou(a: np.ndarray, b: np.ndarray, size: float) -> float:
    """Occupancy IoU over a shared voxel grid -- the metric used on the Confluence page."""
    va, vb = _voxelize(a, size), _voxelize(b, size)
    if not len(va) or not len(vb):
        return 0.0
    inter = np.intersect1d(va, vb, assume_unique=True).size
    return inter / (len(va) + len(vb) - inter)


def crop_keep(ego: np.ndarray) -> np.ndarray:
    inside = np.zeros(ego.shape[0], dtype=bool)
    for x0, y0, z0, x1, y1, z1 in AIP_X2_GEN2_EGO_CROP_BOXES:
        inside |= (
            (ego[:, 0] >= x0)
            & (ego[:, 0] <= x1)
            & (ego[:, 1] >= y0)
            & (ego[:, 1] <= y1)
            & (ego[:, 2] >= z0)
            & (ego[:, 2] <= z1)
        )
    return ~inside


def mask_keep(f: NebulaDownsampleMaskFilter, lidar: str, local: np.ndarray, ch: np.ndarray):
    mask = f._load_mask(lidar, f.lidar_name_to_model[lidar])
    cal = f._load_calibration(lidar)
    az = np.rad2deg(_nebula_azimuth_rad(local.astype(np.float32)))
    if f.use_calibration_azimuth_offsets:
        v = (ch >= 0) & (ch < cal.azimuth_deg.shape[0])
        az = az.copy()
        az[v] -= cal.azimuth_deg[ch[v]]
    az = (az - f.azimuth_start_deg) % 360.0
    x = _round_half_up(az / f.azimuth_extent_deg * mask.shape[1])
    ok = (x >= 0) & (x < mask.shape[1]) & (ch >= 0) & (ch < mask.shape[0])
    keep = np.zeros(local.shape[0], dtype=bool)
    keep[ok] = mask[ch[ok], x[ok]]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t4", required=True, type=Path)
    ap.add_argument("--mcap", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--max-stamp-diff", type=float, default=0.01)
    ap.add_argument("--voxel-size", type=float, default=0.12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scene = args.t4 / "0" if (args.t4 / "0").exists() else args.t4
    ann = scene / "annotation"
    sensors = {s["token"]: s.get("channel") for s in json.loads((ann / "sensor.json").read_text())}
    calib = {c["sensor_token"]: c for c in json.loads((ann / "calibrated_sensor.json").read_text())}
    poses = _load_pose_table(ann / "ego_pose.json")

    info_dir = scene / "data" / "LIDAR_CONCAT_INFO"
    pcd_dir = scene / "data" / "LIDAR_CONCAT"
    frame_ids = sorted(p.stem for p in info_dir.glob("*.json"))

    messages = _load_concat_pointclouds(args.mcap, [CONCAT_TOPIC])[CONCAT_TOPIC]
    concat = {m.stamp: m for m in messages}
    concat_stamps = np.array(sorted(concat))
    print(
        f"concat clouds: {len(concat_stamps)}  "
        f"span {concat_stamps[0]:.3f}..{concat_stamps[-1]:.3f}",
        flush=True,
    )

    f = NebulaDownsampleMaskFilter()
    rows = []
    for fid in frame_ids:
        info = json.loads((info_dir / f"{fid}.json").read_text())
        ref_t = _stamp_to_seconds(info["stamp"])
        i = int(np.argmin(np.abs(concat_stamps - ref_t)))
        if abs(concat_stamps[i] - ref_t) > args.max_stamp_diff:
            continue
        target = concat[concat_stamps[i]]

        dim = int(info.get("num_pts_feats", 7))
        pts = np.fromfile(pcd_dir / f"{fid}.pcd.bin", dtype=np.float32).reshape(-1, dim)
        keep_direct = np.zeros(pts.shape[0], bool)
        keep_inverse = np.zeros(pts.shape[0], bool)

        for src in info["sources"]:
            name = sensors.get(src["sensor_token"])
            cal = calib.get(src["sensor_token"])
            if name is None or cal is None:
                continue
            lidar = _normalize_lidar_name(name)
            b, e = int(src["idx_begin"]), int(src["idx_begin"]) + int(src["length"])
            sp = pts[b:e]
            if not sp.size:
                continue
            trans = np.asarray(cal["translation"], dtype=np.float64)
            rot = _quat_to_rotmat(np.asarray([cal["rotation"]], dtype=np.float64))[0]
            ch = sp[:, CHANNEL_DIM].astype(np.int64)

            cur_local = _ego_to_lidar_points(sp[:, :3].astype(np.float64), trans, rot)
            ptimes = _stamp_to_seconds(src["stamp"]) + _point_offsets_seconds(sp[:, TIME_DIM], "ns")
            raw_ego = _corrected_to_raw_ego(sp[:, :3], ref_t, ptimes, poses)
            raw_local = _ego_to_lidar_points(raw_ego, trans, rot)

            keep_direct[b:e] = mask_keep(f, lidar, cur_local, ch) & crop_keep(sp[:, :3])
            keep_inverse[b:e] = mask_keep(f, lidar, raw_local, ch) & crop_keep(raw_ego)

        actual = target.points.shape[0]
        row = {
            "frame": fid,
            "stamp": ref_t,
            "t4_points": int(pts.shape[0]),
            "actual": int(actual),
            "mask_crop": int(keep_direct.sum()),
            "inverse_mask_crop": int(keep_inverse.sum()),
        }
        for k in ("mask_crop", "inverse_mask_crop"):
            row[f"{k}_delta_pct"] = 100 * (row[k] / actual - 1)
        tgt_xyz = target.points
        for k, keep in (("mask_crop", keep_direct), ("inverse_mask_crop", keep_inverse)):
            row[f"{k}_voxel_iou"] = voxel_iou(pts[keep, :3], tgt_xyz, args.voxel_size)
        rows.append(row)
        print(
            f"  {fid} t4={row['t4_points']:7d} actual={actual:7d} "
            f"direct={row['mask_crop_delta_pct']:+7.2f}% IoU={row['mask_crop_voxel_iou']:.3f}  "
            f"inverse={row['inverse_mask_crop_delta_pct']:+7.2f}% "
            f"IoU={row['inverse_mask_crop_voxel_iou']:.3f}",
            flush=True,
        )
        if len(rows) >= args.frames:
            break

    if not rows:
        print("no frames matched")
        return
    tot = sum(r["actual"] for r in rows)
    summary = {"frames": len(rows), "total_actual": tot}
    for k in ("mask_crop", "inverse_mask_crop"):
        summary[f"{k}_total_delta_pct"] = 100 * (sum(r[k] for r in rows) / tot - 1)
        summary[f"{k}_mean_voxel_iou"] = float(np.mean([r[f"{k}_voxel_iou"] for r in rows]))
    print("\nSUMMARY", json.dumps(summary, indent=1))
    Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
