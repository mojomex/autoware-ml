# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Estimate the error from applying Nebula masks after ego-motion correction.

The vehicle-side order was approximately::

    raw pointcloud -> Nebula mask filter -> ego-motion correction

The backward-compatibility transform has to run on T4Dataset pointclouds that are already
ego-motion corrected::

    raw pointcloud -> ego-motion correction -> Nebula mask filter

This tool approximates the original order by inverting the ego-motion correction with
interpolated T4 ego poses, then comparing mask keep/drop decisions in the raw and corrected
LiDAR frames. It intentionally uses the preserved T4 point ``channel`` feature instead of
reconstructing channel from elevation.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.point_cloud.filters import (
    NebulaDownsampleMaskFilter,
    _nebula_azimuth_rad,
    _normalize_lidar_name,
    _round_half_up,
)


@dataclass(frozen=True)
class PoseTable:
    times: npt.NDArray[np.float64]
    translations: npt.NDArray[np.float64]
    quaternions: npt.NDArray[np.float64]


@dataclass(frozen=True)
class SourceMetrics:
    source_name: str
    num_points: int
    current_kept: int
    original_kept: int
    agree: int
    false_keep: int
    false_drop: int
    jaccard: float
    mismatch_rate: float
    mean_abs_azimuth_shift_deg: float
    p95_abs_azimuth_shift_deg: float
    mean_range_m: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/t4dataset"))
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path(
            "/workspace/data/t4dataset/info/segdet3d/t4dataset_j6gen2_segdet3d_infos_val.pkl"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--point-timestamp-unit",
        choices=("ns", "us", "s"),
        default="ns",
        help="Unit of the T4 per-point timestamp feature.",
    )
    args = parser.parse_args()

    records = _load_records(args.ann_file)
    selected_records = records[args.start_index :: args.frame_stride][: args.max_frames]
    requested_sources = {_normalize_lidar_name(source) for source in args.sources or []}

    filter_impl = NebulaDownsampleMaskFilter()
    frame_results: list[dict[str, Any]] = []
    source_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for frame_index, record in enumerate(selected_records):
        frame_result = analyze_frame(
            record=record,
            data_root=args.data_root,
            filter_impl=filter_impl,
            requested_sources=requested_sources,
            point_timestamp_unit=args.point_timestamp_unit,
        )
        frame_result["input_index"] = args.start_index + frame_index * args.frame_stride
        frame_results.append(frame_result)
        for source in frame_result["sources"]:
            totals = source_totals[source["source_name"]]
            for key in (
                "num_points",
                "current_kept",
                "original_kept",
                "agree",
                "false_keep",
                "false_drop",
            ):
                totals[key] += float(source[key])
            totals["azimuth_shift_sum"] += float(source["mean_abs_azimuth_shift_deg"]) * float(
                source["num_points"]
            )
            totals["range_sum"] += float(source["mean_range_m"]) * float(source["num_points"])

    aggregate = _build_aggregate(source_totals)
    output = {
        "ann_file": str(args.ann_file),
        "data_root": str(args.data_root),
        "max_frames": args.max_frames,
        "frame_stride": args.frame_stride,
        "point_timestamp_unit": args.point_timestamp_unit,
        "num_frames_analyzed": len(frame_results),
        "aggregate_by_source": aggregate,
        "frames": frame_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    if args.output_csv is not None:
        _write_csv(args.output_csv, frame_results)

    print(
        json.dumps(
            {"num_frames_analyzed": len(frame_results), "aggregate_by_source": aggregate}, indent=2
        )
    )


def analyze_frame(
    *,
    record: Mapping[str, Any],
    data_root: Path,
    filter_impl: NebulaDownsampleMaskFilter,
    requested_sources: set[str],
    point_timestamp_unit: str,
) -> dict[str, Any]:
    lidar_points = record["lidar_points"]
    point_path = data_root / (lidar_points.get("lidar_path") or lidar_points["filename"])
    load_dim = int(lidar_points.get("num_pts_feats", 7))
    points = np.fromfile(point_path, dtype=np.float32).reshape(-1, load_dim)
    feature_names = list(lidar_points.get("feature_names", []))
    channel_dim = _feature_dim(feature_names, "channel", default=4)
    point_time_dim = _feature_dim(feature_names, "timestamp", default=6)

    dataset_scene_dir = _dataset_scene_dir(data_root, lidar_points)
    pose_table = _load_pose_table(dataset_scene_dir / "annotation" / "ego_pose.json")
    reference_time = _stamp_to_seconds(record["lidar_sources_info"]["stamp"])

    token_to_source = {
        str(source_meta["sensor_token"]): (str(source_name), source_meta)
        for source_name, source_meta in record["lidar_sources"].items()
    }

    source_outputs = []
    for source_info in record["lidar_sources_info"]["sources"]:
        sensor_token = str(source_info["sensor_token"])
        if sensor_token not in token_to_source:
            continue
        source_name, source_meta = token_to_source[sensor_token]
        normalized_source_name = _normalize_lidar_name(source_name)
        if requested_sources and normalized_source_name not in requested_sources:
            continue

        idx_begin = int(source_info["idx_begin"])
        idx_end = idx_begin + int(source_info["length"])
        source_points = points[idx_begin:idx_end]
        if source_points.size == 0:
            continue

        source_time = _stamp_to_seconds(source_info["stamp"])
        # Matches Autoware distortion_corrector: point time is header.stamp + time_stamp [ns].
        point_offsets = _point_offsets_seconds(
            source_points[:, point_time_dim], point_timestamp_unit
        )
        point_times = source_time + point_offsets

        current_local = _ego_to_lidar_points(
            source_points[:, :3],
            np.asarray(source_meta["translation"], dtype=np.float64),
            np.asarray(source_meta["rotation"], dtype=np.float64),
        )
        raw_ego = _corrected_to_raw_ego(
            source_points[:, :3], reference_time, point_times, pose_table
        )
        raw_local = _ego_to_lidar_points(
            raw_ego,
            np.asarray(source_meta["translation"], dtype=np.float64),
            np.asarray(source_meta["rotation"], dtype=np.float64),
        )

        channels = source_points[:, channel_dim].astype(np.int64)
        current_keep, current_x = _mask_keep(
            filter_impl, normalized_source_name, current_local, channels
        )
        original_keep, original_x = _mask_keep(
            filter_impl, normalized_source_name, raw_local, channels
        )
        metrics = _summarize_source(
            source_name=source_name,
            current_keep=current_keep,
            original_keep=original_keep,
            current_x=current_x,
            original_x=original_x,
            ranges=np.linalg.norm(current_local[:, :3], axis=1),
        )
        source_outputs.append(metrics.__dict__)

    return {
        "token": record.get("token"),
        "scene_token": record.get("scene_token"),
        "scene_name": record.get("scene_name"),
        "timestamp": record.get("timestamp"),
        "lidar_path": str(lidar_points.get("lidar_path") or lidar_points.get("filename")),
        "sources": source_outputs,
    }


def _mask_keep(
    filter_impl: NebulaDownsampleMaskFilter,
    lidar_name: str,
    local_points: npt.NDArray[np.float64],
    channels: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    model_name = filter_impl.lidar_name_to_model[lidar_name]
    calibration = filter_impl._load_calibration(lidar_name)
    mask = filter_impl._load_mask(lidar_name, model_name)
    azimuth_deg = np.rad2deg(_nebula_azimuth_rad(local_points.astype(np.float32)))
    if filter_impl.use_calibration_azimuth_offsets:
        valid_channels = (channels >= 0) & (channels < calibration.azimuth_deg.shape[0])
        azimuth_deg = azimuth_deg.copy()
        azimuth_deg[valid_channels] = (
            azimuth_deg[valid_channels] - calibration.azimuth_deg[channels[valid_channels]]
        )
    azimuth_deg = (azimuth_deg - filter_impl.azimuth_start_deg) % 360.0
    x = _round_half_up(azimuth_deg / filter_impl.azimuth_extent_deg * mask.shape[1])
    valid = (x >= 0) & (x < mask.shape[1]) & (channels >= 0) & (channels < mask.shape[0])
    keep = np.zeros(local_points.shape[0], dtype=bool)
    keep[valid] = mask[channels[valid], x[valid]]
    return keep, x


def _corrected_to_raw_ego(
    corrected_ego: npt.NDArray[np.float32],
    reference_time: float,
    point_times: npt.NDArray[np.float64],
    pose_table: PoseTable,
) -> npt.NDArray[np.float64]:
    ref_t, ref_q = _interpolate_poses(np.array([reference_time], dtype=np.float64), pose_table)
    global_points = corrected_ego.astype(np.float64) @ _quat_to_rotmat(ref_q)[0].T + ref_t[0]
    point_t, point_q = _interpolate_poses(point_times, pose_table)
    point_rot = _quat_to_rotmat(point_q)
    return np.einsum("nj,njk->nk", global_points - point_t, point_rot)


def _ego_to_lidar_points(
    ego_points: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (ego_points - translation) @ rotation


def _interpolate_poses(
    times: npt.NDArray[np.float64], pose_table: PoseTable
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    indices = np.searchsorted(pose_table.times, times, side="right")
    i0 = np.clip(indices - 1, 0, pose_table.times.shape[0] - 1)
    i1 = np.clip(indices, 0, pose_table.times.shape[0] - 1)
    t0 = pose_table.times[i0]
    t1 = pose_table.times[i1]
    denom = np.maximum(t1 - t0, 1e-9)
    alpha = np.clip((times - t0) / denom, 0.0, 1.0)
    translations = (
        pose_table.translations[i0] * (1.0 - alpha[:, None])
        + pose_table.translations[i1] * alpha[:, None]
    )
    quaternions = _slerp(pose_table.quaternions[i0], pose_table.quaternions[i1], alpha)
    return translations, quaternions


def _slerp(
    q0: npt.NDArray[np.float64], q1: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    q0 = _normalize_quaternions(q0)
    q1 = _normalize_quaternions(q1)
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.abs(dot)
    close = dot > 0.9995
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, 1e-12)
    s1 = sin_theta / np.maximum(sin_theta_0, 1e-12)
    out = s0[:, None] * q0 + s1[:, None] * q1
    linear = q0 + alpha[:, None] * (q1 - q0)
    out = np.where(close[:, None], linear, out)
    return _normalize_quaternions(out)


def _quat_to_rotmat(quaternions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    q = _normalize_quaternions(quaternions)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def _normalize_quaternions(quaternions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)


def _summarize_source(
    *,
    source_name: str,
    current_keep: npt.NDArray[np.bool_],
    original_keep: npt.NDArray[np.bool_],
    current_x: npt.NDArray[np.int64],
    original_x: npt.NDArray[np.int64],
    ranges: npt.NDArray[np.float64],
) -> SourceMetrics:
    agree = current_keep == original_keep
    false_keep = current_keep & ~original_keep
    false_drop = ~current_keep & original_keep
    union = current_keep | original_keep
    abs_shift = np.abs(((current_x - original_x + 900) % 1800) - 900) * (360.0 / 1800.0)
    return SourceMetrics(
        source_name=source_name,
        num_points=int(current_keep.size),
        current_kept=int(current_keep.sum()),
        original_kept=int(original_keep.sum()),
        agree=int(agree.sum()),
        false_keep=int(false_keep.sum()),
        false_drop=int(false_drop.sum()),
        jaccard=float((current_keep & original_keep).sum() / max(1, union.sum())),
        mismatch_rate=float((~agree).sum() / max(1, agree.size)),
        mean_abs_azimuth_shift_deg=float(np.mean(abs_shift)) if abs_shift.size else 0.0,
        p95_abs_azimuth_shift_deg=float(np.percentile(abs_shift, 95)) if abs_shift.size else 0.0,
        mean_range_m=float(np.mean(ranges)) if ranges.size else 0.0,
    )


def _build_aggregate(source_totals: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    output = []
    for source_name, totals in sorted(source_totals.items()):
        num_points = max(1.0, totals["num_points"])
        union = totals["current_kept"] + totals["false_drop"]
        intersection = totals["current_kept"] - totals["false_keep"]
        output.append(
            {
                "source_name": source_name,
                "num_points": int(totals["num_points"]),
                "current_kept": int(totals["current_kept"]),
                "original_kept": int(totals["original_kept"]),
                "false_keep": int(totals["false_keep"]),
                "false_drop": int(totals["false_drop"]),
                "mismatch_rate": float((totals["false_keep"] + totals["false_drop"]) / num_points),
                "jaccard": float(intersection / max(1.0, union)),
                "mean_abs_azimuth_shift_deg": float(totals["azimuth_shift_sum"] / num_points),
                "mean_range_m": float(totals["range_sum"] / num_points),
            }
        )
    return output


def _write_csv(output_csv: Path, frame_results: Sequence[Mapping[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_index",
        "token",
        "scene_token",
        "scene_name",
        "timestamp",
        "lidar_path",
        "source_name",
        "num_points",
        "current_kept",
        "original_kept",
        "agree",
        "false_keep",
        "false_drop",
        "jaccard",
        "mismatch_rate",
        "mean_abs_azimuth_shift_deg",
        "p95_abs_azimuth_shift_deg",
        "mean_range_m",
    ]
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frame_results:
            common = {key: frame.get(key) for key in fieldnames if key in frame}
            for source in frame["sources"]:
                writer.writerow(common | source)


def _load_records(ann_file: Path) -> list[dict[str, Any]]:
    with ann_file.open("rb") as file:
        data = pickle.load(file)
    if isinstance(data, dict) and "data_list" in data:
        return list(data["data_list"])
    if isinstance(data, list):
        return data
    raise TypeError(f"Unsupported annotation pickle shape: {type(data).__name__}")


def _load_pose_table(path: Path) -> PoseTable:
    poses = json.loads(path.read_text())
    times = np.asarray([pose["timestamp"] * 1e-6 for pose in poses], dtype=np.float64)
    translations = np.asarray([pose["translation"] for pose in poses], dtype=np.float64)
    quaternions = np.asarray([pose["rotation"] for pose in poses], dtype=np.float64)
    order = np.argsort(times)
    return PoseTable(times[order], translations[order], _normalize_quaternions(quaternions[order]))


def _dataset_scene_dir(data_root: Path, lidar_points: Mapping[str, Any]) -> Path:
    lidar_path = Path(lidar_points.get("lidar_path") or lidar_points["filename"])
    if len(lidar_path.parts) < 3:
        raise ValueError(f"Expected T4 lidar path with dataset/uuid/crop prefix: {lidar_path}")
    return (data_root / lidar_path.parts[0] / lidar_path.parts[1] / lidar_path.parts[2]).resolve()


def _feature_dim(feature_names: Sequence[str], name: str, *, default: int) -> int:
    return feature_names.index(name) if name in feature_names else default


def _stamp_to_seconds(stamp: Mapping[str, int]) -> float:
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9


def _point_offsets_seconds(
    timestamps: npt.NDArray[np.float32], point_timestamp_unit: str
) -> npt.NDArray[np.float64]:
    if point_timestamp_unit == "ns":
        return timestamps.astype(np.float64) * 1e-9
    if point_timestamp_unit == "us":
        return timestamps.astype(np.float64) * 1e-6
    if point_timestamp_unit == "s":
        return timestamps.astype(np.float64)
    raise ValueError(f"Unsupported timestamp unit: {point_timestamp_unit}")


if __name__ == "__main__":
    main()
