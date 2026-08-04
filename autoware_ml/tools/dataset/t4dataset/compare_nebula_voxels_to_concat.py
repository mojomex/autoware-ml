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

"""Compare voxelized Nebula-mask approximations to concatenated rosbag pointclouds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from mcap.reader import make_reader
from scipy.spatial import cKDTree

from autoware_ml.tools.dataset.t4dataset.compare_nebula_order_to_raw import (
    _CdrReader,
    _expand_scene_records,
    _load_records,
)
from autoware_ml.tools.dataset.t4dataset.nebula_order_error import (
    _corrected_to_raw_ego,
    _dataset_scene_dir,
    _ego_to_lidar_points,
    _feature_dim,
    _load_pose_table,
    _mask_keep,
    _point_offsets_seconds,
    _stamp_to_seconds,
)
from autoware_ml.transforms.point_cloud.filters import (
    NebulaDownsampleMaskFilter,
    RingOutlierFilter,
    _normalize_lidar_name,
)


_CONCAT_TOPIC = "/sensing/lidar/concatenated/pointcloud"
_METHODS = (
    "filter_only",
    "inverse_filter",
    "ring_only",
    "filter_only_ring",
    "inverse_filter_ring",
)


@dataclass(frozen=True)
class ConcatPointCloud:
    stamp: float
    points: npt.NDArray[np.float32]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/t4dataset"))
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path(
            "/workspace/data/t4dataset/info/segdet3d/t4dataset_j6gen2_segdet3d_infos_val.pkl"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--scene-frame-count",
        type=int,
        default=0,
        help="Expand the first selected T4 scene to this many 10 Hz intermediate frames.",
    )
    parser.add_argument("--max-stamp-diff-sec", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Comma-separated expanded-frame indices or ranges, e.g. 0-30,42.",
    )
    parser.add_argument(
        "--exclude-frame-indices",
        type=str,
        default=None,
        help="Comma-separated expanded-frame indices or ranges to skip.",
    )
    parser.add_argument(
        "--exclude-radius-m",
        type=float,
        default=0.0,
        help="Drop points with xy radius below this value before counting/voxelizing.",
    )
    parser.add_argument("--icp-align", action="store_true")
    parser.add_argument(
        "--icp-source",
        choices=("t4_unfiltered", "candidate"),
        default="t4_unfiltered",
        help="Estimate one ICP transform from T4 unfiltered, or separately per filtered candidate.",
    )
    parser.add_argument("--icp-max-points", type=int, default=50_000)
    parser.add_argument("--icp-max-iterations", type=int, default=20)
    parser.add_argument("--icp-max-correspondence-m", type=float, default=1.5)
    parser.add_argument("--ring-distance-ratio", type=float, default=1.1)
    parser.add_argument("--ring-object-length-threshold", type=float, default=0.05)
    args = parser.parse_args()

    concat_clouds = _load_concat_pointclouds(args.mcap, [_CONCAT_TOPIC])[_CONCAT_TOPIC]
    records = _load_records(args.ann_file)
    selected_records = records[args.start_index :: args.frame_stride][: args.max_frames]
    if args.scene_frame_count > 0:
        selected_records = _expand_scene_records(
            selected_records[0], args.data_root, args.scene_frame_count
        )
    selected_records = _filter_records_by_index(
        selected_records,
        include=_parse_index_spec(args.frame_indices),
        exclude=_parse_index_spec(args.exclude_frame_indices),
    )
    filter_impl = NebulaDownsampleMaskFilter()

    frame_rows = []
    totals: dict[str, float] = defaultdict(float)
    for input_index, record in zip(
        range(
            args.start_index,
            args.start_index + len(selected_records) * args.frame_stride,
            args.frame_stride,
        ),
        selected_records,
        strict=True,
    ):
        row = _compare_record(
            record=record,
            input_index=input_index,
            data_root=args.data_root,
            concat_clouds=concat_clouds,
            filter_impl=filter_impl,
            ring_distance_ratio=args.ring_distance_ratio,
            ring_object_length_threshold=args.ring_object_length_threshold,
            max_stamp_diff_sec=args.max_stamp_diff_sec,
            voxel_size=args.voxel_size,
            exclude_radius_m=args.exclude_radius_m,
            icp_align=args.icp_align,
            icp_source=args.icp_source,
            icp_max_points=args.icp_max_points,
            icp_max_iterations=args.icp_max_iterations,
            icp_max_correspondence_m=args.icp_max_correspondence_m,
        )
        if row is None:
            continue
        frame_rows.append(row)
        _accumulate(totals, row)

    aggregate = _summarize(totals)
    output = {
        "mcap": str(args.mcap),
        "ann_file": str(args.ann_file),
        "data_root": str(args.data_root),
        "concat_topic": _CONCAT_TOPIC,
        "voxel_size_m": args.voxel_size,
        "max_stamp_diff_sec": args.max_stamp_diff_sec,
        "exclude_radius_m": args.exclude_radius_m,
        "icp_align": args.icp_align,
        "icp_source": args.icp_source,
        "icp_max_points": args.icp_max_points,
        "icp_max_iterations": args.icp_max_iterations,
        "icp_max_correspondence_m": args.icp_max_correspondence_m,
        "ring_distance_ratio": args.ring_distance_ratio,
        "ring_object_length_threshold": args.ring_object_length_threshold,
        "num_frames_requested": len(selected_records),
        "num_frames_matched": len(frame_rows),
        "aggregate": aggregate,
        "frames": frame_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "num_frames_requested": len(selected_records),
                "num_frames_matched": len(frame_rows),
                "aggregate": aggregate,
            },
            indent=2,
        )
    )


def _compare_record(
    *,
    record: Mapping[str, Any],
    input_index: int,
    data_root: Path,
    concat_clouds: Sequence[ConcatPointCloud],
    filter_impl: NebulaDownsampleMaskFilter,
    ring_distance_ratio: float,
    ring_object_length_threshold: float,
    max_stamp_diff_sec: float,
    voxel_size: float,
    exclude_radius_m: float,
    icp_align: bool,
    icp_source: str,
    icp_max_points: int,
    icp_max_iterations: int,
    icp_max_correspondence_m: float,
) -> dict[str, Any] | None:
    lidar_points = record["lidar_points"]
    point_path = data_root / (lidar_points.get("lidar_path") or lidar_points["filename"])
    load_dim = int(lidar_points.get("num_pts_feats", 7))
    points = np.fromfile(point_path, dtype=np.float32).reshape(-1, load_dim)
    feature_names = list(lidar_points.get("feature_names", []))
    channel_dim = _feature_dim(feature_names, "channel", default=4)
    timestamp_dim = _feature_dim(feature_names, "timestamp", default=6)

    reference_time = _stamp_to_seconds(record["lidar_sources_info"]["stamp"])
    concat = _nearest_concat(concat_clouds, reference_time)
    stamp_diff = abs(concat.stamp - reference_time)
    if stamp_diff > max_stamp_diff_sec:
        return None

    dataset_scene_dir = _dataset_scene_dir(data_root, lidar_points)
    pose_table = _load_pose_table(dataset_scene_dir / "annotation" / "ego_pose.json")
    token_to_source = {
        str(source_meta["sensor_token"]): (str(source_name), source_meta)
        for source_name, source_meta in record["lidar_sources"].items()
    }

    method_keep = {
        "filter_only": np.zeros(points.shape[0], dtype=bool),
        "inverse_filter": np.zeros(points.shape[0], dtype=bool),
        "ring_only": np.zeros(points.shape[0], dtype=bool),
        "filter_only_ring": np.zeros(points.shape[0], dtype=bool),
        "inverse_filter_ring": np.zeros(points.shape[0], dtype=bool),
    }
    ring_filter = RingOutlierFilter(
        distance_ratio=ring_distance_ratio,
        object_length_threshold=ring_object_length_threshold,
        channel_dim=channel_dim,
    )
    for source_info in record["lidar_sources_info"]["sources"]:
        sensor_token = str(source_info["sensor_token"])
        if sensor_token not in token_to_source:
            continue
        source_name, source_meta = token_to_source[sensor_token]
        idx_begin = int(source_info["idx_begin"])
        idx_end = idx_begin + int(source_info["length"])
        source_points = points[idx_begin:idx_end]
        if source_points.size == 0:
            continue

        current_local = _ego_to_lidar_points(
            source_points[:, :3],
            np.asarray(source_meta["translation"], dtype=np.float64),
            np.asarray(source_meta["rotation"], dtype=np.float64),
        )
        source_time = _stamp_to_seconds(source_info["stamp"])
        point_times = source_time + _point_offsets_seconds(source_points[:, timestamp_dim], "ns")
        raw_ego = _corrected_to_raw_ego(
            source_points[:, :3], reference_time, point_times, pose_table
        )
        raw_local = _ego_to_lidar_points(
            raw_ego,
            np.asarray(source_meta["translation"], dtype=np.float64),
            np.asarray(source_meta["rotation"], dtype=np.float64),
        )
        channels = source_points[:, channel_dim].astype(np.int64)
        source_filter_only_keep, _ = _mask_keep(
            filter_impl, _normalize_lidar_name(source_name), current_local, channels
        )
        source_inverse_filter_keep, _ = _mask_keep(
            filter_impl, _normalize_lidar_name(source_name), raw_local, channels
        )
        source_ring_keep = ring_filter._ring_outlier_keep_for_local_points(current_local, channels)
        source_filter_only_ring_keep = _apply_ring_after_mask(
            ring_filter, current_local, channels, source_filter_only_keep
        )
        source_inverse_filter_ring_keep = _apply_ring_after_mask(
            ring_filter, current_local, channels, source_inverse_filter_keep
        )
        method_keep["filter_only"][idx_begin:idx_end] = source_filter_only_keep
        method_keep["inverse_filter"][idx_begin:idx_end] = source_inverse_filter_keep
        method_keep["ring_only"][idx_begin:idx_end] = source_ring_keep
        method_keep["filter_only_ring"][idx_begin:idx_end] = source_filter_only_ring_keep
        method_keep["inverse_filter_ring"][idx_begin:idx_end] = source_inverse_filter_ring_keep

    raw_points = _exclude_ego_radius(concat.points, exclude_radius_m)
    t4_points = _exclude_ego_radius(points[:, :3], exclude_radius_m)
    method_points = {
        method: _exclude_ego_radius(points[keep, :3], exclude_radius_m)
        for method, keep in method_keep.items()
    }

    icp_metrics: dict[str, Any] = {}
    if icp_align:
        if icp_source == "candidate":
            filter_only_icp = _icp_align(
                source=method_points["filter_only"],
                target=raw_points,
                max_points=icp_max_points,
                max_iterations=icp_max_iterations,
                max_correspondence_m=icp_max_correspondence_m,
            )
            inverse_filter_icp = _icp_align(
                source=method_points["inverse_filter"],
                target=raw_points,
                max_points=icp_max_points,
                max_iterations=icp_max_iterations,
                max_correspondence_m=icp_max_correspondence_m,
            )
            method_points["filter_only"] = _transform_points(
                method_points["filter_only"], filter_only_icp.rotation, filter_only_icp.translation
            )
            method_points["inverse_filter"] = _transform_points(
                method_points["inverse_filter"],
                inverse_filter_icp.rotation,
                inverse_filter_icp.translation,
            )
            icp_metrics = {
                **_icp_metrics("filter_only", filter_only_icp),
                **_icp_metrics("inverse_filter", inverse_filter_icp),
            }
        else:
            icp_result = _icp_align(
                source=t4_points,
                target=raw_points,
                max_points=icp_max_points,
                max_iterations=icp_max_iterations,
                max_correspondence_m=icp_max_correspondence_m,
            )
            t4_points = _transform_points(t4_points, icp_result.rotation, icp_result.translation)
            method_points = {
                method: _transform_points(
                    method_points_value, icp_result.rotation, icp_result.translation
                )
                for method, method_points_value in method_points.items()
            }
            icp_metrics = _icp_metrics("shared", icp_result)

    raw_voxels = _voxelize(raw_points, voxel_size)
    method_voxels = {
        method: _voxelize(method_points_value, voxel_size)
        for method, method_points_value in method_points.items()
    }
    return {
        "input_index": input_index,
        "token": str(record.get("token")),
        "reference_stamp": reference_time,
        "concat_stamp": concat.stamp,
        "stamp_diff_sec": stamp_diff,
        "raw_points": int(raw_points.shape[0]),
        "raw_voxels": int(raw_voxels.shape[0]),
        "t4_unfiltered_points": int(t4_points.shape[0]),
        **{
            f"{method}_points": int(method_points_value.shape[0])
            for method, method_points_value in method_points.items()
        },
        **icp_metrics,
        **{
            key: value
            for method, voxels in method_voxels.items()
            for key, value in _method_metrics(method, raw_voxels, voxels).items()
        },
    }


def _method_metrics(
    prefix: str, reference_voxels: npt.NDArray[np.void], predicted_voxels: npt.NDArray[np.void]
) -> dict[str, int | float]:
    intersection = int(np.intersect1d(reference_voxels, predicted_voxels).shape[0])
    union = int(reference_voxels.shape[0] + predicted_voxels.shape[0] - intersection)
    false_positive = int(predicted_voxels.shape[0] - intersection)
    false_negative = int(reference_voxels.shape[0] - intersection)
    return {
        f"{prefix}_voxels": int(predicted_voxels.shape[0]),
        f"{prefix}_voxel_intersection": intersection,
        f"{prefix}_voxel_union": union,
        f"{prefix}_voxel_false_positive": false_positive,
        f"{prefix}_voxel_false_negative": false_negative,
        f"{prefix}_voxel_precision": intersection / max(1, predicted_voxels.shape[0]),
        f"{prefix}_voxel_recall": intersection / max(1, reference_voxels.shape[0]),
        f"{prefix}_voxel_iou": intersection / max(1, union),
    }


def _apply_ring_after_mask(
    ring_filter: RingOutlierFilter,
    local_points: npt.NDArray[np.float32],
    channels: npt.NDArray[np.int64],
    mask_keep: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    keep = np.zeros(mask_keep.shape[0], dtype=bool)
    masked_indices = np.flatnonzero(mask_keep)
    if masked_indices.size == 0:
        return keep
    ring_keep = ring_filter._ring_outlier_keep_for_local_points(
        local_points[masked_indices], channels[masked_indices]
    )
    keep[masked_indices[ring_keep]] = True
    return keep


def _accumulate(totals: dict[str, float], row: Mapping[str, Any]) -> None:
    totals["frames"] += 1
    for key in (
        "raw_points",
        "raw_voxels",
        "t4_unfiltered_points",
    ):
        totals[key] += float(row[key])
    for method in _METHODS:
        for suffix in (
            "points",
            "voxels",
            "voxel_intersection",
            "voxel_union",
            "voxel_false_positive",
            "voxel_false_negative",
        ):
            totals[f"{method}_{suffix}"] += float(row[f"{method}_{suffix}"])
    totals["stamp_diff_sum_sec"] += float(row["stamp_diff_sec"])
    if "icp_rmse_m" in row:
        totals["icp_rmse_sum_m"] += float(row["icp_rmse_m"])
        totals["icp_correspondences"] += float(row["icp_correspondences"])
        totals["icp_rotation_sum_deg"] += float(row["icp_rotation_deg"])
        totals["icp_translation_sum_m"] += float(np.linalg.norm(row["icp_translation_m"]))
    for prefix in ("filter_only", "inverse_filter", "shared"):
        key = f"{prefix}_icp_rmse_m"
        if key in row:
            totals[f"{prefix}_icp_rmse_sum_m"] += float(row[key])
            totals[f"{prefix}_icp_correspondences"] += float(row[f"{prefix}_icp_correspondences"])
            totals[f"{prefix}_icp_rotation_sum_deg"] += float(row[f"{prefix}_icp_rotation_deg"])
            totals[f"{prefix}_icp_translation_sum_m"] += float(
                np.linalg.norm(row[f"{prefix}_icp_translation_m"])
            )


def _summarize(totals: Mapping[str, float]) -> dict[str, Any]:
    frames = max(1.0, totals["frames"])
    raw_points = totals["raw_points"]
    output: dict[str, Any] = {
        "frames": int(totals["frames"]),
        "raw_points": int(raw_points),
        "raw_voxels": int(totals["raw_voxels"]),
        "t4_unfiltered_points": int(totals["t4_unfiltered_points"]),
        "mean_stamp_diff_ms": 1000.0 * totals["stamp_diff_sum_sec"] / frames,
    }
    if "icp_rmse_sum_m" in totals:
        output.update(
            {
                "mean_icp_rmse_m": totals["icp_rmse_sum_m"] / frames,
                "mean_icp_correspondences": totals["icp_correspondences"] / frames,
                "mean_icp_rotation_deg": totals["icp_rotation_sum_deg"] / frames,
                "mean_icp_translation_norm_m": totals["icp_translation_sum_m"] / frames,
            }
        )
    for prefix in ("filter_only", "inverse_filter", "shared"):
        if f"{prefix}_icp_rmse_sum_m" in totals:
            output.update(
                {
                    f"{prefix}_mean_icp_rmse_m": totals[f"{prefix}_icp_rmse_sum_m"] / frames,
                    f"{prefix}_mean_icp_correspondences": totals[f"{prefix}_icp_correspondences"]
                    / frames,
                    f"{prefix}_mean_icp_rotation_deg": totals[f"{prefix}_icp_rotation_sum_deg"]
                    / frames,
                    f"{prefix}_mean_icp_translation_norm_m": totals[
                        f"{prefix}_icp_translation_sum_m"
                    ]
                    / frames,
                }
            )
    for prefix in _METHODS:
        predicted_points = totals[f"{prefix}_points"]
        predicted_voxels = totals[f"{prefix}_voxels"]
        intersection = totals[f"{prefix}_voxel_intersection"]
        union = totals[f"{prefix}_voxel_union"]
        output.update(
            {
                f"{prefix}_points": int(predicted_points),
                f"{prefix}_point_count_error_pct": 100.0
                * (predicted_points - raw_points)
                / max(1.0, raw_points),
                f"{prefix}_voxels": int(predicted_voxels),
                f"{prefix}_voxel_intersection": int(intersection),
                f"{prefix}_voxel_union": int(union),
                f"{prefix}_voxel_false_positive": int(totals[f"{prefix}_voxel_false_positive"]),
                f"{prefix}_voxel_false_negative": int(totals[f"{prefix}_voxel_false_negative"]),
                f"{prefix}_voxel_precision": intersection / max(1.0, predicted_voxels),
                f"{prefix}_voxel_recall": intersection / max(1.0, totals["raw_voxels"]),
                f"{prefix}_voxel_iou": intersection / max(1.0, union),
            }
        )
    return output


def _load_concat_pointclouds(
    mcap_path: Path, topics: Iterable[str]
) -> dict[str, list[ConcatPointCloud]]:
    requested_topics = set(topics)
    output = {topic: [] for topic in requested_topics}
    with mcap_path.open("rb") as file:
        reader = make_reader(file)
        for _, channel, message in reader.iter_messages(topics=requested_topics):
            output[channel.topic].append(_parse_concat_pointcloud2(message.data))
    for topic, messages in output.items():
        messages.sort(key=lambda msg: msg.stamp)
        if not messages:
            raise ValueError(f"No messages found for topic {topic}")
    return output


def _parse_concat_pointcloud2(data: bytes) -> ConcatPointCloud:
    reader = _CdrReader(data)
    sec = reader.i32()
    nanosec = reader.u32()
    reader.string()
    height = reader.u32()
    width = reader.u32()
    fields = {}
    for _ in range(reader.u32()):
        name = reader.string()
        fields[name] = (reader.u32(), reader.u8(), reader.u32())
    reader.bool()
    point_step = reader.u32()
    reader.u32()
    blob = reader.byte_sequence()
    reader.bool()
    count = int(width * height)
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [fields["x"][0], fields["y"][0], fields["z"][0]],
            "itemsize": point_step,
        }
    )
    raw = np.frombuffer(blob, dtype=dtype, count=count)
    points = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float32, copy=False)
    finite = np.isfinite(points).all(axis=1)
    return ConcatPointCloud(stamp=sec + nanosec * 1e-9, points=points[finite])


def _nearest_concat(messages: Sequence[ConcatPointCloud], stamp: float) -> ConcatPointCloud:
    stamps = np.asarray([message.stamp for message in messages], dtype=np.float64)
    index = int(np.abs(stamps - stamp).argmin())
    return messages[index]


@dataclass(frozen=True)
class IcpResult:
    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]
    rmse: float
    correspondences: int


def _icp_metrics(prefix: str, result: IcpResult) -> dict[str, Any]:
    return {
        f"{prefix}_icp_rmse_m": result.rmse,
        f"{prefix}_icp_correspondences": result.correspondences,
        f"{prefix}_icp_translation_m": result.translation.tolist(),
        f"{prefix}_icp_rotation_deg": _rotation_angle_deg(result.rotation),
    }


def _icp_align(
    *,
    source: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    max_points: int,
    max_iterations: int,
    max_correspondence_m: float,
) -> IcpResult:
    source_sample = _deterministic_sample(source, max_points).astype(np.float64)
    target_sample = _deterministic_sample(target, max_points).astype(np.float64)
    if source_sample.shape[0] < 3 or target_sample.shape[0] < 3:
        return IcpResult(np.eye(3), np.zeros(3), float("nan"), 0)

    tree = cKDTree(target_sample)
    rotation = np.eye(3)
    translation = np.zeros(3)
    rmse = float("inf")
    correspondences = 0
    for _ in range(max_iterations):
        transformed = source_sample @ rotation.T + translation
        distances, indices = tree.query(transformed, k=1, workers=-1)
        valid = distances <= max_correspondence_m
        if int(valid.sum()) < 3:
            break
        correspondences = int(valid.sum())
        matched_source = transformed[valid]
        matched_target = target_sample[indices[valid]]
        delta_rotation, delta_translation = _best_fit_transform(matched_source, matched_target)
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        rmse = float(np.sqrt(np.mean(distances[valid] ** 2)))
        if np.linalg.norm(delta_translation) < 1e-4 and _rotation_angle_deg(delta_rotation) < 1e-3:
            break
    return IcpResult(rotation, translation, rmse, correspondences)


def _best_fit_transform(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _rotation_angle_deg(rotation: npt.NDArray[np.float64]) -> float:
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def _transform_points(
    points: npt.NDArray[np.float32],
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    if points.size == 0:
        return points
    return (points.astype(np.float64) @ rotation.T + translation).astype(np.float32)


def _exclude_ego_radius(
    points: npt.NDArray[np.float32], radius_m: float
) -> npt.NDArray[np.float32]:
    if radius_m <= 0.0 or points.size == 0:
        return points
    radius = np.linalg.norm(points[:, :2], axis=1)
    return points[radius >= radius_m]


def _deterministic_sample(
    points: npt.NDArray[np.float32], max_points: int
) -> npt.NDArray[np.float32]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    stride = int(np.ceil(points.shape[0] / max_points))
    return points[::stride][:max_points]


def _voxelize(points: npt.NDArray[np.float32], voxel_size: float) -> npt.NDArray[np.void]:
    if points.size == 0:
        return np.empty(0, dtype=np.dtype("V12"))
    coords = np.floor(points[:, :3].astype(np.float64) / voxel_size).astype(np.int32)
    coords = np.ascontiguousarray(coords)
    packed = coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).reshape(-1)
    return np.unique(packed)


def _filter_records_by_index(
    records: Sequence[dict[str, Any]], include: set[int] | None, exclude: set[int] | None
) -> list[dict[str, Any]]:
    output = []
    for index, record in enumerate(records):
        if include is not None and index not in include:
            continue
        if exclude is not None and index in exclude:
            continue
        output.append(record)
    return output


def _parse_index_spec(spec: str | None) -> set[int] | None:
    if spec is None or spec == "":
        return None
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


if __name__ == "__main__":
    main()
