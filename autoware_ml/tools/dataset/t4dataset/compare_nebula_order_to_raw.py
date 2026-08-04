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

"""Compare T4 Nebula-mask approximations against raw filtered LiDAR MCAP messages."""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from mcap.reader import make_reader

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
    _normalize_lidar_name,
)


_SOURCE_TO_TOPIC = {
    "LIDAR_FRONT_UPPER": "/sensing/lidar/front_upper/pointcloud_raw_ex",
    "LIDAR_FRONT_LOWER": "/sensing/lidar/front_lower/pointcloud_raw_ex",
    "LIDAR_LEFT_UPPER": "/sensing/lidar/left_upper/pointcloud_raw_ex",
    "LIDAR_LEFT_LOWER": "/sensing/lidar/left_lower/pointcloud_raw_ex",
    "LIDAR_REAR_UPPER": "/sensing/lidar/rear_upper/pointcloud_raw_ex",
    "LIDAR_REAR_LOWER": "/sensing/lidar/rear_lower/pointcloud_raw_ex",
    "LIDAR_RIGHT_UPPER": "/sensing/lidar/right_upper/pointcloud_raw_ex",
    "LIDAR_RIGHT_LOWER": "/sensing/lidar/right_lower/pointcloud_raw_ex",
}


@dataclass(frozen=True)
class RawPointCloud:
    stamp: float
    topic: str
    width: int
    points: npt.NDArray[np.void]
    channels: npt.NDArray[np.int64]
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    z: npt.NDArray[np.float32]


class _CdrReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 4

    def align(self, alignment: int) -> None:
        self.offset = (self.offset + alignment - 1) & ~(alignment - 1)

    def bool(self) -> bool:
        return bool(self.u8())

    def u8(self) -> int:
        self.align(1)
        value = self.data[self.offset]
        self.offset += 1
        return value

    def i32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return value

    def u32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def string(self) -> str:
        size = self.u32()
        value = self.data[self.offset : self.offset + size]
        self.offset += size
        if value.endswith(b"\0"):
            value = value[:-1]
        return value.decode("utf-8", "replace")

    def byte_sequence(self) -> bytes:
        size = self.u32()
        value = self.data[self.offset : self.offset + size]
        self.offset += size
        return value


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
    parser.add_argument("--skip-identity", action="store_true")
    parser.add_argument("--mask-space-distance", action="store_true")
    parser.add_argument(
        "--vehicle-calibration-root",
        type=Path,
        default=None,
        help="Directory containing per-LiDAR calibration CSVs named front_upper.csv, etc.",
    )
    args = parser.parse_args()

    raw_by_topic = _load_raw_pointclouds(args.mcap, _SOURCE_TO_TOPIC.values())
    records = _load_records(args.ann_file)
    selected_records = records[args.start_index :: args.frame_stride][: args.max_frames]
    if args.scene_frame_count > 0:
        selected_records = _expand_scene_records(
            selected_records[0], args.data_root, args.scene_frame_count
        )
    filter_impl = _make_filter(args.vehicle_calibration_root)

    frame_rows = []
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for input_index, record in zip(
        range(
            args.start_index,
            args.start_index + len(selected_records) * args.frame_stride,
            args.frame_stride,
        ),
        selected_records,
        strict=True,
    ):
        for row in _compare_record(
            record=record,
            input_index=input_index,
            data_root=args.data_root,
            raw_by_topic=raw_by_topic,
            filter_impl=filter_impl,
            max_stamp_diff_sec=args.max_stamp_diff_sec,
            skip_identity=args.skip_identity,
            mask_space_distance=args.mask_space_distance,
        ):
            frame_rows.append(row)
            totals_for_source = totals[row["source_name"]]
            for key in (
                "raw_kept",
                "t4_unfiltered",
                "filter_only_kept",
                "inverse_filter_kept",
            ):
                totals_for_source[key] += row[key]
            for key in ("filter_only_intersection", "inverse_filter_intersection"):
                if row[key] is not None:
                    totals_for_source[key] += row[key]
            for key in (
                "filter_only_mask_chamfer_bins_sum",
                "inverse_filter_mask_chamfer_bins_sum",
                "filter_only_mask_chamfer_degrees_sum",
                "inverse_filter_mask_chamfer_degrees_sum",
            ):
                if row[key] is not None:
                    totals_for_source[key] += row[key]
            totals_for_source["frames"] += 1
            totals_for_source["stamp_diff_sum_sec"] += row["stamp_diff_sec"]

    aggregate = [
        _summarize_source(source_name, values) for source_name, values in sorted(totals.items())
    ]
    output = {
        "mcap": str(args.mcap),
        "ann_file": str(args.ann_file),
        "data_root": str(args.data_root),
        "vehicle_calibration_root": None
        if args.vehicle_calibration_root is None
        else str(args.vehicle_calibration_root),
        "num_frames": len(selected_records),
        "aggregate": aggregate,
        "frames": frame_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(json.dumps({"num_frames": len(selected_records), "aggregate": aggregate}, indent=2))


def _make_filter(vehicle_calibration_root: Path | None) -> NebulaDownsampleMaskFilter:
    if vehicle_calibration_root is None:
        return NebulaDownsampleMaskFilter()
    lidar_names = [
        "front_upper",
        "front_lower",
        "left_upper",
        "left_lower",
        "rear_upper",
        "rear_lower",
        "right_upper",
        "right_lower",
    ]
    return NebulaDownsampleMaskFilter(
        calibration_root=str(vehicle_calibration_root),
        lidar_name_to_calibration={lidar_name: f"{lidar_name}.csv" for lidar_name in lidar_names},
    )


def _compare_record(
    *,
    record: Mapping[str, Any],
    input_index: int,
    data_root: Path,
    raw_by_topic: Mapping[str, Sequence[RawPointCloud]],
    filter_impl: NebulaDownsampleMaskFilter,
    max_stamp_diff_sec: float,
    skip_identity: bool,
    mask_space_distance: bool,
) -> list[dict[str, Any]]:
    lidar_points = record["lidar_points"]
    point_path = data_root / (lidar_points.get("lidar_path") or lidar_points["filename"])
    load_dim = int(lidar_points.get("num_pts_feats", 7))
    points = np.fromfile(point_path, dtype=np.float32).reshape(-1, load_dim)
    feature_names = list(lidar_points.get("feature_names", []))
    channel_dim = _feature_dim(feature_names, "channel", default=4)
    return_type_dim = _feature_dim(feature_names, "return_type", default=5)
    timestamp_dim = _feature_dim(feature_names, "timestamp", default=6)

    dataset_scene_dir = _dataset_scene_dir(data_root, lidar_points)
    pose_table = _load_pose_table(dataset_scene_dir / "annotation" / "ego_pose.json")
    reference_time = _stamp_to_seconds(record["lidar_sources_info"]["stamp"])
    token_to_source = {
        str(source_meta["sensor_token"]): (str(source_name), source_meta)
        for source_name, source_meta in record["lidar_sources"].items()
    }

    rows = []
    for source_info in record["lidar_sources_info"]["sources"]:
        sensor_token = str(source_info["sensor_token"])
        if sensor_token not in token_to_source:
            continue
        source_name, source_meta = token_to_source[sensor_token]
        if source_name not in _SOURCE_TO_TOPIC:
            continue
        topic = _SOURCE_TO_TOPIC[source_name]
        source_stamp = _stamp_to_seconds(source_info["stamp"])
        raw = _nearest_raw(raw_by_topic[topic], source_stamp)
        stamp_diff = abs(raw.stamp - source_stamp)
        if stamp_diff > max_stamp_diff_sec:
            continue

        idx_begin = int(source_info["idx_begin"])
        idx_end = idx_begin + int(source_info["length"])
        source_points = points[idx_begin:idx_end]
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
        filter_only_keep, filter_only_x = _mask_keep(
            filter_impl, _normalize_lidar_name(source_name), current_local, channels
        )
        inverse_filter_keep, inverse_filter_x = _mask_keep(
            filter_impl, _normalize_lidar_name(source_name), raw_local, channels
        )

        raw_ids = raw.points
        if skip_identity:
            filter_only_ids = None
            inverse_filter_ids = None
            raw_counter = None
        else:
            t4_ids = _t4_identity(source_points, channel_dim, return_type_dim, timestamp_dim)
            filter_only_ids = t4_ids[filter_only_keep]
            inverse_filter_ids = t4_ids[inverse_filter_keep]
            raw_counter = _identity_counter(raw_ids)

        rows.append(
            _build_row(
                input_index=input_index,
                token=str(record.get("token")),
                source_name=source_name,
                source_stamp=source_stamp,
                raw_stamp=raw.stamp,
                stamp_diff=stamp_diff,
                t4_unfiltered=len(source_points),
                raw_kept=len(raw_ids),
                filter_only_kept=int(filter_only_keep.sum()),
                inverse_filter_kept=int(inverse_filter_keep.sum()),
                filter_only_ids=filter_only_ids,
                inverse_filter_ids=inverse_filter_ids,
                raw_counter=raw_counter,
                raw_channels=raw.channels,
                raw_x=_raw_project_mask_x(filter_impl, source_name, raw),
                filter_only_channels=channels[filter_only_keep],
                filter_only_x=filter_only_x[filter_only_keep],
                inverse_filter_channels=channels[inverse_filter_keep],
                inverse_filter_x=inverse_filter_x[inverse_filter_keep],
                mask_width=_mask_width(filter_impl, source_name),
                mask_space_distance=mask_space_distance,
            )
        )
    return rows


def _build_row(
    *,
    input_index: int,
    token: str,
    source_name: str,
    source_stamp: float,
    raw_stamp: float,
    stamp_diff: float,
    t4_unfiltered: int,
    raw_kept: int,
    filter_only_kept: int,
    inverse_filter_kept: int,
    filter_only_ids: npt.NDArray[np.void] | None,
    inverse_filter_ids: npt.NDArray[np.void] | None,
    raw_counter: Counter[bytes] | None,
    raw_channels: npt.NDArray[np.int64],
    raw_x: npt.NDArray[np.int64],
    filter_only_channels: npt.NDArray[np.int64],
    filter_only_x: npt.NDArray[np.int64],
    inverse_filter_channels: npt.NDArray[np.int64],
    inverse_filter_x: npt.NDArray[np.int64],
    mask_width: int,
    mask_space_distance: bool,
) -> dict[str, Any]:
    filter_only_intersection = (
        None
        if raw_counter is None or filter_only_ids is None
        else _multiset_intersection_count(filter_only_ids, raw_counter)
    )
    inverse_filter_intersection = (
        None
        if raw_counter is None or inverse_filter_ids is None
        else _multiset_intersection_count(inverse_filter_ids, raw_counter)
    )
    filter_only_mask_chamfer_bins = (
        _mask_space_chamfer(raw_channels, raw_x, filter_only_channels, filter_only_x, mask_width)
        if mask_space_distance
        else None
    )
    inverse_filter_mask_chamfer_bins = (
        _mask_space_chamfer(
            raw_channels, raw_x, inverse_filter_channels, inverse_filter_x, mask_width
        )
        if mask_space_distance
        else None
    )
    filter_only_mask_chamfer_degrees = (
        None
        if filter_only_mask_chamfer_bins is None
        else filter_only_mask_chamfer_bins * 360.0 / mask_width
    )
    inverse_filter_mask_chamfer_degrees = (
        None
        if inverse_filter_mask_chamfer_bins is None
        else inverse_filter_mask_chamfer_bins * 360.0 / mask_width
    )
    return {
        "input_index": input_index,
        "token": token,
        "source_name": source_name,
        "source_stamp": source_stamp,
        "raw_stamp": raw_stamp,
        "stamp_diff_sec": stamp_diff,
        "t4_unfiltered": t4_unfiltered,
        "raw_kept": raw_kept,
        "filter_only_kept": filter_only_kept,
        "inverse_filter_kept": inverse_filter_kept,
        "filter_only_intersection": filter_only_intersection,
        "inverse_filter_intersection": inverse_filter_intersection,
        "filter_only_precision": None
        if filter_only_intersection is None
        else filter_only_intersection / max(1, filter_only_kept),
        "inverse_filter_precision": None
        if inverse_filter_intersection is None
        else inverse_filter_intersection / max(1, inverse_filter_kept),
        "filter_only_recall": None
        if filter_only_intersection is None
        else filter_only_intersection / max(1, raw_kept),
        "inverse_filter_recall": None
        if inverse_filter_intersection is None
        else inverse_filter_intersection / max(1, raw_kept),
        "filter_only_count_error": filter_only_kept - raw_kept,
        "inverse_filter_count_error": inverse_filter_kept - raw_kept,
        "filter_only_mask_chamfer_bins": filter_only_mask_chamfer_bins,
        "inverse_filter_mask_chamfer_bins": inverse_filter_mask_chamfer_bins,
        "filter_only_mask_chamfer_degrees": filter_only_mask_chamfer_degrees,
        "inverse_filter_mask_chamfer_degrees": inverse_filter_mask_chamfer_degrees,
        "filter_only_mask_chamfer_bins_sum": None
        if filter_only_mask_chamfer_bins is None
        else filter_only_mask_chamfer_bins * max(1, raw_kept + filter_only_kept),
        "inverse_filter_mask_chamfer_bins_sum": None
        if inverse_filter_mask_chamfer_bins is None
        else inverse_filter_mask_chamfer_bins * max(1, raw_kept + inverse_filter_kept),
        "filter_only_mask_chamfer_degrees_sum": None
        if filter_only_mask_chamfer_degrees is None
        else filter_only_mask_chamfer_degrees * max(1, raw_kept + filter_only_kept),
        "inverse_filter_mask_chamfer_degrees_sum": None
        if inverse_filter_mask_chamfer_degrees is None
        else inverse_filter_mask_chamfer_degrees * max(1, raw_kept + inverse_filter_kept),
    }


def _summarize_source(source_name: str, values: Mapping[str, float]) -> dict[str, Any]:
    raw_kept = values["raw_kept"]
    filter_only_kept = values["filter_only_kept"]
    inverse_filter_kept = values["inverse_filter_kept"]
    filter_only_intersection = values.get("filter_only_intersection")
    inverse_filter_intersection = values.get("inverse_filter_intersection")
    frames = max(1.0, values["frames"])
    filter_only_chamfer_weight = max(1.0, raw_kept + filter_only_kept)
    inverse_chamfer_weight = max(1.0, raw_kept + inverse_filter_kept)
    return {
        "source_name": source_name,
        "frames": int(values["frames"]),
        "raw_kept": int(raw_kept),
        "t4_unfiltered": int(values["t4_unfiltered"]),
        "filter_only_kept": int(filter_only_kept),
        "inverse_filter_kept": int(inverse_filter_kept),
        "filter_only_count_error_pct": 100.0 * (filter_only_kept - raw_kept) / max(1.0, raw_kept),
        "inverse_filter_count_error_pct": 100.0
        * (inverse_filter_kept - raw_kept)
        / max(1.0, raw_kept),
        "filter_only_precision": None
        if filter_only_intersection is None
        else filter_only_intersection / max(1.0, filter_only_kept),
        "inverse_filter_precision": None
        if inverse_filter_intersection is None
        else inverse_filter_intersection / max(1.0, inverse_filter_kept),
        "filter_only_recall": None
        if filter_only_intersection is None
        else filter_only_intersection / max(1.0, raw_kept),
        "inverse_filter_recall": None
        if inverse_filter_intersection is None
        else inverse_filter_intersection / max(1.0, raw_kept),
        "filter_only_mask_chamfer_bins": None
        if "filter_only_mask_chamfer_bins_sum" not in values
        else values["filter_only_mask_chamfer_bins_sum"] / filter_only_chamfer_weight,
        "inverse_filter_mask_chamfer_bins": None
        if "inverse_filter_mask_chamfer_bins_sum" not in values
        else values["inverse_filter_mask_chamfer_bins_sum"] / inverse_chamfer_weight,
        "filter_only_mask_chamfer_degrees": None
        if "filter_only_mask_chamfer_degrees_sum" not in values
        else values["filter_only_mask_chamfer_degrees_sum"] / filter_only_chamfer_weight,
        "inverse_filter_mask_chamfer_degrees": None
        if "inverse_filter_mask_chamfer_degrees_sum" not in values
        else values["inverse_filter_mask_chamfer_degrees_sum"] / inverse_chamfer_weight,
        "mean_stamp_diff_ms": 1000.0 * values["stamp_diff_sum_sec"] / frames,
    }


def _load_raw_pointclouds(mcap_path: Path, topics: Iterable[str]) -> dict[str, list[RawPointCloud]]:
    requested_topics = set(topics)
    output = {topic: [] for topic in requested_topics}
    with mcap_path.open("rb") as file:
        reader = make_reader(file)
        for _, channel, message in reader.iter_messages(topics=requested_topics):
            output[channel.topic].append(_parse_pointcloud2(message.data, channel.topic))
    for topic, messages in output.items():
        messages.sort(key=lambda msg: msg.stamp)
        if not messages:
            raise ValueError(f"No raw messages found for topic {topic}")
    return output


def _parse_pointcloud2(data: bytes, topic: str) -> RawPointCloud:
    reader = _CdrReader(data)
    sec = reader.i32()
    nanosec = reader.u32()
    reader.string()
    reader.u32()
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
    points = _raw_identity(blob, width, point_step, fields)
    channels, x, y, z = _raw_mask_coords(blob, width, point_step, fields)
    return RawPointCloud(
        stamp=sec + nanosec * 1e-9,
        topic=topic,
        width=width,
        points=points,
        channels=channels,
        x=x,
        y=y,
        z=z,
    )


def _raw_identity(
    blob: bytes, width: int, point_step: int, fields: Mapping[str, tuple[int, int, int]]
) -> npt.NDArray[np.void]:
    if width == 0:
        return np.empty(0, dtype=_identity_dtype())
    dtype = np.dtype(
        {
            "names": ["intensity", "return_type", "channel", "timestamp_bits"],
            "formats": ["u1", "u1", "<u2", "<u4"],
            "offsets": [
                fields["intensity"][0],
                fields["return_type"][0],
                fields["channel"][0],
                fields["time_stamp"][0],
            ],
            "itemsize": point_step,
        }
    )
    raw = np.frombuffer(blob, dtype=dtype, count=width)
    out = np.empty(width, dtype=_identity_dtype())
    out["intensity"] = raw["intensity"]
    out["return_type"] = raw["return_type"]
    out["channel"] = raw["channel"]
    out["timestamp_bits"] = raw["timestamp_bits"].astype(np.float32).view(np.uint32)
    return out


def _raw_mask_coords(
    blob: bytes, width: int, point_step: int, fields: Mapping[str, tuple[int, int, int]]
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    if width == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    channel_format = "<u2" if "channel" in fields else "<u2"
    channel_offset = fields["channel"][0] if "channel" in fields else fields["x"][0]
    dtype = np.dtype(
        {
            "names": ["x", "y", "z", "channel"],
            "formats": ["<f4", "<f4", "<f4", channel_format],
            "offsets": [fields["x"][0], fields["y"][0], fields["z"][0], channel_offset],
            "itemsize": point_step,
        }
    )
    raw = np.frombuffer(blob, dtype=dtype, count=width)
    channels = raw["channel"].astype(np.int64) if "channel" in fields else np.zeros(width, np.int64)
    return channels, raw["x"].copy(), raw["y"].copy(), raw["z"].copy()


def _t4_identity(
    points: npt.NDArray[np.float32],
    channel_dim: int,
    return_type_dim: int,
    timestamp_dim: int,
) -> npt.NDArray[np.void]:
    out = np.empty(points.shape[0], dtype=_identity_dtype())
    out["intensity"] = np.clip(np.rint(points[:, 3]), 0, 255).astype(np.uint8)
    out["return_type"] = np.clip(np.rint(points[:, return_type_dim]), 0, 255).astype(np.uint8)
    out["channel"] = np.clip(np.rint(points[:, channel_dim]), 0, 65535).astype(np.uint16)
    out["timestamp_bits"] = points[:, timestamp_dim].astype(np.float32).view(np.uint32)
    return out


def _identity_dtype() -> np.dtype:
    return np.dtype(
        [
            ("intensity", "u1"),
            ("return_type", "u1"),
            ("channel", "<u2"),
            ("timestamp_bits", "<u4"),
        ]
    )


def _identity_counter(ids: npt.NDArray[np.void]) -> Counter[bytes]:
    return Counter(
        ids.tobytes()[i : i + ids.dtype.itemsize] for i in range(0, ids.nbytes, ids.dtype.itemsize)
    )


def _multiset_intersection_count(ids: npt.NDArray[np.void], reference: Counter[bytes]) -> int:
    counts = _identity_counter(ids)
    return sum(min(count, reference[key]) for key, count in counts.items())


def _mask_width(filter_impl: NebulaDownsampleMaskFilter, source_name: str) -> int:
    lidar_name = _normalize_lidar_name(source_name)
    model_name = filter_impl.lidar_name_to_model[lidar_name]
    return int(filter_impl._load_mask(lidar_name, model_name).shape[1])


def _raw_project_mask_x(
    filter_impl: NebulaDownsampleMaskFilter, source_name: str, raw: RawPointCloud
) -> npt.NDArray[np.int64]:
    lidar_name = _normalize_lidar_name(source_name)
    calibration = filter_impl._load_calibration(lidar_name)
    mask_width = _mask_width(filter_impl, source_name)
    azimuth_deg = np.rad2deg(np.arctan2(raw.x.astype(np.float32), raw.y.astype(np.float32)))
    valid_channels = (raw.channels >= 0) & (raw.channels < calibration.azimuth_deg.shape[0])
    azimuth_deg = azimuth_deg.copy()
    azimuth_deg[valid_channels] = (
        azimuth_deg[valid_channels] - calibration.azimuth_deg[raw.channels[valid_channels]]
    )
    azimuth_deg = (azimuth_deg - filter_impl.azimuth_start_deg) % 360.0
    return np.floor(azimuth_deg / filter_impl.azimuth_extent_deg * mask_width + 0.5).astype(
        np.int64
    )


def _mask_space_chamfer(
    reference_channels: npt.NDArray[np.int64],
    reference_x: npt.NDArray[np.int64],
    predicted_channels: npt.NDArray[np.int64],
    predicted_x: npt.NDArray[np.int64],
    mask_width: int,
) -> float:
    reference_channels, reference_x = _unique_mask_cells(
        reference_channels, reference_x, mask_width
    )
    predicted_channels, predicted_x = _unique_mask_cells(
        predicted_channels, predicted_x, mask_width
    )
    if reference_x.size == 0 and predicted_x.size == 0:
        return 0.0
    if reference_x.size == 0 or predicted_x.size == 0:
        return float(mask_width / 2.0)
    ref_to_pred = _mean_nearest_circular_distance(
        reference_channels, reference_x, predicted_channels, predicted_x, mask_width
    )
    pred_to_ref = _mean_nearest_circular_distance(
        predicted_channels, predicted_x, reference_channels, reference_x, mask_width
    )
    return 0.5 * (ref_to_pred + pred_to_ref)


def _unique_mask_cells(
    channels: npt.NDArray[np.int64], x: npt.NDArray[np.int64], mask_width: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    if x.size == 0:
        return channels, x
    cells = np.unique(channels.astype(np.int64) * mask_width + (x.astype(np.int64) % mask_width))
    return cells // mask_width, cells % mask_width


def _mean_nearest_circular_distance(
    query_channels: npt.NDArray[np.int64],
    query_x: npt.NDArray[np.int64],
    target_channels: npt.NDArray[np.int64],
    target_x: npt.NDArray[np.int64],
    mask_width: int,
) -> float:
    distances = np.full(query_x.shape[0], mask_width / 2.0, dtype=np.float64)
    for channel in np.intersect1d(np.unique(query_channels), np.unique(target_channels)):
        query_indices = np.flatnonzero(query_channels == channel)
        target_values = np.sort(target_x[target_channels == channel] % mask_width)
        if target_values.size == 0:
            continue
        query_values = query_x[query_indices] % mask_width
        insert = np.searchsorted(target_values, query_values)
        right = target_values[insert % target_values.size]
        left = target_values[(insert - 1) % target_values.size]
        right_dist = np.minimum(
            np.abs(right - query_values), mask_width - np.abs(right - query_values)
        )
        left_dist = np.minimum(
            np.abs(query_values - left), mask_width - np.abs(query_values - left)
        )
        distances[query_indices] = np.minimum(left_dist, right_dist)
    return float(np.mean(distances))


def _nearest_raw(messages: Sequence[RawPointCloud], stamp: float) -> RawPointCloud:
    stamps = np.asarray([message.stamp for message in messages], dtype=np.float64)
    index = int(np.abs(stamps - stamp).argmin())
    return messages[index]


def _load_records(ann_file: Path) -> list[dict[str, Any]]:
    with ann_file.open("rb") as file:
        data = pickle.load(file)
    if isinstance(data, dict) and "data_list" in data:
        return list(data["data_list"])
    if isinstance(data, list):
        return data
    raise TypeError(f"Unsupported annotation pickle shape: {type(data).__name__}")


def _expand_scene_records(
    template_record: Mapping[str, Any], data_root: Path, scene_frame_count: int
) -> list[dict[str, Any]]:
    lidar_path = Path(template_record["lidar_points"]["lidar_path"])
    scene_dir = _dataset_scene_dir(data_root, template_record["lidar_points"])
    records = []
    for frame_index in range(scene_frame_count):
        record = dict(template_record)
        lidar_points = dict(template_record["lidar_points"])
        lidar_points["lidar_path"] = str(
            Path(*lidar_path.parts[:3]) / "data" / "LIDAR_CONCAT" / f"{frame_index:05d}.pcd.bin"
        )
        info_path = scene_dir / "data" / "LIDAR_CONCAT_INFO" / f"{frame_index:05d}.json"
        lidar_sources_info = json.loads(info_path.read_text())
        stamp = lidar_sources_info["stamp"]
        record["lidar_points"] = lidar_points
        record["lidar_sources_info"] = lidar_sources_info
        record["timestamp"] = _stamp_to_seconds(stamp)
        record["token"] = f"{template_record.get('token')}:expanded:{frame_index:05d}"
        records.append(record)
    return records


if __name__ == "__main__":
    main()
