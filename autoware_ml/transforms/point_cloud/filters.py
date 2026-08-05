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

"""Point-cloud filters that emulate vehicle-side preprocessing."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform


_ASSET_ROOT = resources.files("autoware_ml.configs").joinpath("assets")
_DEFAULT_MASK_ROOT = _ASSET_ROOT.joinpath("aip_x2_gen2")
_DEFAULT_CALIBRATION_ROOT = _ASSET_ROOT.joinpath("hesai")

_DEFAULT_MASKS = {
    name: f"{name}/generated_30deg_roi.param.png"
    for name in (
        "front_upper",
        "front_lower",
        "left_upper",
        "left_lower",
        "rear_upper",
        "rear_lower",
        "right_upper",
        "right_lower",
    )
}
_DEFAULT_MODELS = {
    "front_upper": "pandar128e4x",
    "left_upper": "pandar128e4x",
    "rear_upper": "pandar128e4x",
    "right_upper": "pandar128e4x",
    "front_lower": "pandar_qt128",
    "left_lower": "pandar_qt128",
    "rear_lower": "pandar_qt128",
    "right_lower": "pandar_qt128",
}
_DEFAULT_CALIBRATIONS = {name: f"{_DEFAULT_MODELS[name]}.csv" for name in _DEFAULT_MASKS}


@dataclass(frozen=True)
class _Calibration:
    elevation_rad: npt.NDArray[np.float32]
    azimuth_deg: npt.NDArray[np.float32]


@dataclass(frozen=True)
class _SourceSlice:
    name: str
    sensor_token: str | None
    point_slice: slice
    translation: npt.NDArray[np.float32] | None
    rotation: npt.NDArray[np.float32] | None


class NebulaDownsampleMaskFilter(BaseTransform):
    """Approximate Nebula's decode-time downsample mask on loaded Cartesian points.

    The vehicle filter runs in packet/range-view space before ego-motion correction. This transform
    reconstructs a per-LiDAR azimuth/channel index from Cartesian points, applies the same bundled
    dithered mask, and keeps the original point rows in their existing coordinate frame.

    Required keys:
        points: Raw point array of shape ``(N, D)``.

    Optional keys:
        lidar_sources: Mapping from LiDAR name to calibration/extrinsic metadata.
        lidar_sources_info: PointCloudMetainfo-like dictionary containing source slices.
        source_name: Name of the single source when the sample has already been sliced.
        translation, rotation: Single-source extrinsics used only for diagnostics; points are
            assumed to already be in that source frame when no concat metadata is present.

    Generated keys:
        points and all aligned per-point numpy arrays filtered to the kept rows.
        nebula_downsample_stats when ``return_stats`` is true.
    """

    _required_keys = ["points"]
    _QUANTIZATION_LEVELS = 10

    def __init__(
        self,
        *,
        mask_root: str | None = None,
        calibration_root: str | None = None,
        lidar_name_to_mask: Mapping[str, str] | None = None,
        lidar_name_to_model: Mapping[str, str] | None = None,
        lidar_name_to_calibration: Mapping[str, str] | None = None,
        use_calibration_azimuth_offsets: bool = False,
        channel_dim: int | None = 4,
        azimuth_start_deg: float = 0.0,
        azimuth_extent_deg: float = 360.0,
        return_stats: bool = False,
    ) -> None:
        """Initialize the Nebula mask-filter approximation.

        Args:
            mask_root: Root directory for mask PNGs. Defaults to bundled J6 Gen2 masks.
            calibration_root: Root directory for calibration CSVs. Defaults to bundled calibration
                files for Pandar128E4X/OT128 and PandarQT128.
            lidar_name_to_mask: LiDAR-name to mask path mapping. Relative paths resolve under
                ``mask_root``.
            lidar_name_to_model: LiDAR-name to model key mapping.
            lidar_name_to_calibration: LiDAR-name to calibration CSV mapping. Relative paths resolve
                under ``calibration_root``.
            use_calibration_azimuth_offsets: Whether to subtract per-channel azimuth calibration
                offsets before sampling the mask x-coordinate. Defaults to ``False``: the driver
                indexes the mask by raw azimuth, and subtracting the offsets moves points into the
                wrong mask column.
            channel_dim: Index of the per-point feature holding the ring/channel number. Set to
                ``None`` to always estimate the ring from elevation instead.
            azimuth_start_deg: Start of the mask azimuth range.
            azimuth_extent_deg: Width of the mask azimuth range.
            return_stats: Whether to attach per-source keep/drop counts.
        """
        self.mask_root = Path(mask_root) if mask_root is not None else Path(_DEFAULT_MASK_ROOT)
        self.calibration_root = (
            Path(calibration_root)
            if calibration_root is not None
            else Path(_DEFAULT_CALIBRATION_ROOT)
        )
        self.lidar_name_to_mask = {
            _normalize_lidar_name(name): path
            for name, path in (lidar_name_to_mask or _DEFAULT_MASKS).items()
        }
        self.lidar_name_to_model = {
            _normalize_lidar_name(name): _normalize_model_name(model)
            for name, model in (lidar_name_to_model or _DEFAULT_MODELS).items()
        }
        self.lidar_name_to_calibration = {
            _normalize_lidar_name(name): path
            for name, path in (lidar_name_to_calibration or _DEFAULT_CALIBRATIONS).items()
        }
        self.use_calibration_azimuth_offsets = use_calibration_azimuth_offsets
        self.channel_dim = channel_dim if channel_dim is None else int(channel_dim)
        self.azimuth_start_deg = float(azimuth_start_deg)
        self.azimuth_extent_deg = float(azimuth_extent_deg)
        if self.azimuth_extent_deg <= 0.0:
            raise ValueError("azimuth_extent_deg must be positive.")
        self.return_stats = return_stats
        self._masks: dict[str, npt.NDArray[np.bool_]] = {}
        self._calibrations: dict[str, _Calibration] = {}

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply the per-LiDAR mask and keep aligned per-point arrays consistent."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        keep_mask = np.zeros(points.shape[0], dtype=bool)
        stats = []

        for source in self._iter_sources(input_dict, points.shape[0]):
            lidar_name = _normalize_lidar_name(source.name)
            model_name = self.lidar_name_to_model.get(lidar_name)
            if model_name is None:
                raise KeyError(f"No Nebula lidar model configured for source {source.name!r}.")
            source_mask = self._source_keep_mask(points, source, lidar_name, model_name)
            keep_mask[source.point_slice] = source_mask
            if self.return_stats:
                stats.append(
                    {
                        "source_name": source.name,
                        "num_input_points": int(source_mask.size),
                        "num_kept_points": int(source_mask.sum()),
                    }
                )

        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[keep_mask]
        if self.return_stats:
            input_dict["nebula_downsample_stats"] = stats
        return input_dict

    def _iter_sources(self, input_dict: Mapping[str, Any], point_count: int) -> list[_SourceSlice]:
        return _iter_pointcloud_sources(input_dict, point_count)

    def _iter_concat_sources(
        self, lidar_sources: Mapping[str, Any], lidar_sources_info: Mapping[str, Any]
    ) -> list[_SourceSlice]:
        return _iter_concat_sources(lidar_sources, lidar_sources_info)

    def _source_keep_mask(
        self,
        points: npt.NDArray[np.float32],
        source: _SourceSlice,
        lidar_name: str,
        model_name: str,
    ) -> npt.NDArray[np.bool_]:
        source_rows = points[source.point_slice]
        source_points = source_rows[:, :3]
        local_points = source_points
        if source.translation is not None and source.rotation is not None:
            local_points = (source_points - source.translation) @ source.rotation

        calibration = self._load_calibration(lidar_name)
        channels = self._channels(source_rows, local_points, calibration)
        azimuth_deg = np.rad2deg(_nebula_azimuth_rad(local_points))
        if self.use_calibration_azimuth_offsets:
            azimuth_deg = azimuth_deg - calibration.azimuth_deg[channels]
        azimuth_deg = (azimuth_deg - self.azimuth_start_deg) % 360.0

        mask = self._load_mask(lidar_name, model_name)
        x = _round_half_up(azimuth_deg / self.azimuth_extent_deg * mask.shape[1])
        valid = (x >= 0) & (x < mask.shape[1]) & (channels >= 0) & (channels < mask.shape[0])
        keep = np.zeros(source_points.shape[0], dtype=bool)
        keep[valid] = mask[channels[valid], x[valid]]
        return keep

    def _channels(
        self,
        source_rows: npt.NDArray[np.float32],
        local_points: npt.NDArray[np.float32],
        calibration: _Calibration,
    ) -> npt.NDArray[np.int64]:
        """Prefer the stored ring index; fall back to estimating it from elevation.

        Estimating the ring by nearest calibration elevation is unreliable -- on a recording where
        the true index is available it agrees for only 14-68% of points on four of eight LiDARs,
        because neighbouring Pandar channels are separated by less than the elevation spread of a
        single return. T4Dataset preserves the ring index, so use it whenever it is present.
        """
        if self.channel_dim is not None and source_rows.shape[1] > self.channel_dim:
            return source_rows[:, self.channel_dim].astype(np.int64)
        return _nearest_channel(local_points, calibration.elevation_rad)

    def _load_mask(self, lidar_name: str, model_name: str) -> npt.NDArray[np.bool_]:
        if lidar_name in self._masks:
            return self._masks[lidar_name]
        mask_path = _resolve_path(self.mask_root, self.lidar_name_to_mask[lidar_name])
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read Nebula downsample mask: {mask_path}")
        calibration = self._load_calibration(lidar_name)
        if image.shape[0] != calibration.elevation_rad.shape[0]:
            raise ValueError(
                f"Mask {mask_path} has {image.shape[0]} rows, but {model_name} calibration has "
                f"{calibration.elevation_rad.shape[0]} channels."
            )
        self._masks[lidar_name] = _dither_mask(image, model_name)
        return self._masks[lidar_name]

    def _load_calibration(self, lidar_name: str) -> _Calibration:
        lidar_name = _normalize_lidar_name(lidar_name)
        if lidar_name in self._calibrations:
            return self._calibrations[lidar_name]
        calibration_path = _resolve_path(
            self.calibration_root, self.lidar_name_to_calibration[lidar_name]
        )
        elevations = []
        azimuths = []
        with open(calibration_path, newline="") as file:
            rows = csv.reader(file)
            header = None
            for row in rows:
                if "Elevation" in row and "Azimuth" in row:
                    header = row
                    break
            if header is None:
                raise ValueError(
                    f"Calibration file has no Elevation/Azimuth header: {calibration_path}"
                )
            reader = csv.DictReader(file, fieldnames=header)
            for row in reader:
                if not row.get("Elevation") or not row.get("Azimuth"):
                    continue
                elevations.append(float(row["Elevation"]))
                azimuths.append(float(row["Azimuth"]))
        if not elevations:
            raise ValueError(f"Calibration file has no channel rows: {calibration_path}")
        self._calibrations[lidar_name] = _Calibration(
            elevation_rad=np.deg2rad(np.asarray(elevations, dtype=np.float32)),
            azimuth_deg=np.asarray(azimuths, dtype=np.float32),
        )
        return self._calibrations[lidar_name]


class RingOutlierFilter(BaseTransform):
    """Reproduce Autoware's CUDA ring outlier filter on loaded Cartesian points.

    .. warning::

       **Not part of the recommended compatibility pipeline.** The upstream ring outlier filter is
       being removed from the vehicle sensing pipeline: it misbehaves independently of which
       implementation is used, so reproducing it faithfully reproduces a defect. Use
       :class:`NebulaDownsampleMaskFilter` followed by :class:`EgoCropBoxFilter` instead.

       This transform is retained because it is validated against the deployed kernel and is the
       reference if the upstream filter is fixed and reinstated. Opt in explicitly; nothing
       constructs it by default.

    AIP X2 Gen2 runs ``autoware_cuda_pointcloud_preprocessor``, whose ring outlier filter is a
    different algorithm from the CPU ``ring_outlier_filter_node``. The CUDA kernel
    (``outlier_kernels.cu::ringOutlierFilterKernel``) makes a *per-point* decision using a
    ``+/-window_size`` sliding window over the organized ring grid: it grows the largest walk that
    passes through the point but stays inside the window, then keeps the point when that walk's
    endpoints are at least ``object_length_threshold`` apart. The CPU node instead segments each
    ring into unbounded walks and keeps or drops each walk as a unit.

    The two disagree most on densely populated rings, where an unbounded walk runs far longer than
    the window. Measured against ``pointcloud_before_sync`` from a CUDA-preprocessor recording, the
    windowed algorithm reproduces the per-ring kept-point distribution more closely on every LiDAR
    (weighted per-ring L1 error 1.12% vs 1.37%; 0.07-0.26% vs 0.49-1.09% on the six LiDARs without
    dense rings).

    Two approximations remain, both stemming from T4Dataset carrying only ego-motion-corrected
    Cartesian points:

    - The kernel gates on the *original* per-point ``azimuth``/``distance`` fields, which the
      distortion corrector never rewrites. Those fields are absent from T4Dataset, so they are
      recomputed from the Cartesian coordinates instead.
    - The kernel's cluster test uses post-undistortion coordinates. That test is a distance between
      two points, so it is invariant to the rigid sensor-to-ego transform and the supplied
      coordinates can be used directly.
    """

    _required_keys = ["points"]

    def __init__(
        self,
        *,
        distance_ratio: float = 1.1,
        object_length_threshold: float = 0.05,
        max_rings_num: int = 128,
        window_size: int = 5,
        channel_dim: int = 4,
        return_stats: bool = False,
    ) -> None:
        """Initialize the RingOutlierFilter transform.

        Args:
            distance_ratio: Maximum ratio between neighbouring point ranges for them to stay in the
                same walk. Matches Autoware's ``distance_ratio``.
            object_length_threshold: Minimum walk endpoint separation in metres for the walk to be
                considered a real object. Matches Autoware's ``object_length_threshold``.
            max_rings_num: Number of LiDAR rings. Points with channels outside this range are
                dropped, matching ``organizeKernel``.
            window_size: Half-width of the CUDA kernel's sliding window, in points. The deployed
                kernel hard-codes 5.
            channel_dim: Index of the per-point feature holding the ring/channel number.
            return_stats: Whether to record per-source kept/input point counts on the sample.
        """
        self.distance_ratio = float(distance_ratio)
        self.object_length_threshold = float(object_length_threshold)
        self.max_rings_num = int(max_rings_num)
        self.window_size = int(window_size)
        self.channel_dim = int(channel_dim)
        self.return_stats = return_stats

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        points = np.asarray(input_dict["points"], dtype=np.float32)
        keep_mask = np.zeros(points.shape[0], dtype=bool)
        stats = []

        for source in _iter_pointcloud_sources(input_dict, points.shape[0]):
            source_keep = self._source_keep_mask(points, source)
            keep_mask[source.point_slice] = source_keep
            if self.return_stats:
                stats.append(
                    {
                        "source_name": source.name,
                        "num_input_points": int(source_keep.size),
                        "num_kept_points": int(source_keep.sum()),
                    }
                )

        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[keep_mask]
        if self.return_stats:
            input_dict["ring_outlier_stats"] = stats
        return input_dict

    def _source_keep_mask(
        self, points: npt.NDArray[np.float32], source: _SourceSlice
    ) -> npt.NDArray[np.bool_]:
        source_points = points[source.point_slice]
        local_points = source_points[:, :3]
        if source.translation is not None and source.rotation is not None:
            local_points = (local_points - source.translation) @ source.rotation

        channels = source_points[:, self.channel_dim].astype(np.int64)
        return self._ring_outlier_keep_for_local_points(local_points, channels)

    def _ring_outlier_keep_for_local_points(
        self, local_points: npt.NDArray[np.float32], channels: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.bool_]:
        """Compute the keep mask for one LiDAR's points expressed in its own frame."""
        keep = np.zeros(local_points.shape[0], dtype=bool)
        in_range = (channels >= 0) & (channels < self.max_rings_num)
        if not in_range.any():
            return keep

        # Mirror organizeKernel: bucket points by ring, preserving acquisition order within a ring.
        ordered = np.flatnonzero(in_range)[np.argsort(channels[in_range], kind="stable")]
        rings = channels[ordered]
        counts = np.bincount(rings, minlength=self.max_rings_num)
        num_slots = int(counts.max())
        if num_slots < 2:
            return keep
        slots = np.arange(rings.shape[0]) - np.repeat(
            np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
        )

        points = local_points[ordered].astype(np.float64, copy=False)
        grid_points = np.zeros((self.max_rings_num, num_slots, 3), dtype=np.float64)
        grid_azimuth = np.zeros((self.max_rings_num, num_slots), dtype=np.float64)
        # Unfilled slots keep distance 0, which fails the ratio test and so terminates a walk --
        # the same effect the kernel gets from gatherKernel zeroing padding slots.
        grid_distance = np.zeros((self.max_rings_num, num_slots), dtype=np.float64)
        grid_valid = np.zeros((self.max_rings_num, num_slots), dtype=bool)
        grid_points[rings, slots] = points
        grid_azimuth[rings, slots] = np.mod(
            _nebula_azimuth_rad(points.astype(np.float32)), 2.0 * np.pi
        )
        grid_distance[rings, slots] = np.linalg.norm(points, axis=1)
        grid_valid[rings, slots] = True

        grid_keep = self._window_keep_mask(grid_azimuth, grid_distance, grid_points, grid_valid)
        keep[ordered] = grid_keep[rings, slots]
        return keep

    def _same_walk(
        self, azimuth: npt.NDArray[np.float64], distance: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Whether each pair of neighbouring slots belongs to the same walk."""
        azimuth_diff = azimuth[:, 1:] - azimuth[:, :-1]
        azimuth_diff = np.where(azimuth_diff < 0.0, azimuth_diff + 2.0 * np.pi, azimuth_diff)
        near, far = distance[:, :-1], distance[:, 1:]
        return (np.maximum(near, far) < np.minimum(near, far) * self.distance_ratio) & (
            azimuth_diff < np.deg2rad(1.0)
        )

    def _window_keep_mask(
        self,
        azimuth: npt.NDArray[np.float64],
        distance: npt.NDArray[np.float64],
        points: npt.NDArray[np.float64],
        valid: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.bool_]:
        """Vectorized transcription of ``ringOutlierFilterKernel`` over the organized ring grid.

        The kernel's serial scan over ``k`` is unrolled: each iteration advances every point's
        candidate walk by one slot, so ``2 * window_size`` vectorized steps cover the whole window.
        """
        num_rings, num_slots = azimuth.shape
        same_walk = np.concatenate(
            [self._same_walk(azimuth, distance), np.zeros((num_rings, 1), dtype=bool)], axis=1
        )

        slot = np.arange(num_slots)
        window_start = np.maximum(slot - self.window_size, 0)
        window_end = np.minimum(slot + self.window_size, num_slots)

        walk_start = np.broadcast_to(window_start, (num_rings, num_slots)).copy()
        walk_end = walk_start + 1
        terminated = np.zeros((num_rings, num_slots), dtype=bool)

        for step in range(2 * self.window_size):
            k = window_start + step
            active = (k <= window_end - 2) & ~terminated
            if not active.any():
                break
            k_clipped = np.clip(k, 0, num_slots - 2)
            linked = np.take_along_axis(
                same_walk, np.broadcast_to(k_clipped, (num_rings, num_slots)), axis=1
            )
            # Linked: the walk extends. Otherwise it either ends here (the break past our own
            # slot) or restarts just after the gap (the gap is still behind us).
            walk_end = np.where(active & linked, walk_end + 1, walk_end)
            terminated |= active & ~linked & (k >= slot)
            restarted = active & ~linked & (k < slot)
            walk_start = np.where(restarted, k_clipped + 1, walk_start)
            walk_end = np.where(restarted, k_clipped + 2, walk_end)

        last_slot = np.clip(walk_end - 1, 0, num_slots - 1)
        squared_length = sum(
            (
                np.take_along_axis(points[..., axis], walk_start, axis=1)
                - np.take_along_axis(points[..., axis], last_slot, axis=1)
            )
            ** 2
            for axis in range(3)
        )
        is_cluster = squared_length >= self.object_length_threshold**2
        return is_cluster & valid


class EgoCropBoxFilter(BaseTransform):
    """Remove points falling inside the ego vehicle's crop boxes.

    Autoware crops the vehicle body and the steered front wheels out of every LiDAR scan before
    concatenation (two negative crop boxes, see ``nebula_node_container.launch.py``). T4Dataset
    carries un-cropped concatenated clouds -- the ego vehicle is plainly visible in them -- so this
    step has to be reapplied to match the inference-time point distribution. It removes up to 18%
    of a single LiDAR's points (``rear_lower`` on AIP X2 Gen2).

    Points are expected in the ego/``base_link`` frame, which is how T4Dataset stores them, and the
    boxes are applied to those coordinates directly.

    .. note::

       On the vehicle the crop mask is computed *before* ego-motion correction, so deciding it on
       corrected coordinates is an approximation. Reconstructing a recorded concatenated cloud from
       an unfiltered T4Dataset, taking the mask and crop decisions in inverse-corrected space is
       markedly closer in space: voxel IoU at 0.12 m rises from 0.702 to 0.878, even though the
       plain point count is slightly worse (+0.32% vs +0.08%). Count alone does not capture this.

       This transform does not invert the correction -- doing so needs per-point timestamps and ego
       poses, which are not available on the transform path. See
       ``autoware_ml/tools/dataset/t4dataset/compare_t4_to_concat.py`` for the measurement.

    Ordering note: if :class:`RingOutlierFilter` is used (it is not in the recommended pipeline),
    this transform must run **after** it. On the vehicle the crop is a mask that is only AND-ed into
    the output at the very end, so cropped points are still present as neighbours while the ring
    filter runs; cropping first discards those neighbours and measurably changes its decisions.
    """

    _required_keys = ["points"]

    def __init__(self, *, crop_boxes: Sequence[Sequence[float]] | None = None) -> None:
        """Initialize the EgoCropBoxFilter transform.

        Args:
            crop_boxes: Boxes to remove, each ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in the
                ego frame. Defaults to the AIP X2 Gen2 self and wheels boxes.
        """
        boxes = AIP_X2_GEN2_EGO_CROP_BOXES if crop_boxes is None else crop_boxes
        self.crop_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
        if self.crop_boxes.size == 0:
            raise ValueError("crop_boxes must contain at least one box")

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Drop points inside any configured box, keeping aligned per-point arrays consistent."""
        points = np.asarray(input_dict["points"], dtype=np.float32)
        inside = np.zeros(points.shape[0], dtype=bool)
        for x_min, y_min, z_min, x_max, y_max, z_max in self.crop_boxes:
            inside |= (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
            )

        keep_mask = ~inside
        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[keep_mask]
        return input_dict


def ego_crop_boxes_from_vehicle_info(
    *,
    wheel_base: float,
    wheel_tread: float,
    wheel_radius: float,
    wheel_width: float,
    front_overhang: float,
    rear_overhang: float,
    left_overhang: float,
    right_overhang: float,
    vehicle_height: float,
    max_steer_angle: float,
) -> list[list[float]]:
    """Derive the self and wheels crop boxes from vehicle dimensions.

    Transcribes ``get_vehicle_info()`` in
    ``aip_x2_gen2_launch/launch/nebula_node_container.launch.py``, so the boxes stay consistent with
    whatever the vehicle's ``vehicle_info.param.yaml`` declares.

    Args:
        wheel_base: Distance between front and rear wheel centres.
        wheel_tread: Distance between left and right wheel centres.
        wheel_radius: Wheel radius.
        wheel_width: Wheel width.
        front_overhang: Front wheel centre to vehicle front.
        rear_overhang: Rear wheel centre to vehicle rear.
        left_overhang: Left wheel centre to vehicle left.
        right_overhang: Right wheel centre to vehicle right.
        vehicle_height: Overall vehicle height.
        max_steer_angle: Maximum tire cut angle in radians.

    Returns:
        Two boxes as ``[x_min, y_min, z_min, x_max, y_max, z_max]``: the vehicle body, then the
        swept volume of the steered front wheels.
    """
    half_width = wheel_width / 2.0
    center_to_corner = float(np.hypot(half_width, wheel_radius))
    corner_angle = float(np.arctan2(half_width, wheel_radius))
    if corner_angle < max_steer_angle:
        max_longitudinal = center_to_corner
    else:
        max_longitudinal = center_to_corner * float(np.cos(max_steer_angle - corner_angle))
    max_lateral = center_to_corner * float(np.sin(max_steer_angle + corner_angle))

    self_box = [
        -rear_overhang,
        -(wheel_tread / 2.0 + right_overhang),
        0.0,
        front_overhang + wheel_base,
        wheel_tread / 2.0 + left_overhang,
        vehicle_height,
    ]
    # The wheel box is scaled to 110% of wheel diameter upstream to absorb suspension travel.
    wheels_box = [
        wheel_base - max_longitudinal,
        -(wheel_tread / 2.0 + max_lateral),
        0.0,
        wheel_base + max_longitudinal,
        wheel_tread / 2.0 + max_lateral,
        wheel_radius * 2.2,
    ]
    return [self_box, wheels_box]


# Derived from j6_gen2_description/config/vehicle_info.param.yaml.
AIP_X2_GEN2_EGO_CROP_BOXES = ego_crop_boxes_from_vehicle_info(
    wheel_base=4.76012,
    wheel_tread=1.754,
    wheel_radius=0.3725,
    wheel_width=0.215,
    front_overhang=0.95099,
    rear_overhang=1.52579,
    left_overhang=0.32358,
    right_overhang=0.34983,
    vehicle_height=3.080,
    max_steer_angle=0.838,
)


def _dither_mask(image: npt.NDArray[np.uint8], model_name: str) -> npt.NDArray[np.bool_]:
    height, width = image.shape
    y, x = np.indices((height, width), dtype=np.int64)
    if model_name == "pandar128e4x":
        positions = (
            (x // 2) * 2 + (y // 4) * 4 + (y % 2)
        ) % NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS
    else:
        positions = (x + y) % NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS

    numerator = image.astype(np.uint32) * NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS // 255
    output = np.zeros(image.shape, dtype=bool)
    for keep_count in range(1, NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS + 1):
        kept_positions = _round_half_up(
            NebulaDownsampleMaskFilter._QUANTIZATION_LEVELS
            / float(keep_count)
            * np.arange(keep_count)
        )
        output |= (numerator == keep_count) & np.isin(positions, kept_positions)
    return output


def _iter_pointcloud_sources(input_dict: Mapping[str, Any], point_count: int) -> list[_SourceSlice]:
    lidar_sources = input_dict.get("lidar_sources")
    lidar_sources_info = input_dict.get("lidar_sources_info")
    if isinstance(lidar_sources, Mapping) and isinstance(lidar_sources_info, Mapping):
        return _iter_concat_sources(lidar_sources, lidar_sources_info)

    source_name = input_dict.get("source_name") or input_dict.get("lidar_source_name")
    if source_name is None:
        sample_name = str(input_dict.get("name", ""))
        source_name = _infer_lidar_name_from_text(sample_name)
    if source_name is None:
        raise KeyError(
            "Pointcloud source-aware filters require concat 'lidar_sources' metadata or a "
            "single-source 'source_name'."
        )
    return [
        _SourceSlice(
            name=str(source_name),
            sensor_token=input_dict.get("sensor_token"),
            point_slice=slice(0, point_count),
            translation=None,
            rotation=None,
        )
    ]


def _iter_concat_sources(
    lidar_sources: Mapping[str, Any], lidar_sources_info: Mapping[str, Any]
) -> list[_SourceSlice]:
    source_ranges = {
        str(source.get("sensor_token")): source for source in lidar_sources_info.get("sources", [])
    }
    output = []
    for source_name, source_meta in lidar_sources.items():
        sensor_token = str(source_meta.get("sensor_token"))
        source_range = source_ranges.get(sensor_token)
        if source_range is None:
            continue
        idx_begin = int(source_range["idx_begin"])
        length = int(source_range["length"])
        output.append(
            _SourceSlice(
                name=str(source_name),
                sensor_token=sensor_token,
                point_slice=slice(idx_begin, idx_begin + length),
                translation=np.asarray(source_meta.get("translation"), dtype=np.float32),
                rotation=np.asarray(source_meta.get("rotation"), dtype=np.float32),
            )
        )
    if not output:
        raise ValueError("lidar_sources_info did not match any configured lidar_sources.")
    return output


def _nearest_channel(
    points: npt.NDArray[np.float32], elevation_rad: npt.NDArray[np.float32]
) -> npt.NDArray[np.int64]:
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    point_elevation = np.arctan2(points[:, 2], xy_norm)
    return np.abs(point_elevation[:, None] - elevation_rad[None, :]).argmin(axis=1).astype(np.int64)


def _nebula_azimuth_rad(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.arctan2(points[:, 0], points[:, 1])


def _normalize_lidar_name(name: str) -> str:
    normalized = str(name).lower()
    if normalized.startswith("lidar_"):
        normalized = normalized[len("lidar_") :]
    return normalized


def _normalize_model_name(name: str) -> str:
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"ot128", "pandar128", "pandar128e4x"}:
        return "pandar128e4x"
    if normalized in {"qt128", "pandarqt128", "pandar_qt128"}:
        return "pandar_qt128"
    return normalized


def _infer_lidar_name_from_text(text: str) -> str | None:
    normalized = text.lower()
    for lidar_name in _DEFAULT_MASKS:
        if lidar_name in normalized:
            return lidar_name
    return None


def _resolve_path(root: Path, path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return root / resolved


def _round_half_up(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int64)
