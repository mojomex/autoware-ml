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

"""Tests for the task config that reapplies the AIP X2 Gen2 sensing filters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.point_cloud.crop_box import CropBoxFilter
from autoware_ml.transforms.point_cloud.ego_motion import InvertEgoMotionCorrection
from autoware_ml.transforms.point_cloud.formatting import PreparePointCloudInput
from autoware_ml.transforms.point_cloud.loading import LoadPointsFromFile
from autoware_ml.transforms.point_cloud.nebula_mask import NebulaDownsampleMaskFilter
from autoware_ml.transforms.point_cloud.sweeps import LoadPointsFromMultiSweeps

CONFIG_NAME = "tasks/segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2_sensing_filters"
BASE_CONFIG_NAME = "tasks/segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2"
SWEEP_CONFIG_NAME = (
    "tasks/detection3d/bevfusion/"
    "lidar_voxel0170_second_secfpn_120m_t4dataset_j6gen2_sensing_filters"
)

# One LiDAR is enough to exercise the wiring; front_upper carries a 128 x 3600 mask.
LIDAR_NAME = "front_upper"
SENSOR_TOKEN = "sensor-front-upper"
NUM_POINT_FEATURES = 7
TIMESTAMP_DIM = 6
CHANNEL_DIM = 4

# Vehicle body box from configs/assets/aip_x2_gen2/crop_boxes.param.yaml.
BODY_BOX_X_MAX = 5.711110
# A point at this range sits just beyond the body box, and 0.5 m nearer once the ego motion is
# undone for a return acquired 50 ms into the sweep at 10 m/s.
PROBE_RANGE_M = 5.9
SWEEP_VELOCITY_MPS = 10.0
LATE_RETURN_NS = 50_000_000.0


def compose_config(config_name: str) -> DictConfig:
    """Compose one of the shipped task configs."""
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        return compose(config_name=config_name)


def write_scene(
    tmp_path: Path,
    points: npt.NDArray[np.float32],
    labels: npt.NDArray[np.uint8],
    velocity_mps: float,
) -> dict[str, Any]:
    """Lay out a minimal T4 scene and return the sample metadata the dataset would emit."""
    scene = tmp_path / "scene"
    lidar_dir = scene / "data" / "LIDAR_CONCAT"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    (scene / "annotation").mkdir(parents=True, exist_ok=True)
    (scene / "annotation" / "ego_pose.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": int((10.0 + index * 0.01) * 1e6),
                    "translation": [velocity_mps * index * 0.01, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                }
                for index in range(20)
            ]
        )
    )

    lidar_path = lidar_dir / "0.pcd.bin"
    points.astype(np.float32).tofile(lidar_path)
    mask_path = lidar_dir / "0.pcd.bin.label"
    labels.astype(np.uint8).tofile(mask_path)

    return {
        "lidar_path": str(lidar_path),
        "name": "sample-0",
        "num_pts_feats": NUM_POINT_FEATURES,
        "pts_semantic_mask_path": str(mask_path),
        "pts_semantic_mask_categories": {"car": 0, "vegetation": 1},
        "lidar_sources": {
            LIDAR_NAME: {
                "sensor_token": SENSOR_TOKEN,
                "translation": [0.0, 0.0, 0.0],
                "rotation": np.eye(3, dtype=np.float32).tolist(),
            }
        },
        "lidar_sources_info": {
            "stamp": {"sec": 10, "nanosec": 0},
            "sources": [
                {
                    "sensor_token": SENSOR_TOKEN,
                    "idx_begin": 0,
                    "length": int(points.shape[0]),
                    "stamp": {"sec": 10, "nanosec": 0},
                }
            ],
        },
    }


def random_cloud(num_points: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Build a synthetic sweep spread over the mask's azimuth and channel range."""
    rng = np.random.default_rng(0)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, num_points)
    radius = rng.uniform(1.0, 60.0, num_points)
    points = np.zeros((num_points, NUM_POINT_FEATURES), dtype=np.float32)
    # Nebula measures azimuth from +y toward +x.
    points[:, 0] = radius * np.sin(azimuth)
    points[:, 1] = radius * np.cos(azimuth)
    points[:, 2] = rng.uniform(-1.0, 3.0, num_points)
    points[:, 3] = rng.uniform(0.0, 255.0, num_points)
    points[:, CHANNEL_DIM] = rng.integers(0, 128, num_points)
    points[:, TIMESTAMP_DIM] = rng.uniform(0.0, 1e8, num_points)
    labels = rng.integers(0, 2, num_points).astype(np.uint8)
    return points, labels


def probe_pair(keep_grid: npt.NDArray[np.bool_]) -> npt.NDArray[np.float32]:
    """Build two returns the mask treats identically but the crop box does not.

    Both sit at the same range, azimuth and ring, just beyond the body box in the stored
    coordinates. Only the second was acquired late enough in the sweep for the ego motion to have
    carried it inside the box by the time the vehicle made the decision.
    """
    # Straight ahead is azimuth 90 deg, a quarter of the way around the grid.
    azimuth_bin = keep_grid.shape[1] // 4
    kept_channels = np.flatnonzero(keep_grid[:, azimuth_bin])
    assert kept_channels.size, "expected the bundled front_upper mask to keep some ring ahead"

    points = np.zeros((2, NUM_POINT_FEATURES), dtype=np.float32)
    points[:, 0] = PROBE_RANGE_M
    points[:, 2] = 1.0
    points[:, 3] = 100.0
    points[:, CHANNEL_DIM] = kept_channels[0]
    points[:, TIMESTAMP_DIM] = [0.0, LATE_RETURN_NS]
    return points


@pytest.fixture(autouse=True)
def data_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Satisfy the data-root interpolation the dataset configs carry."""
    monkeypatch.setenv("AUTOWARE_ML_DATA_PATH", str(tmp_path))


class TestSensingFilterConfig:
    def test_every_split_runs_the_filters_between_loading_and_formatting(self) -> None:
        cfg = compose_config(CONFIG_NAME)

        for split in ("train_transforms", "val_transforms", "test_transforms"):
            pipeline = instantiate(cfg.datamodule[split]).pipeline
            types = [type(transform) for transform in pipeline]
            filters = pipeline[types.index(TransformsCompose)]

            assert [type(step) for step in filters.pipeline] == [
                InvertEgoMotionCorrection,
                NebulaDownsampleMaskFilter,
                CropBoxFilter,
            ], split
            # The filters carry every aligned per-point array through the same mask, so the labels
            # have to be loaded before they run and split into fields after.
            assert types.index(LoadPointsFromFile) < types.index(TransformsCompose), split
            assert types.index(TransformsCompose) < types.index(PreparePointCloudInput), split

    def test_loading_keeps_the_features_the_filters_index_by(self) -> None:
        cfg = compose_config(CONFIG_NAME)

        for split in ("train_transforms", "val_transforms", "predict_transforms"):
            loader = instantiate(cfg.datamodule[split]).pipeline[0]
            assert isinstance(loader, LoadPointsFromFile), split
            assert CHANNEL_DIM in loader.use_dim, split
            assert TIMESTAMP_DIM in loader.use_dim, split

    def test_pipeline_drops_points_the_unfiltered_config_keeps(self, tmp_path: Path) -> None:
        points, labels = random_cloud(4000)
        sample = write_scene(tmp_path, points, labels, velocity_mps=SWEEP_VELOCITY_MPS)

        filtered = instantiate(compose_config(CONFIG_NAME).datamodule.val_transforms)(
            copy.deepcopy(sample)
        )
        unfiltered = instantiate(compose_config(BASE_CONFIG_NAME).datamodule.val_transforms)(
            copy.deepcopy(sample)
        )

        assert 0 < filtered["coord"].shape[0] < unfiltered["coord"].shape[0]
        # Points and their labels stay in lockstep through the filters.
        assert filtered["segment"].shape[0] == filtered["coord"].shape[0]
        assert filtered["strength"].shape[0] == filtered["coord"].shape[0]
        assert filtered["origin_segment"].shape[0] == filtered["origin_coord"].shape[0]

    def test_multi_sweep_config_filters_the_frame_and_every_sweep(self) -> None:
        # A sweep and the current frame are separate scans the vehicle filtered separately, so the
        # block has to appear twice: once in the pipeline for the frame, once inside the sweep
        # loader. Neither position can cover for the other.
        cfg = compose_config(SWEEP_CONFIG_NAME)

        for split in ("train_transforms", "val_transforms", "predict_transforms"):
            pipeline = instantiate(cfg.datamodule[split]).pipeline
            types = [type(transform) for transform in pipeline]
            loader = pipeline[types.index(LoadPointsFromMultiSweeps)]

            # The current frame is loaded explicitly and filtered before the sweep loader runs.
            assert types.index(LoadPointsFromFile) < types.index(TransformsCompose), split
            assert types.index(TransformsCompose) < types.index(LoadPointsFromMultiSweeps), split
            # Every sweep goes through the same block.
            assert [type(step) for step in loader.sweep_transforms.pipeline] == [
                InvertEgoMotionCorrection,
                NebulaDownsampleMaskFilter,
                CropBoxFilter,
            ], split
            # The lag column is the one the filters read as a timestamp, so it can only be
            # overwritten after they have run.
            assert loader.time_dim == TIMESTAMP_DIM, split
            assert tuple(loader.use_dim) == (0, 1, 2, 3, TIMESTAMP_DIM), split

    def test_crop_box_decides_in_pre_correction_space(self, tmp_path: Path) -> None:
        filters = instantiate(compose_config(CONFIG_NAME).datamodule.val_transforms).pipeline[2]
        points = probe_pair(filters.pipeline[1].lidar_masks[LIDAR_NAME].keep)
        assert points[0, 0] > BODY_BOX_X_MAX, "both probes start outside the body box"

        moving = filters(
            write_scene(tmp_path / "moving", points, np.zeros(2, np.uint8), SWEEP_VELOCITY_MPS)
            | {"points": points.copy()}
        )
        stationary = filters(
            write_scene(tmp_path / "stationary", points, np.zeros(2, np.uint8), 0.0)
            | {"points": points.copy()}
        )

        # Undoing the ego motion carries the late return back inside the body box, where the
        # vehicle cropped it. Its twin, acquired at the reference time, does not move.
        assert moving["points"][:, TIMESTAMP_DIM].tolist() == [0.0]
        # With the vehicle at rest the correction is the identity, so neither is cropped.
        assert stationary["points"][:, TIMESTAMP_DIM].tolist() == [0.0, LATE_RETURN_NS]
