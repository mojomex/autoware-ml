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

import numpy as np

from autoware_ml.tools.dataset.t4dataset.nebula_order_error import (
    PoseTable,
    _corrected_to_raw_ego,
    _ego_to_lidar_points,
)


def test_corrected_to_raw_ego_is_identity_at_reference_time():
    corrected = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    pose_table = PoseTable(
        times=np.array([10.0, 11.0], dtype=np.float64),
        translations=np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64),
        quaternions=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )

    raw = _corrected_to_raw_ego(
        corrected,
        reference_time=10.0,
        point_times=np.array([10.0], dtype=np.float64),
        pose_table=pose_table,
    )

    assert np.allclose(raw, corrected)


def test_corrected_to_raw_ego_inverts_interpolated_translation():
    corrected = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    pose_table = PoseTable(
        times=np.array([0.0, 1.0], dtype=np.float64),
        translations=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        quaternions=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )

    raw = _corrected_to_raw_ego(
        corrected,
        reference_time=0.0,
        point_times=np.array([0.25], dtype=np.float64),
        pose_table=pose_table,
    )

    assert np.allclose(raw, np.array([[9.75, 0.0, 0.0]], dtype=np.float64))


def test_ego_to_lidar_points_uses_existing_row_vector_convention():
    ego_points = np.array([[3.0, 4.0, 5.0]], dtype=np.float64)
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)

    local_points = _ego_to_lidar_points(ego_points, translation, rotation)

    assert np.allclose(local_points, np.array([[2.0, 2.0, 2.0]], dtype=np.float64))
