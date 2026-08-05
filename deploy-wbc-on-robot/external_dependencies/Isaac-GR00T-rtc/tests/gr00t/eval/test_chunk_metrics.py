# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for chunk smoothness metrics (jerk / boundary-discontinuity diagnostics)."""

from gr00t.eval.chunk_metrics import (
    compute_chunk_metrics,
    metric_boundary_jump,
    metric_intra_accel,
    metric_momentum_shift,
)
import numpy as np
import pytest


def _linear_chunks(n_chunks=4, chunk_len=8, dim=3, execute_steps=4, slope=0.5):
    """Chunks sampled from one globally linear trajectory: each chunk i covers global
    steps [i * execute_steps, i * execute_steps + chunk_len). Perfectly smooth by
    construction: zero acceleration, zero boundary jump, cosine similarity 1."""
    chunks = np.zeros((n_chunks, chunk_len, dim))
    for i in range(n_chunks):
        start = i * execute_steps
        t = np.arange(start, start + chunk_len)[:, None]
        chunks[i] = slope * t
    return chunks


class TestMetricIntraAccel:
    def test_linear_trajectory_zero_acceleration(self):
        chunks = _linear_chunks()
        assert metric_intra_accel(chunks) == pytest.approx(0.0)

    def test_known_acceleration(self):
        # One chunk, one dim: positions 0, 0, 1 -> acceleration = 1 at the middle step.
        chunks = np.array([[[0.0], [0.0], [1.0]]])
        assert metric_intra_accel(chunks) == pytest.approx(1.0)

    def test_too_short_chunk_raises(self):
        with pytest.raises(ValueError):
            metric_intra_accel(np.zeros((2, 2, 3)))


class TestMetricBoundaryJump:
    def test_continuous_boundaries_zero_jump(self):
        # Chunk i+1's step 0 is co-temporal with chunk i's step execute_steps: the doc
        # metric compares against the last EXECUTED step, so a globally linear
        # trajectory leaves exactly one control step of motion: |slope| * sqrt(dim)...
        # use slope 0 for an exact zero-jump oracle.
        chunks = _linear_chunks(slope=0.0)
        assert metric_boundary_jump(chunks, execute_steps=4) == pytest.approx(0.0)

    def test_known_jump(self):
        # Two chunks, constant positions 0 and 3: jump = 3 per dim -> L2 over 1 dim = 3.
        chunks = np.stack([np.zeros((4, 1)), np.full((4, 1), 3.0)])
        assert metric_boundary_jump(chunks, execute_steps=4) == pytest.approx(3.0)

    def test_single_chunk_raises(self):
        with pytest.raises(ValueError):
            metric_boundary_jump(np.zeros((1, 4, 3)))

    def test_bad_execute_steps_raises(self):
        with pytest.raises(ValueError):
            metric_boundary_jump(np.zeros((2, 4, 3)), execute_steps=5)


class TestMetricMomentumShift:
    def test_consistent_direction_cosine_one(self):
        chunks = _linear_chunks()
        assert metric_momentum_shift(chunks, execute_steps=4) == pytest.approx(1.0, abs=1e-6)

    def test_reversed_direction_cosine_minus_one(self):
        # Previous chunk moves +1/step; next chunk moves -1/step.
        up = np.arange(4, dtype=float)[:, None]
        down = -np.arange(4, dtype=float)[:, None]
        chunks = np.stack([up, down])
        assert metric_momentum_shift(chunks, execute_steps=4) == pytest.approx(-1.0, abs=1e-6)

    def test_execute_steps_one_raises(self):
        with pytest.raises(ValueError):
            metric_momentum_shift(np.zeros((2, 4, 3)), execute_steps=1)


class TestComputeChunkMetrics:
    def test_all_metrics_present(self):
        metrics = compute_chunk_metrics(_linear_chunks(), execute_steps=4)
        assert set(metrics.keys()) == {"intra_accel", "boundary_jump", "momentum_shift"}
        assert all(np.isfinite(v) for v in metrics.values())

    def test_unmet_preconditions_yield_nan_not_raise(self):
        # Single chunk: boundary metrics undefined -> nan, intra_accel still computed.
        metrics = compute_chunk_metrics(np.zeros((1, 4, 3)), execute_steps=4)
        assert np.isfinite(metrics["intra_accel"])
        assert np.isnan(metrics["boundary_jump"])
        assert np.isnan(metrics["momentum_shift"])
