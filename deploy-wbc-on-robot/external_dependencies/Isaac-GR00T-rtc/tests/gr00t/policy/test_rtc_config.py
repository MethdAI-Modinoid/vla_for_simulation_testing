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

"""Tests for RTC configuration, option resolution, and chunk alignment math."""

from gr00t.policy.rtc import (
    RTC_MODEL_OPTION_KEYS,
    RTCConfig,
    compute_prev_chunk_shift,
    merge_rtc_call_options,
    resolve_rtc_options,
    shift_prev_chunk,
)
import pytest
import torch


class TestRTCConfig:
    def test_defaults_valid(self):
        cfg = RTCConfig()
        assert cfg.execution_horizon is None
        assert cfg.overlap_steps is None
        assert cfg.frozen_steps == 0
        assert cfg.ramp_rate == 5.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"execution_horizon": 0},
            {"overlap_steps": 0},
            {"frozen_steps": -1},
            {"ramp_rate": 0.0},
            {"ramp_rate": -1.0},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ValueError):
            RTCConfig(**kwargs)


class TestResolveRTCOptions:
    def test_canonical_full_overlap(self):
        """H=16, e=8 -> overlap defaults to H - e = 8."""
        opts = resolve_rtc_options(unpadded_horizon=16, executed_steps=8)
        assert opts == {
            "action_horizon": 16,
            "rtc_overlap_steps": 8,
            "rtc_frozen_steps": 0,
            "rtc_ramp_rate": 5.0,
        }
        assert set(opts.keys()) == set(RTC_MODEL_OPTION_KEYS)

    def test_values_are_plain_python_types(self):
        """Options must round-trip through msgpack: plain int/float only."""
        import numpy as np

        opts = resolve_rtc_options(
            unpadded_horizon=np.int64(16),
            executed_steps=np.int64(8),
            overlap_steps=np.int64(4),
            frozen_steps=np.int64(2),
            ramp_rate=np.float32(3.0),
        )
        assert type(opts["action_horizon"]) is int
        assert type(opts["rtc_overlap_steps"]) is int
        assert type(opts["rtc_frozen_steps"]) is int
        assert type(opts["rtc_ramp_rate"]) is float

    def test_executed_full_chunk_returns_none(self):
        """e >= H leaves no overlap -> RTC skipped, not an error."""
        assert resolve_rtc_options(unpadded_horizon=16, executed_steps=16) is None
        assert resolve_rtc_options(unpadded_horizon=16, executed_steps=20) is None

    def test_overlap_clamped_to_max(self):
        opts = resolve_rtc_options(unpadded_horizon=16, executed_steps=10, overlap_steps=12)
        assert opts["rtc_overlap_steps"] == 6  # clamped to H - e

    def test_frozen_clamped_to_overlap(self):
        opts = resolve_rtc_options(
            unpadded_horizon=16, executed_steps=8, overlap_steps=4, frozen_steps=10
        )
        assert opts["rtc_frozen_steps"] == 4

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"unpadded_horizon": 0, "executed_steps": 1},
            {"unpadded_horizon": 16, "executed_steps": 0},
            {"unpadded_horizon": 16, "executed_steps": 8, "frozen_steps": -1},
            {"unpadded_horizon": 16, "executed_steps": 8, "ramp_rate": 0.0},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            resolve_rtc_options(**kwargs)


class TestChunkAlignment:
    """The model inpaints from prev[:, H - w : H]; the caller must arrange the cached
    chunk so that slice reads prev[e : e + w] (co-temporal alignment)."""

    def test_canonical_overlap_needs_no_shift(self):
        assert compute_prev_chunk_shift(16, executed_steps=8, overlap_steps=8) == 0

    def test_smaller_overlap_shift(self):
        # H=16, e=8, w=5 -> shift = 3; model slice [11:16] then reads original [8:13].
        assert compute_prev_chunk_shift(16, executed_steps=8, overlap_steps=5) == 3

    def test_overlap_too_large_raises(self):
        with pytest.raises(ValueError):
            compute_prev_chunk_shift(16, executed_steps=8, overlap_steps=9)

    def test_shift_prev_chunk_semantics(self):
        """After shifting by s, the model's slice [H-w : H] equals original [e : e+w]."""
        H, e, w = 16, 8, 5
        prev = torch.arange(H, dtype=torch.float32).reshape(1, H, 1).expand(2, H, 3).clone()
        s = compute_prev_chunk_shift(H, e, w)
        shifted = shift_prev_chunk(prev, s)
        assert torch.equal(shifted[:, H - w : H, :], prev[:, e : e + w, :])

    def test_zero_shift_returns_same_tensor(self):
        prev = torch.randn(1, 16, 3)
        assert shift_prev_chunk(prev, 0) is prev

    def test_negative_shift_raises(self):
        with pytest.raises(ValueError):
            shift_prev_chunk(torch.randn(1, 16, 3), -1)


class TestMergeRTCCallOptions:
    def test_config_defaults_without_call_options(self):
        cfg = RTCConfig(execution_horizon=8, frozen_steps=2, ramp_rate=3.0)
        params = merge_rtc_call_options(cfg, None)
        assert params.enabled is True
        assert params.executed_steps == 8
        assert params.frozen_steps == 2
        assert params.ramp_rate == 3.0

    def test_call_options_override_config(self):
        cfg = RTCConfig(execution_horizon=8, frozen_steps=0)
        params = merge_rtc_call_options(
            cfg,
            {"rtc": {"executed_steps": 5, "frozen_steps": 3, "ramp_rate": 7.0}},
        )
        assert params.executed_steps == 5
        assert params.frozen_steps == 3
        assert params.ramp_rate == 7.0

    def test_disable_per_call(self):
        cfg = RTCConfig(execution_horizon=8)
        params = merge_rtc_call_options(cfg, {"rtc": {"enabled": False}})
        assert params.enabled is False

    def test_no_cadence_anywhere(self):
        params = merge_rtc_call_options(RTCConfig(), {})
        assert params.executed_steps is None

    def test_non_dict_rtc_options_raise(self):
        with pytest.raises(TypeError):
            merge_rtc_call_options(RTCConfig(), {"rtc": 5})

    def test_unrelated_options_ignored(self):
        params = merge_rtc_call_options(RTCConfig(execution_horizon=8), {"other": {"x": 1}})
        assert params.executed_steps == 8
        assert params.enabled is True
