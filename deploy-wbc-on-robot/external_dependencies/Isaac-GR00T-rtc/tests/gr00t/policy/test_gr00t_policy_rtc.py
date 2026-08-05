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

"""
Test Gr00tPolicy Real-Time Chunking integration: previous-chunk caching, model-input
injection, option resolution, reset semantics, and cache invalidation.

Uses mocked model and processor (same conventions as test_gr00t_policy.py) so the RTC
plumbing is exercised without checkpoints or a GPU.
"""

from unittest.mock import MagicMock, patch

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.policy.rtc import RTCConfig
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


EMBODIMENT = "libero_sim"
ACTION_HORIZON = 16

VIDEO_KEYS = ["observation.images.rgb.head_256_256"]
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANGUAGE_KEY = "annotation.human.action.task_description"


def _build_modality_configs(action_configs=None):
    return {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(
                delta_indices=list(range(ACTION_HORIZON)),
                modality_keys=ACTION_KEYS,
                action_configs=action_configs,
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }
    }


class _ModelCallRecorder:
    """Stands in for model.get_action; records the inputs/options of every call and
    returns a fresh, deterministic-per-call action_pred so cache identity is checkable."""

    def __init__(self):
        self.calls: list[dict] = []
        self._counter = 0

    def __call__(self, inputs, options=None):
        # Snapshot: the policy mutates the inputs dict in place before calling.
        self.calls.append({"inputs": dict(inputs), "options": options})
        self._counter += 1
        torch.manual_seed(self._counter)
        return BatchFeature(data={"action_pred": torch.randn(1, ACTION_HORIZON, 7)})


def _make_policy(rtc_config, use_relative_action=False, action_configs=None):
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.dtype = torch.bfloat16
    recorder = _ModelCallRecorder()
    mock_model.get_action = recorder

    mock_processor = MagicMock()
    mock_processor.get_modality_configs.return_value = _build_modality_configs(action_configs)
    mock_processor.use_relative_action = use_relative_action
    mock_processor.eval = MagicMock()

    # Fresh inputs dict per call: the policy injects the cached chunk into it.
    def fake_collate(processed_inputs):
        return BatchFeature(
            data={
                "inputs": {
                    "state": torch.randn(1, 1, 128),
                    "embodiment_id": torch.zeros(1, dtype=torch.long),
                }
            }
        )

    mock_processor.collator = MagicMock(side_effect=fake_collate)

    def fake_decode_action(action, embodiment_tag, state=None):
        batch = action.shape[0]
        return {k: np.zeros((batch, ACTION_HORIZON, 1), dtype=np.float32) for k in ACTION_KEYS}

    mock_processor.decode_action = MagicMock(side_effect=fake_decode_action)

    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
    ):
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoProcessor.from_pretrained.return_value = mock_processor

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag=EMBODIMENT,
            model_path="/fake/path",
            device="cpu",
            rtc_config=rtc_config,
        )
    return policy, recorder


def _make_observation(batch_size=1):
    return {
        "video": {
            k: np.random.randint(0, 255, (batch_size, 1, 256, 256, 3), dtype=np.uint8)
            for k in VIDEO_KEYS
        },
        "state": {k: np.random.randn(batch_size, 1, 1).astype(np.float32) for k in STATE_KEYS},
        "language": {LANGUAGE_KEY: [["pick up the apple"]] * batch_size},
    }


class TestRTCDisabled:
    def test_no_rtc_config_keeps_legacy_model_call(self):
        """rtc_config=None must call model.get_action without an options kwarg and
        return an empty info dict — byte-compatible with pre-RTC behavior."""
        policy, recorder = _make_policy(rtc_config=None)
        action, info = policy.get_action(_make_observation())
        assert info == {}
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["options"] is None
        assert "action" not in recorder.calls[0]["inputs"]
        # No cache is retained when RTC is off.
        assert policy._rtc_prev_action is None


class TestRTCEnabled:
    def _policy(self, **cfg_kwargs):
        cfg = RTCConfig(execution_horizon=8, **cfg_kwargs)
        return _make_policy(rtc_config=cfg)

    def test_first_call_no_injection_but_caches(self):
        policy, recorder = self._policy()
        _, info = policy.get_action(_make_observation())
        assert info["rtc_applied"] is False
        assert "action" not in recorder.calls[0]["inputs"]
        assert recorder.calls[0]["options"] is None
        assert policy._rtc_prev_action is not None
        assert policy._rtc_prev_action.shape == (1, ACTION_HORIZON, 7)

    def test_second_call_injects_prev_chunk_with_options(self):
        policy, recorder = self._policy()
        policy.get_action(_make_observation())
        first_pred = policy._rtc_prev_action.clone()

        _, info = policy.get_action(_make_observation())
        assert info["rtc_applied"] is True
        assert info["rtc_overlap_steps"] == ACTION_HORIZON - 8

        call = recorder.calls[1]
        assert torch.equal(call["inputs"]["action"], first_pred)
        assert call["options"] == {
            "action_horizon": ACTION_HORIZON,
            "rtc_overlap_steps": ACTION_HORIZON - 8,
            "rtc_frozen_steps": 0,
            "rtc_ramp_rate": 5.0,
        }
        # Cache rolled forward to the second prediction.
        assert not torch.equal(policy._rtc_prev_action, first_pred)

    def test_per_call_executed_steps_override(self):
        policy, recorder = self._policy()
        policy.get_action(_make_observation())
        policy.get_action(_make_observation(), {"rtc": {"executed_steps": 12}})
        assert recorder.calls[1]["options"]["rtc_overlap_steps"] == ACTION_HORIZON - 12

    def test_per_call_disable(self):
        policy, recorder = self._policy()
        policy.get_action(_make_observation())
        _, info = policy.get_action(_make_observation(), {"rtc": {"enabled": False}})
        assert info["rtc_applied"] is False
        assert "action" not in recorder.calls[1]["inputs"]
        assert recorder.calls[1]["options"] is None

    def test_reset_clears_cache(self):
        policy, recorder = self._policy()
        policy.get_action(_make_observation())
        policy.reset()
        assert policy._rtc_prev_action is None
        _, info = policy.get_action(_make_observation())
        assert info["rtc_applied"] is False
        assert "action" not in recorder.calls[1]["inputs"]

    def test_batch_size_change_invalidates_cache(self):
        policy, recorder = self._policy()
        policy.get_action(_make_observation(batch_size=1))
        _, info = policy.get_action(_make_observation(batch_size=2))
        assert info["rtc_applied"] is False
        assert "action" not in recorder.calls[1]["inputs"]

    def test_no_cadence_information_skips_rtc(self):
        cfg = RTCConfig()  # no execution_horizon
        policy, recorder = _make_policy(rtc_config=cfg)
        policy.get_action(_make_observation())
        _, info = policy.get_action(_make_observation())  # no per-call executed_steps
        assert info["rtc_applied"] is False
        assert "action" not in recorder.calls[1]["inputs"]

    def test_executed_full_chunk_skips_rtc(self):
        policy, recorder = self._policy()
        policy.get_action(_make_observation())
        _, info = policy.get_action(
            _make_observation(), {"rtc": {"executed_steps": ACTION_HORIZON}}
        )
        assert info["rtc_applied"] is False
        assert "action" not in recorder.calls[1]["inputs"]


class TestRTCRelativeActionGuard:
    def _relative_action_configs(self):
        relative = ActionConfig(
            rep=ActionRepresentation.RELATIVE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
        absolute = ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
        return [relative] + [absolute] * (len(ACTION_KEYS) - 1)

    def test_relative_actions_refuse_rtc(self):
        with pytest.raises(ValueError, match="RELATIVE"):
            _make_policy(
                rtc_config=RTCConfig(execution_horizon=8),
                use_relative_action=True,
                action_configs=self._relative_action_configs(),
            )

    def test_relative_actions_allowed_with_flag(self):
        policy, _ = _make_policy(
            rtc_config=RTCConfig(execution_horizon=8, allow_relative=True),
            use_relative_action=True,
            action_configs=self._relative_action_configs(),
        )
        assert policy.rtc_config is not None

    def test_relative_reps_inert_when_use_relative_action_false(self):
        policy, _ = _make_policy(
            rtc_config=RTCConfig(execution_horizon=8),
            use_relative_action=False,
            action_configs=self._relative_action_configs(),
        )
        assert policy.rtc_config is not None
