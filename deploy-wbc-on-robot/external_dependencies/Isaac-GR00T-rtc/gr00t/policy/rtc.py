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

"""Real-Time Chunking (RTC) configuration and option resolution.

The N1.7 action head ships an RTC primitive (see
``Gr00tN1d7ActionHead.get_action_with_features``): when the model input batch contains an
``"action"`` entry holding the previously predicted chunk (normalized action space) and
``options`` carries ``{"action_horizon", "rtc_overlap_steps", "rtc_frozen_steps",
"rtc_ramp_rate"}``, the sampler inpaints the first ``rtc_overlap_steps`` steps of the noise
latent with the tail of the previous chunk, freezes the first ``rtc_frozen_steps`` steps
(velocity zeroed), and ramps the model's corrections in exponentially over the remaining
overlap. This removes the discontinuity between consecutive action chunks.

This module provides the policy-side plumbing for that primitive: a config dataclass, the
timeline-alignment math, and validation. All resolved option values are plain Python
``int``/``float`` so they can round-trip through the msgpack-based server-client transport.

Timeline alignment
------------------
Let ``H`` be the unpadded action horizon (``len(action delta_indices)``), ``e`` the number of
steps executed since the previous inference, and ``w = rtc_overlap_steps``. Step ``k`` of the
new chunk is co-temporal with step ``e + k`` of the previous chunk. The model inpaints from
``prev[:, H - w : H]``, so:

- With the canonical ``w = H - e`` the cached chunk is passed unshifted and the model reads
  ``prev[:, e : H]`` — exactly the not-yet-executed remainder, perfectly aligned.
- With a smaller ``w`` the cached chunk must be right-shifted by ``s = H - e - w`` so the
  model slice reads ``prev[:, e : e + w]`` (see :func:`shift_prev_chunk`).
"""

from dataclasses import dataclass
import logging
from typing import Any

import torch


logger = logging.getLogger(__name__)

#: Option keys consumed by ``Gr00tN1d7ActionHead.get_action_with_features``.
RTC_MODEL_OPTION_KEYS = (
    "action_horizon",
    "rtc_overlap_steps",
    "rtc_frozen_steps",
    "rtc_ramp_rate",
)


@dataclass
class RTCConfig:
    """Server-side RTC configuration for :class:`~gr00t.policy.gr00t_policy.Gr00tPolicy`.

    Attributes:
        execution_horizon: Default number of chunk steps the controller executes between
            policy queries. Used when a request does not carry a per-call
            ``options["rtc"]["executed_steps"]`` override. Must match the real query cadence
            of the control loop for correct chunk alignment.
        overlap_steps: Number of leading steps of the new chunk seeded from the previous
            chunk. ``None`` (default) resolves to the maximum aligned overlap
            ``H - executed_steps``.
        frozen_steps: Leading steps whose velocity is zeroed during denoising (they stay
            exactly equal to the previous plan). Use 0 for synchronous loops; for async
            loops set to ``ceil(inference_latency / control_period)``.
        ramp_rate: Rate of the exponential ramp ``1 - exp(-ramp_rate * t)`` applied between
            ``frozen_steps`` and ``overlap_steps``.
        allow_relative: Allow RTC even when the checkpoint decodes RELATIVE action groups
            against the current state. The cached chunk is anchored to the *previous*
            inference's state, so inpainting it introduces a reference-frame error equal to
            the state motion between queries. Off by default.
    """

    execution_horizon: int | None = None
    overlap_steps: int | None = None
    frozen_steps: int = 0
    ramp_rate: float = 5.0
    allow_relative: bool = False

    def __post_init__(self):
        if self.execution_horizon is not None and self.execution_horizon < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {self.execution_horizon}")
        if self.overlap_steps is not None and self.overlap_steps < 1:
            raise ValueError(f"overlap_steps must be >= 1 or None, got {self.overlap_steps}")
        if self.frozen_steps < 0:
            raise ValueError(f"frozen_steps must be >= 0, got {self.frozen_steps}")
        if self.ramp_rate <= 0.0:
            raise ValueError(f"ramp_rate must be > 0, got {self.ramp_rate}")


def resolve_rtc_options(
    unpadded_horizon: int,
    executed_steps: int,
    overlap_steps: int | None = None,
    frozen_steps: int = 0,
    ramp_rate: float = 5.0,
) -> dict[str, Any] | None:
    """Resolve and validate the model-level RTC options for one inference call.

    Args:
        unpadded_horizon: ``H`` — the embodiment's action horizon
            (``len(action delta_indices)``), NOT the model's padded ``config.action_horizon``.
        executed_steps: ``e`` — steps of the previous chunk executed since it was predicted.
        overlap_steps: Requested overlap ``w``; ``None`` resolves to ``H - e``.
        frozen_steps: Requested frozen prefix ``f``; clamped to the resolved overlap.
        ramp_rate: Exponential ramp rate.

    Returns:
        The four-key options dict for ``model.get_action`` (plain int/float values), or
        ``None`` when RTC cannot apply this call (no overlap remains).
    """
    horizon = int(unpadded_horizon)
    executed = int(executed_steps)
    if horizon < 1:
        raise ValueError(f"unpadded_horizon must be >= 1, got {horizon}")
    if executed < 1:
        raise ValueError(
            f"executed_steps must be >= 1 (at least one step runs between queries), got {executed}"
        )
    if executed >= horizon:
        # Entire chunk (or more) was executed: no unexecuted remainder to blend with.
        logger.warning(
            "RTC skipped: executed_steps (%d) >= action horizon (%d) leaves no overlap. "
            "Query the policy before the chunk is exhausted to enable blending.",
            executed,
            horizon,
        )
        return None

    max_overlap = horizon - executed
    overlap = max_overlap if overlap_steps is None else int(overlap_steps)
    if overlap > max_overlap:
        logger.warning(
            "RTC overlap_steps (%d) exceeds aligned maximum H - e = %d; clamping.",
            overlap,
            max_overlap,
        )
        overlap = max_overlap
    if overlap < 1:
        return None

    frozen = int(frozen_steps)
    if frozen < 0:
        raise ValueError(f"frozen_steps must be >= 0, got {frozen}")
    if frozen > overlap:
        logger.warning(
            "RTC frozen_steps (%d) exceeds overlap_steps (%d); clamping.", frozen, overlap
        )
        frozen = overlap

    rate = float(ramp_rate)
    if rate <= 0.0:
        raise ValueError(f"ramp_rate must be > 0, got {rate}")

    return {
        "action_horizon": horizon,
        "rtc_overlap_steps": overlap,
        "rtc_frozen_steps": frozen,
        "rtc_ramp_rate": rate,
    }


def compute_prev_chunk_shift(unpadded_horizon: int, executed_steps: int, overlap_steps: int) -> int:
    """Right-shift needed so the model's ``prev[:, H - w : H]`` slice reads ``prev[e : e + w]``.

    Zero for the canonical ``w = H - e``.
    """
    shift = int(unpadded_horizon) - int(executed_steps) - int(overlap_steps)
    if shift < 0:
        raise ValueError(
            f"overlap_steps ({overlap_steps}) > H - e "
            f"({int(unpadded_horizon) - int(executed_steps)}); cannot align previous chunk"
        )
    return shift


def shift_prev_chunk(prev_chunk: torch.Tensor, shift: int) -> torch.Tensor:
    """Right-shift the previous chunk along the time dimension, zero-filling the front.

    ``shifted[:, i] = prev_chunk[:, i - shift]``. The zero-filled front region is never read
    by the model's inpaint slice (``H - w >= shift`` by construction), but is zeroed anyway
    so misuse is loud rather than subtle.

    Args:
        prev_chunk: ``(B, T, D)`` previous predicted chunk (normalized action space).
        shift: Non-negative right shift from :func:`compute_prev_chunk_shift`.
    """
    if shift == 0:
        return prev_chunk
    if shift < 0:
        raise ValueError(f"shift must be >= 0, got {shift}")
    shifted = torch.zeros_like(prev_chunk)
    shifted[:, shift:, :] = prev_chunk[:, : prev_chunk.shape[1] - shift, :]
    return shifted


@dataclass
class RTCCallParams:
    """Effective RTC parameters for one ``get_action`` call (config merged with per-call
    client overrides)."""

    enabled: bool
    executed_steps: int | None
    overlap_steps: int | None
    frozen_steps: int
    ramp_rate: float


def merge_rtc_call_options(config: RTCConfig, call_options: dict[str, Any] | None) -> RTCCallParams:
    """Merge the server-side :class:`RTCConfig` with per-call client options.

    Clients may send ``options["rtc"]`` with each ``get_action`` request; all keys are
    optional and override the config defaults:

    - ``enabled`` (bool): disable RTC for this call.
    - ``executed_steps`` (int): steps of the previous chunk executed since it was
      predicted (falls back to ``config.execution_horizon``).
    - ``overlap_steps`` / ``frozen_steps`` / ``ramp_rate``: per-call overrides — async
      executors use ``frozen_steps`` to track measured inference latency.
    """
    params = RTCCallParams(
        enabled=True,
        executed_steps=config.execution_horizon,
        overlap_steps=config.overlap_steps,
        frozen_steps=config.frozen_steps,
        ramp_rate=config.ramp_rate,
    )

    rtc_opts = (call_options or {}).get("rtc")
    if rtc_opts is not None:
        if not isinstance(rtc_opts, dict):
            raise TypeError(f"options['rtc'] must be a dict, got {type(rtc_opts)}")
        params.enabled = bool(rtc_opts.get("enabled", True))
        if "executed_steps" in rtc_opts:
            params.executed_steps = int(rtc_opts["executed_steps"])
        if "overlap_steps" in rtc_opts:
            params.overlap_steps = int(rtc_opts["overlap_steps"])
        if "frozen_steps" in rtc_opts:
            params.frozen_steps = int(rtc_opts["frozen_steps"])
        if "ramp_rate" in rtc_opts:
            params.ramp_rate = float(rtc_opts["ramp_rate"])

    return params
