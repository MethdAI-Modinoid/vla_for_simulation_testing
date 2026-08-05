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

"""Asynchronous chunk executor for Real-Time Chunking (RTC).

Runs policy inference in a background thread while the control loop keeps executing the
current action chunk, then swaps to the new chunk at the correct temporal offset. Combined
with server-side RTC (``run_gr00t_server.py --rtc``) this removes both chunk-boundary
discontinuities (RTC inpainting) and stop-and-go pauses (asynchronous inference), per the
"Async Inference + RTC" strategy in ``getting_started/real_world_deployment.md``.

Timeline (H = action horizon, e = steps executed at trigger time):

- At trigger, the executor sends the observation with
  ``options["rtc"]["executed_steps"] = e``; the server aligns the new chunk so its step 0
  is co-temporal with the previous chunk's step ``e``.
- While inference runs (k control ticks), the robot keeps executing previous-chunk steps
  ``e .. e+k-1``. These are exactly the new chunk's steps ``0 .. k-1``, which the RTC
  frozen prefix (sent as measured latency in steps) pins to the previous plan.
- On arrival, execution continues from the new chunk at offset ``k`` — no step is
  replayed and no step is skipped.

Usage (robot-side control loop):

    policy = PolicyClient(host=..., port=...)   # server started with --rtc
    executor = AsyncChunkExecutor(policy, control_period_s=1 / 50)
    executor.start_episode(get_observation())   # blocking first inference
    while running:
        action = executor.get_next_action(get_observation())
        robot.execute(action)                   # one control tick
    executor.close()

The executor owns all ``policy.get_action`` calls (a single worker thread ensures at most
one in-flight request — required because the ZMQ REQ socket is not thread-safe). Do not
call ``policy.get_action`` elsewhere while an executor is attached.
"""

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import math
import time
from typing import Any

import numpy as np

from .policy import BasePolicy


logger = logging.getLogger(__name__)


def _slice_step(action_chunk: dict[str, Any], step: int) -> dict[str, Any]:
    """Extract one control step from a chunk dict of (B, T, D) or (T, D) arrays."""
    single = {}
    for key, value in action_chunk.items():
        arr = np.asarray(value)
        if arr.ndim >= 3:
            single[key] = arr[:, step]
        else:
            single[key] = arr[step]
    return single


def _chunk_length(action_chunk: dict[str, Any]) -> int:
    key, value = next(iter(action_chunk.items()))
    arr = np.asarray(value)
    if arr.ndim >= 3:
        return arr.shape[1]
    if arr.ndim >= 1:
        return arr.shape[0]
    raise ValueError(f"Action key '{key}' has no time dimension (shape {arr.shape})")


class AsyncChunkExecutor:
    """Drives a policy asynchronously, one control tick at a time.

    Args:
        policy: The policy (typically a ``PolicyClient`` against an RTC-enabled server).
            The executor takes ownership of its ``get_action``/``reset`` calls.
        control_period_s: Control loop period in seconds (1 / control frequency). Used to
            convert measured inference latency into frozen steps.
        trigger_lead_steps: Trigger the next inference when this many steps remain in the
            current chunk. ``None`` (default) auto-sizes to the measured latency in steps
            plus ``safety_margin_steps`` — large enough that the new chunk normally
            arrives before the old one runs out.
        safety_margin_steps: Extra steps added to both the auto trigger lead and the
            frozen-steps estimate to absorb latency jitter.
        latency_ema_alpha: EMA coefficient for the inference-latency estimate.
        get_action_extra_options: Extra entries merged into every request's options dict
            (outside the ``"rtc"`` key).

    Observability attributes (for control-loop logging; updated as inferences complete
    and chunks swap, read-only for callers):

    - ``chunk_id``: increments every time a new chunk is installed — log a chunk
      boundary on the tick where it changes.
    - ``last_step_index``: index into the current chunk of the step returned by the most
      recent ``get_next_action`` call.
    - ``last_info``: info dict returned by the most recent completed inference
      (``rtc_applied`` / ``rtc_overlap_steps`` / ``rtc_frozen_steps`` from an RTC server).
    - ``last_latency_s``: raw (un-smoothed) latency of that inference; ``latency_s`` is
      the EMA used for frozen-steps/trigger sizing.
    - ``last_rtc_options``: the ``options["rtc"]`` dict actually sent with the most
      recent request (``executed_steps`` / ``frozen_steps`` vary call-to-call).
    - ``last_swap_offset``: co-temporal offset the most recent swap continued from.
    """

    def __init__(
        self,
        policy: BasePolicy,
        control_period_s: float,
        trigger_lead_steps: int | None = None,
        safety_margin_steps: int = 1,
        latency_ema_alpha: float = 0.3,
        get_action_extra_options: dict[str, Any] | None = None,
    ):
        if control_period_s <= 0:
            raise ValueError(f"control_period_s must be > 0, got {control_period_s}")
        self.policy = policy
        self.control_period_s = float(control_period_s)
        self.trigger_lead_steps = trigger_lead_steps
        self.safety_margin_steps = int(safety_margin_steps)
        self.latency_ema_alpha = float(latency_ema_alpha)
        self.get_action_extra_options = dict(get_action_extra_options or {})

        # Single worker => at most one in-flight request, and the (thread-unsafe) client
        # socket is only ever used from that worker.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-infer")
        self._pending: Future | None = None
        self._latency_ema_s: float | None = None

        self._chunk: dict[str, Any] | None = None
        self._chunk_len = 0
        self._index = 0  # next step of the current chunk to return
        self._ticks_since_trigger = 0

        # Observability (see class docstring). Written by the worker thread
        # (last_info/last_latency_s/last_rtc_options) and the control-loop thread;
        # single attribute assignments, so safe to read from the control loop.
        self.chunk_id = 0
        self.last_step_index = -1
        self.last_info: dict[str, Any] = {}
        self.last_latency_s: float | None = None
        self.last_rtc_options: dict[str, Any] = {}
        self.last_swap_offset = 0

    # ------------------------------------------------------------------ lifecycle

    def start_episode(self, observation: dict[str, Any]) -> None:
        """Reset the policy and run the (blocking) first inference of an episode."""
        if self._pending is not None:
            self._pending.result()  # drain any stray in-flight request
            self._pending = None
        self.policy.reset()
        chunk, _ = self._infer(observation, executed_steps=None)
        self._set_chunk(chunk, start_index=0)

    def close(self) -> None:
        self._worker.shutdown(wait=True)

    def __enter__(self) -> "AsyncChunkExecutor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ control loop API

    def get_next_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return the action for one control tick; triggers/swaps chunks as needed.

        Call exactly once per control period. ``observation`` should be the latest
        available observation (it is only sent when this tick triggers an inference).
        """
        if self._chunk is None:
            raise RuntimeError("Call start_episode() before get_next_action().")

        # 1) Swap in a finished inference at the correct temporal offset.
        if self._pending is not None and self._pending.done():
            self._swap_to_pending()

        # 2) Out of actions and still waiting: block (degrades to stop-and-go, but safe).
        if self._index >= self._chunk_len:
            if self._pending is None:
                logger.warning("Chunk exhausted with no inference in flight; blocking.")
                chunk, _ = self._infer(observation, executed_steps=self._chunk_len)
                self._set_chunk(chunk, start_index=0)
            else:
                logger.warning(
                    "Chunk exhausted before the next one arrived (latency > lead); "
                    "blocking on the in-flight inference."
                )
                self._swap_to_pending()

        # 3) Trigger the next inference when few enough steps remain.
        remaining = self._chunk_len - self._index
        if self._pending is None and remaining <= self._trigger_lead():
            self._submit(observation, executed_steps=self._index)

        # 4) Return this tick's action.
        action = _slice_step(self._chunk, self._index)
        self.last_step_index = self._index
        self._index += 1
        self._ticks_since_trigger += 1
        return action

    @property
    def latency_s(self) -> float | None:
        """EMA of measured inference latency in seconds (None before the first request)."""
        return self._latency_ema_s

    @property
    def chunk_length(self) -> int:
        """Length in steps of the currently installed chunk (0 before start_episode)."""
        return self._chunk_len

    # ------------------------------------------------------------------ internals

    def _latency_steps(self) -> int:
        if self._latency_ema_s is None:
            return 1
        return max(1, math.ceil(self._latency_ema_s / self.control_period_s))

    def _trigger_lead(self) -> int:
        if self.trigger_lead_steps is not None:
            return self.trigger_lead_steps
        return self._latency_steps() + self.safety_margin_steps

    def _rtc_options(self, executed_steps: int | None) -> dict[str, Any]:
        options = dict(self.get_action_extra_options)
        if executed_steps is not None:
            options["rtc"] = {
                "executed_steps": int(executed_steps),
                "frozen_steps": self._latency_steps() + self.safety_margin_steps,
            }
        self.last_rtc_options = dict(options.get("rtc", {}))
        return options

    def _infer(
        self, observation: dict[str, Any], executed_steps: int | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Blocking inference through the worker (keeps client usage single-threaded)."""
        return self._worker.submit(
            self._timed_get_action, observation, self._rtc_options(executed_steps)
        ).result()

    def _submit(self, observation: dict[str, Any], executed_steps: int) -> None:
        self._pending = self._worker.submit(
            self._timed_get_action, observation, self._rtc_options(executed_steps)
        )
        self._ticks_since_trigger = 0

    def _timed_get_action(
        self, observation: dict[str, Any], options: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        start = time.monotonic()
        action, info = self.policy.get_action(observation, options)
        latency = time.monotonic() - start
        self.last_latency_s = latency
        self.last_info = dict(info) if info else {}
        if self._latency_ema_s is None:
            self._latency_ema_s = latency
        else:
            alpha = self.latency_ema_alpha
            self._latency_ema_s = alpha * latency + (1 - alpha) * self._latency_ema_s
        return action, info

    def _swap_to_pending(self) -> None:
        assert self._pending is not None
        chunk, _ = self._pending.result()
        self._pending = None
        # Steps of the old chunk executed since the trigger = offset into the new chunk
        # (new step k is co-temporal with old step executed_at_trigger + k).
        offset = self._ticks_since_trigger
        chunk_len = _chunk_length(chunk)
        if offset >= chunk_len:
            logger.warning(
                "Inference latency (%d steps) consumed the entire new chunk (%d steps); "
                "clamping. Increase the chunk length or reduce latency.",
                offset,
                chunk_len,
            )
            offset = chunk_len - 1
        self.last_swap_offset = offset
        self._set_chunk(chunk, start_index=offset)

    def _set_chunk(self, chunk: dict[str, Any], start_index: int) -> None:
        self._chunk = chunk
        self._chunk_len = _chunk_length(chunk)
        self._index = start_index
        self._ticks_since_trigger = 0
        self.chunk_id += 1
