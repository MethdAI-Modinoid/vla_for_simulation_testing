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

"""Tests for the asynchronous chunk executor (RTC Phase 2).

Timing-sensitive assertions are avoided: the fake policy is instantaneous, and where the
test must wait for the worker thread, it polls with a generous deadline.
"""

import threading
import time

from gr00t.policy.async_chunk_executor import AsyncChunkExecutor
import numpy as np
import pytest


HORIZON = 8


class FakeChunkPolicy:
    """Returns chunks whose values encode (call_id, step) so swaps are traceable:
    chunk value at step j = call_id * 100 + j."""

    def __init__(self, horizon: int = HORIZON):
        self.horizon = horizon
        self.calls: list[dict] = []
        self.reset_count = 0

    def get_action(self, observation, options=None):
        call_id = len(self.calls)
        self.calls.append({"observation": observation, "options": options})
        values = call_id * 100 + np.arange(self.horizon, dtype=np.float32)
        info = {"rtc_applied": bool((options or {}).get("rtc"))}
        return {"joints": values.reshape(1, self.horizon, 1)}, info

    def reset(self, options=None):
        self.reset_count += 1
        return {}


class GatedChunkPolicy(FakeChunkPolicy):
    """Blocks every get_action after the first until the test releases the gate,
    emulating inference latency deterministically (no sleeps)."""

    def __init__(self, horizon: int = HORIZON):
        super().__init__(horizon)
        self.gate = threading.Event()

    def get_action(self, observation, options=None):
        if self.calls and not self.gate.wait(timeout=5.0):
            raise TimeoutError("test gate never released")
        return super().get_action(observation, options)


def _wait_for_pending(executor, deadline_s=5.0):
    start = time.monotonic()
    while executor._pending is not None and not executor._pending.done():
        if time.monotonic() - start > deadline_s:
            pytest.fail("in-flight inference did not finish in time")
        time.sleep(0.001)


def _value(action):
    return float(np.asarray(action["joints"]).reshape(-1)[0])


class TestAsyncChunkExecutor:
    def test_start_episode_resets_and_blocks_for_first_chunk(self):
        policy = FakeChunkPolicy()
        with AsyncChunkExecutor(policy, control_period_s=0.01) as executor:
            executor.start_episode({"obs": 0})
            assert policy.reset_count == 1
            assert len(policy.calls) == 1
            # First inference carries no RTC options (no previous chunk to blend).
            assert "rtc" not in (policy.calls[0]["options"] or {})

    def test_swap_continues_at_co_temporal_offset_without_replay(self):
        """The executed step sequence must be old[0..e+k-1] then new[k..] where e is the
        executed count at trigger and k the ticks elapsed during inference — no step
        replayed, none skipped."""
        policy = FakeChunkPolicy()
        with AsyncChunkExecutor(policy, control_period_s=0.01, trigger_lead_steps=2) as executor:
            executor.start_episode({"obs": 0})

            executed = []
            # Consume chunk 0 up to and including the trigger tick: the trigger fires on
            # the tick where remaining == lead, i.e. at index HORIZON - 2 with
            # executed_steps = HORIZON - 2 (steps 0..HORIZON-3 already executed).
            for _ in range(HORIZON - 1):
                executed.append(_value(executor.get_next_action({"obs": 1})))
            # The request runs on the worker thread; wait for it before asserting.
            _wait_for_pending(executor)
            assert len(policy.calls) == 2
            assert policy.calls[1]["options"]["rtc"]["executed_steps"] == HORIZON - 2
            # One tick elapsed since trigger (the trigger tick itself) -> swap lands at
            # new-chunk offset 1: new[0] is co-temporal with the already-executed old[6].
            executed.append(_value(executor.get_next_action({"obs": 2})))

            assert executed == [float(j) for j in range(HORIZON - 1)] + [101.0]

    def test_frozen_steps_sent_with_request(self):
        policy = FakeChunkPolicy()
        with AsyncChunkExecutor(
            policy, control_period_s=0.01, trigger_lead_steps=2, safety_margin_steps=1
        ) as executor:
            executor.start_episode({"obs": 0})
            for _ in range(HORIZON - 1):
                executor.get_next_action({"obs": 1})
            _wait_for_pending(executor)
            rtc = policy.calls[1]["options"]["rtc"]
            assert rtc["frozen_steps"] >= 1  # latency estimate + safety margin

    def test_exhaustion_without_trigger_blocks_and_recovers(self):
        """trigger_lead_steps=0 disables early triggering: the executor must fall back
        to a blocking inference when the chunk runs out (stop-and-go, but safe)."""
        policy = FakeChunkPolicy()
        with AsyncChunkExecutor(policy, control_period_s=0.01, trigger_lead_steps=0) as executor:
            executor.start_episode({"obs": 0})
            executed = [_value(executor.get_next_action({"obs": 1})) for _ in range(HORIZON + 1)]
            # Chunk 0 fully consumed, then blocking re-inference starts chunk 1 at 0.
            assert executed[:HORIZON] == [float(j) for j in range(HORIZON)]
            assert executed[HORIZON] == 100.0

    def test_observability_attributes_through_trigger_swap_cycle(self):
        policy = FakeChunkPolicy()
        with AsyncChunkExecutor(policy, control_period_s=0.01, trigger_lead_steps=2) as executor:
            executor.start_episode({"obs": 0})
            assert executor.chunk_id == 1
            assert executor.last_step_index == -1
            assert executor.last_rtc_options == {}  # first inference has no RTC options
            assert executor.last_info == {"rtc_applied": False}

            for _ in range(HORIZON - 1):
                executor.get_next_action({"obs": 1})
            assert executor.last_step_index == HORIZON - 2
            _wait_for_pending(executor)

            # The triggered request's per-call options were recorded.
            assert executor.last_rtc_options["executed_steps"] == HORIZON - 2
            assert executor.last_rtc_options["frozen_steps"] >= 1
            assert executor.last_latency_s is not None
            assert executor.last_info == {"rtc_applied": True}

            executor.get_next_action({"obs": 2})  # swap tick
            assert executor.chunk_id == 2
            assert executor.last_swap_offset == 1
            assert executor.last_step_index == 1

    def test_g1_timeline_h16_lead8_swaps_at_latency_offset(self):
        """The sim deployment timeline: H=16, trigger_lead=8 (overlap band per the study),
        inference spanning ~3 extra ticks. Trigger must fire at index 8 with
        executed_steps=8 (server-side overlap = 16 - 8 = 8), and the swap must continue
        the new chunk at the co-temporal offset — no step replayed, none skipped."""
        policy = GatedChunkPolicy(horizon=16)
        with AsyncChunkExecutor(policy, control_period_s=0.01, trigger_lead_steps=8) as executor:
            executor.start_episode({"obs": 0})

            executed = []
            # Ticks 1-9 execute old[0..8]; the trigger fires on the tick where index
            # reaches 8 (remaining == lead), before old[8] is returned.
            for _ in range(9):
                executed.append(_value(executor.get_next_action({"obs": 1})))
            # Trigger fired: options were recorded at submit time (the gated request
            # itself is still blocked inside the worker).
            assert executor._pending is not None
            assert executor.last_rtc_options["executed_steps"] == 8

            # Inference still gated: 3 more ticks keep executing the old chunk.
            for _ in range(3):
                executed.append(_value(executor.get_next_action({"obs": 2})))

            policy.gate.set()
            _wait_for_pending(executor)
            assert len(policy.calls) == 2
            assert policy.calls[1]["options"]["rtc"]["executed_steps"] == 8

            # 4 ticks elapsed since trigger (trigger tick + 3) -> swap at new[4]:
            # old[8..11] were co-temporal with new[0..3].
            executed.append(_value(executor.get_next_action({"obs": 3})))
            assert executed == [float(j) for j in range(12)] + [104.0]
            assert executor.last_swap_offset == 4
            assert executor.chunk_id == 2

    def test_get_next_action_before_start_raises(self):
        policy = FakeChunkPolicy()
        with AsyncChunkExecutor(policy, control_period_s=0.01) as executor:
            with pytest.raises(RuntimeError):
                executor.get_next_action({"obs": 0})

    def test_invalid_control_period_raises(self):
        with pytest.raises(ValueError):
            AsyncChunkExecutor(FakeChunkPolicy(), control_period_s=0.0)
