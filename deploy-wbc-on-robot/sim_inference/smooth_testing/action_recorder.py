#!/usr/bin/env python3
"""
action_recorder.py — per-tick data recorder for the GR00T G1 inference loop.

Logs, for every control tick, three 28-D arm+hand vectors so we can measure
jerk and compare smoothing techniques from real data:

    raw_*   : the VLA action for this tick, BEFORE any smoothing
    pub_*   : the command we actually published, AFTER smoothing
    state_* : the robot's actual joint position (from the observation)

28-D layout = [left_arm(7), left_hand(7), right_arm(7), right_hand(7)].

Plus per-tick timing/metadata: iteration, wall_time, t_mono, dt_since_last,
loop_time, new_inference (1 at chunk boundary), horizon, action_idx,
inference_latency_s, mode.

Rows are streamed to disk and flushed every tick, so a Ctrl+C or crash mid-run
still leaves a complete, analyzable CSV. Column layout is compatible with the
existing inference_logs_rtc/exp*.csv analysis (state_* / target_*), with the
extra raw_* (pre-smoothing) and pub_* (== published target) columns.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional

import numpy as np

# The 4 arm+hand groups the upper-body command is built from, in publish order.
GROUPS = [("left_arm", 7), ("left_hand", 7), ("right_arm", 7), ("right_hand", 7)]
DIM = sum(n for _, n in GROUPS)  # 28


def _col_names(prefix: str):
    names = []
    for g, n in GROUPS:
        for j in range(n):
            names.append(f"{prefix}_{g}_{j}")
    return names


def state_vec_from_obs(obs: dict) -> np.ndarray:
    """Assemble the 28-D actual-position vector from an observation dict."""
    parts = []
    for g, n in GROUPS:
        v = obs.get(f"{g}.pos")
        if v is None:
            v = np.zeros(n, dtype=np.float32)
        parts.append(np.asarray(v, dtype=np.float32).flatten()[:n])
    return np.concatenate(parts)


class ActionRecorder:
    """Streaming CSV recorder. One instance per run; call log() each tick."""

    def __init__(self, output_path: str, mode: str, extra_meta: Optional[dict] = None):
        self.output_path = output_path
        self.mode = mode
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        self._meta_cols = [
            "iteration", "wall_time", "t_mono", "dt_since_last", "loop_time",
            "new_inference", "horizon", "action_idx", "inference_latency_s", "mode",
            "rtc_applied",
        ]
        self._cols = (
            self._meta_cols
            + _col_names("raw")
            + _col_names("pub")
            + _col_names("state")
        )

        self._f = open(output_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self._cols)
        self._extra_meta = extra_meta or {}
        self._last_t_mono = None
        self.n_rows = 0

    def log(
        self,
        iteration: int,
        raw: np.ndarray,
        pub: np.ndarray,
        state: np.ndarray,
        *,
        new_inference: bool,
        horizon: int,
        action_idx: int,
        inference_latency_s: float,
        loop_time: float,
        rtc_applied: int = 0,
    ):
        t_mono = time.monotonic()
        dt = 0.0 if self._last_t_mono is None else (t_mono - self._last_t_mono)
        self._last_t_mono = t_mono

        def fix(v):
            a = np.asarray(v, dtype=np.float64).flatten()
            if a.size < DIM:
                a = np.concatenate([a, np.zeros(DIM - a.size)])
            return a[:DIM]

        row = [
            iteration,
            time.time(),
            t_mono,
            dt,
            loop_time,
            1 if new_inference else 0,
            horizon,
            action_idx,
            inference_latency_s,
            self.mode,
            int(rtc_applied),
        ]
        row += fix(raw).tolist()
        row += fix(pub).tolist()
        row += fix(state).tolist()
        self._w.writerow(row)
        self._f.flush()
        self.n_rows += 1

    def close(self):
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
        print(f"[recorder] mode='{self.mode}' wrote {self.n_rows} ticks -> {self.output_path}")
