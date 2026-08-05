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

"""Chunk smoothness / continuity metrics.

Quantifies the trajectory-jitter failure modes described in
``getting_started/real_world_deployment.md`` ("Quantitative diagnostic metrics"):

- :func:`metric_intra_accel` — mean intra-chunk acceleration magnitude (jerk proxy).
- :func:`metric_boundary_jump` — position discontinuity between consecutive chunks.
- :func:`metric_momentum_shift` — velocity-direction consistency across chunk boundaries.

All functions take ``chunks`` of shape ``(N_chunks, chunk_length, joint_dim)`` — the
sequence of predicted action chunks from one trajectory, in prediction order — and
``execute_steps``, the number of steps of each chunk that were actually executed before
re-planning (the receding-horizon stride). Use these to A/B the effect of Real-Time
Chunking (``--rtc``) on chunk-boundary smoothness.
"""

import numpy as np


def _validate_chunks(chunks: np.ndarray, execute_steps: int | None) -> tuple[np.ndarray, int]:
    chunks = np.asarray(chunks, dtype=np.float64)
    if chunks.ndim != 3:
        raise ValueError(
            f"chunks must have shape (N_chunks, chunk_length, joint_dim), got {chunks.shape}"
        )
    exec_steps = chunks.shape[1] if execute_steps is None else int(execute_steps)
    if not 1 <= exec_steps <= chunks.shape[1]:
        raise ValueError(
            f"execute_steps must be in [1, chunk_length={chunks.shape[1]}], got {exec_steps}"
        )
    return chunks, exec_steps


def metric_intra_accel(chunks: np.ndarray) -> float:
    """Mean intra-chunk acceleration magnitude (second-order difference L2 norm).

    Only meaningful under a fixed control frequency. Lower is smoother.

    Args:
        chunks: ``(N_chunks, chunk_length, joint_dim)`` with ``chunk_length >= 3``.
    """
    chunks, _ = _validate_chunks(chunks, None)
    if chunks.shape[1] < 3:
        raise ValueError(f"chunk_length must be >= 3 for acceleration, got {chunks.shape[1]}")
    velocity = np.diff(chunks, axis=1)
    acceleration = np.diff(velocity, axis=1)
    acc_magnitude = np.linalg.norm(acceleration, axis=-1)
    return float(np.mean(acc_magnitude))


def metric_boundary_jump(chunks: np.ndarray, execute_steps: int | None = None) -> float:
    """Mean L2 position jump between the last executed step of chunk ``i`` and step 0 of
    chunk ``i+1``. Lower is smoother; 0 means perfectly continuous boundaries.

    Args:
        chunks: ``(N_chunks, chunk_length, joint_dim)`` with ``N_chunks >= 2``.
        execute_steps: Steps executed per chunk before re-planning; ``None`` = full chunk.
    """
    chunks, exec_steps = _validate_chunks(chunks, execute_steps)
    if chunks.shape[0] < 2:
        raise ValueError(f"Need >= 2 chunks to measure boundaries, got {chunks.shape[0]}")
    last_frame_prev = chunks[:-1, exec_steps - 1, :]
    first_frame_curr = chunks[1:, 0, :]
    jumps = np.linalg.norm(first_frame_curr - last_frame_prev, axis=-1)
    return float(np.mean(jumps))


def metric_momentum_shift(chunks: np.ndarray, execute_steps: int | None = None) -> float:
    """Mean cosine similarity between the velocity at the end of each executed chunk
    segment and the velocity at the start of the next chunk. Closer to 1 is smoother.

    Args:
        chunks: ``(N_chunks, chunk_length, joint_dim)`` with ``N_chunks >= 2``.
        execute_steps: Steps executed per chunk before re-planning; must be >= 2 to
            define an end-of-segment velocity. ``None`` = full chunk.
    """
    chunks, exec_steps = _validate_chunks(chunks, execute_steps)
    if chunks.shape[0] < 2:
        raise ValueError(f"Need >= 2 chunks to measure boundaries, got {chunks.shape[0]}")
    if exec_steps < 2:
        raise ValueError("execute_steps must be >= 2 to compute end velocity")
    if chunks.shape[1] < 2:
        raise ValueError(f"chunk_length must be >= 2 for velocity, got {chunks.shape[1]}")

    idx = exec_steps - 1
    v_end = chunks[:-1, idx, :] - chunks[:-1, idx - 1, :]
    v_start = chunks[1:, 1, :] - chunks[1:, 0, :]

    dot_product = np.sum(v_end * v_start, axis=-1)
    norm_prev = np.linalg.norm(v_end, axis=-1)
    norm_curr = np.linalg.norm(v_start, axis=-1)
    epsilon = 1e-8
    cosine_sim = dot_product / (norm_prev * norm_curr + epsilon)
    return float(np.mean(cosine_sim))


def compute_chunk_metrics(chunks: np.ndarray, execute_steps: int | None = None) -> dict[str, float]:
    """All three chunk metrics as a flat dict (keys: ``intra_accel``, ``boundary_jump``,
    ``momentum_shift``). Metrics whose preconditions are not met (too few chunks/steps)
    are reported as ``nan`` rather than raising, so callers can log unconditionally.
    """
    results: dict[str, float] = {}
    for name, fn in (
        ("intra_accel", lambda c: metric_intra_accel(c)),
        ("boundary_jump", lambda c: metric_boundary_jump(c, execute_steps)),
        ("momentum_shift", lambda c: metric_momentum_shift(c, execute_steps)),
    ):
        try:
            results[name] = fn(chunks)
        except ValueError:
            results[name] = float("nan")
    return results
