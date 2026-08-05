"""
action_smoother1.py  —  GR00T G1 smooth action execution
=========================================================

ROOT CAUSE OF YOUR JUMP PROBLEM (from logs):
─────────────────────────────────────────────
  get_action_chunk: ~75ms   ← policy BLOCKS the control loop every 16 steps
  total_loop_avg:    5.6ms  ← loop is fast when not blocked
  right_arm jump:   0.33 rad every chunk boundary

What actually happens without this fix:
  1. Robot executes steps 1-16 of chunk A smoothly  (16 × 50ms = 800ms)
  2. At step 16: loop BLOCKS 75ms waiting for chunk B  → arm freezes
  3. Chunk B t=0 is 0.33 rad away from chunk A step-16  → arm jumps
  4. Repeat every 800ms  →  arm moves in visible lurching steps

THE FIX — two layers:
  ① AsyncPolicyPrefetcher  — inference in background thread.
       At step 13/16 (prefetch_steps_before_end=3), background fetch starts.
       By step 16 it is already done.  Control loop NEVER blocks.

  ② HorizonInterpolator    — pins chunk t=0 to last robot position.
       Cosine ramp blends from anchor → policy over chunk_blend_steps.
       Eliminates the 0.33 rad jump at every boundary.

  ③ JointFilter            — EMA + per-joint velocity clamp every step.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SmootherConfig:

    # EMA low-pass: 1.0=raw policy, 0.0=frozen. Raise if too sluggish.
    ema_alpha: float = 0.30

    # Max joint change per step (rad).  @ 20Hz: 0.08 rad/step = 1.6 rad/s
    max_delta_rad: float = 0.08
    max_delta_hand_rad: float = 0.12
    wrist_ema_alpha: float = 0.35

    # Cubic spline over the action horizon
    use_cubic_interpolation: bool = True
    horizon_step_substeps: int = 1   # set 2 if policy ~10Hz, control 20Hz

    # Chunk boundary anchor ──────────────────────────────────────────────────
    # 1.0 = fully pin t=0 to last robot pos (no jump)
    # 0.0 = raw policy t=0 (original broken behaviour)
    chunk_anchor_weight: float = 1.0
    # Cosine blend from anchor → policy over N steps  (6 × 50ms = 300ms)
    chunk_blend_steps: int = 6
    # Warn if raw chunk start deviates by more than this (rad)
    max_chunk_start_jump_rad: float = 0.25

    # Async prefetch ─────────────────────────────────────────────────────────
    # Start background inference when this many steps remain in current chunk.
    # With 16-step chunks and ~75ms inference at 20Hz (50ms/step):
    #   need ceil(75/50) = 2 steps lead-time → use 3 for safety margin.
    prefetch_steps_before_end: int = 3

    # Temporal ensemble
    ensemble_size: int = 3
    ensemble_decay: float = 0.7


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

HAND_KEYS  = {"left_hand",  "right_hand"}
WRIST_KEYS = {"left_wrist_pos", "right_wrist_pos",
              "left_wrist_abs_quat", "right_wrist_abs_quat"}


def _clamp_delta(prev, target, max_d):
    return prev + np.clip(target - prev, -max_d, max_d)


def _slerp(q0, q1, t):
    q0 = q0 / (np.linalg.norm(q0) + 1e-9)
    q1 = q1 / (np.linalg.norm(q1) + 1e-9)
    dot = float(np.clip(np.dot(q0, q1), -1, 1))
    if dot < 0: q1, dot = -q1, -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / (np.linalg.norm(r) + 1e-9)
    th0 = np.arccos(dot); th = th0 * t
    return np.sin(th0 - th) / np.sin(th0) * q0 + np.sin(th) / np.sin(th0) * q1


def _wavg(arrays, weights):
    w = np.array(weights, dtype=float); w /= w.sum()
    return sum(a * wi for a, wi in zip(arrays, w))


# ─────────────────────────────────────────────────────────────────────────────
# ① Async Policy Prefetcher
# ─────────────────────────────────────────────────────────────────────────────

class AsyncPolicyPrefetcher:
    """
    Background-thread policy inference so the control loop never blocks.

    Timeline with 16-step chunk, 75ms inference, 50ms/step @ 20Hz:

    step:  1  2 … 12  13  [14  15  16] | 1  2 …
                           ↑
                    prefetch starts (step 13 of 16, 3 steps before end)
                    inference finishes ~75ms later = step 14.5
                    → chunk B ready before step 16 ends  ✓
    """

    def __init__(self, policy_fn: Callable, cfg: SmootherConfig):
        self._policy_fn   = policy_fn
        self._cfg         = cfg
        self._lock        = threading.Lock()
        self._ready_chunk: Optional[Dict] = None
        self._is_fetching = False

    def update_policy_fn(self, fn: Callable):
        """Update the obs-capturing lambda each step (thread-safe)."""
        with self._lock:
            self._policy_fn = fn

    def notify_step(self, current_idx: int, horizon: int):
        """Call every control step. Triggers prefetch when close to end."""
        if (horizon - current_idx) <= self._cfg.prefetch_steps_before_end:
            self._maybe_start()

    def _maybe_start(self):
        with self._lock:
            if self._is_fetching or self._ready_chunk is not None:
                return
            self._is_fetching = True
            fn = self._policy_fn          # capture current fn under lock
        threading.Thread(target=self._worker, args=(fn,), daemon=True).start()

    def _worker(self, fn):
        try:
            chunk, _ = fn()
            with self._lock:
                self._ready_chunk = chunk
                self._is_fetching = False
        except Exception as exc:
            print(f"  ❌ [Prefetcher] {exc}")
            with self._lock:
                self._is_fetching = False

    def pop_ready(self) -> Optional[Dict]:
        """Non-blocking. Returns finished chunk and clears slot, or None."""
        with self._lock:
            chunk = self._ready_chunk
            self._ready_chunk = None
        return chunk

    def fetch_blocking(self) -> Dict:
        """Blocking — only for the very first chunk."""
        chunk, _ = self._policy_fn()
        return chunk


# ─────────────────────────────────────────────────────────────────────────────
# ③ Per-joint EMA + velocity clamp
# ─────────────────────────────────────────────────────────────────────────────

class JointFilter:
    def __init__(self, cfg: SmootherConfig):
        self.cfg = cfg
        self._prev: Dict[str, np.ndarray] = {}
        self._ema:  Dict[str, np.ndarray] = {}

    def _alpha(self, k):
        if k in HAND_KEYS:  return min(1.0, self.cfg.ema_alpha * 1.2)
        if k in WRIST_KEYS: return self.cfg.wrist_ema_alpha
        return self.cfg.ema_alpha

    def _maxd(self, k):
        if k in HAND_KEYS:  return self.cfg.max_delta_hand_rad
        if k in WRIST_KEYS: return self.cfg.max_delta_rad * 2.0
        return self.cfg.max_delta_rad

    def filter(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in action.items():
            v    = np.asarray(v, float).flatten()
            ema  = self._alpha(k) * v + (1 - self._alpha(k)) * self._ema.get(k, v.copy())
            self._ema[k] = ema
            prev = self._prev.get(k, ema.copy())
            out[k] = _clamp_delta(prev, ema, self._maxd(k))
            self._prev[k] = out[k]
        return out

    def reset(self): self._prev.clear(); self._ema.clear()


# ─────────────────────────────────────────────────────────────────────────────
# ② Cubic spline interpolator + chunk-boundary anchor
# ─────────────────────────────────────────────────────────────────────────────

class HorizonInterpolator:
    """
    Builds a cubic spline over T waypoints.

    Chunk boundary fix — knot layout:
      t = [-1,  0,  1, …, T-1]
               ↑ policy chunk starts here (t=0)
      t = -1 is set to last robot position (the "anchor").

    The spline is therefore forced to pass through the robot's current
    position and arc smoothly into the policy trajectory.  No jump.

    Additionally, a cosine ramp over `chunk_blend_steps` blends
    from the anchor position → spline output for extra softness.
    """

    def __init__(self, cfg: SmootherConfig):
        self.cfg         = cfg
        self._splines:   Dict[str, CubicSpline] = {}
        self._anchor:    Dict[str, np.ndarray]  = {}
        self._last_pos:  Dict[str, np.ndarray]  = {}
        self._horizon    = 0
        self._substep    = 0
        self._blend_step = 0
        self._loaded     = False

    def update_last_pos(self, pos: Dict[str, np.ndarray]):
        self._last_pos = {k: v.copy() for k, v in pos.items()}

    def load_chunk(self, chunk: Dict[str, np.ndarray]):
        first_key = next(iter(chunk))
        T         = chunk[first_key].shape[1]
        warned    = False

        self._splines.clear()
        self._anchor.clear()

        for key, arr in chunk.items():
            traj = arr[0].astype(float)                  # (T, D)
            if traj.ndim == 1: traj = traj[:, np.newaxis]
            D = traj.shape[1]

            # anchor = last executed pos, or policy t=0 for first chunk
            anchor = (self._last_pos[key].copy()
                      if key in self._last_pos and self._last_pos[key].shape[0] == D
                      else traj[0].copy())

            # Safety log
            jump = float(np.max(np.abs(traj[0] - anchor)))
            if jump > self.cfg.max_chunk_start_jump_rad and not warned:
                print(f"  ⚠️  [Smoother] chunk boundary {jump:.3f} rad on '{key}'"
                      f" — anchoring to last pos")
                warned = True

            # blended t=0
            t0 = self.cfg.chunk_anchor_weight * anchor + \
                 (1 - self.cfg.chunk_anchor_weight) * traj[0]
            self._anchor[key] = anchor.copy()

            # prepend anchor knot at t = -1
            traj_aug = np.vstack([t0[np.newaxis], traj])   # (T+1, D)
            t_knots  = np.arange(-1, T, dtype=float)

            self._splines[key] = (CubicSpline(t_knots, traj_aug, bc_type='clamped')
                                  if len(t_knots) >= 4
                                  else CubicSpline(t_knots, traj_aug))

        self._horizon    = T * self.cfg.horizon_step_substeps
        self._substep    = 0
        # only use blend ramp if we have a prior position to blend from
        self._blend_step = self.cfg.chunk_blend_steps if self._last_pos else 0
        self._loaded     = True

    def get_action(self) -> Optional[Dict[str, np.ndarray]]:
        if not self._loaded: return None
        t     = self._substep / max(1, self.cfg.horizon_step_substeps)
        t_max = float(self._splines[next(iter(self._splines))].x[-1])
        t     = min(t, t_max)

        out = {}
        for key, spl in self._splines.items():
            val = spl(t).flatten()
            # cosine ease-in: 0=anchor, 1=spline
            if self._blend_step > 0 and key in self._anchor:
                prog  = 1.0 - self._blend_step / self.cfg.chunk_blend_steps
                alpha = 0.5 * (1.0 - np.cos(np.pi * prog))
                val   = alpha * val + (1.0 - alpha) * self._anchor[key]
            out[key] = val
        return out

    def advance(self):
        self._substep   += 1
        self._blend_step = max(0, self._blend_step - 1)

    def needs_update(self) -> bool:
        return not self._loaded or self._substep >= self._horizon

    def steps_remaining(self) -> int:
        return max(0, self._horizon - self._substep)

    def reset(self):
        self._splines.clear(); self._anchor.clear(); self._last_pos.clear()
        self._loaded = False; self._substep = 0; self._blend_step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Temporal ensemble
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEnsemble:
    def __init__(self, cfg: SmootherConfig):
        self.cfg    = cfg
        self._buf: deque = deque(maxlen=cfg.ensemble_size)
        self._step  = 0

    def add(self, chunk): self._buf.append((chunk, self._step))

    def get(self) -> Optional[Dict]:
        if not self._buf: return None
        contrib: Dict[str, list] = {}
        wts:     Dict[str, list] = {}
        for age, (chunk, start) in enumerate(self._buf):
            lt = self._step - start
            H  = next(iter(chunk.values())).shape[1]
            if lt < 0 or lt >= H: continue
            w = np.exp(self.cfg.ensemble_decay * age)
            for k, arr in chunk.items():
                contrib.setdefault(k, []).append(arr[0, lt].flatten())
                wts.setdefault(k, []).append(w)
        if not contrib: return None
        out = {}
        for k in contrib:
            if "quat" in k:
                ws = np.array(wts[k]); ws /= ws.sum()
                q  = contrib[k][0]
                for q2, w in zip(contrib[k][1:], ws[1:]): q = _slerp(q, q2, w)
                out[k] = q
            else:
                out[k] = _wavg(contrib[k], wts[k])
        return out

    def step(self): self._step += 1


# ─────────────────────────────────────────────────────────────────────────────
# SmoothedActionExecutor  — main public API
# ─────────────────────────────────────────────────────────────────────────────

class SmoothedActionExecutor:
    """
    Drop-in replacement for ActionBuffer.

    ASYNC MODE (recommended — eliminates the 75ms block):
    ──────────────────────────────────────────────────────
        executor = SmoothedActionExecutor(cfg)

        # Before loop: get first chunk (blocking once is OK)
        first_chunk = adapter.get_action(obs)[0]
        executor.add_chunk(first_chunk)

        # Set up async prefetch lambda (updates each step with latest obs)
        def make_policy_fn(obs_ref):
            return lambda: adapter.get_action(obs_ref[0])
        obs_ref = [obs]
        executor.setup_async(lambda: adapter.get_action(obs_ref[0]))

        # In loop:
        obs_ref[0] = obs                            # keep obs fresh
        chunk = executor.try_swap_prefetched()      # non-blocking swap
        if chunk: executor.add_chunk(chunk)
        smoothed = executor.get_smoothed_action()
        control_publisher.publish(...)
        executor.step()                             # triggers prefetch internally

    SYNC MODE (simple, but blocks 75ms every 16 steps):
    ─────────────────────────────────────────────────────
        executor = SmoothedActionExecutor(cfg)
        if executor.needs_new_chunk():
            chunk, _ = adapter.get_action(obs)
            executor.add_chunk(chunk)
        smoothed = executor.get_smoothed_action()
        executor.step()
    """

    def __init__(self, cfg: Optional[SmootherConfig] = None):
        self.cfg          = cfg or SmootherConfig()
        self.ensemble     = TemporalEnsemble(self.cfg)
        self.joint_filter = JointFilter(self.cfg)
        self.interpolator = (HorizonInterpolator(self.cfg)
                             if self.cfg.use_cubic_interpolation else None)
        self._last_smoothed: Optional[Dict] = None
        self._prefetcher: Optional[AsyncPolicyPrefetcher] = None

    # ── async setup ─────────────────────────────────────────────────────────

    def setup_async(self, policy_fn: Callable):
        """
        Enable async prefetch.  policy_fn is a zero-arg lambda that calls
        the policy and returns (action_chunk, info).
        Call this once before starting the loop.
        """
        self._prefetcher = AsyncPolicyPrefetcher(policy_fn, self.cfg)

    def update_policy_fn(self, policy_fn: Callable):
        """Update the obs-capturing lambda (call each step with latest obs)."""
        if self._prefetcher is not None:
            self._prefetcher.update_policy_fn(policy_fn)

    def try_swap_prefetched(self) -> Optional[Dict]:
        """
        Non-blocking check for a ready prefetched chunk.
        Returns the chunk if ready (so you can call add_chunk), else None.
        """
        if self._prefetcher is not None:
            return self._prefetcher.pop_ready()
        return None

    # ── main interface ───────────────────────────────────────────────────────

    def needs_new_chunk(self) -> bool:
        if self.cfg.use_cubic_interpolation:
            return self.interpolator.needs_update()
        return self.ensemble.get() is None

    def add_chunk(self, chunk: Dict[str, np.ndarray]):
        """Load a new action chunk (with automatic anchor + spline rebuild)."""
        self.ensemble.add(chunk)
        if self.cfg.use_cubic_interpolation:
            if self._last_smoothed is not None:
                self.interpolator.update_last_pos(self._last_smoothed)
            self.interpolator.load_chunk(chunk)

    def get_smoothed_action(self) -> Optional[Dict[str, np.ndarray]]:
        if self.cfg.use_cubic_interpolation and not self.interpolator.needs_update():
            raw = self.interpolator.get_action()
        else:
            raw = self.ensemble.get()
        if raw is None: return None

        smoothed = self.joint_filter.filter(raw)
        self._last_smoothed = {k: v.copy() for k, v in smoothed.items()}
        return smoothed

    def step(self):
        """Advance state by one control step. Also triggers async prefetch."""
        self.ensemble.step()
        if self.cfg.use_cubic_interpolation:
            self.interpolator.advance()
            # trigger async prefetch based on steps remaining
            if self._prefetcher is not None:
                remaining = self.interpolator.steps_remaining()
                horizon   = self.interpolator._horizon
                self._prefetcher.notify_step(horizon - remaining, horizon)

    def reset(self):
        self.joint_filter.reset()
        self._last_smoothed = None
        if self.cfg.use_cubic_interpolation:
            self.interpolator.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Control goal converter
# ─────────────────────────────────────────────────────────────────────────────

def smoothed_action_to_control_goal(
        smoothed: Dict[str, np.ndarray], now: float, freq: int) -> dict:

    def _get(key, size):
        return np.asarray(smoothed.get(key, np.zeros(size)), float).flatten()[:size]

    cmd = {}
    cmd["target_upper_body_pose"] = np.concatenate([
        _get("left_arm",   7), _get("left_hand",  7),
        _get("right_arm",  7), _get("right_hand", 7),
    ])
    cmd["wrist_pose"]             = _get("left_wrist_pos", 3)
    cmd["base_height_command"]    = float(_get("base_height_command", 1)[0])
    cmd["navigate_cmd"]           = _get("navigate_command", 3).tolist()
    cmd["toggle_policy_action"]   = False
    cmd["toggle_data_collection"] = False
    cmd["toggle_data_abort"]      = False
    cmd["timestamp"]              = now
    cmd["target_time"]            = now + 1.0 / freq
    return cmd