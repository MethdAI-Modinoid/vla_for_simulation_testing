#!/usr/bin/env python3
"""
G1 Jerk Diagnostics — Comprehensive live data collection during inference.

Captures ALL DDS channels at full rate and computes derived jerk metrics.
Run this IN PARALLEL with the inference loop — it only subscribes, never publishes.

Usage (inside gr00t_wbc-bash-root container):
    python jerk_diagnostics.py --output /tmp/jerk_diag_run.npz --duration 120

Outputs:
    - .npz file with raw timeseries + derived metrics
    - Real-time terminal summary every 5 seconds
    - Per-episode breakdown if grasp events are detected
"""

import argparse
import os
import sys
import time
import threading
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Joint index maps (G1 29-body + 14-hand = 43-dim) ──────────────────────
# Body motor order (from URDF/pinocchio): [Lleg 0-5, Rleg 6-11, Waist 12-14, Larm 15-21, Rarm 22-28]
# But the 43-dim action/state layout is interleaved per-side:
#   [Lleg 0-5, Rleg 6-11, Waist 12-14, Larm 15-21, Lhand 22-28, Rarm 29-35, Rhand 36-42]
# Dex3 7-DoF order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1

BODY_JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow", "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow", "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
]

HAND_JOINT_NAMES = [
    "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1",
]

# Joint groups — body indices (0-28) for the 29-body-joint arrays
BODY_GROUPS = {
    "left_leg":   list(range(0, 6)),
    "right_leg":  list(range(6, 12)),
    "waist":      list(range(12, 15)),
    "left_arm":   list(range(15, 22)),
    "right_arm":  list(range(22, 29)),
}

# Right hand finger indices in the 43-dim layout
RH_INDICES = list(range(36, 43))

# Motor-to-joint mapping for G1 29-DOF body
# This comes from the config — we'll hardcode the standard G1 mapping
G1_JOINT2MOTOR = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,       # left leg
    6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,     # right leg
    12: 12, 13: 13, 14: 14,                       # waist
    15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21,  # left arm
    22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28,  # right arm
}

NUM_BODY_JOINTS = 29
NUM_HAND_JOINTS = 7


@dataclass
class TickData:
    """One synchronized snapshot of command + state at a single instant."""
    t_wall: float             # wall-clock timestamp (time.monotonic)
    t_cmd: float              # timestamp from the command message (if available)
    body_q_cmd: np.ndarray    # (29,) commanded body joint positions
    body_dq_cmd: np.ndarray   # (29,) commanded body joint velocities
    body_tau_cmd: np.ndarray  # (29,) commanded body torques
    body_kp: np.ndarray       # (29,) kp gains
    body_kd: np.ndarray       # (29,) kd gains
    body_q_cur: np.ndarray    # (29,) actual body joint positions
    body_dq_cur: np.ndarray   # (29,) actual body joint velocities
    body_tau_cur: np.ndarray  # (29,) actual body estimated torques
    rh_q_cmd: np.ndarray      # (7,) right-hand commanded positions
    rh_q_cur: np.ndarray      # (7,) actual right-hand positions
    rh_dq_cur: np.ndarray     # (7,) actual right-hand velocities
    rh_tau_cur: np.ndarray    # (7,) actual right-hand estimated torques
    lh_q_cmd: np.ndarray      # (7,) left-hand commanded positions
    lh_q_cur: np.ndarray      # (7,) actual left-hand positions
    tick_body: int            # lowstate tick counter
    tick_hand_r: int          # right hand state counter


class RingBuffer:
    """Thread-safe ring buffer for high-frequency DDS data."""
    def __init__(self, maxlen=10000):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, item):
        with self._lock:
            self._buf.append(item)

    def get_all(self):
        with self._lock:
            return list(self._buf)

    def clear(self):
        with self._lock:
            self._buf.clear()

    def __len__(self):
        with self._lock:
            return len(self._buf)


class JerkDiagnostics:
    def __init__(self, output_path: str, duration: float = 120.0, iface: str = "enp5s0"):
        self.output_path = output_path
        self.duration = duration
        self.iface = iface
        self.running = False

        # Raw buffers (full-rate)
        self.body_state_buf = RingBuffer(maxlen=200000)  # rt/lowstate ~1kHz
        self.body_cmd_buf = RingBuffer(maxlen=50000)     # rt/lowcmd ~50Hz
        self.rh_state_buf = RingBuffer(maxlen=200000)    # rt/dex3/right/state ~815Hz
        self.rh_cmd_buf = RingBuffer(maxlen=50000)       # rt/dex3/right/cmd ~50Hz
        self.lh_state_buf = RingBuffer(maxlen=200000)    # rt/dex3/left/state
        self.vla_cmd_buf = RingBuffer(maxlen=10000)      # ControlPolicy/upper_body_pose

        # Derived metrics (computed per synchronized tick)
        self.derived_buf = RingBuffer(maxlen=200000)

        # Rate counters
        self.rate_counters = {
            "lowstate": [0, time.monotonic()],
            "lowcmd": [0, time.monotonic()],
            "rh_state": [0, time.monotonic()],
            "rh_cmd": [0, time.monotonic()],
            "lh_state": [0, time.monotonic()],
            "vla_cmd": [0, time.monotonic()],
        }

        # Last values for delta computation
        self._last_body_cmd = None
        self._last_body_cur = None
        self._last_rh_cmd = None
        self._last_rh_cur = None
        self._last_tick_body = None
        self._last_tick_hand_r = None

        # Episode tracking
        self.episodes = []
        self._episode_active = False
        self._episode_start_idx = 0
        self._episode_grasp_samples = 0
        self._grasp_threshold = 0.3  # rad — right-hand thumb_0 change to detect grasp

    def _init_dds(self):
        """Initialize DDS and create subscribers."""
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
        import unitree_sdk2py.idl.unitree_hg.msg.dds_ as hg_msg

        ChannelFactoryInitialize(0, self.iface)
        time.sleep(0.3)

        # rt/lowstate — body motor state ~1 kHz
        def on_lowstate(msg):
            t = time.monotonic()
            self.body_state_buf.append((t, msg))
            self._tick_rate("lowstate")

        sub = ChannelSubscriber("rt/lowstate", hg_msg.LowState_)
        sub.Init(on_lowstate, 10)

        # rt/lowcmd — body motor command ~50 Hz
        def on_lowcmd(msg):
            t = time.monotonic()
            self.body_cmd_buf.append((t, msg))
            self._tick_rate("lowcmd")

        sub = ChannelSubscriber("rt/lowcmd", hg_msg.LowCmd_)
        sub.Init(on_lowcmd, 10)

        # rt/dex3/right/state — right hand ~815 Hz
        def on_rh_state(msg):
            t = time.monotonic()
            self.rh_state_buf.append((t, msg))
            self._tick_rate("rh_state")

        sub = ChannelSubscriber("rt/dex3/right/state", hg_msg.HandState_)
        sub.Init(on_rh_state, 10)

        # rt/dex3/right/cmd — right hand command ~50 Hz
        def on_rh_cmd(msg):
            t = time.monotonic()
            self.rh_cmd_buf.append((t, msg))
            self._tick_rate("rh_cmd")

        sub = ChannelSubscriber("rt/dex3/right/cmd", hg_msg.HandCmd_)
        sub.Init(on_rh_cmd, 10)

        # rt/dex3/left/state — left hand
        def on_lh_state(msg):
            t = time.monotonic()
            self.lh_state_buf.append((t, msg))
            self._tick_rate("lh_state")

        sub = ChannelSubscriber("rt/dex3/left/state", hg_msg.HandState_)
        sub.Init(on_lh_state, 10)

        print("[diag] DDS subscribers initialized on", self.iface)

    def _tick_rate(self, name):
        """Increment rate counter."""
        c = self.rate_counters[name]
        c[0] += 1

    def _get_rate(self, name):
        """Get instantaneous rate (Hz) and reset counter."""
        c = self.rate_counters[name]
        now = time.monotonic()
        dt = now - c[1]
        rate = c[0] / dt if dt > 0 else 0
        c[0] = 0
        c[1] = now
        return rate

    def _extract_body_state(self, msg):
        """Extract joint positions/velocities/torques from LowState_hg."""
        q = np.zeros(NUM_BODY_JOINTS)
        dq = np.zeros(NUM_BODY_JOINTS)
        tau = np.zeros(NUM_BODY_JOINTS)
        ms = msg.motor_state
        for j in range(NUM_BODY_JOINTS):
            m = ms[G1_JOINT2MOTOR[j]]
            q[j] = m.q
            dq[j] = m.dq
            tau[j] = m.tau_est
        return q, dq, tau, msg.tick

    def _extract_body_cmd(self, msg):
        """Extract commanded positions/velocities/torques/gains from LowCmd_hg."""
        q = np.zeros(NUM_BODY_JOINTS)
        dq = np.zeros(NUM_BODY_JOINTS)
        tau = np.zeros(NUM_BODY_JOINTS)
        kp = np.zeros(NUM_BODY_JOINTS)
        kd = np.zeros(NUM_BODY_JOINTS)
        mc = msg.motor_cmd
        for j in range(NUM_BODY_JOINTS):
            m = mc[G1_JOINT2MOTOR[j]]
            q[j] = m.q
            dq[j] = m.dq
            tau[j] = m.tau
            kp[j] = m.kp
            kd[j] = m.kd
        return q, dq, tau, kp, kd

    def _extract_hand_state(self, msg):
        """Extract hand joint positions/velocities/torques from HandState_."""
        q = np.zeros(NUM_HAND_JOINTS)
        dq = np.zeros(NUM_HAND_JOINTS)
        tau = np.zeros(NUM_HAND_JOINTS)
        ms = msg.motor_state
        for j in range(NUM_HAND_JOINTS):
            if j < len(ms):
                q[j] = ms[j].q
                dq[j] = ms[j].dq
                tau[j] = ms[j].tau_est
        return q, dq, tau

    def _extract_hand_cmd(self, msg):
        """Extract commanded hand positions from HandCmd_."""
        q = np.zeros(NUM_HAND_JOINTS)
        mc = msg.motor_cmd
        for j in range(NUM_HAND_JOINTS):
            if j < len(mc):
                q[j] = mc[j].q
        return q

    def _compute_derived(self, tick: TickData):
        """Compute derived jerk/snap metrics from a synchronized tick."""
        d = {}

        # Per-joint position delta (commanded, one tick)
        if self._last_body_cmd is not None:
            d["body_dq_cmd_delta"] = tick.body_q_cmd - self._last_body_cmd
        else:
            d["body_dq_cmd_delta"] = np.zeros(NUM_BODY_JOINTS)

        if self._last_rh_cmd is not None:
            d["rh_dq_cmd_delta"] = tick.rh_q_cmd - self._last_rh_cmd
        else:
            d["rh_dq_cmd_delta"] = np.zeros(NUM_HAND_JOINTS)

        # Commanded velocity magnitude (what the command asks)
        d["body_cmd_vel_mag"] = np.abs(tick.body_dq_cmd)
        d["rh_cmd_vel_mag"] = np.abs(tick.rh_q_cmd - (self._last_rh_cmd if self._last_rh_cmd is not None else tick.rh_q_cmd))

        # Executed velocity magnitude (what the joint actually does)
        d["body_exe_vel_mag"] = np.abs(tick.body_dq_cur)
        d["rh_exe_vel_mag"] = np.abs(tick.rh_dq_cur)

        # Command/exe velocity ratio (saturation indicator) — body joints only
        for gname, indices in BODY_GROUPS.items():
            cmd_v = np.mean(d["body_cmd_vel_mag"][indices])
            exe_v = np.mean(d["body_exe_vel_mag"][indices])
            d[f"{gname}_cmd_exe_ratio"] = cmd_v / max(exe_v, 1e-6)

        # Right hand: cmd/exe ratio from delta
        rh_cmd_delta_mag = np.mean(np.abs(d["rh_dq_cmd_delta"]))
        rh_exe_delta_mag = np.mean(np.abs(tick.rh_dq_cur)) * 0.02  # approximate delta from velocity
        d["rh_cmd_exe_ratio"] = rh_cmd_delta_mag / max(rh_exe_delta_mag, 1e-6)

        # Torque demand vs estimated (saturation check for fingers)
        # Finger PD: tau = kp * (q_cmd - q_cur) - kd * dq_cur
        # The Kp for fingers is 1.0 (thumb_0 = 2.0)
        finger_kp = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        d["rh_torque_demand"] = finger_kp * (tick.rh_q_cmd - tick.rh_q_cur)
        d["rh_torque_saturated"] = np.abs(d["rh_torque_demand"]) > 1.4  # 1.4 Nm actuator limit

        # Command timing gaps
        if self._last_tick_body is not None:
            d["body_cmd_gap"] = tick.tick_body - self._last_tick_body
        else:
            d["body_cmd_gap"] = 0
        if self._last_tick_hand_r is not None:
            d["rh_cmd_gap"] = tick.tick_hand_r - self._last_tick_hand_r
        else:
            d["rh_cmd_gap"] = 0

        # Large step detection (right hand)
        d["rh_large_step"] = np.any(np.abs(d["rh_dq_cmd_delta"]) > 0.5)  # >0.5 rad in one tick

        # Update tracking
        self._last_body_cmd = tick.body_q_cmd.copy()
        self._last_rh_cmd = tick.rh_q_cmd.copy()
        self._last_tick_body = tick.tick_body
        self._last_tick_hand_r = tick.tick_hand_r

        return d

    def _synchronize_and_log(self):
        """Main logging thread — synchronizes cmd/state at the lowest rate (50 Hz cmd)."""
        print("[diag] Synchronization thread started")
        t_start = time.monotonic()

        while self.running and (time.monotonic() - t_start) < self.duration:
            time.sleep(0.005)  # 200 Hz poll

            # Get latest command (lowest rate = 50 Hz)
            body_cmds = self.body_cmd_buf.get_all()
            if not body_cmds:
                continue

            t_now = time.monotonic()
            latest_cmd_t, latest_cmd = body_cmds[-1]
            body_q_cmd, body_dq_cmd, body_tau_cmd, body_kp, body_kd = self._extract_body_cmd(latest_cmd)

            # Get latest body state (closest to command time)
            body_states = self.body_state_buf.get_all()
            if not body_states:
                continue
            # Find state closest to command time
            best_body_state = None
            best_dt = float('inf')
            for st, sm in reversed(body_states):
                dt = abs(st - latest_cmd_t)
                if dt < best_dt:
                    best_dt = dt
                    best_body_state = (st, sm)
                if dt > 0.01:  # stop searching if too far
                    break

            if best_body_state is None:
                continue

            _, body_state_msg = best_body_state
            body_q_cur, body_dq_cur, body_tau_cur, tick_body = self._extract_body_state(body_state_msg)

            # Get latest right-hand state and command
            rh_states = self.rh_state_buf.get_all()
            rh_cmds = self.rh_cmd_buf.get_all()

            if rh_states:
                rh_q_cur, rh_dq_cur, rh_tau_cur = self._extract_hand_state(rh_states[-1][1])
                tick_hand_r = 0  # HandState has no tick field; use 0
            else:
                rh_q_cur, rh_dq_cur, rh_tau_cur = np.zeros(7), np.zeros(7), np.zeros(7)
                tick_hand_r = 0

            if rh_cmds:
                rh_q_cmd = self._extract_hand_cmd(rh_cmds[-1][1])
            else:
                rh_q_cmd = np.zeros(7)

            # Left hand state
            lh_states = self.lh_state_buf.get_all()
            if lh_states:
                lh_q_cur, _, _ = self._extract_hand_state(lh_states[-1][1])
            else:
                lh_q_cur = np.zeros(7)

            # Left hand command (from body cmd, indices 22-28 in 43-dim)
            # In the body command, left hand joints are at indices 22-28
            # But we get them from the body cmd since they're in the same LowCmd
            # Actually, hands are sent via separate HandCmd messages
            # The left hand command comes from the InterpolationPolicy output
            # We'll use zeros for now since it's idle
            lh_q_cmd = np.zeros(7)

            # Build synchronized tick
            tick = TickData(
                t_wall=t_now,
                t_cmd=latest_cmd_t,
                body_q_cmd=body_q_cmd,
                body_dq_cmd=body_dq_cmd,
                body_tau_cmd=body_tau_cmd,
                body_kp=body_kp,
                body_kd=body_kd,
                body_q_cur=body_q_cur,
                body_dq_cur=body_dq_cur,
                body_tau_cur=body_tau_cur,
                rh_q_cmd=rh_q_cmd,
                rh_q_cur=rh_q_cur,
                rh_dq_cur=rh_dq_cur,
                rh_tau_cur=rh_tau_cur,
                lh_q_cmd=lh_q_cmd,
                lh_q_cur=lh_q_cur,
                tick_body=tick_body,
                tick_hand_r=tick_hand_r,
            )

            # Compute derived metrics
            derived = self._compute_derived(tick)
            self.derived_buf.append((tick, derived))

            # Episode tracking (detect grasp events via right-hand thumb_0 movement)
            thumb_delta = abs(derived["rh_dq_cmd_delta"][0]) if len(derived["rh_dq_cmd_delta"]) > 0 else 0
            if thumb_delta > self._grasp_threshold:
                if not self._episode_active:
                    self._episode_active = True
                    self._episode_start_idx = len(self.derived_buf) - 1
                    self._episode_grasp_samples = 0
                self._episode_grasp_samples += 1
            elif self._episode_active and self._episode_grasp_samples > 5:
                self.episodes.append({
                    "start_idx": self._episode_start_idx,
                    "end_idx": len(self.derived_buf) - 1,
                    "grasp_samples": self._episode_grasp_samples,
                    "start_time": self.derived_buf._buf[self._episode_start_idx][0].t_wall,
                    "end_time": t_now,
                })
                self._episode_active = False
            elif self._episode_active:
                # Episode ended (no large movement for a while)
                if self._episode_grasp_samples > 0:
                    self.episodes.append({
                        "start_idx": self._episode_start_idx,
                        "end_idx": len(self.derived_buf) - 1,
                        "grasp_samples": self._episode_grasp_samples,
                        "start_time": self.derived_buf._buf[self._episode_start_idx][0].t_wall,
                        "end_time": t_now,
                    })
                    self._episode_active = False

    def _print_status(self):
        """Print real-time status every 5 seconds."""
        t_start = time.monotonic()
        last_print = t_start

        while self.running:
            time.sleep(1.0)
            now = time.monotonic()
            if now - last_print < 5.0:
                continue
            last_print = now

            elapsed = now - t_start
            rates = {k: self._get_rate(k) for k in self.rate_counters}
            n_derived = len(self.derived_buf)
            n_episodes = len(self.episodes)

            # Current right-hand stats
            rh_cmd_latest = None
            rh_exe_latest = None
            if self.derived_buf._buf:
                last_tick, last_d = self.derived_buf._buf[-1]
                rh_cmd_latest = last_tick.rh_q_cmd
                rh_exe_latest = last_tick.rh_q_cur

            print(f"\n{'='*70}")
            print(f"[diag] {elapsed:.0f}s elapsed | {n_derived} synced ticks | {n_episodes} episodes detected")
            print(f"  DDS rates: lowstate={rates['lowstate']:.0f}Hz lowcmd={rates['lowcmd']:.0f}Hz "
                  f"rh_state={rates['rh_state']:.0f}Hz rh_cmd={rates['rh_cmd']:.0f}Hz "
                  f"lh_state={rates['lh_state']:.0f}Hz")

            if rh_cmd_latest is not None:
                delta = rh_cmd_latest - rh_exe_latest
                print(f"  RH cmd:  [{', '.join(f'{v:.3f}' for v in rh_cmd_latest)}]")
                print(f"  RH cur:  [{', '.join(f'{v:.3f}' for v in rh_exe_latest)}]")
                print(f"  RH delta:[{', '.join(f'{v:.3f}' for v in delta)}]")
                print(f"  RH |delta| max: {np.max(np.abs(delta)):.3f} rad")
            print(f"{'='*70}")

    def run(self):
        """Start collection."""
        print(f"\n{'#'*70}")
        print(f"# G1 Jerk Diagnostics")
        print(f"# Output: {self.output_path}")
        print(f"# Duration: {self.duration}s")
        print(f"# Interface: {self.iface}")
        print(f"# Press Ctrl+C to stop early")
        print(f"{'#'*70}\n")

        # Initialize DDS
        self._init_dds()
        self.running = True

        # Start sync thread
        sync_thread = threading.Thread(target=self._synchronize_and_log, daemon=True)
        sync_thread.start()

        # Start status printer
        status_thread = threading.Thread(target=self._print_status, daemon=True)
        status_thread.start()

        # Wait for duration or Ctrl+C
        try:
            time.sleep(self.duration)
        except KeyboardInterrupt:
            print("\n[diag] Interrupted by user")

        self.running = False
        time.sleep(0.5)  # Let threads finish

        # Save results
        self._save()

    def _save(self):
        """Save all collected data to .npz file."""
        print(f"\n[diag] Saving data to {self.output_path}...")

        ticks = self.derived_buf.get_all()
        if not ticks:
            print("[diag] WARNING: No data collected!")
            return

        n = len(ticks)

        # Allocate arrays
        t_wall = np.zeros(n)
        t_cmd = np.zeros(n)
        body_q_cmd = np.zeros((n, NUM_BODY_JOINTS))
        body_dq_cmd = np.zeros((n, NUM_BODY_JOINTS))
        body_q_cur = np.zeros((n, NUM_BODY_JOINTS))
        body_dq_cur = np.zeros((n, NUM_BODY_JOINTS))
        body_tau_cur = np.zeros((n, NUM_BODY_JOINTS))
        body_kp = np.zeros((n, NUM_BODY_JOINTS))
        body_kd = np.zeros((n, NUM_BODY_JOINTS))
        rh_q_cmd = np.zeros((n, NUM_HAND_JOINTS))
        rh_q_cur = np.zeros((n, NUM_HAND_JOINTS))
        rh_dq_cur = np.zeros((n, NUM_HAND_JOINTS))
        rh_tau_cur = np.zeros((n, NUM_HAND_JOINTS))
        lh_q_cur = np.zeros((n, NUM_HAND_JOINTS))
        tick_body = np.zeros(n, dtype=np.int64)

        # Derived
        rh_dq_cmd_delta = np.zeros((n, NUM_HAND_JOINTS))
        rh_torque_demand = np.zeros((n, NUM_HAND_JOINTS))
        rh_torque_saturated = np.zeros((n, NUM_HAND_JOINTS), dtype=bool)
        rh_large_step = np.zeros(n, dtype=bool)
        rh_cmd_exe_ratio = np.zeros(n)
        body_cmd_gap = np.zeros(n)

        for i, (tick, derived) in enumerate(ticks):
            t_wall[i] = tick.t_wall
            t_cmd[i] = tick.t_cmd
            body_q_cmd[i] = tick.body_q_cmd
            body_dq_cmd[i] = tick.body_dq_cmd
            body_q_cur[i] = tick.body_q_cur
            body_dq_cur[i] = tick.body_dq_cur
            body_tau_cur[i] = tick.body_tau_cur
            body_kp[i] = tick.body_kp
            body_kd[i] = tick.body_kd
            rh_q_cmd[i] = tick.rh_q_cmd
            rh_q_cur[i] = tick.rh_q_cur
            rh_dq_cur[i] = tick.rh_dq_cur
            rh_tau_cur[i] = tick.rh_tau_cur
            lh_q_cur[i] = tick.lh_q_cur
            tick_body[i] = tick.tick_body
            rh_dq_cmd_delta[i] = derived["rh_dq_cmd_delta"]
            rh_torque_demand[i] = derived["rh_torque_demand"]
            rh_torque_saturated[i] = derived["rh_torque_saturated"]
            rh_large_step[i] = derived["rh_large_step"]
            rh_cmd_exe_ratio[i] = derived["rh_cmd_exe_ratio"]
            body_cmd_gap[i] = derived["body_cmd_gap"]

        # Episode data
        ep_starts = np.array([e["start_idx"] for e in self.episodes]) if self.episodes else np.array([], dtype=int)
        ep_ends = np.array([e["end_idx"] for e in self.episodes]) if self.episodes else np.array([], dtype=int)
        ep_grasp = np.array([e["grasp_samples"] for e in self.episodes]) if self.episodes else np.array([], dtype=int)

        # Save
        np.savez_compressed(
            self.output_path,
            # Metadata
            duration_s=self.duration,
            n_ticks=n,
            n_episodes=len(self.episodes),
            interface=self.iface,
            # Joint name maps (for post-analysis)
            body_joint_names=np.array(BODY_JOINT_NAMES),
            hand_joint_names=np.array(HAND_JOINT_NAMES),
            # Raw timeseries
            t_wall=t_wall,
            t_cmd=t_cmd,
            body_q_cmd=body_q_cmd,
            body_dq_cmd=body_dq_cmd,
            body_q_cur=body_q_cur,
            body_dq_cur=body_dq_cur,
            body_tau_cur=body_tau_cur,
            body_kp=body_kp,
            body_kd=body_kd,
            rh_q_cmd=rh_q_cmd,
            rh_q_cur=rh_q_cur,
            rh_dq_cur=rh_dq_cur,
            rh_tau_cur=rh_tau_cur,
            lh_q_cur=lh_q_cur,
            tick_body=tick_body,
            # Derived metrics
            rh_dq_cmd_delta=rh_dq_cmd_delta,
            rh_torque_demand=rh_torque_demand,
            rh_torque_saturated=rh_torque_saturated,
            rh_large_step=rh_large_step,
            rh_cmd_exe_ratio=rh_cmd_exe_ratio,
            body_cmd_gap=body_cmd_gap,
            # Episodes
            ep_starts=ep_starts,
            ep_ends=ep_ends,
            ep_grasp_samples=ep_grasp,
        )

        size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print(f"[diag] Saved {n} ticks, {len(self.episodes)} episodes -> {self.output_path} ({size_mb:.1f} MB)")

        # Print summary
        self._print_summary(ticks)

    def _print_summary(self, ticks):
        """Print final summary statistics."""
        print(f"\n{'='*70}")
        print("JERK DIAGNOSTICS — SUMMARY")
        print(f"{'='*70}")

        n = len(ticks)
        if n == 0:
            print("No data collected.")
            return

        # Right hand stats
        all_rh_delta = np.array([d["rh_dq_cmd_delta"] for _, d in ticks])
        all_rh_cmd = np.array([t.rh_q_cmd for t, _ in ticks])
        all_rh_cur = np.array([t.rh_q_cur for t, _ in ticks])

        print(f"\nRight Hand (7-DoF Dex3):")
        print(f"  Samples: {n}")
        print(f"  Commanded position range: [{all_rh_cmd.min():.3f}, {all_rh_cmd.max():.3f}] rad")
        print(f"  Executed position range:  [{all_rh_cur.min():.3f}, {all_rh_cur.max():.3f}] rad")
        print(f"  Max |cmd delta|: {np.max(np.abs(all_rh_delta)):.3f} rad/tick")
        print(f"  Mean |cmd delta|: {np.mean(np.abs(all_rh_delta)):.4f} rad/tick")
        print(f"  Large steps (>0.5 rad): {np.sum([d['rh_large_step'] for _, d in ticks])} / {n}")
        print(f"  Torque saturation events: {np.sum([np.any(d['rh_torque_saturated']) for _, d in ticks])} / {n}")

        # Per-finger breakdown
        finger_names = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
        print(f"\n  Per-finger max |cmd delta| (rad):")
        for j, name in enumerate(finger_names):
            max_d = np.max(np.abs(all_rh_delta[:, j]))
            mean_d = np.mean(np.abs(all_rh_delta[:, j]))
            n_large = np.sum(np.abs(all_rh_delta[:, j]) > 0.5)
            print(f"    {name:12s}: max={max_d:.3f}  mean={mean_d:.4f}  large_steps={n_large}")

        # Body stats (arms)
        all_body_delta = np.array([d.get("body_dq_cmd_delta", np.zeros(NUM_BODY_JOINTS)) for _, d in ticks])
        print(f"\nBody joints:")
        for gname, indices in BODY_GROUPS.items():
            g_delta = all_body_delta[:, indices]
            print(f"  {gname:12s}: max|delta|={np.max(np.abs(g_delta)):.3f} rad  mean|delta|={np.mean(np.abs(g_delta)):.4f} rad")

        # Timing
        t_gaps = np.diff([t.t_wall for t, _ in ticks]) * 1000  # ms
        print(f"\nTiming:")
        print(f"  Command interval: mean={np.mean(t_gaps):.1f}ms  max={np.max(t_gaps):.1f}ms  std={np.std(t_gaps):.1f}ms")

        # Episodes
        print(f"\nEpisodes: {len(self.episodes)}")
        for i, ep in enumerate(self.episodes[:10]):
            dur = ep["end_time"] - ep["start_time"]
            print(f"  Episode {i}: {dur:.1f}s, {ep['grasp_samples']} grasp samples")

        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="G1 Jerk Diagnostics")
    parser.add_argument("--output", "-o", default="/tmp/jerk_diag_run.npz",
                        help="Output .npz file path")
    parser.add_argument("--duration", "-d", type=float, default=120.0,
                        help="Collection duration in seconds")
    parser.add_argument("--interface", "-i", default="enp5s0",
                        help="DDS network interface")
    args = parser.parse_args()

    diag = JerkDiagnostics(
        output_path=args.output,
        duration=args.duration,
        iface=args.interface,
    )
    diag.run()


if __name__ == "__main__":
    main()
