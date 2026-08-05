#!/usr/bin/env python3
"""
GR00T Policy Loop — RECORDER + smoothing A/B harness
====================================================

Same control flow as run_Inferene_without_client_for_test.py, but:

  1. Every tick is logged (raw VLA action vs. published command vs. robot state,
     plus timing) via action_recorder.ActionRecorder -> one CSV per run.
  2. A single --mode flag selects WHICH smoothing technique is applied to the
     published command, so each run isolates one technique on identical hardware.
     Analyze the CSVs afterwards with analyze_modes.py to pick the winner.

Modes
-----
  raw            passthrough — no smoothing (the current jerky baseline)
  lowpass        EMA low-pass:      pub = a*raw + (1-a)*prev
  clamp          velocity clamp:    pub = prev + clip(raw-prev, -maxd, +maxd)
  lowpass_clamp  EMA, then clamp
  ramp           cosine blend from last pub into the new chunk over N ticks at
                 every chunk boundary (fills the a16 -> new_a1 gap); interior raw
  smooth         ramp + lowpass + clamp combined (the "everything" per-tick fix)

Usage (run from this directory, one mode per run):
    python run_inference_recorder.py --mode raw
    python run_inference_recorder.py --mode lowpass
    python run_inference_recorder.py --mode clamp
    python run_inference_recorder.py --mode ramp
    python run_inference_recorder.py --mode lowpass_clamp
    python run_inference_recorder.py --mode smooth
    # then: python analyze_modes.py recordings/

Every knob is on the CLI, e.g.:
    python run_inference_recorder.py --mode smooth \
        --ema-alpha 0.4 --max-delta-arm 0.10 --max-delta-hand 0.30 --ramp-steps 6
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
import tyro

# Make sibling modules importable no matter the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gr00t.policy.server_client import PolicyClient
from sdk import create_observation_reader
from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from gr00t_wbc.control.utils.telemetry import Telemetry

from action_recorder import ActionRecorder, state_vec_from_obs, GROUPS, DIM
# Reuse the observation->policy adapter + action buffer from the reference script.
from run_Inferene_without_client_for_test import (
    GR00TG1Adapter,
    ActionBuffer,
)

VALID_MODES = ("raw", "lowpass", "clamp", "lowpass_clamp", "ramp", "smooth")


class RTCAdapter(GR00TG1Adapter):
    """GR00TG1Adapter that forwards per-call options (RTC) to the policy server."""

    def get_action(self, obs: dict, options: dict = None) -> tuple:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input, options)
        return action_chunk, info


class ExecuteStepsBuffer(ActionBuffer):
    """ActionBuffer that re-queries after `execute_steps` actions instead of the
    full horizon, so RTC has an unexecuted remainder of the chunk to blend with.
    execute_steps == horizon reproduces the old exhaust-then-block behaviour
    (RTC cannot fire)."""

    def __init__(self, execute_steps: int):
        super().__init__()
        self.execute_steps = execute_steps

    def needs_update(self) -> bool:
        if self.actions is None:
            return True
        return self.current_idx >= min(self.execute_steps, self.horizon)


@dataclass
class RecorderConfig:
    """Configuration for the recording / smoothing A/B loop."""

    # Which smoothing technique to apply + record this run.
    mode: str = "raw"

    # Policy server
    policy_host: str = "localhost"
    policy_port: int = 5560

    # Camera server
    camera_host: str = "localhost"
    camera_port: int = 5555

    # Control frequency (Hz)
    frequency: int = 20

    # Language instruction
    lang_instruction: str = (
        "observe the entire scene first. identify the red cube, blue cube, and "
        "green cube. then pick up each cube and place it into the basket with the "
        "matching color."
    )

    add_stereo_camera: bool = True
    robot: str = "g1"

    # ── Smoothing knobs ──────────────────────────────────────────────────────
    ema_alpha: float = 0.35        # lowpass: higher = more responsive/less smooth
    max_delta_arm: float = 0.10    # clamp: max arm joint change per tick (rad)
    max_delta_hand: float = 0.30   # clamp: max hand joint change per tick (rad)
    ramp_steps: int = 6            # ramp: cosine blend length at each boundary

    # ── Real-Time Chunking (RTC) ─────────────────────────────────────────────
    # Server-side chunk-boundary blending. Requires the policy server to be
    # started with --rtc (run_gr00t_server.py). This client sends the options
    # either way; info['rtc_applied'] reports whether the server actually did it.
    rtc: bool = False
    execute_steps: int = 8         # re-query after N steps (< horizon) so RTC has
                                   # an unexecuted remainder to blend. 8 for H=16.
    rtc_overlap_steps: int = 0     # 0 = let server use max aligned overlap (H-e)
    rtc_frozen_steps: int = 0      # 0 for this synchronous loop; async: ceil(lat/period)
    rtc_ramp_rate: float = 5.0     # exponential ramp rate for the server blend

    # ── Recording ────────────────────────────────────────────────────────────
    record: bool = True
    output_dir: str = "recordings"
    run_tag: str = ""              # optional suffix, e.g. "trial2"


# ── Per-dimension max-delta vector (arm vs hand) for the velocity clamp ──────
def _build_maxd(cfg: RecorderConfig) -> np.ndarray:
    maxd = np.empty(DIM)
    i = 0
    for g, n in GROUPS:
        val = cfg.max_delta_hand if g.endswith("hand") else cfg.max_delta_arm
        maxd[i:i + n] = val
        i += n
    return maxd


class SmoothingStage:
    """Applies the selected technique to the 28-D published command vector.

    Operates purely on the sequence of published vectors, so every mode is a
    small, transparent, isolated transform. State persists across ticks.
    """

    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg
        self.mode = cfg.mode
        self.maxd = _build_maxd(cfg)
        self._prev = None            # last published vector
        self._ramp_left = 0          # ticks remaining in the current boundary ramp
        self._ramp_anchor = None     # pub vector captured at the boundary

    def _lowpass(self, raw):
        a = self.cfg.ema_alpha
        return a * raw + (1.0 - a) * self._prev

    def _clamp(self, target):
        return self._prev + np.clip(target - self._prev, -self.maxd, self.maxd)

    def _ramp(self, raw, new_chunk):
        # Start a fresh cosine blend at each chunk boundary.
        if new_chunk and self._prev is not None and self.cfg.ramp_steps > 0:
            self._ramp_left = self.cfg.ramp_steps
            self._ramp_anchor = self._prev.copy()
        if self._ramp_left > 0 and self._ramp_anchor is not None:
            prog = 1.0 - self._ramp_left / self.cfg.ramp_steps   # 0 -> 1
            alpha = 0.5 * (1.0 - np.cos(np.pi * prog))           # cosine ease-in
            out = alpha * raw + (1.0 - alpha) * self._ramp_anchor
            self._ramp_left -= 1
            return out
        return raw

    def apply(self, raw: np.ndarray, new_chunk: bool) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64).flatten()
        # First tick: nothing to blend from — publish raw and seed state.
        if self._prev is None:
            self._prev = raw.copy()
            return raw.copy()

        if self.mode == "raw":
            out = raw
        elif self.mode == "lowpass":
            out = self._lowpass(raw)
        elif self.mode == "clamp":
            out = self._clamp(raw)
        elif self.mode == "lowpass_clamp":
            lp = self._lowpass(raw)
            out = self._prev + np.clip(lp - self._prev, -self.maxd, self.maxd)
        elif self.mode == "ramp":
            out = self._ramp(raw, new_chunk)
        elif self.mode == "smooth":
            ramped = self._ramp(raw, new_chunk)
            lp = self.cfg.ema_alpha * ramped + (1.0 - self.cfg.ema_alpha) * self._prev
            out = self._prev + np.clip(lp - self._prev, -self.maxd, self.maxd)
        else:
            out = raw

        self._prev = np.asarray(out, dtype=np.float64).flatten().copy()
        return self._prev


def raw_vec_from_action(current_action: dict) -> np.ndarray:
    """Extract the 28-D [left_arm, left_hand, right_arm, right_hand] from a
    buffered action slice (values shape (B, 1, D))."""
    def t0(key, n):
        v = current_action.get(key)
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            v = v[0, 0]
        if v is None:
            v = np.zeros(n)
        return np.asarray(v, dtype=np.float64).flatten()[:n]
    return np.concatenate([t0(g, n) for g, n in GROUPS])


def build_control_goal(pub: np.ndarray, current_action: dict, now: float, freq: int) -> dict:
    """Build the control command from the (smoothed) 28-D upper-body vector,
    keeping the same auxiliary fields as the reference converter."""
    def t0(key, default):
        v = current_action.get(key, default)
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            v = v[0, 0]
        return v

    cmd = {}
    cmd["target_upper_body_pose"] = np.asarray(pub, dtype=np.float64).flatten()[:DIM]

    lwp = t0("left_wrist_pos", np.zeros(3))
    cmd["wrist_pose"] = np.asarray(lwp, dtype=np.float64).flatten()[:3]

    bh = t0("base_height_command", np.array([0.74]))
    bh = np.asarray(bh).flatten()
    cmd["base_height_command"] = float(bh[0]) if bh.size else 0.74

    nav = t0("navigate_command", np.zeros(3))
    nav = np.asarray(nav).flatten()
    cmd["navigate_cmd"] = nav.tolist()

    cmd["toggle_policy_action"] = False
    cmd["toggle_data_collection"] = False
    cmd["toggle_data_abort"] = False
    cmd["timestamp"] = now
    cmd["target_time"] = now + (1.0 / freq)
    return cmd


def main(config: RecorderConfig):
    if config.mode not in VALID_MODES:
        print(f"❌ Unknown --mode '{config.mode}'. Valid: {', '.join(VALID_MODES)}")
        return

    print("=" * 80)
    print("GR00T G1 POLICY LOOP — RECORDER / SMOOTHING A/B")
    print("=" * 80)
    print(f"Mode:           {config.mode}")
    print(f"Policy server:  {config.policy_host}:{config.policy_port}")
    print(f"Camera server:  {config.camera_host}:{config.camera_port}")
    print(f"Frequency:      {config.frequency} Hz")
    if config.mode != "raw":
        print(f"  ema_alpha={config.ema_alpha}  max_delta_arm={config.max_delta_arm}  "
              f"max_delta_hand={config.max_delta_hand}  ramp_steps={config.ramp_steps}")
    if config.rtc:
        print(f"RTC:            ON  execute_steps={config.execute_steps} "
              f"overlap={config.rtc_overlap_steps or 'auto'} frozen={config.rtc_frozen_steps} "
              f"ramp_rate={config.rtc_ramp_rate}")
        print("                (requires policy server started with --rtc)")
    else:
        print("RTC:            OFF")
    print("=" * 80)

    # Observation reader (initializes ROS internally)
    print("\n[1] Creating observation reader...")
    obs_reader = create_observation_reader(
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        state_topic_name=STATE_TOPIC_NAME,
        frequency=config.frequency,
        add_stereo_camera=config.add_stereo_camera,
    )

    print("[2] Initializing ROS manager and publisher...")
    ros_manager = ROSManager(node_name="GR00TPolicyRecorder")
    node = ros_manager.node
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

    print("[3] Waiting for first observations...")
    if not obs_reader.wait_until_ready(timeout=10.0):
        print("❌ Failed to get observations!")
        obs_reader.shutdown()
        return
    print("✅ Observations ready!")

    print("\n[4] Connecting to policy server...")
    try:
        policy_client = PolicyClient(host=config.policy_host, port=config.policy_port)
        if not policy_client.ping():
            print("❌ Failed to connect to policy server!")
            obs_reader.shutdown()
            return
        print("✅ Connected to policy server!")
    except Exception as e:
        print(f"❌ Error: {e}")
        obs_reader.shutdown()
        return

    # RTC needs a fresh server-side cache so a new run never blends against the
    # previous run's chunk tail.
    rtc_options = None
    if config.rtc:
        print("[5a] Resetting policy (clears server-side RTC chunk cache)...")
        try:
            policy_client.reset()
            print("✅ Policy reset.")
        except Exception as e:
            print(f"⚠️  policy reset failed ({e}); continuing.")
        rtc_inner = {"executed_steps": config.execute_steps, "ramp_rate": config.rtc_ramp_rate,
                     "frozen_steps": config.rtc_frozen_steps}
        if config.rtc_overlap_steps > 0:
            rtc_inner["overlap_steps"] = config.rtc_overlap_steps
        rtc_options = {"rtc": rtc_inner}

    print("\n[5] Initializing adapter, smoother, recorder...")
    if config.rtc:
        adapter = RTCAdapter(policy_client, add_stereo_camera=config.add_stereo_camera)
        action_buffer = ExecuteStepsBuffer(config.execute_steps)
    else:
        adapter = GR00TG1Adapter(policy_client, add_stereo_camera=config.add_stereo_camera)
        action_buffer = ActionBuffer()
    stage = SmoothingStage(config)

    recorder = None
    if config.record:
        tag = f"_{config.run_tag}" if config.run_tag else ""
        rtc_tag = "_rtc" if config.rtc else ""
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(config.output_dir, f"{config.mode}{rtc_tag}{tag}_{ts}.csv")
        recorder = ActionRecorder(out_path, mode=config.mode)
        print(f"✅ Recording to {out_path}")

    rate = node.create_rate(config.frequency)
    telemetry = Telemetry(window_size=100)

    iteration = 0
    last_inference_latency = 0.0
    last_rtc_applied = 0

    print("\n[6] Starting control loop...")
    print("Press Ctrl+C to stop\n")

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()

                obs = obs_reader.get_observation(config.lang_instruction)
                if obs is None:
                    print("⚠️  No observation available")
                    rate.sleep()
                    continue

                # Fetch a new chunk if the buffer ran dry (this BLOCKS on inference).
                new_chunk = False
                if action_buffer.needs_update():
                    t_inf = time.monotonic()
                    if config.rtc:
                        action_chunk, info = adapter.get_action(obs, rtc_options)
                        last_rtc_applied = int(bool(info.get("rtc_applied", False)))
                    else:
                        action_chunk, info = adapter.get_action(obs)
                    last_inference_latency = time.monotonic() - t_inf
                    action_buffer.set_actions(action_chunk)
                    new_chunk = True
                    if iteration == 0:
                        print(f"✅ Horizon: {action_buffer.horizon} | keys: {list(action_chunk.keys())}")
                        if config.rtc:
                            print(f"   rtc_applied={last_rtc_applied} | info={info}")

                current_action = action_buffer.get_current_action()

                # raw VLA target for this tick, then apply the selected smoothing.
                raw_vec = raw_vec_from_action(current_action)
                pub_vec = stage.apply(raw_vec, new_chunk=new_chunk)

                t_now = time.monotonic()
                control_cmd = build_control_goal(pub_vec, current_action, t_now, config.frequency)
                control_publisher.publish(control_cmd)

                action_idx = action_buffer.current_idx
                action_buffer.advance()

                if recorder is not None:
                    recorder.log(
                        iteration,
                        raw=raw_vec,
                        pub=pub_vec,
                        state=state_vec_from_obs(obs),
                        new_inference=new_chunk,
                        horizon=action_buffer.horizon,
                        action_idx=action_idx,
                        inference_latency_s=(last_inference_latency if new_chunk else 0.0),
                        loop_time=time.monotonic() - t_start,
                        rtc_applied=last_rtc_applied,
                    )

                if iteration % config.frequency == 0:
                    rtc_str = f" rtc_applied={last_rtc_applied}" if config.rtc else ""
                    print(f"[{iteration}] mode={config.mode} "
                          f"idx={action_idx}/{action_buffer.horizon} "
                          f"loop={time.monotonic() - t_start:.3f}s{rtc_str}")
                iteration += 1

            rate.sleep()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        if recorder is not None:
            recorder.close()
        print("Cleaning up...")
        obs_reader.shutdown()
        ros_manager.shutdown()
        print("✅ Shutdown complete!")


if __name__ == "__main__":
    main(tyro.cli(RecorderConfig))
