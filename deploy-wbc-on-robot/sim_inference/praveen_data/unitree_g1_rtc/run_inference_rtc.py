#!/usr/bin/env python3
"""
GR00T Policy Loop with RTC (Real-Time Chunking) + boundary logging.

Same control flow as run_Inferene_without_client_for_test.py, with the three
changes RTC needs on the client side:

  1. policy_client.reset() at startup — clears the server-side RTC cache so a
     new episode never blends against the previous episode's chunk tail.
  2. Re-queries the policy after `execute_steps` actions (default 8) instead of
     running the full 16-step chunk to exhaustion. RTC needs the unexecuted
     remainder of the chunk to blend with; exhausting the chunk leaves nothing.
  3. Sends options={"rtc": {"executed_steps": N}} with each query so the server
     knows how far the timeline advanced since the previous chunk.

Whether RTC is actually applied is decided by the SERVER (--rtc flag on
run_gr00t_server.py). This client sends the same requests either way, so the
A/B comparison is a pure server-side toggle. The per-inference `rtc_applied`
CSV column records what the server reported.

Every control tick is logged in the same CSV schema as inference_logs/exp*.csv
(plus a trailing rtc_applied column), so analyze_chunk_boundaries.py works on
both baseline and RTC runs. CSV is written on Ctrl+C.

Run (sim):  conda activate isaaclab; python run_inference_rtc.py --camera-host localhost
Run (real): python run_inference_rtc.py --camera-host 192.168.123.164
"""

import csv
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
import tyro

# Same-directory imports (sdk.py, reference script) work when launched from
# data_collection/; keep the parent on sys.path for the gr00t_wbc package too.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gr00t.policy.server_client import PolicyClient
from sdk import create_observation_reader
from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from gr00t_wbc.control.utils.telemetry import Telemetry

from run_Inferene_without_client_for_test import (
    GR00TG1Adapter,
    ActionBuffer,
    policy_action_to_control_goal,
)


# Logged joint groups (7 DoF each), matching inference_logs/exp*.csv order.
LOG_GROUPS = ["left_arm", "left_hand", "right_arm", "right_hand"]
GROUP_DIM = 7


@dataclass
class RTCInferenceConfig:
    """Configuration for the RTC policy loop."""

    # Policy server
    policy_host: str = "localhost"
    policy_port: int = 5560

    # Camera server ("localhost" for the gear_sonic sim, 192.168.123.164 for the real G1)
    camera_host: str = "localhost"
    camera_port: int = 5555

    # Control frequency
    frequency: int = 20  # Hz

    # RTC
    rtc: bool = True
    """Send RTC options with each query. Harmless if the server has RTC disabled."""

    execute_steps: int = 8
    """Actions executed per chunk before re-querying the policy. Must be < the
    chunk length (16) for RTC to have an overlap to blend. 16 = old exhaust-then-
    block behaviour (RTC cannot fire)."""

    # Logging
    log_dir: str = os.path.join(os.path.dirname(_THIS_DIR), "inference_logs_rtc")
    """Directory for expN.csv logs (auto-incremented per run)."""

    # Language instruction
    lang_instruction: str = "pick up the toy and place inside the basket"

    # Camera configuration
    add_stereo_camera: bool = True

    # Robot configuration
    robot: str = "g1"


class RTCAdapter(GR00TG1Adapter):
    """GR00TG1Adapter that forwards per-call options to the policy server."""

    def get_action(self, obs: dict, options: dict = None) -> tuple:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input, options)
        return action_chunk, info


class ExecuteStepsBuffer(ActionBuffer):
    """ActionBuffer that re-queries after execute_steps actions instead of the
    full horizon."""

    def __init__(self, execute_steps: int):
        super().__init__()
        self.execute_steps = execute_steps

    def needs_update(self) -> bool:
        if self.actions is None:
            return True
        return self.current_idx >= min(self.execute_steps, self.horizon)


def next_log_path(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(log_dir, f"exp{n}.csv")):
        n += 1
    return os.path.join(log_dir, f"exp{n}.csv")


def build_csv_header() -> list:
    header = [
        "iteration",
        "wall_time",
        "loop_time",
        "dt_since_last",
        "new_inference",
        "horizon",
        "inference_latency_s",
        "action_idx",
    ]
    for prefix in ("state", "target"):
        for group in LOG_GROUPS:
            for i in range(GROUP_DIM):
                header.append(f"{prefix}_{group}_{i}")
    header.append("rtc_applied")
    return header


def extract_state_joints(obs: dict) -> list:
    values = []
    for group in LOG_GROUPS:
        arr = obs.get(f"{group}.pos")
        if arr is None:
            arr = np.zeros(GROUP_DIM, dtype=np.float32)
        values.extend(np.asarray(arr).ravel().tolist())
    return values


def extract_target_joints(action: dict) -> list:
    values = []
    for group in LOG_GROUPS:
        arr = action.get(group)
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            arr = arr[0, 0]  # (B, T, D) -> (D,)
        if arr is None:
            arr = np.zeros(GROUP_DIM, dtype=np.float32)
        values.extend(np.asarray(arr).ravel().tolist())
    return values


def main(config: RTCInferenceConfig):
    csv_path = next_log_path(config.log_dir)

    print("=" * 80)
    print("GR00T N1.7 G1 POLICY LOOP — RTC" if config.rtc else "GR00T N1.7 G1 POLICY LOOP")
    print("=" * 80)
    print(f"Policy server:   {config.policy_host}:{config.policy_port}")
    print(f"Camera server:   {config.camera_host}:{config.camera_port}")
    print(f"Frequency:       {config.frequency} Hz")
    print(f"Execute steps:   {config.execute_steps} per chunk")
    print(f"RTC options:     {'sent per query' if config.rtc else 'NOT sent'}")
    print(f"CSV output:      {csv_path}")
    print(f"Language:        '{config.lang_instruction}'")
    print("=" * 80)

    rows = []

    print("\n[1] Creating observation reader...")
    obs_reader = create_observation_reader(
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        state_topic_name=STATE_TOPIC_NAME,
        frequency=config.frequency,
        add_stereo_camera=config.add_stereo_camera,
    )

    print("[2] Initializing ROS manager and publisher...")
    ros_manager = ROSManager(node_name="GR00TPolicyRTC")
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

    # Episode start: clear any server-side RTC cache from a previous run.
    print("[5] Resetting policy (clears server-side RTC chunk cache)...")
    policy_client.reset()
    print("✅ Policy reset.")

    adapter = RTCAdapter(policy_client, add_stereo_camera=config.add_stereo_camera)
    action_buffer = ExecuteStepsBuffer(config.execute_steps)

    rtc_options = (
        {"rtc": {"executed_steps": config.execute_steps}} if config.rtc else None
    )

    rate = node.create_rate(config.frequency)
    telemetry = Telemetry(window_size=100)

    iteration = 0
    last_loop_time = None

    print("\n[6] Starting control loop... Press Ctrl+C to stop and save the CSV\n")

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()

                obs = obs_reader.get_observation(config.lang_instruction)
                if obs is None:
                    print("⚠️  No observation available")
                    rate.sleep()
                    continue

                new_inference = 0
                latency = ""
                horizon = ""
                rtc_applied = ""
                if action_buffer.needs_update():
                    with telemetry.timer("get_action_chunk"):
                        t_inf = time.monotonic()
                        action_chunk, info = adapter.get_action(obs, rtc_options)
                        latency = time.monotonic() - t_inf
                        action_buffer.set_actions(action_chunk)
                        new_inference = 1
                        horizon = action_buffer.horizon
                        rtc_applied = int(bool(info.get("rtc_applied", False)))

                        if iteration == 0:
                            print(f"\n✅ Received action horizon: {horizon} timesteps")
                            print(f"   Action keys: {list(action_chunk.keys())}")
                            print(f"   info: {info}\n")

                current_idx = action_buffer.current_idx
                current_action = action_buffer.get_current_action()

                # ===== RECORD ROW =====
                loop_time = time.monotonic()
                dt = 0.0 if last_loop_time is None else loop_time - last_loop_time
                last_loop_time = loop_time
                row = [
                    iteration,
                    time.time(),
                    loop_time,
                    dt,
                    new_inference,
                    horizon,
                    latency,
                    current_idx,
                ]
                row.extend(extract_state_joints(obs))
                row.extend(extract_target_joints(current_action))
                row.append(rtc_applied)
                rows.append(row)

                with telemetry.timer("convert_to_control_goal"):
                    control_cmd = policy_action_to_control_goal(
                        current_action,
                        time.monotonic(),
                        config.frequency,
                    )

                with telemetry.timer("publish_control"):
                    control_publisher.publish(control_cmd)

                action_buffer.advance()

                if iteration % config.frequency == 0:  # every ~1 s
                    print(
                        f"[Loop {iteration}] "
                        f"Action idx: {action_buffer.current_idx}/{config.execute_steps} "
                        f"(chunk horizon {action_buffer.horizon}) | "
                        f"rtc_applied(last)={rtc_applied if rtc_applied != '' else '-'}"
                    )

                iteration += 1

            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.frequency):
                telemetry.log_timing_info(
                    context="GR00T RTC Policy Loop Missed", threshold=0.001
                )

            rate.sleep()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    finally:
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(build_csv_header())
                writer.writerows(rows)
            print(f"💾 Saved {len(rows)} rows to {csv_path}")
        else:
            print("No rows collected — nothing saved.")
        obs_reader.shutdown()
        ros_manager.shutdown()
        print("✅ Shutdown complete!")


if __name__ == "__main__":
    config = tyro.cli(RTCInferenceConfig)
    main(config)
