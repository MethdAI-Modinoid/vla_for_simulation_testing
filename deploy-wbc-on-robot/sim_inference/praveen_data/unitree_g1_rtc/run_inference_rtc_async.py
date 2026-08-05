#!/usr/bin/env python3
"""
GR00T Policy Loop with ASYNC RTC (Real-Time Chunking) + boundary logging.

The async counterpart of run_inference_rtc.py: instead of blocking the control loop on
every policy query (stop-and-go), an AsyncChunkExecutor requests the next chunk in the
background while the robot keeps executing the current one, then swaps to the new chunk
at the co-temporal offset. Combined with server-side RTC (--rtc on run_gr00t_server.py)
this removes BOTH failure modes measured in the sim study:

  - chunk-boundary discontinuities  -> RTC inpainting/blending (server side), and
  - full-stop pauses + rushed idx0/1/2 dwells -> background inference (this client).

frozen_steps is computed automatically per call from the measured inference latency
(EMA + safety margin) — under async it finally does real work: it pins the steps of the
new chunk that execute while inference runs, so the model cannot rewrite motion that
has already happened.

Defaults follow the consolidated findings doc (2026-07-18): trigger_lead_steps=8 keeps
the blend overlap at H - 8 = 8 (the validated 8-12 band). If the console shows repeated
"chunk exhausted" warnings (inference slower than the lead), relaunch with
--trigger-lead-steps 6.

Every control tick is logged in the same CSV schema as the sync logs (first 64 columns
identical, analyzer-compatible) plus trailing columns: rtc_applied, frozen_steps_used,
executed_steps_sent, latency_ema_s, swap_offset. new_inference=1 marks SWAP ticks (the
first tick executing a new chunk) — under async these land mid-chunk (action_idx > 0),
which analyze_chunk_boundaries.py auto-detects. CSV is written on Ctrl+C.

Run (sim):  conda activate isaaclab; python run_inference_rtc_async.py --camera-host localhost
Run (real): python run_inference_rtc_async.py --camera-host 192.168.123.164

Place this file next to sdk.py and run_Inferene_without_client_for_test.py (it reuses
the adapter/publisher from there), with the RTC branch of Isaac-GR00T importable.
"""

import csv
from dataclasses import dataclass
import os
import sys
import time

import numpy as np
import rclpy
import tyro


# Same-directory imports (sdk.py, reference script) work when launched from
# data_collection/; keep the parent on sys.path for the gr00t_wbc package too.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gr00t.policy.async_chunk_executor import AsyncChunkExecutor
from gr00t.policy.server_client import PolicyClient
from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from gr00t_wbc.control.utils.telemetry import Telemetry
from run_Inferene_without_client_for_test import GR00TG1Adapter, policy_action_to_control_goal
from sdk import create_observation_reader


# Logged joint groups (7 DoF each), matching inference_logs/exp*.csv order.
LOG_GROUPS = ["left_arm", "left_hand", "right_arm", "right_hand"]
GROUP_DIM = 7


@dataclass
class AsyncRTCInferenceConfig:
    """Configuration for the async RTC policy loop."""

    # Policy server (must be started with --rtc for blending to happen)
    policy_host: str = "localhost"
    policy_port: int = 5560

    # Camera server ("localhost" for the gear_sonic sim, 192.168.123.164 for the real G1)
    camera_host: str = "localhost"
    camera_port: int = 5555

    # Control frequency
    frequency: int = 20  # Hz

    # Async RTC
    trigger_lead_steps: int = 8
    """Request the next chunk when this many steps remain in the current one. 8 keeps
    the RTC blend overlap at H - 8 = 8 steps (the validated band). Use 6 if the log
    shows repeated chunk-exhaustion warnings. 0 = auto-size from measured latency
    (gives a smaller overlap; not the validated configuration)."""

    safety_margin_steps: int = 1
    """Extra steps added to the auto frozen_steps (and the auto trigger lead), absorbing
    latency jitter and the obs-read time outside the measured window."""

    # Logging
    log_dir: str = os.path.join(os.path.dirname(_THIS_DIR), "inference_logs_rtc_async")
    """Directory for expN.csv logs (auto-incremented per run)."""

    # Language instruction
    lang_instruction: str = "observe the entire scene first. identify the red cube, blue cube, and green cube. then pick up each cube and place it into the basket with the matching color. place the red cube in the red basket, the blue cube in the blue basket, and the green cube in the green basket. ensure every cube is placed in the correct colored basket."

    # Camera configuration
    add_stereo_camera: bool = True

    # Robot configuration
    robot: str = "g1"


class AsyncPolicyAdapter(GR00TG1Adapter):
    """Policy facade for AsyncChunkExecutor: converts raw observations to the GR00T
    input format inside get_action, which runs on the executor's worker thread — the
    conversion cost stays off the control loop and uses the trigger-tick observation."""

    def get_action(self, obs: dict, options: dict = None) -> tuple:
        model_input = self.obs_to_policy_inputs(obs)
        return self.policy.get_action(model_input, options)

    def reset(self, options: dict = None):
        return self.policy.reset(options)


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
    header.extend(
        ["rtc_applied", "frozen_steps_used", "executed_steps_sent", "latency_ema_s", "swap_offset"]
    )
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


def rewrap_step_action(step_action: dict) -> dict:
    """(B, D) single-step arrays from the executor -> (B, 1, D), the shape
    policy_action_to_control_goal expects (it extracts timestep 0 of (B, T, D))."""
    wrapped = {}
    for key, value in step_action.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            wrapped[key] = value[:, None, :]
        else:
            wrapped[key] = value
    return wrapped


def main(config: AsyncRTCInferenceConfig):
    csv_path = next_log_path(config.log_dir)

    print("=" * 80)
    print("GR00T N1.7 G1 POLICY LOOP — ASYNC RTC")
    print("=" * 80)
    print(f"Policy server:   {config.policy_host}:{config.policy_port} (start it with --rtc)")
    print(f"Camera server:   {config.camera_host}:{config.camera_port}")
    print(f"Frequency:       {config.frequency} Hz")
    print(f"Trigger lead:    {config.trigger_lead_steps or 'auto'} steps before exhaustion")
    print(f"Safety margin:   {config.safety_margin_steps} step(s)")
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
    ros_manager = ROSManager(node_name="GR00TPolicyAsyncRTC")
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

    adapter = AsyncPolicyAdapter(policy_client, add_stereo_camera=config.add_stereo_camera)
    executor = AsyncChunkExecutor(
        adapter,
        control_period_s=1.0 / config.frequency,
        trigger_lead_steps=config.trigger_lead_steps if config.trigger_lead_steps > 0 else None,
        safety_margin_steps=config.safety_margin_steps,
    )

    # Episode start: reset() (clears the server-side RTC cache) + blocking first chunk.
    # After this, the executor owns ALL policy calls — never call get_action elsewhere.
    print("[5] Episode start: policy reset + first (blocking) inference...")
    first_obs = obs_reader.get_observation(config.lang_instruction)
    if first_obs is None:
        print("❌ No observation for the first inference!")
        obs_reader.shutdown()
        return
    executor.start_episode(first_obs)
    prev_chunk_id = executor.chunk_id
    print(f"✅ First chunk ready: {executor.chunk_length} timesteps")

    rate = node.create_rate(config.frequency)
    telemetry = Telemetry(window_size=100)

    iteration = 0
    last_loop_time = None
    swap_count = 0

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

                with telemetry.timer("get_next_action"):
                    step_action = executor.get_next_action(obs)

                # A swap tick = first tick executing a chunk that arrived in the
                # background (or via the blocking fallback).
                new_inference = int(executor.chunk_id != prev_chunk_id)
                prev_chunk_id = executor.chunk_id

                latency = ""
                horizon = ""
                rtc_applied = ""
                frozen_used = ""
                executed_sent = ""
                swap_offset = ""
                if new_inference:
                    swap_count += 1
                    latency = executor.last_latency_s
                    horizon = executor.chunk_length
                    rtc_applied = int(bool(executor.last_info.get("rtc_applied", False)))
                    frozen_used = executor.last_rtc_options.get("frozen_steps", "")
                    executed_sent = executor.last_rtc_options.get("executed_steps", "")
                    swap_offset = executor.last_swap_offset

                current_action = rewrap_step_action(step_action)

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
                    executor.last_step_index,
                ]
                row.extend(extract_state_joints(obs))
                row.extend(extract_target_joints(current_action))
                row.extend(
                    [
                        rtc_applied,
                        frozen_used,
                        executed_sent,
                        executor.latency_s if new_inference else "",
                        swap_offset,
                    ]
                )
                rows.append(row)

                with telemetry.timer("convert_to_control_goal"):
                    control_cmd = policy_action_to_control_goal(
                        current_action,
                        time.monotonic(),
                        config.frequency,
                    )

                with telemetry.timer("publish_control"):
                    control_publisher.publish(control_cmd)

                if iteration % config.frequency == 0:  # every ~1 s
                    ema = executor.latency_s
                    ema_str = f"{ema * 1000:.0f}ms" if ema is not None else "-"
                    print(
                        f"[Loop {iteration}] "
                        f"chunk {executor.chunk_id} idx {executor.last_step_index}/"
                        f"{executor.chunk_length} | swaps={swap_count} | "
                        f"rtc_applied(last)={executor.last_info.get('rtc_applied', '-')} | "
                        f"frozen(last)={executor.last_rtc_options.get('frozen_steps', '-')} | "
                        f"lat_ema={ema_str}"
                    )

                iteration += 1

            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.frequency):
                telemetry.log_timing_info(
                    context="GR00T Async RTC Policy Loop Missed", threshold=0.001
                )

            rate.sleep()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    except Exception as e:
        # A failed background inference surfaces here on the swap tick. Stop cleanly:
        # the WBC holds the last published command; do not keep publishing stale steps.
        print(f"\n\n❌ Control loop error: {e!r} — stopping and saving the log.")

    finally:
        executor.close()
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
    config = tyro.cli(AsyncRTCInferenceConfig)
    main(config)
