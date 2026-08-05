#!/usr/bin/env python3
"""
GR00T Policy Loop — SMOOTH motion edition
=========================================

Same control flow as run_Inferene_without_client_for_test.py, but the raw
ActionBuffer is replaced by SmoothedActionExecutor (action_smoother.py) so the
arm/hand motion sent to the robot is very smooth.

What makes it smooth (all from action_smoother.py):
  ① AsyncPolicyPrefetcher — inference runs in a background thread, so the
     control loop never blocks ~75ms every chunk boundary (kills the "freeze
     then lurch").
  ② HorizonInterpolator — cubic spline over the chunk, with the first knot
     anchored to the robot's last commanded position, so there is no jump at
     the seam between chunks. A cosine ramp eases in on top of that.
  ③ JointFilter — per-step EMA low-pass + per-joint velocity clamp.
  ④ TemporalEnsemble — averages overlapping chunk predictions.

The defaults below are tuned for EXTRA smooth motion. Every knob is exposed on
the CLI so you can trade smoothness against responsiveness:

  --smooth.ema-alpha 0.30           # higher = more responsive, lower = smoother
  --smooth.max-delta-rad 0.06       # per-step arm velocity cap (rad/step)
  --smooth.chunk-blend-steps 8      # longer = softer chunk transitions

Note: more smoothing = more lag. If the arm feels sluggish or can't keep up
with the task, raise ema-alpha / max-delta-rad and lower chunk-blend-steps.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
import tyro

from gr00t.policy.server_client import PolicyClient
from sdk import create_observation_reader
from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from gr00t_wbc.control.utils.telemetry import Telemetry

# Reuse the observation->policy adapter from the reference inference script.
from run_Inferene_without_client_for_test import GR00TG1Adapter

# The smoothing machinery.
from action_smoother import (
    SmootherConfig,
    SmoothedActionExecutor,
    smoothed_action_to_control_goal,
)


def make_smooth_config() -> SmootherConfig:
    """Smooth-AND-accurate preset (overridable via CLI).

    Design principle: get the smoothness from the cubic spline + chunk-boundary
    anchor (which removes jerk WITHOUT losing accuracy, because the spline still
    passes through the VLA waypoints), and keep the EMA / velocity clamp / blend
    as light safety nets so the arm actually reaches the VLA-commanded pose.

    Why the earlier "very smooth" preset couldn't grasp
    ---------------------------------------------------
    Measured on the recorded data, the VLA commands per-step motions up to
    ~1.0 rad (arm) and ~1.5 rad (hand) on the reach/grasp. The old clamp of
    0.06 rad/step (=1.2 rad/s) clipped those fast moves, so the arm fell short
    within each 16-step chunk and the shortfall accumulated chunk over chunk.

    Fixes:
      - max_delta_rad 0.06 -> 0.30  : clamp becomes a safety net (6 rad/s), no
                                      longer clips normal reach motion.
      - max_delta_hand_rad -> 0.40  : hands can close fast enough to grasp.
      - chunk_blend_steps 8 -> 3    : short seam blend, so the arm spends the
                                      chunk tracking the VLA, not dragging back.
      - ema_alpha 0.22 -> 0.50      : far less low-pass lag (spline does the
                                      smoothing; EMA only trims residual noise).
    """
    return SmootherConfig(
        ema_alpha=0.50,            # was 0.22 -> much less lag, still de-noises
        max_delta_rad=0.30,        # was 0.06 -> safety net only (6 rad/s @ 20Hz)
        max_delta_hand_rad=0.40,   # was 0.10 -> allow fast grasp close
        wrist_ema_alpha=0.50,      # was 0.30
        use_cubic_interpolation=True,   # spline = smooth AND passes through VLA
        chunk_anchor_weight=1.0,   # pin chunk start to last pos (kills the jump)
        chunk_blend_steps=3,       # was 8 -> short seam, keeps forward progress
        prefetch_steps_before_end=3,
        ensemble_size=3,
        ensemble_decay=0.7,
    )


@dataclass
class SmoothPolicyConfig:
    """Configuration for the smooth GR00T policy loop."""

    # Policy server
    policy_host: str = "localhost"
    policy_port: int = 5560

    # Camera server
    camera_host: str = "192.168.123.164"
    camera_port: int = 5555

    # Control frequency
    frequency: int = 50  # Hz

    # Language instruction
    lang_instruction: str = "pick up the toy and place inside the basket"

    # Camera configuration
    add_stereo_camera: bool = True

    # Robot configuration
    robot: str = "g1"

    # Smoothing (all sub-fields overridable, e.g. --smooth.ema-alpha 0.3)
    smooth: SmootherConfig = field(default_factory=make_smooth_config)


def main(config: SmoothPolicyConfig):
    """Main smooth GR00T policy loop."""

    print("=" * 80)
    print("GR00T N1.6 G1 POLICY CONTROL LOOP — SMOOTH MOTION")
    print("=" * 80)
    print(f"Policy server:  {config.policy_host}:{config.policy_port}")
    print(f"Camera server:  {config.camera_host}:{config.camera_port}")
    print(f"Frequency:      {config.frequency} Hz")
    print(f"Language:       '{config.lang_instruction}'")
    print("Smoothing:")
    print(f"  ema_alpha           = {config.smooth.ema_alpha}")
    print(f"  max_delta_rad       = {config.smooth.max_delta_rad}")
    print(f"  max_delta_hand_rad  = {config.smooth.max_delta_hand_rad}")
    print(f"  chunk_blend_steps   = {config.smooth.chunk_blend_steps}")
    print(f"  cubic_interpolation = {config.smooth.use_cubic_interpolation}")
    print(f"  async prefetch      = ON")
    print("=" * 80)

    # Create observation reader FIRST (it initializes ROS internally)
    print("\n[1] Creating observation reader...")
    obs_reader = create_observation_reader(
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        state_topic_name=STATE_TOPIC_NAME,
        frequency=config.frequency,
        add_stereo_camera=config.add_stereo_camera,
    )

    # Now create ROS manager (ROS already initialized by obs_reader)
    print("[2] Initializing ROS manager and publisher...")
    ros_manager = ROSManager(node_name="GR00TPolicySmooth")
    node = ros_manager.node
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

    # Wait for observations
    print("[3] Waiting for first observations...")
    if not obs_reader.wait_until_ready(timeout=10.0):
        print("❌ Failed to get observations!")
        obs_reader.shutdown()
        return
    print("✅ Observations ready!")

    # Connect to policy server
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

    # Adapter + smoothed executor
    print("\n[5] Initializing adapter and smoother...")
    adapter = GR00TG1Adapter(policy_client, add_stereo_camera=config.add_stereo_camera)
    executor = SmoothedActionExecutor(config.smooth)

    # The PolicyClient uses a single ZMQ REQ socket, which is NOT thread-safe and
    # requires strict send->recv alternation. The async prefetcher calls the
    # policy from a background thread while the main loop may also call it in the
    # blocking fallback below. Serialize ALL policy calls behind one lock so the
    # socket is only ever used by one thread at a time (no concurrent send).
    policy_lock = threading.Lock()

    def get_action_locked(observation):
        with policy_lock:
            return adapter.get_action(observation)

    # Rate controller + telemetry
    rate = node.create_rate(config.frequency)
    telemetry = Telemetry(window_size=100)

    # ── Prime the first chunk (blocking once is fine) ─────────────────────────
    print("\n[6] Fetching first action chunk...")
    obs = obs_reader.get_observation(config.lang_instruction)
    while obs is None:
        time.sleep(0.05)
        obs = obs_reader.get_observation(config.lang_instruction)
    first_chunk, info = get_action_locked(obs)
    executor.add_chunk(first_chunk)
    print(f"✅ First chunk loaded. Action keys: {list(first_chunk.keys())}")

    # ── Set up async prefetch: lambda reads the freshest obs each call ────────
    # obs_ref is a mutable holder so the background thread always uses the
    # newest observation without us rebuilding the lambda every step.
    obs_ref = [obs]
    executor.setup_async(lambda: get_action_locked(obs_ref[0]))

    iteration = 0

    print("\n[7] Starting SMOOTH control loop...")
    print("Press Ctrl+C to stop\n")

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()

                # Fresh observation
                obs = obs_reader.get_observation(config.lang_instruction)
                if obs is None:
                    print("⚠️  No observation available")
                    rate.sleep()
                    continue

                # Keep the async prefetcher pointed at the latest obs
                obs_ref[0] = obs

                # Non-blocking: swap in a prefetched chunk if the background
                # thread finished one.
                ready = executor.try_swap_prefetched()
                if ready is not None:
                    executor.add_chunk(ready)

                # Fallback: if we somehow ran dry before the prefetch landed,
                # fetch synchronously so we never publish a stale/empty command.
                if executor.needs_new_chunk():
                    with telemetry.timer("get_action_chunk_blocking"):
                        chunk, info = get_action_locked(obs)
                        executor.add_chunk(chunk)

                # Smoothed action for this step
                smoothed = executor.get_smoothed_action()
                if smoothed is None:
                    rate.sleep()
                    continue

                # Convert to control goal and publish
                with telemetry.timer("convert_and_publish"):
                    t_now = time.monotonic()
                    control_cmd = smoothed_action_to_control_goal(
                        smoothed, t_now, config.frequency
                    )
                    control_publisher.publish(control_cmd)

                # Advance smoother state (also triggers async prefetch internally)
                executor.step()

                # Log periodically
                if iteration % config.frequency == 0:  # ~every 1 second
                    print(f"[Loop {iteration}] "
                          f"Time: {time.monotonic() - t_start:.3f}s | "
                          f"steps_remaining: {executor.interpolator.steps_remaining() if executor.interpolator else '-'}")

                iteration += 1

            # Timing check
            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.frequency):
                telemetry.log_timing_info(
                    context="Smooth Policy Loop Missed", threshold=0.001
                )

            rate.sleep()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")

    finally:
        print("\nCleaning up...")
        obs_reader.shutdown()
        ros_manager.shutdown()
        print("✅ Shutdown complete!")


if __name__ == "__main__":
    config = tyro.cli(SmoothPolicyConfig)
    main(config)
