#!/usr/bin/env python3
"""
GR00T Policy Loop - Integrated with WBC Control System

Sends GR00T policy actions to the robot using the same control flow as teleop.
"""

import csv
import os
import time
import numpy as np
import rclpy
import tyro
from dataclasses import dataclass

from gr00t.policy.server_client import PolicyClient
from sdk import create_observation_reader
from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from gr00t_wbc.control.utils.telemetry import Telemetry


@dataclass
class GR00TPolicyConfig:
    """Configuration for GR00T policy loop"""
    
    # Policy server
    policy_host: str = "localhost"
    policy_port: int = 5560
    
    # Camera server
    camera_host: str = "localhost"
    camera_port: int = 5555
    
    # Control frequency
    frequency: int = 20  # Hz   fix 10
    
    # Language instruction
    lang_instruction: str = "observe the entire scene first.  ensure every cube is placed in the correct colored basket."
    
    # Camera configuration
    add_stereo_camera: bool = True
    
    # Robot configuration
    robot: str = "g1"

    # Inference data recording (for debugging / plotting)
    record: bool = True
    record_dir: str = "inference_logs"


class GR00TG1Adapter:
    """Adapter to convert G1 observations to GR00T format and decode actions"""
    
    def __init__(self, policy_client: PolicyClient, add_stereo_camera: bool = True):
        self.policy = policy_client
        self.add_stereo_camera = add_stereo_camera
        
        if add_stereo_camera:
            self.camera_keys = ["ego_view", "ego_view_left_mono", "ego_view_right_mono"]
        else:
            self.camera_keys = ["ego_view"]
        
        self.joint_groups = {
            "left_leg": 6,
            "right_leg": 6,
            "waist": 3,
            "left_arm": 7,
            "left_hand": 7,
            "right_arm": 7,
            "right_hand": 7,
        }
        
        self.wrist_groups = {
            "left_wrist_pos": 3,
            "left_wrist_abs_quat": 4,
            "right_wrist_pos": 3,
            "right_wrist_abs_quat": 4,
        }
    
    def recursive_add_extra_dim(self, obs: dict) -> dict:
        """Add extra dimension for batch and time"""
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs[key] = val[np.newaxis, ...]
            elif isinstance(val, dict):
                obs[key] = self.recursive_add_extra_dim(val)
            else:
                obs[key] = [val]
        return obs
    
    def obs_to_policy_inputs(self, obs: dict) -> dict:
        """Convert robot observation to GR00T policy input format"""
        model_obs = {}
        
        # Cameras
        model_obs["video"] = {}
        for cam_key in self.camera_keys:
            if cam_key in obs:
                model_obs["video"][cam_key] = obs[cam_key]
        
        # State
        model_obs["state"] = {}
        
        # Joint groups
        for group_name in self.joint_groups.keys():
            key = f"{group_name}.pos"
            if key in obs:
                model_obs["state"][group_name] = obs[key]
        
        # Wrist poses
        for wrist_key in self.wrist_groups.keys():
            if wrist_key in obs:
                model_obs["state"][wrist_key] = obs[wrist_key]
        
        # Navigation
        if "base_height_command" in obs:
            model_obs["state"]["base_height_command"] = obs["base_height_command"]
        if "navigate_command" in obs:
            model_obs["state"]["navigate_command"] = obs["navigate_command"]
        
        # Language
        model_obs["language"] = {
            "annotation.human.task_description": obs.get("lang", "")
        }
        
        # Add (B=1, T=1) dimensions
        model_obs = self.recursive_add_extra_dim(model_obs)
        model_obs = self.recursive_add_extra_dim(model_obs)
        
        return model_obs
    
    def get_action(self, obs: dict) -> tuple:
        """Get action chunk from policy"""
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)
        #print("seee the vla output----------------------->",action_chunk)
        return action_chunk, info


def policy_action_to_control_goal(action: dict, now: float, freq: int) -> dict:
    """
    Convert GR00T policy action to control goal format (matching teleop format)
    
    Args:
        action: Dict with keys like 'left_arm', 'right_arm', etc.
               Each value is shape (B, T, D) where we extract timestep 0
        now: Current timestamp
        freq: Control frequency
    
    Returns:
        Control command dict matching teleop format
    """
    control_cmd = {}
    
    # Extract timestep 0 from each action: (B, T, D) -> (D,)
    def extract_t0(arr):
        """Extract first timestep: (B, T, D) -> (D,)"""
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            return arr[0, 0]  # First batch, first timestep
        return arr
    
    # Upper body pose (43 joints total)
    control_cmd["target_upper_body_pose"] = np.concatenate([
         extract_t0(action.get("left_arm", np.zeros(7))),
          extract_t0(action.get("left_hand", np.zeros(7))),
         extract_t0(action.get("right_arm", np.zeros(7))),
         
         extract_t0(action.get("right_hand", np.zeros(7))),
        
        
       
       
        
          
        
        
        
        
       
        
       
         
        
       
        
        
         
    ])
    
    # Wrist pose (14D)
    control_cmd["wrist_pose"] = np.concatenate([
        extract_t0(action.get("left_wrist_pos", np.zeros(3))),
        # extract_t0(action.get("left_wrist_abs_quat", np.array([1, 0, 0, 0]))),
        # extract_t0(action.get("right_wrist_pos", np.zeros(3))),
        # extract_t0(action.get("right_wrist_abs_quat", np.array([1, 0, 0, 0]))),
    ])
    
    # Base height
    base_height = extract_t0(action.get("base_height_command", np.array([0.74])))
    control_cmd["base_height_command"] = float(base_height[0] if isinstance(base_height, np.ndarray) else base_height)
    
    # Navigate command
    nav_cmd = extract_t0(action.get("navigate_command", np.zeros(3)))
    control_cmd["navigate_cmd"] = nav_cmd.tolist() if isinstance(nav_cmd, np.ndarray) else nav_cmd
    
    # Toggles (matching teleop format)
    control_cmd["toggle_policy_action"] = False
    control_cmd["toggle_data_collection"] = False
    control_cmd["toggle_data_abort"] = False
    
    # Timing
    control_cmd["timestamp"] = now
    control_cmd["target_time"] = now + (1.0 / freq)
    
    return control_cmd


class ActionBuffer:
    """Buffer to store and manage action horizon"""
    
    def __init__(self):
        self.actions = None
        self.horizon = 0
        self.current_idx = 0
    
    def set_actions(self, action_chunk: dict):
        """Store new action chunk"""
        self.actions = action_chunk
        # Get horizon from first action
        first_key = next(iter(action_chunk.keys()))
        self.horizon = action_chunk[first_key].shape[1]  # (B, T, D) -> T
        self.current_idx = 0
    
    def get_current_action(self) -> dict:
        """Get action at current timestep"""
        if self.actions is None:
            return None
        
        current_action = {}
        for key, val in self.actions.items():
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                # Extract current timestep: (B, T, D) -> (B, 1, D)
                current_action[key] = val[:, self.current_idx:self.current_idx+1, :]
            else:
                current_action[key] = val
        
        return current_action
    
    def advance(self):
        """Move to next timestep"""
        self.current_idx += 1
    
    def needs_update(self) -> bool:
        """Check if we need new actions from policy"""
        return self.actions is None or self.current_idx >= self.horizon


# Upper-body joint groups logged for both robot state and VLA target.
# Ordering matches policy_action_to_control_goal()'s target_upper_body_pose.
UPPER_BODY_LAYOUT = [
    ("left_arm", 7),
    ("left_hand", 7),
    ("right_arm", 7),
    ("right_hand", 7),
]


class InferenceRecorder:
    """
    Records one CSV row per control tick for offline debugging / plotting.

    Each row captures the robot's current upper-body state, the VLA target,
    the index within the current action chunk, and (only on ticks where a new
    chunk was fetched from the policy) the inference latency and horizon length.

    The `dt_since_last` column spikes on new-chunk ticks, exposing the extra
    time introduced when the policy is queried (the "t16 -> t17" gap).
    """

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.path = self._next_experiment_path(output_dir)

        # Build column layout once.
        self.state_cols = []
        self.target_cols = []
        for group, dim in UPPER_BODY_LAYOUT:
            self.state_cols += [f"state_{group}_{i}" for i in range(dim)]
            self.target_cols += [f"target_{group}_{i}" for i in range(dim)]
        self.state_dim = len(self.state_cols)   # 28
        self.target_dim = len(self.target_cols)  # 28

        self.header = [
            "iteration",
            "wall_time",
            "loop_time",
            "dt_since_last",
            "new_inference",
            "horizon",
            "inference_latency_s",
            "action_idx",
        ] + self.state_cols + self.target_cols

        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)
        self._file.flush()

        self._prev_loop_time = None
        self.rows_written = 0
        print(f"📝 Recording inference data to: {self.path}")

    @staticmethod
    def _next_experiment_path(output_dir: str) -> str:
        """Return output_dir/expN.csv with the smallest unused N (>=1)."""
        n = 1
        while os.path.exists(os.path.join(output_dir, f"exp{n}.csv")):
            n += 1
        return os.path.join(output_dir, f"exp{n}.csv")

    def _fmt_vec(self, vec, dim):
        """Coerce a value to a flat python list of length `dim` (zeros on failure)."""
        if isinstance(vec, np.ndarray):
            flat = vec.reshape(-1).tolist()
        elif vec is None:
            flat = []
        else:
            flat = list(vec)
        if len(flat) < dim:
            flat = flat + [0.0] * (dim - len(flat))
        return flat[:dim]

    def record(self, iteration, wall_time, loop_time, action_idx,
               new_inference, horizon, inference_latency,
               state_vec, target_vec):
        dt = 0.0 if self._prev_loop_time is None else loop_time - self._prev_loop_time
        self._prev_loop_time = loop_time

        row = [
            iteration,
            f"{wall_time:.6f}",
            f"{loop_time:.6f}",
            f"{dt:.6f}",
            1 if new_inference else 0,
            horizon if new_inference else "",
            f"{inference_latency:.6f}" if (new_inference and inference_latency is not None) else "",
            action_idx,
        ]
        row += self._fmt_vec(state_vec, self.state_dim)
        row += self._fmt_vec(target_vec, self.target_dim)

        self._writer.writerow(row)
        self._file.flush()  # robust to Ctrl+C / crashes
        self.rows_written += 1

    def close(self):
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            print(f"📝 Saved {self.rows_written} rows to: {self.path}")


def main(config: GR00TPolicyConfig):
    """Main GR00T policy loop"""
    
    print("=" * 80)
    print("GR00T N1.6 G1 POLICY CONTROL LOOP")
    print("=" * 80)
    print(f"Policy server: {config.policy_host}:{config.policy_port}")
    print(f"Camera server: {config.camera_host}:{config.camera_port}")
    print(f"Control frequency: {config.frequency} Hz")
    print(f"Language: '{config.lang_instruction}'")
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
    ros_manager = ROSManager(node_name="GR00TPolicy")
    node = ros_manager.node
    
    # Create control publisher (same as teleop)
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    
    # Wait for observations
    print("[3] Waiting for first observations...")
    if not obs_reader.wait_until_ready(timeout=10.0):
        print("❌ Failed to get observations!")
        obs_reader.shutdown()
        return
    print("✅ Observations ready!")
    
    # Connect to policy server
    print(f"\n[3] Connecting to policy server...")
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
    
    # Create adapter
    print("\n[5] Initializing adapter...")
    adapter = GR00TG1Adapter(policy_client, add_stereo_camera=config.add_stereo_camera)
    
    # Action buffer for horizon management
    action_buffer = ActionBuffer()

    # Inference data recorder (for debugging / plotting)
    recorder = InferenceRecorder(config.record_dir) if config.record else None
    
    # Create rate controller
    rate = node.create_rate(config.frequency)
    telemetry = Telemetry(window_size=100)
    
    iteration = 0
    
    print("\n[6] Starting control loop...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                
                # Get observation
                obs = obs_reader.get_observation(config.lang_instruction)
                


                
                if obs is None:
                    print("⚠️  No observation available")
                    rate.sleep()
                    continue
                
                # Get new actions if needed
                new_inference = False
                inference_latency = None
                if action_buffer.needs_update():
                    new_inference = True
                    with telemetry.timer("get_action_chunk"):
                        t_inf = time.monotonic()
                        action_chunk, info = adapter.get_action(obs)
                        inference_latency = time.monotonic() - t_inf
                        action_buffer.set_actions(action_chunk)

                        if iteration == 0:
                            print(f"\n✅ Received action horizon: {action_buffer.horizon} timesteps")
                            print(f"   Action keys: {list(action_chunk.keys())}\n")

                # Index within the current chunk being replayed this tick
                action_idx = action_buffer.current_idx

                # Get current action from buffer
                current_action = action_buffer.get_current_action()
                
                # Convert to control goal format
                with telemetry.timer("convert_to_control_goal"):
                    t_now = time.monotonic()
                    control_cmd = policy_action_to_control_goal(
                        current_action, 
                        t_now, 
                        config.frequency
                    )
                
                # Publish control commandt
                #time.sleep(0.1)  # slight delay to ensure proper timing

                with telemetry.timer("publish_control"):
                    control_publisher.publish(control_cmd)

                # Record inference data (state vs target, timing, horizon)
                if recorder is not None:
                    state_vec = np.concatenate([
                        np.asarray(obs.get(f"{group}.pos", np.zeros(dim))).reshape(-1)
                        for group, dim in UPPER_BODY_LAYOUT
                    ])
                    recorder.record(
                        iteration=iteration,
                        wall_time=time.time(),
                        loop_time=t_now,
                        action_idx=action_idx,
                        new_inference=new_inference,
                        horizon=action_buffer.horizon,
                        inference_latency=inference_latency,
                        state_vec=state_vec,
                        target_vec=control_cmd.get("target_upper_body_pose"),
                    )

                # Advance to next action
                action_buffer.advance()
                
                # Log periodically
                if iteration % config.frequency == 0:  # Every 1 second
                    print(f"[Loop {iteration}] "
                          f"Time: {time.monotonic() - t_start:.3f}s | "
                          f"Action idx: {action_buffer.current_idx}/{action_buffer.horizon}")
                
                iteration += 1
                
            # Check timing
            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.frequency):
                telemetry.log_timing_info(context="GR00T Policy Loop Missed", threshold=0.001)
            
            rate.sleep()
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    
    finally:
        print("\nCleaning up...")
        if recorder is not None:
            recorder.close()
        obs_reader.shutdown()
        ros_manager.shutdown()
        print("✅ Shutdown complete!")


if __name__ == "__main__":
    config = tyro.cli(GR00TPolicyConfig)
    main(config)
