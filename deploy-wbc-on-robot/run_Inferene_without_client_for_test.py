#!/usr/bin/env python3
"""
GR00T Policy Loop - Integrated with WBC Control System

Sends GR00T policy actions to the robot using the same control flow as teleop.
"""

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
    camera_host: str = "192.168.123.164"
    camera_port: int = 5555
    
    # Control frequency
    frequency: int = 20  # Hz   fix 10
    
    # Language instruction
    lang_instruction: str = "place the penguin in the beige basket ,the cube in the orange basket, and the blue octopus in the brown basket"
    
    # Camera configuration
    add_stereo_camera: bool = True
    
    # Robot configuration
    robot: str = "g1"


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


def main(config: GR00TPolicyConfig):
    """Main GR00T policy loop"""
    
    print("=" * 80)
    print("GR00T N1.7 G1 POLICY CONTROL LOOP")
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
                if action_buffer.needs_update():
                    with telemetry.timer("get_action_chunk"):
                        action_chunk, info = adapter.get_action(obs)
                        action_buffer.set_actions(action_chunk)
                        
                        if iteration == 0:
                            print(f"\n✅ Received action horizon: {action_buffer.horizon} timesteps")
                            print(f"   Action keys: {list(action_chunk.keys())}\n")
                
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
        obs_reader.shutdown()
        ros_manager.shutdown()
        print("✅ Shutdown complete!")


if __name__ == "__main__":
    config = tyro.cli(GR00TPolicyConfig)
    main(config)