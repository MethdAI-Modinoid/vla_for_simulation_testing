"""
Modular ROS Observation Reader for GR00T Policy
Reads joint states and images from ROS topics and formats them for policy input.
"""

import threading
import time
from typing import Dict, Any, Optional
import numpy as np
import rclpy

from gr00t_wbc.control.sensor.vla_image_client import VLA_DATA_TOPIC_MAP, VLAImageClient
from gr00t_wbc.control.utils.ros_utils import ROSMsgSubscriber


class ROSObservationReader:
    """
    Reads robot state and images from ROS topics and provides them
    in the format expected by GR00T policy.
    """
    
    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        state_topic_name: str,
        frequency: int = 20,
        add_stereo_camera: bool = True,
        auto_start: bool = True,
    ):
        """
        Args:
            camera_host: IP address of camera server
            camera_port: Port of camera server
            state_topic_name: ROS topic name for robot state
            frequency: Polling frequency in Hz
            add_stereo_camera: Whether to include stereo cameras
            auto_start: If True, automatically start the reading thread
        """
        self.frequency = frequency
        self.add_stereo_camera = add_stereo_camera
        
        # Initialize ROS
        rclpy.init(args=None)
        self.node = rclpy.create_node("observation_reader")
        
        # Start ROS spinning in a separate thread
        self._ros_thread = threading.Thread(
            target=rclpy.spin, 
            args=(self.node,), 
            daemon=True
        )
        self._ros_thread.start()
        time.sleep(0.5)  # Give ROS time to initialize
        
        # Initialize subscribers
        self._state_subscriber = ROSMsgSubscriber(state_topic_name)

        # Use the SAME image client as the dataset recorder (run_g1_data_exporter.py)
        # so inference sees frames identical to what the VLA policy was trained on.
        # VLAImageClient runs its own background ZMQ receive thread that keeps only the
        # newest JPEG per camera and decodes lazily in read(), so every read() returns
        # the freshest available frame (no stale backlog, no separate ROS spin needed).
        if self.add_stereo_camera:
            topic_map = VLA_DATA_TOPIC_MAP  # ego_view + ego_left + ego_right
        else:
            # Only the main ego view. read() blocks (returns None) until every topic in
            # the map has a frame, so we must drop the wrist cameras we don't need here.
            topic_map = {"stereo/right": "ego_view"}
        self._image_subscriber = VLAImageClient(
            connect=f"tcp://{camera_host}:{camera_port}",
            topic_map=topic_map,
        )
        
        # Create rate limiter
        self.rate = self.node.create_rate(self.frequency)
        
        # Latest messages
        self._latest_image_msg = None
        self._latest_state_msg = None
        self._lock = threading.Lock()
        
        # Thread control
        self._running = False
        self._read_thread = None
        
        if auto_start:
            self.start()
    
    def start(self):
        """Start the observation reading thread."""
        if self._running:
            print("Reader already running!")
            return
        
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        print(f"✓ Observation reader started at {self.frequency} Hz")
    
    def stop(self):
        """Stop the observation reading thread."""
        self._running = False
        if self._read_thread is not None:
            self._read_thread.join(timeout=2.0)
        print("✓ Observation reader stopped")
    
    def _read_loop(self):
        """Internal loop that continuously reads from ROS topics."""
        try:
            while rclpy.ok() and self._running:
                # Read joint state
                state_msg = self._state_subscriber.get_msg()
                if state_msg is not None:
                    with self._lock:
                        self._latest_state_msg = state_msg
                
                # Read images
                image_msg = self._image_subscriber.read()
                if image_msg is not None:
                    with self._lock:
                        self._latest_image_msg = image_msg
                
                # Sleep at specified rate
                self.rate.sleep()
                
        except Exception as e:
            print(f"Error in read loop: {e}")
            import traceback
            traceback.print_exc()
    
    def get_latest_messages(self) -> tuple:
        """
        Get the latest state and image messages.
        
        Returns:
            (state_msg, image_msg) tuple or (None, None) if not available
        """
        with self._lock:
            return self._latest_state_msg, self._latest_image_msg
    
    def get_observation(self, lang_instruction: str = "") -> Optional[Dict[str, Any]]:
        """
        Get observation dict formatted for GR00T policy.
        
        Args:
            lang_instruction: Language instruction for the task
            
        Returns:
            Observation dict matching the format in full_body_dummy_inference.py
            or None if messages are not yet available
        """
        state_msg, image_msg = self.get_latest_messages()
        
        if state_msg is None or image_msg is None:
            return None
        
        obs = {}
        
        # ========== CAMERAS ==========
        images = image_msg["images"]
        
        # VLAImageClient publishes dataset image keys: ego_view (main ego view),
        # ego_left (left wrist), ego_right (right wrist). Map them onto the obs keys
        # the GR00T policy consumes.
        if "ego_view" in images:
            obs["ego_view"] = images["ego_view"]

        if self.add_stereo_camera:
            # Left stereo / left-wrist camera
            if "ego_left" in images:
                obs["ego_view_left_mono"] = images["ego_left"]

            # Right stereo / right-wrist camera
            if "ego_right" in images:
                obs["ego_view_right_mono"] = images["ego_right"]
        
        # ========== JOINT POSITIONS ==========
        # Extract joint positions from state message
        # Assuming state_msg["q"] contains all 43 joints in order:
        # left_leg (6), right_leg (6), waist (3), left_arm (7), left_hand (7), 
        # right_arm (7), right_hand (7)
        
        q = state_msg["q"]
        
        # Split joints into groups based on indices
        obs["left_leg.pos"] = q[0:6].astype(np.float32)
        obs["right_leg.pos"] = q[6:12].astype(np.float32)
        obs["waist.pos"] = q[12:15].astype(np.float32)
        obs["left_arm.pos"] = q[15:22].astype(np.float32)
        obs["left_hand.pos"] = q[22:29].astype(np.float32)
        obs["right_arm.pos"] = q[29:36].astype(np.float32)
        obs["right_hand.pos"] = q[36:43].astype(np.float32)
        
        # ========== WRIST POSES ==========
        # Extract wrist poses from state message
        # wrist_pose format: [left_pos(3), left_quat(4), right_pos(3), right_quat(4)]
        wrist_pose = state_msg["wrist_pose"]
        
        obs["left_wrist_pos"] = wrist_pose[0:3].astype(np.float32)
        obs["left_wrist_abs_quat"] = wrist_pose[3:7].astype(np.float32)
        obs["right_wrist_pos"] = wrist_pose[7:10].astype(np.float32)
        obs["right_wrist_abs_quat"] = wrist_pose[10:14].astype(np.float32)
        
        # ========== NAVIGATION COMMANDS ==========
        obs["base_height_command"] = np.array(
            [0.74], 
            dtype=np.float32
        )
        obs["navigate_command"] = np.array(
            [0.0, 0.0, 0.0], 
            dtype=np.float32
        )
        
        # ========== LANGUAGE INSTRUCTION ==========
        obs["lang"] = lang_instruction
        
        return obs
    
    def is_ready(self) -> bool:
        """Check if observations are available."""
        state_msg, image_msg = self.get_latest_messages()
        return state_msg is not None and image_msg is not None
    
    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait until observations are available.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if ready, False if timeout
        """
        start_time = time.time()
        while not self.is_ready():
            if time.time() - start_time > timeout:
                print(f"⚠ Timeout waiting for observations")
                return False
            time.sleep(0.1)
        return True
    
    def shutdown(self):
        """Cleanup and shutdown."""
        self.stop()
        self._image_subscriber.close()  # stop the ZMQ receive thread + close socket
        self.node.destroy_node()
        rclpy.shutdown()
        print("✓ Reader shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Convenience function for quick usage
def create_observation_reader(
    camera_host: str = "192.168.123.164",
    camera_port: int = 5555,
    state_topic_name: str = "/robot_state",
    frequency: int = 20,
    add_stereo_camera: bool = True,
) -> ROSObservationReader:
    """
    Create and return an observation reader.
    
    Example:
        reader = create_observation_reader()
        reader.wait_until_ready()
        obs = reader.get_observation("pick up the cube")
    """
    return ROSObservationReader(
        camera_host=camera_host,
        camera_port=camera_port,
        state_topic_name=state_topic_name,
        frequency=frequency,
        add_stereo_camera=add_stereo_camera,
    )


# Example usage
if __name__ == "__main__":
    from gr00t_wbc.control.main.constants import STATE_TOPIC_NAME
    
    print("=" * 80)
    print("ROS OBSERVATION READER TEST")
    print("=" * 80)
    
    # Create reader
    reader = create_observation_reader(
        camera_host="192.168.123.164",
        camera_port=5555,
        state_topic_name=STATE_TOPIC_NAME,
        frequency=20,
        add_stereo_camera=True,
    )
    
    try:
        # Wait for data
        print("\nWaiting for observations...")
        if reader.wait_until_ready(timeout=10.0):
            print("✓ Observations ready!\n")
            
            # Get observation
            obs = reader.get_observation("pick up the red cube")
            
            if obs is not None:
                print("Observation keys:", list(obs.keys()))
                print("\nCamera shapes:")
                for key in ["ego_view", "ego_view_left_mono", "ego_view_right_mono"]:
                    if key in obs:
                        print(f"  {key}: {obs[key].shape}")
                
                print("\nJoint group shapes:")
                for group in ["left_leg", "right_leg", "waist", "left_arm", 
                             "left_hand", "right_arm", "right_hand"]:
                    key = f"{group}.pos"
                    if key in obs:
                        print(f"  {key}: {obs[key].shape}")
                
                print("\nWrist poses:")
                for key in ["left_wrist_pos", "left_wrist_abs_quat", 
                           "right_wrist_pos", "right_wrist_abs_quat"]:
                    if key in obs:
                        print(f"  {key}: {obs[key]}")
                
                print(f"\nLanguage: {obs['lang']}")
                print("\n✓ Observation successfully retrieved!")
            else:
                print("✗ Failed to get observation")
        else:
            print("✗ Timeout waiting for observations")
        
        # Keep reading for a few seconds
        print("\nReading observations for 5 seconds...")
        for i in range(5):
            time.sleep(1)
            obs = reader.get_observation(f"test instruction {i}")
            if obs:
                print(f"  [{i+1}/5] Got observation with {len(obs)} keys")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        reader.shutdown()
        print("\n" + "=" * 80)