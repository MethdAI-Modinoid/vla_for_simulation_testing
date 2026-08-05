# gr00t_wbc

Software stack for loco-manipulation experiments across multiple humanoid platforms, with primary support for the Unitree G1. This repository provides whole-body control policies, a teleoperation stack, and a data exporter. 

---

## System Installation

### Prerequisites
- Ubuntu 22.04
- NVIDIA GPU with a recent driver
- Docker and NVIDIA Container Toolkit (required for GPU access inside the container)

### Repository Setup
Install Git and Git LFS:
```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

Clone the repository:
```bash
mkdir -p ~/Projects
cd ~/Projects
git clone
cd gr00t_wbc
```

### Docker Environment
We provide a Docker image with all dependencies pre-installed.

Install a fresh image and start a container:

cd deploy-wbc-on-robot

```bash
./docker/run_docker.sh --install --root
```
This pulls the latest `gr00t_wbc` image from `docker.io/nvgear`.

Start or re-enter a container:
```bash
./docker/run_docker.sh --root
```

Use `--root` to run as the `root` user. To run as a normal user, build the image locally:
```bash
./docker/run_docker.sh --build
```
---

## Running the Control Stack

Once inside the container, the control policies can be launched directly.

- Run wbc in Simulation:
  ```bash
  export GR00T_WBC_TMUX_SESSION=g1_deployment && \
/root/venv/bin/python /root/Projects/deploy-wbc-on-robot/gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
  --wbc_version gear_wbc \
  --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
  --wbc_policy_class GIDecoupledWholeBodyPolicy \
  --interface lo \
  --simulator None \
  --control_frequency 50 \
  --no-enable_waist \
  --with_hands \
  --no-high_elbow_pose \
  --no-enable_gravity_compensation

  ```
- Real robot: Ensure the host machine network is configured per the [G1 SDK Development Guide](https://support.unitree.com/home/en/G1_developer) and set a static IP at `192.168.123.222`, subnet mask `255.255.255.0`:
  ```bash
  export GR00T_WBC_TMUX_SESSION=g1_deployment && \
/root/venv/bin/python /root/Projects/deploy-wbc-on-robot/gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
  --wbc_version gear_wbc \
  --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
  --wbc_policy_class GIDecoupledWholeBodyPolicy \
  --interface enp5s0 \
  --simulator None \
  --control_frequency 50 \
  --no-enable_waist \
  --with_hands \
  --no-high_elbow_pose \
  --no-enable_gravity_compensation

  ```

Keyboard shortcuts (terminal window):
- `]`: Activate policy
- `o`: Deactivate policy
- `9`: Release / Hold the robot
- `backspace` (viewer): Reset the robot in the visualizer

---

## Running the Teleoperation Stack

The teleoperation policy primarily uses Pico controllers for coordinated hand and body control. It also supports other teleoperation devices, including LeapMotion and HTC Vive with Nintendo Switch Joy-Con controllers.

Keep `run_g1_control_loop.py` running, and in another terminal run:

```bash
/root/venv/bin/python /root/Projects/deploy-wbc-on-robot/gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py \
  --body_control_device pico \
  --hand_control_device pico \
  --body_streamer_ip 10.112.210.229 \
  --body_streamer_keyword knee \
  --no-enable_waist \
  --no-high_elbow_pose \
  --no-enable_visualization \
  --enable_real_device \
  --upper-body-joint-speed 50 \
  --teleop-frequency 50 \
  --control_frequency 50 \
  --no-binary-hand-ik

```

### Pico Setup and Controls
Configure the teleop app on your Pico headset by following the [XR Robotics guidelines](https://github.com/XR-Robotics). 

The necessary PC software is pre-installed in the Docker container. Only the [XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) component is needed.

Prerequisites: Connect the Pico to the same network as the host computer.

Controller bindings:
- `menu + left trigger`: Toggle lower-body policy
- `menu + right trigger`: Toggle upper-body policy
- `Left stick`: X/Y translation
- `Right stick`: Yaw rotation
- `L/R triggers`: Control hand grippers

Pico unit test:
```bash
python gr00t_wbc/control/teleop/streamers/pico_streamer.py
```

---

## Running the Data Collection Stack


```bash
export GR00T_WBC_TMUX_SESSION=g1_data_export && \
/root/venv/bin/python /root/Projects/deploy-wbc-on-robot/gr00t_wbc/control/main/teleop/run_g1_data_exporter.py \
  --data_collection_frequency 20 \
  --root_output_dir outputs \
  --lower_body_policy gear_wbc \
  --wbc_model_path "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx" \
  --camera_host 192.168.123.164 \
  --camera_port 5555



```

Operations on Pico controllers:
- `A`: Start/Stop recording
- `B`: Discard trajectory
