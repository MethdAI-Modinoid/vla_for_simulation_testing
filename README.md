# vla_sim_ws — Sim & Control Stack Setup

Workspace root: `/mnt/drive2/vla_sim_ws`

Two stacks live here and are normally run together:

| Part | What it is | Where | How it's installed |
|---|---|---|---|
| **[Part A](#part-a--gear_sonic-mujoco-sim)** | `gear_sonic` MuJoCo simulator (physics, cameras, DDS bridge) | [gear_sonic/](gear_sonic/) | native venv `.venv_sim` |
| **[Part B](#part-b--gr00t_wbc-control-teleop--data-collection)** | `gr00t_wbc` whole-body control, teleop, data exporter | [deploy-wbc-on-robot/](deploy-wbc-on-robot/) | Docker container |

The sim provides the robot; the WBC stack drives it. They talk over DDS on
loopback — see [§10 Running sim + WBC together](#10-running-sim--wbc-together).

---

# Part A — `gear_sonic` MuJoCo Sim

How to install and run the G1 MuJoCo simulator (`run_sim_loop.py`).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Linux x86_64 (or arm64) | detected automatically by the installer |
| `curl` | used to fetch `uv` if it isn't installed |
| A GPU/display for onscreen rendering | the MuJoCo viewer needs a display; use `--no-enable-onscreen` for headless |

You do **not** need a pre-existing Python — the installer downloads a uv-managed
Python 3.10 (with dev headers).

---

## 2. One-time install

Run from the workspace root:

```bash
cd /mnt/drive2/vla_sim_ws
bash install_scripts/install_mujoco_sim.sh
```

What [install_mujoco_sim.sh](install_scripts/install_mujoco_sim.sh) does:

1. Installs [`uv`](https://astral.sh/uv) if missing.
2. Installs a uv-managed **Python 3.10**.
3. **Deletes any existing `.venv_sim`** and recreates it (prompt: `gear_sonic_sim`).
4. `uv pip install -e "gear_sonic[sim]"` — mujoco, tyro, pin, pyyaml, pyzmq,
   msgpack, msgpack-numpy, opencv-python (plus the base deps: numpy 1.26.4,
   scipy 1.15.3, torch, loguru, …).
5. `uv pip install -e external_dependencies/unitree_sdk2_python` — the DDS bridge
   the sim uses to talk to the WBC controller.

> The script is idempotent but destructive to `.venv_sim`. Re-run it only when you
> want a clean environment.

This venv covers the **simulator only**. Teleop (`pico_manager_thread_server.py`),
data collection, camera and inference each have their own extras in
[gear_sonic/pyproject.toml](gear_sonic/pyproject.toml) (`[teleop]`,
`[data_collection]`, `[camera]`, `[inference]`) and are meant to live in separate
venvs — only the sim installer ships in this workspace.

---

## 3. Activate

```bash
cd /mnt/drive2/vla_sim_ws
source .venv_sim/bin/activate
```

Your prompt should now show `(gear_sonic_sim)`.

Verify:

```bash
python --version                              # Python 3.10.20
python -c "import mujoco; print(mujoco.__version__)"   # 3.11.0
```

---

## 4. Run the sim

The package is installed editable, and the scene path is resolved from the module
location — so you can launch from **any** directory.

**Viewer only (simplest first run):**

```bash
python gear_sonic/scripts/run_sim_loop.py
```

**With offscreen cameras + ZMQ image publishing** (what the VLA / data-collection
pipeline expects):

```bash
python gear_sonic/scripts/run_sim_loop.py \
    --enable-offscreen \
    --enable-image-publish \
    --camera-port 5555 \
    --env-name default
```

`--enable-image-publish` **requires** `--enable-offscreen` (asserted in
[run_sim_loop.py:41-44](gear_sonic/scripts/run_sim_loop.py#L41-L44)).

Three cameras are published when image publishing is on, at 640×480:
`ego_view`, `ego_left`, `ego_right`
([base_sim.py:59-63](gear_sonic/utils/mujoco_sim/base_sim.py#L59-L63)).

**Headless:**

```bash
python gear_sonic/scripts/run_sim_loop.py \
    --no-enable-onscreen --enable-offscreen --enable-image-publish
```

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--env-name` | `default` | Only `default` is registered today ([base_sim.py:685](gear_sonic/utils/mujoco_sim/base_sim.py#L685)) |
| `--enable-offscreen` | off | Offscreen renderers for the camera feeds |
| `--enable-image-publish` | off | Publish frames over ZMQ (needs offscreen) |
| `--camera-port` | `5555` | ZMQ port for published images |
| `--enable-onscreen` | **on** | MuJoCo interactive viewer |
| `--sim-frequency` | `200` | Physics rate (Hz); YAML `SIMULATE_DT` is `1/sim_frequency` |
| `--control-frequency` | `50` | Control loop rate (Hz) |
| `--with-hands` / `--no-with-hands` | on | Dexterous hands |
| `--enable-waist` / `--no-enable-waist` | on | Waist joints in IK |
| `--verbose` | off | Per-iteration logging |
| `--mp-start-method` | `spawn` | Multiprocessing start method for the publisher |

Full list: `python gear_sonic/scripts/run_sim_loop.py --help`.

---

## 5. Viewer controls

| Key | Action |
|---|---|
| `9` | Toggle the elastic band (release the robot from the hanger) |
| `7` / `8` | Shorten / lengthen the elastic band |
| `Backspace` | Reset the episode (re-randomizes the manipulation cubes) |
| `↑ ↓ ← →` | Apply a perturbation force to the robot |
| `v` | Toggle visualization state |

The elastic band is on at startup (`ENABLE_ELASTIC_BAND: True`) — press `9` to
drop the robot onto the floor.

---

## 6. What the sim connects to

The sim is **only physics + rendering + a DDS bridge**. It does not load the WBC
ONNX policy itself; a separate controller process drives it — that's
[Part B](#part-b--gr00t_wbc-control-teleop--data-collection).

```
run_sim_loop.py ──DDS (CycloneDDS, domain 0, iface "lo")──▶ WBC / teleop controller
       │
       └──ZMQ tcp://*:5555 (images) ──▶ camera viewer / data exporter / VLA client
```

- **DDS**: hardcoded to domain `0` on loopback `lo`
  ([simulator_factory.py:18-21](gear_sonic/utils/mujoco_sim/simulator_factory.py#L18-L21)) —
  the `--interface` flag and the YAML `INTERFACE`/`DOMAIN_ID` keys do not override it.
  Any DDS peer must therefore be on the same machine.
- **ZMQ**: images on `--camera-port` (default 5555). Inference defaults to 5558 to
  avoid the clash.

Check the feeds with:

```bash
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
```

(`R` records to MP4, `Q` quits. Needs a venv with the `[data_collection]` or
`[camera]` extra — `opencv-python` and `pyzmq` are already in `[sim]`, so `.venv_sim`
works for viewing.)

---

## 7. Configuration

Defaults come from a YAML that is then overridden by the CLI dataclass:

- YAML: [gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml](gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml)
- CLI dataclass: `SimLoopConfig` in [gear_sonic/utils/mujoco_sim/configs.py](gear_sonic/utils/mujoco_sim/configs.py)
- Merge order: YAML loaded → `override_wbc_config()` stamps CLI values on top.

Key YAML entries:

| Key | Value | Meaning |
|---|---|---|
| `ROBOT_SCENE` | `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml` | Scene loaded by the sim |
| `SIMULATE_DT` | `0.005` | Physics step (overridden by `--sim-frequency`) |
| `VIEWER_DT` | `0.02` | Viewer refresh |
| `ENABLE_ELASTIC_BAND` | `True` | Virtual hanger |
| `DOMAIN_ID` / `INTERFACE` | `0` / `lo` | DDS settings |

The scene includes the pick-and-place table, 3 baskets and 3 cubes — see
[README_manipulation_scene.md](gear_sonic/README_manipulation_scene.md) for the
reset schedule and tuning knobs.

---

## 8. Workspace layout

```
vla_sim_ws/
├── install_scripts/install_mujoco_sim.sh   # the installer
├── .venv_sim/                              # created by the installer
├── gear_sonic/                             # the package (editable install)
│   ├── scripts/run_sim_loop.py             # sim entry point
│   ├── utils/mujoco_sim/                   # simulator, bridge, configs
│   │   ├── base_sim.py                     # DefaultEnv + BaseSimulator
│   │   ├── simulator_factory.py            # DDS init + process launch
│   │   ├── configs.py                      # BaseConfig / SimLoopConfig
│   │   └── wbc_configs/*.yaml
│   ├── data/robot_model/model_data/g1/     # URDF / MJCF scenes + meshes
│   └── pyproject.toml                      # extras: sim, teleop, camera, …
├── external_dependencies/
│   ├── unitree_sdk2_python/                # DDS bridge (installed editable)
│   └── XRoboToolkit-PC-Service-Pybind_.../ # teleop SDK (not needed for sim)
└── deploy-wbc-on-robot/                    # on-robot WBC deployment (separate)
```

---

## 9. Troubleshooting

**`uv: command not found` after the installer ran**
Add uv to PATH and re-run: `export PATH="$HOME/.local/bin:$PATH"`.

**`ModuleNotFoundError: gear_sonic`**
The venv isn't active. `source .venv_sim/bin/activate` from the workspace root.

**Viewer fails to open / GLFW or EGL errors**
No display available. Run headless with `--no-enable-onscreen --enable-offscreen`.

**`AssertionError: enable_offscreen must be True when enable_image_publish is True`**
Add `--enable-offscreen`.

**"No camera configs provided, image publishing subprocess will not be started"**
Offscreen rendering is off, so no cameras were configured. Add `--enable-offscreen`.

**Nothing on port 5555**
The publisher starts ~1 s after launch and only with `--enable-image-publish`.
Confirm with `ss -ltnp | grep 5555`.

**Robot hangs in mid-air**
That's the elastic band. Press `9` in the viewer.

**Address/port already in use**
A previous sim is still running: `pkill -f run_sim_loop.py`, or pick another
`--camera-port`.

---
---

# Part B — `gr00t_wbc` Control, Teleop & Data Collection

Software stack for loco-manipulation experiments across multiple humanoid
platforms, with primary support for the Unitree G1. Provides whole-body control
policies, a teleoperation stack, and a data exporter.

Lives in [deploy-wbc-on-robot/](deploy-wbc-on-robot/) and runs **inside Docker** —
it does not use `.venv_sim`.

---

## B1. System installation

### Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with a recent driver
- Docker and NVIDIA Container Toolkit (required for GPU access inside the container)

### Repository setup

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
git clone <repo-url>
cd gr00t_wbc
```

> In this workspace the repo is already checked out at
> [deploy-wbc-on-robot/](deploy-wbc-on-robot/) — skip the clone and use that path.

### Docker environment

A Docker image with all dependencies pre-installed is provided.

Install a fresh image and start a container:

```bash
cd deploy-wbc-on-robot
./docker/run_docker.sh --install --root
```

This pulls the latest `gr00t_wbc` image from `docker.io/nvgear`
([run_docker.sh:44,315](deploy-wbc-on-robot/docker/run_docker.sh#L44)).

Start or re-enter a container:

```bash
./docker/run_docker.sh --root
```

Use `--root` to run as the `root` user. To run as a normal user, build the image
locally:

```bash
./docker/run_docker.sh --build
```

The container runs with `--network=host` and `--ipc=host`, and mounts the project
at `/root/Projects/deploy-wbc-on-robot`
([run_docker.sh:401-418](deploy-wbc-on-robot/docker/run_docker.sh#L401-L418)) —
that shared network namespace is what lets the containerized WBC reach the
host-side sim over DDS on `lo`.

---

## B2. Running the control stack

Once inside the container, the control policies can be launched directly.

**In simulation** (`--interface lo`, `--simulator None` — it attaches to an
already-running sim rather than starting its own):

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

**On the real robot** — configure the host network per the
[G1 SDK Development Guide](https://support.unitree.com/home/en/G1_developer) and
set a static IP at `192.168.123.222`, subnet mask `255.255.255.0`. Then swap the
interface for your NIC:

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

| Key | Action |
|---|---|
| `]` | Activate policy |
| `o` | Deactivate policy |
| `9` | Release / hold the robot |
| `Backspace` (viewer) | Reset the robot in the visualizer |

---

## B3. Running the teleoperation stack

The teleop policy primarily uses **Pico** controllers for coordinated hand and
body control. LeapMotion and HTC Vive with Nintendo Switch Joy-Cons are also
supported.

Keep `run_g1_control_loop.py` running, and in another terminal:

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

> `--body_streamer_ip` is the Pico headset's IP — change it to match yours.

### Pico setup and controls

Configure the teleop app on your Pico headset per the
[XR Robotics guidelines](https://github.com/XR-Robotics). The necessary PC
software is pre-installed in the Docker container; only the
[XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)
component is needed.

**Prerequisite:** connect the Pico to the same network as the host computer.

Controller bindings:

| Input | Action |
|---|---|
| `menu` + left trigger | Toggle lower-body policy |
| `menu` + right trigger | Toggle upper-body policy |
| Left stick | X/Y translation |
| Right stick | Yaw rotation |
| L/R triggers | Control hand grippers |

Pico unit test:

```bash
python gr00t_wbc/control/teleop/streamers/pico_streamer.py
```

---

## B4. Running the data collection stack

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

Operations on the Pico controllers:

| Button | Action |
|---|---|
| `A` | Start / stop recording |
| `B` | Discard trajectory |

> `--camera_host 192.168.123.164` is the **robot's** camera server. When running
> against the MuJoCo sim instead, point it at the sim's ZMQ publisher:
> `--camera_host localhost --camera_port 5555` (see [§4](#4-run-the-sim)).

---

## 10. Running sim + WBC together

The typical simulation session is three terminals:

```
Terminal 1 (host)       Terminal 2 (container)        Terminal 3 (container)
.venv_sim               ./docker/run_docker.sh --root  (same container)
run_sim_loop.py   ◀──DDS lo, domain 0──▶  run_g1_control_loop.py  ◀──▶  run_teleop_policy_loop.py
       │                --interface lo                        (Pico)
       └── ZMQ :5555 images ──▶ run_g1_data_exporter.py / camera viewer
```

**Terminal 1 — start the sim on the host:**

```bash
cd /mnt/drive2/vla_sim_ws
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py --enable-offscreen --enable-image-publish --camera-port 5555
```

**Terminal 2 — start the WBC in the container:**

```bash
cd /mnt/drive2/vla_sim_ws/deploy-wbc-on-robot
./docker/run_docker.sh --root
# then the "in simulation" command from B2
```

Press `]` in the WBC terminal to activate the policy, and `9` in the MuJoCo viewer
to release the robot from the elastic band.

**Terminal 3 — teleop or data export**, as in B3 / B4.

### Why this works

Both sides use CycloneDDS on **domain 0, interface `lo`**. The sim hardcodes that
([simulator_factory.py:18-21](gear_sonic/utils/mujoco_sim/simulator_factory.py#L18-L21)),
the WBC gets it from `--interface lo`, and the container's `--network=host` puts
them in the same loopback namespace. Consequences worth knowing:

- Sim and WBC **must be on the same machine**. A remote WBC will not see the sim.
- Only **one** sim may run at a time — two publishers on domain 0 will fight.
- Don't launch a second WBC loop against the same sim.

### Which stack owns what

| Concern | Owner |
|---|---|
| Physics, scene, cubes, viewer | `gear_sonic` sim (Part A) |
| Camera rendering + ZMQ image publishing | `gear_sonic` sim (Part A) |
| WBC ONNX policies, balance/walk | `gr00t_wbc` (Part B) |
| Pico / teleop devices | `gr00t_wbc` (Part B) |
| LeRobot dataset export | `gr00t_wbc` data exporter (Part B) |

Note that `gear_sonic` also ships its own `[teleop]`, `[data_collection]` and
`[inference]` extras and scripts (`pico_manager_thread_server.py`,
`run_data_exporter.py`) — a parallel, venv-based path to the same capabilities.
The installer for those venvs is **not** present in this workspace, so the Docker
route in Part B is the working one here.
