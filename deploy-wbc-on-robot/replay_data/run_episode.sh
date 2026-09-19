#!/bin/bash
# ===========================================================================
# Replay recorded episodes on the REAL G1.  ⚠ THIS MOVES THE ROBOT.
# Run INSIDE the deploy container:  docker exec -it gr00t_wbc-bash-root bash
#   bash /root/Projects/deploy-wbc-on-robot/replay_data/run_episode.sh [smooth|raw|all|<parquet>]
#
# Default = smooth (v5 episode 0). Starts the WBC (if not already running), waits
# for init, then feeds the recorded actions. Robot ramps to start pose (~2 s) then
# replays. Ctrl+C stops and shuts the WBC down.
#
#   all  -> replays every episode of $DATASET (see below) in order, reusing one
#           WBC. Confirms GO once, then waits for Enter before each next episode.
#           Single-episode modes still loop forever until Ctrl+C, as before.
# ===========================================================================
# NOTE: no `set -u` here — sourcing ROS's setup.bash references unset vars and would
# make a `set -u` shell exit silently mid-source.
WHICH="${1:-smooth}"
DEP=/root/Projects/deploy-wbc-on-robot
RD="$DEP/replay_data"
PY=/root/venv/bin/python            # absolute — no dependence on PATH/venv activation

# ---------------------------------------------------------------------------
# Dataset replayed by `run_episode.sh all`. Container path (inside the deploy
# container), not the host path. Override without editing: DATASET=... bash ...
# ---------------------------------------------------------------------------
DATASET="${DATASET:-$DEP/outputs/2026-09-19-10-39-50-G1-sim}"

# Collect the episode list up front so we can fail before touching the robot.
EPISODES=()
if [ "$WHICH" = "all" ]; then
  [ -d "$DATASET" ] || { echo "ERROR: dataset not found: $DATASET"; exit 1; }
  while IFS= read -r f; do EPISODES+=("$f"); done \
    < <(find "$DATASET/data" -name 'episode_*.parquet' | sort)
  [ ${#EPISODES[@]} -gt 0 ] || { echo "ERROR: no episode parquets under $DATASET/data"; exit 1; }
else
  case "$WHICH" in
    smooth) PARQUET="$RD/smooth_v2_ep0.parquet" ;;
    raw)    PARQUET="$RD/raw_ep0.parquet" ;;
    *)      PARQUET="$WHICH" ;;
  esac
  [ -f "$PARQUET" ] || { echo "ERROR: parquet not found: $PARQUET"; exit 1; }
  EPISODES=("$PARQUET")
fi

# environment (venv + ROS for rclpy) — safe to re-source
source /root/venv/bin/activate 2>/dev/null || true
source /opt/ros/humble/setup.bash 2>/dev/null || true
cd "$DEP"

echo "=================================================================="
if [ "$WHICH" = "all" ]; then
  echo " Replaying on the REAL robot:  ALL ${#EPISODES[@]} episodes"
  echo " Dataset: $DATASET"
else
  echo " Replaying on the REAL robot:  ${EPISODES[0]}"
fi
echo " The arms/hands WILL move. Clear workspace, e-stop in hand."
echo "=================================================================="
read -r -p " Type GO to proceed: " c
[ "$c" = "GO" ] || { echo "aborted."; exit 1; }

if pgrep -f "run_g1_control_loop.*enp5s0" >/dev/null 2>&1; then
  echo "[wbc] existing WBC on enp5s0 detected — using it."
else
  echo "[wbc] starting WBC controller..."
  "$PY" gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
    --wbc_version gear_wbc \
    --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
    --wbc_policy_class GIDecoupledWholeBodyPolicy \
    --interface enp5s0 --simulator None --control_frequency 50 \
    --no-enable_waist --with_hands --no-high_elbow_pose --no-enable_gravity_compensation \
    > /tmp/wbc_replay.log 2>&1 &
  WBC_PID=$!
  trap 'echo "[cleanup] stopping WBC"; kill $WBC_PID 2>/dev/null' EXIT INT TERM
  echo "[wbc] pid $WBC_PID, waiting 8s to initialize..."
  sleep 8
  if ! kill -0 $WBC_PID 2>/dev/null; then
    echo "[wbc] FAILED to start — last log lines:"; tail -20 /tmp/wbc_replay.log; exit 1
  fi
fi

if [ "$WHICH" = "all" ]; then
  # --no-lerobot_replay_loop makes each episode play once and exit, instead of
  # rewinding forever, so the loop below can advance to the next one.
  total=${#EPISODES[@]}
  for i in "${!EPISODES[@]}"; do
    ep="${EPISODES[$i]}"
    if [ "$i" -gt 0 ]; then
      echo
      read -r -p "[$((i + 1))/$total] next: $(basename "$ep") — press Enter to replay (Ctrl+C to stop): " _
    fi
    echo "[replay $((i + 1))/$total] feeding actions from $(basename "$ep") ..."
    "$PY" gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py \
      --lerobot_replay_path "$ep" --no-lerobot_replay_loop
  done
  echo "[replay] all $total episodes done."
else
  echo "[replay] feeding actions from $(basename "${EPISODES[0]}") ..."
  "$PY" gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py \
    --lerobot_replay_path "${EPISODES[0]}"
  echo "[replay] done."
fi
