# RTC Sim Inference Runbook — gear_sonic MuJoCo + GR00T N1.7 + RTC

Step-by-step launch for the cube-sorting sim with the `sim_sorting` checkpoints and
Real-Time Chunking, covering all three experiment arms:

| Arm | Server | Client | Status |
|---|---|---|---|
| A. Pre-RTC baseline | no `--rtc*` flags | reference loop (e=16) or `run_inference_rtc.py` | measured 2026-07-10 |
| B. Sync RTC | `--rtc` | `run_inference_rtc.py` (e=8, blocking) | measured 2026-07-18 (−63% boundary jump; jerk regressed from stop-and-go) |
| C. **Async RTC** | `--rtc` (same flags) | **`run_inference_rtc_async.py`** (background inference) | ← current round |

Companion docs: `RTC_implementation.md` (repo root — §5.1 baseline numbers, §14 sync
results + async plan) and `RTC_README.md` (concepts + knobs).

Paths below are from the original dev machine; adjust to your checkout. The server
needs the `nvidia/Cosmos-Reason2-2B` backbone reachable (HF access or a warm
`~/.cache/huggingface/hub` — any machine that already ran this checkpoint has it).

---

## Terminal layout (4 terminals)

| T | What | Environment |
|---|------|-------------|
| T1 | MuJoCo sim (camera on :5555) | conda `unitree_sim` + PYTHONPATH fix |
| T2 | WBC controller (docker) — sudo | as you always run it |
| T3 | GR00T policy server (RTC lives here, port **5560**) | Isaac-GR00T uv venv |
| T4 | Inference client + CSV logger | conda `isaaclab` (ROS on PYTHONPATH) |

Start order: T1 → T2 → T3 → T4.

### T1 — Simulator

The `unitree_sim` env's editable `unitree_sdk2py` install points at a deleted
workspace, so the SDK source is supplied via PYTHONPATH:

```bash
cd ~/Projects/r2d2
PYTHONPATH=$HOME/Projects/r2d2:$HOME/Projects/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python \
  ~/miniconda3/envs/unitree_sim/bin/python gear_sonic/scripts/run_sim_loop.py \
  --enable-offscreen --enable-image-publish --camera-port 5555 --env-name default
```

Keys in the sim: **Backspace** = episode reset (cubes re-arranged per the balanced
6-config schedule), **9** = release the elastic-band hanger.

### T2 — WBC controller (docker, needs sudo — launch it yourself)

```bash
cd ~/Projects/GR00T-WholeBodyControl
sudo ./start_wbc.sh
```

### T3 — Policy server (identical for arms B and C; drop --rtc* flags for arm A)

```bash
cd <Isaac-GR00T repo>
env -u PYTHONPATH uv run python gr00t/eval/run_gr00t_server.py \
  --model-path <ckpts>/sim_sorting/checkpoint-22000 \
  --embodiment-tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --port 5560 \
  --rtc --rtc-ramp-rate 2.0
```

- **`--port 5560` is mandatory** — the server default (5555) collides with the sim's
  camera publisher.
- `env -u PYTHONPATH` keeps ROS python packages out of the venv.
- `--rtc-frozen-steps` / `--rtc-execution-horizon` are NOT needed: both clients send
  per-call values that override the server defaults (the async client varies
  frozen_steps with measured latency — that's the point).

### T4 — Inference client + logger

**Arm C — async (current round):**

```bash
conda activate isaaclab
cd <dir with sdk.py + run_Inferene_without_client_for_test.py>   # e.g. r2d2/data_collection
python run_inference_rtc_async.py --camera-host localhost
```

The async client never blocks the control loop: it triggers the next inference when 8
steps remain in the current chunk (`--trigger-lead-steps 8`, keeping the RTC blend
overlap at 16 − 8 = 8, the validated band), keeps executing during the ~170 ms wait,
and swaps to the new chunk at the exactly-elapsed offset. frozen_steps is computed per
call from the latency EMA + 1 safety step. Logs to `inference_logs_rtc_async/expN.csv`
(old schema + `rtc_applied, frozen_steps_used, executed_steps_sent, latency_ema_s,
swap_offset`; `new_inference=1` = swap ticks).

Watch the once-per-second status line:
- `rtc_applied(last)=True` — server blending confirmed (False → T3 missing `--rtc`).
- "Chunk exhausted…" warnings — inference is slower than the lead; relaunch with
  `--trigger-lead-steps 6` (overlap 6, still fine) and note it in the episode log.

**Arm B — sync (only if re-running the sync arm):**

```bash
python run_inference_rtc.py --camera-host localhost
```

**Per episode (both clients):** Backspace in the sim to reset the scene, Ctrl+C the
client (saves the CSV), relaunch it — the restart calls `reset()` so the RTC cache
never blends across episodes.

---

## Protocol for the async round

1. 10 clean episodes with the async client (same task, checkpoint, 20 Hz — change
   nothing else, or attribution breaks).
2. Analyze (auto-detects the async log layout):

```bash
cd <Isaac-GR00T repo>
env -u PYTHONPATH uv run python examples/unitree_g1_rtc/analyze_chunk_boundaries.py \
  --log-dir <...>/inference_logs_rtc_async
```

3. Judge against the Part-8 success criteria (consolidated findings doc, 2026-07-18):

| # | Criterion | Reference |
|---|---|---|
| 1 | `boundary_jump` stays ≤ ~0.120 | sync-RTC already achieved 0.120 (pre-RTC: 0.326) — async must not undo it |
| 2 | State jerk mean AND p95 AND max improve vs BOTH pre-RTC (433.5 / 1289.6 / 4290) and sync-RTC (+15%/+32%/+218%) | if still worse than pre-RTC, the timing-distortion explanation needs rework |
| 3 | Post-swap dwell flat ≈ 50 ms (no more 0.4 ms / 20 ms rushed ticks) | analyzer "DWELL BY TICKS-SINCE-SWAP" table |
| 4 | Zero full stops (no dt > 75 ms stalls) | analyzer stall counter |
| — | Task success does not regress | 10/10 in the pre-RTC logged set |

The analyzer prints a PASS/FAIL table for 1, 3, 4 and both jerk conventions (naive
rows-as-uniform = prior team convention, and resampled-to-true-grid = physical). Judge
jerk *within* analyzer-v2 numbers (run it on the old log dirs too for like-for-like
references — absolute jerk depends on the combination convention).

---

## Verified / outstanding

- ✅ Async executor + observability: CPU-tested (8 tests), incl. the scripted
  H=16 / lead=8 / 3-tick-latency timeline.
- ✅ Analyzer v2 reproduces the 2026-07-10 baseline exactly in legacy mode; async mode
  exercised on synthetic logs.
- ✅ Server side unchanged from the sync round (per-call overrides already supported).
- ⚠ Async client not yet run against a live server — first live run is this round's
  first episode; if the very first swap logs `rtc_applied=0`, check T3 flags.
- ⚠ WBC docker needs sudo — launch as usual.
- Follow-ups: real-hardware cadence/latency re-measurement before any real-robot run;
  TRT port (Phase 3).
