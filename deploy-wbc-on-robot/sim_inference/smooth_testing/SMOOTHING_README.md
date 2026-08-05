# Smoothing A/B recorder

Goal: pick the smoothing technique that makes the arm motion fluid **without**
losing the ability to reach/grasp — decided from recorded numbers, not guesses.

Three files:

| File | Role |
|---|---|
| `action_recorder.py` | streams per-tick `raw` (VLA), `pub` (published), `state` (actual) + timing to a CSV, flushed every tick (Ctrl+C safe) |
| `run_inference_recorder.py` | the control loop + a `--mode` switch selecting one smoothing technique per run |
| `analyze_modes.py` | reads all CSVs, ranks techniques by smoothness vs. fidelity, prints a recommendation |

## 1. Record one run per mode (live, same scene each time)

Run these one at a time. Do the **same motion / scene** each run so they're comparable.
`raw` first — that's your current jerky baseline.

```bash
cd sim_inference/smooth_testing

python run_inference_recorder.py --mode raw            --policy-host <HOST> --camera-host <HOST>
python run_inference_recorder.py --mode lowpass
python run_inference_recorder.py --mode clamp
python run_inference_recorder.py --mode ramp
python run_inference_recorder.py --mode lowpass_clamp
python run_inference_recorder.py --mode smooth
```

Each writes `recordings/<mode>_<timestamp>.csv`. Ctrl+C to stop a run.

What each mode does (all operate on the 28-D `[left_arm, left_hand, right_arm, right_hand]` command):

- **raw** — no smoothing (baseline).
- **lowpass** — EMA: `pub = a·raw + (1-a)·prev`. Kills high-freq noise, adds lag.
- **clamp** — velocity clamp: caps per-tick change (arm vs hand separately). Kills spikes/jumps.
- **ramp** — cosine blend from last command into each new chunk over N ticks. This is the
  "fill the gap between a16 and new_a1" fix — targets the chunk **boundary** only.
- **lowpass_clamp** — EMA then clamp.
- **smooth** — ramp + lowpass + clamp together.

Tunable per run:
```bash
python run_inference_recorder.py --mode smooth \
  --ema-alpha 0.40 --max-delta-arm 0.10 --max-delta-hand 0.30 --ramp-steps 6
```
- `--ema-alpha` ↑ = more responsive, less smooth (1.0 = off).
- `--max-delta-arm` / `--max-delta-hand` = per-tick velocity cap (rad/tick). Hand needs
  a looser cap than the arm because the VLA closes the gripper in a ~1.5 rad step.
- `--ramp-steps` ↑ = softer chunk seams, but more lag reaching the new chunk.

## 2. Compare and pick a winner

```bash
python analyze_modes.py recordings/
```

Reads every CSV and prints, per mode:

- **SMOOTHNESS** (lower = smoother): arm/hand per-tick velocity (mean/p95/max) and
  jerk (3rd difference), plus boundary-step ÷ interior-step ratio (the seam).
- **FIDELITY**: `lag` = mean `|pub-raw|` (smoothing delay; high = sluggish) and
  `reach` = published travel ÷ raw travel per joint (`<1` = motion got shrunk; a grasp
  that no longer closes shows up as low **hand reach**).

The recommendation picks the **smoothest arm among modes that still track the VLA**
(arm lag < 0.05 rad, arm reach > 0.85, hand reach > 0.80). Adjust those thresholds in
`analyze_modes.py` for your task if needed.

## Real-Time Chunking (RTC) — server-side seam fix, composes with smoothing

RTC removes the chunk-boundary jump **at the source** (the N1.7 action head inpaints
the seam during denoising), with **no client-side lag**. It pairs well with `lowpass_clamp`:
RTC handles the seam, the filter handles per-tick jerk (arm snaps + the 1.5 rad grasp).

**Server must be started with `--rtc`** (on `run_gr00t_server.py`). The client sends the
options either way; the `rtc_applied` CSV column / `analyze_modes.py` `rtc_applied=%`
line reports whether the server actually blended.

```bash
# RTC + your chosen filter, recorded:
python run_inference_recorder.py --mode lowpass_clamp \
  --ema-alpha 0.4 --max-delta-arm 0.12 --max-delta-hand 0.35 \
  --rtc --execute-steps 8
```

Key RTC flags:
- `--rtc` — send RTC options + `policy.reset()` at start + re-query early.
- `--execute-steps 8` — re-query after 8 of 16 steps so RTC has an unexecuted
  remainder to blend (must be `< horizon`; `=16` disables RTC).
- `--rtc-frozen-steps 0` — keep 0 for this **synchronous** loop; only raise it in an
  async loop (`ceil(inference_latency / control_period)`).
- `--rtc-overlap-steps 0` / `--rtc-ramp-rate 5.0` — 0 overlap = server auto (`H−e`).

Output CSVs get an `_rtc` suffix and `analyze_modes.py` labels them `<mode>+rtc`, so a
plain run and an RTC run of the same filter compare side by side. Recommended A/B:

```bash
python run_inference_recorder.py --mode lowpass_clamp --ema-alpha 0.4 --max-delta-arm 0.12 --max-delta-hand 0.35            # filter only
python run_inference_recorder.py --mode lowpass_clamp --ema-alpha 0.4 --max-delta-arm 0.12 --max-delta-hand 0.35 --rtc      # filter + RTC
python analyze_modes.py recordings/
```
Expect `lowpass_clamp+rtc` to show a lower `bnd/int` seam ratio than `lowpass_clamp`
alone, with `rtc_applied` near 100% (if it's 0%, the server wasn't started with `--rtc`).

## Notes / what the existing logs already showed

- In the RTC logs (`inference_logs_rtc/exp*.csv`), the chunk boundary was only ~1.3× the
  interior step — the seam alone is **not** the main jerk. The bigger contributors were
  (a) the gripper's ~1.5 rad single-tick grasp steps and (b) inference-latency stalls
  (up to ~1.2 s) freezing the loop. So expect `clamp`/`smooth` (which cap per-tick motion)
  to beat `ramp` alone.
- This harness does **not** yet add async prefetch (background inference) — that removes
  the latency freeze and is already implemented in the repo-root `run_inference_smooth.py`
  + `action_smoother.py`. Once you've picked the per-tick technique here, fold it into that
  async path for the freeze fix too.
