# Isaac GR00T N1.7 — Real-Time Chunking (RTC) for Unitree G1

Private fork of [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (base:
upstream `main` @ `9c7e746`) that adds **inference-time Real-Time Chunking** — smooth
handoffs between action chunks plus a non-blocking control loop — for Unitree G1
whole-body manipulation. No retraining, no checkpoint changes: everything here operates
at inference time. The upstream project README is preserved at
[`README_upstream.md`](README_upstream.md).

## The problem

GR00T predicts actions in chunks of 16 steps. Stock inference regenerates every chunk
from pure noise with no memory of the previous one, and every shipped control loop
blocks while the model thinks. Measured on our G1 cube-sorting sim (20 Hz, 1,055
boundaries): the commanded position **jumps 3.4×** a normal step at every chunk seam,
and the robot **fully stops ~130–170 ms** at every re-plan, rushing the first ticks of
each new chunk when the loop catches up.

## What this fork adds

1. **Server-side RTC blending** — the dormant inpaint/freeze/ramp primitive in the N1.7
   action head is wired into `Gr00tPolicy` and the policy server. The new chunk is
   seeded from the unexecuted tail of the previous one and blends into it instead of
   starting from scratch. Enable with `--rtc` on the server; clients pass per-call
   timeline info via `options={"rtc": {...}}`.
2. **Client-side async execution** — `AsyncChunkExecutor` requests the next chunk in the
   background while the robot keeps moving, swaps to it at the co-temporal offset (no
   step replayed or skipped), and auto-computes the frozen prefix from measured latency.
3. **Measurement tooling** — chunk smoothness metrics (`gr00t/eval/chunk_metrics.py`),
   logging G1 clients, and a boundary/jerk analyzer for A/B evaluation.

## Results so far (sim, checkpoint-22000, cube sorting)

| Metric | Pre-RTC | Sync RTC | Async RTC |
|---|---|---|---|
| Boundary position jump | 0.326 | **0.120 (−63%)** | target: hold ≤ 0.120 |
| Overall jerk (mean / p95 / max) | 433.5 / 1289.6 / 4290 | +15% / +32% / +218% ⚠ | target: beat both arms |
| Full stops per re-plan | every 16 ticks | every 8 ticks | none (background inference) |

Sync RTC proved the blending mechanism (−63% at the seam) but regressed overall jerk —
the robot stopped twice as often, which blending cannot fix. The async round addresses
exactly that; criteria and protocol are in
[`examples/unitree_g1_rtc/RTC_SIM_RUNBOOK.md`](examples/unitree_g1_rtc/RTC_SIM_RUNBOOK.md).

## Quickstart

**Policy server** (RTC lives here; drop the `--rtc*` flags for a baseline arm):

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path <ckpt>/sim_sorting/checkpoint-22000 \
  --embodiment-tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --port 5560 --rtc --rtc-ramp-rate 2.0
```

**Robot client** (from the directory holding your `sdk.py` / reference adapter script;
requires ROS 2 + `gr00t_wbc`):

```bash
# Async RTC (recommended): background inference, no stop-and-go
python run_inference_rtc_async.py --camera-host <localhost | 192.168.123.164>

# Sync RTC (blocking; kept for A/B comparison)
python run_inference_rtc.py --camera-host <...>
```

**Analyze the logged episodes** (auto-detects baseline / sync / async log layouts):

```bash
env -u PYTHONPATH uv run python examples/unitree_g1_rtc/analyze_chunk_boundaries.py \
  --log-dir <dir with exp*.csv>
```

**Tests** (CPU, a few seconds):

```bash
env -u PYTHONPATH uv run python -m pytest \
  tests/gr00t/policy/test_rtc_config.py tests/gr00t/policy/test_gr00t_policy_rtc.py \
  tests/gr00t/policy/test_async_chunk_executor.py tests/gr00t/eval/test_chunk_metrics.py \
  tests/gr00t/model/test_action_head.py tests/gr00t/policy/test_policy_service.py -q
```

## What changed vs upstream

| Area | Files |
|---|---|
| RTC core (new) | `gr00t/policy/rtc.py`, `gr00t/policy/async_chunk_executor.py`, `gr00t/eval/chunk_metrics.py` |
| Wiring (modified) | `gr00t/policy/gr00t_policy.py`, `gr00t/eval/run_gr00t_server.py`, `gr00t/eval/rollout_policy.py`, `gr00t/eval/open_loop_eval.py` |
| G1 clients + tooling | `examples/unitree_g1_rtc/` (sync + async clients, analyzer, sim runbook) |
| Tests | `tests/gr00t/policy/test_rtc_*.py`, `test_async_chunk_executor.py`, `tests/gr00t/eval/test_chunk_metrics.py`, extensions to `test_action_head.py` / `test_policy_service.py` |
| Docs | `RTC_README.md`, `RTC_implementation.md`, `getting_started/real_world_deployment.md` (status update) |

Everything else is untouched upstream Isaac-GR00T — installation, fine-tuning, data
format, and deployment instructions are in [`README_upstream.md`](README_upstream.md).

## Documentation map

- [`RTC_README.md`](RTC_README.md) — concepts and knobs: how the blending works, what
  every parameter does, when to change it.
- [`RTC_implementation.md`](RTC_implementation.md) — engineering passdown: measured
  numbers, design decisions, code locations, porting guide (§13), sync A/B findings and
  the async plan (§14).
- [`examples/unitree_g1_rtc/`](examples/unitree_g1_rtc/) — G1 clients, analyzer, and the
  step-by-step sim runbook with the A/B protocol and success criteria.

## Status

- ✅ Sync RTC end-to-end (server + client), CPU test suite, sim A/B completed
- ▶ **Current: async RTC sim round** — 10 episodes, judged on boundary jump ≤ 0.120,
  overall jerk better than both prior arms, flat post-swap dwell, zero full stops
- ⏭ Real-hardware bring-up (after sim passes; re-measure cadence/latency first)
- ⏭ TensorRT port of the RTC denoising loop

## License

Apache-2.0, same as upstream. This fork retains all NVIDIA copyright headers and
attribution; see [`README_upstream.md`](README_upstream.md) and `LICENSE`.
