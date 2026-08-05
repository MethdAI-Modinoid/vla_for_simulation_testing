# Unitree G1 — RTC client-side files

The client-side companions to the RTC (Real-Time Chunking) implementation in
`gr00t/policy/rtc.py` / `gr00t/policy/async_chunk_executor.py` /
`gr00t/policy/gr00t_policy.py`. This directory is the canonical distribution point:
deployment machines copy the client scripts next to their `sdk.py` / reference adapter
and run them from there.

| File | What it is |
|---|---|
| `run_inference_rtc_async.py` | **The async RTC client (current round).** Never blocks the control loop: `AsyncChunkExecutor` requests the next chunk in the background at `--trigger-lead-steps 8`, swaps at the co-temporal offset, auto-computes `frozen_steps` from measured latency. Logs every tick (`new_inference=1` = swap ticks, plus `frozen_steps_used`/`executed_steps_sent`/`latency_ema_s`/`swap_offset` columns). |
| `run_inference_rtc.py` | The sync RTC client (arm B of the A/B): `reset()` at episode start, blocking re-query after `--execute-steps 8`, `options={"rtc": {"executed_steps": N}}` per query. Kept for comparison runs. |
| `analyze_chunk_boundaries.py` | Offline analyzer for all three log layouts (pre-RTC e=16, sync e=8, async) — auto-detects. Latency tails, dwell regularity, chunk_metrics-matched smoothness numbers, per-joint ranking, state-jerk accounting (naive vs resampled-grid), Part-8 criteria table. |
| `RTC_SIM_RUNBOOK.md` | 4-terminal launch recipe for the gear_sonic MuJoCo cube-sorting sim + WBC + server + client, with the 3-arm protocol and success criteria. |

Dependencies **not** in this repo (expected on the deployment machine): both clients
import `rclpy` (ROS 2), `gr00t_wbc` (GR00T-WholeBodyControl repo), the local `sdk.py`
observation reader, and the reference script `run_Inferene_without_client_for_test.py`
(they reuse `GR00TG1Adapter` / `policy_action_to_control_goal` from it). Run them from
the directory that has those, with this repo's `gr00t` package importable.

The analyzer only needs this repo:

```bash
cd <this repo>
env -u PYTHONPATH uv run python examples/unitree_g1_rtc/analyze_chunk_boundaries.py \
    --log-dir <dir with exp*.csv>
```

Full engineering context: `RTC_implementation.md` (repo root; §13 = porting guide,
§14 = sync A/B results + async plan) and `RTC_README.md` (concepts + knobs).
