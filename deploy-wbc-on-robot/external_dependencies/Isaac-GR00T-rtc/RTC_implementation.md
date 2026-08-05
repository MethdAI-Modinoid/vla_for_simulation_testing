# RTC Implementation — Passdown & Findings (Isaac GR00T N1.7)

> User-facing documentation lives in **`RTC_README.md`** (concept explanation, knobs,
> file-by-file changes, porting guide). This file is the engineering passdown.

> Working document. Holds every vital finding, code location, and decided value so future sessions
> do not need to re-dissect the codebase. Update as implementation proceeds.
> Created 2026-07-10 during the planning phase (no code changes made yet).

---

## 1. Headline findings from the dissection

1. **GR00T N1.7 does NOT ship working Real-Time Chunking** in any inference/deployment path.
   All shipped loops are plain receding-horizon: execute the first `execution_horizon` steps of a
   chunk, throw the rest away, regenerate the next chunk **from pure noise with no memory of the
   previous one**. Hard chunk boundaries → position jumps / velocity reversals / jerk.
2. **BUT the RTC math is already implemented — dormant — inside the action head.**
   `gr00t/model/gr00t_n1d7/gr00t_n1d7.py:358-394` (inside `get_action_with_features`, lines
   326-443). It activates only when `"action"` is present in the model's input batch AND an
   `options` dict carries 4 keys. Nothing in the repo ever supplies these. NVIDIA's own doc admits
   it: `getting_started/real_world_deployment.md:413` — "not wired into Gr00tPolicy or the
   server-client path".
3. **The single blocker is `Gr00tPolicy._get_action`** (`gr00t/policy/gr00t_policy.py:417`):
   it calls `self.model.get_action(**collated_inputs)` with **no `options`** and never feeds the
   previous chunk back. Everything below it (`Gr00tN1d7.get_action` → `action_head.get_action` →
   `get_action_with_features`) already forwards `options`. The ZMQ client/server also already
   transports an `options` dict (`gr00t/policy/server_client.py:387-393`). Only Gr00tPolicy drops it.
4. **All shipped robot loops are synchronous** (robot blocks during inference). No action
   buffering, no async executor. The doc's "Async Inference + RTC" pseudocode
   (`real_world_deployment.md:442-462`) is aspirational, not implemented.
5. **No tests exercise RTC** (grep for `rtc_` in `tests/` → nothing). The assert block + slice math
   in `gr00t_n1d7.py:364-394` is the ONLY spec of the options API.
6. **TensorRT path lacks RTC**: `scripts/deployment/trt_model_forward.py` reimplements the
   denoising loop in plain Python (lines 459-586; dit-only variant 905-988) with vanilla Euler
   (`actions += dt * pred_velocity`, line 584) — no `vel_strength`, no inpaint. Porting is a
   pure-Python edit (the loop is never exported to ONNX; only submodules are). Deferred to Phase 3.

## 2. How the dormant RTC primitive works (the spec)

Flow-matching sampler (`get_action_with_features`):
- Noise init: `actions = torch.randn((B, config.action_horizon, action_dim=132))` (line 349).
- Euler loop: `for t in range(num_inference_timesteps)` (line 397, default 4 steps),
  update `actions = actions + dt * pred_velocity * vel_strength` (line 435), `dt = 1/num_steps`.
- `vel_strength = ones_like(actions)` normally (line 356) → vanilla Euler.

RTC branch (fires only when `"action" in action_input`, line 358):
- Required `options` keys (asserted, lines 364-368): `action_horizon` (H, the UNPADDED horizon),
  `rtc_overlap_steps` (w), `rtc_frozen_steps` (f), `rtc_ramp_rate` (r).
- **Inpaint** (lines 373-378): `actions[:, :w, :] = action_input["action"][:, H-w : H, :]`
  — the first w steps of the noise latent are replaced by the LAST w steps of the supplied
  previous chunk (in NORMALIZED action space).
- **Freeze** (line 379): `vel_strength[:, :f, :] = 0.0` — first f steps never change
  (they model the steps that execute during inference latency).
- **Ramp** (lines 380-394): normalized exponential `1 - exp(-r·t)` over `[f:w)` — the model may
  progressively correct the middle of the overlap; beyond w it plans freely.

## 3. Hyperparameter map — the three "horizon" concepts (don't confuse them)

| Concept | Where set | Stock default | THIS project |
|---|---|---|---|
| Model predicted-chunk length (architectural cap) `action_horizon` | `gr00t/configs/model/gr00t_n1d7.py:76`; persisted in checkpoint `config.json` | 40 | **16** (user's fine-tune) |
| Training action window = `"action"` `ModalityConfig.delta_indices` | `gr00t/configs/data/embodiment_configs.py` (stock G1 full-body: `range(50)`, line 133) or custom config | varies | **16** (must equal model cap; VERIFY, see §5) |
| Execution stride (steps executed before re-query) | `n_action_steps` (`gr00t/eval/rollout_policy.py:649`, default 8); `execution_horizon` (`gr00t/eval/open_loop_eval.py:253`, default 16); DROID `open_loop_horizon=15`; SO100 `action_horizon=8` | — | unknown cadence on real G1 → configurable |

Other key model knobs (`gr00t/configs/model/gr00t_n1d7.py`): `num_inference_timesteps=4` (line 106),
`max_action_dim=132` (75), `num_timestep_buckets=1000` (110), `state_dropout_prob=0.8` (118 — the
PRETRAIN value; finetune default is 0.2 via `gr00t/configs/finetune_config.py:61`).
Horizon contract resolver: `PolicyHorizonSpec`, `gr00t/eval/_horizon_contract.py:71-183`
(note: old `--action-horizon` eval flag was renamed `--execution-horizon`, see lines 33-56).

## 4. Critical code locations (verified this session)

| What | Where |
|---|---|
| Flow-matching sampler + dormant RTC | `gr00t/model/gr00t_n1d7/gr00t_n1d7.py:326-443` (RTC: 358-394; Euler update: 435) |
| Options forwarding chain (already works) | `gr00t_n1d7.py:603` → `:612` → `:446` → `:326` |
| **The blocker** — options dropped, no prev chunk | `gr00t/policy/gr00t_policy.py:417` |
| **The cache tensor** — normalized chunk pre-decode | `gr00t/policy/gr00t_policy.py:418` (`model_pred["action_pred"].float()`, shape `[B, 16, 132]`) — cache THIS, never re-normalize |
| ZMQ transport of options (already works) | `gr00t/policy/server_client.py:387-393`; sync REQ/REP; server `reset` endpoint exists; `Gr00tPolicy.reset` is a no-op (perfect cache-clear hook) |
| Sim consumption loop | `gr00t/eval/sim/wrapper/multistep_wrapper.py:297-327`; driver `gr00t/eval/rollout_policy.py:281-399`; **vectorized envs autoreset WITHOUT policy.reset() (rollout_policy.py:317) → cache unsafe for n_envs>1** |
| G1 sim env mapping | `gr00t/eval/sim/env_utils.py:24-26` (`gr00tlocomanip_g1*` → `EmbodimentTag.UNITREE_G1`) |
| G1 real deployment reference | `examples/GR00TWholeBodyControl/README.md:89-100` (policy server + robot-side controller) |
| Relative-action encode/decode | `gr00t/data/state_action/state_action_processor.py:369-394` (encode: `action_abs − state[-1]`), `:421-518` (decode against CURRENT state) |
| Sampler cadence internals | `gr00t/data/dataset/sharded_single_step_dataset.py:160-231` (episode_sampling_rate semantics) |
| Jerk metrics (described, NOT implemented) | `getting_started/real_world_deployment.md:319-375`; RTC status :413; async pseudocode :442-462 |
| Existing CPU test scaffolds to extend | `tests/gr00t/model/test_action_head.py` (tiny head: horizon 4, 2 denoise steps); `tests/gr00t/policy/test_gr00t_policy.py` (mocked model+processor) |
| TRT loop (Phase 3 target) | `scripts/deployment/trt_model_forward.py:459-586`, dit-only `:905-988`, monkey-patch `:775/:871` |

## 5. User deployment facts (confirmed 2026-07-10)

- Embodiment: **`unitree_g1_full_body_with_waist_height_nav_cmd`** — sim (`gr00tlocomanip_g1*`) AND real G1.
- **PREFLIGHT DONE 2026-07-16** on `~/Projects/r2d2/rahul_ckpts/sim_sorting/checkpoint-22000`
  (a checkpoint-24000 also exists; CSVs came from 22000). **Verdict: RTC-safe, no overrides needed.**
  Two user-stated facts were wrong in letter but right in effect:
  - Model `config.json` `action_horizon = 40` (PADDED), not 16. The 16 is the UNPADDED chunk:
    processor G1 action `delta_indices = [0..15]`. Chunks served to the client are 16 long.
    This is exactly the padded≠unpadded case the implementation handles —
    `_prepare_rtc_injection` sends `options["action_horizon"] = len(delta_indices) = 16`, and
    the model slices the cached (padded, 40-long) chunk at `[16−w : 16]`; covered by
    `test_padded_vs_unpadded_horizon_indexing`. No code change needed.
  - `use_relative_action = True` in the checkpoint processor (not false!) — BUT the fine-tune
    used a custom modality config where **all 7 G1 action groups are `rep: absolute`**
    (stock repo config has RELATIVE arms; this checkpoint overrides them). With no RELATIVE
    groups the flag is inert: no conversion at encode/decode, `relative_action` stats are `{}`,
    and `_validate_rtc_compatibility` passes cleanly (it checks the LOADED action_configs, not
    the flag alone). Effectively behaves as absolute — cached chunks inpaint as-is.
  - Other confirmed: `num_inference_timesteps=4`, `max_action_dim=132`, `use_percentiles=True`,
    `clip_outliers=True`, video/state `delta_indices=[0]`.
- Robot-side client = user's own inference script talking to `run_gr00t_server.py` — **modifiable**
  (can send per-call options and adopt the async executor).
- Control cadence: **MEASURED IN MUJOCO SIM 2026-07-15** (see §5.1) — 20 Hz control loop
  (50 ms/tick). The client executes the full 16-step chunk then blocks on `get_action` (e = 16),
  which leaves ZERO overlap for RTC; the client loop must re-query earlier (sync e < 16) or adopt
  `AsyncChunkExecutor`. **Real-hardware cadence and latency still unverified** — re-run the same
  logging on the real G1 before locking parameters there.
- Training knobs user changed: `episode_sampling_rate 0.1→1.0` (see §8 — did NOT add data),
  `warmup_ratio=0.05` (that IS the default), `state_dropout_prob 0.2→0.5` (more vision-reliant;
  pretrain used 0.8; user observes better robustness in dynamic visual scenes — expected effect).
- Scope decisions: **PyTorch first, TRT later; async inference IN scope; sync RTC ships first.**

## 5.1 Measured discontinuity study — MuJoCo sim (2026-07-15, user's `actionchunk_discontinuity.pdf`)

Setup: **MuJoCo simulation of the G1 (not real hardware)** — cube-sorting task, G1 waist-anchored
(upper body only — 28 commanded DoF: 14 arm + 14 hand), VR teleop demos (96, positive-only),
checkpoint at 22k steps, batch 250, action_horizon=16, state_dropout 0.5, episode_sampling 1.0.
Data: 10 successful episodes, 1,055 chunk boundaries, 16,802 control ticks.

Transferability: the plan-mismatch discontinuity (fresh-noise chunks) is a property of the model +
client loop and transfers to hardware; the latency numbers are specific to this inference machine +
sim pipeline (real camera read / network may differ); 20 Hz is the sim loop rate — confirm the real
controller runs the same rate before reusing the derived parameters on hardware.

**Timing facts (these drive RTC parameters):**
- Control loop: **20 Hz (50 ms/tick)** → `control_period_s = 0.05`.
- Client loop today: executes ALL 16 actions then blocks on `policy.get_action()` — i.e. **e = 16
  = H → `H − e = 0` overlap → sync RTC cannot fire at all until the loop changes.** This is the
  single most consequential finding.
- Full generation-to-execution gap (frame → command published): **mean 129.4 ms, median 128 ms,
  p95 136.9 ms, max 198.7 ms** (n=1,045). Raw model inference alone: mean 79.4 ms, p95 87 ms —
  pipeline overhead ≈ 50 ms. → frozen/lead math must use the FULL gap, not raw inference time.
- Chunk-start dwell irregularity: after the blocking call the loop rushes to resync — idx0 held
  129.4 ms (the gap itself), **idx1 held 0.4 ms, idx2 held 20.6 ms**, idx3–15 = 50 ms ± 0.02.
  The first two actions of every chunk are effectively skipped. Blocking causes this; only the
  async executor removes it (RTC alone does not).

**Discontinuity magnitude (baseline numbers for the A/B):**
- At the boundary tick vs all other ticks: position change **3.41×**, velocity **2.91×**,
  acceleration **2.86×** normal. Single-tick position change median ≈ 0.15 vs 0.05 normalized.
- Jerk (3rd derivative of measured/sim state) peaks at boundary **+2 ticks** (594 vs 413 typical,
  1.44×) — the +2 offset is simulated plant inertia + controller response; separately measured
  tracking lag ≈ 3 ticks. Top-5% jerkiest ticks cluster within ~±2 ticks of boundaries.
- In-chunk tracking drift is mild (1.09× over a full chunk) → the plan does NOT go stale fast →
  a gentle ramp is fine.

**Raw-CSV re-analysis (2026-07-15, `/home/praveenkathirvel/Projects/r2d2/inference_logs/exp1..10.csv`,
1,055 inferences; script: session scratchpad `analyze_rtc_baseline.py`):**
- CSV schema: `iteration, wall_time, loop_time, dt_since_last, new_inference, horizon,
  inference_latency_s, action_idx`, + 28 `state_*` + 28 `target_*` (left/right arm 7 + hand 7).
- Latency tails: raw model p99 = 97.9 ms, max 148.6; FULL gap p95 = 136.9, **p99 = 147.9,
  max = 198.7** → frozen to cover p99 = 3 ticks, worst-case = 4. → `frozen_steps=3` +
  `safety_margin_steps=1` covers the observed maximum.
- **`chunk_metrics.py` baseline for the A/B (execute_steps=16, episode-mean):**
  `intra_accel = 0.1227`, `boundary_jump = 0.2990`, `momentum_shift = 0.3252` (cosine).
  Boundary/intra-step L2 ratio **3.42×** (independently reproduces the PDF's 3.41×).
  exp7 is the worst episode (jump 0.510, max gap 198.7 ms).
- **Per-joint: ARMS carry the discontinuity** — arm mean boundary/intra ratio 3.95× vs hands
  1.59×; worst joints `left_arm_3/5/6`, `right_arm_1/4` (right_arm_4 largest absolute jump
  0.103). → judge the A/B on arm joints primarily.
- **Jumps concentrate during grasps**: corr(hand motion before boundary, jump size) = **0.617**;
  boundaries with active hands jump 3.3× larger (0.663 vs 0.202) — RTC's payoff is concentrated
  exactly where failures happen (contact phases). Supports gentle `ramp_rate ≈ 2.0`.

**Client script found & audited (2026-07-15):**
`/home/praveenkathirvel/Projects/r2d2/data_collection/run_Inferene_without_client_for_test.py`
(+ `sdk.py`, `collect_joint_data.py`) — uses `gr00t.policy.server_client.PolicyClient`, 20 Hz ROS
rate loop, `ActionBuffer.needs_update()` = exhaust-then-block (the e=16 pattern).
- Camera/obs read (`obs_reader.get_observation`) happens BEFORE the timed `get_action` →
  async executor's round-trip EMA misses obs-read time → use `safety_margin_steps=1`.
- **NO `policy_client.reset()` anywhere** — and the server-side `Gr00tPolicy` RTC cache persists
  across client restarts. Client MUST call `reset()` at episode start (method exists:
  `server_client.py:395`) or the first chunk of a new episode blends against the previous
  episode's tail.
- The ROS `create_rate(20)` catch-up after the blocking call is what produces the idx1=0.4 ms /
  idx2=20.6 ms rushed dwells.
- Integration points: sync RTC → pass `options={"rtc": {"executed_steps": 8}}` in
  `GR00TG1Adapter.get_action` and flip `needs_update()` to `current_idx >= 8`; async →
  replace `ActionBuffer` with `gr00t/policy/async_chunk_executor.AsyncChunkExecutor`.

**PDF's own RTC parameter recommendations (consistent with ours):**
- `rtc_frozen_steps = 3` (129.4 ms / 50 ms = 2.6–2.7 → round up; raw-inference-derived 1.6
  would under-commit by ~a full tick). Independently reinforced by the idx0–2 dwell irregularity.
- `rtc_overlap_steps ≈ 8–12`.
- `rtc_ramp_rate ≈ 2.0` (moderate; aggressive ramping unnecessary given mild staleness).

**Derived settings (H=16, 20 Hz — validated for the sim pipeline; re-measure gap/cadence on real
hardware before locking there):**
- Sync A/B (isolates the plan-mismatch fix): client re-queries at e=8 with
  `options={"rtc": {"executed_steps": 8}}`, server `--rtc --rtc-execution-horizon 8` → w=8, f=0,
  try ramp_rate 2.0. The 129 ms stop-and-go pause remains (blocking is unchanged).
- Async (the real fix — removes both plan mismatch AND timing distortion):
  `AsyncChunkExecutor(policy, control_period_s=0.05, trigger_lead_steps=8, safety_margin_steps=0)`
  → triggers at executed_steps=8 → default overlap w=8 (in PDF's 8–12 band); frozen auto =
  ceil(EMA_latency × 20) ≈ 3 (matches PDF; margin 0 because the client-side round-trip EMA
  already ≈ the full pipeline gap; use margin 1 if camera read happens before the call).
  Minimum safe lead is 3 (p95 gap 2.74 ticks) / 4 (max gap); 8 chosen to hit the overlap band.

## 6. Alignment math (the recipe)

H = unpadded horizon (16 here), e = steps executed since last inference, w = overlap, f = frozen.
New-chunk step k is co-temporal with old-chunk step e+k. Model slices `prev[:, H−w : H]`.

- **Canonical: w = H − e, pass the cached chunk UNSHIFTED** → slice `[e : H]` = exactly the
  not-yet-executed remainder, perfectly aligned with the new chunk's start. No shifting.
- Smaller w (< H − e): right-shift cache by `s = H − e − w` so the slice reads `prev[e : e+w]`.
- Constraints: `0 ≤ f ≤ w ≤ H − e`. `w = 0` or no cache (first call of episode) → vanilla sampling.
- Sync loop: `f = 0`. Async loop: `f = ceil(inference_latency / control_period)`.
- Defaults chosen (H=16): sim e=8 → w=8, f=0, ramp_rate=5.0.

## 7. Implementation plan

### Phase 1 — synchronous RTC (independently shippable)
1. **NEW `gr00t/policy/rtc.py`**: `RTCConfig` dataclass (`execution_horizon`, `overlap_steps=None`
   → H−e, `frozen_steps=0`, `ramp_rate=5.0`, `allow_relative=False`);
   `resolve_rtc_options(...) -> dict|None` (returns the 4 model option keys as plain int/float,
   msgpack-safe, validates constraints); `shift_prev_chunk`; `merge_rtc_call_options` (reads
   per-call `options["rtc"] = {"executed_steps": int, "enabled": bool}`).
2. **`gr00t/policy/gr00t_policy.py`**: `Gr00tPolicy.__init__(..., rtc_config=None)`; in
   `_get_action` inject cached chunk as `collated_inputs["inputs"]["action"]` (bf16, device) +
   `options=resolved`; always cache `model_pred["action_pred"].detach().clone()` post-inference;
   `reset()` clears cache; invalidate on batch-size mismatch; relative-rep gate at init; return
   `info={"rtc_applied", "rtc_overlap_steps"}`. Mirror in `Gr00tSimPolicyWrapper` (line 638).
3. **`gr00t/eval/run_gr00t_server.py`**: ServerConfig flags `--rtc`, `--rtc-execution-horizon`,
   `--rtc-overlap-steps`, `--rtc-frozen-steps`, `--rtc-ramp-rate` → RTCConfig → Gr00tPolicy.
   This alone enables RTC for the real G1 (client optionally sends `executed_steps` per call).
4. **`gr00t/eval/rollout_policy.py`** (G1 sim): RTC fields on RolloutConfig; pass
   `executed_steps=n_action_steps` per query; `policy.reset()` per episode; **assert n_envs==1**.
5. **`gr00t/eval/open_loop_eval.py`**: `--rtc` flags; the quantitative A/B harness.
6. **NEW `gr00t/eval/chunk_metrics.py`**: implement the doc's three metrics as tested functions:
   `metric_mean_acceleration` (jerk proxy), `metric_boundary_jump`, `metric_momentum_shift`
   (velocity cosine across boundary); wire into open_loop_eval logging.
7. Docs: update `real_world_deployment.md:413`; add G1 server-flags + client-snippet section.

### Phase 2 — async inference + RTC
- **NEW `gr00t/policy/async_chunk_executor.py`** (client-side, reusable in user's robot script):
  background thread OWNS the PolicyClient (ZMQ REQ is not thread-safe; single in-flight request by
  construction). Trigger `get_action` before chunk exhaustion with anticipated `executed_steps`;
  swap chunks at the elapsed offset on arrival; `frozen_steps = ceil(EMA_latency × control_Hz)`.
  Server unchanged (Phase 1 already supports variable e per call).
- Validate in sim behind `--async-rtc` with simulated latency; G1 client integration example.

### Phase 3 (follow-up) — TensorRT port
- Copy inpaint + vel_strength gate into `trt_model_forward.py` loops (pure Python; `.clone()` the
  shared `init_actions` buffer before in-place writes); gpu-marked torch-vs-TRT parity test.

## 8. Training-knob explanations (asked & answered, for the record)

- **`episode_sampling_rate` (default 0.1; user set 1.0)** — misleading name; it does NOT drop 90%
  of data. `sharded_single_step_dataset.py:184-193`: each episode's timesteps are split into
  `1/rate` interleaved slices and ALL slices are distributed across different shards. All episodes
  + all timesteps remain used; the knob controls how scattered each episode is across shards
  (batch decorrelation). Setting 1.0 keeps each episode as one contiguous block in one shard →
  MORE correlated batches, no extra data. 0.1 was fine.
- **`warmup_ratio = 0.05`** — is already the default (`training_config.py:48`). First 5% of steps
  ramp LR linearly 0→peak to protect pretrained weights from violent early updates.
  (`warmup_steps` overrides it if nonzero.)
- **`state_dropout_prob` (finetune default 0.2; user set 0.5; pretrain used 0.8)** — training-only:
  with prob p the whole proprio-state embedding is zeroed (`gr00t_n1d7.py:220-226`), forcing
  vision reliance and preventing the proprio shortcut / causal confusion. Higher = LESS proprio
  dependence (not "overfit on states"). 0.5 improving dynamic-visual robustness is the expected
  effect; watch precision on vision-ambiguous, body-configuration-dependent motions.
- None of these affect chunk boundaries — jerk is an inference-time problem; RTC needs no retraining.

## 9. Tests & verification

Tests (CPU-safe; run `python -m pytest tests/ -m "not gpu" -v --timeout=300`; lint `pre-commit run --all-files`):
- Extend `tests/gr00t/model/test_action_head.py` (`TestActionHeadRTC`, tiny head): full-freeze
  f==w → output `[0:w)` bit-equals inpainted tail; missing-option asserts; padded≠unpadded case
  (config horizon 6, options horizon 4); w=0 ≡ vanilla under fixed seed.
- NEW `tests/gr00t/policy/test_rtc_config.py`: constraint matrix; shift math ≡ `prev[e:e+w]`; msgpack-plain types.
- NEW `tests/gr00t/policy/test_gr00t_policy_rtc.py` (mocked): 1st call no injection; 2nd call correct
  injection+options; reset clears; batch-size change invalidates; `rtc_config=None` ≡ current behavior.
- Extend `tests/gr00t/policy/test_policy_service.py`: nested `options["rtc"]` serializer round-trip.

A/B verification (RTC off vs on, same checkpoint/seed):
0. Preflight §5 (checkpoint config values).
1. `open_loop_eval.py --execution-horizon 8 [--rtc]` → expect boundary_jump ↓, momentum cosine ↑,
   mean acceleration ↓; MSE/MAE vs GT as no-regression guard.
2. Sim: `rollout_policy.py` on `gr00tlocomanip_g1`, `--n-envs 1 --n-action-steps 8 --seed 42 [--rtc]`
   → success rate must not regress; chunk metrics on logged actions.
3. Real G1: server `--rtc --rtc-execution-horizon <measured stride>`; log commanded joint targets
   both ways; metrics offline. Then repeat with async executor.

## 10. Risks / gotchas

1. `options["action_horizon"]` MUST be the unpadded horizon (silent garbage inpaint otherwise).
   Moot here (16==16) but unit-tested for other embodiments.
2. Cache must be the NORMALIZED pre-decode tensor (`gr00t_policy.py:418`) — never re-encode
   unnormalized actions.
3. Relative action reps + `use_relative_action=true` → stale reference frame on inpainted prefix;
   gated off by default (inert for this checkpoint).
4. Stateful server cache assumes ONE control loop per server; unsafe with vectorized envs
   (assert n_envs==1) and multiple clients (documented).
5. Server-default `executed_steps` must match the real controller's query cadence; per-call
   override provided; measure cadence first.
6. Async: ZMQ REQ socket single-threaded ownership; one in-flight request.

## 11. Implementation status (what was actually built)

### Phase 1 + Phase 2 implemented (2026-07-11)

**New files:**
- `gr00t/policy/rtc.py` — `RTCConfig` (execution_horizon, overlap_steps, frozen_steps,
  ramp_rate, allow_relative), `resolve_rtc_options()` (validates/clamps, returns the 4
  model option keys as plain int/float), `compute_prev_chunk_shift()` + `shift_prev_chunk()`
  (alignment for w < H−e), `merge_rtc_call_options()` → `RTCCallParams` (per-call client
  overrides: enabled, executed_steps, overlap_steps, frozen_steps, ramp_rate).
- `gr00t/policy/async_chunk_executor.py` — `AsyncChunkExecutor` (Phase 2): single-worker
  background inference, trigger at `remaining <= lead`, swap at co-temporal offset
  (`ticks_since_trigger`), frozen_steps = EMA(latency)/control_period + safety margin.
  Fixes the doc pseudocode's replay bug (it restarted new chunks at index 0).
- `gr00t/eval/chunk_metrics.py` — `metric_intra_accel`, `metric_boundary_jump`,
  `metric_momentum_shift`, `compute_chunk_metrics` (nan-safe aggregate).

**Modified files:**
- `gr00t/policy/gr00t_policy.py` — `Gr00tPolicy(..., rtc_config=None)`; caches
  `model_pred["action_pred"].detach().clone()` after every inference; injects it as
  `collated_inputs["inputs"]["action"]` + `options=` on the next call;
  `_validate_rtc_compatibility()` refuses RELATIVE+use_relative_action checkpoints;
  `reset()` clears the cache; cache invalidated on batch-size change; info returns
  `rtc_applied` / `rtc_overlap_steps` / `rtc_frozen_steps`. When `rtc_config=None` the
  model call is byte-identical to the pre-RTC code path (no options kwarg).
- `gr00t/eval/run_gr00t_server.py` — flags: `--rtc --rtc-execution-horizon
  --rtc-overlap-steps --rtc-frozen-steps --rtc-ramp-rate`.
- `gr00t/eval/rollout_policy.py` — same flags on `RolloutConfig` (+ asserts `--n-envs 1`);
  forwards `options={"rtc": {"executed_steps": n_action_steps}}` per query and calls
  `policy.reset()` on episode end.
- `gr00t/eval/open_loop_eval.py` — same flags; per-trajectory `policy.reset()`; collects
  full predicted chunks and logs `compute_chunk_metrics` per trajectory + averaged (the
  A/B harness). `evaluate_single_trajectory` now returns `(mse, mae, chunk_metrics)`.
- `getting_started/real_world_deployment.md` — status paragraph rewritten (RTC wired),
  AsyncChunkExecutor usage snippet added, metrics pointer added.
- `tests/gr00t/policy/test_policy_service.py` — MockPolicy echoes received options;
  nested `options["rtc"]` ZMQ round-trip test.

**New tests (all CPU):** `tests/gr00t/model/test_action_head.py::TestActionHeadRTC`
(full-freeze bit-equality, ramp deviation, padded-vs-unpadded indexing, missing-options
asserts, zero-overlap ≡ vanilla under fixed seed — allclose, since CPU attention is
nondeterministic at 1 ULP), `tests/gr00t/policy/test_rtc_config.py`,
`tests/gr00t/policy/test_gr00t_policy_rtc.py` (mocked policy: caching/injection/reset/
invalidation/guards), `tests/gr00t/policy/test_async_chunk_executor.py`,
`tests/gr00t/eval/test_chunk_metrics.py`. Status: 114 passed.

**Environment gotcha:** ROS Humble on PYTHONPATH breaks pytest plugin loading
(`launch_testing` → missing `lark`). Run tests with `env -u PYTHONPATH uv run python -m
pytest ...`.

### How to use (G1)

```bash
# Real robot: server-side RTC; your client optionally sends
#   options={"rtc": {"executed_steps": N}} per query and calls reset() per episode.
python gr00t/eval/run_gr00t_server.py --model-path <ckpt> \
  --embodiment-tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --rtc --rtc-execution-horizon <your query stride>

# Async client loop: AsyncChunkExecutor (see real_world_deployment.md snippet).

# A/B verification (open loop, needs GPU + checkpoint + dataset):
python gr00t/eval/open_loop_eval.py --model-path <ckpt> --dataset-path <ds> \
  --embodiment-tag <tag> --execution-horizon 8 [--rtc]
# → compare logged boundary_jump (↓), momentum_shift (↑), intra_accel (↓), MSE/MAE (flat)

# Sim closed-loop:
python gr00t/eval/rollout_policy.py --env-name gr00tlocomanip_g1/... \
  --n-envs 1 --n-action-steps 8 --seed 42 [--rtc]
```

### Remaining / follow-ups
- Phase 3: port inpaint + vel_strength gate into the TRT loops
  (`scripts/deployment/trt_model_forward.py:531-584` and dit-only `:932-978`); `.clone()`
  the shared `init_actions` buffer before in-place writes; gpu-marked parity test.
- ~~Preflight~~ **DONE 2026-07-16** (§5): checkpoint-22000 is RTC-safe (padded 40 / unpadded 16
  handled; all action groups absolute). Next: GPU A/B validation — server
  `--model-path ~/Projects/r2d2/rahul_ckpts/sim_sorting/checkpoint-22000 --rtc
  --rtc-execution-horizon 8 --rtc-ramp-rate 2.0`, client at e=8 with reset() per episode.
- Cadence/latency **measured in MuJoCo sim 2026-07-15** (§5.1): 20 Hz, 129.4 ms full gap.
  Still to do on REAL hardware: log the same cadence + frame→execution gap there. Next in sim:
  change the client loop (it currently exhausts the chunk and blocks — e=16 → no overlap);
  adopt `AsyncChunkExecutor` with the §5.1 derived settings, or sync e=8 for the isolated A/B
  first.

## 11.1 Sim environment readiness (2026-07-17)

The sim stack lives across three repos; launch recipe in
**`~/Projects/r2d2/RTC_SIM_RUNBOOK.md`** (4 terminals: sim / WBC docker / server / client).

- Sim: gear_sonic MuJoCo (`~/Projects/r2d2/gear_sonic`, cube-sorting scene from the PDF —
  `README_manipulation_scene.md`), runs in conda `unitree_sim` BUT its editable
  `unitree_sdk2py` points at the deleted `~/g_star/` workspace → launch with
  `PYTHONPATH=$HOME/Projects/r2d2:$HOME/Projects/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python`
  (verified). Camera publishes on ZMQ :5555; Backspace = episode reset; key 9 releases hanger.
- WBC: `~/Projects/GR00T-WholeBodyControl`, docker via `sudo ./start_wbc.sh` (user-run).
- Server: Isaac-GR00T uv venv; **must use `--port 5560`** (default 5555 collides with the
  sim camera). Embodiment tag parses (`EmbodimentTag.UNITREE_G1` == the full-body value).
  **HF status (2026-07-17):** token installed (`praveen-k-etr`) but the account is NOT on
  the authorized list for the gated `nvidia/Cosmos-Reason2-2B` backbone — model load 401s.
  Fix: request access at https://huggingface.co/nvidia/Cosmos-Reason2-2B, or copy
  `~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/` from a machine that has it
  (the fine-tuning box does). `qwen3_backbone.py` has no local-path escape hatch.
- Client: conda `isaaclab` env resolves gr00t_wbc (editable → `~/Projects/GR00T-WholeBodyControl`),
  rclpy (ROS PYTHONPATH), and `gr00t` (→ this repo). NOTE: gear_sonic's own
  `run_vla_inference.py` is for the `unitree_g1_sonic` LATENT embodiment (motion tokens) —
  not usable with this checkpoint; the joint-space client is the r2d2/data_collection one.
- New files created: `~/Projects/r2d2/data_collection/run_inference_rtc.py` (RTC client:
  reset-at-start, e=8 re-query, per-call options, CSV logging in the exp*.csv schema +
  `rtc_applied` column) and `~/Projects/r2d2/analyze_chunk_boundaries.py` (portable analyzer,
  auto-detects executed-steps; validated — reproduces the July-10 baseline exactly).

## 12. Status log

- 2026-07-10 — Dissection + plan complete.
- 2026-07-11 — Phase 1 (sync RTC end-to-end) + Phase 2 (async executor) implemented with
  full CPU test coverage. Full CPU suite: 628 passed (tests/) + 123 passed
  (tests/scripts); the single tests/scripts failure
  (test_device_standalone_inference_script_pytorch) is pre-existing/environmental — it
  needs HF_TOKEN + CUDA to download the gated 3B checkpoint, unrelated to RTC. Ruff
  format + check clean. Docs updated. TRT port (Phase 3) pending.
- 2026-07-15 — User's MuJoCo-sim discontinuity study analyzed (§5.1): 20 Hz, e=16
  exhaust-then-block client, gap 129.4 ms mean / 198.7 max, boundary 3.41×; client script
  audited (no reset(), obs-read before timed call).
- 2026-07-16 — Checkpoint preflight PASSED (§5): padded 40 / unpadded 16; all action
  groups absolute → `use_relative_action=True` inert; RTC-safe with no overrides.
- 2026-07-17 — Sim environment readied (§11.1 + `~/Projects/r2d2/RTC_SIM_RUNBOOK.md`):
  sim/client/server launch verified, RTC client + portable analyzer written. Remaining
  blockers: HF token (user), WBC docker sudo (user). Next: run the A/B.
- 2026-07-17 (later) — Decision: inference will NOT run on this machine. All RTC
  changes committed on branch `rtc-inference` (base: upstream `main` @ 9c7e746) for
  porting to the deployment machine; r2d2-side client/analyzer/runbook copied into
  `examples/unitree_g1_rtc/`. RTC test suite re-run: 106 passed; ruff clean. See §13.
- 2026-07-18 — Sync-RTC A/B ran on the deployment machine; team's consolidated findings
  digested into §14: boundary jump −63% (blending works), overall jerk regressed
  (stop-and-go doubled, not a blending fault). Next step = ASYNC RTC in sim. Built this
  session: executor observability, `run_inference_rtc_async.py`, analyzer v2
  (async-aware + jerk accounting), repo README replaced (upstream → README_upstream.md).

## 13. Porting this work to another machine (2026-07-17)

Inference will run on a different machine that already has: this repo (some upstream
commit), the eval/inference scripts, and the `sim_sorting` checkpoints. Everything
RTC-related is carried by **one commit on branch `rtc-inference`** (based on upstream
`main` @ `9c7e746`, the current NVIDIA tip).

### Complete change manifest (nothing else was touched)

Core (new): `gr00t/policy/rtc.py`, `gr00t/policy/async_chunk_executor.py`,
`gr00t/eval/chunk_metrics.py`.
Core (modified): `gr00t/policy/gr00t_policy.py`, `gr00t/eval/run_gr00t_server.py`,
`gr00t/eval/rollout_policy.py`, `gr00t/eval/open_loop_eval.py`,
`getting_started/real_world_deployment.md`.
Tests (new): `tests/gr00t/policy/test_rtc_config.py`,
`tests/gr00t/policy/test_gr00t_policy_rtc.py`,
`tests/gr00t/policy/test_async_chunk_executor.py`, `tests/gr00t/eval/test_chunk_metrics.py`.
Tests (modified): `tests/gr00t/model/test_action_head.py` (adds `TestActionHeadRTC`),
`tests/gr00t/policy/test_policy_service.py` (options round-trip).
Docs / client-side: `RTC_implementation.md` (this file), `RTC_README.md`,
`examples/unitree_g1_rtc/` (README + ported copies of the r2d2-side RTC client,
analyzer, and sim runbook — see that README for their canonical locations).

### How to land it on the target machine

Preferred (git): add the private remote and fetch the branch.

```bash
cd <target Isaac-GR00T repo>
git remote add etr git@github.com:praveenkathirvel-etr/Isaac-GR00T-rtc.git
git fetch etr rtc-inference
git checkout rtc-inference        # or: git switch -c rtc-inference etr/rtc-inference
```

If the target repo sits on an OLDER upstream commit and must stay there, cherry-pick
instead of checking out (the branch base would otherwise drag the whole tree to
`9c7e746`):

```bash
git fetch etr rtc-inference
git cherry-pick etr/rtc-inference   # single commit; resolve conflicts if base is old
```

No-GitHub fallback: `~/Projects/r2d2/rtc_port/` on the dev machine holds
`rtc-inference.patch` (apply with `git am -3 rtc-inference.patch`) and
`isaac-gr00t-rtc.bundle` (fetch with `git fetch <bundle> rtc-inference:rtc-inference`).

### Post-port checklist (downstream agent)

1. `uv sync --all-extras` (no new dependencies were added — this just rebuilds the venv
   if the base moved).
2. Verify: `env -u PYTHONPATH uv run python -m pytest tests/gr00t/policy/test_rtc_config.py
   tests/gr00t/policy/test_gr00t_policy_rtc.py tests/gr00t/policy/test_async_chunk_executor.py
   tests/gr00t/eval/test_chunk_metrics.py tests/gr00t/model/test_action_head.py
   tests/gr00t/policy/test_policy_service.py -q --timeout=300` → **106 passed** (5–10 s, CPU).
   (`env -u PYTHONPATH` only matters if ROS is on the PYTHONPATH — see §11 gotcha.)
3. HF backbone: the server constructs the gated `nvidia/Cosmos-Reason2-2B` backbone from
   HF at load time. The target machine needs either an HF token whose account has access,
   or the model already in `~/.cache/huggingface/hub/` (a machine that previously ran this
   checkpoint has it; `HF_HUB_OFFLINE=1` forces cache use).
4. Launch (RTC on): `python gr00t/eval/run_gr00t_server.py --model-path
   <...>/sim_sorting/checkpoint-22000 --embodiment-tag
   unitree_g1_full_body_with_waist_height_nav_cmd --port 5560 --rtc
   --rtc-execution-horizon 8 --rtc-ramp-rate 2.0`. Baseline A/B run = same command minus
   the three `--rtc*` flags. Client must re-query at e=8 (NOT chunk exhaustion) and call
   `reset()` per episode — `examples/unitree_g1_rtc/run_inference_rtc.py` does all of it.
5. Analyze logs: `examples/unitree_g1_rtc/analyze_chunk_boundaries.py --log-dir <csvs>`;
   success criteria table is in `examples/unitree_g1_rtc/RTC_SIM_RUNBOOK.md` (baseline
   e=16: boundary_jump 0.299, ratio 3.42×, momentum 0.325 — expect ↓ ~0.10 / ~1× / ↑).

All measured numbers, parameter derivations, and design rationale: §5.1, §6, §7 above.
`RTC_README.md` is the concept/knob reference for anyone new to the work.

## 14. Sync-RTC A/B results & the async plan (2026-07-18)

Source: the team's `Consolidated_RTC_Findings_and_Async_Plan.docx` (sim, checkpoint-22000,
cube sorting, 20 Hz; 9 RTC episodes analyzed, exp1/2 excluded) + this session's code audit.

### Consolidated numbers

| Metric | Pre-RTC baseline | Sync RTC (tested) | Async RTC (target) |
|---|---|---|---|
| Execution horizon | 16 ticks | 8 ticks | 8-ish (trigger-based) |
| Boundary position jump | 0.326 | **0.120 (−63%)** | ≤ 0.120 (hold) |
| Overall mean jerk (team convention) | 433.5 | 499.7 (+15%) | < 433.5 (must improve) |
| Overall p95 jerk | 1289.6 | 1706.6 (+32%) | < 1289.6 |
| Overall max jerk | 4290.0 | 13625.8 (+218%) | < 4290.0 |
| Raw model latency | ~79 ms | ~117 ms | ~same (RTC compute) |
| Generation gap as % of cycle | 16.2% | 41.7% | n/a (no blocking) |
| idx0/1/2 rushed dwell | present | present (unchanged) | must disappear |
| Stop-and-go pauses | every 16 ticks | every 8 ticks (worse) | none |

### The corrected understanding (important)

1. **RTC's blending verdict: works.** −63% at the seam is the mechanism doing exactly its
   job. Do not touch ramp/overlap semantics based on the jerk regression.
2. **The jerk regression is a TIMING problem.** Sync at e=8 stops the robot twice as
   often, each stop is longer (117 ms + transport), and the post-stop ROS catch-up rushes
   idx0/1/2 dwells. RTC has no lever for this; only the client loop pattern does.
3. **frozen_steps = 0 was CORRECT for the sync client** — nothing executes during a
   blocking wait, so there is nothing to freeze. The earlier recommendation to raise it
   applied to the async pattern only. Under async it becomes load-bearing:
   auto-computed per call as `ceil(EMA latency / 50 ms) + safety_margin` (≈ 5 at the
   measured ~170 ms full gap).
4. **Gripper: no special handling** (asked & answered). RTC blends the whole 28-DoF
   action vector at the timeline level — there is no per-joint-group knob, so hands are
   already covered. Half the rough hand moments are mid-chunk deliberate grasp motion
   (the plan itself, not a seam artifact); smoothing that would make grasps mushy.
   Hardware team owns the mechanical/contact part.
5. **Jerk accounting precision** (doc Part 5): team "jerk" = 3rd derivative of measured
   state positions (normalized units, no hardware-limit grounding yet);
   `chunk_metrics.intra_accel` = 2nd derivative of COMMANDED chunks (deliberate,
   noise-robust) — related, not comparable 1:1. `boundary_jump` matches across both
   analyses (0.299 canonical vs 0.31–0.33 theirs — same phenomenon). Analyzer v2 now
   computes the team-convention state jerk two ways (naive rows-as-uniform vs resampled
   onto the true time grid) — the gap between them measures how much "jerk" is timing
   distortion, which tests explanation (2) offline on the existing sync CSVs.

### Async round — what was built this session (all local, CPU-verified)

- `gr00t/policy/async_chunk_executor.py`: observability attributes (`chunk_id`,
  `last_step_index`, `last_info`, `last_latency_s`, `last_rtc_options`,
  `last_swap_offset`, `chunk_length`) — no behavior change; tests extended (8 passed),
  including a scripted H=16 / lead=8 / 3-tick-latency timeline (trigger at index 8 with
  executed_steps=8 → server overlap 8; swap at offset 4; no replay/skip).
- `examples/unitree_g1_rtc/run_inference_rtc_async.py`: the async client.
  `trigger_lead_steps=8` default (overlap stays in the validated 8–12 band; fallback 6
  if chunk-exhaustion warnings appear), `safety_margin_steps=1`, per-episode
  `start_episode()` (= reset + blocking first chunk), single-step re-wrap (B,D)→(B,1,D)
  for `policy_action_to_control_goal`, CSV schema = old 64 columns + `rtc_applied,
  frozen_steps_used, executed_steps_sent, latency_ema_s, swap_offset`, `new_inference=1`
  marks SWAP ticks. Error path: any background-inference failure stops the loop and
  saves the CSV (never keeps publishing a stale stream).
- Analyzer v2 (`examples/unitree_g1_rtc/analyze_chunk_boundaries.py`): auto-detects
  legacy vs async logs; async metrics computed on the executed tick stream with
  chunk_metrics-matched definitions; dwell-by-ticks-since-swap; stall counter; Part-8
  criteria table; state-jerk section (naive vs resampled). Validated: legacy replay
  reproduces the July-10 baseline exactly (0.29901 / 3.42× / 0.617); async path
  exercised on synthetic logs.
- Expected async timeline @ 20 Hz, H=16, lead 8, gap ~170 ms: trigger at index 8
  (overlap 8, frozen ~5, ramp region ~3), arrival ~3.4 ticks later, swap at offset ~4,
  next trigger ~4 ticks after swap → a query every ~200 ms, robot never stops. GPU
  inference rate rises to ~5 Hz — watch for latency creep; the executor degrades to
  blocking (logged warning) rather than failing if a chunk runs out.

### Open items for the async sim round

- Run 10 episodes with `run_inference_rtc_async.py` (server flags unchanged from sync,
  still `--rtc --rtc-ramp-rate 2.0`; `--rtc-execution-horizon` is ignored when the
  client sends per-call values). Same task/checkpoint/rate.
- Judge with analyzer v2 against Part 8: boundary ≤ 0.120 · jerk (both conventions)
  better than BOTH prior arms · dwell flat · zero stalls · task success not regressed.
- Wanted from the team: the sync-RTC CSVs (offline timing-distortion test + analyzer
  validation on real RTC logs), G1 joint velocity/accel/jerk hardware limits (to ground
  normalized jerk before real-hardware runs), deployment-machine GPU model.
