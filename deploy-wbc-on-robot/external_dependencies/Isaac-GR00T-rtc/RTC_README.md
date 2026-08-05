# Real-Time Chunking (RTC) for GR00T N1.7

**Smooth, jerk-free transitions between action chunks — implemented end-to-end in the
PyTorch inference stack, with synchronous and asynchronous modes.**

> Companion documents:
>
> - `RTC_implementation.md` — engineering passdown: findings, code locations, status log.
> - `getting_started/real_world_deployment.md` — the official deployment guide (RTC
>   section updated by this work).

---

## 1. The problem, in plain language

GR00T does not output one action at a time. Each time you ask it for actions, it outputs
an **action chunk** — a short "movie script" of the next N control steps (N = 16 for our
G1 fine-tune). The robot performs the first few steps of that script, then asks for a new
one, and repeats. This is called **receding-horizon execution**.

The problem: **each new script is written from scratch.** The model starts from pure
random noise and refines it (4 denoising passes) into a chunk, with *total amnesia* about
the plan it produced a moment earlier. Two independently-imagined plans almost never line
up where they meet. The result, at every chunk boundary:

```
without RTC:
                     chunk 1 ends here     chunk 2 starts here
                                     ↓     ↓
 joint position:  ─────────╮               ╭───────────
                            ╰──── ✂ JUMP ✂ ╯     ← position step-change = spike in
 time:            ──────────────────|──────────────       velocity/acceleration = JERK
                              boundary
```

You feel this on the robot as a twitch or snap every time a new chunk arrives (every
~0.3–1 s), plus **stop-and-go** if the robot waits for inference to finish.

## 2. How RTC fixes it, in plain language

Keep the last script. When you executed, say, 8 of its 16 steps, the **remaining 8 steps
still describe where the robot was already heading** — and they cover exactly the same
time window as the *beginning* of the next chunk.

RTC exploits this with three mechanisms applied inside the model's denoising loop:

1. **Inpainting** — instead of starting the new chunk's first 8 steps from random noise,
   we *paste in* the old plan's remaining 8 steps. The model doesn't imagine that region
   from scratch; it starts from what it already committed to.
2. **Frozen prefix** — the first `f` of those steps are locked completely (the model's
   corrections are multiplied by 0). These are steps the robot will physically execute
   *while inference is still running* — changing them would be rewriting the past.
3. **Exponential ramp** — between the frozen steps and the end of the overlap, the
   model's corrections are scaled by a smooth 0 → 1 ramp. It may gradually bend the old
   plan toward what the new camera image demands, but it cannot yank it.

```
with RTC:      new chunk (16 steps)
               ┌─────────────┬───────────────────┬─────────────────────────┐
               │  FROZEN (f) │  RAMP (f..w)      │  FREE (w..16)           │
               │  old plan,  │  old plan, gently │  planned freshly from   │
               │  untouched  │  corrected        │  the new observation    │
               └─────────────┴───────────────────┴─────────────────────────┘
               ↑ starts exactly where the old plan was heading
                 → continuous position AND velocity across the boundary
```

The seam disappears *by construction*: the new chunk's first step equals what the old
plan would have done anyway, and the transition to fresh planning is gradual.

**Crucially, this needs no retraining.** It is purely an inference-time technique — the
denoiser is simply constrained about *where it starts* and *how much it may change*.

## 3. What already existed vs. what this work added

A surprising discovery from dissecting the codebase: **NVIDIA had already implemented the
RTC math** — dormant — inside the action head
(`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:358-394`). It activates only if someone passes the
previous chunk plus 4 options into `model.get_action`. Nothing in the shipped repo ever
did: `Gr00tPolicy` dropped the options on the floor, no component remembered the previous
chunk, no consumer passed the cadence, and there were no tests. NVIDIA's own docs said as
much ("not wired into Gr00tPolicy or the server-client path").

**This work wired it up end-to-end** (zero changes to model weights or model code):


| Missing piece                                   | Now provided by                                                 |
| ----------------------------------------------- | --------------------------------------------------------------- |
| Someone must*remember* the previous chunk       | `Gr00tPolicy` caches it after every inference                   |
| Someone must*feed it back* + the 4 options      | `Gr00tPolicy` injects both on the next call                     |
| Someone must know*how many steps were executed* | client sends it per call, or server default                     |
| Config/CLI surface                              | `RTCConfig` + `--rtc*` flags on server/eval/sim entry points    |
| Episode boundaries                              | `policy.reset()` clears the cache (wired into all loops)        |
| Async inference (no stop-and-go)                | `AsyncChunkExecutor`                                            |
| Proof it helps                                  | `chunk_metrics.py` + A/B logging in `open_loop_eval.py`         |
| Safety rails                                    | relative-action guard, vectorized-env guard, cache invalidation |
| Tests                                           | 5 new/extended CPU test files (all passing)                     |

## 4. Quick start

### Real robot (Unitree G1, policy server)

```bash
# Server side — this alone enables RTC; an unmodified client already benefits
# as long as rtc-execution-horizon matches how often the client queries:
python gr00t/eval/run_gr00t_server.py \
  --model-path <your_checkpoint> \
  --embodiment-tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --rtc --rtc-execution-horizon 8        # ← steps your controller executes per query
```

Client side (your inference script), two optional one-liners for exact behavior:

```python
# per query — tells the server exactly how many steps you executed since last query:
actions, info = policy_client.get_action(obs, {"rtc": {"executed_steps": 8}})
# info["rtc_applied"] tells you whether blending happened on this call

# between episodes — start fresh:
policy_client.reset()
```

### Async mode (removes stop-and-go too)

```python
from gr00t.policy.async_chunk_executor import AsyncChunkExecutor
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host=SERVER_IP, port=5555)          # server started with --rtc
with AsyncChunkExecutor(policy, control_period_s=1 / CONTROL_HZ) as executor:
    executor.start_episode(get_observation())             # blocking first inference
    while running:
        action = executor.get_next_action(get_observation())
        robot.execute(action)                             # one control tick
```

The executor requests the next chunk *before* the current one runs out, keeps the robot
moving on the old plan meanwhile, and swaps to the new chunk at the exact time offset —
no step replayed, none skipped. It measures inference latency itself and sets the frozen
prefix accordingly.

Two integration details worth knowing:

- `trigger_lead_steps` controls both *when* the request goes out and *how much overlap*
  the server has to blend with (overlap = H − executed-at-trigger). The default
  auto-sizing (measured latency + margin) minimizes staleness but yields a small
  overlap; pass `trigger_lead_steps=8` to keep the overlap in the validated 8–12 band
  (this is what the G1 client does).
- For logging, the executor exposes read-only attributes updated as it runs:
  `chunk_id` (increments on every swap — detect boundaries), `last_step_index`,
  `last_info` (the server's `rtc_applied` etc.), `last_latency_s` / `latency_s`
  (raw / EMA), `last_rtc_options` (the `executed_steps`/`frozen_steps` actually sent),
  `last_swap_offset`, and `chunk_length`.

A complete, logging-instrumented robot integration is
`examples/unitree_g1_rtc/run_inference_rtc_async.py` (see the README and runbook in
that directory).

### Sim (G1 locomanip environments)

```bash
python gr00t/eval/rollout_policy.py --env-name gr00tlocomanip_g1/<task> \
  --model-path <ckpt> --n-envs 1 --n-action-steps 8 --seed 42 --rtc
```

### Quantitative A/B (does it actually reduce jerk?)

```bash
# Run twice — once without --rtc, once with — same checkpoint, dataset, seed:
python gr00t/eval/open_loop_eval.py --model-path <ckpt> --dataset-path <ds> \
  --embodiment-tag <tag> --execution-horizon 8 [--rtc]
```

It logs three smoothness metrics per trajectory and averaged:


| Metric           | Meaning                                                                           | RTC should make it |
| ---------------- | --------------------------------------------------------------------------------- | ------------------ |
| `boundary_jump`  | position gap between last executed step of chunk*i* and first step of chunk *i+1* | **smaller** (→ 0) |
| `momentum_shift` | cosine similarity of velocity direction across the boundary                       | **closer to 1**    |
| `intra_accel`    | mean acceleration magnitude (jerk proxy)                                          | **smaller**        |

MSE/MAE against ground truth are also logged — they should stay flat (no accuracy cost).

## 5. The knobs — every variable you can change

### RTC parameters (`RTCConfig` in `gr00t/policy/rtc.py`, exposed as `--rtc-*` flags)


| Variable              | Default         | What it does, in plain language                                                                                                                                                                                                                                                                                          |
| --------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `execution_horizon`   | *required*      | How many steps of each chunk your control loop executes before asking again ("e").**Must match your real query cadence** — it tells the blender how far in time the old plan has advanced. Wrong value = blending against the wrong moment. Clients can override per call via `options={"rtc": {"executed_steps": N}}`. |
| `overlap_steps` ("w") | `H − e` (auto) | How many leading steps of the new chunk are seeded from the old plan. The default is the maximum that lines up in time (with H=16 and e=8, that's 8). Smaller = the model gets freedom sooner (more reactive, slightly less smooth); the code auto-shifts the cached chunk so time still lines up.                       |
| `frozen_steps` ("f")  | `0`             | Steps that are*locked* to the old plan. Use 0 in synchronous loops (robot pauses during inference anyway). In async loops, set to `ceil(inference_latency × control_Hz)` — the steps that will execute while inference runs. `AsyncChunkExecutor` computes this automatically per call.                                |
| `ramp_rate`           | `5.0`           | Steepness of the exponential blend between frozen and free regions. Higher = the model gains authority faster (snappier corrections, slightly less smoothing); lower = gentler handover. 5.0 is a sensible middle; tune only if the overlap feels sluggish (raise) or still slightly steppy (lower).                     |
| `allow_relative`      | `False`         | Safety override for checkpoints that use RELATIVE action groups*and* `use_relative_action=true` (yours doesn't — it's `false`, so this is irrelevant for you). See §8.                                                                                                                                                 |

### Pre-existing knobs that interact with RTC (unchanged, for context)


| Variable                                                     | Where                                                              | Relevance                                                                                                                                                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `action_horizon` (H = 16 in your fine-tune)                  | model`config.json` / `gr00t/configs/model/gr00t_n1d7.py`           | The chunk length. Longer chunks give RTC a bigger overlap window (the deployment doc recommends ≥ 32 for heavy async use; 16 works with modest latency). Changing it requires re-finetuning. |
| `"action"` `delta_indices`                                   | `gr00t/configs/data/embodiment_configs.py` (or your custom config) | The training action window; must equal/underflow the model cap. This is the "unpadded horizon" all RTC math is based on.                                                                      |
| `num_inference_timesteps` (4)                                | model config /`--denoising-steps`                                  | Number of denoising passes. Lower = faster inference = fewer frozen steps needed in async mode.                                                                                               |
| `n_action_steps` / `execution_horizon` / `open_loop_horizon` | eval & robot entry points                                          | The receding-horizon stride each loop already had — with RTC on, this**is** `executed_steps`.                                                                                                |

### Rules the code enforces for you

`0 ≤ frozen ≤ overlap ≤ H − executed_steps`. Out-of-range values are clamped with a
logged warning; if no overlap remains (you executed the whole chunk before re-querying),
RTC quietly skips that call and samples vanilla — **query before the chunk is exhausted**
to get blending.

## 6. Every file added or changed

### New files


| File                                                  | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`gr00t/policy/rtc.py`**                             | The brain of the integration.`RTCConfig` (the knobs above), `resolve_rtc_options()` (validates/clamps and produces the exact 4-key options dict the model's RTC branch demands), `compute_prev_chunk_shift()` + `shift_prev_chunk()` (the time-alignment math for custom overlaps), `merge_rtc_call_options()` (merges server defaults with per-call client overrides). Pure logic, fully unit-tested, no model dependencies.                                                                                                |
| **`gr00t/policy/async_chunk_executor.py`**            | Phase 2.`AsyncChunkExecutor` — a class your robot script drives one control tick at a time. Owns a single background worker thread (the ZMQ client socket is not thread-safe, so exactly one request is ever in flight), triggers re-inference early, swaps chunks at the correct temporal offset, and adapts `frozen_steps` from a running average of measured latency. Also fixes a subtle replay bug in the deployment doc's pseudocode (which restarted new chunks at step 0, re-executing steps that had already run). |
| **`gr00t/eval/chunk_metrics.py`**                     | The three jerk/discontinuity metrics from the deployment doc, previously described only as markdown snippets, now importable and tested:`metric_boundary_jump`, `metric_momentum_shift`, `metric_intra_accel`, plus `compute_chunk_metrics()` (nan-safe aggregate for logging).                                                                                                                                                                                                                                              |
| **`tests/gr00t/policy/test_rtc_config.py`**           | Unit tests for the option/alignment math: constraint matrix, clamping, the shift-slice equivalence (`shifted[H−w:H] == original[e:e+w]`), msgpack-plain types.                                                                                                                                                                                                                                                                                                                                                              |
| **`tests/gr00t/policy/test_gr00t_policy_rtc.py`**     | Mocked-policy tests: first call doesn't inject, second call injects the exact cached tensor with the exact options,`reset()` clears, batch-size change invalidates, per-call disable works, `rtc_config=None` is byte-identical to the old behavior, relative-action guard raises/allows correctly.                                                                                                                                                                                                                          |
| **`tests/gr00t/policy/test_async_chunk_executor.py`** | Executor tests: episode start resets + blocks, trigger fires at the right tick with the right`executed_steps`, swap lands at the co-temporal offset with no step replayed or skipped, latency-derived `frozen_steps` is sent, chunk exhaustion degrades safely to blocking.                                                                                                                                                                                                                                                  |
| **`tests/gr00t/eval/test_chunk_metrics.py`**          | Metric oracles: a globally linear trajectory yields zero acceleration/jump and cosine 1; constructed jumps/reversals yield exact known values.                                                                                                                                                                                                                                                                                                                                                                               |
| **`examples/unitree_g1_rtc/`**                        | The G1 deployment kit: `run_inference_rtc.py` (sync client), `run_inference_rtc_async.py` (async client with full logging), `analyze_chunk_boundaries.py` (A/B analyzer for pre-RTC / sync / async logs), `RTC_SIM_RUNBOOK.md` (launch recipe + success criteria).                                                                                                                                                                                                                                                          |
| **`RTC_implementation.md`**                           | Engineering passdown (findings, code locations, decisions, status).                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **`RTC_README.md`**                                   | This document.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

### Modified files


| File                                            | What changed and why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`gr00t/policy/gr00t_policy.py`**              | The core integration, ~90 lines.`Gr00tPolicy.__init__` gains `rtc_config: RTCConfig | None = None`. Inside `_get_action`: **(a)** after every inference the predicted chunk is cached — specifically `model_pred["action_pred"]`, the *normalized, model-space* tensor captured *before* unit conversion, because that is exactly the representation the model's inpaint slot expects (caching the human-readable actions and re-encoding them would be lossy and wrong); **(b)** on the next call, `_prepare_rtc_injection()` merges config + per-call options, resolves/validates them, and injects the cached chunk into the model's input batch plus the `options=` kwarg — which is all the dormant RTC branch ever needed; **(c)** `reset()` now clears the cache (it was a no-op before); **(d)** `_validate_rtc_compatibility()` refuses unsafe checkpoints (§8); **(e)** the returned `info` dict reports `rtc_applied` / `rtc_overlap_steps` / `rtc_frozen_steps` so you can observe it working. With `rtc_config=None` the model call is exactly the pre-RTC code path. |
| **`gr00t/eval/run_gr00t_server.py`**            | Five new`ServerConfig` flags (`--rtc`, `--rtc-execution-horizon`, `--rtc-overlap-steps`, `--rtc-frozen-steps`, `--rtc-ramp-rate`) which build an `RTCConfig` and hand it to `Gr00tPolicy`. No transport changes were needed — the ZMQ server/client already carried an `options` dict and a `reset` endpoint; they were simply unused.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **`gr00t/eval/rollout_policy.py`**              | Same flags on`RolloutConfig`; the rollout loop forwards `options={"rtc": {"executed_steps": n_action_steps}}` on every query and calls `policy.reset()` when an episode ends. Enforces `--n-envs 1` with RTC: vectorized envs auto-reset individual sub-envs without telling the policy, which would silently corrupt a shared chunk cache.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **`gr00t/eval/open_loop_eval.py`**              | Same flags; resets the policy per trajectory; collects every full predicted chunk and logs the three chunk metrics per trajectory and averaged — this is the RTC on/off A/B harness.`evaluate_single_trajectory` now returns `(mse, mae, chunk_metrics)`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **`getting_started/real_world_deployment.md`**  | The "RTC is not wired in" status paragraph replaced with actual usage instructions;`AsyncChunkExecutor` snippet added under the async pseudocode; the metrics section now points at `gr00t/eval/chunk_metrics.py`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **`tests/gr00t/policy/test_policy_service.py`** | The mock policy now echoes received options, and a new test proves a nested`options["rtc"]={...}` dict survives the msgpack/ZMQ round trip intact.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **`tests/gr00t/model/test_action_head.py`**     | New`TestActionHeadRTC` class — the first tests ever for the model's RTC branch: frozen prefix is *bit-identical* to the old plan's tail; the ramp region can deviate; `options["action_horizon"]` is correctly treated as the **unpadded** horizon (using padded values would silently inpaint garbage padding — this is the one true footgun in the primitive); missing options assert loudly; zero overlap reproduces the vanilla sampler under a fixed seed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

**Not changed:** `gr00t/model/**` (the model and its RTC primitive are untouched),
training code, the TensorRT path (see §9).

## 7. How it works internally (the data flow)

```
control loop (robot / sim / eval)
   │  get_action(obs, options={"rtc": {"executed_steps": e}})
   ▼
PolicyClient ──ZMQ/msgpack──► PolicyServer ──► Gr00tPolicy._get_action
                                                 │
                              ┌──────────────────┤
                              │ 1. merge options with RTCConfig defaults
                              │ 2. resolve: w = H − e (clamped), f, ramp
                              │ 3. inject cached chunk into model inputs
                              ▼
                     model.get_action(inputs, options)
                              │    └─► action head denoising loop:
                              │        noise[:, :w] = prev_chunk[:, H−w : H]   ← inpaint
                              │        vel_gate[:, :f] = 0                     ← freeze
                              │        vel_gate[:, f:w] = 1 − e^(−ramp·t)      ← ramp
                              │        4 × Euler steps with gated velocity
                              ▼
                     action_pred  (normalized, [B, H, D])
                              │
                              ├── cached for the NEXT call  ◄── the "memory"
                              ▼
                     decode to physical units ──► back to client
```

The one alignment rule that makes it correct: **step k of the new chunk describes the
same instant as step e + k of the old chunk** (because e steps passed between the two
inferences). The model's inpaint reads the *last w steps* of whatever you hand it, so
handing it the cached chunk with `w = H − e` makes that slice exactly the unexecuted
remainder `prev[e:H]` — perfectly co-temporal with the new chunk's start. For smaller
overlaps the cache is right-shifted first so the slice still lands on `prev[e : e+w]`.

## 8. Limitations and safety rails

- **One control loop per policy/server.** The chunk cache is a single stream of memory.
  Two robots querying one server would blend each other's plans. (The ZMQ REQ/REP
  transport already serializes requests; this is documented, not detected.)
- **`--n-envs 1` in sim** — enforced, for the cache-corruption reason above.
- **Relative action representations.** If a checkpoint stores actions as *offsets from
  the current robot state* (`use_relative_action=true` + RELATIVE groups), a cached chunk
  is anchored to the *old* state — inpainting it applies a stale reference frame, off by
  however far the robot moved between queries. The policy refuses to start in that
  configuration unless you pass `allow_relative=True`. **Your G1 checkpoint has
  `use_relative_action=false`, so this doesn't affect you** — actions are absolute in
  normalized space and inpaint cleanly.
- **The cadence must be honest.** If you tell the server `executed_steps=8` but your
  controller actually executed 12, the blend is misaligned by 4 steps. Measure it, or
  send it per call.
- **First call of every episode is vanilla** (there is nothing to blend with), and RTC
  auto-skips whenever no overlap remains. Call `reset()` between episodes.
- **TensorRT path not yet covered.** The TRT deployment reimplements the denoising loop
  in plain Python without the RTC branch (`scripts/deployment/trt_model_forward.py`) —
  a straightforward follow-up port, no ONNX re-export needed.

## 9. Implementing this on another system (porting guide)

RTC is model-agnostic for any **diffusion or flow-matching action-chunk policy** — any
model that generates its chunk by iteratively refining a noise tensor of shape
`(batch, horizon, action_dim)`. Here is the complete recipe:

**Step 1 — open up the denoising loop.** Find the sampling loop (the `for` loop that
integrates noise into actions). You need two hook points:

```python
noise = randn(B, H, D)
# HOOK 1 (inpaint, once, before the loop):
if prev_chunk is not None:
    noise[:, :w] = prev_chunk[:, H_prev - w : H_prev]     # paste old plan's remainder
    gate = ones(B, H, D)                                  # per-step velocity gate
    gate[:, :f] = 0.0                                     # frozen
    gate[:, f:w] = normalized(1 - exp(-r * linspace(0, 1, w - f)))  # ramp
for t in denoising_steps:
    v = model(noise, t, conditioning)
    # HOOK 2 (gate, every step):
    noise = noise + dt * v * gate                         # gated Euler update
return noise
```

That's the entire model-side change (~15 lines). For DDPM-style diffusion, gate the
predicted update the same way; for a hard variant you can re-inpaint the frozen region
after every step instead of gating.

**Step 2 — cache the previous chunk in the right space.** Store the model's raw output
tensor *before* any un-normalization / coordinate conversion, and feed exactly that back.
Never cache the physical-units actions and re-encode them — normalization parameters,
padding, and reference frames will bite you.

**Step 3 — track the timeline.** Your control loop knows `e` = steps executed since the
last inference. The only alignment law: *new chunk step k ≡ old chunk step e + k*. So the
pasted region must be the old chunk starting at index `e`. Simplest correct choice:
`w = H − e` and paste `prev[e:H]`. Validate `0 ≤ f ≤ w ≤ H − e`.

**Step 4 — reset on episode boundaries** (clear the cache), and skip RTC whenever there
is no cache or no overlap (first call, chunk fully consumed, batch size changed).

**Step 5 (async, optional) — trigger early, swap at the offset.** Request the next chunk
when `remaining_steps ≤ expected_latency_steps + margin`, keep executing the old chunk
while waiting, and when the new chunk arrives after k ticks, continue from its index k
(not 0 — index 0 is a step that already happened). Set `f = expected_latency_steps` so
the model cannot alter the steps that execute during inference. Measure latency with a
running average rather than hard-coding it.

Pitfalls checklist (each one is unit-tested in this repo, copy the tests):

- padded vs. actual horizon (paste using the *real* horizon, not the padded one);
- normalized vs. physical action space (Step 2);
- relative/state-anchored action encodings (stale reference frame — reframe or refuse);
- replaying frozen steps after the swap (Step 5's off-by-k);
- stateful cache vs. parallel/auto-resetting environments.

## 10. Verifying it works

```bash
# All CPU tests (628 + 123 passing as of this work):
env -u PYTHONPATH uv run python -m pytest tests/ -m "not gpu" --timeout=300

# Just the RTC-related suites:
env -u PYTHONPATH uv run python -m pytest tests/gr00t/policy/ \
  tests/gr00t/eval/test_chunk_metrics.py tests/gr00t/model/test_action_head.py -m "not gpu"
```

(The `env -u PYTHONPATH` strips ROS Humble from the path; its pytest plugin otherwise
crashes collection on this machine.)

Then the A/B in §4: expect `boundary_jump` ↓, `momentum_shift` → 1, `intra_accel` ↓, with
MSE/MAE unchanged — and, on hardware, visibly smoother joints at every re-plan.
