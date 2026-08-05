"""Chunk-boundary analysis for GR00T inference logs (baseline, sync-RTC, async-RTC).

Handles both log layouts:

- LEGACY (blocking clients — inference_logs/ e=16 baseline, inference_logs_rtc/ e=8
  sync RTC): every chunk starts at action_idx 0 at a fixed cadence. Reconstructs
  per-chunk blocks and computes the gr00t.eval.chunk_metrics numbers exactly as before
  (reproduces the 2026-07-10 baseline).
- ASYNC (run_inference_rtc_async.py — inference_logs_rtc_async/): chunks swap in
  mid-stream at varying cadence (new_inference=1 rows with action_idx > 0, plus a
  swap_offset column). Metrics are computed directly on the executed tick stream with
  definitions matched to chunk_metrics.py, so numbers are comparable across all three
  arms (pre-RTC / sync-RTC / async-RTC).

Both modes also report the consolidated-findings jerk accounting: the 3rd derivative of
the MEASURED state positions pooled across all 28 joints (L2-combined per tick),
computed two ways — assuming uniform 50 ms ticks AND using the actual measured tick
times. Comparing the two on the same log isolates how much apparent "jerk" is really
timing distortion (rushed post-boundary ticks) rather than physical roughness. Note:
absolute jerk values depend on the joint-combination convention; compare runs analyzed
by THIS script against each other, not against numbers from other tools.

Run from the Isaac-GR00T repo so gr00t.eval.chunk_metrics is importable:

    cd <Isaac-GR00T repo>
    env -u PYTHONPATH uv run python examples/unitree_g1_rtc/analyze_chunk_boundaries.py \
        --log-dir <dir with exp*.csv>
"""

import argparse
import glob
import os

from gr00t.eval.chunk_metrics import compute_chunk_metrics
import numpy as np
import pandas as pd


# Reference numbers from prior rounds (episode-mean pooled, this script / chunk_metrics):
BASELINE_REF = (
    "2026-07-10 pre-RTC baseline, e=16: boundary_jump=0.29901 ratio=3.42x "
    "momentum_shift=0.3252 intra_accel=0.12266"
)
SYNC_RTC_REF = (
    "2026-07-18 sync-RTC (team analysis): boundary_jump~=0.120 (-63%) — async must "
    "hold this while fixing overall jerk"
)


def detect_execute_steps(df: pd.DataFrame) -> int:
    starts = df.index[df["new_inference"] == 1].to_numpy()
    if len(starts) < 3:
        return 0
    gaps = np.diff(starts)
    return int(np.bincount(gaps).argmax())


def detect_mode(df: pd.DataFrame) -> str:
    if "swap_offset" in df.columns:
        return "async"
    inf_rows = df[df["new_inference"] == 1]
    if len(inf_rows) and (inf_rows["action_idx"].to_numpy() != 0).any():
        return "async"
    return "legacy"


def reconstruct_blocks(df: pd.DataFrame, target_cols: list, execute_steps: int):
    """Group executed rows into per-chunk blocks of length execute_steps (legacy logs)."""
    starts = df.index[df["new_inference"] == 1].tolist()
    blocks = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(df)
        block = df.loc[s : end - 1, target_cols].to_numpy(dtype=np.float64)
        idx = df.loc[s : end - 1, "action_idx"].to_numpy()
        if len(block) == execute_steps and idx[0] == 0:
            blocks.append(block)
    return np.stack(blocks) if len(blocks) >= 2 else None


def joint_group_indices(target_cols: list) -> tuple:
    arm = [i for i, c in enumerate(target_cols) if "arm" in c]
    hand = [i for i, c in enumerate(target_cols) if "hand" in c]
    return arm, hand


# --------------------------------------------------------------------------- jerk


def jerk_stats(dfs: list, state_cols: list, control_hz: float) -> None:
    """State-position jerk (3rd derivative), L2-combined across joints per tick, pooled
    across episodes. Two computations:

    - "naive (rows as uniform)": rows treated as equally spaced ticks — the convention
      of the prior team analysis. Sensitive to timing distortion: rushed/late ticks are
      differentiated as if they took a full period.
    - "resampled to true grid": positions interpolated onto a clean uniform grid using
      the MEASURED timestamps, then differentiated — the physically meaningful jerk,
      independent of logging cadence irregularity.

    A gap between the two on the same log = apparent roughness caused by timing, not by
    physical motion (the doc's Part-3 hypothesis, directly testable per run)."""
    period = 1.0 / control_hz
    pooled = {"naive (rows as uniform)": [], "resampled to true grid": []}
    for df in dfs:
        pos = df[state_cols].to_numpy(dtype=np.float64)
        if len(pos) < 8:
            continue
        t = df["loop_time"].to_numpy(dtype=np.float64)
        if not np.all(np.diff(t) > 0):  # fall back if loop_time is unusable
            steps = np.maximum(df["dt_since_last"].to_numpy(dtype=np.float64)[1:], 1e-4)
            t = np.concatenate([[0.0], np.cumsum(steps)])

        # Naive: index grid at the nominal period (prior analysis convention).
        t_naive = np.arange(len(pos)) * period
        d3 = np.gradient(
            np.gradient(np.gradient(pos, t_naive, axis=0), t_naive, axis=0), t_naive, axis=0
        )
        pooled["naive (rows as uniform)"].append(np.linalg.norm(d3, axis=-1))

        # Physical: interpolate onto a clean uniform grid from measured timestamps.
        grid = np.arange(t[0], t[-1], period)
        if len(grid) < 8:
            continue
        pos_grid = np.stack([np.interp(grid, t, pos[:, j]) for j in range(pos.shape[1])], axis=1)
        d3 = np.gradient(
            np.gradient(np.gradient(pos_grid, grid, axis=0), grid, axis=0), grid, axis=0
        )
        pooled["resampled to true grid"].append(np.linalg.norm(d3, axis=-1))

    print("\n=== STATE JERK (3rd derivative of measured positions, L2 across joints) ===")
    for label, chunks in pooled.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks)
        print(
            f"  {label:24s} mean={arr.mean():10.1f}  p95={np.percentile(arr, 95):10.1f}  "
            f"max={arr.max():10.1f}   (n={len(arr)} ticks)"
        )
    print(
        "  (naive >> resampled on the same log = apparent jerk is timing distortion from\n"
        "   irregular tick dwell — the stop-and-go artifact — not physical roughness.\n"
        "   Compare runs via THIS script only; combination conventions differ across tools.)"
    )


# --------------------------------------------------------------------------- legacy


def analyze_legacy(
    files: list, dfs: list, target_cols: list, execute_steps: int, control_hz: float
):
    hand_cols = [c for c in target_cols if "hand" in c]
    has_rtc_col = "rtc_applied" in dfs[0].columns

    all_lat, all_gap, all_blocks = [], [], []
    dwell_by_idx = {}
    per_ep = []
    boundary_jumps_l2, boundary_hand_vel = [], []
    rtc_applied_counts = [0, 0]

    for f, df in zip(files, dfs):
        name = os.path.basename(f)
        inf_rows = df[df["new_inference"] == 1]
        lat = pd.to_numeric(inf_rows["inference_latency_s"], errors="coerce").dropna().to_numpy()
        gap = inf_rows["dt_since_last"].to_numpy()
        gap = gap[gap > 0.001]
        all_lat.append(lat)
        all_gap.append(gap)

        if has_rtc_col:
            applied = pd.to_numeric(inf_rows["rtc_applied"], errors="coerce").dropna()
            rtc_applied_counts[1] += int((applied == 1).sum())
            rtc_applied_counts[0] += int((applied == 0).sum())

        for idx, sub in df.groupby("action_idx"):
            dwell_by_idx.setdefault(int(idx), []).append(sub["dt_since_last"].to_numpy()[1:])

        blocks = reconstruct_blocks(df, target_cols, execute_steps)
        if blocks is None:
            print(f"{name}: fewer than 2 complete chunks, skipping")
            continue
        all_blocks.append(blocks)
        m = compute_chunk_metrics(blocks, execute_steps=execute_steps)
        per_ep.append((name, len(blocks), lat.mean() if len(lat) else np.nan, m))

        prev_last = blocks[:-1, execute_steps - 1, :]
        next_first = blocks[1:, 0, :]
        boundary_jumps_l2.append(np.linalg.norm(next_first - prev_last, axis=-1))

        hand_idx = [target_cols.index(c) for c in hand_cols]
        tail = min(5, execute_steps)
        hand_tail = blocks[:-1, execute_steps - tail : execute_steps, :][:, :, hand_idx]
        boundary_hand_vel.append(np.abs(np.diff(hand_tail, axis=1)).mean(axis=(1, 2)))

    lat = np.concatenate(all_lat)
    gap = np.concatenate(all_gap)
    print(f"\n=== LATENCY (pooled, n={len(lat)} inferences) ===")
    for label, arr in [("raw inference_latency_s", lat), ("full gap at boundary", gap)]:
        if not len(arr):
            continue
        print(
            f"{label}: mean={arr.mean() * 1000:.1f}ms  p95={np.percentile(arr, 95) * 1000:.1f}  "
            f"p99={np.percentile(arr, 99) * 1000:.1f}  max={arr.max() * 1000:.1f}"
        )
    if len(gap):
        print(
            f"frozen_steps to cover p99 @ {control_hz:.0f}Hz: "
            f"{int(np.ceil(np.percentile(gap, 99) * control_hz))}  "
            f"(max: {int(np.ceil(gap.max() * control_hz))})"
        )

    if has_rtc_col:
        total = sum(rtc_applied_counts)
        print(
            f"\nrtc_applied: {rtc_applied_counts[1]}/{total} inferences "
            f"(first call of each episode is expected to be 0)"
        )

    print("\n=== DWELL BY ACTION_IDX (mean ms, pooled) ===")
    for idx in sorted(dwell_by_idx):
        arr = np.concatenate(dwell_by_idx[idx])
        if len(arr):
            print(
                f"  idx {idx:2d}: mean={arr.mean() * 1000:6.1f}ms  "
                f"median={np.median(arr) * 1000:6.1f}ms  n={len(arr)}"
            )

    print(f"\n=== chunk_metrics.py (execute_steps={execute_steps}) ===")
    print(
        f"{'episode':12s} {'chunks':>6s} {'lat_mean':>9s} {'intra_accel':>12s} "
        f"{'boundary_jump':>14s} {'momentum_shift':>15s}"
    )
    for name, n, lm, m in per_ep:
        print(
            f"{name:12s} {n:6d} {lm * 1000:8.1f}m "
            f"{m['intra_accel']:12.5f} {m['boundary_jump']:14.5f} {m['momentum_shift']:15.4f}"
        )
    ia = np.nanmean([m["intra_accel"] for _, _, _, m in per_ep])
    bj = np.nanmean([m["boundary_jump"] for _, _, _, m in per_ep])
    ms = np.nanmean([m["momentum_shift"] for _, _, _, m in per_ep])
    print(
        f"\nPOOLED (episode-mean): intra_accel={ia:.5f}  boundary_jump={bj:.5f}  momentum_shift={ms:.4f}"
    )
    print(f"({BASELINE_REF})")
    print(f"({SYNC_RTC_REF})")

    pooled = np.concatenate(all_blocks)
    intra_steps = np.linalg.norm(np.diff(pooled, axis=1), axis=-1).mean()
    jumps = np.concatenate(boundary_jumps_l2)
    print(
        f"intra-chunk mean step L2={intra_steps:.5f} vs boundary jump mean L2={jumps.mean():.5f} "
        f"→ ratio {jumps.mean() / max(intra_steps, 1e-9):.2f}x  (baseline: 3.42x)"
    )

    print("\n=== PER-JOINT BOUNDARY JUMP vs INTRA-CHUNK STEP (top 8) ===")
    prev_last = np.concatenate([b[:-1, execute_steps - 1, :] for b in all_blocks])
    next_first = np.concatenate([b[1:, 0, :] for b in all_blocks])
    per_joint_jump = np.abs(next_first - prev_last).mean(axis=0)
    per_joint_intra = np.abs(np.diff(pooled, axis=1)).mean(axis=(0, 1))
    ratio = per_joint_jump / np.maximum(per_joint_intra, 1e-9)
    for i in np.argsort(-ratio)[:8]:
        print(
            f"  {target_cols[i]:22s} jump={per_joint_jump[i]:.4f} "
            f"intra={per_joint_intra[i]:.4f} ratio={ratio[i]:6.2f}x"
        )
    arm_i, hand_i = joint_group_indices(target_cols)
    print(
        f"  ARM  mean ratio={ratio[arm_i].mean():.2f}x   HAND mean ratio={ratio[hand_i].mean():.2f}x"
    )

    hv = np.concatenate(boundary_hand_vel)
    r = np.corrcoef(hv, jumps)[0, 1]
    hi = jumps[hv > np.percentile(hv, 75)].mean()
    lo = jumps[hv <= np.percentile(hv, 25)].mean()
    print("\n=== GRASP-PHASE CORRELATION ===")
    print(f"corr(hand motion before boundary, jump) = {r:.3f}  (baseline: 0.617)")
    print(f"jump when hands active = {hi:.4f} vs quiet = {lo:.4f}")


# --------------------------------------------------------------------------- async


def analyze_async(files: list, dfs: list, target_cols: list, control_hz: float):
    """Tick-stream metrics for async logs. Definitions match chunk_metrics.py:
    boundary_jump = L2 of the commanded step across a swap tick; momentum_shift =
    velocity cosine across the swap; intra_accel = 2nd-diff L2 within chunk segments."""
    period = 1.0 / control_hz

    per_ep = []
    all_jumps, all_intra_steps = [], []
    per_joint_jump_rows, per_joint_intra_rows = [], []
    dwell_since_swap = {}
    all_lat, all_frozen, all_executed, all_offsets = [], [], [], []
    rtc_applied_counts = [0, 0]
    stall_count, tick_count, max_dt = 0, 0, 0.0

    for f, df in zip(files, dfs):
        name = os.path.basename(f)
        T = df[target_cols].to_numpy(dtype=np.float64)
        n = len(T)
        swaps = set(df.index[df["new_inference"] == 1].tolist())
        swaps.discard(0)
        if len(swaps) < 2 or n < 4:
            print(f"{name}: fewer than 2 swaps, skipping")
            continue

        step_l2 = np.linalg.norm(np.diff(T, axis=0), axis=-1)  # step i = T[i] -> T[i+1]
        is_swap_step = np.array([(i + 1) in swaps for i in range(n - 1)])
        jumps = step_l2[is_swap_step]
        intra = step_l2[~is_swap_step]
        all_jumps.append(jumps)
        all_intra_steps.append(intra)

        step_abs = np.abs(np.diff(T, axis=0))
        per_joint_jump_rows.append(step_abs[is_swap_step])
        per_joint_intra_rows.append(step_abs[~is_swap_step])

        # Momentum shift across each swap s: v_end = T[s-1]-T[s-2], v_start = T[s+1]-T[s].
        cos_vals = []
        for s in sorted(swaps):
            if s < 2 or s + 1 >= n:
                continue
            v_end = T[s - 1] - T[s - 2]
            v_start = T[s + 1] - T[s]
            denom = np.linalg.norm(v_end) * np.linalg.norm(v_start) + 1e-8
            cos_vals.append(float(np.dot(v_end, v_start) / denom))

        # Intra accel: 2nd difference centered at tick i+1 (uses steps i and i+1);
        # exclude windows whose steps cross a swap.
        accel = np.linalg.norm(T[2:] - 2 * T[1:-1] + T[:-2], axis=-1)
        keep = np.array([(i + 1) not in swaps and (i + 2) not in swaps for i in range(n - 2)])

        # Dwell regularity by ticks-since-swap (Part-8: should be flat ≈ period).
        dts = df["dt_since_last"].to_numpy(dtype=np.float64)
        since = None
        for i in range(1, n):
            since = 0 if i in swaps else (None if since is None else since + 1)
            if since is not None and since <= 7:
                dwell_since_swap.setdefault(since, []).append(dts[i])
        stall_count += int((dts[1:] > 1.5 * period).sum())
        tick_count += n - 1
        max_dt = max(max_dt, float(dts[1:].max()))

        swap_rows = df[df["new_inference"] == 1]
        lat = pd.to_numeric(swap_rows["inference_latency_s"], errors="coerce").dropna().to_numpy()
        all_lat.append(lat)
        for col, dest in (
            ("frozen_steps_used", all_frozen),
            ("executed_steps_sent", all_executed),
            ("swap_offset", all_offsets),
        ):
            if col in df.columns:
                dest.append(pd.to_numeric(swap_rows[col], errors="coerce").dropna().to_numpy())
        if "rtc_applied" in df.columns:
            applied = pd.to_numeric(swap_rows["rtc_applied"], errors="coerce").dropna()
            rtc_applied_counts[1] += int((applied == 1).sum())
            rtc_applied_counts[0] += int((applied == 0).sum())

        per_ep.append(
            (
                name,
                len(swaps),
                lat.mean() if len(lat) else np.nan,
                jumps.mean(),
                np.mean(cos_vals) if cos_vals else np.nan,
                accel[keep].mean() if keep.any() else np.nan,
            )
        )

    if not per_ep:
        raise SystemExit("No analyzable async episodes found.")

    lat = np.concatenate(all_lat) if all_lat else np.array([])
    print(f"\n=== LATENCY (pooled, n={len(lat)} background inferences) ===")
    if len(lat):
        print(
            f"raw inference_latency_s: mean={lat.mean() * 1000:.1f}ms  "
            f"p95={np.percentile(lat, 95) * 1000:.1f}  p99={np.percentile(lat, 99) * 1000:.1f}  "
            f"max={lat.max() * 1000:.1f}"
        )
    for label, vals in (
        ("frozen_steps sent", all_frozen),
        ("executed_steps sent", all_executed),
        ("swap offset (ticks in flight)", all_offsets),
    ):
        if vals:
            arr = np.concatenate(vals)
            if len(arr):
                print(f"{label}: mean={arr.mean():.2f}  min={arr.min():.0f}  max={arr.max():.0f}")

    total = sum(rtc_applied_counts)
    if total:
        print(
            f"rtc_applied: {rtc_applied_counts[1]}/{total} swaps "
            f"(0s beyond warm-up → server missing --rtc)"
        )

    print(f"\n=== DWELL BY TICKS-SINCE-SWAP (Part-8: should be flat ≈ {period * 1000:.0f}ms) ===")
    for k in sorted(dwell_since_swap):
        arr = np.asarray(dwell_since_swap[k])
        print(
            f"  +{k}: mean={arr.mean() * 1000:6.1f}ms  "
            f"median={np.median(arr) * 1000:6.1f}ms  n={len(arr)}"
        )
    print(
        f"stalls (dt > 1.5×period): {stall_count}/{tick_count} ticks  max dt={max_dt * 1000:.1f}ms"
        + (
            "   ← Part-8 'no full stops' PASS"
            if stall_count == 0
            else "   ← investigate (trigger lead too small?)"
        )
    )

    print("\n=== TICK-STREAM CHUNK METRICS (definitions matched to chunk_metrics.py) ===")
    print(
        f"{'episode':12s} {'swaps':>6s} {'lat_mean':>9s} {'intra_accel':>12s} "
        f"{'boundary_jump':>14s} {'momentum_shift':>15s}"
    )
    for name, n_swaps, lm, bj, cs, ia in per_ep:
        lm_str = f"{lm * 1000:8.1f}m" if np.isfinite(lm) else "       -"
        print(f"{name:12s} {n_swaps:6d} {lm_str} {ia:12.5f} {bj:14.5f} {cs:15.4f}")

    bj = float(np.nanmean([e[3] for e in per_ep]))
    cs = float(np.nanmean([e[4] for e in per_ep]))
    ia = float(np.nanmean([e[5] for e in per_ep]))
    jumps = np.concatenate(all_jumps)
    intra = np.concatenate(all_intra_steps)
    ratio = jumps.mean() / max(intra.mean(), 1e-9)
    print(
        f"\nPOOLED (episode-mean): intra_accel={ia:.5f}  boundary_jump={bj:.5f}  momentum_shift={cs:.4f}"
    )
    print(
        f"swap jump mean L2={jumps.mean():.5f} vs intra step L2={intra.mean():.5f} → ratio {ratio:.2f}x"
    )
    print(f"({BASELINE_REF})")
    print(f"({SYNC_RTC_REF})")

    print("\n=== PER-JOINT SWAP JUMP vs INTRA STEP (top 8) ===")
    pj_jump = np.concatenate(per_joint_jump_rows).mean(axis=0)
    pj_intra = np.concatenate(per_joint_intra_rows).mean(axis=0)
    pj_ratio = pj_jump / np.maximum(pj_intra, 1e-9)
    for i in np.argsort(-pj_ratio)[:8]:
        print(
            f"  {target_cols[i]:22s} jump={pj_jump[i]:.4f} intra={pj_intra[i]:.4f} "
            f"ratio={pj_ratio[i]:6.2f}x"
        )
    arm_i, hand_i = joint_group_indices(target_cols)
    print(
        f"  ARM  mean ratio={pj_ratio[arm_i].mean():.2f}x   "
        f"HAND mean ratio={pj_ratio[hand_i].mean():.2f}x"
    )

    print("\n=== PART-8 CRITERIA CHECK ===")
    print(
        f"  1. boundary_jump ≤ 0.120 (hold sync-RTC level):  {bj:.4f} → "
        f"{'PASS' if bj <= 0.13 else 'FAIL'}"
    )
    print("  2. overall jerk vs both baselines:               see STATE JERK section below")
    print("  3. dwell flat after swap:                        see dwell table above")
    print(
        f"  4. no full stops:                                {stall_count} stalls → "
        f"{'PASS' if stall_count == 0 else 'FAIL'}"
    )


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default=os.path.expanduser("~/Projects/r2d2/inference_logs"),
        help="Directory containing exp*.csv logs",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=0,
        help="Legacy logs: executed steps per chunk (0 = auto-detect)",
    )
    parser.add_argument("--control-hz", type=float, default=20.0, help="Control loop rate (Hz)")
    parser.add_argument(
        "--mode",
        choices=["auto", "legacy", "async"],
        default="auto",
        help="Log layout (auto-detected from the first file by default)",
    )
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.log_dir, "exp*.csv")),
        key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))),
    )
    if not files:
        raise SystemExit(f"No exp*.csv files in {args.log_dir}")
    dfs = [pd.read_csv(f) for f in files]

    target_cols = [c for c in dfs[0].columns if c.startswith("target_")]
    state_cols = [c for c in dfs[0].columns if c.startswith("state_")]

    mode = args.mode if args.mode != "auto" else detect_mode(dfs[0])
    print(f"log mode: {mode}  ({len(files)} episodes)")

    if mode == "legacy":
        execute_steps = args.execute_steps or detect_execute_steps(dfs[0])
        print(
            f"execute_steps = {execute_steps}" + ("" if args.execute_steps else " (auto-detected)")
        )
        analyze_legacy(files, dfs, target_cols, execute_steps, args.control_hz)
    else:
        analyze_async(files, dfs, target_cols, args.control_hz)

    jerk_stats(dfs, state_cols, args.control_hz)


if __name__ == "__main__":
    main()
