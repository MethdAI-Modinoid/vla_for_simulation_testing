#!/usr/bin/env python3
"""
analyze_modes.py — compare smoothing techniques from run_inference_recorder.py CSVs.

For each recording it measures the PUBLISHED command trajectory (what the robot
actually tracks) and reports the smoothness/fidelity tradeoff:

  SMOOTHNESS (lower = better):
    vel      mean/p95/max per-tick command change (rad/tick)  -> velocity
    jerk     mean/max 3rd-difference (rad/tick^3)              -> the "jerk" itself
    boundary vs interior per-tick step (the a16 -> new_a1 seam)

  FIDELITY (how faithful the smoothed command stays to the raw VLA intent):
    lag      mean per-tick max|pub - raw| (rad)  -> smoothing delay; high = sluggish
    reach    pub travel / raw travel per joint    -> <1 means motion was shrunk
             (a grasp that no longer closes shows up as low hand-reach)

Arm and hand are reported separately, because the hand carries the big ~1.5 rad
grasp steps and needs a looser clamp than the arm.

Usage:
    python analyze_modes.py recordings/
    python analyze_modes.py recordings/raw_*.csv recordings/smooth_*.csv
"""

import glob
import os
import sys

import numpy as np

GROUPS = [("left_arm", 7), ("left_hand", 7), ("right_arm", 7), ("right_hand", 7)]
DIM = 28
ARM_IDX = list(range(0, 7)) + list(range(14, 21))
HAND_IDX = list(range(7, 14)) + list(range(21, 28))


def _cols(prefix):
    out = []
    for g, n in GROUPS:
        out += [f"{prefix}_{g}_{j}" for j in range(n)]
    return out


def load_csv(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 5:
        return None

    def ff(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return 0.0

    def mat(prefix):
        keys = _cols(prefix)
        return np.array([[ff(r.get(k, 0.0)) for k in keys] for r in rows])

    rtc = np.array([int(ff(r.get("rtc_applied", 0))) for r in rows])
    base_mode = rows[0].get("mode", os.path.basename(path).split("_")[0])
    label = base_mode + ("+rtc" if rtc.max() > 0 else "")
    return {
        "mode": label,
        "path": os.path.basename(path),
        "raw": mat("raw"),
        "pub": mat("pub"),
        "state": mat("state"),
        "new_inf": np.array([int(ff(r.get("new_inference", 0))) for r in rows]),
        "dt": np.array([ff(r.get("dt_since_last", 0)) for r in rows]),
        "lat": np.array([ff(r.get("inference_latency_s", 0)) for r in rows]),
        "rtc_frac": float(rtc.mean()),
    }


def metrics(d):
    pub, raw = d["pub"], d["raw"]
    n = len(pub)
    m = {"mode": d["mode"], "path": d["path"], "n": n}

    def sub(a, idx):
        return a[:, idx]

    for name, idx in [("arm", ARM_IDX), ("hand", HAND_IDX)]:
        p = sub(pub, idx)
        r = sub(raw, idx)
        vel = np.abs(np.diff(p, axis=0)).max(axis=1) if n > 1 else np.zeros(1)
        jerk = np.abs(np.diff(p, n=3, axis=0)).max(axis=1) if n > 3 else np.zeros(1)
        m[f"{name}_vel_mean"] = float(vel.mean())
        m[f"{name}_vel_p95"] = float(np.percentile(vel, 95))
        m[f"{name}_vel_max"] = float(vel.max())
        m[f"{name}_jerk_mean"] = float(jerk.mean())
        m[f"{name}_jerk_max"] = float(jerk.max())
        # fidelity
        m[f"{name}_lag"] = float(np.abs(p - r).max(axis=1).mean())
        raw_travel = (r.max(axis=0) - r.min(axis=0))
        pub_travel = (p.max(axis=0) - p.min(axis=0))
        good = raw_travel > 1e-3
        m[f"{name}_reach"] = float(np.mean(pub_travel[good] / raw_travel[good])) if good.any() else 1.0

    # boundary vs interior on the full published vector
    if n > 1:
        step = np.abs(np.diff(pub, axis=0)).max(axis=1)
        bmask = d["new_inf"][1:] == 1
        m["boundary_step"] = float(step[bmask].mean()) if bmask.any() else 0.0
        m["interior_step"] = float(step[~bmask].mean()) if (~bmask).any() else 0.0
        m["boundary_ratio"] = m["boundary_step"] / max(m["interior_step"], 1e-6)
    m["dt_mean_ms"] = float(d["dt"][d["dt"] > 0].mean() * 1000) if (d["dt"] > 0).any() else 0.0
    m["dt_max_ms"] = float(d["dt"].max() * 1000)
    m["lat_max_ms"] = float(d["lat"].max() * 1000)
    m["rtc_frac"] = d.get("rtc_frac", 0.0)
    return m


def gather(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, "*.csv")))
        else:
            paths += sorted(glob.glob(a))
    return paths


def main():
    args = sys.argv[1:] or ["recordings/"]
    paths = gather(args)
    if not paths:
        print("No CSVs found. Record some runs first with run_inference_recorder.py.")
        return

    results = [metrics(d) for d in (load_csv(p) for p in paths) if d]
    if not results:
        print("No usable recordings.")
        return

    print("\n" + "=" * 118)
    print("SMOOTHING TECHNIQUE COMPARISON  (published command; lower vel/jerk = smoother, lag~0 & reach~1 = faithful)")
    print("=" * 118)
    hdr = (f"{'mode':18s} {'n':>5s} | {'ARM vel(mn/p95/mx)':>20s} {'ARM jerk(mn/mx)':>16s} "
           f"{'lag':>6s} {'reach':>6s} | {'HAND jerk(mn/mx)':>16s} {'lag':>6s} {'reach':>6s} | "
           f"{'bnd/int':>8s}")
    print(hdr)
    print("-" * 118)
    for m in results:
        print(f"{m['mode']:18s} {m['n']:5d} | "
              f"{m['arm_vel_mean']:6.3f}/{m['arm_vel_p95']:5.3f}/{m['arm_vel_max']:5.3f} "
              f"{m['arm_jerk_mean']:7.3f}/{m['arm_jerk_max']:6.3f} "
              f"{m['arm_lag']:6.3f} {m['arm_reach']:6.2f} | "
              f"{m['hand_jerk_mean']:7.3f}/{m['hand_jerk_max']:6.3f} "
              f"{m['hand_lag']:6.3f} {m['hand_reach']:6.2f} | "
              f"{m.get('boundary_ratio', 0):6.1f}x")
    print("-" * 118)
    print("Timing:")
    for m in results:
        print(f"  {m['mode']:18s} dt_mean={m['dt_mean_ms']:6.1f}ms dt_max={m['dt_max_ms']:7.1f}ms "
              f"inference_max={m['lat_max_ms']:7.1f}ms  rtc_applied={100*m['rtc_frac']:.0f}%")

    # ── Recommendation: smoothest arm among modes that stay faithful ──────────
    # Faithful = arm lag < 0.05 rad, arm reach > 0.85, hand reach > 0.80 (grasp
    # still closes). Among those, pick lowest arm jerk_mean.
    faithful = [m for m in results
                if m["arm_lag"] < 0.05 and m["arm_reach"] > 0.85 and m["hand_reach"] > 0.80]
    print("\n" + "=" * 118)
    if faithful:
        best = min(faithful, key=lambda m: m["arm_jerk_mean"])
        base = next((m for m in results if m["mode"] == "raw"), None)
        print(f"RECOMMENDATION: '{best['mode']}' — smoothest arm that still tracks the VLA "
              f"(lag {best['arm_lag']:.3f} rad, reach {best['arm_reach']:.2f}, hand reach {best['hand_reach']:.2f}).")
        if base:
            impr = 100 * (1 - best["arm_jerk_mean"] / max(base["arm_jerk_mean"], 1e-9))
            print(f"                arm jerk {base['arm_jerk_mean']:.3f} -> {best['arm_jerk_mean']:.3f} "
                  f"({impr:+.0f}% vs raw); arm vel_max {base['arm_vel_max']:.3f} -> {best['arm_vel_max']:.3f}.")
    else:
        print("RECOMMENDATION: no mode met the fidelity bar (arm lag<0.05, reach>0.85, hand reach>0.80).")
        print("                Loosen smoothing (raise --ema-alpha / --max-delta-*) and re-record,")
        print("                or relax the thresholds in this script for your task.")
    print("=" * 118 + "\n")


if __name__ == "__main__":
    main()
