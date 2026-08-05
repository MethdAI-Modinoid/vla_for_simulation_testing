"""
Online causal smoothing of the right-hand grasp at data-collection time.

Port of the offline v5/v6 grasp smoother (Melvin_ws/scripts/smooth_bang_bang_v5.py,
Melvin_ws/apply_v6_smoothing.py) into the live control loop, so the smoothing
happens where the data is created instead of as post-hoc label surgery:

  * the robot executes the smoothed hand command
      -> measured observation.state is REAL and naturally consistent (no offline
         servo-model regeneration needed), and
  * the recorded `action` label is the same smoothed command the robot executed
      -> no train/deploy distribution shift.

Only the right-hand joints (36..42, Dex3) are touched. The per-joint limits are
v5's DMAX (rad/tick @ 20 Hz) expressed as a max velocity in rad/s:

    DMAX = {36:0.30, 37:0.30, 38:0.30, 39:0.30, 40:0.20, 41:0.25, 42:0.25} rad/tick
    -> {36:6.0, 37:6.0, 38:6.0, 39:6.0, 40:4.0, 41:5.0, 42:5.0} rad/s

which sits inside the measured finger envelope (~4.7-9 rad/s).

Profile generator: instead of the offline corner-rounding (zero-phase gaussian,
Savitzky-Golay), which is non-causal and cannot run online, each hand joint runs
a jerk-limited (S-curve) motion profile with provable bounds:

    max velocity  = per-joint DMAX above
    max accel     = 30 rad/s^2
    max jerk      = 300 rad/s^3

Every control tick the profile is replanned from the current (q, v, a) state to
the latest target ("reflex" replanning), which makes it naturally handle moving
targets and direction reversals without ever looking ahead. A hard per-tick slew
clamp (|dq| <= DMAX*dt) is kept as a backstop.

Trade-off vs the previous slew+EMA version: accel/decel ramps add a little time
to a full close (~0.5 s for 1.5 rad vs ~0.25 s) in exchange for bounded jerk.
"""

import numpy as np

RIGHT_HAND_JOINTS = np.arange(36, 43)
DMAX_RADPS = np.array([6.0, 6.0, 6.0, 6.0, 4.0, 5.0, 5.0])


class HandGraspSmoother:
    """Per-joint jerk-limited S-curve profile for the right-hand grasp.

    Stateful: must be called once per control tick with the full 43-dim joint
    target (or any array whose indices 36..42 are the right-hand joints). Smooths
    the hand joints in place and returns the same array.

    Each joint keeps its own scalar trajectory state (q, v, a). Per tick:

      1. target error d = target - q is read (may move arbitrarily; teleop).
      2. a stopping-feasibility check decides whether to accelerate toward the
         target or to start/continue the maximum-jerk deceleration, such that the
         joint stops exactly at the target (reflex replan -> time-optimal and
         jerk-bounded S-curve, no overshoot for a fixed target).
      3. q is integrated with the commanded accel, velocity-clamped to the
         per-joint DMAX, and a hard slew backstop clips the per-tick step.
    """

    def __init__(
        self,
        dt: float,
        dmax_radps=DMAX_RADPS,
        max_acc: float = 30.0,
        max_jerk: float = 300.0,
        d_tol: float = 1e-3,
        v_tol: float = 0.02,
        max_stop_iters: int = 400,
    ):
        self.dt = float(dt)
        self.dmax_radps = np.asarray(dmax_radps, dtype=float)
        if self.dmax_radps.shape != (7,):
            raise ValueError("dmax_radps must have 7 entries (right-hand joints 36..42)")
        self.v_max = self.dmax_radps.copy()
        self.a_max = float(max_acc)
        self.j_max = float(max_jerk)
        self.d_tol = float(d_tol)
        self.v_tol = float(v_tol)
        self.max_stop_iters = int(max_stop_iters)
        self._q = None
        self._v = None
        self._a = None

    def reset(self, q: np.ndarray) -> None:
        """Re-seed the profile state from a measured/desired hand q (no initial jump)."""
        q = np.asarray(q, dtype=float)
        self._q = q[RIGHT_HAND_JOINTS].copy()
        self._v = np.zeros(7)
        self._a = np.zeros(7)

    def _stop_dist(self, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Distance covered while stopping from (v, a) at max jerk then max accel.

        Simulates the exact same discrete integration the main loop will execute,
        so the feasibility check is self-consistent with the executed trajectory.
        Works on the speed |v| with the accel component along the direction of
        motion (a*sign(v)), so it is sign-correct for both travel directions and
        always accumulates non-negative distances. Returns >= 0 for each joint;
        joints with |v| <= v_tol contribute 0.
        """
        s = np.zeros(7)
        vv = np.abs(v)
        aa = a * np.sign(v)
        done = np.zeros(7, dtype=bool)
        for _ in range(self.max_stop_iters):
            active = (~done) & (vv > self.v_tol)
            if not active.any():
                break
            a_next = np.clip(aa - self.j_max * self.dt, -self.a_max, self.a_max)
            v_next = vv + a_next * self.dt
            crosses = active & (v_next <= 0.0) & (a_next < -1e-12)
            if crosses.any():
                t_stop = np.zeros(7)
                t_stop[crosses] = -vv[crosses] / a_next[crosses]
                s[crosses] += vv[crosses] * t_stop[crosses] + 0.5 * a_next[crosses] * t_stop[crosses] ** 2
            keep = active & ~crosses
            s[keep] += vv[keep] * self.dt + 0.5 * a_next[keep] * self.dt ** 2
            upd = active & ~crosses
            vv = np.where(upd, v_next, vv)
            aa = np.where(upd, a_next, aa)
            done |= active & crosses
        return np.maximum(s, 0.0)

    def smooth(self, q: np.ndarray) -> np.ndarray:
        """Advance the jerk-limited S-curve one tick and write the hand joints of `q`."""
        q = np.asarray(q, dtype=float)
        if self._q is None:
            self.reset(q)
            return q

        target = q[RIGHT_HAND_JOINTS]
        v = self._v
        a = self._a
        q_prev = self._q

        d = target - q_prev
        d_abs = np.abs(d)
        v_abs = np.abs(v)

        settle = (d_abs <= self.d_tol) & (v_abs <= self.v_tol)
        moving = v_abs > self.v_tol
        at_target = d_abs <= self.d_tol
        toward = d * v > 0.0

        # Greedy one-tick feasibility: pick j in {+J, 0, -J} so that the state
        # left after this tick is still inside the reachable set, i.e. its max-
        # jerk stopping distance fits in the distance that remains after this
        # tick's travel. The reachable set certificate is _stop_dist(v, a) <= d,
        # and since the certificate is re-checked every tick, any trajectory that
        # stays feasible provably reaches the target (no overshoot, no reliance
        # on the arrival guard).
        dir_toward = np.where(d >= 0.0, 1.0, -1.0)

        # Arrival speed bound: if a candidate reaches the target this tick, the
        # position guard below will zero the remaining velocity in a single tick
        # (stored a = -v/dt, jerk = -v/dt^2), so arriving at more than v_arr
        # would exceed the jerk limit. Reject fast landing steps; this forces the
        # profile to slow to <= v_arr before the final approach.
        v_arr = self.j_max * self.dt ** 2
        land_eps = self.d_tol * 1e-3

        a_acc = np.clip(a + dir_toward * self.j_max * self.dt, -self.a_max, self.a_max)
        v_acc = np.clip(v + a_acc * self.dt, -self.v_max, self.v_max)
        rem_acc = d_abs - np.abs(v_acc) * self.dt
        accel_ok = ((rem_acc <= land_eps) & (np.abs(v_acc) <= v_arr)) | \
                   (self._stop_dist(v_acc, a_acc) <= rem_acc)

        v_hold = np.clip(v + a * self.dt, -self.v_max, self.v_max)
        rem_hold = d_abs - np.abs(v_hold) * self.dt
        hold_ok = ((rem_hold <= land_eps) & (np.abs(v_hold) <= v_arr)) | \
                  (self._stop_dist(v_hold, a) <= rem_hold)

        # Decelerate only when neither accelerating nor holding keeps us feasible;
        # this is what sheds the accel at exactly the right moment. Once engaged,
        # the decel command tracks the exact-landing law a_target = -v^2/(2d)
        # (jerk-limited ramp) instead of bang-bang max braking: max braking from
        # the latest-feasible switch point always over-brakes the low-speed tail,
        # stranding the joint short of the target and then re-creeping (each
        # transition a jerk spike). The landing law returns a to ~0 as v and d
        # shrink together, so the profile ends at v ~= 0 exactly on the target.
        decel = moving & (~accel_ok) & (~hold_ok)
        decel |= moving & at_target
        hold = moving & (~accel_ok) & hold_ok & (~at_target)
        accel = accel_ok & (~at_target)

        a_land = -dir_toward * np.minimum(
            v_abs * v_abs / (2.0 * np.maximum(d_abs, self.d_tol)), self.a_max)
        a_cmd = np.zeros(7)
        a_cmd = np.where(accel, a + dir_toward * self.j_max * self.dt, a_cmd)
        a_cmd = np.where(decel, a + np.clip(a_land - a, -self.j_max * self.dt,
                                            self.j_max * self.dt), a_cmd)
        a_cmd = np.where(hold, a, a_cmd)

        # Bleed leftover accel: if the joint stopped short of the target with a
        # deep accel still stored (v=0, d>d_tol), ramp it toward 0 at max jerk
        # instead of letting a_cmd fall to 0 (which would be a jerk spike). The
        # direction guard below keeps position pinned while the accel bleeds.
        stopped_short = (v_abs <= self.v_tol) & (d_abs > self.d_tol) & \
                        (np.abs(a) > self.j_max * self.dt)
        a_cmd = np.where(stopped_short, a - np.sign(a) * self.j_max * self.dt, a_cmd)

        # Settle at the target: pin v to 0 so the trajectory goes flat. No accel
        # "bleed" here (a nonzero command while v ~= 0 would push q away from the
        # target); any leftover accel is simply dropped since only q is recorded.
        a_settle = np.clip(-v / self.dt, -self.a_max, self.a_max)
        a_cmd = np.where(settle, a_settle, a_cmd)

        # Integrate. Ease off accel as v approaches the per-joint ceiling so a
        # returns to 0 exactly as v reaches v_max (jerk-bounded: a is shed at
        # max jerk J, which from a=A takes A/J s and gains A^2/(2J) of velocity,
        # so the shed starts at v_max - a^2/(2J)). A hard proportional cap on a
        # would otherwise produce jerk up to a/dt at the start of cruise.
        a_cmd = np.where(
            (~decel) & (v_abs >= self.v_max - a * a / (2.0 * self.j_max)) & (a * dir_toward > 0.0),
            a - dir_toward * self.j_max * self.dt,
            a_cmd,
        )
        a_lim = a_cmd.copy()
        a_lim = np.where(a_cmd > 0.0, np.minimum(a_lim, (self.v_max - v) / self.dt), a_lim)
        a_lim = np.where(a_cmd < 0.0, np.maximum(a_lim, (-self.v_max - v) / self.dt), a_lim)
        a_lim = np.clip(a_lim, -self.a_max, self.a_max)

        v_cmd = v + a_lim * self.dt
        v_cmd = np.clip(v_cmd, -self.v_max, self.v_max)

        # Arrival guards: never step past the current target, and never let an
        # over-aggressive command push velocity through zero while the joint is
        # still approaching (this is what used to cause arrival overshoot/ring).
        pos = d >= 0.0
        neg = d < 0.0
        v_cmd = np.where(pos, np.minimum(v_cmd, d / self.dt), v_cmd)
        v_cmd = np.where(neg, np.maximum(v_cmd, d / self.dt), v_cmd)
        v_cmd = np.where(pos & (v >= 0.0), np.maximum(v_cmd, 0.0), v_cmd)
        v_cmd = np.where(neg & (v <= 0.0), np.minimum(v_cmd, 0.0), v_cmd)

        q_cmd = q_prev + v_cmd * self.dt

        # Hard slew backstop: never step more than DMAX*dt per tick.
        q_cmd = np.clip(q_cmd, q_prev - self.dmax_radps * self.dt,
                        q_prev + self.dmax_radps * self.dt)

        v_cmd = (q_cmd - q_prev) / self.dt
        self._q = q_cmd
        self._v = v_cmd
        self._a = (v_cmd - v) / self.dt
        q[RIGHT_HAND_JOINTS] = q_cmd
        return q
