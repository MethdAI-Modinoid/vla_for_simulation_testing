# G1 Pick-and-Place Manipulation Scene

Documentation for the colored-cube manipulation task added to
[`scene_43dof.xml`](scene_43dof.xml). The scene places a table in front of the
robot with **3 baskets** (red, green, blue) and **3 matching cubes**. On every
simulation reset the cubes are re-placed following a balanced schedule so a
pick-and-place policy can be trained/evaluated across every color arrangement.

Related code: `DefaultEnv._init_manipulation_cubes` and
`DefaultEnv.randomize_cubes` in
[`utils/mujoco_sim/base_sim.py`](../../../../utils/mujoco_sim/base_sim.py).

---

## Scene layout

Robot faces **+x**, so everything is placed in front of it (positive x).

| Object | x (m) | y (m) | z (m) | Notes |
|---|---|---|---|---|
| Table top | 0.41 | 0.0 | top at 0.76 | static |
| Basket 🔴 Red | 0.33 | −0.13 | 0.76 | **fixed**, open-top container |
| Basket 🟢 Green | 0.33 | 0.0 | 0.76 | **fixed** |
| Basket 🔵 Blue | 0.33 | +0.13 | 0.76 | **fixed** |
| Cube 🔴 Red (slot A) | 0.20 | −0.10 | 0.785 | 5 cm, `freejoint` |
| Cube 🟢 Green (slot B) | 0.20 | 0.0 | 0.785 | 5 cm, `freejoint` |
| Cube 🔵 Blue (slot C) | 0.20 | +0.10 | 0.785 | 5 cm, `freejoint` |

The three cube start positions are the **slots**, tuned for the robot's arm
reach. Ordered by y they are:

- **Pos A** = near Red basket (y = −0.10)
- **Pos B** = near Green basket (y = 0.0)
- **Pos C** = near Blue basket (y = +0.10)

---

## Reset behavior

A reset happens on **Backspace**, when the robot **falls**, or on a programmatic
`reset()`. On every reset the cubes are:

1. **Arranged** by the current config in the balanced schedule (see below), i.e.
   which cube goes into slot A / B / C.
2. **Jittered** by an independent uniform **±5 cm** offset in x and y around the
   slot, so no two episodes are identical.

Baskets never move. Key **9** still releases the robot from the elastic-band
hanger (unchanged).

> The reset is detected in `sim_step` by the MuJoCo sim clock rewinding to 0,
> which also catches the viewer's built-in Backspace reset. Each reset counts as
> one **episode**; the episode counter starts at 0 when the sim process launches
> and is **not** persisted across restarts.

---

## Balanced collection schedule

With 3 cubes there are **3! = 6** possible arrangements. Instead of random
sampling (which comes out lopsided), the sim cycles **deterministically** through
all 6 arrangements, **16 episodes each → 96 episodes total**, then repeats.

| Config | Pos A (near 🔴) | Pos B (near 🟢) | Pos C (near 🔵) | Episodes |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 🔴 Red | 🟢 Green | 🔵 Blue | 16 |
| 2 | 🔴 Red | 🔵 Blue | 🟢 Green | 16 |
| 3 | 🟢 Green | 🔴 Red | 🔵 Blue | 16 |
| 4 | 🟢 Green | 🔵 Blue | 🔴 Red | 16 |
| 5 | 🔵 Blue | 🔴 Red | 🟢 Green | 16 |
| 6 | 🔵 Blue | 🟢 Green | 🔴 Red | 16 |
| | | | **Total** | **96** |

Episodes 0–15 → config 1, 16–31 → config 2, … 80–95 → config 6, then it loops.
On each reset the sim prints a line so you can track collection:

```
[cube reset] episode 17, config 2/6, cubes at (A,B,C) = (0, 2, 1)
```

(cube index: red = 0, green = 1, blue = 2)

---

## The math

**Number of arrangements.** Placing 3 distinct cubes into 3 distinct slots is a
permutation of 3 items:

```
3! = 3 × 2 × 1 = 6 configurations
```

**Episodes per config.** To collect each arrangement equally:

```
episodes_per_config = 16
total = 6 configs × 16 = 96 episodes
```

**Per-cube balance.** Each cube appears in each slot in exactly **2 of the 6**
configs (e.g. Red is in slot A for configs 1 & 2, slot B for configs 3 & 5,
slot C for configs 4 & 6). So over 96 episodes each cube occupies each position:

```
2 configs × 16 episodes = 32 times per position
```

| Cube | Pos A | Pos B | Pos C |
|---|:---:|:---:|:---:|
| 🔴 Red | 32 | 32 | 32 |
| 🟢 Green | 32 | 32 | 32 |
| 🔵 Blue | 32 | 32 | 32 |

Perfectly balanced: **32 + 32 + 32 = 96** per cube.

### Aside: why random sampling was not balanced

If instead each reset drew a **uniform random** permutation, each cube would land
in each slot with probability 1/3 (≈ 33 per 100). But "nearest basket by
distance" is not the same as "slot," because the baskets are spaced **0.13 m**
apart in y while the slots are **0.10 m** apart, and the ±5 cm jitter can push a
cube past the midpoint between two baskets. The middle (green) basket has the
widest catchment, giving a skewed **≈ 28 / 43 / 28** split per cube. The
deterministic 6-config schedule above avoids this skew entirely.

---

## Tuning

All knobs are at the top of `_init_manipulation_cubes` in
[`utils/mujoco_sim/base_sim.py`](../../../../utils/mujoco_sim/base_sim.py):

| Parameter | Meaning | Default |
|---|---|---|
| `cube_xy_jitter` | ± jitter in x and y (m) | `0.05` |
| `episodes_per_config` | episodes collected per arrangement | `16` |
| `cube_configs` | the 6 cube→slot arrangements (and their order) | see table |

Cube / basket / table positions and colors are edited directly in
[`scene_43dof.xml`](scene_43dof.xml); the slot positions are read from the model
automatically, so retuning the XML needs no code change.
