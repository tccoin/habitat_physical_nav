# PID Global Pose/Trajectory Tracking

## What this is

A new, simpler action interface alongside the existing pure-pursuit path
follower: given a global `[x, y, theta]` reference pose in the same planar
frame as the saved `pose2d` data, a PID controller servos the holonomic
TidyBot base toward it — one control tick per reference, the same "position
control" action space standard in robotic manipulation. Feeding the controller
a single setpoint is one control step; feeding it a stream of setpoints at a
fixed cadence (an action chunk, or a whole trajectory) turns the same
primitive into trajectory tracking. There is no separate "trajectory mode" —
`track_trajectory()` is just a loop over `PIDPoseController.step()`.

This is deliberately isolated from the existing pure-pursuit code
(`sim.py: HolonomicRobotController` / `pure_pursuit.py`), which solves a
different problem ("here is a path, follow it well, in whatever time it
takes" — used to turn the discrete, time-free human demos into continuous
replays). **`sim.py` was not modified.**

## Files added

- `pid_pose_controller.py` — `PIDPoseControllerConfig`, `PIDPoseController`,
  `track_trajectory()`. Built entirely on existing public `HabitatRobotSim`
  methods (`get_2d_pose()`, `get_joint_state(..., is_vel=True)`, `step()`);
  no new simulator plumbing was needed.
  - Independent PID loops on position (shared x/y gains — the base is
    holonomic and symmetric) and heading.
  - **Derivative-on-measurement** (subtracts current joint velocities rather
    than differencing the error) so a setpoint that jumps every tick — the
    normal case when tracking a trajectory — doesn't cause derivative kick.
  - Optional reference-velocity **feedforward**, computed by
    finite-differencing the reference trajectory when not supplied explicitly.
  - Output is velocity- and acceleration-limited (`v_max`/`w_max`,
    `a_max`/`aw_max`) and returned as `{"chassis_twist": [vx, vy, w],
    "twist_frame": "world"}` — directly usable as `robot_sim.step(**step_args,
    dt=dt)`.
- `eval_pid_tracking.py` — standalone validation script. Reuses
  `load_demo_objectnav`, `build_robot_sim`, `extract_demo`,
  `resolve_scene_path`, `find_scene_dataset_config`, `DEFAULT_URDF_PATH` from
  `replay_human_demo.py` rather than reimplementing scene/sim setup. No
  rendering or frame capture — only `step()` + `get_2d_pose()` — so it's fast
  and adds negligible disk usage.

## Validation methodology

Rather than re-running the pure-pursuit replay, this reads back trajectories
**already generated and already validated as physically plausible** by
`generate_pose2d_dataset.py` and asks: can a much simpler PID controller track
them?

The shard used (`metadata/17DRP5sb8fy.jsonl`) is a **pre-schema corpus
recorded at 30 Hz** — off habitat's fixed 0.008 s physics grid, so every
saturated tick in it carries the ~24% velocity ripple described in
`sim.py::step`, and the numbers below inherit that. The pipeline now records
at 25 Hz (`dt = 0.04 s` = exactly 5 physics steps) and writes the rate into
the metadata, so `eval_pid_tracking.py` reads it and needs no flag; this old
shard has no such field and must be told, once: `--fps 30`. The controller
limits actually in force while the data was recorded are **2.0 m/s and
2.0 rad/s (114.6 deg/s)** — not the 1.28/73 or 1.92/110 of earlier findings
docs, both of which were `0.064/dt` for a wrong `dt`, i.e. artifacts of the
ripple's lower mode rather than real caps.

5 episodes were selected deterministically by trajectory length (229 to 1664
steps, i.e. ~8s to ~55s of demo) to cover short/medium/long variety.

**A real gotcha found and fixed during validation:** resetting the robot via
the raw human-demo start position (`path_3d[0]`) with navmesh snapping
re-applied does *not* reliably reproduce the exact spawn point the stored
`pose2d` trajectory started from — for one test episode the snap now resolves
to a point ~1m away from where it did when `data/` was generated (693
re-simulated poses vs. 710 stored for the same episode). The root cause
wasn't fully diagnosed — it's consistent with navmesh/pathfinder state or
version drift between when `data/` was generated and now, but that's an
inference, not a confirmed mechanism. Whatever the cause, it briefly showed up
as a spurious 1m tracking "error" at step 0 and (when accel limits were
raised) an unstable-looking 3.8m position spike downstream while the
controller tried to catch up. Fixed by spawning directly from the recorded
`pose2d[0]` (holding height from `path_3d[0]`, which comes from the human
demo and isn't subject to snapping) with `snap_to_navmesh=False` — this
sidesteps the drift for this eval, but **the underlying non-reproducibility of
sim state across runs is a separate, unresolved concern worth flagging for
anyone planning RL rollouts or VLA eval in this simulator**, where exact
episode reproducibility may matter more than it does here.

## Results (5 episodes, final gains)

| Episode (suffix) | Steps | ADE (m) | FDE (m) | Heading RMSE (deg) | Max pos err (m) | Max cross-track (m) |
|---|---:|---:|---:|---:|---:|---:|
| ...D03GGCWF20XBJA | 229 | 0.079 | 0.100 | 4.2 | 0.199 | 0.086 |
| ...JOCST0GSBOKZ89S | 452 | 0.047 | 0.124 | 5.5 | 0.206 | 0.061 |
| ...F790JR0BAAJJ947JG | 709 | 0.043 | 0.074 | 12.1 | 0.204 | 0.127 |
| ...MPB616E32TSJP | 1060 | 0.060 | 0.146 | 10.6 | 0.210 | 0.100 |
| ...VJ9S5DHCP2USZ6NYKU | 1664 | 0.027 | 0.073 | 5.5 | 0.211 | 0.103 |
| **Aggregate mean** | | **0.051** | **0.103** | **7.6** | **0.206** | **0.095** |

(Full per-episode data, including saturation rates, in
`data/pid_eval/summary.json`; BEV plots — reference vs. actual trajectory —
in `data/pid_eval/plots/<episode_id>.png`, regenerated by rerunning
`eval_pid_tracking.py`. `data/` is gitignored per this repo's convention for
generated artifacts, so these aren't checked in.)

**Takeaways:**
- ADE stays in the 3–8 cm range and max position error is bounded around
  ~0.2 m across all 5 episodes regardless of trajectory length — the
  controller doesn't accumulate drift over long trajectories.
- Max cross-track error (perpendicular distance to the nearest point on the
  *reference polyline*, not just the same-time-index point) is smaller than
  max position error, confirming the dominant error mode is timing lag on
  fast turns, not the base wandering off the intended path.
- Heading RMSE (4–12 deg) is higher on episodes with more in-place scanning
  (`TURN_LEFT`/`TURN_RIGHT` "look" phases), where the reference yaw rate is
  large; angular saturation was tuned down substantially (see below) but not
  fully eliminated (`aw` saturation ~27–33% of steps) — the demo's turn rate
  during scans occasionally exceeds even a generous `aw_max`.
- Final gains: `kp_pos=2.5, kd_pos=0.3` (position), `kp_th=4.0, kd_th=0.2`
  (heading), `v_max=2.0 m/s`, `a_max=8.0 m/s²`, `w_max=180°/s`,
  `aw_max=400°/s`. Raising `a_max`/`w_max`/`aw_max` from more conservative
  starting values (`a_max=3.0`, `w_max=70°/s`, `aw_max=100°/s²`) was the
  single biggest lever — it cut angular saturation from ~60–68% of steps down
  to ~1–6%, and roughly halved ADE on the episodes with heavy scanning.

## Step-response check (single-setpoint regulation, not just on-trajectory tracking)

The 5-episode validation above always tracks a *feasible* reference — spawned
exactly on `pose2d[0]`, with consecutive setpoints only `v_max * dt` ≈ 0.067 m
apart on this 30 Hz shard (0.08 m on a 25 Hz corpus), so the feedback loop is always correcting a near-zero error. That's
the trajectory-tracking case, but it doesn't exercise the single-step
"position control" case the task also names (and that a VLA/RL caller
commanding one target pose at a time will always hit): servoing to a
setpoint you're not already at, under the same aggressive `a_max`/`w_max`
limits tuned above.

Checked separately: spawned on `pose2d[0]` of the shortest episode, then held
a constant target 0.4 m/0.4 m/30° away for 90 ticks (3 s) with no
feedforward. Error decreased monotonically with no overshoot: 0.56 m/29.6°
at tick 0 → 0.16 m/3.7° at tick 20 → 0.03 m/0.2° at tick 45 → 0.002 m/0.001°
at tick 89. The raised accel/velocity limits do not cause oscillation on a
genuine step setpoint.

## How to reproduce

```bash
conda activate habitat
export HABITAT_SCENE_ROOT=/path/to/data/scene_datasets
python3 eval_pid_tracking.py 17DRP5sb8fy.json.gz \
    --metadata data/metadata/17DRP5sb8fy.jsonl --num-episodes 5
```

Disk footprint of a run: ~170 KB (5 PNGs + one JSON summary); no videos or
per-frame images are written.

## Limitations / not covered (by design, per scope)

- No relative-pose tracking, no VLA-model-in-the-loop evaluation — this is
  the reference-tracking primitive those would sit on top of.
- Gains are hand-tuned against this single scene (`17DRP5sb8fy`); not
  validated across scenes.
- Angular accel/velocity limits are still occasionally saturated during fast
  in-place scans; if heading tracking during scans matters for a downstream
  use case, revisit `w_max`/`aw_max` or add heading-rate feedforward
  smoothing.
