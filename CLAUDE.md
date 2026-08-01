# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone package that replays Habitat-web ObjectNav human demonstrations
(`17DRP5sb8fy.json.gz`, discrete `MOVE_FORWARD`/`TURN_LEFT`/.../`STOP` actions)
as **continuous** robot trajectories for a holonomic mobile base (TidyBot),
driven through Habitat-Sim by a pure-pursuit path controller. Downstream
scripts turn those continuous replays into an image + pose (or image +
action-sequence) dataset for training.

This folder is meant to work standalone (its own `.git`), decoupled from the
rest of the parent `habitat/` checkout — see `README.md` "Included files" /
"Not included files". Don't casually add imports back to the parent repo's
Python files.

## Pipeline (in order)

1. `replay_human_demo.py` — replay one episode of a `.json.gz` dataset through
   Habitat-Sim + `sim.py`, render a debug MP4. This is the manual/debug entry
   point for a single episode.
2. `generate_pose2d_dataset.py` — batch version: replays every episode of a
   dataset, dumps per-frame RGB images plus aligned continuous `[X, Y, theta]`
   poses to `<output-dir>/{images,metadata,videos?}`. Resumable — skips
   episode ids already present in `metadata/<scene>.jsonl` or logged in
   `failed_episodes.log`.
3. `render_pose2d_videos.py` — reads back a `generate_pose2d_dataset.py`
   output dir and stitches frames into MP4s for visual QA (`--episode-index`,
   `--episode-id`, or `--all`).
4. `build_pose2d_dataset.py` / `build_uniform_dataset.py` — turn the
   images+metadata into a HuggingFace `DatasetDict` (arrow table), converting
   to a conversation format (`to_convo`) and sampling episodes to a uniform
   distribution over trajectory length (`sample_uniformly_by_length`).
   `build_uniform_dataset.py` targets the discrete `action_sequence` column;
   `build_pose2d_dataset.py` targets the continuous `pose2d` column.
5. `verify_replay_format.py` — standalone empirical check of the raw
   `reference_replay` format's action/state ordering and axis conventions
   against a `.json.gz` dataset; the executable proof behind
   `REPLAY_DATA_FORMAT.md`. Re-run this on any new dataset shard before
   trusting the alignment recipe.

Separately, `eval_pid_tracking.py` validates a different, simpler action
interface: `src/continuous_demos/pid_pose_controller.py`'s PID-based global
pose/trajectory tracker (position-control, one target pose per tick — the
kind of action space a VLA/RL policy would emit), as opposed to the
pure-pursuit path follower steps 1-2 use. It replays already-generated
`pose2d` trajectories from step 2's output and reports tracking-error
metrics. See `docs/PID_TRACKING_REPORT.md` for the full design/validation
writeup, including a known sim non-reproducibility gotcha: navmesh snapping
can resolve to a different point across runs/environments for the same
episode, which looks like tracking error if you don't spawn from the
recorded trajectory's own first pose. Relevant to anyone building further
RL/VLA-eval work in this simulator, not just PID tracking.

Also separate from the replay pipeline: `scripts/eval_objectnav.py` +
`src/continuous_demos/objectnav.py` port the **ObjectNav task** onto this
simulator (episode loading, spawn, success/SPL/soft-SPL/oracle-success/
oracle-SPL). See "ObjectNav task layer" below.

`batch_gen.bash` fans step 2 out across every `.json.gz` shard in a dataset
directory (`xargs -P` workers, one `generate_pose2d_dataset.py` process per
scene file). Its `PYTHON_BIN`/`INPUT_DIR`/`OUTPUT_DIR` are hardcoded paths —
edit them in place rather than parameterizing, since this is a one-off
launch script, not a library entry point.

`data/` (generated output, gitignored) and `videos/` / `*.mp4` (gitignored)
are pipeline artifacts, not source.

## Running things

There is no linter configured. Most verification is still empirical/visual:
run a script and inspect the printed diagnostics or the output MP4/BEV plots
-- that's inherent to a physics simulator and can't be replaced by unit tests.

There is a small `tests/` suite (`pip install -e ".[test]"`, then `pytest`)
covering what *can* be checked without a simulator instance: invariants and
known-value/step-response checks on the pure-math pieces (`pure_pursuit.py`,
`pid_pose_controller.py`, `hab_quat_to_heading`), plus a regression test that
re-runs `verify_replay_format.py`'s action/state-ordering check against the
bundled example dataset. Everything except the `hab_quat_to_heading` test
runs in any Python env with the package installed -- no `habitat_sim`/
`magnum` needed -- because those modules don't import them; that test skips
cleanly where those aren't installed rather than failing. Add tests here
alongside new pure-logic code (controllers, metrics, path/pose math); don't
try to unit-test `HabitatRobotSim` itself or full replay runs -- those stay
empirical.

This is a pip-installable package (`src/continuous_demos/`, `pyproject.toml`)
plus a `scripts/` directory of CLI entry points that import it — install it
editable once per env before running anything: `pip install -e .`. Scripts
are meant to be run from the repo root as `python scripts/<name>.py`, and
never import from each other (only from `continuous_demos.*`); if you add a
new script that needs something another script defines, move that thing into
an appropriate `src/continuous_demos/` module instead of importing the
sibling script.

```bash
# One-time setup (per Python env)
pip install -e .

# Single-episode debug replay (run from this folder)
python scripts/replay_human_demo.py examples/17DRP5sb8fy.json.gz --episode-index 4 --output demo.mp4 --debug-draw

# Same, with scenes living outside the parent repo's data/ layout
python scripts/replay_human_demo.py examples/17DRP5sb8fy.json.gz --episode-index 4 --scene-root /path/to/data/scene_datasets --output demo.mp4

# Re-verify the reference_replay ordering/axis assumptions on a dataset shard
python scripts/verify_replay_format.py examples/17DRP5sb8fy.json.gz

# Batch-generate the pose2d dataset, then spot-check via video
python scripts/generate_pose2d_dataset.py examples/17DRP5sb8fy.json.gz --output-dir data
python scripts/render_pose2d_videos.py --output-dir data --episode-index 4

# Score human demos as ObjectNav episodes (success/SPL/oracle metrics)
python scripts/eval_objectnav.py examples/17DRP5sb8fy.json.gz --num-episodes 5
```

`replay_human_demo.py` resolves relative scene paths (if `--scene-root` is
omitted) in this order: `HABITAT_SCENE_ROOT` env var → `./data/scene_datasets`
→ `../data/scene_datasets`.

Requires a Python env with `habitat_sim`, `magnum`, `numpy`, `imageio`
already installed, plus MP3D scene assets on disk somewhere resolvable by the
scene-root logic above. `build_*_dataset.py` additionally need `datasets`
(HuggingFace) and `PIL`.

## `reference_replay` data format — read before touching indexing

Full details, empirically verified over all 1141 episodes, are in
`REPLAY_DATA_FORMAT.md`. The one thing to internalize before writing any code
that indexes into `reference_replay`:

- Entry `i`'s `action` is the action that **produced** entry `i`'s state (the
  incoming action, not the outgoing one).
- Entry 0 is a `STOP` seed holding the true initial state `s_0`.
- Entry 1 **duplicates** entry 0's state exactly, but already carries the
  human's first real action — that action's effect only shows up in entry 2.
- Naively pairing `(state[i], action[i])` or `(state[i], action[i+1])` is off
  by one. Collapse the index-1 duplicate first (`states[0] + states[2:]`,
  same for actions); from then on `actions[i]` applied to `states[i-1]` yields
  `states[i]` exactly (verified over 147,898 transitions, 0 violations).
- Action vocabulary is six actions, not four: `MOVE_FORWARD`, `TURN_LEFT`,
  `TURN_RIGHT`, `LOOK_UP`, `LOOK_DOWN`, `STOP`. `LOOK_*` only changes
  `sensor_data[*].rotation` (camera pitch), never base `position`/`rotation`,
  and the pitch persists across later frames.
- Coordinates are **right-handed, y-up, forward = local −z** (not the
  "x-forward, y-left, z-up" some earlier notes assumed — that was wrong).
  Heading/yaw is rotation about `+y`.

`extract_path_3d` (in `replay_human_demo.py`) drops entry 1, which is
sufficient for a position-only path but does **not** by itself fix
action↔state alignment for `(s, a, s')` work — use the collapse recipe above
for that.

`extract_notes.md` (parent dir) documents the surrounding dataset schema:
this `.json.gz` does not load via Habitat's stock
`ObjectNavDatasetV1.from_json()` (fails on `is_thda`), so it must be parsed
as raw JSON; `goals_by_category` must be joined back onto each episode via
`f"{os.path.basename(scene_id)}_{object_category}"`.

## Coordinate frames in `sim.py` — three, not one

`sim.py`'s module docstring documents three distinct frames; conflating them
is the main source of bugs in this file (see comments at `HabitatRobotSim`
and its `reset()`):

1. **Habitat world frame** (right-handed, y-up): `robot.translation`, scene
   geometry, every quaternion in `reference_replay`. Forward is local `-z`
   (per `habitat_sim.geo`: RIGHT=+x, UP=+y, FRONT=-z).
2. **Robot body frame** (ROS/URDF-style, right-handed): +x forward, +y left,
   +z up. `HabitatRobotSim.step()` takes `chassis_twist = [v_x, v_y, w]` in
   this frame.
3. **Planar control / world frame** (2D, right-handed, up = habitat +y): what
   `get_2d_pose` returns and what path-following controllers operate in.
   Axes: `X = habitat_x`, `Y = -habitat_z`, `theta` measured CCW from `+X`.

Frames 2 and 3 share the same axis convention; frame 2 is frame 3 carried by
and yawed with the base (`joint_x`, `joint_y`, `joint_th` all zero ⇒ they
coincide).

`sim.py` itself now contains only `HabitatRobotSim` — the simulator: it owns
the Habitat-Sim instance, the URDF-loaded robot, sensors, and
stepping/kinematic integration. Everything that drives the simulator but
isn't the simulator lives in sibling modules under `src/continuous_demos/`:

- `controllers.py` — `HolonomicRobotController` (wraps
  `pure_pursuit.PurePursuitController` to drive the base along a 2D path;
  `HolonomicControllerConfig` tunes velocity/acceleration/look-ahead limits)
  and `HeadingRotationController` (drives pure in-place rotation through a
  sequence of target yaw headings, used for the `LOOK_UP`/`LOOK_DOWN`/turn
  segments of a replay, as opposed to path-following).
- `replay.py` — `record_segmented_demo` / `record_path`, the top-level replay
  recording used by `scripts/replay_human_demo.py` and
  `scripts/generate_pose2d_dataset.py`, plus `hab_quat_to_heading` (the
  habitat-quaternion → planar-heading conversion).
- `episode_io.py` — dataset/episode IO and scene-resolution helpers shared by
  `replay_human_demo.py`, `generate_pose2d_dataset.py`, and
  `eval_pid_tracking.py` (`load_demo_objectnav`, `resolve_scene_path`,
  `find_scene_dataset_config`, `build_robot_sim`, `extract_demo`,
  `find_looking_slices`, `DEFAULT_URDF_PATH`). Pulled out of
  `replay_human_demo.py` specifically so scripts stop importing from other
  scripts — a script should only ever be a `scripts/` entry point, never an
  import target.
- `pid_pose_controller.py` — the PID global-pose/trajectory tracker (see
  `docs/PID_TRACKING_REPORT.md`), independent of the pure-pursuit stack above.
- `objectnav.py` / `objectnav_dataset.py` — the ObjectNav task layer (below).
  These are the **only** modules allowed to import `habitat.*` (habitat-lab).
  The "keep this folder standalone" rule targets the parent repo's loose Python
  files; habitat-lab is an installed package and is a deliberate dependency
  here.

`scripts/sim_debug_drive.py` holds `sim.py`'s former manual/debug entry points
(`main()`/`test_rotation_controller()`) — ad hoc driving scripts, not part of
the documented pipeline, and not imported by anything.

`pure_pursuit.py` implements Pure Pursuit / Adaptive / Regulated / Dynamic
Window Pure Pursuit variants (ported from
https://github.com/decwest/dwpp) behind one `PurePursuitController` class
selected via `method_name`.

## ObjectNav task layer (`objectnav.py`, `objectnav_dataset.py`)

Ports the classic ObjectNav benchmark onto `HabitatRobotSim` **without
modifying `sim.py`** and without buying into habitat-lab's hydra/`Env` stack.

It is deliberately *between* a sim and an RL env: it makes **no assumption about
the action space** and never wraps stepping or rendering. Continuous control here
means action chunking, real-time trajectory mixing, and simulated inference lag,
none of which fit `env.step(action)`. The caller keeps calling
`robot_sim.step(...)` / `robot_sim.get_obs()` itself and pulls from the task:

```python
task = ObjectNavTask(robot_sim, ObjectNavConfig(), scene_id=scene_id)
task.reset(episode)      # spawn + reset measures; returns t=0 metrics
task.observe_pose(pose)  # O(1) path-length sample -- call per physics tick
task.call_stop()         # agent declares termination (what Success gates on)
task.evaluate()          # navmesh queries + measure update -- only when asked
task.evaluate_trajectory(episode, poses)  # offline scoring, no stepping
```

Physics stepping therefore never triggers a geodesic query or a render. Path
length — the one quantity needing integration over time — is tracked by the
cheap `PathTracker` instead of inside SPL (upstream accumulates it inside
`update_metric`, which would tie path length to evaluation cadence); hence
`ContinuousSPL` / `ContinuousSoftSPL`, which keep habitat's formulas but read
`task.path_tracker.length`. Verified: `success`/`spl`/`soft_spl` are identical at
`evaluate_every=1` and `25`; only the oracle metrics (which see just the
evaluated samples, and are therefore lower bounds at the chosen cadence) move.

What is reused vs. written here:

- **Reused from habitat-lab**: `DistanceToGoal`, `Success`, `SPL`, `SoftSPL`,
  `Measure`/`Measurements` — they instantiate fine with a plain config object
  and a duck-typed sim, so numbers match published ObjectNav results.
  `NavSimAdapter` supplies the only three sim methods they touch
  (`geodesic_distance`, copied verbatim from `HabitatSim`; `get_agent_state`;
  `previous_step_collided`, a stub — `HabitatRobotSim` has no collision flag).
- **Written here**: `OracleSuccess` / `OracleSPL` (habitat-lab has none) — an
  imaginary oracle terminates the episode the moment the agent is close enough,
  so oracle success ignores `STOP` and oracle SPL freezes path length at that
  first success.
- `get_agent_state` is the *one* place the planar↔habitat inversion happens
  (`position = [X, base_y, -Y]`, and `base_y` subtracts the same
  `-cumulative_bb.min.y` URDF lift that `reset()` adds). Get this wrong and
  every metric is silently garbage — `tests/test_objectnav.py` pins it.

Episode loading (`objectnav_dataset.py`): these `.json.gz` shards **are** stock
ObjectNav-v1 datasets; the *only* reason `ObjectNavDatasetV1.from_json` rejects
them is four extra episode keys (`reference_replay`, `is_thda`, `scene_dataset`,
`scene_state`). So `ObjectNavReplayDataset` subclasses it with an adapted
`from_json` (unknown keys → `episode.extra`, so future shards degrade instead of
failing) and we get the whole `Dataset` abstraction — `scene_ids`,
`get_scene_episodes`, `filter_episodes` — plus real `ObjectGoal`/
`ObjectViewLocation` objects. **habitat's `from_json` overwrites `episode_id`
with `str(index)`; ours preserves the real id** (the pipeline resumes by it) and
records the index as `episode_index`. `load_objectnav_episodes()` falls back to
`episode_io.load_demo_objectnav` + light dataclasses with the same attribute
surface if habitat-lab is missing or a shard trips the parser.

Two things to know before reading the numbers:

- Distances are computed on the navmesh `build_robot_sim` recomputes for the
  *robot* (radius 0.28 m, height 1.0 m), not the point agent the dataset was
  authored with. Start distance therefore differs from the episode's recorded
  `info["geodesic_distance"]` by ~0.2 m typically and several metres in narrow
  spaces. That is a real difference in what is traversable, not a bug — the
  eval script prints it per episode.
- A `HabitatRobotSim` is pinned to one scene (`reset(scene_id=...)` does not
  rebuild the robot), so build one sim + task per scene and drive the outer loop
  with `episodes_by_scene`. `ObjectNavTask.reset` raises on a scene mismatch.

Validation: `python scripts/eval_objectnav.py examples/17DRP5sb8fy.json.gz
--scene-root <...> --num-episodes 5` replays human demos through the pure-pursuit
stack and scores them. These demos are successful human trajectories, so success
should be 1.0 — it is (5/5, mean SPL 0.59, oracle SPL 1.0). Note long demos need
a generous `--timeout`; truncated replays are flagged `replay_truncated` rather
than silently counted as navigation failures.
