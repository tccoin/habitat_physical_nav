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

`batch_gen.bash` fans step 2 out across every `.json.gz` shard in a dataset
directory (`xargs -P` workers, one `generate_pose2d_dataset.py` process per
scene file). Its `PYTHON_BIN`/`INPUT_DIR`/`OUTPUT_DIR` are hardcoded paths —
edit them in place rather than parameterizing, since this is a one-off
launch script, not a library entry point.

`data/` (generated output, gitignored) and `videos/` / `*.mp4` (gitignored)
are pipeline artifacts, not source.

## Running things

There is no build/lint/test tooling configured (no pytest, no linter config).
Verification is empirical/visual: run a script and inspect the printed
diagnostics or the output MP4.

```bash
# Single-episode debug replay (run from this folder)
python replay_human_demo.py 17DRP5sb8fy.json.gz --episode-index 4 --output demo.mp4 --debug-draw

# Same, with scenes living outside the parent repo's data/ layout
python replay_human_demo.py 17DRP5sb8fy.json.gz --episode-index 4 --scene-root /path/to/data/scene_datasets --output demo.mp4

# Re-verify the reference_replay ordering/axis assumptions on a dataset shard
python verify_replay_format.py 17DRP5sb8fy.json.gz

# Batch-generate the pose2d dataset, then spot-check via video
python generate_pose2d_dataset.py 17DRP5sb8fy.json.gz --output-dir data
python render_pose2d_videos.py --output-dir data --episode-index 4
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

Key pieces in `sim.py`:
- `HabitatRobotSim` — owns the Habitat-Sim instance, the URDF-loaded robot,
  sensors, and stepping/kinematic integration.
- `HolonomicRobotController` — wraps `pure_pursuit.PurePursuitController` to
  drive the base along a 2D path (`HolonomicControllerConfig` tunes
  velocity/acceleration/look-ahead limits).
- `HeadingRotationController` — drives pure in-place rotation through a
  sequence of target yaw headings (used for the `LOOK_UP`/`LOOK_DOWN`/turn
  segments of a replay, as opposed to path-following).
- `record_segmented_demo` / `record_path` — top-level replay recording used
  by `replay_human_demo.py`.

`pure_pursuit.py` implements Pure Pursuit / Adaptive / Regulated / Dynamic
Window Pure Pursuit variants (ported from
https://github.com/decwest/dwpp) behind one `PurePursuitController` class
selected via `method_name`.
