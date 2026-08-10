# PointNav demonstration generation

Generating PointNav demonstrations with no input demonstrations, in a corpus shaped so it
flows through the same downstream as the ObjectNav one, and trains with the same
behavioural-cloning loss.

## The shape of an episode

An episode is a chain of **segments**, physically continuous: segment `k + 1` starts from
wherever segment `k` finished, so the pose stream never jumps and the video never cuts.
Each segment is

    sample a goal -> plan a shortest path -> turn in place onto it -> drive it

and the episode ends when the frame target is reached. Chaining is what makes every
episode roughly the same length, which is the point — a fixed-length row wastes no
training window. Segment lengths are whatever the geometry gives (median ~100 frames at
the shipped geodesic band), so a 2000-frame episode holds roughly 19 segments.

Everything is sampled on the **robot** navmesh (r = 0.28 m, h = 1.0 m) that
`build_robot_sim` recomputes per scene, not the point-agent mesh the ObjectNav datasets
were cut on. That is what makes the demonstrations traversable by the machine that has to
imitate them, and it is also the source of the one systematic failure mode (below).

## Layout

    src/continuous_demos/
        pointnav.py           # decisions: sampling, screening, budget, rollback. No habitat.
        pointnav_episode.py   # the tick loop that drives them through the simulator
        topdown.py            # navmesh floor plan + episode overlay
        viewport_overlay.py   # the path projected into the RGB frame
        robot_pose.py         # where the robot is, and at which height to ask the pathfinder
        corpus_io.py          # the raw corpus's on-disk contract, shared with ObjectNav
    scripts/
        pointnav_debug_episode.py    # one episode -> an MP4 with a HUD. The design gate.
        generate_pointnav_dataset.py # one process, N scenes -> a raw corpus
        batch_generate_pointnav.py   # a work queue over scenes, for a large run

The split between `pointnav.py` and `pointnav_episode.py` is what makes the design
testable: "is a 1.4 m goal too close", "does a rollback leave the recorded arrays
consistent", and "does a chained episode land on 2000 frames" are all answerable without
loading a scene.

`pointnav_episode.py` deliberately owns its own tick loop rather than adding flags to
`replay.record_segmented_demo`. The two share ~35 lines; `replay.py` is verified
frame-for-frame against a pristine worktree, and that property cannot be re-established
from a unit test. What must not drift are the *constants*, and those are imported
(`MOVE_STALL_TICKS`, `MOVE_HEADING_GAIN`, `MOVE_MAX_VW`, `LOCAL_TIMEOUT_MPS`,
`TIPPED_OVER_DEG`, `is_stalled`), which makes drift impossible rather than unlikely.

The tick ordering is identical to `record_segmented_demo`'s, because the two corpora have
to mean the same thing: observation **before** the step (frame `i` is what the controller
acted on), pose **after** it.

## Rejection, in two places

**Sampling time** screens a candidate goal before the robot moves — island, floor,
geodesic band, loop-back — and redraws. Free, ~2 redraws per accepted segment against a
budget of 30.

**Drive time** means the robot tried and failed. The episode is not thrown away: the
failed segment's ticks are erased, the robot is physically returned to where that segment
started, and a different goal is drawn.

    goal2d stays piecewise-constant because the excursion is ERASED, not relabelled.

Relabelling it under the new goal would teach a policy that goals move when you fail. Two
consequences the rest of the design honours:

* the health monitor is **per segment**, so a rolled-back segment's ticks cannot poison
  the episode's escape / ref-miss verdict. This also makes falling through a mesh hole a
  property of the route rather than of the episode.
* a rolled-back segment is still **counted**, and appears in `segments` with
  `outcome="rolled_back"`. A corpus that hid them would read cleaner than the scenes are,
  and every threshold re-cut from a pilot would be fit to a lie.

### `--stratified-pool N` — how far the goals are, as a choice rather than an accident

Default `0`, which is the original behaviour: draw a navigable point, screen it, keep the
first that passes. That returns the distance distribution the **scene** has, not the one
the window asks for. Measured over six HM3D scenes (4,000 same-island point pairs each),
the geodesic between two random points has median 5.6 m, reaches 15 m in 7.1% of draws and
**30 m in 0.0%** — so raising `--max-geodesic` from 8 to 30 lengthens the tail without
changing the short-skewed shape, and is nearly indistinguishable from raising it to 15.

`N > 0` draws a target distance uniformly from `[--min-geodesic, --max-geodesic]`, pools up
to `N` candidates that pass the **same** screening, and keeps the one nearest the target.
It only ever reweights: no setting can admit a goal an unstratified draw would reject.

Measured on 4 scenes x 8 episodes of 2000 frames, matched seed:

| setting | segment geodesic (med) | agent→goal (med / p90 / max) | segments/episode |
| --- | --- | --- | --- |
| `1.5–8`, pool 0 (production) | 4.03 | 2.76 / 5.70 / 7.96 | 21.2 |
| `2–30`, pool 0 | 4.91 | 3.59 / 8.51 / 16.40 | 17.0 |
| `2–15`, pool 0 | 4.91 | 3.55 / 8.18 / 13.55 | 17.3 |
| `2–15`, pool 24 | 7.21 | 4.37 / 9.11 / 13.57 | 14.9 |
| `2–30`, pool 24 | **10.59** | **5.70 / 12.25 / 18.55** | 11.8 |

The cost is fewer goals per episode at a fixed frame budget — a longer goal takes longer to
reach — so this trades goal-announcement count against goal distance. Yield falls slightly
too (26/32 kept against 28/32).

**What it does not fix.** The pose the encoder receives is anchor-relative
(`relative_se2` against the episode's first observation), and *that* distribution is
almost invariant across all five settings above: median 3.80 → 4.98 m, max 18.5 → 17.8 m.
With 12–21 chained segments the robot covers the scene either way, so the anchor-relative
extent is bounded by **scene diameter**, not by goal distance. No goal-sampling setting
reaches beyond it; that ceiling is a property of indoor scenes and has to be answered on
the encoder side rather than here.

## What the gates actually catch

Measured over a 72-episode MP3D pilot and the first 160 episodes of the HM3D run:

| gate | fires? |
|---|---|
| `not_reached` (0.5 m) | **this is the one doing all the work** |
| segment SPL >= 0.55 | never — completed segments run at SPL 1.000, mean and min |
| ref-miss >= 1.5 s at 1.0 m | never — worst off-path excursion measured is 0.13 m |
| escape / tipped | rare, and recoverable by rollback |

Pure-pursuit tracks a navmesh shortest path essentially exactly, so the efficiency and
path-adherence gates are cheap insurance rather than active filters. They stay armed
because the failure they guard against (a robot fighting geometry the navmesh calls free)
is exactly what must not enter a corpus meant to be imitated — and unlike ObjectNav, where
the reference is a human demonstration, here the reference is a path we chose ourselves,
so a sustained miss has no innocent explanation.

**The systematic failure mode is per-scene navmesh optimism.** Rejections cluster: over
the first 18 HM3D scenes, 3 scenes produced 29 of 40 rejections while the other 15 ran at
91%. Those scenes have navmesh the robot cannot actually traverse — `agent_max_climb =
0.2` admits a step the base will not take, or a doorway passable to a planner and not to a
0.28 m radius. Raising `--segment-rollbacks` spends *more* time on those scenes rather
than recovering episodes, so the right response is to accept the loss.

## Two visualisations

`scripts/pointnav_debug_episode.py --layout {topdown,viewport,both}`.

**`topdown`** rasterises the navmesh floor plan beside the frame, with the planned path,
the executed trajectory, the goal and the agent arrow on it. It answers "is this
trajectory inside the building", which the RGB alone cannot.

The axis convention is the one thing here that fails silently.
`pathfinder.get_topdown_view` returns `bool[z_index, x_index]`; the planar frame is
`X = habitat_x, Y = -habitat_z`, so the raster is already `+X` right / `+Y` up with **no
flip**. Verified two ways: a hand-built array in `tests/test_topdown.py`, and empirically —
100% of an episode's executed trajectory and all its goals land on free pixels, against
39% mirrored in X and 0% transposed. A mirrored map renders perfectly plausibly; the only
symptom is that every left turn is a right turn.

**`viewport`** projects the path and goal through the camera that rendered the frame and
strokes them on with cv2. Cheaper (2.06 ms/frame against the panel's 5.94) and the frame
stays the render's own size, so it scales to a batch worth skimming.

> `HabitatRobotSim.draw_path` / `draw_axes` are the obvious way to do this and **do not
> work for it**. Habitat's `DebugLineRender` draws into the interactive viewer's pass, not
> into offscreen sensor observations: measured, a `draw_path` and a `draw_axes` issued
> immediately before `get_obs()` change **0 pixels** of the returned observation, while
> `sim.navmesh_visualization = True` (a scene-graph drawable, not a debug line) changes
> 1771 in the same frame. This is also why `replay.record_path`'s `debug_draw` branch has
> never put its axes or its purple path into a recorded frame — the BEV panel it hstacks
> is what carries the information.

`--show-navmesh` burns habitat's navmesh wireframe into the render, which is the tool for
looking at the per-scene optimism above.

## The corpus

Same layout as ObjectNav's, plus three per-tick keys and two provenance ones:

```jsonc
{
  "episode_id": "kfPV7w3FaU5.basis_pn_00007", "episode_index": 7,
  "scene_id": "hm3d/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb",
  "goal_text": "",                          // present and empty: PointNav has no object
  "rgb_paths": [...], "pose2d": [...],      // N
  "goal2d": [[gx, gy, gtheta], ...],        // N, piecewise constant per segment
  "segment_index": [0, 0, ..., 1, ...],     // N
  "segments": [{...}],                      // incl. rolled-back ones
  "start_pose2d": [...], "sampling": {...},
  "recording": {"task": "pointnav", "schema_version": 3, ...},
  "quality": {...}
}
```

`goal2d`'s theta is the planned path's **final tangent**, not a constant. That matters: a
constant survives the model's `relative_se2` as `wrap(0 - theta_anchor)`, which *is* the
scene's compass at the window origin — precisely the leak the frame conversion exists to
remove.

`recording.task` is consistency-checked (`recording_metadata.CONSISTENCY_KEYS`), so a
PointNav corpus cannot be poured into an ObjectNav output directory or vice versa.
**`None` means objectnav** and the ObjectNav generator must keep stamping `None`: every
corpus written before the field existed reads back as `None`, and stamping `"objectnav"`
instead would make the whole `v2_25hz` lineage unappendable.

Determinism: `episode_id` and the episode's seed are both pure functions of
`(corpus_seed, scene, episode_index)`, so resume is exact and a re-run at the same seed
reproduces the corpus byte for byte (verified). **The retry-loop structure is part of the
reproducibility contract** — the seed fixes the sequence of draws, but how much of it is
consumed depends on how many candidates are rejected, so changing a screening rule or its
order changes the corpus a seed produces.

## Downstream

`build_action_chunk_episode_dataset.py` carries `goal2d` / `segment_index` through as
`obs_goals` / `segment_indices`. `obs_goal` is the goal at the **anchor** frame, never at
the chunk's tail — the image was rendered at the anchor, and taking the tail's goal would
condition on which goal the agent is *about to* be given.

A chunk spanning a goal change (`crosses_segment`) is physically valid but not
supervisorily valid; `--boundary-policy drop` (the default) removes them, about 1% of rows
at `clen=10 / stride=5`.

An ObjectNav episode has neither column and gets neither — absent, not empty — so its
table is byte-for-byte what it was.

## The conversation

`<pose>` is **one modality type**, playing the role `<image>` plays for vision: many
occurrences, one value row each, bound by occurrence order. A goal pose is simply another
occurrence of it. There is no `<goal>` token and no change to `modality_embed.py`,
`vector_sft.py` or `pose_frame.py`.

Two placements, `format_action_chunk_dataset.py --goal-placement`:

* **`segment`** (default) — its own user message wherever a new goal is introduced, so the
  conversation reads the way the task does:

      user  : <the PointNav system prompt>
      user  : You are at <pose>. Go to <pose>.
      user  : Observation 0: [image] <pose> Action:
      assistant: **____**
      ...
      user  : You are at <pose>. Go to <pose>.      <- segment 1

  Value rows, in text order: `n_obs + 2 * n_segments`.

* **`observation`** — the goal repeated in every turn, like a PointGoal sensor reading.
  Value rows: `2 * n_obs`.

**The announcement carries the agent's own pose as well as the goal's, and that is
load-bearing.** It puts an *agent* pose at row 0 of the value column, so `relative_se2`'s
anchor means the same thing in a PointNav row as in an ObjectNav row. With `Go to <pose>.`
alone, row 0 would be the goal and the two tasks would sit in different frames with
nothing to detect it.

The row order **is** the text order — that is the whole contract — so `modality_values`
and `build_messages` are adjacent in the file and pinned against each other by
`tests/test_pose_injection.py`. Reversing a pair binds every value to the wrong slot with
the counts still matching and nothing raising.

The system prompt is chosen per row from the corpus's own `task` column, not from a flag,
so a table cannot describe one task with the other's prompt.

## Running it

```bash
# The design gate: watch this before generating anything.
python scripts/pointnav_debug_episode.py \
    --scene hm3d/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb \
    --scene-root /Projects/data/scene_datasets \
    --episodes 6 --layout viewport --output pointnav_debug.mp4

# A large run: a work queue over scenes, sized to share the box.
python scripts/batch_generate_pointnav.py \
    --scene-root /Projects/data/scene_datasets \
    --scene-glob 'hm3d/train/*/*.basis.glb' \
    --output-dir /Projects/data/pointnav_hm3d \
    --episodes-per-scene 13 --frames-per-episode 2000 \
    --image-ext jpeg --segment-rollbacks 8 \
    --workers 8 --gpus 0,1,2,3,4,5,6,7 --nice 10

# Then the two downstream steps, exactly as for ObjectNav.
python scripts/build_action_chunk_episode_dataset.py \
    --data-dir /Projects/data/pointnav_hm3d --output-dir <out>/chunks \
    --action-chunk-len 10 --obs-stride-frames 5 --val-scene-count 20

$LONGNAV_PY /Projects/spatial_training/data_scripts/format_action_chunk_dataset.py \
    --in-dir <out>/chunks --out-dir <out>/formatted \
    --split train --val-split validation \
    --modality-marker '<pose>' --modality-column obs_poses \
    --goal-marker '<pose>' --goal-values-column obs_goals --goal-placement segment
```

`batch_generate_pointnav.py` exists rather than `xargs -P` for three reasons `xargs`
cannot cover: per-worker GPU assignment (habitat renders on GPU, and on a busy node the
workers must be spread rather than piled onto device 0 — a DDP job runs at the speed of
its slowest rank); pre-launch backpressure on load, free disk and free RAM, withholding
*new* workers rather than killing running ones; and resume by scene completeness, so a
restart does not pay a simulator startup per finished scene.

### Cost, measured

At 640×480 on HM3D:

| | |
|---|---|
| GPU render + readback | 3.03 ms/frame |
| jpeg encode + write | 8.01 ms/frame |
| one worker | ~74 frames/s, ~22% duty cycle of one GPU |
| bytes per frame | 38.7 KB (against 21 KB for the MP3D ObjectNav corpus) |

So an 8-worker run offers roughly +22% load per GPU across 8 devices, and 10k episodes of
2000 frames is ~800 GB and ~11 h.

## Out of scope, needed later

1. **Rollout.** `objectnav_eval/bridge._PoseFeed` calls `relative_se2_last`, which on a
   column carrying goal rows returns the **goal**, not the agent. It needs to keep both
   streams and assemble them in text order, and to install `RolloutConfig.goal_marker`
   from the spec the way it already installs `modality_marker`. Until then a
   PointNav-trained checkpoint can be trained but not rolled out.
2. A PointNav task layer in `objectnav_eval` (success/SPL against a sampled point, an
   `EpisodeSource` for point goals) and a held-out evaluation split — the generator makes
   demonstrations, not an eval set.
3. Runtime mixing of the two corpora. Deliberately not built yet: the tables stay separate
   on disk, and a union that aligns the column sets is a runtime concern.
4. `vector_sft`'s `episode_stop_labels` derives stops from the *episode* end, which is
   wrong for chained segments — a segment-boundary stop label would need a real column.
5. `render_pose2d_videos.py` does not know about `goal2d`, so a PointNav corpus renders
   back without the goal overlay.
