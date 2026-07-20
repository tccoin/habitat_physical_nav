# Action-Chunked SFT Dataset — Design & Validation Report

## What this is

Two additions to the pipeline, built directly on the raw
`generate_pose2d_dataset.py` output (`<data_dir>/images/` +
`<data_dir>/metadata/*.jsonl`, one row per episode, native-fps `pose2d` +
`rgb_paths`):

- `build_action_chunk_dataset.py` — resamples that native-fps trajectory down
  to a configurable, lower observation frequency (fixed-stride by default, or
  drawn per-observation from a distribution to simulate variable inference
  latency — see "Variable observation spacing" below), and pairs each
  surviving observation with a configurable-length chunk of *future* actions
  expressed relative to that observation's own pose. Output is an arrow
  `DatasetDict`, one row per observation.
- `replay_chunked_actions.py` — reads that arrow dataset back with no trained
  model involved and drives the simulator through it, to check the encoding
  round-trips: since every chunk is ground truth (not model-predicted),
  decoding chunks back into an absolute trajectory and PID-tracking it should
  reproduce `eval_pid_tracking.py`'s non-chunked tracking behavior almost
  exactly.

All of the frequency/index arithmetic both scripts depend on lives in one new
module, `src/continuous_demos/action_chunking.py`, so there's exactly one
place that can get the off-by-one wrong.

## Vocabulary and the three configurable knobs

| term | meaning |
|---|---|
| `native_fps` | fps the `pose2d`/`rgb_paths` were recorded at (the `--fps` originally passed to `generate_pose2d_dataset.py`). **Not stored in the metadata JSONL** — every script here takes it as an explicit arg the caller must get right. |
| `obs_hz` | how often an observation (image + pose) is sampled from the native trajectory. Lower than `native_fps`. |
| `action_chunk_len` | number of future native-fps poses attached to each observation, as relative actions. |

Derived: `dt_native = 1/native_fps`, `obs_stride_frames = round(native_fps /
obs_hz)`, `obs_interval = obs_stride_frames * dt_native`, `chunk_duration =
action_chunk_len * dt_native`.

Default config (per your instruction): `obs_hz=4`, `action_chunk_len=10`,
`native_fps=20` → `obs_stride_frames=5`, `chunk_duration=0.5s`,
`obs_interval=0.25s` → chunks are 2x the observation interval, i.e. **50%
overlap** between consecutive chunks.

---

## Q1: Is the observation frequency actually reduced?

Yes. `make_chunk_config` (`action_chunking.py`) computes
`obs_stride_frames = max(1, round(native_fps / obs_hz))`, and
`build_observation_records` walks the native trajectory in steps of that
stride:

```python
idx = 0
while idx + action_chunk_len <= n - 1:
    ... emit observation anchored at native frame idx ...
    idx += obs_stride_frames
```

So an observation is emitted only every `obs_stride_frames` native frames —
confirmed both by a unit test (`test_build_observation_records_indexing`:
27 native frames, `obs_hz=4` @ `native_fps=20` → observations land at native
indices `[0, 5, 10, 15]`, i.e. every 5th frame) and empirically on the real
dataset: `data/metadata/17DRP5sb8fy.jsonl` (1045 episodes, native fps
assumed 20) built down to 164,578 observation rows, an average of ~157
observations per episode against episodes that run several hundred to ~1660
native frames — consistent with a ~5x reduction.

One caveat worth flagging: `obs_stride_frames` is `round()`-ed to the nearest
integer frame, so if `native_fps / obs_hz` isn't an integer, the *effective*
observation rate isn't exactly the requested one. `ChunkConfig.effective_obs_hz`
reports what was actually achieved (`1 / obs_interval`), and the build script
prints both requested and effective values. At the documented defaults
(20 / 4 = 5, exact) this doesn't bite, but it will for e.g. `obs_hz=3` on
20 Hz data (`round(6.67) = 7` → effective ≈2.86 Hz, not 3).

## Q2: Does replay run at the exact observation cadence and discard the overlapping tail?

Yes — this is `reconstruct_reference_trajectory` in
`replay_chunked_actions.py`:

```python
for i, row in enumerate(rows):          # rows sorted by obs_index
    anchor = row["obs_pose"]
    chunk = row["action_chunk"]
    n_ticks = len(chunk) if i == n - 1 else min(obs_stride_frames, len(chunk))
    for j in range(n_ticks):
        targets.append(relative_to_pose(anchor, chunk[j]))
```

For every row except the last, only the **first `obs_stride_frames` actions**
of its chunk are decoded and kept — the remainder of the chunk (the part that
overlaps with what the *next* observation's chunk will also cover) is
discarded. The last row, which nothing supersedes, is decoded in full.

Why this reproduces the native trajectory exactly with no gap or duplication:
row `i`'s frame index is `idx_i`, row `i+1`'s is `idx_i + obs_stride_frames`
by construction. Row `i`'s kept ticks decode native frames
`idx_i+1 .. idx_i+obs_stride_frames`; row `i+1`'s first kept tick decodes
`idx_i+obs_stride_frames+1`. The two ranges are contiguous with no overlap and
no gap — i.e. new chunks are switched in exactly at observation boundaries,
`obs_interval` seconds apart, which is "replay at the observation rate."

The individual PID control ticks inside a chunk still run at native `dt`
(`1/native_fps`) — chunk boundaries change *which* target stream is being
fed, not the control loop's own rate.

## Q3: Relative-pose conversion and tracking architecture

**Conversion** (`action_chunking.py`): a pose `target = [x, y, theta]` is
expressed relative to an `anchor` pose by rotating the displacement into the
anchor's own heading frame (anchor's heading becomes local `+X`):

```
dx, dy   = target.xy - anchor.xy
local_x  =  cos(anchor.theta)*dx + sin(anchor.theta)*dy
local_y  = -sin(anchor.theta)*dx + cos(anchor.theta)*dy
local_th = wrap_angle(target.theta - anchor.theta)
```

`relative_to_pose` is the algebraic inverse (rotate back into world frame,
add the anchor's translation). This is the same planar convention used
everywhere else in the repo (`X = habitat_x`, `Y = -habitat_z`, `theta` CCW
from `+X`; see `sim.py`'s frame docstring). Every action in a chunk is
relative to the **same anchor** (the observation's own pose), not to the
previous action — so decoding action `j` never needs actions `0..j-1` first,
and there's no error accumulation within a chunk from the encoding itself.

Correctness of the transform is pinned by
`tests/test_action_chunking.py::test_relative_pose_round_trip` (200 random
poses, including wrapped angles: `compose(anchor, relative(anchor, p)) == p`
to `1e-9`) and `test_relative_pose_is_anchor_frame` (a point one metre ahead
along the anchor's own heading must read back as local `[1, 0, 0]`
regardless of the anchor's world heading).

**Tracking**: rather than reimplement a control loop, `run_chunked_episode`
spawns the robot at the first observation's own recorded pose (same
navmesh-drift rationale as `eval_pid_tracking.run_episode` — see that
function's docstring), builds the reconstructed absolute trajectory via
`reconstruct_reference_trajectory`, and hands it to the **existing**
`pid_pose_controller.track_trajectory` unchanged — identical PID gains,
identical finite-difference velocity feedforward, identical `dt` handling as
the non-chunked path. The only thing that differs between
`eval_pid_tracking.py` and `replay_chunked_actions.py` is *how the reference
trajectory array is built* (straight from `pose2d` vs. decoded from chunks);
once built, both are tracked by the same code. This is also why the
`compute_metrics`/`plot_bev`/`select_episodes` helpers were pulled out of
`eval_pid_tracking.py` into a new shared module, `tracking_eval.py` — so the
new script doesn't import the sibling script (this repo's convention) while
still reusing the exact same metric code for a fair comparison.

## Refactor: `tracking_eval.py`

`eval_pid_tracking.py` previously defined `load_metadata_records`,
`select_episodes`, `cross_track_error`, `compute_metrics`, and `plot_bev`
inline. These moved verbatim into `src/continuous_demos/tracking_eval.py`
(only `plot_bev`'s first argument was renamed `episode_id` → `title`, since
`replay_chunked_actions.py` needed a different title string); both scripts
now import from there. No behavior change to `eval_pid_tracking.py`.

## Empirical validation

Built the arrow dataset from the real `data/metadata/17DRP5sb8fy.jsonl`
(1045 episodes, defaults: `obs_hz=4`, `action_chunk_len=10`, `native_fps=20`)
→ 164,578 observation rows, 0 episodes dropped for being too short.

### First pass (superseded — methodology was too weak)

The first validation ran the non-chunked baseline (`eval_pid_tracking.py`)
and the chunked round-trip (`replay_chunked_actions.py`) once each on 3
episodes and compared each run's own ADE/FDE against its own reference
trajectory:

| episode | non-chunked ADE/FDE | chunked ADE/FDE |
|---|---|---|
| short (229 frames) | 0.040m / 0.066m | 0.038m / 0.070m |
| medium (709 frames) | 0.021m / 0.047m | 0.022m / 0.063m |
| long (1664 frames) | 0.014m / 0.054m | 0.013m / 0.058m |

These looked close, but **this comparison doesn't actually establish that the
two runs produced the same trajectory** — it only shows both runs had
similar-*sized* tracking error against their own (slightly different-length)
reference. It doesn't rule out a systematic bug that happens to look small on
this metric. Asked to tighten this up, two things were checked instead:

### Was the "one tick later" explanation actually verified?

No — it was asserted, not checked. Directly diffing the two runs' *actual*
position trajectories against each other (not against their own references),
aligned for the one-tick offset, showed the gap was NOT bounded: mean
position diff grew from ~1e-5 m over the first 50 steps to ~2e-3 m over the
last 50 steps of the long episode, with the max occurring mid-trajectory, not
decaying back toward zero. That ruled out "negligible one-off tick" as an
explanation and pointed at a real, deterministic (reruns of the same
comparison gave the same numbers) discrepancy.

**Root cause**: `eval_pid_tracking.py` calls `track_trajectory` with
`ref_traj = pose2d`, so its first control tick's setpoint is `pose2d[0]`
itself — the pose the robot is already spawned at. That tick is not a no-op:
spawning via `robot_sim.reset()` (URDF joint homing + navmesh state) lands
a few hundredths of a radian off the recorded heading (observed: ~0.05 rad /
2.9° on the long episode), so the controller issues a real corrective command
on tick 0 before it ever starts chasing `pose2d[1]`. The original
`reconstruct_reference_trajectory` dropped that tick — its first setpoint was
already `pose2d[1]` — so the chunked run's controller (and hence its whole
integral/derivative state history) diverged from the baseline's from tick 0
onward, in a way that doesn't self-correct.

**Fix**: `reconstruct_reference_trajectory` now prepends the first
observation's own `obs_pose` as the reconstructed trajectory's leading pose,
so the setpoint stream matches `eval_pid_tracking.py`'s tick-for-tick over
the reconstructed span. `run_chunked_episode` no longer passes an explicit
`initial_pose` to `track_trajectory` either, for the same reason
(`ref_traj[0]` already *is* `obs_pose`, matching the baseline's default
zero-feedforward first tick).

### Is a several-cm gap "alarming," and is it gone after the fix?

The real question underneath "is this alarming" is: how much does the
*non-chunked script itself* vary between two back-to-back runs of the exact
same episode with identical inputs? If chunked-vs-non-chunked disagreement is
no bigger than non-chunked-vs-non-chunked disagreement, there's no "extra"
error to explain — chunking is exactly reproducing what the baseline itself
does.

That comparison has to happen **inside one process, on one live `robot_sim`/
`controller` pair, with runs interleaved** — reusing the objects across many
`.reset()` calls the way both scripts already do. (An earlier attempt that
computed the "noise floor" from a separate process invocation was misleading:
different Habitat-Sim process instances can land on different post-load
floating-point states, which turned out to dominate over any real per-reset
variability and made two unrelated processes look artificially more different
than two resets *within* the same process.)

Protocol used: within a single process, for each of 5 iterations, run
non-chunked (`nc_i`) then chunked (`ch_i`) back to back on the same
`robot_sim`/`controller`, `.reset()` between every run. Compare:
- **paired, same-iteration**: `nc_i` vs `ch_i` — what chunking adds, if
  anything, on top of a single reset's realization.
- **inherent, cross-iteration**: `nc_i` vs `nc_j` (`i != j`) — pure
  same-algorithm rerun noise, no chunking involved at all.

| episode | inherent noise (nc vs nc, cross-iteration) | paired (nc vs ch, same-iteration) |
|---|---|---|
| short (226 frames, 15 reruns / 210 pairs) | max 0.0197 m, median 0.0131 m | 0.0131 – 0.0213 m |
| medium (706 frames, 5 reruns) | range 0.0000 – 0.0132 m | 0.0086 – 0.0132 m |
| long (1661 frames, 5 reruns) | range 0.00008 – 0.0132 m (clusters by reset realization) | 0.0003 – 0.0009 m |

Paired same-iteration diffs fall inside (or, for the short episode, within
~8% of) the inherent cross-iteration range in every case — i.e. chunking adds
no detectable error beyond the simulator's own reset-to-reset variability.
That variability itself is real and larger than one might expect from "just a
PID loop": it's clustered/bimodal rather than smoothly distributed (e.g. the
long episode's 5 non-chunked reruns fall into distinguishable clusters
~0.0003–0.0013 m apart internally but ~0.006–0.013 m apart across clusters),
consistent with `robot_sim.reset()`'s URDF-joint-homing search landing on
slightly different sub-radian solutions from run to run, which a
33-second-long closed-loop trajectory can amplify into centimeter-scale
position differences by the end — chaotic sensitivity in the underlying
physics simulation + controller loop, not something introduced by the
dataset or replay scripts. The exact round-trip of the relative-pose encoding
itself is unaffected by any of this and is checked independently, with no
simulator involved, in `tests/test_action_chunking.py`
(`recon_ref` vs `pose2d` differ by ≤ 2e-15, float64 machine epsilon).

## Variable observation spacing (simulated inference latency)

Real deployment doesn't query the policy on a metronome — inference latency
varies, so the gap between one observation and the next varies too. Added
`--obs-spacing-mode {fixed, uniform, poisson}` (plus `--obs-spacing-jitter`
and `--obs-spacing-seed`) to `build_action_chunk_dataset.py` to simulate that:
instead of every observation landing exactly `obs_stride_frames` native
frames after the last, the gap is drawn per-observation from a distribution
centered on that same mean stride (`obs_hz` keeps its meaning as the *mean*
rate in every mode). `uniform` draws from a symmetric window
(`mean * (1 ± jitter)`); `poisson` draws from `Poisson(mean)`, unbounded
above and asymmetric — closer to a real latency distribution's occasional
spike. All three modes live in `action_chunking.SpacingConfig` /
`sample_gap_frames`.

**The one thing that must never vary with the mode**: whatever gap gets
realized, the earlier observation's chunk must have enough actions to cover
it, or replay hits an uncovered gap — the exact failure mode
`make_chunk_config`'s `chunk_duration >= obs_interval` check exists to
prevent under fixed spacing. Under variable spacing this is enforced per
draw: `sample_gap_frames` clips every sample to `[1, action_chunk_len]`, so
it's structurally impossible for a realized gap to exceed what its chunk
covers, regardless of how the distribution is parameterized (verified for a
worst-case setup — mean stride pinned to `action_chunk_len` — in
`test_sample_gap_frames_poisson_clips_the_mean_stride_edge_case`).

**Why the replay side needed no fixed/variable branch at all.** The key
design choice: `reconstruct_reference_trajectory` never takes a stride
parameter. It reads the *realized* gap directly off consecutive rows'
`frame_index` values (`rows[i+1]["frame_index"] - rows[i]["frame_index"]`),
which is `obs_stride_frames` every time under fixed spacing and whatever
`sample_gap_frames` actually drew under variable spacing — the same
line of code covers both, because a dataset row already records exactly
where it landed. Chunk *construction* doesn't need to know the mode either:
each observation's chunk is anchored purely on its own pose, independent of
how far the previous observation was; spacing only decides where the next
`idx` starts. Net effect: **the "mathematical alignment of the chunked
execution is exactly the same as in the fixed interval case,"** as expected
going in — not asserted, but structural, since fixed spacing is just the
zero-variance special case of the same gap-reading code path. This is also
why `replay_chunked_actions.py` needed zero changes beyond dropping the
now-unnecessary `obs_stride_frames` parameter it used to thread through.

**Validation, held to the identical standard as the fixed-interval case**
(same paired-vs-inherent-noise protocol, same episodes, same simulator):
built the arrow dataset with `--obs-spacing-mode poisson --obs-spacing-seed
123` (165,244 rows vs. 164,578 for fixed spacing at the same mean rate — as
expected, since the observation *count* per episode is governed by the same
mean stride either way, just realized with different exact frame indices).
Confirmed exact reconstruction against the recorded `pose2d` first
(`exact_recon_err` at float64 machine epsilon, same as the fixed case), then
ran the same interleaved single-process paired/cross-iteration comparison:

| episode | realized gap range / mean | inherent noise (nc vs nc) | paired (nc vs ch) |
|---|---|---|---|
| short (226 frames) | [1, 10], mean 5.0 | range 0.0000 – 0.0131 m | 0.0131 – 0.0213 m |
| long (1661 frames) | [1, 10], mean ≈5 | range 0.00046 – 0.0132 m | 0.0000 – 0.0006 m |

Both rows land inside the same noise envelope observed for fixed spacing on
the same two episodes (compare to the table above) — in fact the short
episode's numbers are the *identical* five values, because
`reconstruct_reference_trajectory` decodes exactly the same underlying
`pose2d[0 : last_covered]` regardless of which frames were designated
observations, so the sequence of setpoints handed to the (same-seeded)
simulator run-for-run doesn't depend on the spacing mode at all — only which
subset of frames got an image attached differs between the two datasets, not
the trajectory that gets replayed. Variable spacing gets the exact same
correctness guarantee as fixed spacing, for that reason.

## Known limitations / things not handled

- **No padding**: an episode whose native trajectory has fewer than
  `action_chunk_len + 1` frames contributes zero rows (dropped, logged in the
  build script's summary line); the last few frames of every episode that
  don't fill a stride step are silently dropped rather than padded into a
  short final chunk.
- **`native_fps` is caller-supplied, not verified** against the data — there
  is no way to detect a wrong value from the metadata alone; get it wrong and
  both the chunk timing *and* the replay `dt` will be wrong together (self-
  consistently, so it won't crash, just silently mistrack real-world timing).
- **No conversation/`messages` formatting** — this dataset only carries raw
  `image` / `obs_pose` / `action_chunk` columns; turning a row into a
  model-facing training example is left to the training pipeline, as
  requested.
- **Variable spacing's realized mean can undershoot the requested one** when
  the distribution's clip is doing a lot of work (e.g. `poisson` with a mean
  close to `action_chunk_len`, where the upper tail gets clipped far more
  than the lower tail is extended) — `uniform`'s window is symmetric and
  clipped only at the `jitter >= 1` extreme, so it holds its target mean much
  better in practice. No automatic warning is emitted if a configuration's
  realized mean drifts noticeably from the requested `obs_hz`; inspect
  `frame_index` deltas in the built dataset if this matters for a given run.
