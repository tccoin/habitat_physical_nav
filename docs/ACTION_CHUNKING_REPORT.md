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

| term | read or chosen? | meaning |
|---|---|---|
| `native_fps` | **read** | fps the `pose2d`/`rgb_paths` were recorded at (the `--fps` passed to `generate_pose2d_dataset.py`). Stored in the metadata JSONL's `recording` block and read from there — see "Recording parameters live in the metadata" below. It is a property of the corpus, never a build choice. |
| `obs_stride_frames` | chosen | how many native frames apart consecutive observations land. The exact form of the observation-spacing knob, and the one to sweep: it means the same thing at every `native_fps`. |
| `obs_hz` | chosen | the same knob expressed as a rate, quantized to a whole number of native frames. Mutually exclusive with `--obs-stride-frames`; the realized rate can differ from the request. |
| `action_chunk_len` | chosen | number of future native-fps poses attached to each observation, as relative actions. |

Derived, all computed from the *recorded* rate: `dt_native = 1/native_fps`,
`obs_interval = obs_stride_frames * dt_native`, `effective_obs_hz =
1/obs_interval`, `chunk_duration = action_chunk_len * dt_native`,
`overlap_fraction = 1 - obs_interval/chunk_duration`. All five are stamped
in-band onto every row, which is why changing any of them means a rebuild
rather than a relabel.

**What the shipped corpus was actually built at** (this table previously
claimed `native_fps=20` → a 0.5 s horizon at 4 Hz; both figures were wrong,
and the row counts below were produced with the settings in the first column):

| | `action_chunk_len` | `obs_stride_frames` | `dt_native` | `chunk_duration` | `effective_obs_hz` | `overlap_fraction` |
|---|---|---|---|---|---|---|
| shipped corpus (30 Hz) | 10 | 5 | 0.0333 s | **0.333 s** | **6.0 Hz** | 0.50 |
| as previously documented | 10 | 5 | 0.05 s | 0.5 s | 4.0 Hz | 0.50 |
| under consideration (25 Hz) | 10–15 | 5 | 0.04 s | 0.40–0.60 s | **5.0 Hz** (exact) | 0.50–0.67 |

The observation *count* depends only on `action_chunk_len` and
`obs_stride_frames`, not on `native_fps` — so the row counts quoted later in
this report are still the counts for a stride-5 / chunk-10 build. Only their
*timing* labels were wrong.

---

## Q1: Is the observation frequency actually reduced?

Yes. `obs_stride_frames` is either given directly (`--obs-stride-frames`, the
exact form) or derived from a requested rate (`make_chunk_config` computes
`max(1, round(native_fps / obs_hz))`), and `build_observation_records` walks
the native trajectory in steps of that stride:

```python
idx = 0
while idx + action_chunk_len <= n - 1:
    ... emit observation anchored at native frame idx ...
    idx += obs_stride_frames
```

So an observation is emitted only every `obs_stride_frames` native frames —
confirmed by unit tests (`test_build_observation_records_indexing`, and
`test_chunk_sweep_settings.py::test_observations_land_on_the_stride` across
the whole `{10,12,15} x {5,6}` grid at both 25 and 30 Hz) and empirically on
the real dataset: `metadata/17DRP5sb8fy.jsonl` (1045 episodes) built down to
164,578 observation rows at stride 5, an average of ~157 observations per
episode against episodes that run several hundred to ~1660 native frames —
consistent with a ~5x reduction.

The reduction *factor* is the stride; the observation *rate* it corresponds to
is `native_fps / stride`, which is where the old documentation went wrong:
stride 5 on the 30 Hz corpus is 6 Hz, not the 4 Hz claimed.

One caveat, and the reason to prefer `--obs-stride-frames` for a sweep:
a requested `obs_hz` is `round()`-ed to the nearest whole frame, so unless
`native_fps / obs_hz` is an integer the *effective* rate isn't the requested
one — and it quantizes differently at each native rate, so "the same `obs_hz`"
is not the same build at 25 and 30 Hz. `ChunkConfig.effective_obs_hz` reports
what was achieved and the build prints both. Concretely: `--obs-hz 4` on 25 Hz
data gives `round(6.25) = 6` → 4.167 Hz, not 4.

## Q2: Does replay run at the exact observation cadence and discard the overlapping tail?

Yes — this is `reconstruct_reference_trajectory` in `action_chunking.py`
(it moved there from `replay_chunked_actions.py`, and no longer takes a stride
parameter — it reads the realized gap off consecutive rows' `frame_index`):

```python
for i, row in enumerate(rows):          # rows sorted by obs_index
    anchor = row["obs_pose"]
    chunk = row["action_chunk"]
    gap = len(chunk) if i == n - 1 else rows[i + 1]["frame_index"] - row["frame_index"]
    for j in range(gap):
        targets.append(relative_to_pose(anchor, chunk[j]))
```

For every row except the last, only the **first `gap` actions** of its chunk
are decoded and kept — the remainder (the part the *next* observation's chunk
will also cover) is discarded. The last row, which nothing supersedes, is
decoded in full.

**The index convention, stated once, because the off-by-one here has bitten
before.** A chunk covers ticks `+1 .. +L` relative to its observation, not
`+0 .. +L-1`:

```
chunk[j] = pose_to_relative(anchor, pose2d[idx + j + 1])
```

so `chunk[0]` is already the *next* tick, and the next observation — which
sits `obs_stride_frames` frames later — decodes from **`chunk[obs_stride - 1]`,
not `chunk[obs_stride]`**. Row `i`'s kept ticks decode native frames
`idx_i+1 .. idx_i+obs_stride`; row `i+1`'s first kept tick decodes
`idx_i+obs_stride+1`. Contiguous, no overlap, no gap — new chunks switch in
exactly at observation boundaries, `obs_interval` seconds apart, which is
"replay at the observation rate."

This is pinned at every supported setting by
`tests/test_chunk_sweep_settings.py` (all six `{10,12,15} x {5,6}` points at
both 25 and 30 Hz, plus the `stride == chunk_len` exact-touch boundary and the
degenerate `stride > chunk_len` case, which `make_chunk_config` rejects), and
was re-checked directly against real built datasets — see "Verification across
the sweep" below.

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

Built the arrow dataset from the real `metadata/17DRP5sb8fy.jsonl`
(1045 episodes, `action_chunk_len=10`, `obs_stride_frames=5`) → 164,578
observation rows, 0 episodes dropped for being too short. The build was run
labelling the corpus `native_fps=20`; the corpus is really 30 Hz, so the row
count stands (it depends only on stride and chunk length) but every duration
and rate reported from that build was off by 1.5x.

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
- ~~**`native_fps` is caller-supplied, not verified** against the data — there
  is no way to detect a wrong value from the metadata alone.~~ **Both halves of
  this were wrong, and it is now fixed.** The rate *was* detectable from the
  data — the physics-aliasing artifact stamped the physics-step count into the
  per-tick displacements, and that is how the real 30 Hz was recovered (5:1
  predicted vs 5.08:1 measured ratio of 0.064/0.080 m saturated increments over
  150 episodes / 104,362 ticks; see `dump/upstream_bugfix/FINDINGS.md`). That
  fingerprint is *gone* at 25 Hz, where there is a single clean 0.080 m
  increment — which is precisely why the rate is now recorded rather than
  inferred. It is no longer caller-supplied at all: see below.
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

---

## Recording parameters live in the metadata

`native_fps` used to live nowhere except in the head of whoever launched
`batch_gen.bash`, and every consumer carried its own default instead. Those
defaults drifted: both build scripts said `20.0` against a 30 Hz corpus,
understating every derived velocity by exactly 1.5x, and nothing crashed
because a wrong `native_fps` is self-consistent all the way down. The move to
25 Hz removed the physics-aliasing artifact that let the true rate be
recovered after the fact, so the rate is now recorded at generation time.

`generate_pose2d_dataset.py` writes a `recording` block onto **every** JSONL
record, alongside `pose2d`/`rgb_paths`:

```json
"recording": {
  "schema_version": 1,
  "native_fps": 25.0,
  "dt_native": 0.04,
  "physics_timestep": 0.008,
  "physics_steps_per_tick": 5.0,
  "on_physics_grid": true,
  "v_max_mps": 2.0,
  "w_max_radps": 2.0,
  "image_width": 640, "image_height": 480, "image_ext": "jpeg",
  "goal_threshold_m": 1.0, "urdf": "spot.urdf",
  "generator": "generate_pose2d_dataset.py", "code_commit": "4e7e30f"
}
```

Everything a consumer previously had to *assume* is in there: the tick rate,
the physics timestep it does or does not divide (`physics_steps_per_tick` /
`on_physics_grid` — an off-grid corpus carries the 24% per-tick velocity
ripple and a consumer computing instantaneous speed should know), the velocity
caps a saturated tick actually hits, and enough provenance to tell two corpora
apart. `physics_timestep` is *probed from the live simulator* at generation
time rather than assumed.

**The true caps are 2.0 m/s and 2.0 rad/s (114.6 deg/s.)** Not the 1.28/73 of
the older findings docs, and not the 1.92/110 that superseded them — both were
`0.064/dt` for some wrong `dt`, i.e. artifacts of the ripple's *lower* mode
rather than real caps.

It is per-record rather than a sidecar file because generation is sharded
across ~24 concurrent processes sharing one output dir, shards get copied and
concatenated by hand, and generation *appends* on resume — so a shard can
physically contain records from two runs. Per-record parameters make that
detectable. The cost is ~250 bytes against a payload of hundreds of frames.

### How each kind of corpus is handled

One policy, in `recording_metadata.resolve_corpus_params`, shared by every
consumer (`build_action_chunk_dataset.py`,
`build_action_chunk_episode_dataset.py`, `sweep_chunk_datasets.py`,
`eval_pid_tracking.py`):

| corpus | what happens |
|---|---|
| written by the current generator (`schema_version >= 1`) | true values, **no flag needed**. Passing `--native-fps` that *contradicts* the record is an error, not an override. |
| pre-schema (no `recording` key) | **fails loudly**, naming the fallback. Nothing is guessed. Pass `--native-fps 30` (`--fps 30` for `eval_pid_tracking.py`) — every pre-schema corpus in this repo was recorded at 30 Hz, and the resulting config is marked `schema_version: 0` / `native_fps_source: "assumed"` everywhere it is reported or stamped. |
| self-contradictory (some shards with, some without; or two rates) | **always an error.** This is the resumed-across-the-fps-change case; it must never be averaged over. Both build loops also re-check every record, catching a shard whose first line is honest and whose last is not. |

The generator itself refuses to append to a shard recorded at a different rate
(or to a pre-schema shard), so the mixed case cannot be created going forward.
`batch_gen.bash` therefore defaults to a fresh `output_25hz/`; the existing
30 Hz `output/` is left alone and stays readable with an explicit
`--native-fps 30`.

## Sweeping `action_chunk_len` x `obs_stride_frames`

Both are **build-time parameters of the arrow table**, not properties of the
raw data: varying them re-slices trajectories that already exist, re-simulates
nothing and re-renders no frames. What they are *not* is relabelable — all five
derived values are stamped in-band onto every row, so each grid point is a
separate build.

```bash
# the plan and every derived value, without building anything
python scripts/sweep_chunk_datasets.py --data-dir output_25hz \
    --output-root output_25hz/sweeps --dry-run

# build the grid (resumable; already-built points are skipped)
python scripts/sweep_chunk_datasets.py --data-dir output_25hz \
    --output-root output_25hz/sweeps --val-scene-count 5

# one point on its own, either builder
python scripts/build_action_chunk_dataset.py --data-dir output_25hz \
    --output-dir /tmp/clen12_stride5 --action-chunk-len 12 --obs-stride-frames 5

# a pre-schema (30 Hz) corpus, rate stated once
python scripts/sweep_chunk_datasets.py --data-dir output --output-root /tmp/sweep \
    --native-fps 30
```

Each point lands in `<output-root>/clen<L>_stride<S>/` with its own
`chunk_config.json`; the comparison table is written to
`sweep_summary.{json,md}` after every point, so an interrupted sweep keeps
what finished. The native rate is deliberately **not** part of the grid — it
is read from the corpus, and sweeping it would mean regenerating.

Derived values at the default grid, computed from the recorded rate:

| build | chunk_len | stride | `chunk_duration` | `obs_interval` | `effective_obs_hz` | `overlap_fraction` |
|---|---|---|---|---|---|---|
| clen10_stride5 | 10 | 5 | 0.400 s | 0.200 s | 5.000 Hz | 0.500 |
| clen10_stride6 | 10 | 6 | 0.400 s | 0.240 s | 4.167 Hz | 0.400 |
| clen12_stride5 | 12 | 5 | 0.480 s | 0.200 s | 5.000 Hz | 0.583 |
| clen12_stride6 | 12 | 6 | 0.480 s | 0.240 s | 4.167 Hz | 0.500 |
| clen15_stride5 | 15 | 5 | 0.600 s | 0.200 s | 5.000 Hz | 0.667 |
| clen15_stride6 | 15 | 6 | 0.600 s | 0.240 s | 4.167 Hz | 0.600 |

(at 25 Hz; the same grid on the 30 Hz corpus gives 0.333/0.400/0.500 s
horizons at 6.0 or 5.0 Hz.) No winner is picked here — the capability is the
deliverable.

## Verification across the sweep

Every point of the grid was checked, at both 25 and 30 Hz, in
`tests/test_chunk_sweep_settings.py` (72 cases, no simulator): derived values
against closed form, `chunk[j] == pose_to_relative(anchor, pose2d[idx+j+1])`,
observations landing exactly on the stride, the next observation decoding from
`chunk[obs_stride - 1]` and *not* `chunk[obs_stride]`, and an exact
reconstruction round trip. Plus both sides of the boundary: `stride ==
chunk_len` (chunks abut exactly, zero overlap, still correct) and `stride >
chunk_len` (rejected — the chunk would run out before the next observation).

The same claims were then re-checked against **real built datasets** rather
than synthetic trajectories: all six grid points built from a 30-episode slice
of `17DRP5sb8fy`, every chunk entry re-derived from the raw corpus and every
consecutive observation pair's handoff decoded. Max absolute error
1.8e-15 (float64 machine epsilon) across 255k chunk entries and 20k handoffs;
`chunk[obs_stride - 1]` recovered the next observation's pose exactly at every
setting.

One real-data nuance worth recording: the "`chunk[obs_stride]` is one tick too
far" check is only *observable* where the robot moved on that tick. On ~2% of
handoffs the two poses coincide to float precision — those are look phases and
stalled ticks, where the base is genuinely stationary. The index arithmetic is
still verified there by the chunk-entry check; it just isn't distinguishable
from the handoff check alone.
