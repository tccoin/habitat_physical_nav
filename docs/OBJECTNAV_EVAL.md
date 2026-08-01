# ObjectNav policy evaluation (`objectnav_eval`)

Closed-loop evaluation of a navigation policy on standard ObjectNav episodes,
driven through `HabitatRobotSim` and scored with the habitat-lab metrics that
`continuous_demos.objectnav` already provides.

The task layer answers *"what are this episode's metrics?"*. This package
answers *"what happens when you run a policy over a whole split, on several
GPUs, and want to believe the number at the end?"* — which turns out to be a
different question with its own failure modes, most of them silent.

---

## Quick start

```bash
# 1. What would run, and what would be skipped. No model, no rollout.
python scripts/eval_objectnav_policy.py --dry-run \
    --episodes /path/to/objectnav/hm3d/v1/val_mini \
    --scene-root /path/to/data/scene_datasets

# 2. The harness controls. Run these FIRST on any new setup.
python scripts/eval_objectnav_policy.py --policy forward \
    --episodes .../val_mini --scene-root .../scene_datasets \
    --output-dir dump/eval_forward --max-steps 200 --gpus 0,1

# 3. A checkpoint, four GPUs, real-time video.
python scripts/eval_objectnav_policy.py \
    --ckpt dump/vector_sft/checkpoint-2500 \
    --episodes .../val_mini --scene-root .../scene_datasets \
    --output-dir dump/eval_ckpt2500 --max-steps 200 --record-video \
    --gpus 4,5,6,7 --policy-python /workspace/conda/envs/longnav_vlm/bin/python

# 4. Recover the summary of a partial or crashed run, without re-running it.
python scripts/eval_objectnav_policy.py --report dump/eval_ckpt2500/shards

# 5. Is the difference between two runs bigger than this episode set can resolve?
python scripts/compare_policies.py --baseline dump/eval_a --runs dump/eval_b
```

Run from the repo root, in the environment that has `habitat_sim` and
habitat-lab. `--policy-python` points at the environment that can load the
checkpoint (see [Two environments](#two-environments)).

## Output

| File | What it is |
| --- | --- |
| `objectnav_eval_results[_<policy>].json` | Summary, every per-episode record in full, every skipped episode, and the exact config the run used. |
| `objectnav_eval_results[_<policy>].jsonl` | One compact line per episode — **including skipped ones**, with their reason. This is the file to read. Carries the outcome (`success`, `spl`, distances), the tracking summary (`undershoot`, `lag_m`) and the behavioural summary (`frozen_frac`, `turn_frac`, `drive_frac`, `obstructed_frac`, `collisions_per_m`, `flip_rate`). The full probe output is in the `.json`. |
| `videos/<uid>.mp4` | One frame per control tick, so playback at `1/dt` fps is real time. |
| `shards/shard_N*.{json,log}` | Per-worker spec, raw results and log. `--report` re-aggregates from here. |
| `logs/policy_shardN.log` | The model-side interpreter's stdout/stderr. |

---

## Architecture

Each stage is a module with a small interface, so a new dataset, policy head,
metric or diagnostic is an addition rather than an edit.

```
 episodes.py ──► screening.py ──► harness.py ──► runner.py ──► reporting.py
  EpisodeSource   EpisodeScreener   scene loop     rollout      JSONL + summary
       │                                             │
       │                                    ┌────────┼────────┐
       │                                    ▼        ▼        ▼
       │                              policies.py executor.py probes.py
       │                                    │      ChunkExecutor  tracking + behaviour
       │                                    ▼
       │                               bridge.py ──► scripts/policy_server.py
       ▼                                                (other interpreter)
  parallel.py  shard + fan out over GPUs        compare.py ──► scripts/compare_policies.py
                                                 two runs, paired, with an MDE
```

| Module | Responsibility |
| --- | --- |
| `episodes.py` | `EvalEpisode`, the `EpisodeSource` interface + registry, `EpisodeFilter`. |
| `navmesh.py` | Scene/navmesh resolution, `robot_position_3d`, geodesic queries. |
| `screening.py` | Which episodes this robot can win, with machine-readable reasons. |
| `policies.py` | `Policy` / `Observation` / `ActionChunk`, the image-blind controls, the registry. |
| `bridge.py` | The sim↔model socket, and the `PolicyBackend` interface for head types. |
| `executor.py` | `ActionChunk` → setpoints → PID-tracked motion, emitting `TickRecord`s. |
| `probes.py` | Per-episode diagnostics: `TrackingDiagnostic`, and the behavioural probes `MotionTaxonomy` / `CollisionProbe` / `MotionCoherence`. |
| `compare.py` | Paired comparison of two runs, and what this episode count can resolve. |
| `task.py` | `RobotObjectNavTask`: the task layer with a live query height + a metric seam. |
| `runner.py` | One episode: the rollout loop and the result record. |
| `harness.py` | The scene loop; `--dry-run` lives here too. |
| `parallel.py` | Load-balanced sharding and worker processes. |
| `reporting.py` | Aggregation, JSONL, printed summary, `--report`. |
| `config.py` | `EvalConfig` — every knob, in one serialisable object. |

### Extension points

**A new dataset.** Subclass `EpisodeSource`, return `EvalEpisode`s, and
`register_source("mydata", MySource)`. Then `--episode-source mydata`. The only
contracts are that `uid` is unique across the whole return value and
`view_points` is an `(N, 3)` array in the habitat world frame.

**A new policy head type** (e.g. a normalizing-flow head). Implement
`bridge.PolicyBackend` — `reset`, `act(rgb) -> (T, 3)`, optionally
`wants_stop()` — and `register_backend("flow", loader)`. Run with
`--policy-backend flow`. Nothing on the simulator side changes, because the
wire format does not. Head-specific outputs (log-probs, sample sets) ride in
`ActionChunk.extra` and land in the episode record without widening the
interface. A head with a termination output sets `ActionChunk.stop` and the
runner ends the episode there — `success` then means what it means in habitat.

**A new metric.** Write a habitat `Measure` subclass (see `OracleSuccess` in
`continuous_demos/objectnav.py` for the pattern) and pass it in
`build_task(..., extra_measures=[MyMeasure])`. It appears in every episode
record automatically. Add its name to `reporting.RATE_KEYS` or
`DISTANCE_KEYS` for it to be aggregated.

**A new diagnostic.** Implement `probes.RolloutProbe` — `reset`, `on_tick`,
`on_chunk`, `summary()` — and add it to `probes.default_probes()`. Probes see
every control tick and never touch the simulator, so they cost nothing and
cannot perturb a run. A diagnostic that genuinely needs to *query* the simulator
(the collision probe does) has the executor sample it and hand the result over
on the `TickRecord`, which keeps that rule intact.

Because every head runs through this one suite — `vector_rollout`, `bin_rollout`
and whatever registers next — a probe added here is inherited by all of them and
its numbers are comparable across them by construction. That is the reason to
put a diagnostic here rather than in the experiment that wanted it.

---

## The things that are easy to get wrong

These are not style notes. Each one silently produces plausible, wrong numbers.

### 1. The robot's position is not `robot.translation`

The base link is fixed at the spawn point; the robot travels on world prismatic
joints. `robot.translation` therefore never changes during an episode, and
reading it pins every goal distance to the spawn — the distance-to-goal trace
comes out flat and every policy looks equally useless. The live position is
`get_2d_pose()` → `[X, Y, theta]` with `X = habitat_x`, `Y = -habitat_z`, and
the vertical is the base translation plus `joint_z`. See
`navmesh.robot_position_3d`.

The same applies to the *query height* used for geodesics. The task layer's
`NavSimAdapter._height()` is `translation[1] - vertical_offset`, which is exactly
the navmesh-level height **at spawn** and stays there all episode.
`task.LiveHeightNavAdapter` adds `joint_z` so it tracks vertical travel;
`--frozen-height` restores the task layer's own behaviour if you want to compare.

### 2. Two navmeshes, and episodes that are unwinnable on ours

`build_robot_sim` recomputes the navmesh for the robot's footprint (radius
0.28 m, height 1.0 m). The episodes were generated on the mesh shipped with the
scene, cut for a 0.1 m point agent. The robot's mesh is strictly smaller and
more fragmented. On HM3D `val_mini`, **16 of 30 episodes have the goal on a
different navmesh island from the start** — the robot does not fit through
whatever connects them.

Those episodes are screened out before any compute, each with a reason code,
and both distances are recorded (`geodesic_start_m` on the robot mesh,
`geodesic_start_dataset_m` on the shipped one) so the gap stays visible. They
are counted in the summary and present in the JSONL with `status: "skipped"`.
They are never silently dropped and never counted as failures — a harness that
did either would report a very different, and much worse, number.

A handful of episodes also *start* inside the success radius (the dataset
guarantees a minimum start distance on **its** navmesh, not on ours); those are
screened as `trivially_successful`. `--keep-trivial` scores them anyway.

### 3. The shipped navmesh has a different filename per dataset

HM3D is `<scene>.basis.glb` → `<scene>.basis.navmesh`; MP3D is `<scene>.glb` →
`<scene>.navmesh`. A `str.replace(".basis.glb", ".basis.navmesh")` returns the
path *unchanged* for MP3D — and that path exists, because it is the `.glb`. An
existence check alone therefore hands the navmesh loader a scene file.
`navmesh.dataset_pathfinder` checks the resolved suffix.

### 4. MP3D reuses `episode_id` within a scene

`x8F5xyUWy9e` has two episodes numbered 3, for different goal categories, so
`scene:id` is not a key. `EvalEpisode.uid` carries an occurrence suffix
(`scene:3#2`). Without it the two share a video filename and a shard-membership
test, and one silently overwrites the other.

### 5. Non-finite distances poison averages

A run can end with the robot off its navmesh — in a nook the mesh does not
cover, or on a disconnected island — and the geodesic is then `inf`. That is a
legitimate failure and stays a failure in `success`/`spl`, but averaging it into
a mean distance makes the whole mean `inf`. Distances are averaged over the
finite episodes only and the excluded count is reported as
`episodes_off_navmesh`. Non-finite values serialise as `null`, so every JSONL
line is strict JSON (`json.dumps` otherwise emits bare `Infinity`, which most
parsers reject).

### 6. Episode order matters — the simulator carries state across episodes

**`HabitatRobotSim.reset()` does not fully restore the physics state.** Measured
with the *stateless* `forward` control (so the policy cannot be the cause), the
same episode executes a different trajectory depending on whether another
episode ran before it in the same simulator:

| episode | run alone | after 1–2 other episodes | Δ |
| --- | --- | --- | --- |
| `TEEsavR23oF:87` | 1.798399 m | 1.824174 m | +1.4% |
| `TEEsavR23oF:93` | 2.900846 m | 2.939657 m | +1.3% |

`--fresh-sim-per-episode` rebuilds the simulator before each episode and
reproduces the run-alone trajectory **exactly** (Δ = 0.000000 m on both), which
both confirms the diagnosis and provides the workaround. It costs one scene load
per episode, which is why it is off by default.

Under a closed-loop policy this amplifies: on `val_mini`, episode
`TEEsavR23oF:75` reaches the goal (min distance ~0.00 m, a success) when run
alone and fails (min distance 3.3–3.7 m) when run after another episode — in
*both* this system and the reference harness, which share the simulator.

Consequences, none of which are specific to this harness:

* **Aggregate metrics are stable; per-episode ones are not.** Compare runs at the
  aggregate level, and hold shard composition fixed when comparing checkpoints.
* Worker shards resolve their episode list against the **selected** (canonical,
  sorted) order rather than file order, so a parallel run and a single-process
  run execute the same episodes in the same order.
* For a per-episode number you intend to quote, re-run that episode with
  `--episode-ids <uid>` or use `--fresh-sim-per-episode`.

### 7. Sharding by scene idles workers

Scene load dominates a short episode, so episodes of one scene are kept
together — but only up to a point. `val_mini` has two scenes; splitting purely
by scene leaves two of four GPUs doing nothing. `parallel.shard_episodes` cuts
scenes into blocks no larger than an even share and packs blocks onto the
least-loaded worker, so a scene is reloaded only when that is the price of using
an otherwise idle GPU.

Workers receive **the whole `EvalConfig` as JSON** in their shard file, not a
rebuilt command line. Rebuilding argv flag by flag is how a flag silently stops
being forwarded, and the resulting run looks fine while measuring something
else.

---

## Two environments

The simulator lives in the `vln` conda env (habitat_sim, habitat-lab,
transformers 4.51) and the policy in `longnav_vlm` (transformers 4.57 +
flash-attn, no habitat_sim). This is not incidental — the dependency sets
genuinely conflict — so the boundary is part of the design.

`bridge.PolicyBridge` creates a unix socket, launches
`<policy-python> scripts/policy_server.py`, and blocks on `accept()` until the
model has loaded and connected, so a checkpoint that fails to load surfaces as a
startup error with the server log attached rather than as a hang mid-episode.
Three message types (`reset` / `act` / `stop`); frames cross as raw bytes plus
shape and dtype rather than pickled arrays, because the two envs also disagree
about numpy and unpickling arrays across that is asking for trouble.

`scripts/policy_server.py` imports only `objectnav_eval.bridge` (numpy +
stdlib), so it loads in an environment with no habitat at all. Only these two
files know the boundary exists.

---

## Validating the harness before trusting a number

Run the image-blind controls first. They are **not** baselines worth reporting;
they are how you find out whether the loop itself works.

```bash
--policy zero      # stands still. Path length must be ~0 and success 0.
--policy forward   # drives straight ahead at --control-speed.
```

`zero` pins the floor: it bounds what a policy that has learned nothing scores,
and its path length (~0.01 m over 14 episodes × 1000 ticks) proves the executor
does not inject motion of its own. `forward` proves the opposite — that chunk
decoding, anchoring and PID tracking actually move the robot and that the goal
distance responds. If `forward` cannot reduce the distance to a goal in front of
it, no model number measured through this loop means anything.

Read `forward`'s *task* numbers with care, though: on a cluttered split a blind
forward walk can score surprisingly well, so it is also a useful sanity floor
for the model.

---

## The PID tracking diagnostic

Training demonstrations were generated by a DWPP path controller; evaluation
executes the policy's chunks through a PID *pose* controller. If the PID cannot
keep up, closed-loop numbers understate the policy and we are measuring the
executor rather than the model.

`TrackingDiagnostic` records, per control tick, the commanded displacement
(setpoint to setpoint), the achieved displacement (pose to pose) and the lag
(pose to setpoint), plus the same for heading and the controller's own
saturation counters. It is reported per episode under `"tracking"` and
aggregated in the summary:

```
  --- PID tracking diagnostic (14 episodes, 14000 control ticks) ---
  per tick         commanded 0.0500 m  ->  achieved 0.0393 m   (undershoot 0.787)
  per chunk        commanded net 0.2500 m  ->  achieved 0.1933 m   (undershoot 0.773)
  tracking lag     mean 0.0481 m   p95 0.1308 m   end-of-chunk 0.0833 m
  saturated        v 0.0%  w 0.0%  a 5.0%  aw 0.0%
```

**It is a diagnostic, not a gate.** Nothing in the harness branches on it.

Two caveats when reading it:

* **Undershoot conflates controller lag with physical blocking.** A robot
  pressed against a wall commands 0.05 m/tick and achieves ~0, which reads as an
  undershoot of 0.05. Compare `undershoot_ratio` against a run in free space, or
  restrict to episodes whose path length is close to the commanded total.
* **Re-anchoring resets the transient.** Every chunk is anchored on the robot's
  *live* pose, so the tracker starts each chunk from zero lag and never reaches
  steady state. The transient loss is therefore paid once per chunk and does not
  amortise — which is why `chunk_undershoot_ratio` depends strongly on `--gap`.

See `/Projects/spatial_training/dump/eval_system/FINDINGS.md` for the measured
numbers.

---

## The behavioural diagnostics

Success, SPL and distance say whether the robot arrived. They say nothing about
*how it looked getting there* — and that is where policy heads differ. A
regression head creeps forward without ever committing to motion; a discrete-bin
head moves confidently but wiggles, crashes into things and freezes. Both can
score the same SPL. Three probes turn those observations into numbers.

All three are **on by default** and all three are **diagnostics, not gates**.

### 1. Motion taxonomy (`"motion"`)

Every control tick is classified with the tick taxonomy the demonstration corpus
is characterised with (`dump/data_diagnostics/croppability/segment_decomposition.py`,
via `dump/regen_readiness/tick_decomposition.py`), so an eval number reads
straight against the corpus composition table:

| class | body-frame test |
| --- | --- |
| `DRIVE` | `|dxy| >= XY_T` — translating |
| `TURN` | `|dxy| < XY_T` and `|dtheta| >= TH_T` — turning in place |
| `SETTLE` | `|dxy| < XY_T` and `EPS <= |dtheta| < TH_T` — controller ringing |
| `FROZEN` | `|dxy| < EPS` and `|dtheta| < EPS` — truly idle |

Two things had to change to make the corpus thresholds mean the same thing here.

* **They are stored as speeds, not per-tick displacements.** `XY_T = 0.01 m` and
  `TH_T = 0.02 rad` are defined *per tick at 20 Hz*, so at face value they denote
  0.2 m/s there and something else at any other rate — which would silently
  change what "frozen" means the moment `--dt` changed. They live in `probes.py`
  as **0.20 m/s** and **0.40 rad/s** and are multiplied back by the run's `dt`.
* **`frozen` and `turning in place` are kept apart.** Rotating without
  translating is not standing still — it is how the robot acquires a new view —
  and merging them under one "displacement below threshold" test has misled this
  project before.

The headline "effectively not moving" number is therefore
`frac_stationary = frac_settle + frac_frozen`, which is identical by
construction to `frac_stationary` in `analyze_chunk_motion.py`. `frac_turn` sits
beside it, never inside it.

`FROZEN` itself is near zero in any physics rollout, and that is expected rather
than a bug: the corpus's `EPS` is "the generator wrote the same pose twice", and
a Bullet base always jitters at the solver's noise floor. Read `frac_stationary`.

Everything is computed **twice**, on the commanded setpoints and on the achieved
poses. The gap between them is `blocked_drive_frac` — commanded `DRIVE`, achieved
not-`DRIVE` — which is the blocked-motion signal `undershoot_ratio` documents
itself as unable to separate from PID lag.

### 2. Collisions (`"contact"`)

**The robot is always in contact with the floor.** A healthy episode sits at
roughly 6 broadphase pairs and 2 contact points with nothing whatsoever wrong
(`dump/drive_failure_detection/FINDINGS.md`), so a raw contact count is not a
collision signal — the baseline is non-zero. `executor.robot_contacts` splits
ground from obstruction on the **verticality of the contact normal** (within 45°
of vertical is something the robot is standing on; anything else is something it
is pressing into) and drops self-collisions between the gantry's own links.

Collisions are counted as **maximal runs, not ticks**: leaning on one wall for
200 ticks is one collision, and counting ticks would let a single stuck episode
dominate every aggregate. `obstructed_while_driving_frac` is the number the
question usually means — of the ticks where the policy was trying to go
somewhere, how many were spent against something that was not the floor.

That findings document *rejected* contact-based signals for escape detection,
because that had to be bulletproof enough to accept or reject episodes. This
gates nothing, so a contact measure is the right tool here. Its limits, stated
rather than hidden:

* It is an **instantaneous end-of-tick sample**. A glancing contact entirely
  inside a tick is invisible.
* The 45° normal split **misreads a steep ramp or stair tread** as an
  obstruction, and a low sill the robot rides over as ground.
* It says the robot is *touching* something, not that the something stopped it.
  Cross it with `blocked_drive_frac` for that.

`--no-collision-probe` turns it off. It is on by default because the cost was
measured, not guessed: over four runs of 3,000 control ticks it is inside
run-to-run noise (6.65 / 6.64 s with, 6.83 / 6.72 s without), and in a model run
policy latency dominates by two orders of magnitude anyway.

### 3. Motion coherence (`"coherence"`)

The direct quantification of "drunken". A head that models its output slots as
**fully independent** — the bin head's 30 slots are 10 ticks × 3 channels — has
no correlation between consecutive ticks, and none between forward speed and
rotation within a tick, despite the two being strongly anti-correlated in real
driving. Each individual tick is plausible; the sequence is incoherent.

| field | what it is |
| --- | --- |
| `flip_rate_within_chunk` | of adjacent tick pairs where both genuinely rotate, how often commanded rotation **reverses**. The headline. |
| `flip_rate_all_adjacent` | the same without the within-chunk restriction. |
| `rotation_directness` | per chunk, `\|Σdθ\| / Σ\|dθ\|`. 1.0 is a committed turn; ~0 is rotation that cancels itself out. |
| `forward_rotation_corr` | `corr(\|forward\|, \|dθ\|)` over ticks. Negative in real driving; ~0 for independent sampling. |

**Within-chunk is the headline, deliberately.** Across a chunk boundary the
policy has seen a new image and re-anchored, so reversing there is legitimate
information. Within a chunk it is one forward pass contradicting itself, which is
exactly the signature of independent-per-tick sampling.

Rotations below a deadband (a quarter of the `TURN` threshold, 0.1 rad/s) are
excluded from the pair count: asking which way a 10⁻⁵ rad drift pointed is
meaningless. `flip_rate_within_chunk_turning` is the same statistic at the full
`TURN` threshold, which cannot be inflated by near-deadband jitter.

### What "coherent" looks like

None of these numbers mean anything without a reference. Computed on 1,500
validation demonstrations with the *same estimator*
(`/Projects/spatial_training/dump/eval_system/reference_behaviour.py`, 1.18 M
ticks):

| statistic | real demonstrations |
| --- | --- |
| `flip_rate_within_chunk` | **0.005** |
| `rotation_directness` | **0.983** (median 1.000) |
| `forward_rotation_corr` | **−0.124** |
| `frac_drive` / `frac_turn` / `frac_stationary` | 0.302 / 0.293 / 0.405 |

A human-driven demonstration essentially never reverses its turn direction
mid-chunk. That is the bar.

---

## Reference: the knobs that change what is measured

| Flag | Why it matters |
| --- | --- |
| `--gap` | Control ticks executed per chunk. Must equal the training observation stride, or the policy is evaluated at a speed it never saw. |
| `--dt` | Control tick length. Must match the demonstrations' true control rate. |
| `--success-distance` | Also the screening threshold for trivially-successful episodes. |
| `--evaluate-every` | Ticks between measure updates. `1` (default) matches the reference harness; higher is cheaper but makes the oracle metrics lower bounds. |
| `--snap-start` | Off by default: the start is navigable on the dataset's mesh, and snapping onto the robot's smaller mesh would silently move the agent away from the pose the episode specifies. |
| `--auto-stop` | Ends the episode on arrival, making `success == oracle_success` by construction. Off by default so both numbers stay informative. |
| `--policy-decode` | Decode rule for a distributional head (`bin_rollout`): `argmax` / `sample` / `mean`. Changes the policy as much as changing the checkpoint does — a sampled chunk and a mean chunk from the same weights behave nothing alike. Ignored by backends that do not accept it. |
| `--policy-seed` | Base seed for a backend's per-episode seeded sampling. Combined with an episode counter, so re-running an episode list reproduces itself and an episode's result does not depend on how many ran before it. |
| `--no-collision-probe` | Drops the per-tick contact sample. Measured at inside run-to-run noise, so there is rarely a reason. |
| `--max-scenes` / `--sample-per-scene` | Stratified subsetting for big splits (MP3D val is 2195 episodes over 11 scenes). Seeded, so the same flags name the same episodes. Sampling per scene, not a head slice — the files are grouped by scene, so a head slice would evaluate one scene and call it a split. |

---

## What this cost the numbers, measured

`--fresh-sim-per-episode` is not a theoretical concern. Full 14-episode run of
`checkpoint-2500` on `val_mini`, identical in every respect except that flag:

| metric | shared simulator | fresh simulator |
| --- | --- | --- |
| success | 0.214 | **0.357** |
| oracle_success | 0.214 | **0.357** |
| spl | 0.095 | **0.166** |
| oracle_spl | 0.168 | **0.299** |

Two episodes flip from failure to success. The reference harness shares the same
simulator and the same defect, so its published numbers are depressed the same
way. Quote numbers produced with `--fresh-sim-per-episode`, or state that shard
composition was held fixed.

Full write-up, including the bisection that established this:
`/Projects/spatial_training/dump/eval_system/FINDINGS.md`.
