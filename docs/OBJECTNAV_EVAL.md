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

# 3b. A non-regression head: pick the backend, and (optionally) its decode rule.
python scripts/eval_objectnav_policy.py \
    --policy-backend ar_rollout --policy-decode sample --policy-seed 0 \
    --ckpt dump/autoregressive_head/run_vq256_ga8/checkpoint-225 \
    --episodes .../val_mini --scene-root .../scene_datasets \
    --output-dir dump/eval_ar225 --max-steps 200 --gpus 4,5,6,7 \
    --policy-python /workspace/conda/envs/longnav_vlm/bin/python \
    --policy-sys-path /Projects/spatial_training/src

# 3c. The flow-matching head. One generation mode, so no --policy-decode; the pose
#     modality is read off the checkpoint, so no flag for that either.
python scripts/eval_objectnav_policy.py \
    --policy-backend flow_rollout --policy-seed 0 \
    --ckpt /Projects/spatial_training/dump/pose_injection/run_v8_flow_planar_pose/checkpoint-400 \
    --episodes .../val_mini --scene-root .../scene_datasets \
    --output-dir dump/eval_flow_v8_400 --max-steps 200 --auto-stop --gpus 4,5,6,7 \
    --policy-python /workspace/conda/envs/longnav_vlm/bin/python \
    --policy-sys-path /Projects/spatial_training/src

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
| `objectnav_eval_results[_<policy>].json` | Summary, compact per-episode metrics, every skipped episode, exact config, and references to compressed trajectory/detail files. |
| `objectnav_eval_results[_<policy>].jsonl` | One compact line per episode — **including skipped ones**, with their reason. This is the file to read. Carries the outcome (`success`, `spl`, distances), the tracking summary (`undershoot`, `lag_m`) and the behavioural summary (`frozen_frac`, `turn_frac`, `drive_frac`, `obstructed_frac`, `collisions_per_m`, `flip_rate`). |
| `episode_details/<uid>.json.gz` | Compressed per-step diagnostics such as policy/execution latency, goal-distance, stop-score, injected-pose, and SG mapping/action traces. `result.json` and the JSONL retain scalar summaries plus a `details_file` reference. |
| `videos/<uid>.mp4` | One frame per control tick, so playback at `1/dt` fps is real time. By default it is first-person RGB beside a scene-RGB top-down BEV with travelled route, target, and all success view points; `--no-video-bev` restores first-person-only video. |
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
| `pointnav.py` | PointNav as a one-goal ObjectNav episode: `PointGoal`, `PointNavEpisodeSource`, the `goal_pursuit` control. |
| `goal_sources.py` | `GoalSource`: where the *next* point goal comes from, for the free-running rollout. The seam a GUI would drive. |
| `navmesh.py` | Scene/navmesh resolution, `robot_position_3d`, geodesic queries. |
| `screening.py` | Which episodes this robot can win, with machine-readable reasons. |
| `policies.py` | `Policy` / `Observation` / `ActionChunk`, the image-blind controls, the registry. |
| `bridge.py` | The sim↔model socket, and the `PolicyBackend` interface for head types. |
| `executor.py` | `ActionChunk` → setpoints → PID-tracked motion, emitting `TickRecord`s. |
| `probes.py` | Per-episode diagnostics: `TrackingDiagnostic`, and the behavioural probes `MotionTaxonomy` / `CollisionProbe` / `MotionCoherence`. |
| `budget.py` | `GeometricBudget` / `BudgetRule`: what an episode is allowed to cost, in discrete-step equivalents. See [docs/GEOMETRIC_BUDGET.md](GEOMETRIC_BUDGET.md). |
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
`view_points` is an `(N, 3)` array in the habitat world frame. A source needing
more than a path takes it from `--source-set FIELD=VALUE` (repeatable), which is
forwarded to the factory as keyword arguments — rather than adding a flag per
source to this CLI. `objectnav_eval.pointnav.PointNavEpisodeSource` is the
worked example: `--episode-source pointnav --source-set episodes_per_scene=5
--source-set seed=101`.

**A different task.** PointNav is one, and it is deliberately not a second
harness: a point goal is the degenerate one-view-point ObjectNav episode, so
`objectnav_eval/pointnav.py` supplies an episode object habitat's measures can
read and nothing else changes — `success`, `spl`, `soft_spl`, `oracle_*`,
`npwpl`, the budget and every probe are the same code computing the same
statistic. The two things it does add are a goal fed *per step* rather than as a
category string (`Observation.goal_pose`, forwarded over the bridge) and
`--goal-placement`, which tells the model side where the goal's `<pose>` marker
sits. See `CLAUDE.md` "PointNav rollout and evaluation".

**A new policy head type** (e.g. a normalizing-flow head). Implement
`bridge.PolicyBackend` — `reset`, `act(rgb, pose=None) -> (T, 3)`, optionally
`wants_stop()` — and `register_backend("flow", loader)`. Run with
`--policy-backend flow`. Nothing on the simulator side changes, because the
wire format does not. Head-specific outputs (log-probs, sample sets) ride in
`ActionChunk.extra` and land in the episode record without widening the
interface. A head with a termination output sets `ActionChunk.stop` and the
runner ends the episode there — `success` then means what it means in habitat.

Registered today:

| `--policy-backend` | head | `--policy-decode` |
| --- | --- | --- |
| `vector_rollout` (default) | turn-vector regression (`longnav.utils.vector_rollout`) | n/a — not distributional |
| `bin_rollout` | discrete bins, one categorical per (tick, dimension) (`longnav.utils.bin_rollout`) | `argmax` (default) / `sample` / `mean` |
| `ar_rollout` | autoregressive over ticks: one joint VQ code per tick, decoded sequentially through a small causal transformer (`longnav.utils.ar_action_head`) | `sample` (default) / `argmax` / `mean` |
| `flow_rollout` | flow matching: the whole chunk generated jointly by integrating a learned velocity field from noise (t=1) down to data (t=0) (`longnav.utils.flow_matching_head`, loaded by `longnav.utils.flow_rollout`) | `sample` only — and that is a design decision, not a gap |

All four reach the simulator through `VectorRolloutPolicy` unmodified, because
each head's `normalizer` slot already answers `denormalize(...) -> (T, 3)`. The
backend's whole job is loading the right model class, fixing the decode rule and
seeding the per-episode sampling stream — which is why a new head of this family
is ~40 lines here and nothing anywhere else.

Two things about `ar_rollout` fail *silently* if copied from `bin_rollout`. Its
decode rule lives on the **normalizer** (`model.normalizer.decode`), not on
`model.codec`: for this head `model.codec` is the `ChunkVQCodec` lookup table
*inside* the `ARActionCodec`, so writing `model.codec.decode` sets an attribute
nothing reads and the head decodes under its own default instead. And its
`"context"` decode is the offline passthrough
`dump/autoregressive_head/predict_ar.py` uses, returning the raw context vector
rather than a chunk; the backend rejects it instead of sending a
`(context_dim,)` array down the wire as an action. `"mean"` is accepted but is
computed from a greedy-decoded prefix — the head's own docs call it
reporting-only, not a deployable rule, and it is here for the same
control-within-the-control the bin head kept it for.

`flow_rollout` inverts the first of those traps and refuses the third. Its
`.codec` **is** the normalizer-slot object (this head has no lookup table, so
`model.codec is model.normalizer`), which is why `flow_rollout.seed_sampling`
follows `bin_rollout`'s pattern rather than the AR backend's — with
`codec.action_scales.device` where the bin head reads `codec.centroids.device`.
And `FlowActionCodec.DECODES` is `("sample", "context")` with **no `"mean"`, on
purpose**: averaging several ODE solutions is not a draw from the conditional,
it is the conditional mean — the creeping failure this head exists to remove. So
`--policy-decode mean` is rejected here rather than quietly approximated, and
`"context"` is rejected for exactly the reason it is on the AR head. Do not add
a mean, and do not average chunks at the call site.

Two more things are specific to it:

* **Seeding is unconditional.** The bin and AR backends reseed only under
  `--policy-decode sample` because their other rules consume no randomness. This
  head has one generation mode and it always draws `x_1 ~ N(0, I)`, so every
  episode is reseeded from `--policy-seed + <episode index>`. Without it a
  closed-loop number is not reproducible at all.
* **The number of Euler steps is part of the policy.** It is a checkpoint field
  (`FlowMatchingConfig.num_inference_steps`, restored by `from_pretrained`), so
  by default an eval integrates the ODE the way the run that produced the
  training metrics did. `--policy-inference-steps` sweeps it — the same weights
  at 4 steps and at 10 are different policies — and `describe()` prints the value
  that actually loaded, next to the `action_scales` that turn the field's
  unit-scale output back into metres:

```
  [flow decode=sample seed_base=1000 n_ticks=10 memory=unlimited]
  [flow ODE euler_steps=10 action_scales=(0.03, 0.03, 0.05)]
  [flow field d_model=128 n_layers=4 n_heads=4 n_context_tokens=8 use_incoming_motion=False n_dims=3]
```

**A checkpoint that reads more than the image, or that can stop.** Neither is a
new backend. Both are *properties of the checkpoint*, so both are switched on by
reading it — see [Pose injection and the stop
head](POSE_INJECTION_EVAL.md) for the full writeup.

* **Pose.** `bridge._PoseFeed` looks for a modality spec declaring the
  `pose_relative_first` transform. When one is there, every `act()` passes the
  agent's **raw** planar `[X, Y, theta]` (straight from `get_2d_pose()`) as
  `step(obs_pose=...)`, and the model's own `relative_se2` converts it — the
  same function the training collator calls, never a copy of it. There is no
  flag: a flag could disagree with the weights. Checkpoints without such a spec
  take the identical code path they always did.
* **A point goal.** `--goal-placement` switches the same `_PoseFeed` to the
  interleaved value column PointNav trains on: a goal is another occurrence of
  the *same* `<pose>` marker, so agent rows and goal rows go through one
  accumulator and `relative_se2` anchors on row 0 of the column training saw.
  This one **does** need a flag, for the opposite reason pose does not: the
  placement is a property of how the corpus's text was written
  (`format_action_chunk_dataset.py --goal-placement`) and is recorded nowhere in
  the checkpoint. The row order itself is never decided here — it comes from
  `longnav.utils.vector_rollout.PointNavValueStream`, the incremental form of the
  `modality_value_rows` the training formatter imports.
* **Stop.** `bridge._StopHeadControl` finds `model.stop_head`. The stop
  probability and its logit are recorded every step regardless; `--stop-head` is
  what lets the decision actually end an episode. It is mutually exclusive with
  `--auto-stop` (the config raises), because with both, an episode ends at
  whichever fired first and neither can be read on its own.

#### The AR decoder's architecture is not fixed, and the eval log has to say which one ran

`CausalActionDecoder` carries four options that all default OFF, so a checkpoint
written before any of them existed loads under the new code unchanged:

| option | default | what it changes |
| --- | --- | --- |
| `n_context_tokens` | `1` | the context vector is split into this many transformer prefix tokens, lifting the hard `d_model` ceiling on how much context reaches the decoder |
| `use_prefix_pose` | `False` | each tick additionally receives a 7-dim state `[x, y, cosθ, sinθ, dx_prev, dy_prev, dθ_prev]`, composed from the prefix's own decoded codes |
| `attn_mode` | `"causal"` | `"markov"` restricts each tick to the context slots plus itself, making history-based heuristics structurally impossible |
| `direct_context` | `False` | drops `context_proj`; the readout MLP's output reshapes straight into prefix tokens |

**`ar_rollout` needs no change for any of them, and this is not luck.** The
backend only ever calls `TurnARActionClassifier.from_pretrained` (which rebuilds
the decoder from the checkpoint's own `ar_decoder_kwargs`) and sets
`normalizer.decode`; it never constructs a decoder, never reads `context_proj` —
which `direct_context=True` sets to `None` — and never touches the attention
mask. The one option with an external requirement, `use_prefix_pose`, needs
`centroids` passed into `decoder.forward()`/`decoder.decode()`, and
`ARActionCodec.denormalize` already supplies them on both the sequential and the
`"mean"` path. All of it verified end to end: every decode rule against a real
`n_context_tokens=8, use_prefix_pose=True` checkpoint and against the previous
architecture, plus synthetic checkpoints for `markov` and `direct_context`, which
no trained run reaches yet.

One thing that got *looser* rather than broken: `"mean"` with `use_prefix_pose=True`
is now a double approximation. It was already the probability-weighted centroid
given a greedy prefix; the pose/velocity features it conditions on are now also
composed from those greedy codes, while the chunk it returns corresponds to no
code sequence at all. It runs and it is finite, but it is even further from a
deployable rule than before — which does not change how it should be used, since
it was already reporting-only.

What *did* need adding is the record of which one ran. `n_codes` and `n_ticks` are
identical across these runs, so `describe()` — the only architecture statement in
an eval run's output, printed once into
`<output-dir>/logs/policy_shard<N>.log` — now prints the decoder config as well:

```
  [ar decode=sample seed_base=1000 n_codes=1024 n_ticks=10]
  [ar decoder d_model=128 n_layers=4 n_heads=4 n_context_tokens=8 use_prefix_pose=True attn_mode=causal direct_context=False]
```

It reads `CausalActionDecoder.to_config()` — the same dict `save_pretrained`
stores and `from_pretrained` rebuilds from — rather than restating the CLI flags,
so it reports the architecture that actually loaded. Missing keys are printed at
their defaults on purpose: a pre-option checkpoint stores nothing for them, and a
line that silently shortens is not comparable by eye to one that does not.

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

Because every head runs through this one suite — `vector_rollout`, `bin_rollout`,
`ar_rollout`, `flow_rollout` and whatever registers next — a probe added here is inherited by all of them and
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

#### Which one to measure on: `--navmesh`

Both meshes are available, and neither is correct in general — they answer
different questions, so the run records and prints which it used.

| | `--navmesh robot` | `--navmesh dataset` (**default**) |
| --- | --- | --- |
| footprint | r=0.28 m, h=1.0 m — this robot | r=0.1 m point agent |
| connectivity | fragmented; 28% of HM3D `val` screened `goal_unreachable` | connected; screening nearly never fires |
| honest about | where the robot **fits** | comparability with published ObjectNav |
| fails by | *overstating* difficulty, and detours | *understating* difficulty |

The robot mesh's failure mode is worth stating concretely, because it produces a
plausible wrong number rather than an error. **The robot is a physics body, not
a navmesh agent** — nothing constrains it to the mesh. Where the mesh routes
around an obstacle the body can simply drive through, the geodesic overstates
the distance. On MP3D `8194nk5LbLH:39` the geodesic start was 9.15 m to a goal
**2.20 m away in a straight line**; the robot arrived after 1.9 m of travel, the
distance trace fell 9.99 → 0.02 in three policy steps, and **SPL pinned at
1.000** because its reference path was longer than the route actually taken.

`detour_ratio` (geodesic ÷ euclidean at the start) is recorded per episode as
the signal for this. It gates nothing. On the current sets roughly 10% of
episodes exceed 3×.

The dataset mesh removes the detours and the island screening, at the cost of
assuming a robot that fits through gaps this one does not — goals it calls
reachable may be unreachable in fact, and those episodes become honest failures
rather than screened-out ones.

**`dataset` is the default, and `robot` should not be used for any number you
intend to report.** That is a change: every run before 2026-08-06 used `robot`,
because it was the default and this section described both meshes as merely
answering different questions. They do — but only one of them answers its
question *consistently*, and the section above understated how badly the other
fails.

The robot mesh's failure is not only that it overstates distance. The robot is
not constrained to the mesh, so `snap_point` (`objectnav.py:154-157`, unbounded
and island-unconstrained) can resolve the agent's proxy **across an island
boundary**. On `wcojb4TFT35:0#26` a 4.4 cm real step moved the proxy 2.73 m,
`distance_to_goal` fell 6.193 -> 1.189 in one tick, and the metric spent the
rest of the episode measuring a *different bed instance* — one the robot navmesh
calls unreachable from the start. 168 of 320 ticks violated the 1-Lipschitz
bound. Scored on the dataset mesh the same trajectory has **zero** violations.

`detour_ratio` is the cheap signal and it is recorded per episode: on the
`sample101` set the median is 1.39, **28.6% of episodes exceed 2x and 15.7%
exceed 3x**. That is the scale of the distortion, and it is an order of
magnitude more episodes than the outright-impossible ones a
`start - min <= path_at_min` check catches.

The dataset mesh has its own bias — it assumes a robot that fits through gaps
this one does not, so it understates difficulty. But that bias is *stable*: it
does not change which goal is being measured partway through an episode.

**Runs using different `--navmesh` values are not comparable.** Switching it
changes every geodesic, SPL's denominator, and which episodes are screened.

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

### What they resolve that the headline metrics do not

The bin head's two decode rules, checkpoint-7250, same 45 HM3D episodes, paired
(`scripts/compare_policies.py --behaviour`):

| metric | mean | sample | Δ | 95% CI | MDE |
| --- | --- | --- | --- | --- | --- |
| **flip_rate** | 0.000 | 0.114 | **+0.114** | **[+0.105, +0.123]** | 0.013 |
| frozen_frac | 0.558 | 0.497 | −0.061 | [−0.115, −0.010] | 0.076 |
| obstructed_frac | 0.380 | 0.287 | −0.093 | [−0.173, −0.016] | 0.112 |
| success | 0.156 | 0.200 | +0.044 | [−0.111, +0.200] | 0.233 |
| spl | 0.078 | 0.070 | −0.007 | [−0.088, +0.075] | 0.118 |

Success cannot tell these two apart at this episode count — the difference is a
fifth of the minimum detectable effect. The sign-flip rate separates them by
**8.6× the MDE**, and says *why*: sampling 30 independent slots produces a chunk
that contradicts itself, while taking their mean produces one that is smooth by
construction (0.000, below even the demonstrations).

Reading them honestly also means reporting where they disagree with the
impression that motivated them: mean decode turns out to freeze **more** than
sample (0.558 vs 0.497 stationary, longest stall 9.6 s vs 3.4 s), and to spend
more of its time in contact, in longer episodes of it. Sample collides more
*often* (12.2 vs 8.0 events per episode) but is stuck for less of the run. Full
write-up in `/Projects/spatial_training/dump/eval_system/FINDINGS.md` §6.

---

## Reference: the knobs that change what is measured

| Flag | Why it matters |
| --- | --- |
| `--episode-budget` / `--budget-units` / `--budget-time-limit` | **Which currency the episode is capped in.** `steps` (default) is `--max-steps` and nothing else — historical behaviour, and `gap`-dependent, so runs at different observation rates are not comparable under it. `geometric` caps at `--budget-units` units of motion, one unit being 0.25 m driven or 30 deg turned: the geometry of one classic ObjectNav discrete step, so 500 units is the benchmark's own 500-step cap measured on a continuous agent. It always carries `--budget-time-limit` as a backstop (the budget is blind to a frozen agent and to one pushing into a wall, both of which spend nothing) and the config raises without one. `both` keeps the step cap too and reports whichever fired, as `terminated_by`. The budget spent is recorded on **every** run regardless, so a finished run can be re-scored under a different cap with `scripts/rescore_budget.py` rather than re-run. Full derivation, including why there is no scalar seconds↔units conversion, in [docs/GEOMETRIC_BUDGET.md](GEOMETRIC_BUDGET.md). |
| `--gap` | Control ticks executed per chunk. Must equal the training observation stride, or the policy is evaluated at a speed it never saw. |
| `--dt` | Control tick length. Must match the demonstrations' true control rate. |
| `--success-distance` | Also the screening threshold for trivially-successful episodes. |
| `--evaluate-every` | Ticks between measure updates. `1` (default) matches the reference harness; higher is cheaper but makes the oracle metrics lower bounds. |
| `--navmesh` | **Which navmesh every distance is measured on** — see [Two navmeshes](#2-two-navmeshes-and-episodes-that-are-unwinnable-on-ours). `robot` (default) recomputes it for this robot's footprint; `dataset` uses the point-agent mesh shipped with the scene. It does *not* change how the robot moves. Recorded in every results file and printed per scene. |
| `--snap-start` | Off by default: the start is navigable on the dataset's mesh, and snapping onto the robot's smaller mesh would silently move the agent away from the pose the episode specifies. |
| `--auto-stop` | **Oracle termination.** Ends the episode on the control *tick* the agent first comes within `--success-distance` — the stop a policy with no termination head cannot make for itself. Off by default so `success` and `oracle_success` stay independent readings. At `--evaluate-every 1` this makes `success == oracle_success` by construction; it cuts the chunk short rather than running it out, because a policy that arrives on tick 1 and drifts back out by tick 5 would otherwise score below the oracle that had already fired. Above `--evaluate-every 1` arrival is noticed late, so the identity weakens exactly as the oracle metrics do. |
| `--policy-decode` | Decode rule for a distributional head (`bin_rollout`, `ar_rollout`): `argmax` / `sample` / `mean`. Changes the policy as much as changing the checkpoint does — a sampled chunk and a mean chunk from the same weights behave nothing alike. Ignored by backends that do not accept it. Unset means the backend's own default, and those differ: `argmax` for `bin_rollout`, `sample` for `ar_rollout` (its codec's own default, and the only rule its design treats as an executable sequential policy). `flow_rollout` accepts **only** `sample`: it has one generation mode, and it deliberately has no mean because averaging ODE solutions *is* the creeping failure that head was built to remove. |
| `--policy-inference-steps` | Euler steps for the flow-matching head's ODE (`flow_rollout` only). Unset means the checkpoint's own `num_inference_steps`, i.e. the integration its training metrics were computed under — so an eval agrees with the run's own numbers unless it says otherwise. Setting it is a deliberate sweep: the same weights at 4 steps and at 10 are different policies, and the value that ran is printed in the policy shard log. |
| `--stop-head` | **Lets the checkpoint's own stop classifier end the episode.** Mutually exclusive with `--auto-stop` — the config raises rather than picking a precedence. `oracle_success`/`oracle_spl` are unaffected either way (they latch on ever reaching `--success-distance`), so they stay comparable to every auto-stop run; `success`/`spl` stop being pinned equal to them and start measuring the policy's own termination. Expect longer episodes: nothing ends them on arrival any more. |
| `--stop-inference` / `--stop-threshold` / `--stop-temperature` | Runtime knobs on that classifier, unset meaning the checkpoint's own values. They are post-hoc scalars by design and do not change its AP or AUC. Each episode record carries the per-step `(probability, logit)` trace under `"stop"` whether or not the head is acting, so a threshold or temperature can be swept offline instead of by re-running the split. |
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
