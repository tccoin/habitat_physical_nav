# Closed-loop evaluation of a pose-injecting policy, and of its stop head

The pose run injects **ground-truth agent pose** into the conversation at a
`<pose>` marker, one per turn, and carries a **stop classifier** on the same
pooled turn context. Both are new things for this harness to feed and to read.
This is what was built for that, how the one silent failure mode was ruled out,
and what the runs measured.

Mechanism side: `/Projects/spatial_training/dump/modality_embed/DESIGN.md`
(design), `src/longnav/utils/modality_embed.py` (implementation),
`src/longnav/utils/pose_frame.py` (the frame convention),
`src/longnav/utils/stop_head.py` (the classifier).

> **The result this supports is a diagnostic, not a system.** Injecting
> ground-truth pose grants noiseless GPS+compass. That changes the task
> definition, and any number from it is incomparable with a no-GPS baseline.
> This document does not repeat that warning next to every table; it applies to
> all of them.

---

## 1. The one thing that had to be right

Training shows the model its pose **relative to the first observation in
context**, expressed in that observation's own frame. The harness had no
reference to agent pose at all, so it had to start producing that value — and if
the value it produced differed from the one training used, *nothing would
raise*. The model would read a well-formed vector meaning something else, the
rollout would complete, and the run would print a plausible number.

Two ways to get it wrong that are individually reasonable and jointly fatal:

* **previous-frame delta instead of an episode-start anchor.** Every value is
  small, smooth and plausible. It is simply a different quantity.
* **an unwrapped heading difference.** On the real corpus this reaches
  **5.955 rad where the true rotation is −0.328 rad** — measured below, on the
  first validation episode, not quoted from the design doc.

The defence is structural rather than careful: **the harness does not compute the
frame at all.** It sends the raw scene-frame pose and the policy's own
`pose_values()` → `pose_frame.relative_se2` converts it, which is the identical
function `ModalityEmbedSpec.apply_transform` calls at collate time. There is one
implementation and both paths reach it. `bridge._PoseFeed` exists to make that
non-negotiable, and says so in its docstring.

---

## 2. What changed

### `src/objectnav_eval/bridge.py`

Two small classes and a serialisation helper, plus one extra wire field. Neither
class is a backend: **pose and stopping are properties of a checkpoint, not of a
head type**, so a pose-injecting AR head and a pose-injecting vector head are the
same eval problem and share one implementation.

| addition | what it does |
| --- | --- |
| `_PoseFeed` | Finds the spec declaring the `pose_relative_first` transform (by transform, **not** by token name — the transform *is* the contract `step(obs_pose=...)` implements). Turns a raw `[X, Y, theta]` into `{"obs_pose": [...]}`, or `{}` when the checkpoint declares no pose. Raises if a checkpoint that needs a pose is handed none, rather than letting the marker be injected with nothing behind it. |
| `_StopHeadControl` | Finds `model.stop_head`. Validates and writes the runtime knobs (`temperature`, `threshold`, `inference`) onto the loaded `StopHeadConfig` — validated here because assigning to a dataclass field after construction skips its `__post_init__`. Seeds the sampled decision per episode. Gates whether `wants_stop()` can ever be true. Adds `stop_logit` next to `stop_prob`. |
| `_wire_safe` | `last_stats["pose_value"]` is a **torch tensor**. The simulator environment has no torch, so left alone it fails at `conn.recv()` in the other process, an episode away from its cause. Converted, not dropped: it is the one value that proves what the model was shown. |
| `PolicyBackend.act(rgb, pose=None)` | The signature gains the pose. All three backends (`vector_rollout`, `bin_rollout`, `ar_rollout`) take it and ignore it unless their checkpoint asked for it. |
| wire | `PolicyBridge.act` sends `"pose"` as three plain floats — not a numpy array, because the two environments disagree about numpy (1.26 vs 2.x). `serve` reads it with `.get`, so a sim side predating this still talks to a server that has it. |

`stop_logit` is recovered by inverting the reported probability under the
configured temperature, not by a second forward pass. The reason it exists at
all: the episode record rounds floats to 3 decimals, and a head trained with
`pos_weight 90` can easily sit entirely inside `(0.000, 0.005)` — which would be
saved as a column of zeros, and no threshold could be fitted on it afterwards.
The logit has no such dynamic-range problem, and temperature *divides* it, so it
is the only quantity a post-hoc temperature refit can work from.

### Everything else

| file | change |
| --- | --- |
| `policies.py` | `BridgedPolicy`/`InProcessPolicy` pass `obs.pose` through. `Observation.pose` already existed and is already `get_2d_pose()` — nothing transforms it on the way, which is the point (§3). |
| `config.py` | `--stop-head`, `--stop-inference`, `--stop-threshold`, `--stop-temperature`. `__post_init__` **rejects `--stop-head` with `--auto-stop`**. No pose flag, deliberately: a flag could disagree with the weights. |
| `runner.py` | Records the per-step `(stop_prob, stop_logit)` trace under `"stop"`, and the per-step injected pose under `"pose_injected"`. Both stay absent on a checkpoint with neither, so no existing record grows a column. |
| `scripts/policy_server.py` | Forwards the four stop flags, and only when set — so a backend predating them sees exactly the kwargs it always saw. |
| `tests/test_pose_injection_bridge.py` | New. §3. |
| `scripts/verify_pose_injection.py` | New. §3, levels 1 and 2. |

**Production paths are untouched.** Every addition is either gated on a
checkpoint declaring something (`_PoseFeed`, `_StopHeadControl` both return
immediately when the checkpoint has nothing) or on a flag that defaults off. §4
is the empirical check of that claim, not just the argument for it.

---

## 3. How train-vs-rollout agreement was verified

Three levels, each strictly stronger. Levels 1 and 2 are runnable:

```bash
# level 1 -- no model, no simulator, no socket
/workspace/conda/envs/vln/bin/python scripts/verify_pose_injection.py --level transform

# level 2 -- the same episode through a live policy server and a real checkpoint
/workspace/conda/envs/vln/bin/python scripts/verify_pose_injection.py --level bridge \
    --ckpt <pose-checkpoint> \
    --policy-python /workspace/conda/envs/longnav_vlm/bin/python \
    --policy-sys-path /Projects/spatial_training/src --steps 40

# the same claims as assertions
/workspace/conda/envs/vln/bin/python -m pytest tests/test_pose_injection_bridge.py -q
```

### Level 1 — the transform, on a real episode

Both sides run their **shipped** implementation. The training side is
`ModalityEmbedSpec.apply_transform`, which is literally what
`TurnVectorCollator` calls per example after windowing. The rollout side is
`VectorRolloutPolicy.pose_values`, invoked as an unbound method on a stub
carrying only `_raw_poses` — that runs the production code with no model to
load, and avoids writing a "known-good equivalent", which would be the second
implementation the whole design exists to prevent.

Episode `AZPZDX26SYMFY:3EO896NRAYYH3D4GDMU1G621Y2VJTM`, scene
`mp3d/D7G3Y4RVNrH`, goal `cabinet`, 80 observations, from
`data/v2_25hz/formatted_pose` `validation`:

| input | max &#124;Δxy&#124; | max &#124;Δθ&#124; |
| --- | --- | --- |
| float32 raw poses (what the collator actually has) | **0** | **0** |
| float64 raw poses (what the simulator actually reports) | 4.77e-07 m | 1.19e-07 rad |

The first row is the claim: with the input pinned, the two paths are
**bit-identical**, so any difference would be a divergence in the transform.

The second row is a real, bounded difference and is worth stating rather than
tolerancing away. `TurnVectorCollator` casts the arrow column to float32 before
transforming it (`vector_sft.py`: `torch.as_tensor(ex[s.source_column],
dtype=torch.float32)`); the simulator reports float64 and the bridge sends
float64. So the two disagree by the **float32 quantisation of a stored scene
coordinate** — half a micron here. The rollout is *not* rounded to match: that
would be matching an artefact of how the dataset was stored, not the convention.
`test_only_input_precision_differs_on_the_deployed_path` asserts the difference
stays inside 4 float32 ulps at the episode's own coordinate magnitude.

Also confirmed on that episode:

* row 0 is exactly `(0, 0, 0)` — the first observation *is* the frame;
* **55 of 80 rows would be wrong** with an unwrapped heading difference, worst
  case **5.955 rad reported for a true −0.328 rad**. This reproduces the design
  document's figure independently, on the first episode looked at.

The pytest suite adds the cases a single real episode does not exercise: that
the *rotation* into the start frame happens at all (translate-only is the
plausible half-implementation — starting faced along +y and stepping one metre
along it must read as one metre *ahead*, not one metre left), the wrap boundary,
`_PoseFeed`'s inertness and its two error paths, a real socket round trip
through `serve()` with a recording backend (without which nothing would catch a
missing `pose` key in the payload), and that `BridgedPolicy` passes
`Observation.pose` through unaltered.

`tests/test_pose_injection_bridge.py`: **17 passed**. Full suite: **155 passed**.

### Level 2 — the whole two-process path

`--level bridge` drives a real checkpoint over the episode's own frames and
poses and compares `last_stats["pose_value"]` — *the vector the encoder was
handed* — against the collator's rows. That covers the wire, the accumulator,
the per-episode reset and the checkpoint's own spec, i.e. everything except
habitat.

**Status: not yet run** — it needs a pose checkpoint, and none existed at the
time of writing (§6). This is the check that closes the loop, and it should be
run against the first checkpoint that lands, before any metric from it is
quoted.

### Level 3 — simulator frame vs dataset frame

DESIGN.md 8.1(2) calls this "the biggest practical risk". In this codebase it is
**an identity rather than an agreement**, which is stronger than any measurement:

```
replay.py:216            poses.append(robot_sim.get_2d_pose())
generate_pose2d_dataset.py:344   pose2d = [[float(p[0]), float(p[1]), float(p[2])] for p in poses]
build_action_chunk_episode_dataset.py:81   "obs_poses": [rec["obs_pose"] ...]   # = pose2d[idx], verbatim
                                            (action_chunking.py:290, :336)
runner.py:167            pose=ctx.robot_sim.get_2d_pose()
```

The training column and the eval observation are the **same method on the same
class**, with no conversion on either side. There is no origin convention or
axis convention that could differ, because neither side has one to choose. The
planar frame itself is documented in `sim.py` and in `CLAUDE.md`: `X =
habitat_x`, `Y = −habitat_z`, `θ` CCW from `+X`.

What that argument does *not* cover is stated in §6.

---

## 4. Non-regression of the production path

The pose and stop additions touch `act()` on every backend, so "does the
unmodified path still produce the same numbers" was checked empirically, not
only argued.

`run_v3_ctx8_pose/checkpoint-2600` re-run under the new code with the **exact**
command and shard layout that produced the stored `ar_v3_ckpt2600_hm3d_n41_autostop`
result (HM3D n41, 2 shards, `decode=sample`, `seed 1000`, `--auto-stop`,
`--dt 0.04 --gap 5 --max-steps 350`, fresh sim per episode). Same GPU *count* and
episode order, so the per-worker episode counters — and therefore every
per-episode seed — are identical.

**41 of 41 episodes are identical**, field for field, on `success`,
`oracle_success`, `spl`, `oracle_spl`, `soft_spl`, `distance_to_goal_m`,
`min_distance_to_goal_m`, `path_length_m`, `steps`, `ticks`, `sim_seconds`,
`terminated`, `first_reach_step` and `geodesic_start_m`. Not "within noise" —
equal. Summary line for line:

| | stored `ar_v3_ckpt2600` | re-run under the pose code |
| --- | --- | --- |
| oracle_success | 0.4878 | 0.4878 |
| oracle_spl | 0.3030 | 0.3030 |
| path_length_m | 20.61 | 20.61 |
| steps | 239.5 | 239.5 |

That is the strongest form of this check available: these rollouts are
stochastic (`decode=sample`), so identity requires that not one extra draw was
taken from any RNG. It rules out the whole class of "the refactor perturbed
something upstream of sampling" — which is what `_StopHeadControl.seed()`
would have done had it not returned early on a checkpoint with no stop head.

---

## 5. The stop head, and why it is one arm and not two

Every evaluation in this project so far ran with `--auto-stop`, the harness's
oracle termination. Under it `success == oracle_success` and `spl == oracle_spl`
on **every** episode of **every** stored run — verified, not assumed. So
stopping has never actually been measured here.

The pose run's checkpoint has a stop classifier, so it is run **once**, with
`--auto-stop` off and `--stop-head` on. That is not a break in comparability,
because the oracle measures latch independently of how the episode ends:
`OracleSuccess` on *ever* coming within `--success-distance`, `OracleSPL` by
freezing path length at that moment. Turning the oracle stop off cannot change
either. What it changes is `success` and `spl`, because path length keeps
accruing and the episode now ends wherever the robot actually stopped.

**Read the two families separately.**

| family | comparable to the `ar_v2` series? |
| --- | --- |
| `oracle_success`, `oracle_spl` | **Yes** — the continuation of the historical series. |
| `success`, `spl` | **No.** New. Every prior run had them pinned equal to their oracle counterparts, so there is no past number to compare against. |

Running both terminations at once is rejected by the config rather than resolved
by a precedence rule: with both, an episode ends at whichever fires first and
one `success` column would be shared between the oracle and the policy.

Practical consequence: **episodes get longer.** Previously an episode ended the
moment it arrived; now it runs until the head fires or it hits `--max-steps 350`.
Budget more wall clock per checkpoint.

Calibration is expected to need work, and is treated as a fitting problem rather
than a failure: `stop_head.py` judges the head on AP and AUC precisely because
they are threshold-free, and both temperature and threshold are single scalars
fitted post hoc. Every episode record therefore carries the per-step
`(probability, logit)` trace under `"stop"` **whether or not the head is
acting**, so a threshold sweep is an offline pass over the results file rather
than a re-run of the split.

---

## 6. Results

### Settings, held fixed across every run here

Matching `dump/eval_system/ar_v2` exactly, so the oracle numbers are comparable:

```
--dt 0.04 --gap 5 --max-steps 350 --success-distance 1.0
--fresh-sim-per-episode --navmesh robot --evaluate-every 1
--policy-decode sample --policy-seed 1000  (seed 0)
episode set: dump/eval_system/episode_sets/hm3d_val_n41.txt   (41 episodes)
```

`--dt 0.04`, not the harness default of 0.05: these checkpoints are trained on
the 25 Hz corpus, and executing them at 0.05 runs every commanded tick 25% slow
with nothing erroring.

### Commands

```bash
cd /Projects/spatial_training/dump/eval_system

# the control: the exact weight state the pose run warm-starts from
EPISODE_SET_OVERRIDE=$PWD/episode_sets/hm3d_val_n41.txt \
bash run_ar_v2_eval.sh \
    /Projects/spatial_training/dump/autoregressive_head/run_v3_ctx8_pose/checkpoint-3400 \
    ar_v3_ckpt3400_hm3d_n41_autostop 1,5

# the production-path regression check (§4)
EPISODE_SET_OVERRIDE=$PWD/episode_sets/hm3d_val_n41.txt \
bash run_ar_v2_eval.sh \
    /Projects/spatial_training/dump/autoregressive_head/run_v3_ctx8_pose/checkpoint-2600 \
    regress_v3_2600_posecode 0,7
```

### The control: `run_v3_ctx8_pose/checkpoint-3400`

This is the exact weight state `run_v5_ar_pose` warm-starts from, so it is
matched to the pose run by construction rather than by argument. It had not been
evaluated under these settings before; `checkpoint-2600` had, and is shown
beside it because it is the nearest point in the same series.

| | ckpt-2600 | **ckpt-3400 (the control)** |
| --- | --- | --- |
| oracle_success | 0.488 | **0.390** |
| oracle_spl | 0.303 | **0.191** |
| soft_spl | 0.266 | 0.175 |
| path_length_m | 20.61 | 23.60 |
| steps | 239.5 | 273.8 |
| min distance to goal (m) | 2.66 | 3.39 |

`success == oracle_success` and `spl == oracle_spl` in both, because both ran
with `--auto-stop`. That identity is what §5 is about.

Paired over the 41 shared episodes, ckpt-3400 minus ckpt-2600:

| metric | delta | 95% CI | p | MDE | resolved? |
| --- | --- | --- | --- | --- | --- |
| oracle_success | −0.098 | [−0.268, +0.073] | 0.424 | 0.256 | no |
| oracle_spl | −0.112 | [−0.230, +0.005] | — | 0.171 | no (just) |
| progress_m | −0.728 | [−1.377, −0.127] | — | 0.906 | **yes** |
| min_m | +0.728 | [+0.127, +1.377] | — | 0.906 | **yes** |
| flip_rate | −0.009 | [−0.012, −0.006] | — | 0.005 | **yes** |
| path_m | +2.99 | [−1.09, +7.15] | — | 5.98 | no |

**Read this as "the control is 0.390, not as "training got worse".** At n=41 the
outcome metrics cannot resolve a step-2600-to-3400 difference — the MDE is 0.256
and the observed gap is 0.098, so this pair is consistent with no change at all.
What *is* resolved is behavioural and points the same way: the policy gets 0.73 m
less far from the start, ends 0.73 m further from the goal at its closest
approach, and takes 34 more steps. The one improvement is `flip_rate`, down from
0.037 to 0.028 against an MDE of 0.005 — mid-chunk rotation reversals, the
"drunken" motion signature, are genuinely rarer.

This is the same pattern already recorded for this architecture: eval loss falls
while navigation does not follow it. Use ckpt-3400's numbers as the matched
baseline for the pose arm and do not read a trend into the 2600→3400 pair.

### The pose arm: `run_v5_ar_pose`

**Pending.** The run started 2026-08-03 ~07:08 UTC at ~19 s/step with
`--save-steps 200`, so checkpoints land roughly every 65 minutes. Nothing to
report yet, and nothing should be reported before the level-2 check in §3 has
run against the first checkpoint.

---

## 7. What could not be verified

Stated plainly, because each of these is a place a later reader could assume
more was established than was.

1. **Level 2 (end-to-end injected-value parity) has not been run.** It needs a
   pose checkpoint and none existed yet. Levels 1 and 3 together cover the
   transform and the frame; level 2 is what would additionally prove the value
   survives the socket *for a real checkpoint's own spec*. The socket itself is
   covered by `test_pose_survives_the_socket`, with a stub backend.
2. **The folder's `launch.sh` described a run that never happened.** It declared
   `train_vector_sft.py` → `dump/pose_injection/run_v4_pose`, a **regression**
   head. What actually launched was `ar_launch.sh` →
   `dump/pose_injection/run_v5_ar_pose`, an **AR** head warm-started from
   `run_v3_ctx8_pose/checkpoint-3400` — and `ar_launch.sh` did not exist when
   this directory was first read. Confirmed from `ps` and from which `run_*`
   directory is being written, not from the script. Backend: `ar_rollout`.
   This is why pose and stop support went into *all three* backends rather than
   only the one the script implied.
3. **Level 3 is a provenance argument, not a measurement.** It is airtight for
   the *frame*: both sides call the same `get_2d_pose()` with no conversion. It
   does **not** establish that a rollout's poses have the same *distribution* as
   the demonstrations' — a policy driven off-distribution visits poses the
   training data never contained, and the encoder's Fourier features are
   periodic, so a pose far outside the training range is not extrapolated so much
   as aliased. That is a property of the experiment, not a bug, but it is not
   something this verification rules out.
4. **Stop-head calibration is untuned.** The runs below use the checkpoint's own
   `inference`/`threshold`/`temperature`. A miscalibrated head that never fires
   (or fires immediately) would depress `success`/`spl` without touching the
   oracle numbers, and the fix is a sweep over the recorded traces, not a
   conclusion about the head.
