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
| `PolicyBackend.act(rgb, pose=None)` | The signature gains the pose. Every backend (`vector_rollout`, `bin_rollout`, `ar_rollout`, `flow_rollout`) takes it and ignores it unless its checkpoint asked for it — which is the whole reason a new head is a backend and pose is not. `flow_rollout` needed no pose code of its own: `--policy-backend flow_rollout` through `verify_pose_injection.py --level bridge` matched the collator to 1.1e-7 m on a real episode against `run_v8_flow_planar_pose/checkpoint-400`. |
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
--fresh-sim-per-episode --navmesh robot --evaluate-every 1   # NOTE: --navmesh robot
# Every number in this document was produced on the ROBOT navmesh, which is no
# longer the default -- see docs/OBJECTNAV_EVAL.md 'Which one to measure on'.
# It overstates geodesics and, worse, lets snap_point switch which goal instance
# is being measured mid-episode. These numbers are internally consistent with
# each other and NOT comparable with anything run on --navmesh dataset.
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

### The pose arm: `run_v5_ar_pose/checkpoint-4400`

The run is `dump/pose_injection/run_v5_ar_pose` -- an **AR** head warm-started
from `run_v3_ctx8_pose/checkpoint-3400`, launched by `ar_launch.sh` (see §7.2).
Backend `ar_rollout`.

**Termination had to be reworked before any number here was readable.** With the
classifier acting ungated it ended every episode by step 46 and dragged
`oracle_success` to 0.073 -- not because navigation failed but because the
episode was over before it could start. §5 covers the fix; the arms below all use
`--auto-stop --auto-stop-delay 30 --stop-head --stop-requires-oracle
--stop-inference threshold --stop-threshold 0.5`, which lets the oracle open a
30-step window on arrival and lets the classifier declare only inside it.

| arm | n | oracle_success | oracle_spl |
| --- | --- | --- | --- |
| n41, true frame | 36 | 0.472 | 0.262 |
| **sample101, true frame** | **70** | **0.486** | **0.271** |
| v3 ckpt-3400 control (no injection) | 41 | 0.390 | 0.191 |

`sample101` is the number to quote: 70 fresh episodes over 20 scenes, no splice
and no shifted seeds. (An earlier n41 figure of 0.537 was a splice of 36 fresh
episodes plus 5 carried over *because they had succeeded* -- an upward bias.
Do not use it.)

The +0.10 over the control is **not** resolvable at this n (MDE ~0.20), and in any
case a pose-injecting policy has noiseless GPS+compass and is not comparable to
one without it.

### The ablations: does the policy use the injected pose?

Three arms, same checkpoint, same gated config, paired uid for uid:

| ablation | what it destroys | paired n | delta oracle_success | 95% CI | p | delta oracle_spl |
| --- | --- | --- | --- | --- | --- | --- |
| mirror + 90 deg | the frame's orientation only (an isometry) | 36 | **+0.000** | [-0.167, +0.139] | 1.000 | -0.004 |
| foreign poses | everything -- an unrelated trajectory | 70 | **-0.057** | [-0.200, +0.086] | 0.557 | **+0.001** |
| foreign poses | " | 36 | **+0.056** | [-0.139, +0.250] | 0.774 | +0.014 |

**Null across the board.** Every interval spans zero, no p below 0.55, and the two
foreign arms disagree in sign -- the signature of noise, not a small real effect.
Behaviour is flat too: on the n=70 arm `turn_frac` +0.013 [-0.015, +0.042],
`drive_frac` -0.024 [-0.065, +0.016], `flip_rate` +0.000.

Why two different corruptions, and why the second is the one that counts: mirror
+ rotation is an **isometry**, so distances travelled, revisit structure and
coverage all survive it. A policy using pose only for "have I been here before"
is untouched by it, and a null there means "pose is not bound to visual
direction", not "pose is unused". The foreign-trajectory arm removes that escape
-- the injected walk is a real corpus trajectory, so still on the encoder's
training distribution (unlike zeros or noise, which would confound "no
information" with "input never seen"), but it carries nothing about where the
robot is. It too is null.

**So: at checkpoint-4400 the injected pose is not contributing to navigation.**

What that does *not* establish, and must not be collapsed into it:

1. **Unused vs. unusable.** `FourierSE2Encoder`'s output projection is
   zero-initialised by design, so "the model cannot yet read the pose" predicts
   exactly this result. No behavioural arm can separate the two; the decodability
   probe (DESIGN.md section 9) is the only thing that can, and it does not exist yet.
2. **Power.** MDE is 0.204 on the largest arm. This rules out a large
   contribution, not a modest one.
3. **One checkpoint**, 22% into a 20k-step run.

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

---

## 8. The encoder rebuild, and what it settled (2026-08-05)

Sections 1-7 end with three null ablations on `run_v5_ar_pose` and the caveat that
they could not distinguish "localisation was not the bottleneck" from "the
encoding was unusable". An audit resolved that: **the encoding was unusable**, and
a second one that is not produces a sharply different picture -- though not the
one that was hoped for.

### 8.1 What was wrong with `FourierSE2Encoder`

Measured on the trained `run_v5_ar_pose` encoder over real relative poses:

| defect | measurement |
| --- | --- |
| heading nearly invisible | reversing the agent 180 deg moved the injected vector **3.6x LESS** than sliding it 10 cm (1.81 vs 6.48). Heading was ~1% of the varying energy. |
| geometry inverted | corr(pose distance, embedding distance) = **-0.218**. Poses further apart in metres landed *closer* in embedding space. |
| aliasing | the corpus's modal step is 0.4004 m against a 0.25 m shortest basis period; 39.6% of consecutive observations alias. Effective rank of the varying output: **2.02 of 2048**. |
| norm mismatch | injected norm reached **16.5** against **0.362** for the tokens actually present in a sequence -- 45x, still growing, `grad_norm` median 10.7 vs 1.9 for a comparable run. |

Cause of the first: two bounded heading channels `(cos, sin)` next to an
*unbounded* raw `xy` channel with std 3.0 m, which alone was 55% of the first
layer's input variance. Not a bandwidth problem -- a survey of 17 navigation
repos found **k=1 heading is universal** (habitat DD-PPO, PIRLNav, VLN-DUET,
VLN-HAMT, Recurrent-VLN-BERT all use exactly `(cos, sin)`), and none of them
Fourier-expands position either.

### 8.2 The `max_norm` bug that killed `run_v6_regress_pose`

`run_v6_regress_pose` trained ~4200 steps with its pose encoder **completely
frozen**. `zero_init` makes the output exactly zero; the `max_norm` soft cap
multiplies by `tanh(||out||/m) * m / (||out|| + 1e-12)`, which is **0** at
`||out|| == 0`; backward carries the same factor. Output 0 -> grad 0 -> weights
never move, permanently. Encoder weights are bit-identical at steps 200, 2000 and
3600. Three separate gradient paths all vanish, which is why it is airtight
rather than merely fragile.

The same cap is dead at the *other* end too: at v5's real output norms (12-19),
fp32 `tanh` returns exactly 1.0 above 9.011, so the radial gradient is exactly
zero there as well.

Consequence: **v6 is a free, exactly matched no-pose control** -- same seed, same
data order, same architecture, one line of config different.

### 8.3 `PlanarSE2Encoder`

`[x/s, y/s, cos theta, sin theta]`, four features. Position raw (as the field
does), `pos_scale=4.0` to balance channel variance (x/4 std 0.77 vs heading's
0.71), output normalised to `out_norm` so the norm cannot drift, and a scalar
`out_gain` that starts at zero but whose *gradient does not*. Verified on the
trained ckpt-8200 equivalent:

| | fourier (v5) | planar (v7) |
| --- | --- | --- |
| heading share of varying energy | 1.0% | **62%** |
| corr(pose dist, embedding dist) | -0.218 | **+0.589** |
| output norm | 16.5, growing | **0.400 exactly** |

### 8.4 It works -- on the training objective

`run_v7_planar_pose` vs its matched `run_v6_regress_pose` control, identical
batches:

* v7 lower `eval_loss` on **9 of 10** eval points (only step 200 favours v6 --
  the transient as the pathway comes online)
* the gain is **heading**: `rmse_dtheta` lower at every point from step 400,
  by 8-11%, while `rmse_dx` is roughly tied
* the gap **widens** with training: at step 4200, `mae_dtheta` 0.0975 vs 0.1216
  (20% better) while v6's own curve had stalled
* at step 6000 every component is still at an all-time low (loss 0.0729,
  `mae_dtheta` 0.0890) -- 41.6% improvement over steps 200-2000, then a further
  20.6% over 2000-6000. The rate bends but does not flatten.

### 8.5 And does not convert into navigation

Four checkpoints x pose condition, paired on the same 70 `sample101` episodes:

| ckpt | NPWPL true | foreign | delta | oracle true | foreign |
| --- | --- | --- | --- | --- | --- |
| 600 | 0.338 | 0.351 | +0.013 | 0.543 | 0.543 |
| 1000 | 0.292 | 0.350 | +0.058 | 0.543 | 0.543 |
| 1400 | 0.375 | 0.342 | -0.033 | 0.514 | 0.457 |
| 2000 | 0.325 | 0.378 | +0.053 | 0.371 | 0.586 |
| 5800 | 0.327 | pending | | 0.414 | |

Pooled NPWPL change from replacing the pose with an unrelated episode's:
**+0.023, 95% CI [-0.011, +0.057]**. And ckpt-2000 -> ckpt-5800 on the true arm,
across nearly 3x the training and a 31% fall in eval loss: **NPWPL +0.002,
CI [-0.066, +0.070]**.

**Caution on `oracle_success`.** At ckpt-2000 it read +0.214 for the foreign arm,
which looked like a large effect; NPWPL on the same episodes read +0.053, CI
spanning zero. The binary metric counts an episode ending at 1.01 m identically
to one ending at 8 m, and the arms' episodes cluster either side of that line.
Do not read checkpoint-to-checkpoint deltas off it.

### 8.6 What the pose *does* change: rotation, and getting stuck

Pooled over four checkpoints, replacing the true pose with a foreign one:

| metric | real pose | fake pose | change | 95% CI |
| --- | --- | --- | --- | --- |
| median_turnrate_degps | 25.7 | 38.1 | **+13.8** | [+12.5, +15.2] |
| frac_turn | 0.315 | 0.384 | +0.069 | [+0.055, +0.083] |
| median_speed_mps | 0.295 | 0.230 | -0.065 | [-0.099, -0.031] |
| flip_rate_within_chunk | 0.250 | 0.180 | -0.070 | [-0.088, -0.052] |
| **frac_drive** | 0.496 | 0.495 | -0.001 | [-0.018, +0.016] |
| **undershoot_ratio_moving** | 0.800 | 0.801 | +0.001 | [-0.011, +0.013] |

The last two are the controls that make the rest credible: translation share and
controller tracking are **unchanged**, so this is the policy commanding different
motion, not the executor behaving differently. `commanded_frac_turn` moves on its
own (+0.032), which is what the policy asked for before the controller touched it.

**The mechanism is pose *dynamics*, not frame correctness.** A three-way at
ckpt-2000 separates them, because mirror+90deg is an isometry of the robot's own
trajectory (so a stuck robot still sees a frozen pose) while a foreign trajectory
never stops moving:

| arm | stuck-share | cap@350 | turn deg/s |
| --- | --- | --- | --- |
| TRUE | 0.428 | 45 | 19.0 |
| MIRROR90 | 0.497 | 44 | 15.4 |
| FOREIGN | 0.234 | 32 | 37.2 |

`stationary_share_in_runs_ge_10`, paired: TRUE->MIRROR90 **+0.069, CI
[-0.010, +0.147] (not significant)**; TRUE->FOREIGN **-0.194, CI
[-0.272, -0.116]**; MIRROR90->FOREIGN **-0.263, CI [-0.342, -0.183]**.

So a *completely wrong frame* behaves like the correct pose, and only breaking
the dynamics changes the stuck behaviour. The working hypothesis this supports:
when the robot stalls, the injected pose goes near-stationary, which is a regime
the training corpus never contains -- expert demonstrations are always moving. An
always-moving foreign pose keeps the policy out of that regime.

**Not yet verified directly.** The mechanism is inferred from aggregates. The
runs from ckpt-5800 onward carry the per-tick trajectory log (section 8.7), so
"is the injected pose actually frozen during the long stalls" is now checkable
rather than assumed. Do that before treating it as established.

### 8.7 Instrumentation added

* **`npwpl`** -- normalised progress to the *closest approach*, weighted by path
  efficiency at that same instant. `soft_spl` measures progress to the *final*
  pose, which charges a stop-head-less checkpoint for drifting away after
  arrival; 22 of 38 successes at one checkpoint did exactly that. MDE 0.095 vs
  `oracle_success`'s 0.180 on this episode set. `path_at_min_m` is recorded to
  make the denominator exact; before it existed the metric mixed a frozen
  numerator with a total-path denominator and was wrong.
* **`TrajectoryLog`** -- per control tick: `[sim_time, X, Y, theta]`, the habitat
  **world camera pose** `[x, y, z, qw, qx, qy, qz]` (1.19 m up and 0.10 m behind
  the base, carrying camera pitch that appears nowhere else), `joint_z`, and
  every action chunk **as emitted** before truncation to `gap`. ~230 kB/episode
  against ~700 kB of MP4 the same episode already writes. On by default.
* **drive-conditioned motion** -- `drive_disp_mean_m`, `drive_speed_mean_mps`,
  `drive_distance_m`. The pre-existing `achieved_disp_mean_moving_m` pools DRIVE
  and TURN ticks, so it is diluted by whatever drive/turn mix a policy happens to
  have -- exactly the confound it would be used to rule out.
* **`model_metrics`** (training side) -- per-module `grad_norm`, `weight_delta`,
  `act_norm_mean`, `act_over_embed`, `cap_sat_frac`. Any one of these logged at
  step 1 would have caught the v6 dead encoder immediately.

### 8.8 Open

1. The stuck-pose mechanism is unverified at the trajectory level (8.6).
2. `MIRROR90` has only been run at ckpt-2000. ckpt-600 is the informative one --
   TRUE and FOREIGN were exactly tied there, so a split would be attributable to
   the corruption rather than to checkpoint quality.
3. The `constant`-encoder control (same marker, same rate, no value) has never
   been run. It would separate "reads the pose" from "reads that a marker is
   present", which the ablations here cannot.
4. Whether any of this changes past 30% of training. Every navigation number is
   flat from ckpt-600 to ckpt-5800 while the loss falls monotonically.
