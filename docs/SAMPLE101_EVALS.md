# Running the standard policy evaluation (`sample101`)

Every closed-loop policy number quoted in this project since 2026-08-06 comes from one
episode set, one navmesh and one command shape. This document is how to reproduce them
and how to read them. If you are about to evaluate a checkpoint, read the two
"silently wrong" sections first -- both failure modes produce plausible numbers rather
than errors.

## The episode set

**`sample101`** -- 101 episode ids from HM3D ObjectNav `val`, drawn once and reused so
that every run is paired against every other:

    /Projects/spatial_training/dump/eval_system/episode_sets/sample101_ids.txt

Pass it verbatim; do not re-sample. Runs that draw their own episodes are not
comparable to anything in the results table, and a differently-packed set silently
changes which seed each episode gets (this happened once and invalidated ~19 of 36
"paired" episodes).

    IDS=$(cat /Projects/spatial_training/dump/eval_system/episode_sets/sample101_ids.txt)

On `--navmesh dataset` all 101 are scored. On `--navmesh robot` only ~70 survive
screening, which is one of several reasons those two are not comparable.

## The command

```bash
cd /Projects/habitat_physical_nav

/workspace/conda/envs/vln/bin/python scripts/eval_objectnav_policy.py \
  --episodes   /Projects/spatial_training/data/datasets/objectnav/hm3d/v1/val \
  --scene-root /Projects/spatial_training/data/scene_datasets \
  --output-dir /Projects/spatial_training/dump/eval_system/pose_v5/<RUN_NAME> \
  --episode-ids "$IDS" \
  --ckpt <checkpoint dir> \
  --policy-python   /workspace/conda/envs/longnav_vlm/bin/python \
  --policy-sys-path /Projects/spatial_training/src \
  --policy-backend <BACKEND> --policy-seed 1000 \
  --max-steps <CAP> --dt 0.04 --gap <GAP> \
  --success-distance 1.0 --distance-to VIEW_POINTS \
  --seed 0 --navmesh dataset --fresh-sim-per-episode \
  --auto-stop --auto-stop-delay <DELAY> \
  --record-video --pid-preset baseline --keep-tracking-series
```

The simulator (`vln`, habitat_sim) and the model (`longnav_vlm`, transformers +
flash-attn) **cannot share a Python environment** -- their dependency sets conflict.
The harness runs the policy in a second interpreter over a unix socket, which is what
`--policy-python` and `--policy-sys-path` are for.

`--fresh-sim-per-episode` is not optional for comparable numbers: without it
`HabitatRobotSim.reset()` leaks physics state between episodes.

Working launch scripts for every run below live in
`/Projects/spatial_training/dump/eval_system/run_*.sh` -- copy the nearest one rather
than assembling flags by hand.

## Silently wrong #1: `--gap` must match the corpus

**`--dt` is 0.04 for every model in this project.** The control tick rate has never
changed. What changes is `--gap`, the number of ticks executed per observation, and it
**must equal the `obs_stride_frames` of the corpus the checkpoint trained on**:

| trained on | obs rate | chunk len | `--gap` |
|---|---|---|---|
| `v2_25hz` | 5 Hz | 10 | **5** |
| `v2_25hz_obs2.5hz` | 2.5 Hz | 20 | **10** |
| `v2_25hz_obs1hz` | 1 Hz | 30 | **25** |

A wrong `--gap` executes a fraction of each chunk and discards the rest. It does not
error. It presents as a slow, weak policy. (The same trap in `--dt` once cost a run
that was silently executing at 0.8x speed.)

Confirm from the checkpoint itself rather than the run name:

```bash
python -c "import json; c=json.load(open('<ckpt>/turn_vector_head_config.json'));
print(c.get('fm_n_ticks') or c.get('target_shape'))"
```

## Silently wrong #2: budget is in POLICY steps, not seconds

`--max-steps` and `--auto-stop-delay` count **policy steps**, and one policy step is
`gap * dt` seconds. So the same `--max-steps` buys different amounts of sim time at
different rates. **Compare models at matched sim time, not matched steps.**

| rate | `--gap` | 70 s | 140 s | 350 s |
|---|---|---|---|---|
| 5 Hz | 5 | `--max-steps 350` | 700 | 1750 |
| 2.5 Hz | 10 | `--max-steps 175` | 350 | 875 |
| 1 Hz | 25 | `--max-steps 70` | 140 | 350 |

`--auto-stop-delay` scales the same way; **6.0 s** is the convention (30 / 15 / 6 steps
at gap 5 / 10 / 25). A delay mismatch changes `success` without changing the oracle
metrics, which is one more reason to read the oracle ones.

## Backends

| head | flags |
|---|---|
| flow matching (v8/v9/v10) | `--policy-backend flow_rollout` (no `--policy-decode`; one generation mode by design) |
| autoregressive (v11) | `--policy-backend ar_rollout --policy-decode sample` |
| regression / vector | `--policy-backend vector_rollout` |
| bin head | `--policy-backend bin_rollout --policy-decode sample` |
| **CNF** | **no backend exists.** `VectorRolloutPolicy.from_checkpoint` fails (it routes through `TurnVectorRegressor.from_pretrained`, which strictly loads the flow's ~170 tensors into a `TargetNormalizer`). `VectorRolloutPolicy(TurnFlowPolicy.from_pretrained(dir, proc), proc, cfg)` does load and decode -- so this is a construction change, not a new rollout implementation. |
| blind controls | `--policy forward` or `--policy zero` |

`--policy-decode` matters more than the checkpoint for sampling heads: on one bin-head
checkpoint, `sample` gave 0.178 success against `argmax`'s 0.067. Always state which.

## Which metrics to read

**Read the oracle metrics.** None of the current checkpoints has a stop classifier, and
evaluation runs with `--auto-stop`, so termination is driven by the harness. `success`
measures whether the agent was still within 1 m when forced termination fired -- i.e. it
measures `--auto-stop-delay`. `spl` inherits that.

| metric | use it? | why |
|---|---|---|
| `oracle_success` | **yes** | ever reached within `--success-distance`. Stop-independent. |
| `npwpl` | **yes** | progress toward goal, normalised by start distance, weighted by path length. Reads `min_m` and `path_at_min_m` only -- stop-independent. |
| `min_m` | yes | closest approach |
| `oracle_spl` | **no, it is broken** | see below |
| `success`, `spl`, `soft_spl` | no | harness-driven termination |

### `oracle_spl` is currently wrong

`OracleSPL.update_metric` freezes path length incorrectly and returns **1.000** on
episodes where the agent travelled further than the geodesic before first reaching the
goal -- 12 of 13 such episodes in one run. Recompute it:

```python
ospl = start_m / max(start_m, path_at_min_m) if oracle_success else 0.0
```

Both columns appear as `oSPL_rep` (as reported) and `oSPL_fix` (recomputed) in the
results below. The defect inverts conclusions: on the budget comparison the reported
column makes a longer episode look like an *efficiency gain*, where the corrected one
shows it flat or falling.

### `--navmesh dataset`, not `robot`

`dataset` is the default since 2026-08-06. On `robot` the agent is a physics body that is
**not constrained to the mesh**, so `snap_point` can resolve its proxy across an island
boundary and the metric silently starts measuring a *different goal instance*: on
`wcojb4TFT35:0#26` a 4.4 cm step moved the proxy 2.73 m and `distance_to_goal` fell
6.193 -> 1.189 in one tick, with 168 of 320 ticks breaking the 1-Lipschitz bound. Zero
violations for the same trajectory on `dataset`.

Screening also differs (101 scored vs ~70), so **runs on different meshes are not
comparable at all**. Everything before 2026-08-06 used `robot`; see
`docs/OBJECTNAV_EVAL.md`.

## Results

All: `sample101`, `--navmesh dataset`, n=101, flow-matching head with planar-SE(2) pose
injection. `turns_k` = thousands of observation-turns of supervision consumed
(`steps x obs_per_episode`), which is the honest training measure since slower corpora
yield fewer turns per step.

### Decision frequency, at matched 70 s

| run | Hz | ckpt | turns_k | oracle_succ | oSPL_rep | oSPL_fix | NPWPL | min_m | path_m |
|---|---|---|---|---|---|---|---|---|---|
| v8 | 5.0 | 5800 | 594 | 0.366 | 0.164 | 0.139 | 0.266 | 3.49 | 23.6 |
| v9 | 2.5 | 2000 | 101 | 0.426 | 0.230 | 0.191 | 0.296 | 3.31 | 27.6 |
| **v9** | **2.5** | **9600** | 485 | 0.535 | 0.334 | **0.265** | **0.358** | 2.94 | 23.4 |
| v10 | 1.0 | 5400 | 109 | 0.495 | 0.271 | 0.217 | 0.308 | 3.13 | 23.1 |
| v10 | 1.0 | 8400 | 169 | 0.525 | 0.306 | 0.248 | 0.333 | 2.64 | 24.5 |
| **v10** | **1.0** | **20000** | 402 | **0.564** | 0.306 | 0.246 | 0.341 | **2.52** | 22.3 |

5 Hz is last on every oracle metric **while having consumed the most supervision of any
run**. 2.5 Hz and 1 Hz are statistically indistinguishable (all paired CIs span zero).

Behavioural, same runs: `flip_rate` falls monotonically with rate -- 0.022 / 0.017 /
0.014 against the training corpus's 0.0076. All policies under-drive (0.34-0.45 vs the
corpus's 0.517).

### Episode budget, same checkpoint

| Hz | ckpt | sim s | oracle_succ | oSPL_fix | NPWPL | min_m | path_m |
|---|---|---|---|---|---|---|---|
| 2.5 | 9600 | 70 | 0.535 | 0.265 | **0.358** | 2.94 | 23.4 |
| 2.5 | 9600 | 140 | 0.653 | 0.265 | 0.317 | 2.18 | 37.1 |
| 1.0 | 5400 | 70 | 0.495 | 0.217 | **0.308** | 3.13 | 23.1 |
| 1.0 | 5400 | 350 | **0.743** | 0.205 | 0.238 | 1.76 | **64.7** |

More time buys arrivals and destroys efficiency, at both rates (all paired deltas
significant). The extreme case is 1 Hz at 350 s: the highest oracle success anywhere at
the lowest NPWPL anywhere -- 64.7 m of path to reach goals a straight line would cover
in ~7. That is searching, not navigating.

### Training direction

5 Hz models get **worse** with training (v7: spl 0.141 -> 0.115 -> 0.067 across
ck5800/9800/11200, `frozen_frac` tripling to 0.514); 2.5 and 1 Hz improve monotonically
(v9 spl 0.104 -> 0.196; v10 0.149 -> 0.171 -> 0.185). A training confound cannot flip the
sign of a derivative, which is the strongest evidence that the rate effect is real.

## Where the raw data lives

    /Projects/spatial_training/dump/eval_system/pose_v5/<RUN>/
        objectnav_eval_results.json    aggregate metrics
        objectnav_eval_results.jsonl   one line per episode (the summary record)
        shards/shard_N.json            the exact config the workers ran
        shards/shard_N_results.json    FULL per-episode records: trajectory, tracking,
                                       motion, contact, coherence, pose_injected
        shards/shard_N.log             per-episode lines as they complete
        videos/<scene>_<ep>.mp4        real-time playback
        logs/policy_shard<N>.log       the policy server, incl. which head loaded

The run directories for the table above:

    pose_v8_flow_ckpt5800_sample101_dsnavmesh
    pose_v9_2p5hz_ckpt2000_sample101_dsnavmesh
    pose_v9_2p5hz_ckpt9600_sample101_dsnavmesh
    pose_v9_2p5hz_ckpt9600_sample101_dsnavmesh_350st
    pose_v10_1hz_ckpt5400_sample101_dsnavmesh
    pose_v10_1hz_ckpt5400_sample101_dsnavmesh_STEPPARITY
    pose_v10_1hz_ckpt8400_sample101_dsnavmesh
    pose_v10_1hz_ckpt20000_sample101_dsnavmesh

**Field names differ between the two levels.** The summary `.jsonl` uses `start_m`,
`min_m`, `path_at_min_m`; the shard `_results.json` uses `geodesic_start_m`,
`min_distance_to_goal_m`, `path_at_min_m`. Reading one set against the other silently
yields zeros.

Per-tick trajectories are in the shard records (`trajectory.tick_pose`,
`tick_camera`, `tick_height`, `chunks`) whenever `--log-trajectory` was on (default), so
a finished run can usually answer a new question without being re-run.

## Verifying before you trust a run

```bash
# 1. what did the workers actually run?
python -c "import json; c=json.load(open('<RUN>/shards/shard_0.json')); c=c.get('config',c)
print({k:c.get(k) for k in ['ckpt','policy_backend','policy_decode','gap','dt',
      'max_steps','auto_stop_delay','navmesh','success_distance','seed','policy_seed']})"

# 2. per-scene banner: navmesh and PID actually applied
grep -h "navmesh \|pid " <RUN>/shards/shard_0.log | head -2

# 3. the impossibility check -- start - min can never exceed distance travelled
python - <<'EOF'
import json
bad=[r['uid'] for r in (json.loads(l) for l in open('<RUN>/objectnav_eval_results.jsonl'))
     if isinstance(r.get('npwpl'),(int,float))
     and r['start_m']-r['min_m'] > r['path_at_min_m']+0.25]
print(f'{len(bad)} impossible episodes', bad[:5])
EOF
```

That third check catches the navmesh/goal-instance defect. Expect **0-1** on
`--navmesh dataset` and ~5 of 70 on `robot`.

## Pairing runs for comparison

Always compare on the intersection of scored uids, never on aggregate means from
different runs:

```python
import json, statistics as st, math
def load(d):
    return {r['uid']: r for r in (json.loads(l) for l in
            open(f'{d}/objectnav_eval_results.jsonl'))
            if isinstance(r.get('npwpl'), (int, float))}
A, B = load(RUN_A), load(RUN_B)
common = sorted(set(A) & set(B))
d = [B[u]['npwpl'] - A[u]['npwpl'] for u in common]
mu = st.mean(d); h = 1.96 * st.stdev(d) / math.sqrt(len(d))
print(f'{mu:+.3f} [{mu-h:+.3f}, {mu+h:+.3f}]  n={len(common)}')
```

At n=101 the MDE on a success-rate difference is roughly 0.10. **n=45 cannot resolve
success-rate differences at all** (MDE ~0.23), and `val_mini` (n=14) is unusable for
ranking -- it once put a checkpoint at 0.357 where n=280 gave 0.139.

Older runs on `--navmesh robot` live in the same directory with `_oracledelay`,
`_MIRROR90`, `_FOREIGN` or `_PID_*` suffixes. They are internally consistent and **not
comparable** to anything above.

## Related

* `docs/OBJECTNAV_EVAL.md` -- the harness architecture and every extension point
* `docs/EVAL_QUICKSTART.md` -- the one-command version
* `docs/GEOMETRIC_BUDGET.md` -- the alternative episode budget and its conversions
* `scripts/compare_policies.py` -- paired comparison with McNemar
* `scripts/rescore_budget.py` -- re-score finished runs under a geometric cap
