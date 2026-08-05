# Evaluating a policy — quickstart

One harness, one command shape, for every policy type. Full architecture in
`docs/OBJECTNAV_EVAL.md`; this is just how to run it.

## The command shape

```bash
cd /Projects/habitat_physical_nav

/workspace/conda/envs/vln/bin/python scripts/eval_objectnav_policy.py \
  --episodes  /Projects/spatial_training/data/datasets/objectnav/hm3d/v1/val \
  --scene-root /Projects/spatial_training/data/scene_datasets \
  --output-dir /Projects/spatial_training/dump/<run_name> \
  --max-steps 200 --fresh-sim-per-episode --gpus 0,1 \
  --policy-python /workspace/conda/envs/longnav_vlm/bin/python \
  --policy-sys-path /Projects/spatial_training/src \
  <POLICY FLAGS>          # <- the only part that changes
```

`--fresh-sim-per-episode` is not optional for numbers you intend to compare:
without it, `HabitatRobotSim.reset()` leaks physics state and episodes contaminate
each other. It moved one checkpoint from 0.214 to 0.357 success.

The simulator runs in `vln` and the policy in `longnav_vlm` (their dependencies
genuinely conflict), bridged over a unix socket — hence the two `--policy-*` flags.

## Policy flags, per type

| policy | flags |
|---|---|
| **regression head** (continuous) | `--ckpt <dir>` |
| **bin head** (discrete, per-slot) | `--ckpt <dir> --policy-backend bin_rollout --policy-decode sample --policy-seed 1000` |
| **AR head** (autoregressive) | `--ckpt <dir> --policy-backend ar_rollout --policy-decode sample --policy-seed 1000` |
| **flow-matching head** | `--ckpt <dir> --policy-backend flow_rollout --policy-seed 1000`. No `--policy-decode`: one generation mode, and no mean by design (averaging ODE samples *is* the creeping failure it removes). `--policy-inference-steps N` sweeps the Euler integration; unset uses the checkpoint's own. |
| **image-blind control** | `--policy forward` or `--policy zero` |

Two things are **not** flags in that table, because they are properties of the
checkpoint rather than of the run — full writeup in `docs/POSE_INJECTION_EVAL.md`:

* **Pose injection.** If the checkpoint declares a `<pose>` modality spec, the
  harness feeds it the agent's raw planar pose every step and the model's own
  `relative_se2` converts it to the episode-start frame. No flag, deliberately: a
  flag could disagree with the weights. Confirm it happened — the server log
  prints `pose=<pose> ...`, and each episode record grows a `pose_injected`
  block whose first row must be exactly `(0, 0, 0)`.
* **A stop head.** Its probability and logit are recorded every step regardless.
  Add `--stop-head` to let it actually end episodes; it is **mutually exclusive
  with `--auto-stop`** (the config raises). Swapping one for the other leaves
  `oracle_success`/`oracle_spl` untouched — they latch on ever reaching
  `--success-distance` — but `success`/`spl` stop being pinned equal to them, so
  do not compare a `--stop-head` `success` against an `--auto-stop` one. Tune the
  threshold offline with `scripts/sweep_stop_threshold.py` rather than re-running.

**Always run the blind controls** on the same episodes. `forward` once beat a
trained checkpoint on SPL at n=14 — without that floor you cannot tell a working
policy from a lucky one.

### Decode mode matters more than the checkpoint

For any sampling head, `--policy-decode` changes results enormously. Same weights,
same episodes, bin head at checkpoint-7250:

| decode | success | path |
|---|---|---|
| `sample` | **0.178** | 11.12 m |
| `mean` | 0.156 | 8.48 m |
| `argmax` | 0.067 | **2.15 m** |

`sample` is the one that works; `mean` reintroduces the creeping the distributional
head exists to fix; `argmax` under-commits to motion and barely moves. Always state
which you used.

## Adding a new head

Implement `PolicyBackend` and call `register_backend("<name>", …)` in
`src/objectnav_eval/bridge.py` — see `VectorRolloutBackend` and `BinRolloutBackend`.
Nothing else changes; all metrics and probes are inherited.

### Which AR architecture did this run use?

The AR head has grown decoder options that all default OFF, so two checkpoints
with the same `n_codes` and `n_ticks` can be different architectures. The policy
server log — `<output-dir>/logs/policy_shard<N>.log`, one per shard — states which
one loaded, on the second `[ar decoder …]` line:

```
  [ar decode=sample seed_base=1000 n_codes=1024 n_ticks=10]
  [ar decoder d_model=128 n_layers=4 n_heads=4 n_context_tokens=8 use_prefix_pose=True attn_mode=causal direct_context=False]
```

`n_context_tokens=1 use_prefix_pose=False attn_mode=causal direct_context=False`
is the original architecture. Record that line with the run — it is the only
place the eval output distinguishes one AR run from another.

## Reading the output

- `objectnav_eval_results.json` — aggregate metrics
- `objectnav_eval_results.jsonl` — one line per episode
- `videos/` — with `--record-video`, real-time playback

Beyond success/SPL, three probes run by default:

| probe | what it catches |
|---|---|
| `motion` | frozen vs turning-in-place vs driving; `blocked_drive_frac` = commanding motion into geometry |
| `contact` | collisions as *runs*, ground contact excluded by normal direction |
| `coherence` | `flip_rate_within_chunk` — rotation reversing mid-chunk. Real data 0.005; an independent-sampling head ~0.10. This is "drunken" motion, quantified. |

**Coherence discriminates where success cannot.** At n=45 it separated two decode
modes with an effect 8.6× the minimum detectable, while success could not separate
them at all. If you are comparing heads at small n, look there first.

## Sample size

n=45 resolves large behavioural differences; **it cannot resolve success-rate
differences** (MDE ≈ 0.23). n=280 resolves ≈ 4.5 points of success. `val_mini`
(n=14) is unusable for ranking — it put a checkpoint at 0.357 where n=280 gave
0.139.
