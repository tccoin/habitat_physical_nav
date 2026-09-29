"""The model-environment half of the two-environment bridge.

Run by :class:`objectnav_eval.bridge.PolicyBridge` under a *different*
interpreter -- the one that can load the checkpoint. It must therefore import
nothing from ``habitat_sim``/``habitat``/``continuous_demos``: only
``objectnav_eval.bridge`` (numpy + stdlib) and whatever the chosen backend
needs.

Normally launched for you. To debug the model side alone:

    <model-python> scripts/policy_server.py --socket /tmp/x/sock --ckpt <dir>

``--backend`` selects the policy head implementation; see
``objectnav_eval.bridge.register_backend``.
"""

import argparse
import os
import sys
from pathlib import Path

# Set before torch is imported (the backend imports it lazily, inside its
# __init__), because the allocator reads this once at initialisation.
#
# A rollout is a pathological allocation pattern for the default caching
# allocator: the KV cache grows monotonically across a 350-step episode and is
# then released at reset, so each episode leaves the reserved pool more
# fragmented than the last. On a box where the GPUs are shared with training
# this fragments into an OOM after a handful of episodes -- observed at ~7 GB
# nominal headroom, with 4.19 GiB sitting *reserved but unallocated* in the
# failing process. Expandable segments let the allocator give that back.
#
# It matters more here than it would elsewhere because a policy-server OOM is
# not a lost episode: the error crosses the socket, `PolicyBridge._rpc` raises,
# and the exception takes down the whole worker shard along with every episode
# it had not yet run. `train_ar_action_sft.py` sets the same variable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Each evaluator worker owns a policy server.  Unbounded CPU math pools make
# several otherwise independent GPU workers oversubscribe the host.
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

# The model environment has no reason to have this package installed, and
# installing it there would drag habitat-lab expectations into an env that
# cannot satisfy them. Adding src/ to the path keeps the boundary one-way.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from objectnav_eval.bridge import available_backends, build_backend, serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="unix socket to connect back to")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--backend", default="vector_rollout",
                        help=f"policy head implementation; one of {available_backends()}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-lora", action="store_true",
                        help="fold LoRA into the base model before inference")
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--placeholder", default=None)
    parser.add_argument("--decode", default=None,
                        help="decode rule for a distributional head (bin_rollout, "
                             "ar_rollout); unset means the backend's own default")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Euler steps for the flow-matching head's ODE (flow_rollout); "
                            "unset means the checkpoint's own num_inference_steps")
    parser.add_argument("--ode-rng-preconsume-size", type=int, default=0,
                        help="flow_rollout compatibility: consume a raw FlowSDE ODE "
                             "permutation before each base-noise draw")
    parser.add_argument("--context-reset-every", type=int, default=None,
                        help="memory-length ablation: drop rollout history (KV cache and "
                             "visual dedup DB) every N policy steps, keeping the system "
                             "prompt. Unset/0 = unlimited memory, the production default. "
                             "In STEPS because that is what the backend can count; the "
                             "harness converts from --policy-memory-seconds using dt*gap")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed for a backend's per-episode seeded sampling "
                        "(bin_rollout / ar_rollout with --decode sample)")
    parser.add_argument("--seed-by-episode-uid", action="store_true",
                        help="flow_rollout only: derive ODE noise from the stable "
                             "episode UID instead of worker-local ordering")
    parser.add_argument("--pose-mirror", action="store_true",
                        help="pose-frame ablation: mirror the injected pose about the "
                             "start heading")
    parser.add_argument("--pose-rotate-deg", type=float, default=0.0,
                        help="pose-frame ablation: rotate the injected pose frame, "
                             "applied after --pose-mirror")
    parser.add_argument("--pose-foreign", default=None,
                        help="pose-information ablation: inject another episode's pose "
                             "sequence, sampled from this load_from_disk() dataset")
    parser.add_argument("--pose-foreign-split", default="train")
    parser.add_argument("--no-pose-injection", action="store_true",
                        help="make the pose feed INERT on a checkpoint that declares a "
                             "pose spec: no <pose> marker in the rendered turn and no "
                             "values supplied. This is the pose-free arm of a mixture "
                             "trained with and without pose -- not a corruption ablation, "
                             "which feeds wrong values rather than none")
    parser.add_argument("--anchor-offset", default=None,
                        help="DIAGNOSTIC 'DX,DY[,DTHETA_RAD]'. Moves the origin that "
                             "relative_se2 anchors on, by seeding the pose accumulator "
                             "with a fictitious row that is never emitted -- so the text, "
                             "the images, the marker count and the number of injected "
                             "value rows are all unchanged, and only the numeric pose "
                             "values move. Used to ask whether the same task gets harder "
                             "when the anchor is far away or pointing elsewhere")
    parser.add_argument("--goal-placement", default=None,
                        choices=["segment", "observation"],
                        help="PointNav: where the goal's <pose> marker sits, which must "
                             "be the placement the CORPUS was formatted with "
                             "(format_action_chunk_dataset.py --goal-placement). It is a "
                             "property of the data, not of the weights, and is not "
                             "recorded in the checkpoint -- so it has to be passed, and "
                             "passing it wrong writes the markers somewhere the model "
                             "never saw them with the counts still adding up. Unset = "
                             "ObjectNav, which is what every run before this did")
    parser.add_argument("--stop-head", action="store_true",
                        help="let the checkpoint's stop classifier terminate episodes. "
                             "Off = the probability is still computed and reported in "
                             "last_stats, but wants_stop() is always False")
    parser.add_argument("--stop-inference", default=None,
                        choices=["sample", "threshold", "argmax"])
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--stop-temperature", type=float, default=None)
    parser.add_argument("--stop-head-min-steps", type=int, default=0)
    parser.add_argument("--trajectory-stop-threshold-m", type=float, default=None)
    parser.add_argument("--trajectory-stop-yaw-threshold-rad", type=float, default=None)
    parser.add_argument("--trajectory-stop-min-steps", type=int, default=0)
    parser.add_argument("--trajectory-stop-prefix-points", type=int, default=10)
    parser.add_argument("--sys-path", default=None,
                        help="colon-separated paths to prepend to sys.path before "
                             "loading the backend -- the checkout the checkpoint's code "
                             "lives in, which this environment need not have installed")
    args = parser.parse_args()

    for entry in reversed((args.sys_path or "").split(":")):
        # Prepended, not appended: a same-named namespace package from another
        # checkout may already be importable, and it would shadow the intended
        # one as a *missing submodule*, which is a confusing way to fail.
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)

    print(f"loading {args.ckpt} on {args.device} (backend {args.backend})...", flush=True)
    kwargs = dict(
        ckpt=args.ckpt,
        device=args.device,
        merge_lora=args.merge_lora,
        max_context_tokens=args.max_context_tokens,
        placeholder=args.placeholder,
    )
    # Only forwarded when set: backends that don't accept `decode`/`seed`
    # (vector_rollout) must see no change at all when these flags are absent.
    if args.decode is not None:
        kwargs["decode"] = args.decode
    if args.num_inference_steps is not None:
        kwargs["num_inference_steps"] = args.num_inference_steps
    if args.ode_rng_preconsume_size:
        kwargs["ode_rng_preconsume_size"] = args.ode_rng_preconsume_size
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.seed_by_episode_uid:
        kwargs["seed_by_episode_uid"] = True
    if args.context_reset_every is not None:
        kwargs["context_reset_every"] = args.context_reset_every
    # Same rule as above: only forwarded when the run asked for it, so a backend
    # that predates the stop head sees exactly the kwargs it always saw.
    if args.stop_head:
        kwargs["stop_head"] = True
    for name in ("stop_inference", "stop_threshold", "stop_temperature"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.stop_head_min_steps:
        if not args.stop_head:
            raise ValueError("--stop-head-min-steps requires --stop-head")
        if args.stop_head_min_steps < 0:
            raise ValueError("--stop-head-min-steps must be non-negative")
        kwargs["stop_head_min_steps"] = args.stop_head_min_steps
    trajectory_stop_m = args.trajectory_stop_threshold_m
    trajectory_stop_yaw = args.trajectory_stop_yaw_threshold_rad
    if trajectory_stop_m is not None or trajectory_stop_yaw is not None:
        if trajectory_stop_m is None or trajectory_stop_yaw is None:
            raise ValueError(
                "trajectory STOP requires both --trajectory-stop-threshold-m and "
                "--trajectory-stop-yaw-threshold-rad"
            )
        kwargs["trajectory_stop_threshold_m"] = trajectory_stop_m
        kwargs["trajectory_stop_yaw_threshold_rad"] = trajectory_stop_yaw
        kwargs["trajectory_stop_min_steps"] = args.trajectory_stop_min_steps
        kwargs["trajectory_stop_prefix_points"] = args.trajectory_stop_prefix_points
    if args.pose_mirror:
        kwargs["pose_mirror"] = True
    if args.pose_rotate_deg:
        kwargs["pose_rotate_deg"] = args.pose_rotate_deg
    if args.pose_foreign:
        kwargs["pose_foreign"] = args.pose_foreign
        kwargs["pose_foreign_split"] = args.pose_foreign_split
    if args.goal_placement:
        kwargs["goal_placement"] = args.goal_placement
    if args.no_pose_injection:
        kwargs["disable_pose"] = True
    if args.anchor_offset:
        kwargs["anchor_offset"] = [float(v) for v in args.anchor_offset.split(",")]
    backend = build_backend(args.backend, **kwargs)
    print(backend.describe(), flush=True)
    serve(args.socket, backend)


if __name__ == "__main__":
    main()
