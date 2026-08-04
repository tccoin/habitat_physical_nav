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
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--placeholder", default=None)
    parser.add_argument("--decode", default=None,
                        help="decode rule for a distributional head (bin_rollout, "
                             "ar_rollout); unset means the backend's own default")
    parser.add_argument("--context-reset-every", type=int, default=None,
                        help="memory-length ablation: drop rollout history (KV cache and "
                             "visual dedup DB) every N policy steps, keeping the system "
                             "prompt. Unset/0 = unlimited memory, the production default. "
                             "In STEPS because that is what the backend can count; the "
                             "harness converts from --policy-memory-seconds using dt*gap")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed for a backend's per-episode seeded sampling "
                             "(bin_rollout / ar_rollout with --decode sample)")
    parser.add_argument("--stop-head", action="store_true",
                        help="let the checkpoint's stop classifier terminate episodes. "
                             "Off = the probability is still computed and reported in "
                             "last_stats, but wants_stop() is always False")
    parser.add_argument("--stop-inference", default=None,
                        choices=["sample", "threshold", "argmax"])
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--stop-temperature", type=float, default=None)
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
        max_context_tokens=args.max_context_tokens,
        placeholder=args.placeholder,
    )
    # Only forwarded when set: backends that don't accept `decode`/`seed`
    # (vector_rollout) must see no change at all when these flags are absent.
    if args.decode is not None:
        kwargs["decode"] = args.decode
    if args.seed is not None:
        kwargs["seed"] = args.seed
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
    backend = build_backend(args.backend, **kwargs)
    print(backend.describe(), flush=True)
    serve(args.socket, backend)


if __name__ == "__main__":
    main()
