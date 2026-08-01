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
import sys
from pathlib import Path

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
    args = parser.parse_args()

    print(f"loading {args.ckpt} on {args.device} (backend {args.backend})...", flush=True)
    backend = build_backend(
        args.backend,
        ckpt=args.ckpt,
        device=args.device,
        max_context_tokens=args.max_context_tokens,
        placeholder=args.placeholder,
    )
    print(backend.describe(), flush=True)
    serve(args.socket, backend)


if __name__ == "__main__":
    main()
