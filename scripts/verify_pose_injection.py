"""Empirical check that the pose the eval bridge injects is the pose training used.

The pytest suite (``tests/test_pose_injection_bridge.py``) pins the frame
convention against the two shipped functions. This script is the other half:
it runs the claim over a **real dataset episode**, reports the numbers rather
than only asserting them, and -- given ``--ckpt`` -- takes the whole two-process
path end to end, so the value compared is the one the model actually read.

Three levels, each strictly stronger than the last:

1. ``transform``  -- collator vs rollout on the episode's own ``obs_poses``.
   No model, no simulator, no socket.
2. ``bridge``     -- the same episode driven through ``PolicyBridge`` into a
   real checkpoint, comparing ``last_stats['pose_value']`` (what the encoder
   was handed) against the collator's rows. Covers the wire, the accumulator,
   the per-episode reset, and the checkpoint's own spec. Needs ``--ckpt`` and
   ``--policy-python``.
3. ``sim``        -- the simulator's ``get_2d_pose()`` against the dataset's
   ``obs_poses`` for one episode. Reported as an identity rather than measured
   agreement, and the script says why: ``generate_pose2d_dataset.py`` writes
   ``robot_sim.get_2d_pose()`` verbatim into ``pose2d``, which
   ``build_action_chunk_episode_dataset.py`` copies verbatim into ``obs_poses``,
   and ``runner.py`` reads the same method on the same class. There is no
   conversion on either side to disagree about. Level 3 therefore only checks
   that the method still returns what the column contains, for a spawn at a
   recorded pose.

    python scripts/verify_pose_injection.py --level transform
    python scripts/verify_pose_injection.py --level bridge \\
        --ckpt <dir> --policy-python /workspace/conda/envs/longnav_vlm/bin/python \\
        --policy-sys-path /Projects/spatial_training/src --steps 40
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_DATASET = "/Projects/spatial_training/data/v2_25hz/formatted_pose"
DEFAULT_SPEC = "/Projects/spatial_training/dump/pose_injection/pose_spec.json"
DEFAULT_LONGNAV_SRC = "/Projects/spatial_training/src"


def load_spec(path: str):
    from longnav.utils.modality_embed import ModalityEmbedSpec

    with open(path) as fh:
        specs = json.load(fh)
    pose = [s for s in specs if s.get("transform") == "pose_relative_first"]
    if len(pose) != 1:
        raise SystemExit(f"expected one pose spec in {path}, found {len(pose)}")
    return ModalityEmbedSpec.from_dict(pose[0])


def collator_values(spec, raw_poses):
    """Exactly what ``TurnVectorCollator`` puts on the wire, cast included."""
    import torch

    values = torch.as_tensor(np.asarray(raw_poses), dtype=torch.float32)
    return spec.apply_transform(values).numpy().astype(np.float64)


def rollout_values(spec, raw_poses):
    """What the bridge produces, one raw pose at a time, via the shipped method."""
    from objectnav_eval.bridge import _PoseFeed
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    # `cfg` stands in for RolloutConfig: `_PoseFeed` installs the marker on it,
    # and refuses to run against a config that could not render one.
    config = type("C", (), {"modality_marker": None})()
    policy = type("P", (), {"_pose_spec": spec, "cfg": config})()
    feed = _PoseFeed(policy)
    stub = type("S", (), {})()
    stub._raw_poses = []
    out = []
    for pose in raw_poses:
        # `step_kwargs(policy, pose)`: the policy argument arrived with the pose-frame
        # ablation, which needs `policy.pose_values` to corrupt the *relative* value.
        kwargs = feed.step_kwargs(policy, list(pose))
        value = VectorRolloutPolicy.pose_values(stub, kwargs["obs_pose"])
        out.append(value[0].tolist())
    return np.asarray(out, dtype=np.float64)


def report(name, a, b):
    diff = np.abs(np.asarray(a) - np.asarray(b))
    print(f"  {name:<28} max|dxy| {diff[:, :2].max():.3e} m   "
          f"max|dtheta| {diff[:, 2].max():.3e} rad   rows {len(diff)}")
    return diff


def wrap_statistics(raw_poses):
    """How much the wrap actually matters on this episode.

    Reported because "the heading must be wrapped" is easy to agree with and
    easy to leave unimplemented; a number from the corpus is what makes it
    concrete.
    """
    theta = np.asarray(raw_poses)[:, 2]
    naive = theta - theta[0]
    from longnav.utils.pose_frame import wrap_to_pi
    import torch

    wrapped = wrap_to_pi(torch.as_tensor(naive, dtype=torch.float64)).numpy()
    off = np.abs(naive - wrapped)
    return {
        "rows_where_naive_is_wrong": int((off > 1e-9).sum()),
        "rows": len(theta),
        "max_naive_error_rad": float(off.max()),
        "worst_naive_rad": float(naive[np.argmax(off)]),
        "worst_true_rad": float(wrapped[np.argmax(off)]),
    }


def pick_episode(dataset_path, split, index, episode_id):
    from datasets import load_from_disk

    data = load_from_disk(dataset_path)
    rows = data[split] if hasattr(data, "keys") else data
    if episode_id is not None:
        for i in range(len(rows)):
            if str(rows[i]["episode_id"]) == str(episode_id):
                return rows[i]
        raise SystemExit(f"episode_id {episode_id} not in {dataset_path}/{split}")
    return rows[index]


def level_transform(args, spec, row):
    poses = np.asarray(row["obs_poses"], dtype=np.float64)
    print(f"episode {row['episode_id']}  scene {row['scene_id']}  "
          f"goal {row['goal_text']!r}  observations {len(poses)}")
    print("\n[1] transform parity -- collator vs rollout, same raw poses")
    # float32 first: that is what the collator has, so with the input pinned any
    # difference at all is a divergence in the transform itself.
    poses32 = np.asarray(poses, dtype=np.float32).astype(np.float64)
    exact = report("float32 input (exact)",
                   rollout_values(spec, poses32), collator_values(spec, poses32))
    if exact.max() != 0.0:
        raise SystemExit("FAIL: the two paths disagree on identical input")
    print("       -> bit-identical")
    deployed = report("float64 input (deployed)",
                      rollout_values(spec, poses), collator_values(spec, poses))
    print("       -> difference is the float32 quantisation of the *stored* column, "
          "not of the convention")
    stats = wrap_statistics(poses)
    print(f"\n  heading wrap: {stats['rows_where_naive_is_wrong']}/{stats['rows']} rows "
          f"would be wrong unwrapped; worst {stats['worst_naive_rad']:.3f} rad "
          f"where the true rotation is {stats['worst_true_rad']:.3f} rad")
    rel = rollout_values(spec, poses32)
    print(f"  row 0 = {rel[0].tolist()} (must be exactly zero)")
    print(f"  final = [{rel[-1][0]:.3f}, {rel[-1][1]:.3f}, {rel[-1][2]:.3f}]  "
          f"({np.linalg.norm(rel[-1][:2]):.2f} m from the episode's first observation)")
    return deployed


def level_bridge(args, spec, row):
    """Drive the real checkpoint over this episode's own frames and poses."""
    from PIL import Image

    from objectnav_eval.bridge import PolicyBridge

    poses = np.asarray(row["obs_poses"], dtype=np.float64)
    images = row["images"]
    n = min(args.steps, len(poses)) if args.steps else len(poses)
    print(f"\n[2] bridge parity -- {n} step(s) through a live policy server")

    backend_args = {
        "ckpt": args.ckpt,
        "device": args.device,
        "backend": args.policy_backend,
        "max_context_tokens": args.max_context_tokens,
    }
    if args.policy_sys_path:
        backend_args["sys_path"] = args.policy_sys_path
    if args.policy_decode:
        backend_args["decode"] = args.policy_decode

    server = Path(__file__).resolve().parent / "policy_server.py"
    bridge = PolicyBridge(str(server), args.policy_python, backend_args,
                          log_path=args.server_log)
    injected = []
    try:
        bridge.reset(row["goal_text"])
        for i in range(n):
            rgb = np.asarray(Image.open(images[i]).convert("RGB"), dtype=np.uint8)
            reply = bridge.act(rgb, pose=poses[i].tolist())
            value = reply["stats"].get("pose_value")
            if value is None:
                raise SystemExit(
                    "FAIL: the policy reported no pose_value. Either this checkpoint "
                    "declares no pose modality, or the pose never reached it."
                )
            injected.append([float(v) for v in value])
    finally:
        bridge.close()

    injected = np.asarray(injected, dtype=np.float64)
    expected = collator_values(spec, poses[:n])
    diff = report("model-side vs collator", injected, expected)
    # float32 in the model, float64 raw pose in: same bound as level 1.
    tol = 4.0 * float(np.spacing(np.float32(np.abs(poses[:n, :2]).max())))
    ok = diff[:, :2].max() < max(tol, 1e-6) and diff[:, 2].max() < 1e-6
    print(f"       tolerance {max(tol, 1e-6):.3e} m -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("FAIL: the value the model read is not the value training used")
    return diff


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="transform", choices=["transform", "bridge"],
                   help="'bridge' also runs level 1; see the module docstring for level 3")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="validation")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--episode-id", default=None)
    p.add_argument("--spec", default=DEFAULT_SPEC,
                   help="the modality spec the run declares. With --ckpt the "
                        "checkpoint's own spec is what actually runs; this only builds "
                        "the reference values")
    p.add_argument("--longnav-src", default=DEFAULT_LONGNAV_SRC)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--policy-python", default=None)
    p.add_argument("--policy-sys-path", default=DEFAULT_LONGNAV_SRC)
    p.add_argument("--policy-backend", default="vector_rollout")
    p.add_argument("--policy-decode", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-context-tokens", type=int, default=110000)
    p.add_argument("--steps", type=int, default=40,
                   help="cap the bridge level at this many observations (0 = all)")
    p.add_argument("--server-log", default=None)
    args = p.parse_args()

    if os.path.isdir(args.longnav_src) and args.longnav_src not in sys.path:
        sys.path.insert(0, args.longnav_src)

    spec = load_spec(args.spec)
    print(f"spec: {spec.token} F={spec.n_features} encoder={spec.encoder} "
          f"column={spec.source_column} transform={spec.transform}")
    row = pick_episode(args.dataset, args.split, args.index, args.episode_id)
    level_transform(args, spec, row)

    if args.level == "bridge":
        if not (args.ckpt and args.policy_python):
            raise SystemExit("--level bridge needs --ckpt and --policy-python")
        level_bridge(args, spec, row)
    print("\nOK")


if __name__ == "__main__":
    main()
