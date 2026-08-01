"""Evaluate a closed-loop policy on standard ObjectNav episodes.

This is the entry point of :mod:`objectnav_eval`. It selects episodes, screens
them against the robot's own recomputed navmesh, runs the policy in the
simulator, and writes an aggregate summary, a per-episode JSONL and (optionally)
a real-time video per episode.

    # what would run, and what would be skipped -- no model, no rollout
    python scripts/eval_objectnav_policy.py --dry-run \
        --episodes .../objectnav/hm3d/v1/val_mini --scene-root .../scene_datasets

    # the harness controls: image-blind, and the first thing to run on a new setup
    python scripts/eval_objectnav_policy.py --policy forward \
        --episodes .../val_mini --scene-root .../scene_datasets \
        --output-dir dump/eval_forward --max-steps 40

    # a checkpoint, over four GPUs, with video
    python scripts/eval_objectnav_policy.py --ckpt dump/vector_sft/checkpoint-2500 \
        --episodes .../val_mini --scene-root .../scene_datasets \
        --output-dir dump/eval_ckpt2500 --max-steps 200 --record-video \
        --gpus 4,5,6,7 --policy-python /workspace/conda/envs/longnav_vlm/bin/python

    # re-aggregate a finished or partial run without repeating any rollout
    python scripts/eval_objectnav_policy.py --report dump/eval_ckpt2500/shards \
        --episodes .../val_mini

See docs/OBJECTNAV_EVAL.md for the architecture and the extension points.
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from objectnav_eval.config import EvalConfig, add_arguments, config_from_args  # noqa: E402
from objectnav_eval.episodes import (  # noqa: E402
    EpisodeFilter,
    build_episode_source,
    select_by_uid,
)
from objectnav_eval.harness import dry_run, run_episodes  # noqa: E402
from objectnav_eval.parallel import (  # noqa: E402
    read_shard_spec,
    run_parallel,
    write_shard_output,
)
from objectnav_eval.reporting import (  # noqa: E402
    load_and_merge,
    print_summary,
    summarize,
    write_episode_lines,
    write_results,
)

SERVER_SCRIPT = Path(__file__).resolve().parent / "policy_server.py"


def load_selected(config: EvalConfig):
    source = build_episode_source(config.episode_source, path=config.episodes)
    episodes = source.load()
    selection = EpisodeFilter(
        **config.selection_lists(),
        max_scenes=config.max_scenes,
        sample_per_scene=config.sample_per_scene,
        num_episodes=config.num_episodes,
        seed=config.seed,
    )
    return episodes, selection.apply(episodes)


def worker_main(shard_path: str) -> None:
    """The ``--worker-shard`` role: run exactly the episodes this shard lists."""
    spec = read_shard_spec(shard_path)
    config = EvalConfig.from_dict(spec["config"])
    all_episodes, _ = load_selected(config)
    episodes = select_by_uid(all_episodes, spec["episodes"])
    results, skipped = run_episodes(episodes, config, str(SERVER_SCRIPT))
    write_shard_output(spec["out"], results, skipped)
    print(
        f"[shard {config.shard_index}] done: {len(results)} evaluated, "
        f"{len(skipped)} skipped",
        flush=True,
    )


def report_only(path: str, output_dir: str) -> None:
    merged = load_and_merge(Path(path))
    results, skipped = merged["results"], merged["skipped"]
    summary = summarize(results, skipped, len(results) + len(skipped))
    print_summary(summary, results, skipped)
    out_path = write_results(
        merged["path"], summary, results, skipped, merged.get("config")
    )
    lines_path = write_episode_lines(out_path.with_suffix(".jsonl"), results, skipped)
    print(f"\nResults -> {out_path}")
    print(f"Per-episode -> {lines_path} (one line per episode)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--worker-shard", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true",
                        help="screen the episodes and print what would run: no model, no "
                             "rollout, one scene load per scene")
    parser.add_argument("--report", default=None,
                        help="re-aggregate an existing results JSON (or a shards/ "
                             "directory) and exit, recovering the summary after a partial "
                             "run without repeating any rollout")
    add_arguments(parser)
    args = parser.parse_args()

    if args.worker_shard:
        return worker_main(args.worker_shard)
    if args.report:
        return report_only(args.report, args.output_dir)

    config = config_from_args(args)
    _all, episodes = load_selected(config)
    if not episodes:
        raise SystemExit("no episodes selected")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(episodes)} episode(s) from {config.episodes}")

    if args.dry_run:
        dry_run(episodes, config)
        return
    if config.policy == "model" and not config.ckpt:
        raise SystemExit("--ckpt is required unless --dry-run or --policy is a control")

    slots = config.gpu_slots()
    if len(slots) == 1 and slots[0] is None:
        results, skipped = run_episodes(episodes, config, str(SERVER_SCRIPT))
    else:
        results, skipped, _failed = run_parallel(episodes, config, __file__)

    summary = summarize(results, skipped, len(episodes))
    print_summary(summary, results, skipped)

    out_path = write_results(
        output_dir / f"objectnav_eval_results{config.tag}.json",
        summary, results, skipped, config.to_dict(),
    )
    lines_path = write_episode_lines(out_path.with_suffix(".jsonl"), results, skipped)
    print(f"\nResults -> {out_path}")
    print(f"Per-episode -> {lines_path} (one line per episode)")
    if config.record_video:
        print(f"Videos  -> {output_dir / 'videos'} (real time at {config.fps} fps)")


if __name__ == "__main__":
    main()
