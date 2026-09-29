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
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Imported for its registrations: the `pointnav` episode source and the `goal_pursuit`
# control. A registry entry is not worth a plugin loader, and importing it here rather
# than from `episodes.py` keeps the dependency pointing the way the package does.
import objectnav_eval.pointnav  # noqa: E402,F401
from objectnav_eval.config import (  # noqa: E402
    EvalConfig,
    add_arguments,
    config_from_args,
)
from objectnav_eval.episodes import (  # noqa: E402
    EpisodeFilter,
    build_episode_source,
    select_by_uid,
)
from objectnav_eval.harness import (  # noqa: E402
    dry_run,
    run_episode_queue,
    run_episodes,
)
from objectnav_eval.parallel import (  # noqa: E402
    append_shard_progress,
    claim_task,
    complete_task,
    load_shard_progress,
    read_shard_spec,
    run_parallel,
    shard_progress_path,
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
    # `source_kwargs` is empty for every source that takes only a path, so this is the
    # same call it always was; a source needing more (PointNav needs a seed and a count)
    # gets it from `--source-set` rather than from a flag per source on this CLI.
    source = build_episode_source(
        config.episode_source, path=config.episodes, **(config.source_kwargs or {})
    )
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
    # Resolve against the *selected* list, not the raw source list, so a worker
    # runs its episodes in the same canonical order a single-process run would.
    # This is not cosmetic: episode results depend on what ran before them in
    # the same worker (see docs/OBJECTNAV_EVAL.md, "Episode order matters"), so
    # resolving against source order would make a parallel run irreproducible
    # from a single-process one.
    _all, selected = load_selected(config)
    if spec.get("queue"):
        return queue_worker_main(spec, config, selected)
    episodes = select_by_uid(selected, spec["episodes"])
    out_path = Path(spec["out"])
    prior_results = []
    prior_skipped = []
    if out_path.exists():
        payload = json.loads(out_path.read_text())
        prior_results.extend(payload.get("results", []))
        prior_skipped.extend(payload.get("skipped", []))
    progress_path = shard_progress_path(out_path)
    progress_results, progress_skipped = load_shard_progress(progress_path)
    prior_results.extend(progress_results)
    prior_skipped.extend(progress_skipped)

    results_by_uid = {str(record["uid"]): record for record in prior_results}
    skipped_by_uid = {
        str(record["uid"]): record
        for record in prior_skipped
        if str(record["uid"]) not in results_by_uid
    }
    done = set(results_by_uid) | set(skipped_by_uid)
    pending = [episode for episode in episodes if episode.uid not in done]
    if done:
        print(
            f"[shard {config.shard_index}] resume: {len(done)} complete, "
            f"{len(pending)} pending",
            flush=True,
        )

    if pending:
        new_results, new_skipped = run_episodes(
            pending,
            config,
            str(SERVER_SCRIPT),
            progress_callback=lambda kind, record: append_shard_progress(
                progress_path, kind, record
            ),
        )
        results_by_uid.update(
            (str(record["uid"]), record) for record in new_results
        )
        for record in new_skipped:
            uid = str(record["uid"])
            if uid not in results_by_uid:
                skipped_by_uid[uid] = record

    order = {episode.uid: index for index, episode in enumerate(episodes)}
    results = sorted(results_by_uid.values(), key=lambda record: order[str(record["uid"])])
    skipped = sorted(skipped_by_uid.values(), key=lambda record: order[str(record["uid"])])
    write_shard_output(spec["out"], results, skipped)
    print(
        f"[shard {config.shard_index}] done: {len(results)} evaluated, "
        f"{len(skipped)} skipped",
        flush=True,
    )


def queue_worker_main(
    spec: dict, config: EvalConfig, selected: list
) -> None:
    """Claim one episode at a time while keeping the policy process alive."""
    queue_path = Path(spec["queue"])
    episodes_by_uid = {episode.uid: episode for episode in selected}
    out_path = Path(spec["out"])
    progress_path = shard_progress_path(out_path)
    prior_results = []
    prior_skipped = []
    if out_path.exists():
        payload = json.loads(out_path.read_text())
        prior_results.extend(payload.get("results", []))
        prior_skipped.extend(payload.get("skipped", []))
    progress_results, progress_skipped = load_shard_progress(progress_path)
    prior_results.extend(progress_results)
    prior_skipped.extend(progress_skipped)

    def next_episode(last_scene_id):
        while True:
            status, task = claim_task(
                queue_path, config.shard_index, last_scene_id
            )
            if status == "done":
                return None
            if status == "wait":
                time.sleep(0.5)
                continue
            uid = str(task["uid"])
            if uid not in episodes_by_uid:
                raise ValueError(f"task queue returned unknown episode {uid}")
            print(
                f"[worker {config.shard_index}] claimed {uid}", flush=True
            )
            return episodes_by_uid[uid]

    def record_progress(kind, record):
        append_shard_progress(progress_path, kind, record)
        complete_task(queue_path, config.shard_index, str(record["uid"]))

    new_results, new_skipped = run_episode_queue(
        next_episode,
        config,
        str(SERVER_SCRIPT),
        progress_callback=record_progress,
    )
    results_by_uid = {
        str(record["uid"]): record
        for record in [*prior_results, *new_results]
    }
    skipped_by_uid = {
        str(record["uid"]): record
        for record in [*prior_skipped, *new_skipped]
        if str(record["uid"]) not in results_by_uid
    }
    order = {episode.uid: index for index, episode in enumerate(selected)}
    results = sorted(results_by_uid.values(), key=lambda row: order[str(row["uid"])])
    skipped = sorted(skipped_by_uid.values(), key=lambda row: order[str(row["uid"])])
    write_shard_output(out_path, results, skipped)
    print(
        f"[worker {config.shard_index}] done: {len(new_results)} evaluated, "
        f"{len(new_skipped)} skipped this session",
        flush=True,
    )


def report_only(path: str, output_dir: str) -> None:
    merged = load_and_merge(Path(path))
    results, skipped = merged["results"], merged["skipped"]
    summary = summarize(results, skipped, len(results) + len(skipped))
    out_path = write_results(
        merged["path"], summary, results, skipped, merged.get("config")
    )
    lines_path = write_episode_lines(out_path.with_suffix(".jsonl"), results, skipped)
    print_summary(summary, results, skipped)
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
    if not config.episodes:
        raise SystemExit("--episodes is required")
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
    failed_shards = []
    if len(slots) == 1 and slots[0] is None:
        results, skipped = run_episodes(episodes, config, str(SERVER_SCRIPT))
    else:
        results, skipped, failed_shards = run_parallel(episodes, config, __file__)

    summary = summarize(results, skipped, len(episodes))
    out_path = write_results(
        output_dir / f"result{config.tag}.json",
        summary, results, skipped, config.to_dict(),
    )
    lines_path = write_episode_lines(out_path.with_suffix(".jsonl"), results, skipped)
    print_summary(summary, results, skipped)
    print(f"\nResults -> {out_path}")
    print(f"Per-episode -> {lines_path} (one line per episode)")
    if config.record_video:
        print(f"Videos  -> {output_dir / 'videos'} (real time at {config.fps} fps)")
    if failed_shards:
        raise SystemExit(f"evaluation failed in shard(s): {failed_shards}")


if __name__ == "__main__":
    main()
