"""Running many episodes over many GPUs.

Workers are separate processes because ``CUDA_VISIBLE_DEVICES`` has to be set
before ``habitat_sim`` and torch initialise; a thread pool cannot pin a GPU.
Each worker re-runs the entry-point script with ``--worker-shard <file>``, and
that file carries **the whole configuration as JSON** rather than a rebuilt
command line -- rebuilding a worker's argv flag by flag is how a flag silently
stops being forwarded, and the resulting run looks fine while measuring
something else.

Sharding is load-balanced, not by scene. Scene load dominates a short episode,
so episodes of one scene are kept together -- but only up to a point: sharding
purely by scene idles every worker beyond the scene count (val_mini has two
scenes, so four GPUs would leave two doing nothing). Scenes are cut into blocks
no larger than an even share, and the blocks are packed onto the least-loaded
worker, so a scene is reloaded only when that is the price of using an otherwise
idle GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from objectnav_eval.episodes import EvalEpisode


def shard_episodes(
    episodes: Sequence[EvalEpisode], num_slots: int
) -> List[List[EvalEpisode]]:
    """Split episodes across workers, keeping scene locality but filling every
    worker."""
    if num_slots <= 1:
        return [list(episodes)]
    by_scene: Dict[str, List[EvalEpisode]] = defaultdict(list)
    for episode in episodes:
        by_scene[episode.scene_id].append(episode)
    block = max(1, -(-len(episodes) // num_slots))
    groups = [
        eps[i : i + block]
        for _scene, eps in sorted(by_scene.items(), key=lambda kv: -len(kv[1]))
        for i in range(0, len(eps), block)
    ]
    shards: List[List[EvalEpisode]] = [[] for _ in range(num_slots)]
    for group in sorted(groups, key=len, reverse=True):
        min(shards, key=len).extend(group)
    return shards


def run_parallel(
    episodes: Sequence[EvalEpisode],
    config: Any,
    entry_script: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Fan the episodes out over the configured GPU slots and collect results.

    Returns ``(results, skipped, failed_shard_indices)``. A failed shard is
    reported loudly and excluded; the summary then covers only the shards that
    completed, and says so.
    """
    slots = config.gpu_slots()
    shards = shard_episodes(episodes, len(slots))
    shard_dir = Path(config.output_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for index, (gpu, shard) in enumerate(zip(slots, shards)):
        if not shard:
            continue
        spec_path = shard_dir / f"shard_{index}.json"
        out_path = shard_dir / f"shard_{index}_results.json"
        worker_config = config.to_dict()
        worker_config["shard_index"] = index
        # A worker never re-shards, so it must not re-subsample either: it is
        # given uids and resolves exactly those.
        spec_path.write_text(
            json.dumps(
                {
                    "out": str(out_path),
                    "episodes": [e.uid for e in shard],
                    "config": worker_config,
                },
                indent=2,
            )
        )
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("MAGNUM_LOG", "quiet")
        env.setdefault("HABITAT_SIM_LOG", "quiet")
        log_path = shard_dir / f"shard_{index}.log"
        command = [sys.executable, str(Path(entry_script).resolve()),
                   "--worker-shard", str(spec_path)]
        print(f"  shard {index}: {len(shard)} episode(s) on GPU {gpu} -> {log_path}")
        log = open(log_path, "w")
        processes.append(
            (index, out_path, log_path, log,
             subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        )

    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[int] = []
    for index, out_path, log_path, log, process in processes:
        code = process.wait()
        log.close()
        if code != 0 or not out_path.exists():
            failed.append(index)
            print(f"  shard {index} FAILED (exit {code}); see {log_path}")
            try:
                tail = Path(log_path).read_text().splitlines()[-15:]
                print("  " + "\n  ".join(tail))
            except OSError:
                pass
            continue
        payload = json.loads(out_path.read_text())
        results.extend(payload["results"])
        skipped.extend(payload["skipped"])
        print(
            f"  shard {index} ok: {len(payload['results'])} evaluated, "
            f"{len(payload['skipped'])} skipped"
        )
    if failed:
        print(
            f"\nWARNING: {len(failed)} shard(s) failed: {failed}. "
            "The summary below covers only the shards that completed."
        )
    return results, skipped, failed


def read_shard_spec(path: str) -> Dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    for key in ("out", "episodes", "config"):
        if key not in spec:
            raise ValueError(f"shard spec {path} is missing '{key}'")
    return spec


def write_shard_output(
    path: str, results: Sequence[Dict[str, Any]], skipped: Sequence[Dict[str, Any]]
) -> None:
    """Write a shard's raw results.

    ``allow_nan=True`` deliberately: this file is an *internal* handoff read
    back by :func:`run_parallel` in the same Python, and a genuinely infinite
    goal distance must survive it unchanged. Turning ``inf`` into ``null`` here
    would make a parallel run and a single-process run disagree about the
    distance means. The user-facing results file and JSONL are strict JSON --
    that conversion happens once, at the end, in :mod:`objectnav_eval.reporting`.
    """
    Path(path).write_text(
        json.dumps(
            {"results": list(results), "skipped": list(skipped)},
            indent=2, default=float, allow_nan=True,
        )
    )
