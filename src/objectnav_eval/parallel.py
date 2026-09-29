"""Running many episodes over many GPUs.

Workers are separate processes because ``CUDA_VISIBLE_DEVICES`` has to be set
before ``habitat_sim`` and torch initialise; a thread pool cannot pin a GPU.
Each persistent worker re-runs the entry-point script with ``--worker-shard
<file>`` and claims one episode at a time from a durable shared queue. The queue
prefers the worker's current scene when possible, records completion only after
the episode progress is fsynced, and returns a crashed worker's claim for another
worker to finish. The shard file carries the whole configuration as JSON rather
than rebuilding worker arguments flag by flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from objectnav_eval.episodes import EvalEpisode

QUEUE_WAIT = "wait"
QUEUE_DONE = "done"


def shard_progress_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_progress.jsonl")


@contextmanager
def _live_result_lock(run_dir: Path):
    """Serialize progress appends with the run-level live-result snapshot."""
    import fcntl

    lock_path = run_dir / ".result.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_live_result(run_dir: Path) -> None:
    """Atomically publish all completed shard records as ``result.json``."""
    from objectnav_eval.reporting import summarize, write_results

    shard_dir = run_dir / "shards"
    results_by_uid: Dict[str, Dict[str, Any]] = {}
    skipped_by_uid: Dict[str, Dict[str, Any]] = {}
    for progress_path in sorted(shard_dir.glob("*_results_progress.jsonl")):
        results, skipped = load_shard_progress(progress_path)
        for record in results:
            uid = str(record["uid"])
            results_by_uid[uid] = record
            skipped_by_uid.pop(uid, None)
        for record in skipped:
            uid = str(record["uid"])
            if uid not in results_by_uid:
                skipped_by_uid[uid] = record

    expected = 0
    config = None
    queue_path = shard_dir / "task_queue.json"
    if queue_path.exists():
        try:
            expected = len(json.loads(queue_path.read_text()).get("tasks", []))
        except (OSError, json.JSONDecodeError):
            expected = 0
    for spec_path in sorted(shard_dir.glob("shard_*.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not queue_path.exists():
            expected += len(spec.get("episodes", []))
        config = config or spec.get("config")

    results = list(results_by_uid.values())
    skipped = list(skipped_by_uid.values())
    summary = summarize(results, skipped, expected)
    summary["live"] = {
        "complete": len(results) + len(skipped) == expected,
        "completed": len(results) + len(skipped),
        "expected": expected,
    }
    result_path = run_dir / "result.json"
    write_results(result_path, summary, results, skipped, config)


def append_shard_progress(
    path: str | Path, kind: str, record: Dict[str, Any]
) -> None:
    """Append one completed episode so a failed worker can resume its shard."""
    if kind not in {"result", "skipped"}:
        raise ValueError(f"unsupported shard progress kind {kind!r}")
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"kind": kind, "record": record},
        default=float,
        allow_nan=True,
        separators=(",", ":"),
    )
    with _live_result_lock(progress_path.parent.parent):
        with progress_path.open("ab+") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(payload.encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _write_live_result(progress_path.parent.parent)


def load_shard_progress(
    path: str | Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load valid progress lines, ignoring a truncated final write."""
    results: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, Dict[str, Any]] = {}
    progress_path = Path(path)
    if not progress_path.exists():
        return [], []
    for line in progress_path.read_text().splitlines():
        try:
            payload = json.loads(line)
            kind = payload["kind"]
            record = payload["record"]
            uid = str(record["uid"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if kind == "result":
            results[uid] = record
            skipped.pop(uid, None)
        elif kind == "skipped" and uid not in results:
            skipped[uid] = record
    return list(results.values()), list(skipped.values())


@contextmanager
def _task_queue_lock(queue_path: Path):
    """Serialize claims and completions across independently pinned workers."""
    import fcntl

    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_task_queue(queue_path: Path, state: Dict[str, Any]) -> None:
    temporary = queue_path.with_name(f".{queue_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")))
    os.replace(temporary, queue_path)


def initialize_task_queue(
    queue_path: str | Path,
    episodes: Sequence[EvalEpisode],
    completed_uids: Sequence[str] = (),
) -> None:
    """Create or resume the durable work queue, releasing stale claims."""
    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [{"uid": episode.uid, "scene_id": episode.scene_id} for episode in episodes]
    expected = [task["uid"] for task in tasks]
    completed = set(map(str, completed_uids))
    with _task_queue_lock(path):
        if path.exists():
            state = json.loads(path.read_text())
            existing = [task["uid"] for task in state.get("tasks", [])]
            if existing != expected:
                raise ValueError("task queue episode list differs from this run")
            completed.update(map(str, state.get("completed", [])))
        state = {
            "version": 1,
            "tasks": tasks,
            "pending": [task for task in tasks if task["uid"] not in completed],
            "claimed": {},
            "completed": [uid for uid in expected if uid in completed],
        }
        _write_task_queue(path, state)


def claim_task(
    queue_path: str | Path,
    worker_id: int | str,
    last_scene_id: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, str]]]:
    """Claim one episode, preferring the worker's currently loaded scene."""
    path = Path(queue_path)
    worker = str(worker_id)
    with _task_queue_lock(path):
        state = json.loads(path.read_text())
        claimed = state["claimed"]
        if worker in claimed:
            return "task", claimed[worker]
        pending = state["pending"]
        if pending:
            index = 0
            if last_scene_id is not None:
                index = next(
                    (
                        i
                        for i, task in enumerate(pending)
                        if task["scene_id"] == last_scene_id
                    ),
                    0,
                )
            task = pending.pop(index)
            claimed[worker] = task
            _write_task_queue(path, state)
            return "task", task
        if claimed:
            return QUEUE_WAIT, None
        return QUEUE_DONE, None


def complete_task(
    queue_path: str | Path, worker_id: int | str, episode_uid: str
) -> None:
    """Acknowledge the worker's current episode after its progress is durable."""
    path = Path(queue_path)
    worker = str(worker_id)
    with _task_queue_lock(path):
        state = json.loads(path.read_text())
        task = state["claimed"].get(worker)
        if task is None or task["uid"] != episode_uid:
            raise ValueError(
                f"worker {worker} cannot complete unclaimed episode {episode_uid}"
            )
        state["claimed"].pop(worker)
        if episode_uid not in state["completed"]:
            state["completed"].append(episode_uid)
        _write_task_queue(path, state)


def requeue_worker_task(
    queue_path: str | Path,
    worker_id: int | str,
    durable_uids: Sequence[str] = (),
) -> bool:
    """Return a crashed worker's claim to the front of the queue."""
    path = Path(queue_path)
    worker = str(worker_id)
    with _task_queue_lock(path):
        state = json.loads(path.read_text())
        task = state["claimed"].pop(worker, None)
        if task is None:
            return False
        if task["uid"] in set(map(str, durable_uids)):
            if task["uid"] not in state["completed"]:
                state["completed"].append(task["uid"])
        elif task["uid"] not in state["completed"]:
            state["pending"].insert(0, task)
        _write_task_queue(path, state)
        return True


def task_queue_snapshot(queue_path: str | Path) -> Dict[str, int]:
    path = Path(queue_path)
    with _task_queue_lock(path):
        state = json.loads(path.read_text())
    return {
        "expected": len(state["tasks"]),
        "pending": len(state["pending"]),
        "claimed": len(state["claimed"]),
        "completed": len(state["completed"]),
    }


def completed_uids(shard_dir: str | Path) -> set[str]:
    """Collect durable results from both completed and interrupted workers."""
    directory = Path(shard_dir)
    done: set[str] = set()
    for progress_path in directory.glob("*_results_progress.jsonl"):
        results, skipped = load_shard_progress(progress_path)
        done.update(str(record["uid"]) for record in results)
        done.update(str(record["uid"]) for record in skipped)
    for output_path in directory.glob("*_results.json"):
        try:
            payload = json.loads(output_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        done.update(str(record["uid"]) for record in payload.get("results", []))
        done.update(str(record["uid"]) for record in payload.get("skipped", []))
    return done


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
    """Run a persistent model worker per GPU slot over one shared task queue."""
    slots = config.gpu_slots()
    shard_dir = Path(config.output_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    queue_path = shard_dir / "task_queue.json"
    initialize_task_queue(queue_path, episodes, completed_uids(shard_dir))

    processes: Dict[int, Tuple[Path, Path, Any, subprocess.Popen]] = {}
    for index, gpu in enumerate(slots):
        spec_path = shard_dir / f"shard_{index}.json"
        out_path = shard_dir / f"shard_{index}_results.json"
        worker_config = config.to_dict()
        worker_config["shard_index"] = index
        spec_path.write_text(
            json.dumps(
                {
                    "out": str(out_path),
                    "episodes": [],
                    "queue": str(queue_path),
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
        print(f"  worker {index}: shared task queue on GPU {gpu} -> {log_path}")
        log = open(log_path, "a")
        processes[index] = (
            out_path,
            log_path,
            log,
            subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT),
        )

    failed_workers: List[int] = []
    while processes:
        for index, (_out_path, log_path, log, process) in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            processes.pop(index)
            if code != 0:
                failed_workers.append(index)
                requeue_worker_task(queue_path, index, completed_uids(shard_dir))
                print(f"  worker {index} FAILED (exit {code}); see {log_path}")
                try:
                    tail = Path(log_path).read_text().splitlines()[-15:]
                    print("  " + "\n  ".join(tail))
                except OSError:
                    pass
            else:
                print(f"  worker {index} finished")
        if processes:
            time.sleep(0.2)

    results_by_uid: Dict[str, Dict[str, Any]] = {}
    skipped_by_uid: Dict[str, Dict[str, Any]] = {}
    for progress_path in sorted(shard_dir.glob("*_results_progress.jsonl")):
        worker_results, worker_skipped = load_shard_progress(progress_path)
        for record in worker_results:
            uid = str(record["uid"])
            results_by_uid[uid] = record
            skipped_by_uid.pop(uid, None)
        for record in worker_skipped:
            uid = str(record["uid"])
            if uid not in results_by_uid:
                skipped_by_uid[uid] = record

    order = {episode.uid: index for index, episode in enumerate(episodes)}
    results = sorted(results_by_uid.values(), key=lambda row: order[str(row["uid"])])
    skipped = sorted(skipped_by_uid.values(), key=lambda row: order[str(row["uid"])])
    snapshot = task_queue_snapshot(queue_path)
    expected = snapshot["expected"]
    completed = snapshot["completed"]
    unresolved = failed_workers if completed != expected else []
    if unresolved:
        print(
            f"\nWARNING: task queue stopped at {completed}/{expected} after "
            f"worker failure(s): {unresolved}."
        )
    return results, skipped, unresolved


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
    output_path = Path(path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            {"results": list(results), "skipped": list(skipped)},
            indent=2, default=float, allow_nan=True,
        )
    )
    temporary_path.replace(output_path)
