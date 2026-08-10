"""Run ``generate_pointnav_dataset.py`` across many scenes, without starving the box.

A work queue over scenes, one worker subprocess per in-flight scene. Sized for an
overnight run on a node that is *already busy*: the assumption throughout is that
something more important is running and this must yield to it.

------------------------------------------------------------------------
Why a launcher rather than ``xargs -P`` (the ``batch_gen.bash`` pattern)
------------------------------------------------------------------------
Three things ``xargs`` cannot do, each of which matters here:

* **GPU assignment.** habitat-sim renders on a GPU. On a node whose GPUs are already
  saturated by training, workers must be spread evenly rather than piled onto whichever
  device is device 0 -- a DDP job runs at the speed of its slowest rank, so concentrating
  the interference on two GPUs slows the whole job as much as spreading it over eight and
  is far less fair.
* **Backpressure.** Load average, free disk and free memory are all checked before each
  worker is started, so a run that starts safe stays safe when something else ramps up.
  A worker already running is never killed -- only new ones are withheld.
* **Resume by completeness.** A scene is skipped when its shard already holds the target
  episode count. The generator resumes *within* a scene by episode id anyway; this avoids
  paying a simulator startup to discover there is nothing to do.

Workers are ``nice``d, so on a contended box they lose to anything at default priority.

    python scripts/batch_generate_pointnav.py \
        --scene-root /Projects/data/scene_datasets \
        --scene-glob 'hm3d/train/*/*.basis.glb' \
        --output-dir /Projects/data/pointnav_hm3d \
        --episodes-per-scene 25 --frames-per-episode 2000 \
        --workers 12 --gpus 0,1,2,3,4,5,6,7

Ctrl-C once asks for a graceful stop: no new workers start, the running ones are left to
finish the scene they are on. Everything is resumable, so a hard kill costs at most the
episodes in flight.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from continuous_demos.corpus_io import load_done_episode_ids, load_failed_episode_ids

#: Passed through to the per-scene generator untouched. Anything not in here is a
#: launcher concern (parallelism, throttling, bookkeeping) and never reaches a worker.
PASSTHROUGH_FLAGS = (
    "frames-per-episode", "min-frames", "frame-tolerance", "min-segment-frames",
    "fps", "width", "height", "panoramic", "urdf-path", "image-ext",
    "corpus-seed", "min-island-area", "min-island-radius",
    "min-geodesic", "max-geodesic", "min-euclid", "max-height-delta", "min-turn-deg",
    "stratified-pool",
    "segment-retries", "tail-speed-mps", "goal-threshold", "min-segment-spl",
    "segment-rollbacks", "time-backstop-s", "fall-speed-mps", "fall-duration-s",
    "ref-miss-flag-m", "reject-ref-miss-s",
)


@dataclass
class Worker:
    scene: str
    proc: subprocess.Popen
    gpu: Optional[int]
    started: float
    log: Path

    @property
    def elapsed(self) -> float:
        return time.time() - self.started


@dataclass
class Throttle:
    """The conditions under which a new worker may start.

    Checked before each launch, never against a worker already running: killing work in
    flight to satisfy a threshold wastes the simulator startup and the episodes in
    progress, and the whole point is to be a good neighbour rather than a twitchy one.
    """

    max_load: float = 0.0  #: 0 disables. Compared against the 1-minute average.
    min_free_gb: float = 200.0
    min_free_mem_gb: float = 64.0
    cpus: int = field(default_factory=lambda: os.cpu_count() or 1)

    def blocked(self, output_dir: Path) -> Optional[str]:
        free_gb = shutil.disk_usage(output_dir).free / 1e9
        if free_gb < self.min_free_gb:
            return f"only {free_gb:.0f} GB free on {output_dir} (floor {self.min_free_gb:.0f})"
        if self.max_load:
            load1 = os.getloadavg()[0]
            if load1 > self.max_load:
                return f"load {load1:.1f} over ceiling {self.max_load:.1f}"
        free_mem = _free_mem_gb()
        if free_mem is not None and free_mem < self.min_free_mem_gb:
            return f"only {free_mem:.0f} GB RAM available (floor {self.min_free_mem_gb:.0f})"
        return None


def _free_mem_gb() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def discover_scenes(args) -> List[str]:
    """Scene ids to cover, in a stable order. Same three ways in as the generator."""
    scenes: List[str] = []
    for entry in args.scenes or []:
        scenes.extend(s for s in entry.split(",") if s)
    if args.scene_glob:
        root = Path(args.scene_root)
        scenes.extend(str(p.relative_to(root)) for p in sorted(root.glob(args.scene_glob)))
    seen, ordered = set(), []
    for scene in scenes:
        if scene not in seen:
            seen.add(scene)
            ordered.append(scene)
    if args.max_scenes:
        ordered = ordered[: args.max_scenes]
    return ordered


def scene_stem(scene_id: str) -> str:
    """The shard name the generator will use. Mirrors ``corpus_io.scene_name``."""
    return os.path.splitext(os.path.basename(scene_id))[0]


def scene_is_done(scene_id: str, metadata_dir: Path, log_file: Path, target: int) -> bool:
    """Has this scene already had its full quota attempted?

    Counted as written + previously rejected, because a rejected episode is not retried:
    the generator's own resume logic skips both, so a scene whose remainder is all
    rejections would otherwise be relaunched on every restart forever.
    """
    stem = scene_stem(scene_id)
    done = len(load_done_episode_ids(metadata_dir, stem))
    failed = sum(1 for ep in load_failed_episode_ids(log_file) if ep.startswith(f"{stem}_pn_"))
    return done + failed >= target


def build_command(args, scene: str) -> List[str]:
    cmd = [args.python, str(Path(__file__).with_name("generate_pointnav_dataset.py")),
           "--scene-root", args.scene_root,
           "--scenes", scene,
           "--output-dir", str(args.output_dir),
           "--episodes-per-scene", str(args.episodes_per_scene)]
    for flag in PASSTHROUGH_FLAGS:
        value = getattr(args, flag.replace("-", "_"), None)
        if value is not None:
            cmd += [f"--{flag}", str(value)]
    if args.adopt_existing_corpus:
        cmd.append("--adopt-existing-corpus")
    return cmd


def launch(args, scene: str, gpu: Optional[int], log_dir: Path) -> Worker:
    env = dict(os.environ)
    if gpu is not None:
        # habitat-sim picks its device from the visible set, so this is what spreads the
        # render load. Even spreading, not concentration: a DDP job runs at the speed of
        # its slowest rank.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # One BLAS thread per worker. Without this every worker spawns a thread pool sized to
    # the whole machine, and N workers oversubscribe the box by N x cores.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    log = log_dir / f"{scene_stem(scene)}.log"
    handle = open(log, "a", encoding="utf-8")
    handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} =====\n")
    handle.flush()
    proc = subprocess.Popen(
        build_command(args, scene), env=env, stdout=handle, stderr=subprocess.STDOUT,
        # Below everything else on the box, so this loses every scheduling contest.
        preexec_fn=(lambda: os.nice(args.nice)) if args.nice else None,
        start_new_session=True,
    )
    return Worker(scene=scene, proc=proc, gpu=gpu, started=time.time(), log=log)


def episode_count(metadata_dir: Path, scenes: List[str]) -> int:
    return sum(len(load_done_episode_ids(metadata_dir, scene_stem(s))) for s in scenes)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene-root", required=True)
    p.add_argument("--scenes", action="append", default=None)
    p.add_argument("--scene-glob", default=None)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--episodes-per-scene", type=int, default=25)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--adopt-existing-corpus", action="store_true")

    par = p.add_argument_group("parallelism and neighbourliness")
    par.add_argument("--workers", type=int, default=8,
                     help="scenes in flight. Each is one habitat simulator, so this is "
                          "also the number of processes competing for the GPUs")
    par.add_argument("--gpus", default=None,
                     help="comma-separated device ids to spread workers over "
                          "(default: whatever CUDA_VISIBLE_DEVICES already says)")
    par.add_argument("--nice", type=int, default=10,
                     help="worker niceness; 0 disables. Default loses every contest "
                          "against a default-priority training job")
    par.add_argument("--load-ceiling", type=float, default=0.0,
                     help="hold off starting workers while the 1-min load average is "
                          "above this. 0 disables")
    par.add_argument("--min-free-gb", type=float, default=200.0)
    par.add_argument("--min-free-mem-gb", type=float, default=64.0)
    par.add_argument("--poll-seconds", type=float, default=5.0)
    par.add_argument("--status-seconds", type=float, default=120.0)

    gen = p.add_argument_group("passed through to the per-scene generator")
    gen.add_argument("--frames-per-episode", type=int, default=2000)
    gen.add_argument("--min-frames", type=int, default=None)
    gen.add_argument("--frame-tolerance", type=int, default=100)
    gen.add_argument("--min-segment-frames", type=int, default=50)
    gen.add_argument("--fps", type=int, default=25)
    gen.add_argument("--width", type=int, default=640)
    gen.add_argument("--height", type=int, default=480)
    gen.add_argument("--panoramic", type=int, default=1)
    gen.add_argument("--urdf-path", default=None)
    gen.add_argument("--image-ext", default="jpeg")
    gen.add_argument("--corpus-seed", type=int, default=0)
    gen.add_argument("--min-island-area", type=float, default=None)
    gen.add_argument("--min-island-radius", type=float, default=None)
    gen.add_argument("--min-geodesic", type=float, default=None)
    gen.add_argument("--max-geodesic", type=float, default=None)
    gen.add_argument("--stratified-pool", type=int, default=None,
                     help="see generate_pointnav_dataset.py: 0/unset draws goals uniformly "
                          "on the navmesh, N>0 draws a target distance uniformly from "
                          "[min,max] geodesic and keeps the nearest of N screened "
                          "candidates. Without it, widening --max-geodesic lengthens the "
                          "tail but leaves the scene's own short-skewed shape")
    gen.add_argument("--min-euclid", type=float, default=None)
    gen.add_argument("--max-height-delta", type=float, default=None)
    gen.add_argument("--min-turn-deg", type=float, default=None)
    gen.add_argument("--segment-retries", type=int, default=None)
    gen.add_argument("--tail-speed-mps", type=float, default=None)
    gen.add_argument("--goal-threshold", type=float, default=None)
    gen.add_argument("--min-segment-spl", type=float, default=None)
    gen.add_argument("--segment-rollbacks", type=int, default=8,
                     help="a 2000-frame episode holds ~19 segments, so the 3 that suits "
                          "a 400-frame episode is five times too tight")
    gen.add_argument("--time-backstop-s", type=float, default=None)
    gen.add_argument("--fall-speed-mps", type=float, default=None)
    gen.add_argument("--fall-duration-s", type=float, default=None)
    gen.add_argument("--ref-miss-flag-m", type=float, default=None)
    gen.add_argument("--reject-ref-miss-s", type=float, default=None)

    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and one example worker command, then exit")
    args = p.parse_args(argv)

    # Line-buffered, because this is always redirected to a file and always long-running.
    # Block buffering makes an overnight run look dead for hours and leaves nothing behind
    # if it dies -- the progress log is the only evidence it was ever alive.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover - very old Python
        pass

    scenes = discover_scenes(args)
    if not scenes:
        raise SystemExit("no scenes selected")
    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"
    log_dir = output_dir / "worker_logs"
    log_file = output_dir / "failed_episodes.log"

    gpus: List[Optional[int]] = [None]
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]

    target_total = len(scenes) * args.episodes_per_scene
    print(f"{len(scenes)} scene(s) x {args.episodes_per_scene} episode(s) "
          f"= {target_total} episodes of {args.frames_per_episode} frames")
    print(f"workers={args.workers} gpus={gpus} nice={args.nice}")
    if args.dry_run:
        print("\nexample worker command:")
        print("  " + " ".join(build_command(args, scenes[0])))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    throttle = Throttle(max_load=args.load_ceiling, min_free_gb=args.min_free_gb,
                        min_free_mem_gb=args.min_free_mem_gb)

    pending = [s for s in scenes
               if not scene_is_done(s, metadata_dir, log_file, args.episodes_per_scene)]
    print(f"{len(scenes) - len(pending)} scene(s) already complete; {len(pending)} to do\n")

    stopping = {"flag": False}

    def request_stop(signum, frame):
        if stopping["flag"]:
            print("\n! second interrupt -- leaving workers to be reaped by the OS")
            raise KeyboardInterrupt
        stopping["flag"] = True
        print("\n! stopping: no new workers; running ones will finish their scene")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    running: List[Worker] = []
    finished = failed = 0
    gpu_cursor = 0
    started_at = time.time()
    last_status = 0.0
    # Episodes a previous run already wrote. The rate and the ETA are computed against
    # what THIS run produced; counting resumed episodes as if they were made in the last
    # few seconds gives a six-figure ep/h and an ETA of zero, which is not merely ugly --
    # it is the number someone checks at 3am to decide whether the run is healthy.
    baseline = episode_count(metadata_dir, scenes)

    while (pending and not stopping["flag"]) or running:
        for worker in list(running):
            if worker.proc.poll() is None:
                continue
            running.remove(worker)
            finished += 1
            if worker.proc.returncode not in (0, 1):  # 1 == "nothing written", not a crash
                failed += 1
                print(f"  ! {scene_stem(worker.scene)} exited {worker.proc.returncode} "
                      f"-- see {worker.log}")

        while pending and len(running) < args.workers and not stopping["flag"]:
            reason = throttle.blocked(output_dir)
            if reason:
                if time.time() - last_status > args.status_seconds:
                    print(f"  holding off: {reason}")
                    last_status = time.time()
                break
            scene = pending.pop(0)
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            running.append(launch(args, scene, gpu, log_dir))

        if time.time() - last_status > args.status_seconds:
            last_status = time.time()
            written = episode_count(metadata_dir, scenes)
            elapsed = time.time() - started_at
            rate = (written - baseline) / max(elapsed, 1e-9) * 3600
            free_gb = shutil.disk_usage(output_dir).free / 1e9
            # ETA from SCENES remaining, not episodes. `target_total` is the number of
            # episodes *attempted*, and a sizeable fraction are rejected -- some scenes'
            # navmesh is optimistic for a 0.28 m robot -- so a run finishes when the
            # scenes run out and never reaches `target_total` written. Projecting off
            # episodes therefore reports an ETA that never arrives.
            done_scenes = len(scenes) - len(pending) - len(running)
            scene_rate = done_scenes / max(elapsed, 1e-9)
            eta = ((len(pending) + len(running)) / scene_rate / 3600
                   if scene_rate > 0 else float("inf"))
            print(f"[{elapsed/3600:5.2f}h] {written:6d} episodes, "
                  f"{done_scenes}/{len(scenes)} scenes  "
                  f"{len(running)} running, {len(pending)} queued, {failed} worker fault(s)  "
                  f"{rate:6.0f} ep/h  ETA {eta:5.1f}h  load {os.getloadavg()[0]:5.1f}  "
                  f"free {free_gb:5.0f} GB")

        time.sleep(args.poll_seconds)

    written = episode_count(metadata_dir, scenes)
    print(f"\nDone in {(time.time() - started_at)/3600:.2f}h: {written} episode(s) written "
          f"across {finished} scene run(s), {failed} worker fault(s).")
    print(f"Worker logs: {log_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
