"""Generate continuous ObjectNav demonstrations and dump them as a frames + 2D
pose dataset, in a layout compatible with ``build_uniform_dataset.py``.

For every episode of a Habitat-web ObjectNav ``.json.gz`` dataset we replay the
human demonstration as a continuous trajectory (the same segmented pure-pursuit /
in-place-rotation replay used by ``replay_human_demo.py``), then persist:

  * the per-frame RGB observations as image files, and
  * the aligned per-frame planar pose ``[X, Y, theta]`` (see ``sim.get_2d_pose``).

Unlike the original ``pos_rots`` / ``actions`` columnar format, this dataset has
no discrete actions -- the demonstration is continuous -- so each frame carries a
single ``pose2d`` entry instead.

Everything lives under a single ``--output-dir``; the image root defaults to
``<output_dir>/images`` (override with ``--image-root``) and the per-scene JSONL
files are tucked into ``<output_dir>/metadata`` to keep the top level tidy. The
two-level ``scene / episode`` image structure (and the relative ``rgb_paths``
stored in the JSONL) is identical regardless of where the image root points:

    <output_dir>/
        images/                    # == image_root (default)
            <scene>/
                <episode_id>/
                    frame_00000.png
                    frame_00001.png
                    ...
        metadata/
            <scene>.jsonl          # one JSON object per successful episode
        failed_episodes.log        # episodes skipped or that failed the goal check

Each JSONL line (one episode) has:

    {
      "episode_id":    "...",              # metadata
      "episode_index": 42,                  # index within data["episodes"] (debug)
      "scene_id":      "mp3d/....glb",
      "goal_text":     "chair",            # object_category
      "rgb_paths":     ["<scene>/<ep>/frame_00000.png", ...],  # relative to image_root
      "pose2d":        [[x, y, theta], ...],                   # aligned with rgb_paths
      "recording":     {"schema_version": 1, "native_fps": 25.0, ...},
      "quality":       {"schema_version": 1, "success": true, "final_dist_m": ...}
    }

``recording`` is the block described in ``continuous_demos.recording_metadata``:
how this corpus was recorded -- tick rate, the physics timestep that rate does
or does not divide, the velocity caps in force while driving, and provenance.
It is repeated on every record (cheap next to ``pose2d``) so a consumer never
has to be *told* the frame rate, a shard survives being copied or concatenated,
and a shard that was appended to across a parameter change is detectable rather
than silently mixed. Consumers read it via
``recording_metadata.resolve_corpus_params``; corpora written before the schema
existed carry no such key and were all recorded at 30 Hz.

``quality`` is the block described in ``continuous_demos.drive_failure``: how
this episode actually went. It carries the outcome, the final-pose distance the
goal check used (previously computed and thrown away, so a shipped corpus could
not be audited at all), the escape/support signals, the reference-path tracking
error, and the episode's motion composition.

Two rejection rules run per episode, and both are recorded rather than just
acted on:

  * the **goal check** -- final replayed pose within ``--goal-threshold`` metres
    of the demonstration's reference final position;
  * the **escape check** -- the robot must not leave the world. This cannot be
    a tighter goal threshold: the base is a virtual gantry, so a robot that has
    fallen out of the scene keeps tracking the demonstration *perfectly* in
    ``pose2d`` and lands well inside the goal threshold with a normal path
    length. See ``continuous_demos.drive_failure``.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from continuous_demos.replay import record_segmented_demo
from continuous_demos.drive_failure import (
    QUALITY_KEY,
    DriveFailureConfig,
    EpisodeHealth,
    EpisodeHealthMonitor,
)
from continuous_demos.recording_metadata import (
    RecordingParams,
    format_grid_note,
    params_for_generation,
    shard_params,
)
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    extract_demo,
    find_looking_slices,
    find_scene_dataset_config,
    load_demo_objectnav,
    resolve_scene_path,
)


def scene_name(scene_id: str) -> str:
    """Short, filesystem-safe scene name, e.g. ``mp3d/17DRP5sb8fy.glb`` -> ``17DRP5sb8fy``."""
    return os.path.splitext(os.path.basename(scene_id))[0]


def sanitize(name: str) -> str:
    """Make an arbitrary id safe to use as a single path component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def load_done_episode_ids(metadata_dir: Path, scene: str) -> set:
    """Episode ids already completed for ``scene``, read from its own JSONL only.

    Because the metadata is sharded per scene (``<scene>.jsonl``), resuming a run
    -- even one targeting a single scene -- never has to scan the other scenes'
    files: we open just the shard that matches the scene being generated.
    """
    path = metadata_dir / f"{scene}.jsonl"
    done: set = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
            ep = rec.get("episode_id")
            if ep is not None:
                done.add(ep)
    return done


def assert_shard_is_appendable(jsonl_path: Path, params: RecordingParams) -> None:
    """Refuse to append records to a shard that was recorded differently.

    Generation is resumable and appends, so without this check a rerun after a
    parameter change (the 30 Hz -> 25 Hz move being exactly the case that
    motivated all of this) would leave one shard holding two time bases with
    nothing to mark the boundary. Aborting is deliberate: it is an operator
    error about the whole ``--output-dir``, not a per-episode failure, and the
    fix is a fresh output dir rather than anything this script can do.
    """
    existing = shard_params(jsonl_path) if jsonl_path.exists() else None
    if existing is None:
        if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
            raise RuntimeError(
                f"{jsonl_path} already holds records with no 'recording' block, i.e. from a "
                f"generation that predates the schema (those were all recorded at 30 Hz). "
                f"Appending {params.native_fps:g} Hz records to it would produce a shard with "
                "two time bases and no way to tell which record is which. Write to a fresh "
                "--output-dir."
            )
        return
    if not existing.agrees_with(params):
        raise RuntimeError(
            f"{jsonl_path} was recorded with {existing.consistency_view()} but this run is "
            f"{params.consistency_view()}. Resuming would mix two generations in one shard. "
            "Write to a fresh --output-dir."
        )


def load_failed_episode_ids(log_file: Path) -> set:
    """Episode ids previously logged as skipped/failed (so we don't retry them)."""
    failed: set = set()
    if not log_file.exists():
        return failed
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                failed.add(parts[1])  # episode_id column
    return failed


def reference_final_xy(path_3d: np.ndarray) -> np.ndarray:
    """Planar (X, Y) of the demo's final reference position.

    Uses the same planar convention as ``HabitatRobotSim.get_2d_pose``:
    ``X = habitat_x``, ``Y = -habitat_z`` (see the frame notes in sim.py).
    """
    hx, _hy, hz = path_3d[-1]
    return np.array([hx, -hz], dtype=float)


def group_episodes_by_scene(
    episodes: List[Dict[str, Any]], max_length: Optional[int]
) -> Tuple[Dict[str, List[Tuple[int, Dict[str, Any]]]], List[Tuple[int, str, str]]]:
    """Bucket ``(episode_index, episode)`` pairs by scene, applying the length filter.

    Returns ``(by_scene, too_long)`` where ``too_long`` is a list of
    ``(episode_index, episode_id, reason)`` for episodes dropped up front so the
    caller can log them.
    """
    by_scene: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    too_long: List[Tuple[int, str, str]] = []
    for idx, episode in enumerate(episodes):
        replay_len = len(episode.get("reference_replay", []))
        if max_length is not None and replay_len > max_length:
            too_long.append(
                (idx, episode.get("episode_id", str(idx)),
                 f"length {replay_len} > max_length {max_length}")
            )
            continue
        by_scene[episode["scene_id"]].append((idx, episode))
    return by_scene, too_long


def replay_one_episode(
    robot_sim,
    episode: Dict[str, Any],
    fps: int,
    max_steps: Optional[int],
    failure_config: Optional[DriveFailureConfig] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, EpisodeHealth]:
    """Replay a single episode and return ``(frames, poses, path_3d, health)``.

    ``frames[i]`` and ``poses[i]`` are aligned 1:1 (one planar pose per RGB
    frame). ``path_3d`` is the reference demo path, used for the goal check.
    ``health`` is the per-episode quality record built by
    :class:`~continuous_demos.drive_failure.EpisodeHealthMonitor` -- including
    whether the robot fell out of the world, which no planar quantity can show.
    """
    path_3d, path_headings, actions = extract_demo(episode, max_steps=max_steps)
    looking_slices = find_looking_slices(actions)

    robot_sim.reset(
        robot_translation=path_3d[0].copy(),
        robot_rotation=episode["start_rotation"].copy(),
        snap_to_navmesh=False,
        vertical_offset=None,
    )
    # Constructed after reset() so its spawn-height baseline is the episode's
    # own spawn, not wherever the previous episode left the robot.
    monitor = EpisodeHealthMonitor(robot_sim, fps=fps, config=failure_config)
    frames, poses = record_segmented_demo(
        robot_sim,
        path_3d,
        path_headings,
        looking_slices,
        timeout=60,
        fps=fps,
        debug_draw=False,
        health_monitor=monitor,
    )
    return frames, poses, path_3d, monitor.finish(poses)


def save_episode_frames(
    image_root: Path,
    scene: str,
    episode_dir_name: str,
    frames: List[np.ndarray],
    image_ext: str,
) -> List[str]:
    """Write frames to ``<image_root>/<scene>/<episode>/`` and return relative paths."""
    rel_dir = Path(scene) / episode_dir_name
    abs_dir = image_root / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths: List[str] = []
    for i, frame in enumerate(frames):
        fname = f"frame_{i:05d}.{image_ext}"
        imageio.imwrite(abs_dir / fname, frame)
        rgb_paths.append(str(rel_dir / fname))
    return rgb_paths


def generate(
    dataset_path: str,
    output_dir: Path,
    image_root: Optional[Path] = None,
    scene_root: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    max_length: Optional[int] = None,
    max_steps: Optional[int] = None,
    goal_threshold: float = 1.0,
    urdf_path: str = str(DEFAULT_URDF_PATH),
    image_ext: str = "png",
    log_file: Optional[Path] = None,
    max_episodes: Optional[int] = None,
    failure_config: Optional[DriveFailureConfig] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_root is None:
        image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = output_dir / "failed_episodes.log"

    data = load_demo_objectnav(dataset_path)
    episodes = data["episodes"]
    by_scene, too_long = group_episodes_by_scene(episodes, max_length)

    n_ok = 0
    n_failed = 0
    n_resumed = 0  # episodes skipped because a prior run already handled them
    processed = 0
    n_rejected: Dict[str, int] = defaultdict(int)  # rejections, by rule

    # Resume support: episodes already completed (per-scene JSONL shards, read
    # lazily below) or previously failed (the log) are skipped. Load the failed
    # set once up front; done ids are loaded per scene so a single-scene run only
    # ever touches that scene's shard.
    failed_ids = load_failed_episode_ids(log_file)
    log_is_new = not log_file.exists() or log_file.stat().st_size == 0

    # Append so failures accumulate across resumed runs instead of being wiped.
    with open(log_file, "a", encoding="utf-8") as log:
        if log_is_new:
            log.write(f"# Failed / skipped episodes for {dataset_path}\n")
            log.write("# columns: episode_index\tepisode_id\tscene\treason\n")
        for idx, ep_id, reason in too_long:
            if ep_id in failed_ids:
                continue  # already recorded on a previous run
            log.write(f"{idx}\t{ep_id}\t-\t{reason}\n")
            failed_ids.add(ep_id)
            n_failed += 1
        log.flush()

        for scene_id, scene_episodes in by_scene.items():
            scene = scene_name(scene_id)
            done_ids = load_done_episode_ids(metadata_dir, scene)
            print(
                f"\n=== Scene {scene} ({len(scene_episodes)} episode(s)"
                + (f", {len(done_ids)} already done" if done_ids else "")
                + ") ==="
            )

            try:
                scene_path = resolve_scene_path(scene_id, scene_root)
                scene_dataset_config = find_scene_dataset_config(scene_path, scene_root)
                robot_sim = build_robot_sim(
                    scene_path=scene_path,
                    scene_dataset_config=scene_dataset_config,
                    width=width,
                    height=height,
                    urdf_path=urdf_path,
                )
            except Exception as exc:  # scene unresolved / sim build failed
                print(f"  ! Skipping scene {scene}: {exc}")
                for idx, episode in scene_episodes:
                    ep_id = episode.get("episode_id", str(idx))
                    if ep_id in done_ids or ep_id in failed_ids:
                        continue
                    log.write(f"{idx}\t{ep_id}\t{scene}\tscene load failed: {exc}\n")
                    failed_ids.add(ep_id)
                    n_failed += 1
                log.flush()
                continue

            jsonl_path = metadata_dir / f"{scene}.jsonl"
            try:
                # Probe the recording parameters off the live simulator (the
                # physics timestep in particular is asked, not assumed) and
                # stamp them onto every record this scene writes. Inside the
                # try so the appendability check below cannot leak the sim.
                recording = params_for_generation(
                    fps,
                    robot_sim,
                    image_width=width,
                    image_height=height,
                    image_ext=image_ext,
                    goal_threshold_m=goal_threshold,
                    urdf=urdf_path,
                    generator="generate_pose2d_dataset.py",
                )
                print(f"  recording: {format_grid_note(recording)}")
                recording_blob = recording.as_dict()
                assert_shard_is_appendable(jsonl_path, recording)

                # Append so records from prior runs are preserved (resume).
                with open(jsonl_path, "a", encoding="utf-8") as jf:
                    for idx, episode in scene_episodes:
                        ep_id = episode.get("episode_id", str(idx))
                        # Resume: skip anything a prior run already finished or logged.
                        if ep_id in done_ids or ep_id in failed_ids:
                            n_resumed += 1
                            continue
                        if max_episodes is not None and processed >= max_episodes:
                            break
                        processed += 1
                        ep_dir_name = sanitize(ep_id)

                        try:
                            frames, poses, path_3d, health = replay_one_episode(
                                robot_sim, episode, fps=fps, max_steps=max_steps,
                                failure_config=failure_config,
                            )
                        except Exception as exc:
                            print(f"  [ep {idx} / {ep_id}] replay error: {exc}")
                            log.write(f"{idx}\t{ep_id}\t{scene}\treplay error: {exc}\n")
                            log.flush()
                            n_failed += 1
                            continue

                        if not frames:
                            log.write(f"{idx}\t{ep_id}\t{scene}\tno frames produced\n")
                            log.flush()
                            n_failed += 1
                            continue

                        # Goal check: final replayed pose vs reference final position.
                        # Note this passes for a robot that fell out of the world --
                        # the gantry keeps tracking the demo perfectly once nothing
                        # touches it -- which is exactly why the health monitor's
                        # escape check below is a separate, non-planar test and not a
                        # tighter threshold on this one.
                        final_xy = np.asarray(poses[-1][:2], dtype=float)
                        target_xy = reference_final_xy(path_3d)
                        dist = float(np.linalg.norm(final_xy - target_xy))
                        health.final_dist_m = dist
                        health.goal_threshold_m = float(goal_threshold)
                        if dist > goal_threshold:
                            health.success = False
                            health.outcome = "goal"

                        if not health.success:
                            reason = {
                                "goal": (f"final pose {dist:.3f}m from goal "
                                         f"(> {goal_threshold}m)"),
                                "escaped": (
                                    f"left the world: unsupported for "
                                    f"{health.unsupported_run_s:.2f}s "
                                    f"({health.unsupported_run_ticks} ticks), fell "
                                    f"{health.max_drop_m:.2f}m"
                                ),
                                "ref_miss": (
                                    f"off its reference path by > "
                                    f"{health.ref_miss_flag_m:g}m for "
                                    f"{health.ref_miss_worst_run_s:.2f}s"
                                ),
                            }.get(health.outcome, health.outcome)
                            print(f"  [ep {idx} / {ep_id}] FAIL ({health.outcome}) {reason}"
                                  " -- skipping")
                            log.write(f"{idx}\t{ep_id}\t{scene}\t{reason}\n")
                            log.flush()
                            n_failed += 1
                            n_rejected[health.outcome] += 1
                            continue

                        rgb_paths = save_episode_frames(
                            image_root, scene, ep_dir_name, frames, image_ext
                        )
                        pose2d = [
                            [float(p[0]), float(p[1]), float(p[2])] for p in poses
                        ]

                        record = {
                            "episode_id": ep_id,
                            "episode_index": idx,
                            "scene_id": scene_id,
                            "goal_text": episode.get("object_category", ""),
                            "rgb_paths": rgb_paths,
                            "pose2d": pose2d,
                            "recording": recording_blob,
                            QUALITY_KEY: health.as_dict(),
                        }
                        jf.write(json.dumps(record) + "\n")
                        jf.flush()
                        n_ok += 1
                        print(
                            f"  [ep {idx} / {ep_id}] OK  {len(frames)} frames, "
                            f"final-pose dist {dist:.2f}m, ref-miss p95 "
                            f"{(health.ref_miss_p95_m or 0.0):.2f}m, "
                            f"drive/turn {(health.frac_drive or 0):.0%}/"
                            f"{(health.frac_turn or 0):.0%}"
                        )
            finally:
                robot_sim.sim.close(destroy=True)

    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(n_rejected.items())) or "none"
    print(
        f"\nDone. {n_ok} episode(s) written, {n_failed} skipped/failed, "
        f"{n_resumed} skipped (already done/failed on a prior run). "
        f"Rejections by rule: {breakdown}. Log: {log_file}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Habitat human demos and save frames + 2D pose dataset."
    )
    parser.add_argument("dataset_path", help="Path to the ObjectNav .json.gz dataset.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Everything is written under here: images/, metadata/*.jsonl, and the log.",
    )
    parser.add_argument(
        "--image-root", type=Path, default=None,
        help="Root directory for saved frames (paths in the JSONL are relative to it). "
             "Defaults to <output_dir>/images.",
    )
    parser.add_argument(
        "--scene-root", default=None,
        help="Optional root used to resolve relative scene_id paths (see replay_human_demo).",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Replay/record FPS. MUST sit on habitat's fixed 0.008s Bullet "
             "physics grid, or every saturated tick's displacement alternates "
             "between two values (see HabitatRobotSim.step). The only integer "
             "rates whose period is a whole multiple of 0.008s are 125, 25, 5 "
             "and 1 Hz -- 25 Hz (dt=0.04s = 5 physics steps) is the only "
             "usable one. The previous 30 Hz default was off-grid and produced "
             "a 24%% per-tick velocity ripple. Whatever is passed here is "
             "written into every metadata record's 'recording' block, so "
             "consumers never have to be told it again.",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Skip episodes whose reference_replay is longer than this (pre-filter).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Optional cap on replay steps rendered per episode.",
    )
    parser.add_argument(
        "--goal-threshold", type=float, default=1.0,
        help="Keep a demo only if its final pose is within this many metres of the "
             "reference final position.",
    )
    parser.add_argument(
        "--urdf-path", default=str(DEFAULT_URDF_PATH), help="Robot URDF to use.",
    )
    parser.add_argument(
        "--image-ext", default="png", choices=["png", "jpg", "jpeg"],
        help="Image file format for saved frames.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Where to write the failed/skipped-episode log "
             "(default: <output_dir>/failed_episodes.log).",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Optional cap on total episodes processed (for quick testing).",
    )
    parser.add_argument(
        "--escape-unsupported-s", type=float,
        default=DriveFailureConfig.escape_unsupported_s,
        help="Reject an episode in which the robot has no contact with the scene "
             "for this many consecutive seconds -- the signature of it leaving the "
             "world, which no planar quantity can see. The longest contact-free "
             "run measured in a healthy episode is 1.0s; the shortest in an "
             "escaped one is 3.0s.",
    )
    parser.add_argument(
        "--reject-ref-miss-s", type=float, default=None,
        help="Additionally reject an episode that stays more than "
             "--ref-miss-flag-m off its reference path for this long. OFF by "
             "default: sustained tracking error measures demonstration QUALITY, "
             "not validity, and it cannot see an escape at all (an escaped robot "
             "tracks its path better than a healthy one). The statistic is "
             "recorded on every episode either way, so a cut can be chosen from "
             "the measured distribution.",
    )
    parser.add_argument(
        "--ref-miss-flag-m", type=float, default=DriveFailureConfig.ref_miss_flag_m,
        help="Reference-path miss counted as 'not following the path' when "
             "reporting (and optionally acting on) sustained tracking error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        image_root=args.image_root,
        scene_root=args.scene_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_length=args.max_length,
        max_steps=args.max_steps,
        goal_threshold=args.goal_threshold,
        urdf_path=args.urdf_path,
        image_ext=args.image_ext,
        log_file=args.log_file,
        max_episodes=args.max_episodes,
        failure_config=DriveFailureConfig(
            escape_unsupported_s=args.escape_unsupported_s,
            ref_miss_flag_m=args.ref_miss_flag_m,
            reject_ref_miss_s=args.reject_ref_miss_s,
        ),
    )


if __name__ == "__main__":
    main()
