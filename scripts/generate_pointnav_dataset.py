"""Generate continuous PointNav demonstrations as a frames + 2D pose + goal corpus.

The PointNav counterpart to ``generate_pose2d_dataset.py``, writing the *same* raw layout
so everything downstream -- the action-chunk builders, ``render_pose2d_videos.py``, the
resume logic -- reads it without knowing which task made it.

Unlike the ObjectNav generator this takes **no input demonstrations**. For each episode it
samples a start and a chain of goals on the robot navmesh, plans a shortest path to each,
turns in place onto it and drives it with the same DWPP pure-pursuit stack. Chaining is
what lets every episode be roughly the same length: segment ``k + 1`` starts where segment
``k`` finished, so the trajectory is physically continuous and the video never cuts.

    <output_dir>/
        images/<scene>/<episode_id>/frame_00000.png      # one file per tick
        metadata/<scene>.jsonl                            # one object per kept episode
        failed_episodes.log                               # TSV, one row per rejection
        corpus_params.json                                # the guard's marker

Each JSONL line is the ObjectNav record plus three PointNav-specific keys:

    {
      "episode_id":     "17DRP5sb8fy_pn_00042",
      "episode_index":  42,
      "scene_id":       "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
      "goal_text":      "",                     # present and empty: PointNav has no object
      "rgb_paths":      [...],                                    # N
      "pose2d":         [[x, y, theta], ...],                     # N
      "goal2d":         [[gx, gy, gtheta], ...],   # N, piecewise constant per segment
      "segment_index":  [0, 0, ..., 1, 1, ...],    # N
      "segments":       [{...}, ...],              # per-segment provenance and outcome
      "start_pose2d":   [x, y, theta],
      "sampling":       {...},                     # seed, island, screening config
      "recording":      {"task": "pointnav", "schema_version": 3, ...},
      "quality":        {...}
    }

``goal2d`` is in the planar control frame (``X = habitat_x``, ``Y = -habitat_z``) and its
theta is the path's final tangent -- the heading the robot is expected to hold on arrival.
That must be a genuine SE(2) quantity: a constant survives the model's
``relative_se2`` as ``wrap(0 - theta_anchor)``, which *is* the scene's compass at the
window origin, and is precisely the leak that frame conversion exists to remove.

------------------------------------------------------------------------
Scene selection is dataset-agnostic, by construction
------------------------------------------------------------------------
There is no MP3D or HM3D branch anywhere in this pipeline. Episodes are sampled off the
robot navmesh that ``build_robot_sim`` recomputes for every scene, so any scene habitat
can load is a scene this can generate from. Name scenes with ``--scenes`` (ids or paths),
discover them with ``--scene-glob``, or point ``--scene-dataset-config`` at a habitat
scene-dataset config and let it enumerate.

------------------------------------------------------------------------
Rejection, in two places
------------------------------------------------------------------------
**Sampling time** screens a candidate goal before the robot moves -- island, floor,
geodesic band, loop-back -- and simply redraws. Free.

**Drive time** means the robot tried and failed. The episode is not thrown away: the
failed segment's ticks are erased, the robot is put back where that segment started, and
a different goal is drawn (``--on-segment-failure rollback``, the default). ``goal2d``
stays piecewise-constant because the excursion is *erased* rather than relabelled -- the
alternative teaches a policy that goals move when you fail.

Rolled-back segments are still counted and still appear in ``segments`` with
``outcome="rolled_back"``. A corpus that hid them would read cleaner than the scenes are,
and every threshold re-cut from a pilot would be fit to a lie.

    python scripts/generate_pointnav_dataset.py \
        --scene-root /Projects/data/scene_datasets \
        --scene-glob 'mp3d/*/*.glb' \
        --output-dir data/pointnav --episodes-per-scene 200
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from continuous_demos import pointnav as pn
from continuous_demos.corpus_guard import (
    assert_corpus_marker_agrees,
    assert_output_dir_is_safe,
    assert_shard_is_appendable,
)
from continuous_demos.corpus_io import (
    FAILURE_LOG_NAME,
    FailureLog,
    append_metadata_record,
    load_done_episode_ids,
    sanitize,
    save_episode_frames,
    scene_name,
)
from continuous_demos.drive_failure import QUALITY_KEY, DriveFailureConfig
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    find_scene_dataset_config,
    resolve_scene_path,
)
from continuous_demos.pointnav_episode import record_pointnav_episode
from continuous_demos.recording_metadata import format_grid_note, params_for_generation

#: Written into every record's ``recording`` block and consistency-checked, so a PointNav
#: corpus cannot be poured into an ObjectNav output directory (or vice versa) unnoticed.
TASK_NAME = "pointnav"


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------
def discover_scenes(args) -> List[str]:
    """Scene ids to generate from, in a stable order.

    Three ways in, none of them dataset-specific:
      * ``--scenes`` -- explicit ids or paths, comma-separated or repeated;
      * ``--scene-glob`` -- a glob under ``--scene-root``;
      * ``--scene-dataset-config`` -- a habitat scene-dataset config, whose ``stages`` or
        ``scene_instances`` paths are globbed relative to the config's own directory.
    """
    scenes: List[str] = []
    for entry in args.scenes or []:
        scenes.extend(s for s in entry.split(",") if s)
    if args.scene_glob:
        if not args.scene_root:
            raise SystemExit("--scene-glob needs --scene-root to glob under")
        root = Path(args.scene_root)
        scenes.extend(str(p.relative_to(root)) for p in sorted(root.glob(args.scene_glob)))
    if args.scene_dataset_config:
        scenes.extend(scenes_from_dataset_config(Path(args.scene_dataset_config)))
    # Stable and de-duplicated: a scene named twice must not be generated twice, and the
    # order fixes which scenes a --max-scenes run picks.
    seen, ordered = set(), []
    for scene in scenes:
        if scene not in seen:
            seen.add(scene)
            ordered.append(scene)
    if args.max_scenes:
        ordered = ordered[: args.max_scenes]
    return ordered


def scenes_from_dataset_config(config_path: Path) -> List[str]:
    """Scene paths named by a habitat scene-dataset config.

    Reads the config's own glob patterns rather than assuming a directory layout, which
    is what keeps this working for a scene dataset nobody here has seen.
    """
    blob = json.loads(config_path.read_text())
    base = config_path.parent
    patterns: List[str] = []
    for key in ("stages", "scene_instances"):
        section = blob.get(key) or {}
        paths = section.get("paths") or {}
        for globs in paths.values():
            patterns.extend(globs if isinstance(globs, list) else [globs])
    found: List[str] = []
    for pattern in patterns:
        found.extend(str(p) for p in sorted(base.glob(pattern)))
    if not found:
        raise SystemExit(
            f"{config_path} named no scenes (looked at stages/scene_instances globs "
            f"{patterns or 'none'}). Use --scenes or --scene-glob instead."
        )
    return found


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_configs(args):
    """The three config objects, assembled once so every scene gets identical settings."""
    sample_config = pn.PointNavSampleConfig(
        min_island_area_m2=args.min_island_area,
        min_island_radius_m=args.min_island_radius,
        min_geodesic_m=args.min_geodesic,
        max_geodesic_m=args.max_geodesic,
        min_euclid_m=args.min_euclid,
        max_height_delta_m=args.max_height_delta,
        min_turn_deg=args.min_turn_deg,
        segment_retries=args.segment_retries,
        tail_speed_mps=args.tail_speed_mps,
    )
    failure_config = DriveFailureConfig(
        fall_speed_mps=args.fall_speed_mps,
        fall_duration_s=args.fall_duration_s,
        ref_miss_flag_m=args.ref_miss_flag_m,
        # ON by default here, unlike ObjectNav. There the reference is a human
        # demonstration, so a sustained miss measures imitation fidelity and its false
        # positives are real demonstrations. Here the reference is the navmesh shortest
        # path we chose ourselves, so being a metre off it for over a second means the
        # robot is fighting geometry the navmesh calls free -- exactly what a policy must
        # not learn from.
        reject_ref_miss_s=args.reject_ref_miss_s,
    )
    return sample_config, failure_config


def make_budget(args) -> pn.FrameBudget:
    return pn.FrameBudget(
        target_frames=args.frames_per_episode,
        min_frames=(args.min_frames if args.min_frames is not None
                    else int(args.frames_per_episode * 0.75)),
        min_segment_frames=args.min_segment_frames,
        exact=not args.no_exact_frames,
        tolerance_frames=args.frame_tolerance,
    )


def generate(args) -> int:
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root) if args.image_root else None

    # Before anything is created: this must not be writing into the scene tree it reads,
    # or into a corpus some other run owns. See continuous_demos.corpus_guard. Passing the
    # scene root as `dataset_path` protects the whole scene-datasets directory.
    assert_output_dir_is_safe(output_dir, args.scene_root or str(output_dir),
                              image_root=image_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_root is None:
        image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output_dir / FAILURE_LOG_NAME

    scenes = discover_scenes(args)
    if not scenes:
        raise SystemExit("no scenes selected; pass --scenes, --scene-glob or "
                         "--scene-dataset-config")
    print(f"{len(scenes)} scene(s), {args.episodes_per_scene} episode(s) each, "
          f"target {args.frames_per_episode} frames")

    sample_config, failure_config = build_configs(args)
    n_ok = n_failed = n_resumed = 0
    rejected: Dict[str, int] = defaultdict(int)
    rolled_back: Dict[str, int] = defaultdict(int)

    with FailureLog(log_file, source=f"pointnav over {len(scenes)} scene(s)") as log:
        for scene_id in scenes:
            scene = scene_name(scene_id)
            done_ids = load_done_episode_ids(metadata_dir, scene)
            print(f"\n=== Scene {scene}"
                  + (f" ({len(done_ids)} already done)" if done_ids else "") + " ===")

            try:
                scene_path = resolve_scene_path(scene_id, args.scene_root)
                scene_dataset_config = find_scene_dataset_config(scene_path, args.scene_root)
                robot_sim = build_robot_sim(
                    scene_path=scene_path,
                    scene_dataset_config=scene_dataset_config,
                    width=args.width, height=args.height,
                    urdf_path=args.urdf_path, num_cameras=args.panoramic,
                )
            except Exception as exc:
                print(f"  ! Skipping scene {scene}: {exc}")
                for index in range(args.episodes_per_scene):
                    ep_id = episode_id_for(scene, index)
                    if ep_id in done_ids:
                        continue
                    if log.record(index, ep_id, scene,
                                  pn.describe("scene_load_failed", str(exc))):
                        n_failed += 1
                continue

            try:
                recording = params_for_generation(
                    args.fps, robot_sim,
                    image_width=args.width, image_height=args.height,
                    image_ext=args.image_ext, goal_threshold_m=args.goal_threshold,
                    urdf=args.urdf_path, generator=Path(__file__).name,
                    num_cameras=args.panoramic, phase_free=False, task=TASK_NAME,
                )
                if not recording.on_physics_grid:
                    print(f"  ! {format_grid_note(recording)}")
                jsonl_path = metadata_dir / f"{scene}.jsonl"
                assert_corpus_marker_agrees(output_dir, metadata_dir, recording,
                                            args.adopt_existing_corpus)
                assert_shard_is_appendable(jsonl_path, recording)

                pathfinder = robot_sim.sim.pathfinder
                print(f"  navmesh: {pathfinder.navigable_area:.1f} m2 over "
                      f"{pathfinder.num_islands} island(s)")

                with open(jsonl_path, "a", encoding="utf-8") as shard:
                    for index in range(args.episodes_per_scene):
                        ep_id = episode_id_for(scene, index)
                        if ep_id in done_ids or ep_id in log.known:
                            n_resumed += 1
                            continue
                        if args.max_episodes is not None and n_ok >= args.max_episodes:
                            break

                        try:
                            rec = record_pointnav_episode(
                                robot_sim, scene=scene, episode_index=index,
                                corpus_seed=args.corpus_seed,
                                config=sample_config, budget=make_budget(args),
                                fps=args.fps, goal_threshold_m=args.goal_threshold,
                                min_segment_spl=args.min_segment_spl,
                                failure_config=failure_config,
                                rollback_limit=args.segment_rollbacks,
                                sim_time_limit_s=args.time_backstop_s,
                                verbose=args.verbose,
                            )
                        except Exception as exc:  # a scene fault must not kill the run
                            print(f"  [ep {index}] error: {exc}")
                            if log.record(index, ep_id, scene,
                                          pn.describe("record_error", str(exc))):
                                n_failed += 1
                            continue

                        for code, count in (rec.rollbacks.reasons if rec.rollbacks
                                            else {}).items():
                            rolled_back[code] += count

                        if not rec.ok:
                            print(f"  [ep {index}] REJECTED ({rec.reason}) {rec.detail}")
                            log.record(index, ep_id, scene,
                                       pn.describe(rec.reason, rec.detail))
                            n_failed += 1
                            rejected[rec.reason] += 1
                            continue

                        record = write_episode(
                            rec, shard, image_root=image_root, scene=scene,
                            scene_id=scene_id, ep_id=ep_id, episode_index=index,
                            image_ext=args.image_ext, recording=recording,
                        )
                        n_ok += 1
                        summary = pn.segments_summary(rec.segments)
                        print(f"  [ep {index}] OK  {len(record['rgb_paths'])} frames, "
                              f"{summary['num_segments']} segment(s), "
                              f"{summary['travelled_m']:.1f}m travelled"
                              + (f", {summary['num_rolled_back']} rolled back"
                                 if summary["num_rolled_back"] else ""))
            finally:
                robot_sim.sim.close(destroy=True)

    print(f"\nDone. {n_ok} episode(s) written, {n_failed} rejected, "
          f"{n_resumed} skipped (already done/failed on a prior run).")
    print(f"Rejections by rule: {dict(rejected) or 'none'}")
    print(f"Segment rollbacks by rule: {dict(rolled_back) or 'none'}")
    print(f"Log: {log_file}")
    return 0 if n_ok else 1


def episode_id_for(scene: str, index: int) -> str:
    """A pure function of scene and index, which is what makes resume exact.

    The episode's random seed is derived from the same two (plus ``--corpus-seed``), so a
    re-run re-derives the same episodes and skips exactly the ones already present.
    """
    return f"{scene}_pn_{index:05d}"


def write_episode(rec, shard, *, image_root, scene, scene_id, ep_id, episode_index,
                  image_ext, recording) -> dict:
    """Persist one accepted episode's frames and metadata record."""
    rgb_paths = save_episode_frames(image_root, scene, sanitize(ep_id), rec.frames,
                                    image_ext)
    record = {
        "episode_id": ep_id,
        "episode_index": episode_index,
        "scene_id": scene_id,
        # Present and empty rather than absent: every existing reader does
        # `.get("goal_text", "")` and a PointNav episode has no object category. The
        # conversational formatter picks its prompt from `recording.task`, not from this.
        "goal_text": "",
        "rgb_paths": rgb_paths,
        "pose2d": [[float(p[0]), float(p[1]), float(p[2])] for p in rec.poses],
        "goal2d": [[float(g[0]), float(g[1]), float(g[2])] for g in rec.goal2d],
        "segment_index": [int(s) for s in rec.segment_index],
        "segments": [s.as_dict() for s in rec.segments],
        "start_pose2d": [float(v) for v in rec.start_pose2d],
        "sampling": rec.sampling,
        "recording": recording.as_dict(),
        QUALITY_KEY: rec.health.as_dict() if rec.health is not None else {},
    }
    append_metadata_record(shard, record)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    scenes = p.add_argument_group("scenes (dataset-agnostic; pick any one way in)")
    scenes.add_argument("--scene-root", default=None,
                        help="root the scene ids/globs are relative to")
    scenes.add_argument("--scenes", action="append", default=None,
                        help="explicit scene ids or paths, comma-separated or repeated")
    scenes.add_argument("--scene-glob", default=None,
                        help="glob under --scene-root, e.g. 'mp3d/*/*.glb'")
    scenes.add_argument("--scene-dataset-config", default=None,
                        help="a habitat scene-dataset config to enumerate scenes from")
    scenes.add_argument("--max-scenes", type=int, default=None)

    out = p.add_argument_group("output")
    out.add_argument("--output-dir", required=True, type=Path)
    out.add_argument("--image-root", default=None, type=Path,
                     help="default <output-dir>/images")
    out.add_argument("--log-file", default=None, type=Path)
    out.add_argument("--image-ext", default="png", choices=("png", "jpg", "jpeg"))
    out.add_argument("--adopt-existing-corpus", action="store_true",
                     help="write a corpus marker into a directory that already has "
                          "shards but no marker")

    ep = p.add_argument_group("episodes")
    ep.add_argument("--episodes-per-scene", type=int, default=200)
    ep.add_argument("--max-episodes", type=int, default=None,
                    help="stop after this many are written, across all scenes")
    ep.add_argument("--corpus-seed", type=int, default=0)
    ep.add_argument("--frames-per-episode", type=int, default=400)
    ep.add_argument("--min-frames", type=int, default=None,
                    help="only with --no-exact-frames; defaults to 75%% of the target")
    ep.add_argument("--frame-tolerance", type=int, default=25,
                    help="how far short of the target an episode may land and be kept")
    ep.add_argument("--min-segment-frames", type=int, default=50)
    ep.add_argument("--no-exact-frames", action="store_true",
                    help="keep whole segments instead of cutting the tail to the target")

    render = p.add_argument_group("rendering")
    render.add_argument("--fps", type=int, default=25,
                        help="25 Hz is the only integer rate on habitat's physics grid")
    render.add_argument("--width", type=int, default=640)
    render.add_argument("--height", type=int, default=480)
    render.add_argument("--panoramic", type=int, default=1,
                        help="N cameras concatenated into one strip per tick")
    render.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))

    sample = p.add_argument_group("sampling and screening")
    sample.add_argument("--min-island-area", type=float, default=8.0)
    sample.add_argument("--min-island-radius", type=float, default=2.0)
    sample.add_argument("--min-geodesic", type=float, default=1.5)
    sample.add_argument("--max-geodesic", type=float, default=8.0)
    sample.add_argument("--min-euclid", type=float, default=1.0)
    sample.add_argument("--max-height-delta", type=float, default=0.5)
    sample.add_argument("--min-turn-deg", type=float, default=0.0,
                        help="off by default: gating on it deletes straight corridors, "
                             "which are real. The spawn heading is randomised instead")
    sample.add_argument("--segment-retries", type=int, default=30)
    sample.add_argument("--tail-speed-mps", type=float, default=0.8,
                        help="assumed rate when sizing the last goal; a bias, not a gate")

    gate = p.add_argument_group("quality gates")
    gate.add_argument("--goal-threshold", type=float, default=0.5,
                      help="how close a segment must finish to count as reached")
    gate.add_argument("--min-segment-spl", type=float, default=0.55)
    gate.add_argument("--segment-rollbacks", type=int, default=3,
                      help="drive-time segment failures one episode may absorb")
    gate.add_argument("--time-backstop-s", type=float, default=300.0,
                      help="mandatory: a rollback loop spends sim time and no frames")
    gate.add_argument("--fall-speed-mps", type=float, default=2.0)
    gate.add_argument("--fall-duration-s", type=float, default=1.0)
    gate.add_argument("--ref-miss-flag-m", type=float, default=1.0)
    gate.add_argument("--reject-ref-miss-s", type=float, default=1.5,
                      help="reject a segment off its own planned path this long. ON here, "
                           "unlike ObjectNav, because the reference is a path we chose")

    p.add_argument("--verbose", action="store_true", help="print every segment")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return generate(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
