"""Render one generated PointNav episode as an MP4 with a top-down-map HUD.

This is the gate before any corpus gets generated. The RGB view alone cannot tell you
whether an episode is *good* -- a robot pushing through a wall and a robot walking down a
corridor look similar from inside -- so every frame is composited next to the navmesh
floor plan with the planned path, the executed trajectory, the current goal and the agent
arrow drawn on it.

Five things to look for, each of which fails silently otherwise:

  1. the map is not mirrored (a mirrored map still looks like a clean trajectory; every
     left turn is just a right turn). Pinned by ``tests/test_topdown.py`` as well.
  2. the magenta planned path lies on free space.
  3. the turn-in-place at each segment start lands on the path's first leg.
  4. the robot does not clip through walls the navmesh calls solid.
  5. the goal star steps exactly at segment boundaries, and the segment counter with it.

    python scripts/pointnav_debug_episode.py \
        --scene mp3d/17DRP5sb8fy/17DRP5sb8fy.glb \
        --scene-root /Projects/data/scene_datasets \
        --episode-index 0 --frames-per-episode 400 --output pointnav_debug.mp4

``--episodes N`` renders N consecutive episode indices into one file, back to back, which
is the fastest way to look at a sample. ``--print-segments`` writes the per-segment table
that the pilot's thresholds get re-cut from.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from continuous_demos import pointnav as pn
from continuous_demos.drive_failure import DriveFailureConfig
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    find_scene_dataset_config,
    resolve_scene_path,
)
from continuous_demos.pointnav_episode import record_pointnav_episode
from continuous_demos.robot_pose import navmesh_height
from continuous_demos.topdown import TopDownMap, TopDownState

# The streaming MP4 writer lives in objectnav_eval. A *script* may import both packages;
# `continuous_demos.topdown` deliberately may not (see its module docstring).
from continuous_demos.viewport_overlay import draw_goal_in_view, draw_path_in_view
from objectnav_eval.video import OverlayState, Recorder, annotate


def build_frame(rgb, panel):
    """RGB view beside the map panel, one shape for every frame.

    Same idiom as ``replay.py``'s BEV composite: the panel is square and sized to the
    RGB's height, so an ``imageio`` writer sees a constant frame size.
    """
    return np.hstack([rgb, panel])


def render_viewport(tick, episode_index, frame_target, goal_radius_m):
    """The path drawn into the frame itself, plus the eval harness's caption bar.

    The path and the goal ring are projected through the camera the frame was rendered
    from (`viewport_overlay`), which is why the recorder hands over the camera matrix as
    it was at render time rather than letting this re-read it a physics step later.

    The caption reuses `objectnav_eval.video.annotate` -- it already draws a step counter
    and a distance colour-coded against its starting value, and a second implementation
    here would be a second thing to keep consistent.
    """
    goal = np.asarray(tick["goal2d"], dtype=float)[:2]
    pose = np.asarray(tick["pose2d"], dtype=float)[:2]
    rgb = tick["frame"]
    camera = tick.get("camera")
    if camera is not None:
        cam_to_world, (width, height, hfov) = camera
        rgb = draw_path_in_view(rgb, tick["path_3d"], cam_to_world, hfov)
        rgb = draw_goal_in_view(rgb, tick["goal_3d"], cam_to_world, hfov,
                                radius_m=goal_radius_m)
    return annotate(rgb, OverlayState(
        step=tick["frame_index"],
        sim_time=tick["sim_time"],
        goal=f"s{tick['segment_index']}",
        distance=float(np.linalg.norm(goal - pose)),
        start_distance=tick["geodesic_m"],
        extra=f"ep {episode_index} {tick['phase']} {tick['frame_index']}/{frame_target}",
    ))


def render_episode(robot_sim, recorder, args, episode_index, topdown_cache):
    """Drive one episode, compositing every tick into ``recorder``.

    Two layouts, and the difference is where the path is drawn rather than what is drawn:

      * ``topdown`` rasterises the navmesh floor plan and composites it beside the frame.
        It answers "is this trajectory inside the building", which the RGB alone cannot.
      * ``viewport`` draws the path and goal as 3D lines *in the scene*, so the frame
        stays the render's own size and nothing is rasterised or composited per tick.
        Cheaper and much smaller on disk, so it scales to a batch worth skimming.
    """
    executed = []
    past_goals = []
    state_box = {"map": None, "last_segment": -1}
    want_topdown = args.layout in ("topdown", "both")

    def on_tick(tick):
        pose = tick["pose2d"]
        executed.append([float(pose[0]), float(pose[1])])
        if tick["segment_index"] != state_box["last_segment"]:
            if state_box["last_segment"] >= 0 and state_box.get("last_goal") is not None:
                past_goals.append(state_box["last_goal"])
            state_box["last_segment"] = tick["segment_index"]
        state_box["last_goal"] = list(tick["goal2d"])

        if not want_topdown:
            recorder.add(render_viewport(tick, episode_index, args.frames_per_episode,
                                         args.goal_threshold))
            return
        rgb = tick["frame"]
        if args.layout == "both" and tick.get("camera") is not None:
            cam_to_world, (_w, _h, hfov) = tick["camera"]
            rgb = draw_path_in_view(rgb, tick["path_3d"], cam_to_world, hfov)
            rgb = draw_goal_in_view(rgb, tick["goal_3d"], cam_to_world, hfov,
                                    radius_m=args.goal_threshold)

        tdm = state_box["map"]
        if tdm is None:
            # Built at the robot's own navmesh height, on the first tick, so the slice is
            # the storey the episode is actually on.
            key = round(navmesh_height(robot_sim), 1)
            if key not in topdown_cache:
                topdown_cache[key] = TopDownMap.for_pathfinder(
                    robot_sim.sim.pathfinder, key,
                    meters_per_pixel=args.meters_per_pixel,
                )
            tdm = state_box["map"] = topdown_cache[key]

        panel = tdm.panel(
            TopDownState(
                pose2d=pose,
                goal2d=tick["goal2d"],
                planned_path=tick["planned_path"],
                executed=np.asarray(executed),
                past_goals=past_goals,
                segment_index=tick["segment_index"],
                frame_index=tick["frame_index"],
                frame_target=args.frames_per_episode or None,
                note=f"ep {episode_index}  phase={tick['phase']}",
            ),
            side_px=rgb.shape[0],
            # Upsample the raster before drawing, so the overlay lines are crisp rather
            # than a 0.05 m/px floor plan's worth of stair-stepping blown up to panel size.
            scale=args.map_scale,
        )
        recorder.add(build_frame(rgb, panel))

    budget = pn.FrameBudget(
        target_frames=args.frames_per_episode,
        # Derived from the target rather than fixed, so lowering --frames-per-episode
        # for a quick look does not trip the "no episode could ever be accepted" guard.
        min_frames=(args.min_frames if args.min_frames is not None
                    else int(args.frames_per_episode * 0.75)),
        exact=not args.no_exact_frames,
    )
    config = pn.PointNavSampleConfig(
        min_geodesic_m=args.min_geodesic,
        max_geodesic_m=args.max_geodesic,
        max_height_delta_m=args.max_height_delta,
        min_turn_deg=args.min_turn_deg,
        segment_retries=args.segment_retries,
    )
    return record_pointnav_episode(
        robot_sim,
        scene=Path(args.scene).stem,
        episode_index=episode_index,
        corpus_seed=args.corpus_seed,
        config=config,
        budget=budget,
        fps=args.fps,
        goal_threshold_m=args.goal_threshold,
        min_segment_spl=args.min_segment_spl,
        failure_config=DriveFailureConfig(
            fall_speed_mps=args.fall_speed_mps,
            fall_duration_s=args.fall_duration_s,
            ref_miss_flag_m=args.ref_miss_flag_m,
            reject_ref_miss_s=args.reject_ref_miss_s,
        ),
        rollback_limit=args.segment_rollbacks,
        sim_time_limit_s=args.time_backstop_s,
        on_tick=on_tick,
        viewport_debug=args.layout in ("viewport", "both"),
        show_navmesh=args.show_navmesh,
        verbose=True,
    )


def print_segments(rec, episode_index):
    print(f"  episode {episode_index}: {rec.num_frames} frames, "
          f"{'OK' if rec.ok else 'REJECTED (' + str(rec.reason) + ': ' + rec.detail + ')'}")
    if not rec.segments:
        return
    print("    idx  frames  geodesic  travelled     spl   turn   final  outcome")
    for s in rec.segments:
        print(f"    {s.index:3d}  {s.num_frames:6d}  {s.geodesic_m:8.2f}  "
              f"{s.travelled_m:9.2f}  {s.spl():6.3f}  "
              f"{np.degrees(s.turn_rad):+5.0f}  {s.final_dist_m:6.2f}  "
              f"{s.outcome}" + (f" ({s.reason})" if s.reason else ""))
    summary = pn.segments_summary(rec.segments)
    print(f"    -> {json.dumps(summary)}")
    rejections = {}
    for s in rec.segments:
        for code, n in s.sampling_rejections.items():
            rejections[code] = rejections.get(code, 0) + n
    if rejections:
        print(f"    sampling rejections: {json.dumps(rejections, sort_keys=True)}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene", required=True,
                   help="scene id or .glb path, e.g. mp3d/17DRP5sb8fy/17DRP5sb8fy.glb")
    p.add_argument("--scene-root", default=None)
    p.add_argument("--output", default="pointnav_debug.mp4")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1,
                   help="render this many consecutive episode indices into one file")
    p.add_argument("--corpus-seed", type=int, default=0)

    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--panoramic", type=int, default=1)
    p.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))
    p.add_argument("--layout", choices=("topdown", "viewport", "both"), default="topdown",
                   help="topdown: RGB beside a rasterised navmesh floor plan -- answers "
                        "'is this inside the building'. viewport: the path and goal drawn "
                        "as 3D lines in the scene, so the frame is a plain RGB and nothing "
                        "is composited -- cheaper and smaller, for skimming a batch. "
                        "both: viewport lines AND the panel")
    p.add_argument("--show-navmesh", action="store_true",
                   help="burn habitat's navmesh wireframe into the render. Shows what the "
                        "planner thinks is traversable next to what the robot manages, "
                        "which is the question behind every not_reached rejection")
    p.add_argument("--meters-per-pixel", type=float, default=0.05)
    p.add_argument("--map-scale", type=int, default=3,
                   help="upsample the floor plan this many times before drawing on it")
    p.add_argument("--video-scale", type=float, default=1.0,
                   help="downscale the written MP4 by this factor")

    p.add_argument("--frames-per-episode", type=int, default=400)
    p.add_argument("--min-frames", type=int, default=None,
                   help="only used with --no-exact-frames; defaults to 75%% of the target")
    p.add_argument("--no-exact-frames", action="store_true",
                   help="keep whole segments instead of cutting the tail to the target")
    p.add_argument("--min-geodesic", type=float, default=1.5)
    p.add_argument("--max-geodesic", type=float, default=8.0)
    p.add_argument("--max-height-delta", type=float, default=0.5)
    p.add_argument("--min-turn-deg", type=float, default=0.0)
    p.add_argument("--segment-retries", type=int, default=30)
    p.add_argument("--segment-rollbacks", type=int, default=3)
    p.add_argument("--goal-threshold", type=float, default=0.5)
    p.add_argument("--min-segment-spl", type=float, default=0.55)
    p.add_argument("--time-backstop-s", type=float, default=300.0)

    p.add_argument("--fall-speed-mps", type=float, default=2.0)
    p.add_argument("--fall-duration-s", type=float, default=1.0)
    p.add_argument("--ref-miss-flag-m", type=float, default=1.0)
    p.add_argument("--reject-ref-miss-s", type=float, default=1.5,
                   help="reject a segment off its own planned path this long "
                        "(PointNav plans the reference itself, unlike ObjectNav)")
    p.add_argument("--print-segments", action="store_true", default=True)
    args = p.parse_args()

    scene_path = resolve_scene_path(args.scene, args.scene_root)
    scene_dataset_config = find_scene_dataset_config(scene_path, args.scene_root)
    print(f"scene: {scene_path}")
    robot_sim = build_robot_sim(
        scene_path=scene_path, scene_dataset_config=scene_dataset_config,
        width=args.width, height=args.height, urdf_path=args.urdf_path,
        num_cameras=args.panoramic,
    )
    pathfinder = robot_sim.sim.pathfinder
    print(f"robot navmesh: {pathfinder.navigable_area:.1f} m2 over "
          f"{pathfinder.num_islands} island(s); areas "
          f"{[round(pathfinder.island_area(i), 1) for i in range(pathfinder.num_islands)]}")

    topdown_cache = {}
    kept = 0
    try:
        with Recorder(args.output, fps=args.fps, scale=args.video_scale) as recorder:
            for offset in range(args.episodes):
                index = args.episode_index + offset
                rec = render_episode(robot_sim, recorder, args, index, topdown_cache)
                if args.print_segments:
                    print_segments(rec, index)
                kept += int(rec.ok)
    finally:
        robot_sim.sim.close(destroy=True)

    print(f"\nWrote {args.output}: {kept}/{args.episodes} episode(s) would be kept")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
