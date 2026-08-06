import argparse
import json
from typing import Any, Dict, Optional

import imageio.v2 as imageio
import numpy as np

from continuous_demos.drive_failure import motion_composition, near_stop_runs
from continuous_demos.panoramic import describe_rig
from continuous_demos.replay import (
    MOVE_STALL_TICKS,
    PHASE_FREE_STALL_TICKS,
    record_segmented_demo,
    record_path,
)
from continuous_demos.controllers import HolonomicRobotController
from continuous_demos.rotation_canonicalize import maybe_canonicalize
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    demo_looking_slices,
    extract_demo,
    find_scene_dataset_config,
    load_demo_objectnav,
    resolve_scene_path,
)


def extract_path_3d(
    episode: Dict[str, Any], max_steps: Optional[int] = None
) -> np.ndarray:
    replay = episode.get("reference_replay", [])
    if len(replay) <2:
        raise ValueError("Selected episode has no reference_replay data.")
    # Skip entry 0: it is the pure initial state s_0 (its action is a placeholder
    # STOP). In every other entry, `action` is the action that PRODUCED that
    # entry's state, not the action taken from it. See REPLAY_DATA_FORMAT.md.
    replay = replay[1:]
    if max_steps is not None:
        replay = replay[:max_steps]
    
    path = np.array(
        [step["agent_state"]["position"] for step in replay], dtype=np.float32
    )
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("reference_replay positions are not a valid Nx3 path")

    if len(path) == 1:
        path = np.concatenate([path, path], axis=0)

    return path


def resolve_episode(episodes: list[Dict[str, Any]], episode_index: int, episode_id: Optional[str]) -> Dict[str, Any]:
    if episode_id is not None:
        for episode in episodes:
            if episode["episode_id"] == episode_id:
                return episode
        raise ValueError(f"Episode id '{episode_id}' not found.")

    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode_index {episode_index} out of range for {len(episodes)} episodes.")

    return episodes[episode_index]


def trajectory_summary(poses, fps: float) -> Dict[str, Any]:
    """What the replayed trajectory cost, in the quantities the rotation
    problem is stated in.

    ``gross_rotation`` is total absolute yaw change (so a scan there-and-back
    counts twice, unlike net rotation), and ``rad_per_m`` is that divided by
    distance travelled -- the ratio the phase-free option exists to reduce.
    ``frac_turn`` is the share of ticks spent turning in place. The near-stop
    figures are reported because a phase-free replay is expected to produce
    some, wherever the demonstrated path doubles back on itself.
    """
    p = np.asarray(poses, dtype=np.float64)
    composition = motion_composition(p, fps)
    travelled = composition["path_len_m"]
    gross = float(np.abs(np.diff(np.unwrap(p[:, 2]))).sum()) if len(p) > 1 else 0.0
    stops = near_stop_runs(p, fps, min_duration_s=2.0 / fps)
    return {
        "ticks": int(len(p)),
        "duration_s": round(len(p) / fps, 2),
        "path_len_m": round(travelled, 2),
        "gross_rotation_rad": round(gross, 2),
        "rad_per_m": round(gross / travelled, 3) if travelled > 0.05 else None,
        "frac_turn_in_place": round(composition["frac_turn"], 4),
        "frac_drive": round(composition["frac_drive"], 4),
        "near_stops": len(stops),
        "near_stop_total_s": round(float(sum(stops)), 2),
        "near_stop_median_s": round(float(np.median(stops)), 2) if stops else None,
        "near_stop_max_s": round(float(max(stops)), 2) if stops else None,
    }


def replay_episode(
    dataset_path: str,
    output_path: str,
    episode_index: int = 0,
    episode_id: Optional[str] = None,
    scene_root: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    max_steps: Optional[int] = None,
    urdf_path: str = str(DEFAULT_URDF_PATH),
    debug_draw: bool = False,
    canonicalize_rotation: bool = False,
    num_cameras: int = 1,
    phase_free: bool = False,
    timeout: int = 15,
    summary_path: Optional[str] = None,
) -> None:
    data = load_demo_objectnav(dataset_path)
    episode = resolve_episode(data["episodes"], episode_index, episode_id)
    scene_path = resolve_scene_path(episode["scene_id"], scene_root)
    scene_dataset_config = find_scene_dataset_config(scene_path, scene_root)
    # path_3d[m] and path_headings[m] are the demo position/heading at path point
    # m; actions[j] carries the robot from path point j to j+1 (see extract_demo).
    path_3d, path_headings, actions = extract_demo(episode, max_steps=max_steps)
    # OFF by default; see rotation_canonicalize and dump/rotation_blunders.
    path_3d, path_headings, actions, canon = maybe_canonicalize(
        path_3d, path_headings, actions, canonicalize_rotation
    )
    if canon is not None and canon.touched:
        print(
            f"Canonicalised {canon.runs_altered}/{canon.runs} rotation runs: "
            f"-{canon.actions_removed} actions, "
            f"-{canon.rotation_deg_removed:.0f} deg of {canon.rotation_deg_before:.0f}"
        )
    # "looking" slices live in action-index space, which maps directly onto path
    # segments: a slice [a, b) covers path points a..b (see record_segmented_demo).
    # With --no-phases there are none, and the demo replays as a single move
    # phase over the whole path (see episode_io.demo_looking_slices).
    looking_slices = demo_looking_slices(actions, phase_free=phase_free)
    robot_sim = build_robot_sim(
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        width=width,
        height=height,
        urdf_path=urdf_path,
        num_cameras=num_cameras,
    )

    try:
        robot_sim.reset(
            scene_id=str(scene_path),
            robot_translation=path_3d[0].copy(),
            robot_rotation=episode["start_rotation"].copy(),
            snap_to_navmesh=False,
            vertical_offset=None,
        )
        print(f"Goal category: {episode['object_category']}")
        print(f"Episode info:\n {episode['info']}")
        print(f"Camera rig: {describe_rig(num_cameras)}")
        if phase_free:
            print(
                f"Phase-free: replaying all {len(actions)} actions as ONE move "
                "phase over the full path; recorded headings and LOOK_* unused"
            )
        else:
            print(
                f"Found {len(looking_slices)} looking slice(s) among {len(actions)} "
                f"actions: {looking_slices}"
            )
        # Switch controllers per phase: pure-pursuit path following while moving,
        # in-place heading tracking while looking (scans/peeks).
        frames, poses = record_segmented_demo(
            robot_sim,
            path_3d,
            path_headings,
            looking_slices,
            timeout=timeout,
            fps=fps,
            debug_draw=debug_draw,
            # The move-stall rule bails out of a phase that has stopped making
            # progress. With one phase there is nothing to bail out TO, so it
            # must be off -- see replay.PHASE_FREE_STALL_TICKS for what it
            # costs either way.
            move_stall_ticks=PHASE_FREE_STALL_TICKS if phase_free else MOVE_STALL_TICKS,
        )
        # frames,_poses = record_path(robot_sim, HolonomicRobotController(robot_sim,pure_pursuit_config=None), path_3d,timeout=timeout, fps=fps, debug_draw=debug_draw)
        imageio.mimsave(output_path, frames, fps=fps)

        summary = {
            "episode_id": episode.get("episode_id"),
            "episode_index": episode_index,
            "goal_text": episode.get("object_category", ""),
            "phase_free": phase_free,
            "num_cameras": num_cameras,
            "fps": fps,
            "video": output_path,
            "demo_actions": len(actions),
            "looking_slices": len(looking_slices),
            **trajectory_summary(poses, fps),
        }
        print("SUMMARY " + json.dumps(summary))
        if summary_path is not None:
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
    finally:
        robot_sim.sim.close(destroy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Habitat human demonstration and save RGB video.")
    parser.add_argument("dataset_path", help="Path to the .json.gz dataset file.")
    parser.add_argument("--output", default="human_demo_replay.mp4", help="Output video path.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--episode-id", type=str, default=None, help="Episode id to replay instead of using --episode-index.")
    parser.add_argument(
        "--scene-root",
        default=None,
        help=(
            "Optional root directory used to resolve relative scene_id paths. "
            "If omitted, the script checks HABITAT_SCENE_ROOT, ./data/scene_datasets, "
            "and ../data/scene_datasets."
        ),
    )
    parser.add_argument("--width", type=int, default=640, help="Output render width.")
    parser.add_argument("--height", type=int, default=480, help="Output render height.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum number of replay steps to render.")
    parser.add_argument(
        "--urdf-path",
        default=str(DEFAULT_URDF_PATH),
        help="URDF to use for the robot replay.",
    )
    parser.add_argument("--debug-draw", action="store_true", help="Render path and axes overlays during replay.")
    parser.add_argument(
        "--panoramic", type=int, default=1, metavar="N",
        help="Number of RGB cameras evenly spaced around the base, sharing the "
             "one camera's intrinsics. Default 1 = the shipped single forward "
             "camera, unchanged. N > 1 renders all N views concatenated into a "
             "panorama strip (forward in the middle, view sweeping left to "
             "right). At the configured hfov of 79 degrees, N=5 covers the full "
             "360 with 7 degrees of neighbour overlap. See "
             "continuous_demos.panoramic.",
    )
    parser.add_argument(
        "--no-phases", dest="phase_free", action="store_true",
        help="Replay the demonstration as ONE continuous pure-pursuit "
             "trajectory over its full 2D path, instead of alternating move / "
             "turn-in-place / look phases. The demo's recorded rotation and its "
             "action properties are discarded -- LOOK_UP/LOOK_DOWN included -- "
             "and heading is slaved to the direction of travel. OFF by default. "
             "Independent of --panoramic.",
    )
    parser.add_argument(
        "--timeout", type=int, default=15,
        help="Episode-wide budget in seconds of replayed time. Phase-free "
             "replays of long demonstrations need more than the default.",
    )
    parser.add_argument(
        "--summary-json", default=None,
        help="Also write the printed SUMMARY line to this path as JSON.",
    )
    parser.add_argument(
        "--no-canonicalize-rotation", dest="canonicalize_rotation",
        action="store_false", default=True,
        help="Disable rotation canonicalisation, which is ON by default so that "
             "what you debug here matches what the generator produces. "
             "Canonicalisation replaces every maximal run of pure "
             "TURN_LEFT/TURN_RIGHT with the shortest run covering the same "
             "heading range and ending on the same heading, before replay. See "
             "continuous_demos.rotation_canonicalize.",
    )
    args = parser.parse_args()

    replay_episode(
        dataset_path=args.dataset_path,
        output_path=args.output,
        episode_index=args.episode_index,
        episode_id=args.episode_id,
        scene_root=args.scene_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_steps=args.max_steps,
        urdf_path=args.urdf_path,
        debug_draw=args.debug_draw,
        canonicalize_rotation=args.canonicalize_rotation,
        num_cameras=args.panoramic,
        phase_free=args.phase_free,
        timeout=args.timeout,
        summary_path=args.summary_json,
    )


if __name__ == "__main__":
    main()