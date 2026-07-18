import argparse
from typing import Any, Dict, Optional

import imageio.v2 as imageio
import numpy as np

from continuous_demos.replay import record_segmented_demo, record_path
from continuous_demos.controllers import HolonomicRobotController
from continuous_demos.episode_io import (
    DEFAULT_URDF_PATH,
    build_robot_sim,
    extract_demo,
    find_looking_slices,
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
) -> None:
    data = load_demo_objectnav(dataset_path)
    episode = resolve_episode(data["episodes"], episode_index, episode_id)
    scene_path = resolve_scene_path(episode["scene_id"], scene_root)
    scene_dataset_config = find_scene_dataset_config(scene_path, scene_root)
    # path_3d[m] and path_headings[m] are the demo position/heading at path point
    # m; actions[j] carries the robot from path point j to j+1 (see extract_demo).
    path_3d, path_headings, actions = extract_demo(episode, max_steps=max_steps)
    # "looking" slices live in action-index space, which maps directly onto path
    # segments: a slice [a, b) covers path points a..b (see record_segmented_demo).
    looking_slices = find_looking_slices(actions)
    robot_sim = build_robot_sim(
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        width=width,
        height=height,
        urdf_path=urdf_path,
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
        print(
            f"Found {len(looking_slices)} looking slice(s) among {len(actions)} "
            f"actions: {looking_slices}"
        )
        timeout = 15
        # Switch controllers per phase: pure-pursuit path following while moving,
        # in-place heading tracking while looking (scans/peeks).
        frames, _poses = record_segmented_demo(
            robot_sim,
            path_3d,
            path_headings,
            looking_slices,
            timeout=timeout,
            fps=fps,
            debug_draw=debug_draw,
        )
        # frames,_poses = record_path(robot_sim, HolonomicRobotController(robot_sim,pure_pursuit_config=None), path_3d,timeout=timeout, fps=fps, debug_draw=debug_draw)
        imageio.mimsave(output_path, frames, fps=fps)
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
    )


if __name__ == "__main__":
    main()