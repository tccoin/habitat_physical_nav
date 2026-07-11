import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import imageio.v2 as imageio
import numpy as np

import habitat_sim

from sim import HabitatRobotSim, HolonomicRobotController, record_path
from habitat_sim.utils.settings import default_sim_settings


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_URDF_PATH = (
    PACKAGE_DIR / "tidybot_description" / "urdf" / "tidybot_base_minimal.urdf"
)


def load_demo_objectnav(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = json.load(file)

    goals_by_category = data["goals_by_category"]
    for episode in data["episodes"]:
        goals_key = f"{os.path.basename(episode['scene_id'])}_{episode['object_category']}"
        episode["resolved_goals"] = goals_by_category.get(goals_key, [])

    return data


def iter_scene_roots(scene_root: Optional[str]) -> Iterable[Path]:
    if scene_root is not None:
        yield Path(scene_root).expanduser()
        return

    env_scene_root = os.environ.get("HABITAT_SCENE_ROOT")
    candidate_roots = []
    if env_scene_root:
        candidate_roots.append(Path(env_scene_root).expanduser())

    candidate_roots.extend(
        [
            PACKAGE_DIR / "data" / "scene_datasets",
            PACKAGE_DIR.parent / "data" / "scene_datasets",
            Path.cwd() / "data" / "scene_datasets",
        ]
    )

    seen = set()
    for root in candidate_roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        yield root


def iter_scene_candidates(scene_id: str, scene_root: Path) -> Iterable[Path]:
    scene_rel = scene_id.lstrip("./")
    stripped_mp3d_rel = scene_rel[len("mp3d/") :] if scene_rel.startswith("mp3d/") else scene_rel

    candidate_roots = [scene_root]
    mp3d_example_root = scene_root / "mp3d_example"
    if mp3d_example_root.exists():
        candidate_roots.append(mp3d_example_root)

    relative_candidates = [scene_rel, stripped_mp3d_rel]

    seen = set()
    for root in candidate_roots:
        for rel in relative_candidates:
            candidate = root / rel
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def resolve_scene_path(scene_id: str, scene_root: Optional[str]) -> Path:
    scene_path = Path(scene_id)
    if scene_path.is_absolute() and scene_path.exists():
        return scene_path

    searched_roots = []
    for root in iter_scene_roots(scene_root):
        searched_roots.append(str(root))
        for candidate in iter_scene_candidates(scene_id, root):
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Could not resolve scene_id '{scene_id}'. Searched scene roots: {searched_roots}."
    )


def find_scene_dataset_config(
    scene_path: Path, scene_root: Optional[str]
) -> Optional[str]:
    parent_candidates = [scene_path.parent, scene_path.parent.parent]
    for root in iter_scene_roots(scene_root):
        parent_candidates.extend([root, root / "mp3d_example"])

    seen = set()
    for directory in parent_candidates:
        if directory in seen or not directory.exists():
            continue
        seen.add(directory)
        for name in ("mp3d.scene_dataset_config.json", "default.scene_dataset_config.json"):
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    return None


def build_robot_sim(
    scene_path: Path,
    scene_dataset_config: Optional[str],
    width: int,
    height: int,
    urdf_path: str,
) -> HabitatRobotSim:
    sim_settings = default_sim_settings.copy()
    sim_settings["scene"] = str(scene_path)
    sim_settings["window_width"] = width
    sim_settings["window_height"] = height
    sim_settings["default_agent"] = 0
    sim_settings["enable_physics"] = True

    if scene_dataset_config is not None:
        sim_settings["scene_dataset_config_file"] = scene_dataset_config

    color_sensor_spec = habitat_sim.sensor.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.sensor.SensorType.COLOR
    color_sensor_spec.resolution = [height, width]
    color_sensor_spec.hfov = 79
    sensor_setup = {"base_camera_link": [color_sensor_spec]}

    robot_sim = HabitatRobotSim(
        sim_settings=sim_settings,
        urdf_path=urdf_path,
        sensor_setup=sensor_setup,
        fixed_base=True,
    )
    robot_sim.navmesh_config_and_recompute()
    return robot_sim


def extract_path_3d(
    episode: Dict[str, Any], max_steps: Optional[int] = None
) -> np.ndarray:
    replay = episode.get("reference_replay", [])
    if len(replay) == 0:
        raise ValueError("Selected episode has no reference_replay data.")

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
    path_3d = extract_path_3d(episode, max_steps=max_steps)
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
            snap_to_navmesh=True,
            vertical_offset=None,
        )
        controller = HolonomicRobotController(robot_sim, pure_pursuit_config=None)
        print(f"Goal category: {episode['object_category']}")
        print(f"Episode info:\n {episode['info']}")
        timeout = 20
        frames, _poses = record_path(
            robot_sim,
            controller,
            path_3d,
            timeout=timeout,
            fps=fps,
            debug_draw=debug_draw,
        )
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