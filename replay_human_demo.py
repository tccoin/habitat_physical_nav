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

from sim import (
    HabitatRobotSim,
    record_segmented_demo, record_path, HolonomicRobotController,
    hab_quat_to_heading,
)
from habitat_sim.utils.settings import default_sim_settings


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_URDF_PATH = (
    PACKAGE_DIR / "tidybot_description" / "urdf" / "tidybot_base_stabilized.urdf"
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


# Actions that are neither locomotion (MOVE_FORWARD) nor termination (STOP);
# these are the in-place re-orientations that make up "looking" behavior.
_LOCOMOTION_ACTIONS = {"MOVE_FORWARD", "STOP"}


def find_looking_slices(actions: list[str]) -> list[tuple[int, int]]:
    """Identify non-overlapping ``[start, end)`` action-index slices of "looking".

    An index slice is "looking" if it is a maximal run of non-locomotion actions
    (anything that is not ``MOVE_FORWARD`` or ``STOP`` -- i.e. ``TURN_LEFT``,
    ``TURN_RIGHT``, ``LOOK_UP``, ``LOOK_DOWN``) that is either:

      * a **panoramic scan** -- a run of ``>= 3`` such actions; or
      * a **peek** -- a run of exactly 2 actions that is a left/right reversal
        (``TURN_LEFT`` then ``TURN_RIGHT`` or vice versa).

    Both patterns are derived from *maximal* runs, so a scan always takes
    precedence over a peek: e.g. ``[..., L, R, R, R, ...]`` yields a single
    4-action scan, never an ``L, R`` peek plus leftovers. Runs of length 1, and
    length-2 runs that are not L/R reversals (e.g. ``L, L`` or ``LOOK_UP,
    LOOK_DOWN``), are not looking behavior. Returned slices are non-overlapping
    and in ascending order.
    """
    slices: list[tuple[int, int]] = []
    n = len(actions)
    i = 0
    while i < n:
        if actions[i] in _LOCOMOTION_ACTIONS:
            i += 1
            continue
        # Extend a maximal run of non-locomotion (re-orientation) actions.
        j = i
        while j < n and actions[j] not in _LOCOMOTION_ACTIONS:
            j += 1
        run_len = j - i
        if run_len >= 3:
            slices.append((i, j))  # panoramic scan (takes precedence)
        elif run_len == 2 and {actions[i], actions[i + 1]} == {"TURN_LEFT", "TURN_RIGHT"}:
            slices.append((i, j))  # peek: left-then-right or right-then-left
        i = j
    return slices


def extract_demo(
    episode: Dict[str, Any], max_steps: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build mutually-aligned ``(path_3d, path_headings, actions)`` for a demo.

    Returns:
      * ``path_3d[m]``       -- world position at path point ``m`` (Nx3 float32)
      * ``path_headings[m]`` -- ``joint_th``-frame yaw at path point ``m``
      * ``actions[j]``       -- the ``j``-th realized action string

    with the crucial alignment that ``actions[j]`` is the action carrying the
    robot from ``path_3d[j]`` to ``path_3d[j + 1]``. Hence there are
    ``M = len(path_3d) - 1`` actions and ``M + 1`` path points.

    Why the indices line up this way (see REPLAY_DATA_FORMAT.md for the full
    story). Raw replay entries ``r[0 .. N-1]`` store ``(action, state)`` where a
    given entry's ``state`` is the one *produced* by that entry's ``action`` --
    except for two startup artifacts of the collection code:

      * ``r[0]`` is a ``STOP`` seed carrying the true initial state ``s_0``;
      * ``r[1]`` is a stale duplicate of ``s_0`` -- its action's effect is lost.

    So the causally-clean realized rollout is
    ``s_0 = r[1].state  --r[2].action-->  r[2].state  --r[3].action-->  ...``.
    We therefore index:

      * path/heading points by the STATE entries ``r[1 .. N-1]``
        (``path_3d[m] = r[1 + m].state``), and
      * actions by ``r[2 .. N-1]`` (``actions[j] = r[2 + j].action``).

    Then ``actions[j] = r[2 + j]`` maps the transition
    ``r[1 + j].state -> r[2 + j].state`` = ``path_3d[j] -> path_3d[j + 1]``, an
    off-by-one-free segment<->action correspondence. The first (lost) action and
    both ``STOP`` seeds are dropped; at most one demo transition is discarded.
    """
    replay = episode.get("reference_replay", [])
    if len(replay) < 2:
        raise ValueError("Selected episode has no reference_replay data.")

    state_entries = replay[1:]   # r[1 .. N-1] -> path points 0 .. M
    action_entries = replay[2:]  # r[2 .. N-1] -> actions   0 .. M-1
    if max_steps is not None:
        # Keep alignment: M actions require M + 1 path points.
        state_entries = state_entries[: max_steps + 1]
        action_entries = action_entries[:max_steps]

    path_3d = np.array(
        [e["agent_state"]["position"] for e in state_entries], dtype=np.float32
    )
    if path_3d.ndim != 2 or path_3d.shape[1] != 3:
        raise ValueError("reference_replay positions are not a valid Nx3 path")
    path_headings = np.array(
        [hab_quat_to_heading(e["agent_state"]["rotation"]) for e in state_entries],
        dtype=np.float32,
    )
    actions = [e["action"] for e in action_entries]
    return path_3d, path_headings, actions


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