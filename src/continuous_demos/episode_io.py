"""Dataset/episode IO and scene-resolution helpers shared by the pipeline
scripts (``generate_pose2d_dataset.py``, ``eval_pid_tracking.py``, and
``replay_human_demo.py``). Split out into the installable package because a
script should never be imported by another script -- these were previously
defined in ``replay_human_demo.py`` and imported sibling-style from it.
"""

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings

from continuous_demos import ASSETS_DIR
from continuous_demos.sim import HabitatRobotSim
from continuous_demos.replay import hab_quat_to_heading


DEFAULT_URDF_PATH = (
    ASSETS_DIR / "tidybot_description" / "urdf" / "tidybot_base_stabilized.urdf"
)

# Legacy dev-convenience fallback used by iter_scene_roots() below. Before the
# src-layout move this module's code lived in scripts/replay_human_demo.py,
# one level below the repo root, so PACKAGE_DIR.parent reached the repo root.
# This file is two levels below the repo root (src/continuous_demos/), so
# parents[1] is used to keep pointing at that same conceptual location.
PACKAGE_DIR = Path(__file__).resolve().parent


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
            PACKAGE_DIR.parents[1] / "data" / "scene_datasets",
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

# Actions that are neither locomotion (MOVE_FORWARD) nor termination (STOP);
# these are the in-place re-orientations that make up "looking" behavior.
_LOCOMOTION_ACTIONS = {"MOVE_FORWARD", "STOP"}

def find_looking_slices(
    actions: list[str],
    min_run: int = 3,
    peek_reversals: bool = True,
) -> list[tuple[int, int]]:
    """Identify non-overlapping ``[start, end)`` action-index slices of "looking".

    A candidate slice is a *maximal* run of non-locomotion actions -- anything
    that is not ``MOVE_FORWARD`` or ``STOP``, i.e. ``TURN_LEFT``, ``TURN_RIGHT``,
    ``LOOK_UP``, ``LOOK_DOWN``. A run is returned if it is at least ``min_run``
    actions long, or (when ``peek_reversals`` is set) if it is exactly 2 actions
    forming a left/right reversal. Returned slices are non-overlapping and in
    ascending order.

    The defaults are the shipped rule: a **panoramic scan** of ``>= 3`` actions,
    or a **peek** of exactly 2 that reverses direction. Because both patterns are
    derived from *maximal* runs a scan always takes precedence over a peek
    (``[L, R, R, R]`` is one 4-action scan, never a peek plus leftovers), and
    that holds for any ``min_run``.

    Runs below the threshold are not dropped from the replay; they are swallowed
    by the surrounding move phase, where they become zero-arc-length corners that
    pure pursuit rounds off -- measured at 0.14 control ticks per 30-degree turn,
    i.e. not executed as rotation at all. Whether that is a defect depends
    entirely on what those runs *are*, and the answer is: mostly not looking.
    ``dump/rotation_segmentation/FINDINGS.md`` classifies them with a
    course-deviation discriminator and finds 87.5% of sub-threshold runs are
    path-following re-aims -- the human turning because the corridor is not
    aligned to the 30-degree action grid. Executing those in place would shatter
    a fast, correct DWPP glide into a stutter of in-place rotations. Lowering
    ``min_run`` was measured and **rejected**; see that document before touching
    it. ``min_run`` is a parameter only so the arms of that A/B stay reproducible.
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
        if run_len >= min_run:
            slices.append((i, j))
        elif (
            peek_reversals
            and run_len == 2
            and {actions[i], actions[i + 1]} == {"TURN_LEFT", "TURN_RIGHT"}
        ):
            slices.append((i, j))  # peek: left-then-right or right-then-left
        i = j
    return slices

def extract_demo(
    episode: Any, max_steps: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build mutually-aligned ``(path_3d, path_headings, actions)`` for a demo.

    ``episode`` may be a raw dict (``load_demo_objectnav``) or an episode object
    with a ``reference_replay`` attribute (``objectnav_dataset``).

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
    if isinstance(episode, dict):
        replay = episode.get("reference_replay", [])
    else:
        replay = getattr(episode, "reference_replay", None) or []
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
