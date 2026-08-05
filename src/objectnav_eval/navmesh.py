"""Scene resolution, navmesh access, and reading the robot's position.

Three things live here because getting any of them wrong silently invalidates
every metric downstream:

1. :func:`robot_position_3d` -- the robot's position is **not**
   ``robot.translation``.
2. :func:`dataset_pathfinder` -- the shipped navmesh has a different filename
   suffix per dataset, and a naive ``str.replace`` no-ops on one of them.
3. :func:`resolve_scene` -- HM3D needs the episode's own scene-dataset config,
   which ``continuous_demos.episode_io.find_scene_dataset_config`` (MP3D-only
   filenames) will not find.

Requires ``habitat_sim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

import habitat_sim

from objectnav_eval.episodes import EvalEpisode

# The robot footprint build_robot_sim recomputes the navmesh for. Recorded so
# results can say which mesh they were measured on.
ROBOT_NAVMESH_RADIUS = 0.28
ROBOT_NAVMESH_HEIGHT = 1.0


#: Valid values for ``EvalConfig.navmesh``. See :func:`apply_navmesh_choice`.
NAVMESH_CHOICES = ("robot", "dataset")


def apply_navmesh_choice(robot_sim, scene_path, choice: str) -> str:
    """Choose which navmesh the *metrics* are measured on. Returns a description.

    This is the single most consequential switch in the harness, because it
    changes what every distance means, and it is easy to forget which one a
    number was produced on. It does **not** change how the robot moves: the
    robot is a physics body driven by the controller, never constrained to a
    navmesh. It changes geodesic distances, SPL's reference path, and screening.

    ``"robot"`` (the default) recomputes the mesh for the robot's own footprint
    (r=0.28 m, h=1.0 m). It is the honest answer to "where can *this* robot
    go", and it is strictly smaller and more fragmented than the mesh the
    episodes were cut on. That fragmentation has two costs:

    * It screens out episodes whose goal is on another island -- 28% of HM3D
      `val` and 12% of MP3D `val`.
    * Where it routes around something the physics body can simply drive
      through, the geodesic overstates the true distance. On MP3D
      ``8194nk5LbLH:39`` it said 9.15 m to a goal 2.20 m away in a straight
      line; the robot arrived after 1.9 m of travel, the distance trace fell
      9.99 -> 0.02 in three policy steps, and SPL pinned at **1.000** because
      its reference path was longer than the route actually taken.

    ``"dataset"`` uses the navmesh shipped with the scene -- what habitat and
    every published ObjectNav number use, cut for a 0.1 m point agent. It is
    connected and unfragmented, so the absurd jumps and the island screening
    both go away, and distances become comparable with the literature.

    **The trade is real and runs the other way**: the point-agent mesh assumes a
    robot that fits through gaps this one does not, so it *understates*
    difficulty. A goal it calls reachable may be unreachable in fact, and those
    episodes become honest failures rather than screened-out ones.

    Neither choice is correct in general; they answer different questions. What
    matters is that a run says which it used, so ``navmesh`` is written into
    every results file and printed in the run header.
    """
    if choice not in NAVMESH_CHOICES:
        raise ValueError(f"navmesh must be one of {NAVMESH_CHOICES}, got {choice!r}")
    if choice == "robot":
        return f"robot (recomputed, r={ROBOT_NAVMESH_RADIUS} m h={ROBOT_NAVMESH_HEIGHT} m)"

    pathfinder = dataset_pathfinder(Path(scene_path))
    if pathfinder is None:
        raise FileNotFoundError(
            f"--navmesh dataset needs the navmesh shipped with {Path(scene_path).name}, "
            "and none was found next to it. Fail loudly rather than silently "
            "measuring on the robot mesh, which is what the caller did not ask for."
        )
    navmesh_file = _dataset_navmesh_path(Path(scene_path))
    robot_sim.sim.pathfinder.load_nav_mesh(str(navmesh_file))
    if not robot_sim.sim.pathfinder.is_loaded:
        raise RuntimeError(f"failed to load {navmesh_file} into the simulator")
    # HabitatRobotSim.reset(scene_id=...) re-runs navmesh_config_and_recompute()
    # whenever this flag is set, which would silently put the robot mesh back on
    # the next scene switch. Clearing it means a switch falls back to the
    # scene's stock navmesh -- which is exactly this one.
    robot_sim._navmesh_recomputed = False
    return f"dataset (shipped point-agent mesh, {navmesh_file.name})"


def _dataset_navmesh_path(scene_path: Path) -> Optional[Path]:
    """The shipped navmesh file for a scene, or None. See dataset_pathfinder."""
    candidates = []
    hm3d = Path(str(scene_path).replace(".basis.glb", ".basis.navmesh"))
    if hm3d != scene_path:
        candidates.append(hm3d)
    candidates.append(scene_path.with_suffix(".navmesh"))
    for nav in candidates:
        if nav.suffix == ".navmesh" and nav.exists():
            return nav
    return None


def robot_position_3d(robot_sim) -> np.ndarray:
    """The robot's habitat-frame ``[x, y, z]`` position.

    NOT ``robot.translation``: the base link is fixed at the spawn point and the
    robot moves through its world prismatic joints, so ``translation`` never
    changes during an episode. The live planar position comes from
    ``get_2d_pose()`` -- ``[X, Y, theta]`` with ``X = habitat_x`` and
    ``Y = -habitat_z`` -- and the vertical is the base translation plus the
    passive ``joint_z``. Reading ``translation`` directly pins every goal
    distance to the spawn, which looks like a policy that never makes progress.
    """
    x, y_planar, _theta = robot_sim.get_2d_pose()
    return np.array([x, robot_height(robot_sim), -y_planar], dtype=np.float32)


def robot_height(robot_sim) -> float:
    """Habitat-world ``y`` of the robot base, tracking vertical motion."""
    height = float(robot_sim.robot.translation[1])
    try:
        height += float(robot_sim.get_joint_state("joint_z"))
    except (ValueError, KeyError):
        pass  # passive joint without a motor; the base height is close enough
    return height


def urdf_vertical_offset(robot_sim) -> float:
    """The lift ``HabitatRobotSim.reset`` adds so the model rests on the floor.

    ``robot.translation[1]`` is the navmesh-level spawn height *plus* this, so
    subtracting it recovers the height goals and view points are expressed at.
    """
    aabb = robot_sim.robot.root_scene_node.cumulative_bb
    return float(-aabb.min.y) if aabb.min.y < 0 else 0.0


def navmesh_height(robot_sim) -> float:
    """Navmesh-level habitat ``y`` of the robot, live.

    ``translation[1] - vertical_offset`` is the spawn height and never changes;
    ``joint_z`` is the prismatic vertical travel since then. Their sum is the
    height a geodesic query should be issued at.
    """
    return robot_height(robot_sim) - urdf_vertical_offset(robot_sim)


# ---------------------------------------------------------------------------
# Scene + navmesh resolution
# ---------------------------------------------------------------------------


def resolve_scene(
    episode: EvalEpisode, scene_root: Optional[str]
) -> Tuple[Path, Optional[str]]:
    """Absolute ``.glb`` path and the scene-dataset config to load it with.

    ``continuous_demos.find_scene_dataset_config`` only knows the MP3D config
    filenames, so the episode's own ``scene_dataset_config`` field is preferred
    -- for HM3D that is ``hm3d_annotated_basis.scene_dataset_config.json``,
    without which the semantic annotations and the basis-compressed meshes do
    not resolve.
    """
    from continuous_demos.episode_io import (
        find_scene_dataset_config,
        resolve_scene_path,
    )

    scene_path = resolve_scene_path(episode.scene_id, scene_root)
    cfg = episode.scene_dataset_config
    if cfg:
        candidate = Path(str(cfg).lstrip("./"))
        roots = [Path.cwd()]
        if scene_root:
            roots = [Path(scene_root).parent.parent, Path(scene_root)] + roots
        relatives = [candidate]
        if len(candidate.parts) > 2:
            # './data/scene_datasets/hm3d/x.json' also addressed relative to the
            # scene root itself, which is what 'data/scene_datasets' names.
            relatives.append(Path(*candidate.parts[2:]))
        for base in roots:
            for rel in relatives:
                resolved = base / rel
                if resolved.exists():
                    return scene_path, str(resolved)
    return scene_path, find_scene_dataset_config(scene_path, scene_root)


def dataset_pathfinder(scene_path: Path):
    """The navmesh that ships with the scene -- the one episodes were cut on.

    Kept only as a reference distance: it is built for a 0.1 m point agent and
    so overstates what a 0.28 m robot can reach.

    HM3D is ``<scene>.basis.glb`` -> ``<scene>.basis.navmesh``; MP3D is
    ``<scene>.glb`` -> ``<scene>.navmesh``. Testing the HM3D form first only
    works if a non-match is rejected: ``str.replace`` returns the path unchanged
    for MP3D, and that path exists -- it is the ``.glb`` -- so an existence
    check alone would hand the mesh loader a scene file. Hence the explicit
    suffix test.
    """
    scene_path = Path(scene_path)
    candidates = []
    hm3d = Path(str(scene_path).replace(".basis.glb", ".basis.navmesh"))
    if hm3d != scene_path:
        candidates.append(hm3d)
    candidates.append(scene_path.with_suffix(".navmesh"))
    for nav in candidates:
        if nav.suffix != ".navmesh" or not nav.exists():
            continue
        pathfinder = habitat_sim.nav.PathFinder()
        pathfinder.load_nav_mesh(str(nav))
        if pathfinder.is_loaded:
            return pathfinder
    return None


def navmesh_stats(pathfinder) -> dict:
    if pathfinder is None or not pathfinder.is_loaded:
        return {}
    return {
        "navigable_area_m2": float(pathfinder.navigable_area),
        "num_islands": int(pathfinder.num_islands),
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def geodesic(pathfinder, start: Sequence[float], ends: Any) -> float:
    """Shortest navmesh distance from ``start`` to the nearest of ``ends``.

    ``inf`` when no path exists (different islands), ``nan`` when the query is
    not well posed (no pathfinder, or an empty goal set).
    """
    ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
    if pathfinder is None or len(ends) == 0:
        return float("nan")
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_ends = ends
    pathfinder.find_path(path)
    return float(path.geodesic_distance)


def snap_points(pathfinder, points: Any) -> np.ndarray:
    """Project points onto the navmesh, dropping the ones that fall off it."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pathfinder is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    snapped = np.asarray([pathfinder.snap_point(p) for p in points], dtype=np.float32)
    return snapped[~np.isnan(snapped[:, 0])]


class GoalTracker:
    """Distance from a 3D position to the nearest goal view point.

    Geodesic on the given navmesh (so it respects walls), with the queried
    position snapped onto the mesh first. The euclidean distance is carried
    alongside because the robot is a physics body that can end up slightly off
    the navmesh, in which case the snap -- and therefore the geodesic -- can
    jump; a large gap between the two is the signal that this happened.

    This is the *screening* view of distance-to-goal. The metric that gets
    reported comes from the task layer's ``DistanceToGoal``; the two are
    compared per episode (see :mod:`objectnav_eval.reporting`).
    """

    def __init__(self, pathfinder, view_points: Any):
        self.pathfinder = pathfinder
        self.goals = snap_points(pathfinder, view_points)
        self.raw_goals = np.asarray(view_points, dtype=np.float32).reshape(-1, 3)

    def from_position(self, position_3d: Any) -> Tuple[float, float]:
        position_3d = np.asarray(position_3d, dtype=np.float32)
        if len(self.goals) == 0:
            return float("nan"), float("nan")
        euclidean = float(
            np.linalg.norm(self.raw_goals - position_3d[None, :], axis=1).min()
        )
        snapped = self.pathfinder.snap_point(position_3d)
        if np.isnan(snapped[0]):
            return float("inf"), euclidean
        return geodesic(self.pathfinder, snapped, self.goals), euclidean


import magnum as _mn  # noqa: E402  (only needed by camera_pose)


def camera_pose(robot_sim, sensor_uuid: str = "color_sensor"):
    """The camera's **habitat world** pose: ``[x, y, z, qw, qx, qy, qz]``.

    Distinct from everything else recorded per tick, and none of those substitute
    for it:

      * ``get_2d_pose()`` is the planar *control* frame (``X = habitat_x``,
        ``Y = -habitat_z``), and is the base, not the camera.
      * ``joint_z`` is a joint value, not a world coordinate.
      * ``robot.translation`` is pinned at the spawn (see :func:`robot_position_3d`).

    The sensor is reparented onto a robot link at setup (``sim.py``
    ``_setup_robot_and_sensors``) and carries its own fixed rotation, so its world
    transform is the link's chain times that offset -- which is what
    ``absolute_transformation()`` returns. This is the pose the rendered image was
    actually taken from, including the ``LOOK_UP``/``LOOK_DOWN`` camera pitch that
    never appears in the base pose at all.

    Returns ``None`` rather than raising if the sensor cannot be found: a missing
    diagnostic must never take down a rollout.
    """
    try:
        agent = robot_sim.sim.get_agent(robot_sim.agent_id)
        node = agent.scene_node.subtree_sensors.get(sensor_uuid)
        if node is None:
            return None
        transform = node.object.absolute_transformation()
        translation = transform.translation
        quaternion = _mn.Quaternion.from_matrix(transform.rotation())
        vector = quaternion.vector
        return [
            float(translation.x), float(translation.y), float(translation.z),
            float(quaternion.scalar), float(vector.x), float(vector.y), float(vector.z),
        ]
    except Exception:  # noqa: BLE001 - a diagnostic must not break a run
        return None
