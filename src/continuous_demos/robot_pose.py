"""Where the robot actually is -- the four readings every caller gets wrong once.

``HabitatRobotSim`` builds the URDF with ``fixed_base=True`` and drives the machine
through world prismatic/revolute joints (``joint_x``, ``joint_y``, ``joint_th``) plus a
passive vertical ``joint_z``. Two consequences, and both of them read as plausible
numbers rather than as errors:

* **``robot.translation`` is the spawn, not the position.** The base link is pinned, so
  ``translation`` is identical at tick 0 and tick 4000 -- measured identical to 1e-9 for a
  robot driving across a scene and for one spawned 30 m outside it. Reading it pins every
  goal distance to the spawn, which looks exactly like a policy that never makes progress.
  The live planar position is ``get_2d_pose()``; the live vertical is
  ``translation[1] + joint_z``.

* **The height a *pathfinder* query wants is not the height the robot is at.**
  ``reset()`` lifts the model by ``-cumulative_bb.min.y`` so it rests on the floor rather
  than sinking into it, so ``robot_height_m`` sits that much above the navmesh.
  ``navmesh_height`` subtracts the lift back off. Issuing a geodesic or a ``snap_point``
  at the un-subtracted height is usually absorbed by the snap on a flat floor and, in a
  multi-storey scene, silently resolves to the storey above.

The distinction only started to matter when we began *sampling* points against that
height (PointNav) rather than reading them out of an episode file (ObjectNav), which is
why this module exists now and did not before.

This is the single definition. ``continuous_demos.drive_failure`` and
``objectnav_eval.navmesh`` re-export from here; they each carried their own copy, with
their own docstring saying so.
"""

from __future__ import annotations

import math

import numpy as np


def robot_height_m(robot_sim) -> float:
    """The robot's live habitat-frame ``y`` (metres), tracking vertical motion.

    ``translation[1]`` is the spawn height (see the module docstring) and ``joint_z`` is
    the passive prismatic travel since then. Their sum is where the base is now.

    A joint that cannot be read falls back to the base height rather than raising: this
    feeds the escape detector, and a failed query must never be able to delete an episode.
    """
    height = float(robot_sim.robot.translation[1])
    try:
        height += float(robot_sim.get_joint_state("joint_z"))
    except Exception:
        pass  # passive joint without a motor: the base height is what we have
    return height


def urdf_vertical_offset(robot_sim) -> float:
    """The lift ``HabitatRobotSim.reset`` adds so the model rests on the floor.

    ``robot.translation[1]`` is the navmesh-level spawn height *plus* this, so subtracting
    it recovers the height goals, view points and navmesh polygons are expressed at.
    """
    aabb = robot_sim.robot.root_scene_node.cumulative_bb
    return float(-aabb.min.y) if aabb.min.y < 0 else 0.0


def navmesh_height(robot_sim) -> float:
    """Navmesh-level habitat ``y`` of the robot, live.

    This -- not :func:`robot_height_m` -- is the height a geodesic, a ``snap_point`` or a
    ``get_topdown_view`` slice should be issued at.
    """
    return robot_height_m(robot_sim) - urdf_vertical_offset(robot_sim)


def robot_position_3d(robot_sim) -> np.ndarray:
    """The robot's live habitat-frame ``[x, y, z]``, at the *robot's* height.

    Planar components come from ``get_2d_pose()`` (``X = habitat_x``, ``Y = -habitat_z``);
    the vertical is :func:`robot_height_m`. Use this to place things relative to the
    machine -- a camera, a debug marker, a distance to a goal already expressed at robot
    height. For anything the pathfinder has to answer, use
    :func:`robot_navmesh_position`.
    """
    x, y_planar, _theta = robot_sim.get_2d_pose()
    return np.array([x, robot_height_m(robot_sim), -y_planar], dtype=np.float32)


def robot_navmesh_position(robot_sim) -> np.ndarray:
    """The robot's live habitat-frame ``[x, y, z]``, at *navmesh* height.

    The query point for ``pathfinder.find_path`` / ``snap_point`` / ``island_radius``.
    Differs from :func:`robot_position_3d` only in the ``y`` component, by the URDF lift
    -- which is roughly 0.25 m on the shipped TidyBot, small enough for a snap to absorb
    on a flat floor and large enough to resolve to the wrong storey in a stairwell.
    """
    x, y_planar, _theta = robot_sim.get_2d_pose()
    return np.array([x, navmesh_height(robot_sim), -y_planar], dtype=np.float32)


def planar_heading_to_quat_xyzw(theta: float) -> np.ndarray:
    """Planar heading (radians, CCW from +X) -> a habitat yaw quaternion ``[x, y, z, w]``.

    The inverse of :func:`continuous_demos.replay.hab_quat_to_heading`, and the format
    ``HabitatRobotSim.reset(robot_rotation=...)`` accepts. Needed by anything that
    *chooses* a spawn heading rather than reading one out of a demonstration.

    The half-turn offset is the whole content: habitat's forward is local ``-z`` and the
    planar frame measures theta from ``+X``, so a quaternion of ``theta`` about ``+y``
    points the robot 90 degrees away from where the caller meant. Writing it out here
    rather than reaching for ``objectnav.planar_heading_to_habitat_quat`` keeps this
    module free of the habitat-lab import that one carries;
    ``tests/test_robot_pose.py`` pins the two against each other via the existing
    inverse rather than trusting that they agree.
    """
    half = (float(theta) - math.pi / 2.0) / 2.0
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float32)


def planar_to_habitat(pose2d, height: float) -> np.ndarray:
    """Planar ``[X, Y, theta]`` (theta ignored) + a habitat ``y`` -> habitat ``[x, y, z]``.

    The inverse of the planar projection above, spelled out once so callers that hold a
    recorded pose rather than a live simulator do not each re-derive the sign on ``z``.
    """
    return np.array([float(pose2d[0]), float(height), -float(pose2d[1])], dtype=np.float32)
