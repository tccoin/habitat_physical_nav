"""Replay-recording orchestration: drives :class:`~continuous_demos.sim.HabitatRobotSim`
and the controllers in :mod:`continuous_demos.controllers` together to record a
continuous demonstration, plus the small geometry helper (``hab_quat_to_heading``)
used to turn a demo's habitat quaternions into planar headings. Split out of
``sim.py``, which owns only the simulator itself. See ``sim.py``'s module
docstring for the three coordinate frames referenced throughout.
"""

import math
from typing import Optional

import numpy as np
from PIL import Image

from continuous_demos.sim import HabitatRobotSim
from continuous_demos.controllers import (
    HolonomicControllerConfig,
    HolonomicRobotController,
    HeadingRotationController,
)

def record_path(robot_sim: HabitatRobotSim, controller: HolonomicRobotController, path_3d, timeout=10, fps=30, debug_draw=True):
    frames = []
    poses = []
    controller.set_hab_path3(path_3d)
    robot_sim.sim.navmesh_visualization = debug_draw
    for _ in range(timeout * fps):
        if debug_draw:
            robot_sim.draw_axes(path_3d[0], axis_len=1)
            robot_sim.draw_axes(path_3d[-1], axis_len=1)
        obs =robot_sim.get_obs()
        step_args = controller.step(dt=1.0/fps, heading_alignment_gain=1.0,max_vw=1.5)
        if debug_draw:
            controller.draw_path()
        robot_sim.step(
            **step_args,
            dt=1.0 / fps
        )
        rgb_frame = obs["color_sensor"][..., :3]
        pose_2d = robot_sim.get_2d_pose()
        if debug_draw:
            bev_frame = controller.visualize_bev()
            bev_frame = np.array(
                Image.fromarray(bev_frame).resize((rgb_frame.shape[0], rgb_frame.shape[0]))
            )
            frame = np.hstack([rgb_frame, bev_frame])
        else:
            frame = rgb_frame
        frames.append(frame)
        poses.append(pose_2d)
        if controller.is_goal_reached() and robot_sim.is_stationary(lin_vel_threshold=0.01, ang_vel_threshold=0.01):
            print("Goal reached!")
            break
        if robot_sim.is_tipped_over(angle_threshold_degrees=45.0):
            print("Robot tipped over! Stopping simulation.")
            break
    return frames, poses

def hab_quat_to_heading(rotation_xyzw) -> float:
    """Convert a Habitat ``[x, y, z, w]`` agent rotation to the robot's 2D heading.

    The heading is returned in the planar control frame (frame 3 in the module
    header) -- the frame ``joint_th`` drives motion in and that
    ``HolonomicRobotController`` aligns to: axes ``X = habitat_x``,
    ``Y = -habitat_z`` (see ``get_2d_pose`` / ``set_hab_path3``). The agent's
    forward axis is habitat local ``-z`` (``habitat_sim.geo.FRONT``); we rotate it
    by the quaternion and take its heading in the control plane as
    ``atan2(-forward_z, forward_x)``. Verified to match the actual per-step
    displacement of every ``MOVE_FORWARD`` in the demos to 0.000 deg, so
    ``HeadingRotationController`` tracks these headings faithfully and a look phase
    leaves the robot facing the direction it must next walk.

    This is the same convention ``HabitatRobotSim.reset`` now uses to seed
    ``joint_th`` (both take habitat FRONT = -z as forward), so spawn heading and
    tracked headings agree.
    """
    x, y, z, w = (float(c) for c in rotation_xyzw)
    # World-frame forward direction = quaternion applied to habitat local -z.
    forward_x = -(2.0 * (x * z + w * y))
    forward_z = -(1.0 - 2.0 * (x * x + y * y))
    # Control/motion plane is (X = habitat_x, Y = -habitat_z).
    return float(math.atan2(-forward_z, forward_x))

def _build_phases(num_actions, looking_slices):
    """Tile the action timeline ``[0, num_actions)`` with ordered move/look phases.

    ``looking_slices`` are ``[start, end)`` half-open ranges in ACTION-index space
    (ascending, non-overlapping, as returned by
    ``replay_human_demo.find_looking_slices``). Every gap between them -- plus the
    head and tail -- becomes a ``move`` phase. Returns a list of
    ``(kind, start, end)`` tuples, ``kind in {'move', 'look'}``, that exactly
    covers ``[0, num_actions)`` in order.
    """
    phases = []
    cursor = 0
    for start, end in looking_slices:
        if start > cursor:
            phases.append(("move", cursor, start))
        phases.append(("look", start, end))
        cursor = end
    if cursor < num_actions:
        phases.append(("move", cursor, num_actions))
    return phases


def record_segmented_demo(
    robot_sim: HabitatRobotSim,
    path_3d,
    path_headings,
    looking_slices,
    timeout: int = 15,
    fps: int = 30,
    debug_draw: bool = False,
    pure_pursuit_config: Optional[HolonomicControllerConfig] = None,
):
    """Replay a discrete demonstration as a continuous trajectory, switching
    between a path-following controller (while moving) and an in-place rotation
    controller (while "looking": panoramic scans / peeks).

    ------------------------------------------------------------------------
    Index conventions and the action -> path-index conversion (read carefully)
    ------------------------------------------------------------------------
    The arrays are built by ``replay_human_demo.extract_demo`` so that they are
    all mutually aligned:

      * ``path_3d[m]``       -- world position at demo path point ``m``
      * ``path_headings[m]`` -- ``joint_th``-frame yaw at demo path point ``m``
        for ``m = 0 .. M``  (there are ``M + 1`` path points).
      * The action timeline has ``M = len(path_3d) - 1`` entries. Action ``j``
        is exactly the demo action that carries the robot FROM path point ``j``
        TO path point ``j + 1``. ``looking_slices`` are ``[start, end)`` ranges
        in this action space, ``0 <= start < end <= M``.

    Converting an action-index phase ``[p, q)`` to reference data:

      * MOVE phase -> follow the sub-path ``path_3d[p : q + 1]``. Actions
        ``p .. q-1`` span path segments ``(p->p+1), ..., (q-1->q)``, i.e. path
        points ``p`` through ``q`` inclusive; hence the ``q + 1`` slice end.
      * LOOK phase -> rotate in place through ``path_headings[p : q + 1]``. Turns
        and looks do not translate, so path points ``p .. q`` share one position;
        instead of a path we track the heading at the phase's anchor point ``p``
        followed by one heading per look action (points ``p+1 .. q``). The robot
        never moves, only re-orients.

    Because move phases start their sub-path at point ``p`` and the preceding look
    phase ended (position-wise) at that same point ``p``, and vice versa, phases
    chain together with no positional gap -- continuity is carried by the shared
    ``robot_sim`` state, which each controller reads live.
    """
    path_3d = np.asarray(path_3d, dtype=np.float32)
    path_headings = np.asarray(path_headings, dtype=float)
    num_actions = len(path_3d) - 1
    if num_actions < 1:
        return [], []

    phases = _build_phases(num_actions, looking_slices)

    # One controller instance of each type, reconfigured (via reset + set_*) per
    # phase; the robot's live pose provides continuity across the handoffs.
    move_controller = HolonomicRobotController(robot_sim, pure_pursuit_config)
    look_controller = HeadingRotationController(robot_sim)

    robot_sim.sim.navmesh_visualization = debug_draw
    frames = []
    poses = []

    for kind, p, q in phases:
        if kind == "move":
            move_controller.reset()
            move_controller.set_hab_path3(path_3d[p : q + 1])
            local_timeout = int(np.ceil(move_controller.controller._path_distances[-1] / 0.5 + 0.5))
            # print(local_timeout)
            # print(type(local_timeout))
            # print(f"Move phase: {p} -> {q}, path length: {move_controller.controller._path_distances[-1]:.2f}m, timeout: {local_timeout:.1f}s")
        else:
            look_controller.reset()
            look_controller.set_heading_trajectory(
                path_headings[p : q + 1], remove_redundant=True
            )
            local_timeout =  2 * (q - p + 1)  # allow more time for longer look phases
            # print(f"Look phase: {p} -> {q}, num headings: {len(look_controller.headings)}, timeout: {local_timeout:.1f}s")
        for _ in range(local_timeout * fps):
            obs = robot_sim.get_obs()
            if debug_draw:
                robot_sim.draw_axes(path_3d[0], axis_len=1)
                robot_sim.draw_axes(path_3d[-1], axis_len=1)

            if kind == "move":
                step_args = move_controller.step(
                    dt=1.0 / fps, heading_alignment_gain=3.0, max_vw=2.0
                )
                if debug_draw:
                    move_controller.draw_path()
            else:
                step_args = look_controller.step(dt=1.0 / fps)

            robot_sim.step(**step_args, dt=1.0 / fps)

            rgb_frame = obs["color_sensor"][..., :3]
            if debug_draw:
                if kind == "move":
                    panel = move_controller.visualize_bev()
                else:
                    # HeadingRotationController has no BEV; pad with a blank panel
                    # so every frame keeps the same shape for imageio.mimsave.
                    panel = np.zeros(
                        (rgb_frame.shape[0], rgb_frame.shape[0], 3), dtype=rgb_frame.dtype
                    )
                panel = np.array(
                    Image.fromarray(panel).resize((rgb_frame.shape[0], rgb_frame.shape[0]))
                )
                frame = np.hstack([rgb_frame, panel])
            else:
                frame = rgb_frame
            frames.append(frame)
            poses.append(robot_sim.get_2d_pose())

            if kind == "move":
                phase_done = move_controller.is_goal_reached() and robot_sim.is_stationary(
                    lin_vel_threshold=0.01, ang_vel_threshold=0.01
                )
            else:
                phase_done = look_controller.is_trajectory_complete() and robot_sim.is_stationary(
                    lin_vel_threshold=0.01, ang_vel_threshold=0.01
                )
            if phase_done:
                break
            if robot_sim.is_tipped_over(angle_threshold_degrees=45.0):
                # print("Robot tipped over! Stopping simulation.")
                return frames, poses
        if len(frames) >= timeout * fps:
            # print("Timeout reached! Stopping simulation.")
            break
    return frames, poses
