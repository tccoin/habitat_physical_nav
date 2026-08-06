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
    HeadingRotationConfig,
    HeadingRotationController,
)

# ---------------------------------------------------------------------------
# Phase-transition defaults (see dump/data_diagnostics/rotation_tuning/FINDINGS.md).
#
# A move phase ends early if, over the last MOVE_STALL_TICKS control ticks, the
# robot advanced less than MOVE_STALL_EPS metres ALONG THE REFERENCE PATH *and*
# never exceeded MOVE_STALL_MAX_SPEED. A phase can be blocked by geometry, or by
# the path controller simply having nothing left to give. Without this check such
# phases burn their whole local timeout -- measured at 15-56% of an episode's
# ticks -- emitting frames of a completely motionless robot. Because the speed
# gate fires only when the base is genuinely not moving, bailing out changes
# nothing about where the robot goes; it only stops recording the frozen frames,
# and it frees episode-timeout budget for the phases that follow.
#
# Re-derived AFTER the DWPP constant-braking-distance fix
# (pure_pursuit._decide_accel_or_decel). Before that fix, every move phase with a
# reference path shorter than 0.67 m was commanded v = 0 for its whole life, so
# this rule was partly compensating for a controller that could not move. With the
# root cause gone it is a smaller, cleaner effect: over 192 held-out episodes at
# 25 Hz it fires on 21% of them, and on those it cuts duration to 0.70.
#
# The window barely matters and the SPEED GATE does all the work. Measured
# against the otherwise-identical arm with the rule switched off, so that the
# rule's own cost is isolated: 30 ticks and 45 ticks are indistinguishable on
# every metric (duration 0.669 vs 0.673 of baseline, both 0 goal-check
# regressions, both +1 recovery), while dropping the gate at 30 ticks causes 3
# regressions and loses 3.4 pp of path length. 45 is kept as the more
# conservative of two equals.
#
# Note this supersedes an earlier claim, made at 30 Hz before the DWPP fix, that
# the rule cost one goal-check regression in 80 and that widening to 45 removed
# it. Neither half survives re-measurement: post-fix the gated rule costs no
# regressions at either window, and the single regression in that earlier sample
# is attributable to the rotation change, not to this rule.
MOVE_STALL_TICKS = 45
MOVE_STALL_EPS = 0.05
MOVE_STALL_MAX_SPEED = 0.02

# Recording tick rate. Habitat advances physics only in whole multiples of its
# fixed 0.008 s timestep, so an off-grid dt makes the physics time delivered per
# tick alternate between two values and puts a ~24% ripple on a constant
# commanded velocity (see HabitatRobotSim.step). 25 Hz (dt = 0.04 s = 5 physics
# steps) is the only usable integer rate on the grid; it is also
# generate_pose2d_dataset.py's default, and the stall-window and rotation-gain
# defaults above are expressed in ticks at this rate.
RECORD_FPS = 25
# ---------------------------------------------------------------------------

def record_path(robot_sim: HabitatRobotSim, controller: HolonomicRobotController, path_3d, timeout=10, fps=RECORD_FPS, debug_draw=True):
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

def _is_stalled(path_progress, speeds, stall_ticks: int, stall_eps: float,
                stall_max_speed: Optional[float]) -> bool:
    """Has the move controller stopped making progress along its reference path?

    Both arguments are appended once per control tick: ``path_progress`` is the
    cumulative along-path distance of the controller's own projected index, and
    ``speeds`` is the base's linear speed. The phase counts as stalled when, over
    the last ``stall_ticks`` ticks,

      * the projected index advanced by less than ``stall_eps`` metres, AND
      * (if ``stall_max_speed`` is set) the base never exceeded that speed.

    Progress is measured along the path rather than as raw displacement so that a
    genuinely slow corner is not mistaken for a stall, and the speed gate is what
    makes the rule safe: a dead phase leaves the base at *exactly* zero velocity,
    whereas a robot merely crawling at the path controller's 0.1 m/s minimum
    approach speed is making real progress and must be left alone. Re-measured at
    25 Hz with the DWPP braking fix in, over 192 held-out episodes and scored
    against the otherwise-identical arm with the rule off: with the gate, 0 goal
    regressions and +1 recovery; without it, 3 regressions and 3.4 pp less path
    length. The gate is the load-bearing part of this rule, not the window.

    ``stall_ticks == 0`` disables the check.
    """
    if not stall_ticks or len(path_progress) <= stall_ticks:
        return False
    if path_progress[-1] - path_progress[-1 - stall_ticks] >= stall_eps:
        return False
    if stall_max_speed is None:
        return True
    return max(speeds[-stall_ticks:]) < stall_max_speed


def _build_phases(num_actions, looking_slices):
    """Tile the action timeline ``[0, num_actions)`` with ordered move/look phases.

    ``looking_slices`` are ``[start, end)`` half-open ranges in ACTION-index space
    (ascending, non-overlapping, as returned by
    ``episode_io.find_looking_slices``). Every gap between them -- plus the head
    and tail -- becomes a ``move`` phase. Returns a list of
    ``(kind, start, end)`` tuples, ``kind in {'move', 'look'}``, that exactly
    covers ``[0, num_actions)`` in order.

    Note that a move phase's reference sub-path can be degenerate: the demos
    contain MOVE_FORWARDs the collection-time agent could not execute (it walked
    into geometry), which leave the recorded position unchanged, and 3.9% of move
    phases are shorter than 1 cm end to end (measured over 6,950 episodes,
    ``dump/rotation_segmentation/FINDINGS.md``). These are harmless: densify_path
    drops the duplicate points, the path controller finds itself inside
    ``goal_tolerance_dist`` immediately, and the phase ends after one tick. They
    are deliberately not filtered here -- doing so would cost a code path to save
    roughly a quarter of a tick per episode.
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
    fps: int = RECORD_FPS,
    debug_draw: bool = False,
    pure_pursuit_config: Optional[HolonomicControllerConfig] = None,
    rotation_config: Optional[HeadingRotationConfig] = None,
    move_stall_ticks: int = MOVE_STALL_TICKS,
    move_stall_eps: float = MOVE_STALL_EPS,
    move_stall_max_speed: Optional[float] = MOVE_STALL_MAX_SPEED,
    health_monitor=None,
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

    -------------------------
    When a phase hands over
    -------------------------
    A phase ends on the first tick that any of these holds:

      * MOVE: the path controller is inside its goal tolerance and the base is
        stationary;
      * MOVE: the phase has stalled -- no progress along the reference path, and
        no motion at all, for ``move_stall_ticks`` ticks (see ``_is_stalled``);
      * LOOK: the heading trajectory is complete. There is deliberately NO
        stationarity requirement here: :class:`HeadingRotationController` commands
        exactly zero on the tick it accepts the last heading, and the base's yaw
        joint is a velocity motor with no measurable lag or coast, so the robot is
        already stopped. The move phase that follows overwrites the yaw command
        every tick in any case;
      * the local timeout expires.

    ``health_monitor``, when given, is a
    :class:`continuous_demos.drive_failure.EpisodeHealthMonitor`; its ``observe``
    is called once per recorded tick, immediately after the physics step and the
    pose append, so its signals are 1:1 with ``poses``. It is what lets the
    generator tell a demonstration that failed catastrophically -- the robot
    leaving the world -- from one that merely looks unremarkable in ``pose2d``;
    see that module's docstring for why no planar quantity can.

    A look phase that is followed by a move phase is told so
    (``handoff_to_path=True``), which lets it accept its FINAL heading at the
    looser ``HeadingRotationConfig.handoff_tolerance``. This is cheap because the
    move phase applies its translation in the path controller's own virtual-heading
    frame -- the base is holonomic -- so a heading residual at handoff costs a few
    ticks of camera pointing and nothing positional, and the path controller
    re-aligns the true heading while it drives.
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
    look_controller = HeadingRotationController(robot_sim, rotation_config)

    robot_sim.sim.navmesh_visualization = debug_draw
    frames = []
    poses = []

    for phase_idx, (kind, p, q) in enumerate(phases):
        # A look phase that hands straight over to a move phase may accept its
        # final heading loosely: the path controller re-aligns while it drives,
        # and it translates in its own virtual-heading frame, so a residual costs
        # camera pointing for a few ticks and nothing positional.
        next_is_move = (
            phase_idx + 1 < len(phases) and phases[phase_idx + 1][0] == "move"
        )
        # per-tick history backing the move-phase stall check (see _is_stalled)
        path_progress = []
        speeds = []
        if kind == "move":
            move_controller.reset()
            move_controller.set_hab_path3(path_3d[p : q + 1])
            local_timeout = int(np.ceil(move_controller.controller._path_distances[-1] / 0.5 + 0.5))
        else:
            look_controller.reset()
            look_controller.set_heading_trajectory(
                path_headings[p : q + 1],
                remove_redundant=True,
                handoff_to_path=next_is_move,
            )
            local_timeout =  2 * (q - p + 1)  # allow more time for longer look phases
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
            if health_monitor is not None:
                health_monitor.observe(
                    move_controller=move_controller if kind == "move" else None
                )

            if kind == "move":
                phase_done = move_controller.is_goal_reached() and robot_sim.is_stationary(
                    lin_vel_threshold=0.01, ang_vel_threshold=0.01
                )
            else:
                # No stationarity guard: the rotation law commands exactly zero
                # once the trajectory is complete, and the yaw joint is a velocity
                # motor with no measurable lag or coast, so the robot is already
                # stopped on the tick the last heading is accepted. Waiting for
                # is_stationary() only re-recorded frames of an already-stopped
                # robot. The move phase that follows overwrites the yaw command
                # every tick regardless.
                phase_done = look_controller.is_trajectory_complete()
            if phase_done:
                break
            if kind == "move" and move_stall_ticks:
                controller = move_controller.controller
                path_progress.append(
                    float(controller._path_distances[controller.last_idx])
                )
                speeds.append(math.hypot(
                    robot_sim.get_joint_state("joint_x", is_vel=True),
                    robot_sim.get_joint_state("joint_y", is_vel=True),
                ))
                if _is_stalled(path_progress, speeds, move_stall_ticks,
                               move_stall_eps, move_stall_max_speed):
                    break
            if robot_sim.is_tipped_over(angle_threshold_degrees=45.0):
                # print("Robot tipped over! Stopping simulation.")
                return frames, poses
        if len(frames) >= timeout * fps:
            # print("Timeout reached! Stopping simulation.")
            break
    return frames, poses
