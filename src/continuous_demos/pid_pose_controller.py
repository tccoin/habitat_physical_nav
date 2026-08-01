"""PID-based global pose/trajectory tracking -- a position-control action space.

This is intentionally separate from ``sim.py``'s ``HolonomicRobotController``
(pure-pursuit / DWPP path following). That controller solves "here is a path,
follow it smoothly, in whatever time it takes." This module solves a different,
simpler problem: "here is a target pose (or a stream of target poses at a fixed
cadence), servo the base toward it/them now," the kind of position-control
action space standard in robotic manipulation (target pose in, PID/servo
toward it). A single call to ``PIDPoseController.step()`` is one control tick
against one reference pose; calling it repeatedly against a stream of
reference poses (via ``track_trajectory``) is what turns the same primitive
into trajectory tracking -- there is no separate "trajectory mode."

References are consumed in the same planar frame as the saved ``pose2d``
data and as ``HabitatRobotSim.get_2d_pose()``: ``[X, Y, theta]`` with
``X = habitat_x``, ``Y = -habitat_z``, ``theta`` CCW from ``+X``. See the
frame-convention header in ``sim.py`` for the full picture.

This module depends only on public ``HabitatRobotSim`` methods
(``get_2d_pose``, ``get_joint_state``, ``step``) -- no changes to ``sim.py``
were needed or made.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Default control period. Habitat advances physics only in whole multiples of its
# fixed 0.008 s timestep, so an off-grid dt makes the physics time delivered per
# tick alternate between two values and puts a ~24% ripple on a constant
# commanded velocity (see HabitatRobotSim.step). 25 Hz is the only usable integer
# rate on that grid and is what the pipeline records at, so a controller default
# of 1/30 was both off-grid and 20% wrong against the data it is validated
# against. Callers that know their shard's rate should still pass dt explicitly:
# a PID gain set is only meaningful at the period it was tuned for.
DEFAULT_DT = 1.0 / 25.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle (radians) to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class PIDPoseControllerConfig:
    """Gains and limits for :class:`PIDPoseController`.

    Position gains (``kp_pos``/``ki_pos``/``kd_pos``) are shared between the X
    and Y axes since the base is holonomic and symmetric in those directions.
    Heading gains are separate.
    """

    kp_pos: float = 2.5
    ki_pos: float = 0.0
    kd_pos: float = 0.3

    kp_th: float = 4.0
    ki_th: float = 0.0
    kd_th: float = 0.2

    v_max: float = 2.0           # m/s
    a_max: float = 8.0           # m/s^2
    w_max: float = math.radians(180.0)   # rad/s
    aw_max: float = math.radians(400.0)  # rad/s^2

    integral_limit_pos: float = 0.5   # meters worth of accumulated error
    integral_limit_th: float = math.radians(30.0)

    feedforward_gain: float = 1.0


class PIDPoseController:
    """Drives a holonomic :class:`~sim.HabitatRobotSim` base toward a stream of
    global ``[x, y, theta]`` reference poses using independent PID loops on
    position and heading, plus optional reference-velocity feedforward.
    """

    def __init__(self, robot_sim, config: Optional[PIDPoseControllerConfig] = None):
        self.robot_sim = robot_sim
        self.config = config or PIDPoseControllerConfig()
        self.reset()

    def reset(self) -> None:
        self._integral_pos = np.zeros(2, dtype=float)
        self._integral_th = 0.0
        self._prev_cmd = np.zeros(3, dtype=float)  # [vx, vy, w], world frame

    def step(
        self,
        ref_pose: np.ndarray,
        ref_vel: Optional[np.ndarray] = None,
        dt: float = DEFAULT_DT,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute one control tick toward ``ref_pose``.

        Args:
            ref_pose: ``[x, y, theta]`` target in the planar control frame.
            ref_vel: optional ``[vx, vy, w]`` feedforward velocity (world
                frame). If omitted, feedforward is zero (pure feedback).
            dt: control period; must match the ``dt`` passed to
                ``robot_sim.step()``.

        Returns:
            ``(step_args, debug)`` where ``step_args`` is a dict directly
            usable as ``robot_sim.step(**step_args, dt=dt)``, and ``debug``
            carries the raw errors and saturation flags for logging/eval.
        """
        cfg = self.config
        cur_pose = self.robot_sim.get_2d_pose()
        cur_vx = self.robot_sim.get_joint_state("joint_x", is_vel=True)
        cur_vy = self.robot_sim.get_joint_state("joint_y", is_vel=True)
        cur_w = self.robot_sim.get_joint_state("joint_th", is_vel=True)

        error_xy = np.asarray(ref_pose[:2], dtype=float) - cur_pose[:2]
        error_th = wrap_angle(float(ref_pose[2]) - float(cur_pose[2]))

        # Anti-windup: clamp accumulated integral before it can dominate.
        self._integral_pos = np.clip(
            self._integral_pos + error_xy * dt,
            -cfg.integral_limit_pos,
            cfg.integral_limit_pos,
        )
        self._integral_th = float(
            np.clip(
                self._integral_th + error_th * dt,
                -cfg.integral_limit_th,
                cfg.integral_limit_th,
            )
        )

        # Derivative-on-measurement (not on error): the setpoint can jump every
        # tick when tracking a trajectory, and differencing the error directly
        # would spike ("derivative kick") on every such jump.
        ff = np.zeros(3, dtype=float) if ref_vel is None else np.asarray(ref_vel, dtype=float)

        cmd_xy = (
            cfg.kp_pos * error_xy
            + cfg.ki_pos * self._integral_pos
            - cfg.kd_pos * np.array([cur_vx, cur_vy])
            + cfg.feedforward_gain * ff[:2]
        )
        cmd_w = (
            cfg.kp_th * error_th
            + cfg.ki_th * self._integral_th
            - cfg.kd_th * cur_w
            + cfg.feedforward_gain * float(ff[2])
        )

        # Velocity clamp.
        speed = float(np.linalg.norm(cmd_xy))
        v_saturated = speed > cfg.v_max
        if v_saturated:
            cmd_xy = cmd_xy * (cfg.v_max / speed)
        w_saturated = abs(cmd_w) > cfg.w_max
        cmd_w = float(np.clip(cmd_w, -cfg.w_max, cfg.w_max))

        # Accel (rate) limit against the previously commanded velocity, so
        # tracking a physically-realistic trajectory doesn't teleport.
        cmd = np.array([cmd_xy[0], cmd_xy[1], cmd_w], dtype=float)
        max_delta_lin = cfg.a_max * dt
        max_delta_ang = cfg.aw_max * dt
        delta = cmd - self._prev_cmd
        delta_lin = delta[:2]
        delta_lin_norm = float(np.linalg.norm(delta_lin))
        a_saturated = delta_lin_norm > max_delta_lin
        if a_saturated:
            delta_lin = delta_lin * (max_delta_lin / delta_lin_norm)
        delta_ang = delta[2]
        aw_saturated = abs(delta_ang) > max_delta_ang
        delta_ang = float(np.clip(delta_ang, -max_delta_ang, max_delta_ang))

        cmd = self._prev_cmd + np.array([delta_lin[0], delta_lin[1], delta_ang])
        self._prev_cmd = cmd

        step_args = {
            "chassis_twist": cmd,
            "twist_frame": "world",
        }
        debug = {
            "error_xy": error_xy,
            "error_th": error_th,
            "cur_pose": cur_pose,
            "saturated": {
                "v": bool(v_saturated),
                "w": bool(w_saturated),
                "a": bool(a_saturated),
                "aw": bool(aw_saturated),
            },
        }
        return step_args, debug


def track_trajectory(
    robot_sim,
    controller: PIDPoseController,
    ref_traj: np.ndarray,
    dt: float = DEFAULT_DT,
    ref_vel_traj: Optional[np.ndarray] = None,
    initial_pose: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Drive ``robot_sim`` through ``ref_traj`` (an ``(N, 3)`` array of
    ``[x, y, theta]`` setpoints at fixed ``dt``), one setpoint per control tick.

    This is the "single pose per step vs. whole trajectory" distinction made
    concrete: a trajectory is nothing but repeated single-pose ``controller
    .step()`` calls, each one a normal position-control tick.

    If ``ref_vel_traj`` isn't provided, feedforward velocity is computed by
    finite-differencing ``ref_traj`` (using ``initial_pose`` as the point
    before ``ref_traj[0]`` if given, else zero feedforward for the first tick).

    Returns a dict with the actual pose history (``(N, 3)``), the reference
    trajectory (echoed back for convenience), and per-step debug info.
    """
    ref_traj = np.asarray(ref_traj, dtype=float)
    n = len(ref_traj)
    if ref_vel_traj is None:
        ref_vel_traj = np.zeros((n, 3), dtype=float)
        prev = initial_pose if initial_pose is not None else ref_traj[0]
        for i in range(n):
            cur_ref = ref_traj[i]
            d_xy = (cur_ref[:2] - prev[:2]) / dt
            d_th = wrap_angle(float(cur_ref[2]) - float(prev[2])) / dt
            ref_vel_traj[i] = [d_xy[0], d_xy[1], d_th]
            prev = cur_ref

    actual_poses = np.zeros((n, 3), dtype=float)
    saturation_counts = {"v": 0, "w": 0, "a": 0, "aw": 0}
    for i in range(n):
        step_args, debug = controller.step(ref_traj[i], ref_vel_traj[i], dt=dt)
        robot_sim.step(**step_args, dt=dt)
        actual_poses[i] = robot_sim.get_2d_pose()
        for k, v in debug["saturated"].items():
            saturation_counts[k] += int(v)

    return {
        "actual_poses": actual_poses,
        "ref_traj": ref_traj,
        "saturation_counts": saturation_counts,
        "num_steps": n,
    }
