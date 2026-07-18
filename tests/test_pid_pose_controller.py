"""Invariant / step-response checks for pid_pose_controller.py.

No habitat_sim/magnum dependency: PIDPoseController only calls
get_2d_pose()/get_joint_state()/step() on whatever "robot_sim" object it's
given, so it's tested here against a small kinematic double-integrator stub
rather than the real HabitatRobotSim -- exactly the property that made this
controller easy to validate manually (see docs/PID_TRACKING_REPORT.md)
turned into an automated regression test.
"""

import math

import numpy as np
import pytest

from continuous_demos.pid_pose_controller import (
    PIDPoseController,
    PIDPoseControllerConfig,
    track_trajectory,
    wrap_angle,
)


class FakePlanarRobotSim:
    """Kinematic double-integrator stand-in for HabitatRobotSim.

    Implements exactly the surface PIDPoseController/track_trajectory touch:
    get_2d_pose(), get_joint_state(name, is_vel=True), and
    step(chassis_twist, twist_frame, dt). World-frame twist is integrated
    directly (twist_frame="world" is the only mode PIDPoseController uses).
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x, self.y, self.theta = x, y, theta
        self.vx, self.vy, self.vth = 0.0, 0.0, 0.0

    def get_2d_pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)

    def get_joint_state(self, joint_name: str, is_vel: bool = False) -> float:
        assert is_vel, "PIDPoseController only reads joint velocities"
        return {"joint_x": self.vx, "joint_y": self.vy, "joint_th": self.vth}[joint_name]

    def step(self, chassis_twist, twist_frame: str = "world", dt: float = 1.0 / 30.0) -> None:
        assert twist_frame == "world"
        vx, vy, w = chassis_twist
        self.x += vx * dt
        self.y += vy * dt
        self.theta += w * dt
        self.vx, self.vy, self.vth = vx, vy, w


class TestWrapAngle:
    @pytest.mark.parametrize(
        "angle,expected",
        [
            (0.0, 0.0),
            (math.pi, -math.pi),  # wrap_angle maps to the half-open interval [-pi, pi)
            (-math.pi, -math.pi),
            (2 * math.pi, 0.0),
            (3 * math.pi, -math.pi),
            (-2 * math.pi, 0.0),
        ],
    )
    def test_known_values(self, angle, expected):
        assert wrap_angle(angle) == pytest.approx(expected, abs=1e-9)

    def test_always_in_range(self):
        for angle in np.linspace(-10 * math.pi, 10 * math.pi, 200):
            wrapped = wrap_angle(angle)
            assert -math.pi - 1e-9 <= wrapped <= math.pi + 1e-9


class TestPIDPoseControllerStepResponse:
    """Mirrors the manual step-response check from PID_TRACKING_REPORT.md:
    a constant, unreached setpoint should be approached monotonically-ish and
    without overshoot, even under the aggressive accel/velocity limits tuned
    for trajectory tracking.
    """

    def test_converges_to_constant_setpoint_without_overshoot(self):
        sim = FakePlanarRobotSim(x=0.0, y=0.0, theta=0.0)
        controller = PIDPoseController(sim, PIDPoseControllerConfig())
        target = np.array([0.4, 0.4, math.radians(30.0)])
        dt = 1.0 / 30.0

        errors = []
        for _ in range(90):  # 3 seconds
            step_args, _debug = controller.step(target, ref_vel=None, dt=dt)
            sim.step(**step_args, dt=dt)
            pos_err = float(np.linalg.norm(target[:2] - sim.get_2d_pose()[:2]))
            errors.append(pos_err)

        initial_error = errors[0]
        final_error = errors[-1]

        assert final_error < 0.01, "should converge close to the setpoint within 3s"
        # Bounded overshoot: error should never exceed the initial gap by more
        # than a small margin (a real step response settles in, it doesn't
        # diverge past where it started).
        assert max(errors) <= initial_error * 1.05

    def test_zero_error_setpoint_produces_no_drift(self):
        sim = FakePlanarRobotSim(x=1.0, y=-2.0, theta=0.5)
        controller = PIDPoseController(sim, PIDPoseControllerConfig())
        target = sim.get_2d_pose().copy()
        dt = 1.0 / 30.0

        for _ in range(30):
            step_args, _debug = controller.step(target, ref_vel=None, dt=dt)
            sim.step(**step_args, dt=dt)

        np.testing.assert_allclose(sim.get_2d_pose(), target, atol=1e-6)


class TestTrackTrajectory:
    def test_output_shape_matches_reference(self):
        sim = FakePlanarRobotSim()
        controller = PIDPoseController(sim, PIDPoseControllerConfig())
        n = 50
        ref_traj = np.stack(
            [np.linspace(0.0, 2.0, n), np.zeros(n), np.zeros(n)], axis=1
        )
        result = track_trajectory(sim, controller, ref_traj, dt=1.0 / 30.0)

        assert result["actual_poses"].shape == ref_traj.shape
        assert result["num_steps"] == n

    def test_tracks_straight_line_with_small_error(self):
        sim = FakePlanarRobotSim()
        controller = PIDPoseController(sim, PIDPoseControllerConfig())
        n = 200
        # Slow-enough reference (0.5 m/s) that the controller's velocity/accel
        # limits are never binding, so tracking error should stay small.
        dt = 1.0 / 30.0
        speed = 0.5
        xs = np.arange(n) * speed * dt
        ref_traj = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)

        result = track_trajectory(sim, controller, ref_traj, dt=dt)
        pos_err = np.linalg.norm(result["actual_poses"][:, :2] - ref_traj[:, :2], axis=1)

        assert float(np.mean(pos_err)) < 0.05
