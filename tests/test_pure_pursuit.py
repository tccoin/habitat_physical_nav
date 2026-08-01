"""Invariant / known-limit checks for pure_pursuit.py.

No habitat_sim/magnum dependency -- this module is pure numpy/math, so these
tests run in any Python env with the package installed (`pip install -e .`),
not just the `habitat` conda env used for actual simulation.

These aren't "is the DWPP output numerically correct" tests (no ground truth
for that without a reference implementation) -- they check properties that
must hold regardless of tuning: densify_path preserves endpoints and respects
the requested resolution, and a straight-ahead goal produces forward, not
backward or wildly turning, velocity commands.
"""

import math

import numpy as np
import pytest

from continuous_demos.pure_pursuit import ControllerConfig, PurePursuitController


def make_config(**overrides) -> ControllerConfig:
    defaults = dict(
        min_look_ahead_distance=0.05,
        max_look_ahead_distance=0.4,
        look_ahead_time=0.2,
        v_max=2.0,
        v_min=0.0,
        w_max=70.0,
        w_min=-70.0,
        a_max=3.0,
        aw_max=100.0,
        dt=1.0 / 30.0,
        approach_velocity_scaling_dist=0.5,
        min_approach_linear_velocity=0.1,
        goal_tolerance_dist=0.15,
        regulated_linear_scaling_min_radius=0.01,
        regulated_linear_scaling_min_speed=0.1,
    )
    defaults.update(overrides)
    return ControllerConfig(**defaults)


class TestDensifyPath:
    def test_preserves_endpoints(self):
        controller = PurePursuitController(make_config(), method_name="dwpp")
        path = np.array([[0.0, 0.0], [5.0, 0.0]])
        dense = controller.densify_path(path, resolution=0.5)
        np.testing.assert_allclose(dense[0], path[0])
        np.testing.assert_allclose(dense[-1], path[-1])

    def test_respects_resolution(self):
        controller = PurePursuitController(make_config(), method_name="dwpp")
        path = np.array([[0.0, 0.0], [5.0, 0.0]])
        resolution = 0.5
        dense = controller.densify_path(path, resolution=resolution)
        gaps = np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1)
        # Interpolation splits each segment into equal sub-steps <= resolution;
        # allow a small tolerance for floating point.
        assert np.all(gaps <= resolution + 1e-9)

    def test_skips_near_duplicate_points(self):
        controller = PurePursuitController(make_config(), method_name="dwpp")
        path = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
        dense = controller.densify_path(path, resolution=0.5)
        gaps = np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1)
        assert np.all(gaps > 1e-9)


class TestStraightLineSanity:
    def test_commands_forward_velocity_toward_distant_goal(self):
        controller = PurePursuitController(make_config(), method_name="dwpp")
        controller.set_path(np.array([[0.0, 0.0], [5.0, 0.0]]))

        current_pose = np.array([0.0, 0.0, 0.0])  # at start, facing +x
        current_velocity = np.array([0.0, 0.0])  # [v, w]
        (v_ref, w_ref), _debug = controller.compute_velocity(current_pose, current_velocity)

        assert v_ref > 0.0, "should command forward motion toward a goal straight ahead"
        assert abs(w_ref) < 1.0, "heading is already aligned with a straight path; yaw command should be small"

    def test_look_ahead_heading_matches_straight_path_direction(self):
        controller = PurePursuitController(make_config(), method_name="dwpp")
        controller.set_path(np.array([[0.0, 0.0], [5.0, 0.0]]))
        current_pose = np.array([0.0, 0.0, 0.0])
        controller.compute_velocity(current_pose, np.array([0.0, 0.0]))

        heading = controller.get_look_ahead_heading()
        assert abs(heading) < 0.1, "look-ahead heading along +x path should be ~0 rad"


def _drive_straight(controller, length, max_ticks=900, dt=1.0 / 25.0):
    """Closed-loop roll-out along a straight path, mirroring record_segmented_demo:
    stop once inside the goal tolerance and at rest. Returns (distance travelled,
    peak speed, ticks)."""
    pose = np.array([0.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.0])
    goal = np.array([length, 0.0])
    speeds = [0.0]
    ticks = 0
    for ticks in range(max_ticks):
        at_goal = np.linalg.norm(pose[:2] - goal) < controller.config.goal_tolerance_dist
        if at_goal and abs(velocity[0]) < 0.01:
            break
        (v, w), _ = controller.compute_velocity(pose, velocity)
        pose = pose + np.array([v * math.cos(pose[2]), v * math.sin(pose[2]), 0.0]) * dt
        pose[2] += w * dt
        velocity = np.array([v, w])
        speeds.append(v)
    return float(pose[0]), max(speeds), ticks


class TestBrakingDistanceUsesCurrentSpeed:
    """`_decide_accel_or_decel` must brake from the CURRENT speed, not from v_max.

    Regression test for a logic error that silently froze short move phases. The
    predicate was `goal_distance > v_max**2 / (2*a_max)` -- a constant 0.667 m for
    the default config that never referenced the robot's speed. Any path shorter
    than that constant was therefore classified as "decelerating" from its first
    tick while the robot was still at rest, and the dynamic-window search then
    picked its lowest-velocity candidate, which at rest is zero. The robot never
    moved at all and sat out the phase's entire timeout.
    """

    def test_at_rest_it_always_accelerates(self):
        controller = PurePursuitController(make_config(), method_name="dwpp")
        controller.set_path(np.array([[0.0, 0.0], [0.3, 0.0]]))
        # braking distance from v = 0 is 0, so any remaining distance means "go"
        assert controller._decide_accel_or_decel(0, np.array([0.0, 0.0])) is True

    def test_stopping_distance_matches_the_discrete_braking_ladder(self):
        """The controller sheds at most a_max*dt per tick, so the true stopping
        distance is the sum of that ladder, not the continuous v**2/(2a)."""
        config = make_config()
        controller = PurePursuitController(config, method_name="dwpp")
        dv = config.a_max * config.dt
        for v in (0.0, 0.1, 0.5, 1.0, 2.0):
            # simulate the ladder directly
            speed, expected = v, 0.0
            while speed > 0.0:
                expected += speed * config.dt
                speed = max(speed - dv, 0.0)
            assert controller._discrete_stopping_distance(v) == pytest.approx(expected)
            # and it is never optimistic relative to the continuous formula
            assert controller._discrete_stopping_distance(v) >= v ** 2 / (2 * config.a_max)

    def test_at_speed_it_brakes_within_its_stopping_distance(self):
        config = make_config()
        controller = PurePursuitController(config, method_name="dwpp")
        controller.set_path(np.array([[0.0, 0.0], [10.0, 0.0]]))
        stopping = config.v_max ** 2 / (2 * config.a_max)
        n = len(controller.path)
        idx_far = int(np.searchsorted(controller._path_distances,
                                      controller._path_distances[-1] - 2 * stopping))
        idx_near = int(np.searchsorted(controller._path_distances,
                                       controller._path_distances[-1] - 0.5 * stopping))
        at_v_max = np.array([config.v_max, 0.0])
        assert controller._decide_accel_or_decel(idx_far, at_v_max) is True
        assert controller._decide_accel_or_decel(idx_near, at_v_max) is False
        assert idx_near < n

    @pytest.mark.parametrize("length", [0.2, 0.25, 0.4, 0.5, 0.6, 0.667])
    def test_short_paths_actually_move(self, length):
        """These are the lengths the old constant froze completely.

        0.25 m is one MOVE_FORWARD primitive, so this band is extremely common in
        the demonstrations.
        """
        controller = PurePursuitController(make_config(), method_name="dwpp")
        controller.set_path(np.array([[0.0, 0.0], [length, 0.0]]))
        travelled, peak_v, ticks = _drive_straight(controller, length)

        assert peak_v > 0.0, "robot never moved on a short path"
        # lands inside the controller's own goal tolerance. The residual is
        # projection-index lag in _calc_index, not braking: `goal_distance` is
        # measured from the projected path point, which trails the true pose.
        assert abs(travelled - length) < controller.config.goal_tolerance_dist
        assert ticks < 100, "should reach the goal promptly, not idle out the phase"

    @pytest.mark.parametrize("length", [1.0, 3.0, 10.0])
    def test_long_paths_still_reach_the_goal_without_overshooting(self, length):
        controller = PurePursuitController(make_config(), method_name="dwpp")
        controller.set_path(np.array([[0.0, 0.0], [length, 0.0]]))
        travelled, peak_v, _ = _drive_straight(controller, length)

        assert abs(travelled - length) < controller.config.goal_tolerance_dist
        # a long path has room to reach the speed cap
        assert peak_v == pytest.approx(controller.config.v_max, rel=0.2)

    def test_the_old_constant_would_have_frozen_a_short_path(self):
        """Pins the mechanism, so the regression cannot come back unnoticed."""
        config = make_config()
        controller = PurePursuitController(config, method_name="dwpp")
        controller.set_path(np.array([[0.0, 0.0], [0.5, 0.0]]))

        legacy_decel_distance = config.v_max ** 2 / (2 * config.a_max)
        assert legacy_decel_distance == pytest.approx(0.667, abs=0.001)
        # the whole path is shorter than the old constant -> old code decelerated
        # from tick 0; the fixed predicate accelerates instead.
        assert controller._path_distances[-1] < legacy_decel_distance
        assert controller._decide_accel_or_decel(0, np.array([0.0, 0.0])) is True
