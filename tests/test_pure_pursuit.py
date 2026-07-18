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
