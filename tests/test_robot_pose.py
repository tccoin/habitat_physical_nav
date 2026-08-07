"""Where the robot is, and the heading<->quaternion conversion that spawns it.

The height helpers need a simulator to be interesting, so they are exercised against a
stub with just the attributes they read. The quaternion conversion needs nothing: it is
pinned against `replay.hab_quat_to_heading`, the inverse that already exists and that
every recorded demonstration was read through.
"""

import math

import numpy as np
import pytest

from continuous_demos.replay import hab_quat_to_heading
from continuous_demos.robot_pose import (
    navmesh_height,
    planar_heading_to_quat_xyzw,
    planar_to_habitat,
    robot_height_m,
    robot_navmesh_position,
    robot_position_3d,
    urdf_vertical_offset,
)


class FakeAABB:
    def __init__(self, min_y):
        self.min = type("V", (), {"y": min_y})()


class FakeNode:
    def __init__(self, min_y):
        self.cumulative_bb = FakeAABB(min_y)


class FakeRobot:
    def __init__(self, translation, min_y=-0.25):
        self.translation = translation
        self.root_scene_node = FakeNode(min_y)


class FakeSim:
    """Only the four things `robot_pose` reads."""

    def __init__(self, pose2d=(1.0, 2.0, 0.5), spawn_y=0.3, joint_z=0.0, min_y=-0.25,
                 joint_readable=True):
        self.robot = FakeRobot([0.0, spawn_y, 0.0], min_y=min_y)
        self._pose2d = np.array(pose2d, dtype=float)
        self._joint_z = joint_z
        self._joint_readable = joint_readable

    def get_2d_pose(self):
        return self._pose2d

    def get_joint_state(self, name, is_vel=False):
        if not self._joint_readable:
            raise ValueError(f"no joint {name}")
        return self._joint_z


# ---------------------------------------------------------------------------
# The height stack
# ---------------------------------------------------------------------------
def test_height_is_the_spawn_plus_the_passive_vertical_joint():
    """`translation[1]` alone never changes during an episode -- the base link is pinned."""
    assert robot_height_m(FakeSim(spawn_y=0.3, joint_z=0.0)) == pytest.approx(0.3)
    assert robot_height_m(FakeSim(spawn_y=0.3, joint_z=-4.0)) == pytest.approx(-3.7)


def test_an_unreadable_joint_falls_back_rather_than_raising():
    """This feeds the escape detector; a failed query must never delete an episode."""
    assert robot_height_m(FakeSim(spawn_y=0.3, joint_readable=False)) == pytest.approx(0.3)


def test_navmesh_height_subtracts_the_urdf_lift():
    """The height a pathfinder query must use, as opposed to where the robot is."""
    sim = FakeSim(spawn_y=0.3, joint_z=0.0, min_y=-0.25)
    assert urdf_vertical_offset(sim) == pytest.approx(0.25)
    assert robot_height_m(sim) == pytest.approx(0.30)
    assert navmesh_height(sim) == pytest.approx(0.05)


def test_a_model_already_resting_on_the_origin_has_no_lift():
    assert urdf_vertical_offset(FakeSim(min_y=0.0)) == 0.0
    assert urdf_vertical_offset(FakeSim(min_y=0.4)) == 0.0


def test_the_two_position_helpers_differ_only_in_the_vertical():
    sim = FakeSim(pose2d=(1.0, 2.0, 0.5), spawn_y=0.3, min_y=-0.25)
    live = robot_position_3d(sim)
    query = robot_navmesh_position(sim)
    assert live[0] == pytest.approx(query[0]) == pytest.approx(1.0)
    assert live[2] == pytest.approx(query[2]) == pytest.approx(-2.0)  # z = -Y
    assert live[1] - query[1] == pytest.approx(0.25)


def test_planar_to_habitat_negates_z():
    assert list(planar_to_habitat([3.0, 4.0, 1.23], 0.5)) == pytest.approx([3.0, 0.5, -4.0])


# ---------------------------------------------------------------------------
# Heading <-> quaternion. Pinned against the inverse the corpus was read through.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "theta",
    [0.0, 0.1, math.pi / 4, math.pi / 2, 2.0, math.pi - 1e-6, -math.pi / 2, -2.5, -0.3],
)
def test_heading_round_trips_through_the_existing_inverse(theta):
    """`hab_quat_to_heading` is what every recorded demonstration was read with, so
    agreeing with it *is* the definition of correct here."""
    quat = planar_heading_to_quat_xyzw(theta)
    assert hab_quat_to_heading(quat) == pytest.approx(theta, abs=1e-6)


def test_it_is_a_unit_quaternion_about_the_up_axis_only():
    for theta in (0.0, 1.0, -2.0, 3.0):
        q = planar_heading_to_quat_xyzw(theta)
        assert q[0] == 0.0 and q[2] == 0.0, "yaw only: no roll, no pitch"
        assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=1e-6)


def test_zero_heading_is_not_the_identity_quaternion():
    """The half-turn offset is the whole content of this function: habitat's forward is
    local -z while the planar frame measures theta from +X. Dropping it points every
    spawned robot 90 degrees away from where the caller meant."""
    q = planar_heading_to_quat_xyzw(0.0)
    assert not np.allclose(q, [0.0, 0.0, 0.0, 1.0])
    assert hab_quat_to_heading([0.0, 0.0, 0.0, 1.0]) == pytest.approx(math.pi / 2)


def test_it_agrees_with_the_objectnav_task_layers_version():
    """Two implementations exist because this module must not import habitat-lab.
    That is a reason to check they agree, not a reason to assume it."""
    habitat_sim = pytest.importorskip("habitat_sim")
    from continuous_demos.objectnav import planar_heading_to_habitat_quat

    for theta in (0.0, 0.7, -1.9, math.pi / 3):
        theirs = planar_heading_to_habitat_quat(theta)
        mine = planar_heading_to_quat_xyzw(theta)
        # habitat's quaternion is (w, x, y, z)-ordered in its own type; compare through
        # the heading each denotes rather than through a component ordering.
        as_xyzw = np.array([theirs.imag[0], theirs.imag[1], theirs.imag[2], theirs.real])
        assert hab_quat_to_heading(as_xyzw) == pytest.approx(
            hab_quat_to_heading(mine), abs=1e-6
        )
