"""Unit tests for the pure logic in ``continuous_demos.drive_failure``.

Everything here runs without habitat_sim: the monitor talks to the simulator
through three tiny accessors (contact count, height, planar pose), so a stub
robot is enough to exercise every rejection rule. What cannot be tested here --
whether the contact query really goes to zero when the robot leaves an MP3D
scene -- is empirical and lives in
``dump/drive_failure_detection/FINDINGS.md`` in the spatial_training repo.
"""

import math

import numpy as np
import pytest

from continuous_demos.drive_failure import (
    DriveFailureConfig,
    EpisodeHealthMonitor,
    motion_composition,
)

FPS = 25.0


class _StubRobot:
    def __init__(self):
        self.translation = [0.0, 1.0, 0.0]
        self.link_object_ids = {1: 0}
        self.object_id = 1


class _StubSim:
    """A simulator whose contact count and joint_z are scripted per tick."""

    def __init__(self, contacts, joint_z, pose=None):
        self.robot = _StubRobot()
        self._contacts = list(contacts)
        self._joint_z = list(joint_z)
        self._pose = pose
        self.i = -1
        self.sim = self  # monitor reads robot_sim.sim.get_physics_contact_points

    # --- HabitatRobotSim surface used by the monitor ---
    def get_joint_state(self, name, is_vel=False):
        if name != "joint_z":
            raise ValueError(name)
        return self._joint_z[self.i]

    def get_2d_pose(self):
        if self._pose is None:
            return np.array([0.0, 0.0, 0.0])
        return np.asarray(self._pose[self.i], dtype=float)

    def get_physics_contact_points(self):
        class _C:
            object_id_a, object_id_b = 1, -1
        return [_C() for _ in range(self._contacts[self.i])]

    @property
    def pathfinder(self):
        raise RuntimeError("no pathfinder in this stub")

    # --- driving the script ---
    def tick(self):
        self.i += 1


def run(contacts, joint_z, config=None, poses=None):
    sim = _StubSim(contacts, joint_z)
    sim.i = 0  # so the constructor's spawn-height read is well defined
    monitor = EpisodeHealthMonitor(sim, fps=FPS, config=config, track_navmesh=False)
    sim.i = -1
    for _ in range(len(contacts)):
        sim.tick()
        monitor.observe()
    return monitor.finish(poses)


def test_supported_episode_is_healthy():
    n = 200
    health = run([4] * n, [-0.05] * n)
    assert health.success
    assert health.outcome == "ok"
    assert not health.escaped
    assert health.unsupported_run_ticks == 0


def test_brief_airborne_hop_is_not_an_escape():
    """A robot dropping off a step goes contact-free for a moment. The longest
    such run measured over 479 real episodes is 25 ticks (1.0 s); the default
    2.0 s window must clear it with room to spare."""
    contacts = [4] * 100 + [0] * 25 + [4] * 100
    joint_z = [-0.05] * 100 + list(np.linspace(-0.05, -3.1, 25)) + [-3.1] * 100
    health = run(contacts, joint_z)
    assert not health.escaped
    assert health.success
    assert health.unsupported_run_ticks == 25
    # It really did descend 3 m -- height alone would have condemned it.
    assert health.max_drop_m == pytest.approx(3.05, abs=1e-6)


def test_sustained_loss_of_contact_is_an_escape():
    contacts = [4] * 100 + [0] * 60
    joint_z = [-0.05] * 100 + list(np.linspace(-0.05, -20.0, 60))
    health = run(contacts, joint_z)
    assert health.escaped
    assert not health.success
    assert health.outcome == "escaped"
    assert health.unsupported_run_s == pytest.approx(60 / FPS)


def test_terminal_escape_shorter_than_the_main_window():
    """An escape in the last second still has to be caught, but only when the
    robot has also fallen away from its last supported height."""
    contacts = [4] * 100 + [0] * 30
    joint_z = [-0.05] * 100 + list(np.linspace(-0.05, -6.0, 30))
    assert run(contacts, joint_z).escaped

    # Same 30 contact-free ticks, no descent: a hop the episode happens to end
    # on, not an escape. Both halves of the terminal clause are required.
    flat = [-0.05] * 130
    health = run(contacts, flat)
    assert not health.escaped
    assert health.unsupported_terminal_ticks == 30


def test_a_failed_contact_query_never_deletes_an_episode():
    class _Broken(_StubSim):
        def get_physics_contact_points(self):
            raise RuntimeError("bullet said no")

    sim = _Broken([0] * 200, [-0.05] * 200)
    sim.i = 0
    monitor = EpisodeHealthMonitor(sim, fps=FPS, track_navmesh=False)
    sim.i = -1
    for _ in range(200):
        sim.tick()
        monitor.observe()
    health = monitor.finish()
    assert not health.escaped
    assert health.success


def test_ref_miss_rule_is_off_by_default_and_arms_on_request():
    class _PP:
        path = np.array([[0.0, 0.0], [1.0, 0.0]])
        last_idx = 0

    class _Ctrl:
        controller = _PP()

    n = 100
    sim = _StubSim([4] * n, [-0.05] * n, pose=[(5.0, 0.0, 0.0)] * n)
    sim.i = 0
    monitor = EpisodeHealthMonitor(sim, fps=FPS, track_navmesh=False)
    sim.i = -1
    for _ in range(n):
        sim.tick()
        monitor.observe(move_controller=_Ctrl())
    health = monitor.finish()
    assert health.ref_miss_mean_m == pytest.approx(5.0)
    assert health.ref_miss_worst_run_s == pytest.approx(n / FPS)
    assert health.success  # recorded, not enforced

    sim.i = 0
    armed = EpisodeHealthMonitor(sim, fps=FPS, track_navmesh=False,
                                 config=DriveFailureConfig(reject_ref_miss_s=1.0))
    sim.i = -1
    for _ in range(n):
        sim.tick()
        armed.observe(move_controller=_Ctrl())
    health = armed.finish()
    assert not health.success
    assert health.outcome == "ref_miss"


def test_motion_composition_classes_are_exclusive_and_exhaustive():
    fps = 25.0
    poses = [(0.0, 0.0, 0.0)]
    # 10 driving ticks (0.5 m/s), 10 turning in place (1.0 rad/s), 10 frozen
    for i in range(10):
        poses.append((poses[-1][0] + 0.5 / fps, 0.0, 0.0))
    for i in range(10):
        poses.append((poses[-1][0], 0.0, poses[-1][2] + 1.0 / fps))
    for i in range(10):
        poses.append(poses[-1])
    comp = motion_composition(poses, fps)
    total = (comp["frac_drive"] + comp["frac_turn"]
             + comp["frac_settle"] + comp["frac_frozen"])
    assert total == pytest.approx(1.0)
    assert comp["frac_drive"] == pytest.approx(10 / 30)
    assert comp["frac_turn"] == pytest.approx(10 / 30)
    assert comp["frac_frozen"] == pytest.approx(10 / 30)
    assert comp["path_len_m"] == pytest.approx(10 * 0.5 / fps)


def test_motion_composition_handles_heading_wraparound():
    fps = 25.0
    poses = [(0.0, 0.0, math.pi - 0.02), (0.0, 0.0, -math.pi + 0.02)]
    comp = motion_composition(poses, fps)
    # 0.04 rad in one tick == 1.0 rad/s: a turn, not a 6.24 rad/s spin.
    assert comp["frac_turn"] == pytest.approx(1.0)
