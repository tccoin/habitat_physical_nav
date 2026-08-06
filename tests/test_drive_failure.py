"""Unit tests for the pure logic in ``continuous_demos.drive_failure``.

Everything here runs without habitat_sim: the monitor talks to the simulator
through three tiny accessors (joint_z position and velocity, planar pose), so a
stub robot exercises every rejection rule. The claim these tests cannot make --
that ``joint_z``'s velocity really does free-fall when the robot is outside an
MP3D scene and really does stay at zero when it is not -- is checked against the
simulator itself in ``test_drive_failure_in_sim.py``, and at corpus scale in
``dump/drive_failure_detection/`` in the spatial_training repo.
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
G = 9.81


class _StubRobot:
    def __init__(self):
        self.translation = [0.0, 1.0, 0.0]


class _StubSim:
    """A simulator whose joint_z position and velocity are scripted per tick."""

    def __init__(self, joint_z, joint_vz, pose=None):
        self.robot = _StubRobot()
        self._z = list(joint_z)
        self._vz = list(joint_vz)
        self._pose = pose
        self.i = 0

    def get_joint_state(self, name, is_vel=False):
        if name != "joint_z":
            raise ValueError(name)
        return self._vz[self.i] if is_vel else self._z[self.i]

    def get_2d_pose(self):
        if self._pose is None:
            return np.array([0.0, 0.0, 0.0])
        return np.asarray(self._pose[self.i], dtype=float)


def run(joint_z, joint_vz, config=None, poses=None):
    sim = _StubSim(joint_z, joint_vz)
    monitor = EpisodeHealthMonitor(sim, fps=FPS, config=config)
    for i in range(len(joint_vz)):
        sim.i = i
        monitor.observe()
    return monitor.finish(poses)


def free_fall(n_ticks, z0=0.0):
    """(positions, velocities) of an unactuated joint falling from rest."""
    t = np.arange(1, n_ticks + 1) / FPS
    return list(z0 - 0.5 * G * t ** 2), list(-G * t)


def test_a_supported_robot_is_healthy():
    """joint_z pinned by the floor: velocity noise around zero, no fall run."""
    n = 200
    vz = list(0.02 * np.sin(np.arange(n)))
    health = run([-0.05] * n, vz)
    assert health.success
    assert health.outcome == "ok"
    assert not health.escaped
    assert health.fall_run_ticks == 0
    assert health.max_fall_speed_mps < 0.1


def test_free_fall_is_an_escape():
    z, vz = free_fall(60)
    health = run(z, vz)
    assert health.escaped
    assert not health.success
    assert health.outcome == "escaped"
    # -2 m/s is crossed after 0.2 s, so 60 ticks of fall leave a 2.2 s run.
    assert health.fall_run_s == pytest.approx(60 / FPS - 0.2, abs=2 / FPS)
    assert health.max_fall_speed_mps > 20.0


def test_the_deepest_healthy_fall_measured_is_not_an_escape():
    """pRbA3pwrgk9 episode 29: a real 4.2 m drop down a stairwell that lands,
    peak descent 6.5 m/s, 14 consecutive ticks below -2 m/s. It is the closest
    any of 2,204 healthy episodes gets to the rule, and it must survive."""
    n = 14
    z, vz = free_fall(n + 5)
    vz = list(np.clip(vz, -6.5, None))
    z += [z[-1]] * 100
    vz += [0.0] * 100
    health = run(z, vz)
    assert not health.escaped
    assert health.success
    assert health.fall_run_ticks < DriveFailureConfig().fall_duration_s * FPS


def test_a_step_down_is_not_an_escape():
    """Habitat builds the navmesh with agent_max_climb = 0.2 m, so the tallest
    step the robot can legitimately take reaches sqrt(2*g*0.2) = 1.98 m/s -- and
    only instantaneously. It must not be condemned."""
    drop = 0.2
    t_fall = math.sqrt(2 * drop / G)               # 0.20 s
    n = max(1, int(round(t_fall * FPS)))
    z = list(-0.5 * G * (np.arange(1, n + 1) / FPS) ** 2)
    vz = list(-G * np.arange(1, n + 1) / FPS)
    # ... then it lands and joint_z is pinned again
    z += [z[-1]] * 100
    vz += [0.0] * 100
    health = run(z, vz)
    assert not health.escaped
    assert health.success
    assert health.max_fall_speed_mps < 2.1
    assert health.max_drop_m == pytest.approx(drop, abs=0.05)


def test_a_long_staircase_descent_is_not_an_escape():
    """Three metres of descent, taken in contact one step at a time. Descent
    depth alone would condemn this; the free-fall signature does not."""
    z, vz = [0.0], [0.0]
    for _ in range(20):                     # 20 steps of 0.15 m
        z += [z[-1] - 0.05, z[-1] - 0.10, z[-1] - 0.15]
        vz += [-1.2, -1.5, 0.0]
        z += [z[-1]] * 10                   # rolls along the tread
        vz += [0.0] * 10
    health = run(z, vz)
    assert not health.escaped
    assert health.success
    assert health.max_drop_m == pytest.approx(3.0, abs=1e-6)


def test_a_falling_robot_that_lands_is_not_an_escape():
    """0.6 s of free flight -- a 1.8 m drop off a ledge -- then contact. Below
    the 1.0 s window, deliberately: it is a rough demonstration, not a broken
    one, and every false positive here deletes a real episode."""
    n = int(0.6 * FPS)
    z, vz = free_fall(n)
    z += [z[-1]] * 100
    vz += [0.0] * 100
    health = run(z, vz)
    assert not health.escaped
    assert 0 < health.fall_run_ticks < 0.5 * FPS


def test_a_failed_joint_query_never_deletes_an_episode():
    class _Broken(_StubSim):
        def get_joint_state(self, name, is_vel=False):
            raise RuntimeError("no such joint")

    sim = _Broken([0.0] * 50, [0.0] * 50)
    monitor = EpisodeHealthMonitor(sim, fps=FPS)
    for i in range(50):
        sim.i = i
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
    sim = _StubSim([-0.05] * n, [0.0] * n, pose=[(5.0, 0.0, 0.0)] * n)
    monitor = EpisodeHealthMonitor(sim, fps=FPS)
    for i in range(n):
        sim.i = i
        monitor.observe(move_controller=_Ctrl())
    health = monitor.finish()
    assert health.ref_miss_mean_m == pytest.approx(5.0)
    assert health.ref_miss_worst_run_s == pytest.approx(n / FPS)
    assert health.success  # recorded, not enforced

    armed = EpisodeHealthMonitor(sim, fps=FPS,
                                 config=DriveFailureConfig(reject_ref_miss_s=1.0))
    for i in range(n):
        sim.i = i
        armed.observe(move_controller=_Ctrl())
    health = armed.finish()
    assert not health.success
    assert health.outcome == "ref_miss"


def test_an_escaped_robot_can_track_its_reference_path_perfectly():
    """The asymmetry that decides the design: tracking error catches the stuck
    case and is blind to the escape. Here the robot is in free fall and dead on
    its reference path -- only the vertical signal fires."""
    class _PP:
        path = np.array([[0.0, 0.0], [1.0, 0.0]])
        last_idx = 0

    class _Ctrl:
        controller = _PP()

    n = 60
    z, vz = free_fall(n)
    sim = _StubSim(z, vz, pose=[(0.0, 0.0, 0.0)] * n)
    monitor = EpisodeHealthMonitor(
        sim, fps=FPS, config=DriveFailureConfig(reject_ref_miss_s=1.0))
    for i in range(n):
        sim.i = i
        monitor.observe(move_controller=_Ctrl())
    health = monitor.finish()
    assert health.ref_miss_max_m == pytest.approx(0.0)
    assert health.ref_miss_worst_run_s == 0.0   # the stuck rule sees nothing
    assert health.escaped                       # the fall rule condemns it
    assert health.outcome == "escaped"


def test_motion_composition_classes_are_exclusive_and_exhaustive():
    fps = 25.0
    poses = [(0.0, 0.0, 0.0)]
    for _ in range(10):                                   # driving at 0.5 m/s
        poses.append((poses[-1][0] + 0.5 / fps, 0.0, 0.0))
    for _ in range(10):                                   # turning at 1.0 rad/s
        poses.append((poses[-1][0], 0.0, poses[-1][2] + 1.0 / fps))
    for _ in range(10):                                   # frozen
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
