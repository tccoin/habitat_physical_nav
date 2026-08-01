"""The escape detector, validated against the simulator itself rather than a stub.

The point of this test is that the free-fall signal is testable **in isolation**:
it needs no failing episode and no replay, only a robot deliberately instantiated
outside the world. That is what makes ``joint_z`` velocity the right primitive --
the thing it measures is a direct physical consequence of the failure, so it can
be provoked on demand.

It also pins the trap that made the failure invisible for so long:
``robot.translation[1]`` is bit-identical between a robot standing on the navmesh
and one falling 30 m outside the scene, because ``fixed_base=True`` pins the base
link at spawn. Only ``joint_z`` moves.

Skipped wherever habitat_sim or the MP3D scene assets are not installed, like the
other simulator-touching tests in this suite.
"""

import numpy as np
import pytest

habitat_sim = pytest.importorskip("habitat_sim")

from continuous_demos.drive_failure import (  # noqa: E402
    DriveFailureConfig,
    EpisodeHealthMonitor,
    robot_height_m,
    vertical_velocity_mps,
)

FPS = 25.0
TICKS = 75


@pytest.fixture(scope="module")
def robot_sim():
    from continuous_demos.episode_io import (
        DEFAULT_URDF_PATH,
        build_robot_sim,
        find_scene_dataset_config,
        resolve_scene_path,
    )

    scene_id = "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    try:
        scene_path = resolve_scene_path(scene_id, None)
        cfg = find_scene_dataset_config(scene_path, None)
        sim = build_robot_sim(scene_path=scene_path, scene_dataset_config=cfg,
                              width=64, height=64, urdf_path=str(DEFAULT_URDF_PATH))
    except Exception as exc:  # scene assets not on this machine
        pytest.skip(f"MP3D scene not resolvable here: {exc}")
    yield sim
    sim.sim.close(destroy=True)


def _spawn(robot_sim, xyz):
    robot_sim.reset(robot_translation=np.asarray(xyz, dtype=np.float32),
                    snap_to_navmesh=False, vertical_offset=0.0)


def _navmesh_point(robot_sim):
    pf = robot_sim.sim.pathfinder
    assert pf.is_loaded
    return np.asarray(pf.get_random_navigable_point(), dtype=np.float32)


def _drive(robot_sim, monitor, ticks=TICKS):
    trace = []
    for _ in range(ticks):
        robot_sim.step(chassis_twist=np.array([0.5, 0.0, 0.0]),
                       twist_frame="robot", dt=1.0 / FPS)
        monitor.observe()
        trace.append((robot_height_m(robot_sim), vertical_velocity_mps(robot_sim),
                      float(robot_sim.robot.translation[1])))
    return np.array(trace)


def test_robot_on_the_navmesh_does_not_fall(robot_sim):
    _spawn(robot_sim, _navmesh_point(robot_sim))
    monitor = EpisodeHealthMonitor(robot_sim, fps=FPS)
    trace = _drive(robot_sim, monitor)
    health = monitor.finish()
    assert not health.escaped
    assert health.fall_run_ticks == 0
    # The floor pins the unactuated joint: no sustained descent at all.
    assert health.max_fall_speed_mps < DriveFailureConfig().fall_speed_mps
    assert trace[10:, 1].min() > -0.5      # past the spawn settle


def test_robot_instantiated_outside_the_world_free_falls(robot_sim):
    """The signal, provoked on demand: spawn 30 m below the scene so nothing can
    ever catch the robot, and watch joint_z accelerate under gravity."""
    _spawn(robot_sim, _navmesh_point(robot_sim) - np.array([0.0, 30.0, 0.0], np.float32))
    monitor = EpisodeHealthMonitor(robot_sim, fps=FPS)
    trace = _drive(robot_sim, monitor)
    health = monitor.finish()

    assert health.escaped
    assert health.outcome == "escaped"
    assert health.fall_run_ticks >= DriveFailureConfig().fall_duration_s * FPS
    assert health.max_fall_speed_mps > 10.0

    # Free fall, not merely "moving down": the velocity ramps at about -g.
    # Only over the first half-second -- the descent does taper off later (the
    # URDF's joint_z still carries a habitat-created velocity motor, so the fall
    # levels out near -14.5 m/s rather than accelerating forever), and it is the
    # onset that identifies the failure.
    vz = trace[:, 1]
    accel = np.diff(vz[:13]) * FPS
    assert np.median(accel) == pytest.approx(-9.81, abs=1.5)
    # ... and the speed keeps growing well past anything a supported robot sees.
    assert vz[-1] < vz[12] < vz[4] < -1.0


def test_the_base_link_translation_shows_nothing(robot_sim):
    """Why the failure was invisible. ``robot.translation`` is pinned at spawn,
    so it is identical for a supported robot and a falling one -- the vertical
    only exists in ``joint_z``."""
    p = _navmesh_point(robot_sim)

    _spawn(robot_sim, p)
    on_mesh = _drive(robot_sim, EpisodeHealthMonitor(robot_sim, fps=FPS))
    _spawn(robot_sim, p - np.array([0.0, 30.0, 0.0], np.float32))
    outside = _drive(robot_sim, EpisodeHealthMonitor(robot_sim, fps=FPS))

    assert np.ptp(on_mesh[:, 2]) == pytest.approx(0.0, abs=1e-9)
    assert np.ptp(outside[:, 2]) == pytest.approx(0.0, abs=1e-9)
    # ... while the real heights differ by the whole fall.
    assert (on_mesh[-1, 0] - on_mesh[0, 0]) == pytest.approx(0.0, abs=0.2)
    assert (outside[0, 0] - outside[-1, 0]) > 10.0
