"""Regression tests for episode-to-episode state leakage through ``reset()``.

These are the tests that exposed the defect. A ``HabitatRobotSim`` is reused
across episodes -- which is what ``generate_pose2d_dataset.py`` and the
evaluation harnesses do, one simulator per scene -- and the SAME episode is
run at several different points in that simulator's history. The commands are
a fixed, stateless, image-blind open-loop sequence, so a stateless policy
cannot explain any difference between the runs: whatever differs is simulator
state carried across ``reset()``.

What the fix guarantees, and what it does not, is split across three tests
deliberately:

  * ``test_reset_leaves_no_duplicate_articulated_object`` and
    ``test_spawn_state_is_a_function_of_the_spawn_pose`` are the contract
    ``reset()`` can actually keep, and they are strict. The second one is the
    strong statement: every quantity Bullet will read at the start of the
    episode -- joint state, root transform, the broadphase pair count and the
    whole ordered contact-point list -- is identical no matter what ran before.

  * ``test_trajectory_is_independent_of_execution_order`` is the property we
    want but cannot guarantee, and it is a non-strict xfail. What is left
    lives below habitat's Python API: the order in which Bullet's broadphase
    hands the solver the robot's contacts depends on btDbvtBroadphase's
    internal state, which no binding can reset. It reproduces most of the time
    -- 6 of 8 episodes bit-identical between a forward and a reverse batch,
    against 1 of 8 before the fix -- so this test is left live as a probe
    rather than deleted. See dump/sim_reset_fix/FINDINGS.md.

Unlike the rest of ``tests/``, these need a live simulator and MP3D scene
assets on disk, so they skip cleanly when either is missing -- the suite still
runs anywhere. Point them at scenes with ``HABITAT_SCENE_ROOT`` or the usual
``data/scene_datasets`` layout.
"""

import json
import math

import numpy as np
import pytest

pytest.importorskip("habitat_sim", reason="needs a live simulator")
pytest.importorskip("magnum", reason="needs a live simulator")

from continuous_demos.episode_io import (  # noqa: E402
    DEFAULT_URDF_PATH,
    build_robot_sim,
    find_scene_dataset_config,
    resolve_scene_path,
)

SCENE_ID = "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
DT = 0.04           # 25 Hz, the on-grid recording tick (see replay.RECORD_FPS)
TICKS = 150
IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def robot_sim():
    try:
        scene_path = resolve_scene_path(SCENE_ID, None)
    except FileNotFoundError as exc:
        pytest.skip(f"MP3D scene assets not available: {exc}")
    sim = build_robot_sim(
        scene_path=scene_path,
        scene_dataset_config=find_scene_dataset_config(scene_path, None),
        width=64,
        height=64,
        urdf_path=str(DEFAULT_URDF_PATH),
    )
    yield sim
    sim.sim.close()


@pytest.fixture(scope="module")
def spawn_points(robot_sim):
    pathfinder = robot_sim.sim.pathfinder
    pathfinder.seed(7)
    spawn = list(map(float, pathfinder.get_random_navigable_point()))
    elsewhere = list(map(float, pathfinder.get_random_navigable_point()))
    return spawn, elsewhere


def _drive_forward(n):
    """A stateless open-loop episode: hold 1 m/s forward for n ticks."""
    return [[1.0, 0.0, 0.0]] * n


def _wander(n, seed=0):
    """A disturbance episode -- different path, different contacts."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(0.02, 0.12, 3)
    return [
        [1.5 * math.cos(a * i), 0.8 * math.sin(b * i), 1.2 * math.sin(c * i)]
        for i in range(n)
    ]


def _reset(sim, spawn_xyz):
    sim.reset(
        robot_translation=np.array(spawn_xyz, dtype=np.float64),
        robot_rotation=np.array(IDENTITY_QUAT, dtype=np.float64),
        snap_to_navmesh=False,
        vertical_offset=None,
    )


def _run(sim, spawn_xyz, twists):
    _reset(sim, spawn_xyz)
    poses = []
    for twist in twists:
        sim.step(np.array(twist, dtype=np.float64), dt=DT)
        poses.append(sim.get_2d_pose())
    return np.array(poses)


def _spawn_state(sim):
    """Everything the first physics step of the episode will read."""
    return json.dumps(
        {
            "joint_positions": [float(x) for x in sim.robot.joint_positions],
            "joint_velocities": [float(x) for x in sim.robot.joint_velocities],
            "translation": [float(x) for x in sim.robot.translation],
            "overlapping_pairs": int(sim.sim.get_physics_num_active_overlapping_pairs()),
            "contacts": [
                [
                    int(p.object_id_a), int(p.object_id_b),
                    int(p.link_id_a), int(p.link_id_b),
                    [float(x) for x in p.position_on_b_in_ws],
                    [float(x) for x in p.contact_normal_on_b_in_ws],
                    float(p.contact_distance),
                ]
                for p in sim.sim.get_physics_contact_points()
            ],
        },
        sort_keys=True,
    )


def test_reset_leaves_no_duplicate_articulated_object(robot_sim, spawn_points):
    """reset() must reuse the robot, not instantiate another one.

    __init__ used to blank self.robot right after building it, so the first
    reset() added a second robot from the URDF and abandoned the first in the
    physics world as a dynamic multibody at the origin, permanently in contact
    with the scene and sharing the solver with the real robot.
    """
    spawn, _ = spawn_points
    aom = robot_sim.sim.get_articulated_object_manager()
    for _ in range(3):
        _run(robot_sim, spawn, _drive_forward(1))
        assert aom.get_num_objects() == 1, list(aom.get_object_handles())


def test_spawn_state_is_a_function_of_the_spawn_pose(robot_sim, spawn_points):
    """Nothing the first physics step reads may depend on the previous episode.

    Before the fix this failed on the collision state: habitat's sim.reset()
    restores joint positions and the root transform but leaves Bullet's
    broadphase pairs and persistent contact manifolds in place -- 6 overlapping
    pairs and 2 active contact points at the end of an episode were still
    there, untouched, after sim.reset().
    """
    spawn, elsewhere = spawn_points
    reference = None
    for history_ticks in (0, 50, 97, 211, 303):
        if history_ticks:
            _run(robot_sim, elsewhere, _wander(history_ticks, seed=history_ticks))
        _reset(robot_sim, spawn)
        state = _spawn_state(robot_sim)
        if reference is None:
            reference = state
        else:
            assert state == reference, (
                f"the spawn state changed after a {history_ticks}-tick episode; "
                "reset() is carrying physics state across episodes"
            )


@pytest.mark.xfail(
    strict=False,
    reason="residual contact-ordering non-determinism inside btDbvtBroadphase; "
           "no habitat binding can reset it (see dump/sim_reset_fix/FINDINGS.md)",
)
def test_trajectory_is_independent_of_execution_order(robot_sim, spawn_points):
    """The headline property: same episode, same result, whatever preceded it."""
    spawn, elsewhere = spawn_points
    episode = _drive_forward(TICKS)

    alone = _run(robot_sim, spawn, episode)
    # The episode must have actually gone somewhere, or this proves nothing.
    assert np.linalg.norm(alone[-1, :2] - alone[0, :2]) > 0.5

    after_itself = _run(robot_sim, spawn, episode)
    _run(robot_sim, elsewhere, _wander(TICKS, seed=1))
    after_wander = _run(robot_sim, spawn, episode)
    _run(robot_sim, elsewhere, _wander(TICKS // 2, seed=2))
    after_short_wander = _run(robot_sim, spawn, episode)

    for label, poses in [
        ("immediate repeat", after_itself),
        ("after a wander episode", after_wander),
        ("after a shorter wander episode", after_short_wander),
    ]:
        drift = np.linalg.norm(alone[:, :2] - poses[:, :2], axis=1).max()
        assert np.array_equal(alone, poses), (
            f"episode outcome changed {label}: max {drift:.6f} m from the "
            "run-alone trajectory"
        )
