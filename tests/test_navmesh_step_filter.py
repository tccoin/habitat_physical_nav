import numpy as np

from continuous_demos.sim import clip_navmesh_world_velocity


class _ClippedPathfinder:
    is_loaded = True

    def try_step(self, start, target):
        return np.array([min(target[0], 0.25), target[1], min(target[2], 0.5)])


def test_navmesh_filter_clips_planar_velocity_at_boundary():
    vx, vy, blocked, rejected_m = clip_navmesh_world_velocity(
        _ClippedPathfinder(), np.array([0.0, 1.0, 0.0]), 2.0, -3.0, 0.5
    )

    assert blocked
    assert vx == 0.5
    assert vy == -1.0
    assert rejected_m == np.hypot(0.75, 1.0)


def test_navmesh_filter_is_inert_without_a_loaded_mesh():
    vx, vy, blocked, rejected_m = clip_navmesh_world_velocity(
        None, np.array([0.0, 1.0, 0.0]), 2.0, -3.0, 0.5
    )

    assert (vx, vy, blocked, rejected_m) == (2.0, -3.0, False, 0.0)
