"""The debug overlays must never reach the policy's observation.

Every rollout renders one frame per tick and uses it TWICE: once for the recorded video,
with the planned path, the goal marker and the commanded chunk drawn on it, and once as
the policy's observation. In both `scripts/pointnav_rollout.py` and
`objectnav_eval/runner.py` the video composition happens **first**, on the same array
object that is handed to `policy.act` a few lines later:

    rgb = render()
    recorder.add(compose(robot_sim, rgb, state))     # <- draws path, goal, chunk
    chunk = policy.act(Observation(rgb=rgb, ...))    # <- same object

So the only thing preventing the policy from seeing the answer drawn on its own input is
that every draw function copies before it strokes. That is a convention, and cv2's
drawing primitives mutate their first argument in place, so a single
``cv2.line(rgb, ...)`` added to a future overlay would reintroduce the leak.

It would be an unusually nasty leak. A policy shown the planned path to its goal would
*improve*, so the symptom is a better number, not a crash -- and the frames would also be
out of distribution relative to training, which never contained an overlay. Nothing else
in the harness would notice.

These tests pin the invariant at the level that matters: the bytes of the array the policy
receives, before and after the overlay code has run over it.
"""

import hashlib

import numpy as np
import pytest

from continuous_demos.viewport_overlay import (
    chunk_to_world_xy,
    draw_goal_in_view,
    draw_path_in_view,
)


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()


@pytest.fixture
def frame():
    """A noisy frame, so any stroke anywhere changes the hash."""
    return np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)


@pytest.fixture
def camera():
    """A camera 1.1 m up looking down -z, matching the shipped rig."""
    cam = np.eye(4)
    cam[1, 3] = 1.1
    return cam, 79.0


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------
def test_draw_path_does_not_touch_its_input(frame, camera):
    cam, hfov = camera
    path = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -4.0], [1.0, 0.0, -6.0]])
    before = digest(frame)
    out = draw_path_in_view(frame, path, cam, hfov)
    assert digest(frame) == before, "draw_path_in_view mutated the frame in place"
    assert out is not frame


def test_draw_goal_does_not_touch_its_input(frame, camera):
    cam, hfov = camera
    before = digest(frame)
    out = draw_goal_in_view(frame, np.array([0.0, 0.0, -3.0]), cam, hfov)
    assert digest(frame) == before, "draw_goal_in_view mutated the frame in place"
    assert out is not frame


def test_the_overlays_actually_draw_something(frame, camera):
    """Guards the guard: a no-op overlay would pass every test above trivially."""
    cam, hfov = camera
    path = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, -6.0]])
    assert not np.array_equal(draw_path_in_view(frame, path, cam, hfov), frame)
    assert not np.array_equal(
        draw_goal_in_view(frame, np.array([0.0, 0.0, -3.0]), cam, hfov), frame
    )


def test_chained_overlays_still_leave_the_original_alone(frame, camera):
    """The real call pattern: `rgb = f(rgb)` several times over."""
    cam, hfov = camera
    before = digest(frame)
    rgb = frame
    rgb = draw_path_in_view(rgb, np.array([[0.0, 0.0, -2.0], [0.0, 0.0, -7.0]]), cam, hfov)
    rgb = draw_goal_in_view(rgb, np.array([0.0, 0.0, -7.0]), cam, hfov)
    rgb = draw_path_in_view(rgb, np.array([[0.0, 0.0, -2.0], [0.5, 0.0, -3.0]]), cam, hfov)
    assert digest(frame) == before
    assert not np.array_equal(rgb, frame)


# ---------------------------------------------------------------------------
# The eval harness's overlay hook
# ---------------------------------------------------------------------------
def test_runner_overlay_returns_the_same_object_when_disabled(frame):
    """Default off must be a true no-op, not a copy: existing videos are unchanged."""
    from objectnav_eval.config import EvalConfig
    from objectnav_eval.runner import EpisodeRunner

    runner = EpisodeRunner.__new__(EpisodeRunner)
    config = EvalConfig()
    assert config.overlay_chunk is False
    assert runner._overlay(config, None, frame) is frame


def test_runner_overlay_does_not_touch_its_input_when_enabled(frame):
    """With no chunk emitted yet it must still hand the frame straight back."""
    from objectnav_eval.config import EvalConfig
    from objectnav_eval.runner import EpisodeRunner

    runner = EpisodeRunner.__new__(EpisodeRunner)
    config = EvalConfig()
    config.overlay_chunk = True
    before = digest(frame)
    assert runner._overlay(config, None, frame) is frame
    assert digest(frame) == before


# ---------------------------------------------------------------------------
# The composer, against a live simulator -- the path a real rollout takes
# ---------------------------------------------------------------------------
def test_frame_composer_leaves_the_policys_observation_untouched():
    """The end-to-end invariant, with every overlay active on a real render.

    Skipped without a simulator and scene assets; the primitive tests above cover the
    same property without them. When it does run, this is the one that would catch a
    leak introduced anywhere in the composition chain, including the top-down panel.
    """
    pytest.importorskip("habitat_sim")
    import importlib.util
    import os
    from pathlib import Path

    root = os.environ.get("HABITAT_SCENE_ROOT", "/Projects/data/scene_datasets")
    scene = Path(root) / "hm3d/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb"
    if not scene.exists():
        pytest.skip(f"scene assets not present under {root}")

    from continuous_demos.episode_io import (
        DEFAULT_URDF_PATH,
        build_robot_sim,
        find_scene_dataset_config,
        resolve_scene_path,
    )
    from continuous_demos.robot_pose import robot_navmesh_position

    spec = importlib.util.spec_from_file_location(
        "pointnav_rollout",
        Path(__file__).resolve().parents[1] / "scripts" / "pointnav_rollout.py",
    )
    rollout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rollout)

    scene_path = resolve_scene_path(str(scene), root)
    sim = build_robot_sim(
        scene_path=scene_path,
        scene_dataset_config=find_scene_dataset_config(scene_path, root),
        width=160, height=120, urdf_path=str(DEFAULT_URDF_PATH),
    )
    try:
        pathfinder = sim.sim.pathfinder
        pathfinder.seed(0)
        sim.reset(
            robot_translation=np.asarray(
                pathfinder.get_random_navigable_point(), dtype=np.float32
            ).copy(),
            snap_to_navmesh=False, vertical_offset=None,
        )
        rgb = np.asarray(sim.get_obs()["color_sensor"])[..., :3]
        before = digest(rgb)

        here = robot_navmesh_position(sim)
        path_3d = np.array(
            [here + np.array([0, 0, -i * 0.5], dtype=np.float32) for i in range(10)],
            dtype=np.float32,
        )
        composer = rollout.FrameComposer("both", goal_radius_m=0.5)
        composer.last_chunk = (sim.get_2d_pose(),
                               np.tile(np.array([[0.1, 0.0, 0.02]]), (20, 1)))
        composer.executed = [[0.0, 0.0], [0.1, 0.1]]
        state = {
            "pose2d": sim.get_2d_pose(),
            "goal2d": [float(here[0]), float(-here[2]) - 3.0, 0.0],
            "planned_path": np.stack([path_3d[:, 0], -path_3d[:, 2]], axis=1),
            "path_3d": path_3d, "goal_3d": path_3d[-1],
            "step": 0, "tick": 0, "sim_time": 0.0, "goal_index": 0,
            "geodesic_m": 3.0, "reached": False, "is_observation": True, "note": "",
        }
        composed = composer.compose(sim, rgb, state)

        assert digest(rgb) == before, (
            "the frame handed to policy.act was modified by the video overlay -- the "
            "policy would see its own planned path drawn on its observation"
        )
        assert composed.shape != rgb.shape or not np.array_equal(composed, rgb), (
            "the overlay drew nothing, so this test proves nothing"
        )
    finally:
        sim.sim.close(destroy=True)


def test_chunk_to_world_xy_does_not_touch_its_inputs():
    anchor = np.array([1.0, 2.0, 0.5])
    chunk = np.array([[0.1, 0.0, 0.0], [0.2, 0.1, 0.05]])
    a_before, c_before = digest(anchor), digest(chunk)
    chunk_to_world_xy(anchor, chunk)
    assert digest(anchor) == a_before and digest(chunk) == c_before
