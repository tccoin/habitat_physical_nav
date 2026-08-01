"""Sim-free checks for the ObjectNav task layer.

Everything here runs without a simulator instance: the measures are driven
through a fake adapter whose ``geodesic_distance`` is a straight line, and the
frame conversions are fed a stub robot. What can only be checked empirically
(that the metrics on a real navmesh are sane) lives in
``scripts/eval_objectnav.py`` instead -- see the repo's testing note in
CLAUDE.md.

Skips cleanly where habitat-lab / habitat_sim aren't installed.
"""

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

objectnav = pytest.importorskip(
    "continuous_demos.objectnav",
    reason="continuous_demos.objectnav requires habitat-lab and habitat_sim",
)
objectnav_dataset = pytest.importorskip(
    "continuous_demos.objectnav_dataset",
    reason="requires habitat-lab",
)

from continuous_demos.objectnav import (  # noqa: E402
    ObjectNavConfig,
    OracleSPL,
    OracleSuccess,
    PathTracker,
    _NavAdapterBase,
    _TrajectoryScorer,
    NavSimAdapter,
    planar_heading_to_habitat_quat,
    planar_to_habitat_position,
)

DATASET = Path(__file__).resolve().parents[1] / "examples" / "17DRP5sb8fy.json.gz"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StraightLineAdapter(_NavAdapterBase):
    """Adapter with a Euclidean stand-in for geodesic distance."""

    def __init__(self):
        super().__init__(pathfinder=None, snap_agent_position=False)
        self.pose = np.zeros(3)

    def set_pose(self, pose2d: Sequence[float]) -> None:
        self.pose = np.asarray(pose2d, dtype=float)

    def _planar_pose(self):
        return self.pose

    def _height(self):
        return 0.0

    def geodesic_distance(self, position_a, position_b, episode=None) -> float:
        ends = np.atleast_2d(np.asarray(position_b, dtype=float))
        return float(np.min(np.linalg.norm(ends - np.asarray(position_a), axis=1)))


class FakeGoal:
    def __init__(self, position):
        self.position = position
        self.view_points = []


class FakeEpisode:
    def __init__(self, goal_position):
        self.episode_id = "fake"
        self.scene_id = "fake.glb"
        self.goals = [FakeGoal(goal_position)]
        self._shortest_path_cache = None


def drive(scorer, adapter, episode, poses, stop_at_end=False):
    """Feed a planar trajectory through a scorer, evaluating at every sample."""
    adapter.set_pose(poses[0])
    scorer.path_tracker.reset(poses[0])
    scorer.measurements.reset_measures(episode=episode, task=scorer)
    for index, pose in enumerate(poses[1:], start=1):
        adapter.set_pose(pose)
        scorer.path_tracker.observe(pose)
        if stop_at_end and index == len(poses) - 1:
            scorer.is_stop_called = True
        scorer.measurements.update_measures(episode=episode, task=scorer)
    return dict(scorer.measurements.get_metrics())


def make_scorer(**config_kwargs):
    config = ObjectNavConfig(distance_to="POINT", **config_kwargs)
    adapter = StraightLineAdapter()
    return _TrajectoryScorer(adapter, config), adapter


# ---------------------------------------------------------------------------
# PathTracker
# ---------------------------------------------------------------------------


class TestPathTracker:
    def test_accumulates_euclidean_length(self):
        tracker = PathTracker()
        tracker.reset([0.0, 0.0, 0.0])
        tracker.observe([3.0, 0.0, 0.0])
        tracker.observe([3.0, 4.0, 0.0])
        assert tracker.length == pytest.approx(7.0)
        assert tracker.num_samples == 3

    def test_heading_only_motion_adds_no_length(self):
        tracker = PathTracker()
        tracker.reset([1.0, 2.0, 0.0])
        tracker.observe([1.0, 2.0, math.pi])
        assert tracker.length == pytest.approx(0.0)

    def test_coarse_sampling_undercounts(self):
        """Documented property: sampling coarser than the motion cuts corners."""
        fine = PathTracker()
        fine.reset([0.0, 0.0, 0.0])
        for pose in ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]):
            fine.observe(pose)
        coarse = PathTracker()
        coarse.reset([0.0, 0.0, 0.0])
        coarse.observe([1.0, 1.0, 0.0])
        assert coarse.length < fine.length

    def test_reset_clears_state(self):
        tracker = PathTracker()
        tracker.reset([0.0, 0.0, 0.0])
        tracker.observe([5.0, 0.0, 0.0])
        tracker.reset([10.0, 10.0, 0.0])
        tracker.observe([10.0, 11.0, 0.0])
        assert tracker.length == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Frame conversions
# ---------------------------------------------------------------------------


class TestFrameConversions:
    def test_position_inverts_planar_mapping(self):
        # planar X = habitat_x, Y = -habitat_z (sim.py header).
        position = planar_to_habitat_position([1.5, -2.5, 0.3], height=0.07)
        assert position == pytest.approx([1.5, 0.07, 2.5])

    @pytest.mark.parametrize("theta_deg", [0.0, 45.0, 90.0, -120.0, 179.0])
    def test_rotation_maps_habitat_front_onto_planar_heading(self, theta_deg):
        import quaternion  # noqa: F401  (registers np.quaternion)

        theta = math.radians(theta_deg)
        quat = planar_heading_to_habitat_quat(theta)
        rotation = quaternion.as_rotation_matrix(quat)
        forward = rotation @ np.array([0.0, 0.0, -1.0])  # habitat FRONT
        # Same direction expressed in the planar frame: (X, Y) = (fx, -fz).
        assert forward[0] == pytest.approx(math.cos(theta), abs=1e-6)
        assert -forward[2] == pytest.approx(math.sin(theta), abs=1e-6)
        assert forward[1] == pytest.approx(0.0, abs=1e-6)


class TestNavSimAdapter:
    """get_agent_state must undo both the planar mapping and the URDF lift."""

    class _StubRobotSim:
        def __init__(self, pose2d, translation_y, bb_min_y):
            self._pose = np.asarray(pose2d, dtype=float)

            class _BB:
                min = type("V", (), {"y": bb_min_y})()

            class _Node:
                cumulative_bb = _BB()

            class _Robot:
                root_scene_node = _Node()
                translation = [0.0, translation_y, 0.0]

            class _Pathfinder:
                is_loaded = False

            class _Sim:
                pathfinder = _Pathfinder()

            self.robot = _Robot()
            self.sim = _Sim()

        def get_2d_pose(self):
            return self._pose

    def test_removes_vertical_mounting_offset(self):
        # reset() lifts the robot by -bb.min.y; get_agent_state must undo it.
        robot_sim = self._StubRobotSim([2.0, -3.0, 0.0], translation_y=0.5, bb_min_y=-0.4)
        adapter = NavSimAdapter(robot_sim, snap_agent_position=False)
        state = adapter.get_agent_state()
        assert state.position == pytest.approx([2.0, 0.1, 3.0], abs=1e-6)

    def test_no_offset_when_model_sits_above_origin(self):
        robot_sim = self._StubRobotSim([0.0, 0.0, 0.0], translation_y=1.0, bb_min_y=0.2)
        adapter = NavSimAdapter(robot_sim, snap_agent_position=False)
        assert adapter.get_agent_state().position[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


class TestMeasures:
    def test_success_requires_stop_called(self):
        scorer, adapter = make_scorer(success_distance=0.2)
        episode = FakeEpisode([10.0, 0.0, 0.0])
        poses = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]

        metrics = drive(scorer, adapter, episode, poses, stop_at_end=False)
        assert metrics["success"] == 0.0
        assert metrics["spl"] == 0.0
        # The oracle stops for you: reaching the goal is enough.
        assert metrics["oracle_success"] == 1.0
        assert metrics["oracle_spl"] == pytest.approx(1.0)

    def test_success_and_optimal_spl_on_straight_line(self):
        scorer, adapter = make_scorer(success_distance=0.2)
        episode = FakeEpisode([10.0, 0.0, 0.0])
        poses = [[float(x), 0.0, 0.0] for x in range(11)]

        metrics = drive(scorer, adapter, episode, poses, stop_at_end=True)
        assert metrics["success"] == 1.0
        assert metrics["spl"] == pytest.approx(1.0)
        assert metrics["soft_spl"] == pytest.approx(1.0)

    def test_spl_penalizes_a_detour(self):
        scorer, adapter = make_scorer(success_distance=0.2)
        episode = FakeEpisode([10.0, 0.0, 0.0])
        # 10 m of optimal path walked as a 20 m dog-leg.
        poses = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [10.0, 5.0, 0.0], [10.0, 0.0, 0.0]]

        metrics = drive(scorer, adapter, episode, poses, stop_at_end=True)
        assert scorer.path_tracker.length == pytest.approx(20.0)
        assert metrics["success"] == 1.0
        assert metrics["spl"] == pytest.approx(10.0 / 20.0)

    def test_spl_reads_path_length_from_the_tracker(self):
        """The decoupling that matters: SPL must not accumulate on its own."""
        scorer, adapter = make_scorer(success_distance=0.2)
        episode = FakeEpisode([10.0, 0.0, 0.0])
        poses = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]

        adapter.set_pose(poses[0])
        scorer.path_tracker.reset(poses[0])
        scorer.measurements.reset_measures(episode=episode, task=scorer)
        adapter.set_pose(poses[1])
        scorer.is_stop_called = True
        # Path length tracked at a finer cadence than the measure update: SPL
        # sees the tracker's 40 m, not the 10 m it would have accumulated itself.
        scorer.path_tracker.reset([0.0, 0.0, 0.0])
        for pose in ([0.0, 20.0, 0.0], [10.0, 20.0, 0.0], [10.0, 0.0, 0.0]):
            scorer.path_tracker.observe(pose)
        scorer.measurements.update_measures(episode=episode, task=scorer)

        metrics = dict(scorer.measurements.get_metrics())
        assert metrics["spl"] == pytest.approx(10.0 / 50.0)

    def test_oracle_spl_freezes_path_length_at_first_success(self):
        scorer, adapter = make_scorer(success_distance=0.2)
        episode = FakeEpisode([10.0, 0.0, 0.0])
        # Reach the goal optimally, then wander off for another 20 m.
        poses = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 20.0, 0.0]]

        metrics = drive(scorer, adapter, episode, poses, stop_at_end=True)
        # Agent stopped 20 m away -> plain success fails, oracle keeps its 1.0.
        assert metrics["success"] == 0.0
        assert metrics["oracle_success"] == 1.0
        assert metrics["oracle_spl"] == pytest.approx(1.0)

    def test_soft_spl_rewards_partial_progress(self):
        scorer, adapter = make_scorer(success_distance=0.2)
        episode = FakeEpisode([10.0, 0.0, 0.0])
        poses = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]

        metrics = drive(scorer, adapter, episode, poses, stop_at_end=True)
        assert metrics["success"] == 0.0
        assert metrics["soft_spl"] == pytest.approx(0.5)

    def test_measure_uuids(self):
        assert OracleSuccess.cls_uuid == "oracle_success"
        assert OracleSPL.cls_uuid == "oracle_spl"


class TestSceneSwitching:
    """`_prepare_scene` decides whether reset() must load a different scene.

    Whether the *simulator* really switches is empirical -- see
    `scripts/sim_debug_drive.py::check_scene_switch`.
    """

    class _StubRobotSim:
        def __init__(self, scene_id):
            self.cfg = type(
                "Cfg", (), {"sim_cfg": type("S", (), {"scene_id": scene_id})()}
            )()
            # pathfinder is resolved lazily by NavSimAdapter, so a stub sim
            # never needs one until a geodesic query happens.
            self.sim = None

    @staticmethod
    def _episode(scene_id):
        episode = FakeEpisode([1.0, 0.0, 0.0])
        episode.scene_id = scene_id
        return episode

    def _task(self, scene_id, **config_kwargs):
        from continuous_demos.objectnav import ObjectNavTask

        return ObjectNavTask(
            self._StubRobotSim(scene_id),
            ObjectNavConfig(**config_kwargs),
            scene_id=scene_id,
        )

    def test_same_scene_needs_no_switch(self):
        task = self._task("mp3d/a/a.glb")
        assert task._prepare_scene(self._episode("mp3d/a/a.glb")) == {}

    def test_switch_resolves_scene_path(self, tmp_path):
        scene = tmp_path / "mp3d" / "b" / "b.glb"
        scene.parent.mkdir(parents=True)
        scene.write_bytes(b"")
        task = self._task("mp3d/a/a.glb", scene_root=str(tmp_path))

        kwargs = task._prepare_scene(self._episode("mp3d/b/b.glb"))
        assert kwargs["scene_id"] == str(scene)
        assert task.scene_id == "mp3d/b/b.glb"

    def test_strict_mode_raises_on_mismatch(self):
        task = self._task("mp3d/a/a.glb", auto_switch_scene=False)
        with pytest.raises(ValueError, match="auto_switch_scene is off"):
            task._prepare_scene(self._episode("mp3d/b/b.glb"))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DATASET.exists(), reason="bundled example dataset missing")
class TestObjectNavReplayDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        return objectnav_dataset.load_objectnav_dataset(str(DATASET))

    def test_loads_every_episode(self, dataset):
        assert len(dataset.episodes) == 1141

    def test_preserves_original_episode_id(self, dataset):
        # habitat's from_json overwrites episode_id with str(index); the
        # pipeline resumes by the real id, so this must survive.
        episode = dataset.episodes[0]
        assert episode.episode_id.startswith("A16K7X677N4WN6:")
        assert episode.episode_index == 0

    def test_keeps_reference_replay(self, dataset):
        assert len(dataset.episodes[0].reference_replay) == 895

    def test_goals_are_habitat_objects(self, dataset):
        from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation

        goal = dataset.episodes[0].goals[0]
        assert isinstance(goal, ObjectGoal)
        assert isinstance(goal.view_points[0], ObjectViewLocation)
        assert len(goal.view_points[0].agent_state.position) == 3

    def test_habitat_dataset_helpers_work(self, dataset):
        assert dataset.scene_ids == ["mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"]
        assert len(dataset.get_scene_episodes(dataset.scene_ids[0])) == 1141

    def test_unknown_keys_go_to_extra(self):
        episode = objectnav_dataset.ObjectNavReplayDataset._make_episode(
            {
                "episode_id": "abc",
                "scene_id": "s.glb",
                "start_position": [0, 0, 0],
                "start_rotation": [0, 0, 0, 1],
                "is_thda": True,
                "some_future_field": 7,
            },
            index=3,
        )
        assert episode.episode_id == "abc"
        assert episode.episode_index == 3
        assert episode.is_thda is True
        assert episode.extra == {"some_future_field": 7}

    def test_fallback_matches_habitat_surface(self, dataset):
        raw = objectnav_dataset.load_raw_episodes(str(DATASET))
        assert len(raw) == len(dataset.episodes)
        for attribute in ("episode_id", "scene_id", "object_category"):
            assert getattr(raw[0], attribute) == getattr(dataset.episodes[0], attribute)
        assert raw[0].goals[0].position == dataset.episodes[0].goals[0].position
        assert raw[0].goals[0].view_points[0].agent_state.position == (
            dataset.episodes[0].goals[0].view_points[0].agent_state.position
        )
        assert raw[0]._shortest_path_cache is None

    def test_episodes_by_scene(self, dataset):
        grouped = objectnav_dataset.episodes_by_scene(dataset.episodes)
        assert list(grouped) == dataset.scene_ids
        assert len(grouped[dataset.scene_ids[0]]) == 1141
