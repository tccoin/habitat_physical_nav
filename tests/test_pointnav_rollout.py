"""The PointNav rollout's pure logic: goal sources, the goal-pursuit control, the
point-goal episode, and -- the load-bearing one -- the order the ``<pose>`` value rows
reach the model in.

No simulator anywhere in this file. What each group is defending:

* **Row order.** ``bridge._PoseFeed`` used to feed one row per turn and take the last row
  of the accumulated column. That is right only because the column held nothing but agent
  poses; interleave a goal and "the last row" stops meaning "the agent's pose" while
  staying a perfectly well-formed three-vector. The order now comes from
  ``longnav.utils.vector_rollout.PointNavValueStream``, which is the incremental form of
  the same ``modality_value_rows`` the training formatter calls -- and this file asserts
  that the *bridge*, driven turn by turn, reproduces that batch function row for row. The
  model-side half of the same claim is in
  ``/Projects/spatial_training/tests/test_pose_injection.py``.
* **Goal sources.** The seam a GUI would drive. Its contract is one method returning
  ``[X, Y, theta]`` or ``None``, and the two things easy to get wrong from a UI (the
  frame, and a constant heading) are cheap to pin here.
* **The control.** ``pursuit_chunk`` is what proves the plumbing without a checkpoint, so
  it has to be right about the one thing controls get wrong: rows are cumulative offsets
  from the anchor, not per-tick deltas.

    pytest tests/test_pointnav_rollout.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

from objectnav_eval.config import _parse_source_set
from objectnav_eval.goal_sources import (
    NavmeshGoalSource,
    QueueGoalSource,
    ScriptedGoalSource,
    available_goal_sources,
    build_goal_source,
    heading_toward,
    planar_distance,
)
from objectnav_eval.pointnav import (
    PointGoalPursuitPolicy,
    build_point_episode,
    pursuit_chunk,
    to_eval_episode,
)
from objectnav_eval.policies import Observation

# The checkpoint's own code lives in the training checkout, which this environment need
# not have installed -- the same reason `--policy-sys-path` exists.
_LONGNAV_SRC = os.environ.get("LONGNAV_SRC", "/Projects/spatial_training/src")
if os.path.isdir(_LONGNAV_SRC) and _LONGNAV_SRC not in sys.path:
    sys.path.insert(0, _LONGNAV_SRC)


# ===========================================================================
# Goal sources
# ===========================================================================
class TestScriptedGoalSource:
    def test_goals_are_issued_in_order_then_exhausted(self):
        source = ScriptedGoalSource([[1.0, 2.0, 0.5], [3.0, 4.0, -0.5]])
        pose = [0.0, 0.0, 0.0]
        assert source.next_goal(None, pose) == [1.0, 2.0, 0.5]
        assert source.next_goal(None, pose) == [3.0, 4.0, -0.5]
        # None, not a repeat and not an exception: exhaustion is how a route ends.
        assert source.next_goal(None, pose) is None

    def test_reset_rewinds(self):
        source = ScriptedGoalSource([[1.0, 2.0, 0.0]])
        source.next_goal(None, [0.0, 0.0, 0.0])
        source.reset()
        assert source.next_goal(None, [0.0, 0.0, 0.0]) == [1.0, 2.0, 0.0]

    def test_looping_never_exhausts(self):
        source = ScriptedGoalSource([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], loop=True)
        pose = [0.0, 0.0, 0.0]
        got = [source.next_goal(None, pose)[0] for _ in range(5)]
        assert got == [1.0, 2.0, 1.0, 2.0, 1.0]

    def test_a_two_element_goal_gets_a_real_heading_not_a_constant(self):
        """A constant heading survives `relative_se2` as the scene's compass at the window
        origin -- exactly the leak the frame conversion exists to remove. So a goal named
        without one is completed from the bearing, not from 0.0."""
        source = ScriptedGoalSource([[0.0, 5.0]])
        goal = source.next_goal(None, [0.0, 0.0, 1.234])
        assert len(goal) == 3
        assert goal[2] == pytest.approx(math.pi / 2)

    def test_a_malformed_goal_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match=r"\[X, Y\] or \[X, Y, theta\]"):
            ScriptedGoalSource([[1.0, 2.0, 3.0, 4.0]])


class TestQueueGoalSource:
    """The GUI seam. Nothing here knows what pushes; that is the point."""

    def test_a_pushed_goal_comes_back(self):
        source = QueueGoalSource()
        source.push([2.0, 3.0, 0.25])
        assert source.next_goal(None, [0.0, 0.0, 0.0]) == [2.0, 3.0, 0.25]

    def test_an_empty_queue_ends_the_episode_rather_than_blocking(self):
        """timeout 0 is the batch-job default: a rollout hung on an empty queue is a hung
        worker, which is worse than one that stops."""
        assert QueueGoalSource(timeout_s=0.0).next_goal(None, [0.0, 0.0, 0.0]) is None

    def test_an_empty_queue_falls_back_when_one_is_configured(self):
        fallback = ScriptedGoalSource([[9.0, 9.0, 0.0]])
        source = QueueGoalSource(timeout_s=0.0, fallback=fallback)
        assert source.next_goal(None, [0.0, 0.0, 0.0]) == [9.0, 9.0, 0.0]

    def test_a_pushed_none_ends_the_episode(self):
        source = QueueGoalSource(timeout_s=0.0)
        source.push(None)
        assert source.next_goal(None, [0.0, 0.0, 0.0]) is None

    def test_a_two_element_push_is_completed_with_a_bearing(self):
        source = QueueGoalSource(timeout_s=0.0)
        source.push([3.0, 0.0])
        assert source.next_goal(None, [0.0, 0.0, 2.0])[2] == pytest.approx(0.0)

    def test_reset_forwards_to_the_fallback(self):
        fallback = ScriptedGoalSource([[1.0, 0.0, 0.0]])
        source = QueueGoalSource(timeout_s=0.0, fallback=fallback)
        source.next_goal(None, [0.0, 0.0, 0.0])
        source.reset(None)
        assert source.next_goal(None, [0.0, 0.0, 0.0]) == [1.0, 0.0, 0.0]


class TestGoalGeometry:
    def test_heading_toward_is_the_planar_bearing(self):
        assert heading_toward([0, 0], [1, 0]) == pytest.approx(0.0)
        assert heading_toward([0, 0], [0, 1]) == pytest.approx(math.pi / 2)
        assert heading_toward([2, 2], [1, 2]) == pytest.approx(math.pi)

    def test_reaching_a_goal_is_a_question_about_position_only(self):
        """The demonstrations arrive on the path's final tangent, but a policy that
        arrives facing elsewhere has still arrived. Gating on heading would fail episodes
        the corpus itself calls successful."""
        assert planar_distance([1.0, 1.0, 0.0], [1.0, 1.0, math.pi]) == 0.0
        assert planar_distance([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) == pytest.approx(5.0)


class TestGoalSourceRegistry:
    def test_the_three_shipped_sources_are_registered(self):
        assert available_goal_sources() == ["navmesh", "queue", "scripted"]

    def test_build_resolves_by_name(self):
        assert isinstance(build_goal_source("scripted", goals=[[1.0, 1.0]]),
                          ScriptedGoalSource)

    def test_an_unknown_source_lists_what_exists(self):
        with pytest.raises(KeyError, match="unknown goal source"):
            build_goal_source("telepathy")

    def test_the_navmesh_sampler_reuses_the_corpus_screening_config(self):
        """Not a separate idea of what a goal is: a sampler with its own thresholds would
        evaluate the policy on a distribution it was never trained for."""
        from continuous_demos.pointnav import PointNavSampleConfig

        source = NavmeshGoalSource()
        assert isinstance(source.config, PointNavSampleConfig)
        assert source.config.min_geodesic_m == PointNavSampleConfig().min_geodesic_m


# ===========================================================================
# The goal-pursuit control
# ===========================================================================
class TestPursuitChunk:
    ANCHOR = [10.0, -4.0, 0.3]

    def test_shape_and_finiteness(self):
        chunk = pursuit_chunk(self.ANCHOR, [12.0, -4.0, 0.0], chunk_len=10, dt=0.05)
        assert chunk.shape == (10, 3)
        assert np.isfinite(chunk).all()

    def test_rows_are_cumulative_offsets_not_per_tick_deltas(self):
        """The mistake that makes a control crawl at one tick's displacement per chunk and
        look like a broken executor. Distance from the anchor must grow along the chunk."""
        chunk = pursuit_chunk(self.ANCHOR, [20.0, -4.0, 0.0], chunk_len=10, dt=0.05,
                              v_max=0.6)
        radii = np.linalg.norm(chunk[:, :2], axis=1)
        assert np.all(np.diff(radii) > 0)

    def test_it_turns_toward_a_goal_behind_it_before_translating(self):
        """Turn-then-drive: translating while badly misaligned is how a controller ends up
        circling a goal it could have walked to."""
        behind = [self.ANCHOR[0] - math.cos(0.3) * 5, self.ANCHOR[1] - math.sin(0.3) * 5,
                  0.0]
        chunk = pursuit_chunk(self.ANCHOR, behind, chunk_len=6, dt=0.05, v_max=0.6,
                              w_max=1.5)
        assert abs(chunk[0, 2]) > 0.0, "it must start rotating immediately"
        assert np.allclose(chunk[:3, :2], 0.0, atol=1e-9), (
            "and must not translate while pointing the wrong way"
        )

    def test_the_heading_rotates_the_short_way_round(self):
        """An unwrapped bearing error sends the robot 350 degrees round for a 10-degree
        correction, and the chunk is still finite and still executable."""
        anchor = [0.0, 0.0, math.pi - 0.05]
        goal = [-1.0, -0.2, 0.0]  # just across the +pi/-pi seam
        chunk = pursuit_chunk(anchor, goal, chunk_len=4, dt=0.05, w_max=1.5)
        assert abs(chunk[-1, 2]) < 0.5

    def test_it_does_not_command_a_fly_past(self):
        """A chunk that overshoots is faithfully executed and then has to be undone."""
        goal = [self.ANCHOR[0] + math.cos(0.3) * 0.10,
                self.ANCHOR[1] + math.sin(0.3) * 0.10, 0.3]
        chunk = pursuit_chunk(self.ANCHOR, goal, chunk_len=10, dt=0.05, v_max=2.0)
        travelled = np.linalg.norm(chunk[:, :2], axis=1)
        assert travelled.max() <= 0.10 + 1e-6

    def test_standing_on_the_goal_commands_nothing(self):
        chunk = pursuit_chunk(self.ANCHOR, self.ANCHOR, chunk_len=5, dt=0.05)
        assert np.allclose(chunk, 0.0)

    def test_a_long_chunk_converges_rather_than_orbiting(self):
        chunk = pursuit_chunk([0.0, 0.0, math.pi], [4.0, 0.0, 0.0], chunk_len=200,
                              dt=0.05, v_max=0.6, w_max=1.5)
        # In the anchor frame the goal is 4 m behind, i.e. at (-4, 0).
        final = np.linalg.norm(chunk[-1, :2] - np.array([-4.0, 0.0]))
        assert final < 0.1

    def test_an_empty_chunk_is_rejected(self):
        with pytest.raises(ValueError, match="chunk_len"):
            pursuit_chunk(self.ANCHOR, [1.0, 1.0, 0.0], chunk_len=0)


class TestPursuitPolicy:
    def test_it_reads_the_goal_and_nothing_else(self):
        policy = PointGoalPursuitPolicy(chunk_len=4, dt=0.05, speed=0.5)
        assert policy.uses_observation is False
        chunk = policy.act(Observation(
            rgb=np.zeros((2, 2, 3), np.uint8), goal=None, step=0,
            pose=np.array([0.0, 0.0, 0.0]), sim_time=0.0,
            goal_pose=np.array([2.0, 0.0, 0.0]),
        ))
        assert chunk.poses.shape == (4, 3)
        assert chunk.poses[-1, 0] > 0

    def test_without_a_goal_it_says_so_instead_of_standing_still(self):
        """A control that silently emitted zeros would be indistinguishable from `zero`,
        and the run would look like a policy failure rather than a misconfiguration."""
        policy = PointGoalPursuitPolicy()
        with pytest.raises(ValueError, match="goal_pose"):
            policy.act(Observation(rgb=np.zeros((2, 2, 3), np.uint8), goal=None, step=0,
                                   pose=np.zeros(3), sim_time=0.0))


# ===========================================================================
# The point-goal episode
# ===========================================================================
class TestPointEpisode:
    def test_the_planar_goal_becomes_habitat_world_exactly_once(self):
        """`X = habitat_x`, `Y = -habitat_z`. A sign error here is a goal that is a
        perfect mirror image of the intended one, and every metric stays plausible."""
        episode = build_point_episode(
            "scene.glb", start2d=[1.0, 2.0, 0.0], goal2d=[5.0, -3.0, 0.75],
            height_m=0.25,
        )
        np.testing.assert_allclose(episode.goals[0].position, [5.0, 0.25, 3.0])
        np.testing.assert_allclose(episode.start_position, [1.0, 0.25, -2.0])

    def test_the_view_point_is_the_goal_itself(self):
        """The degenerate one-view-point case, so `distance_to` VIEW_POINTS and POINT give
        the identical number and a run cannot be misconfigured into measuring something
        else."""
        episode = build_point_episode("s.glb", [0, 0, 0], [3.0, 4.0, 0.0], height_m=0.0)
        goal = episode.goals[0]
        assert len(goal.view_points) == 1
        np.testing.assert_array_equal(goal.view_points[0].agent_state.position,
                                      goal.position)

    def test_a_goal_on_another_level_keeps_its_own_height(self):
        episode = build_point_episode("s.glb", [0, 0, 0], [1.0, 0.0, 0.0], height_m=0.0,
                                      goal_height_m=1.5)
        assert float(episode.goals[0].position[1]) == pytest.approx(1.5)
        assert float(episode.start_position[1]) == pytest.approx(0.0)

    def test_the_start_rotation_is_the_quaternion_the_simulator_accepts(self):
        """`HabitatRobotSim.reset(robot_rotation=...)` takes xyzw, and habitat's forward is
        local -z -- the half-turn offset is the whole content of the conversion."""
        from continuous_demos.replay import hab_quat_to_heading

        for theta in (0.0, 1.0, -2.5, math.pi - 1e-6):
            episode = build_point_episode("s.glb", [0, 0, theta], [1, 1, 0], height_m=0.0)
            # `hab_quat_to_heading` reads [x, y, z, w] -- the order `reset` takes, and the
            # order the recorded demonstrations are stored in. Agreeing with it IS the
            # definition of correct here.
            recovered = hab_quat_to_heading(episode.start_rotation)
            assert recovered == pytest.approx(theta, abs=1e-6)

    def test_the_eval_wrapper_carries_the_planar_goal_for_the_policy(self):
        """`runner.EpisodeRunner` reads `extra["goal2d"]`; without it the policy gets a
        `<pose>` marker with nothing behind it."""
        episode = build_point_episode("hm3d/val/00800-X/X.basis.glb", [0, 0, 0],
                                      [2.0, 1.0, 0.5], height_m=0.1, episode_id="pn_7")
        wrapped = to_eval_episode(episode)
        assert wrapped.uid == "X:pn_7"
        assert wrapped.extra["goal2d"] == [2.0, 1.0, 0.5]
        assert wrapped.view_points.shape == (1, 3)
        np.testing.assert_allclose(wrapped.view_points[0], [2.0, 0.1, -1.0])

    def test_a_malformed_pose_is_rejected(self):
        with pytest.raises(ValueError, match="planar"):
            build_point_episode("s.glb", [0, 0], [1, 1, 0], height_m=0.0)


# ===========================================================================
# THE ROW ORDER, through the bridge
# ===========================================================================
longnav = pytest.importorskip(
    "longnav.utils.vector_rollout",
    reason=f"longnav not importable (looked in {_LONGNAV_SRC}; set LONGNAV_SRC)",
)


class _FakeTokenizer:
    """Enough of a tokenizer for `queue_user_block`. The ids are irrelevant here -- what
    is under test is which rows are bound to which markers, not what the text tokenizes
    to (that is pinned token-exactly on the model side)."""

    def encode(self, text, add_special_tokens=False):
        return [len(text)]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(
            part.get("text") or "<image>"
            for message in messages for part in message["content"]
        )


class _StubPolicy:
    """A ``VectorRolloutPolicy`` reduced to what ``_PoseFeed`` touches.

    The accumulator and the transform are the **production** functions
    (``accumulate_pose_rows`` -> ``relative_se2_tail``), for the same reason
    ``test_pose_injection_bridge.py`` drives ``pose_values`` unbound: a hand-written
    equivalent here would be the second implementation the whole design exists to prevent.
    Only the backbone is stubbed, because loading one to check three floats is absurd.
    """

    def __init__(self, spec, goal_placement):
        from longnav.utils.vector_rollout import RolloutConfig

        self._pose_spec = spec
        self.cfg = RolloutConfig()
        self.processor = _FakeProcessor()
        self._raw_poses = []
        self.queued = []
        self.goal_placement = goal_placement

    def pose_rows(self, rows):
        from longnav.utils.vector_rollout import accumulate_pose_rows

        return accumulate_pose_rows(self._raw_poses, rows)

    def pose_values(self, pose):
        return self.pose_rows([pose])

    def queue_user_block(self, text, modality=None):
        self.queued.append((text, modality))


def _pose_spec():
    from longnav.utils.modality_embed import ModalityEmbedSpec

    return ModalityEmbedSpec(
        token="<pose>", n_features=3, encoder="fourier_se2",
        column="obs_poses", transform="pose_relative_first",
    )


def _episode(n_obs=7, segment_lengths=(3, 4)):
    segments = [seg for seg, length in enumerate(segment_lengths) for _ in range(length)]
    assert len(segments) == n_obs
    return {
        "poses": [[float(i), 0.5 * i, 0.1 * i] for i in range(n_obs)],
        "goals": [[100.0 + segments[i], 2.0, 0.25] for i in range(n_obs)],
        "segments": segments,
    }


def _drive(feed, policy, episode):
    """One `step_kwargs` per observation, exactly as a rollout calls it."""
    values = []
    for pose, goal in zip(episode["poses"], episode["goals"]):
        kwargs = feed.step_kwargs(policy, pose, goal)
        values.append(kwargs["modality"][feed.spec.key])
    return values


@pytest.mark.parametrize("placement", ["segment", "observation"])
class TestPoseFeedRowOrder:
    """THE regression. The bridge, driven turn by turn, must lay the value column out
    exactly as the training formatter does for the same conversation."""

    def test_the_accumulated_column_is_the_formatters_column(self, placement):
        from objectnav_eval.bridge import _PoseFeed

        episode = _episode()
        policy = _StubPolicy(_pose_spec(), placement)
        feed = _PoseFeed(policy, goal_placement=placement)
        feed.on_episode_reset()
        _drive(feed, policy, episode)

        expected = longnav.modality_value_rows(
            episode["poses"], episode["goals"], episode["segments"], placement
        )
        got = [[float(v) for v in row] for row in policy._raw_poses]
        assert got == [[float(v) for v in row] for row in expected]

    def test_the_injected_values_are_relative_se2_of_that_column(self, placement):
        """And row 0 is exactly the origin -- the cheap end-to-end assertion that the
        anchor is an AGENT pose and not a goal."""
        import torch

        from longnav.utils.pose_frame import relative_se2

        from objectnav_eval.bridge import _PoseFeed

        episode = _episode()
        policy = _StubPolicy(_pose_spec(), placement)
        feed = _PoseFeed(policy, goal_placement=placement)
        feed.on_episode_reset()
        per_step = _drive(feed, policy, episode)

        column = torch.tensor(
            longnav.modality_value_rows(
                episode["poses"], episode["goals"], episode["segments"], placement
            ),
            dtype=torch.float64,
        )
        expected = relative_se2(column)
        assert torch.allclose(expected[0], torch.zeros(3))

        # Reassemble what the model was handed, in the order it was handed it: an
        # announcement's rows are queued before the turn's own.
        got = []
        queued = list(policy.queued)
        starts = longnav.segment_starts(episode["segments"])
        for i, step_values in enumerate(per_step):
            if i in starts and queued and placement == "segment":
                got.append(queued.pop(0)[1]["pose"])
            got.append(step_values)
        assert torch.allclose(torch.cat(got, dim=0), expected.float(), atol=1e-6)

    def test_the_row_count_matches_the_marker_count(self, placement):
        from objectnav_eval.bridge import _PoseFeed

        episode = _episode(n_obs=6, segment_lengths=(2, 2, 2))
        policy = _StubPolicy(_pose_spec(), placement)
        feed = _PoseFeed(policy, goal_placement=placement)
        feed.on_episode_reset()
        _drive(feed, policy, episode)
        assert len(policy._raw_poses) == longnav.expected_modality_len(6, 3, placement)

    def test_a_repeated_goal_is_not_re_announced(self, placement):
        """A goal in force is not a new goal. Announcing it again would write a message
        (and two rows) the training text does not have at that point."""
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), placement)
        feed = _PoseFeed(policy, goal_placement=placement)
        feed.on_episode_reset()
        for _ in range(4):
            feed.step_kwargs(policy, [0.0, 0.0, 0.0], [5.0, 5.0, 0.0])
        assert feed.goals_announced == 1

    def test_a_new_episode_starts_a_new_column(self, placement):
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), placement)
        feed = _PoseFeed(policy, goal_placement=placement)
        feed.on_episode_reset()
        feed.step_kwargs(policy, [0.0, 0.0, 0.0], [5.0, 5.0, 0.0])
        feed.on_episode_reset()
        policy._raw_poses.clear()  # what VectorRolloutPolicy.reset() does
        assert feed.goal is None
        feed.step_kwargs(policy, [1.0, 1.0, 0.0], [5.0, 5.0, 0.0])
        assert feed.goals_announced == 1


class TestPoseFeedPointNavGuards:
    def test_the_goal_announcement_is_its_own_queued_message(self):
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), "segment")
        feed = _PoseFeed(policy, goal_placement="segment")
        feed.on_episode_reset()
        feed.step_kwargs(policy, [0.0, 0.0, 0.0], [5.0, 5.0, 0.0])
        assert len(policy.queued) == 1
        text, modality = policy.queued[0]
        assert "You are at" in text and "Go to" in text
        assert modality["pose"].shape == (2, 3)

    def test_the_observation_placement_queues_no_message_at_all(self):
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), "observation")
        feed = _PoseFeed(policy, goal_placement="observation")
        feed.on_episode_reset()
        kwargs = feed.step_kwargs(policy, [0.0, 0.0, 0.0], [5.0, 5.0, 0.0])
        assert policy.queued == []
        assert kwargs["modality"]["pose"].shape == (2, 3)

    def test_the_marker_and_the_placement_are_installed_on_the_config(self):
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), "segment")
        _PoseFeed(policy, goal_placement="segment")
        assert policy.cfg.modality_marker == "<pose>"
        assert policy.cfg.goal_marker == "<pose>"
        assert policy.cfg.goal_placement == "segment"

    def test_a_missing_goal_is_an_error_not_an_empty_injection(self):
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), "segment")
        feed = _PoseFeed(policy, goal_placement="segment")
        feed.on_episode_reset()
        with pytest.raises(RuntimeError, match="carried no goal"):
            feed.step_kwargs(policy, [0.0, 0.0, 0.0], None)

    def test_an_unknown_placement_is_rejected(self):
        from objectnav_eval.bridge import _PoseFeed

        with pytest.raises(ValueError, match="goal_placement"):
            _PoseFeed(_StubPolicy(_pose_spec(), "x"), goal_placement="x")

    def test_a_checkpoint_with_no_pose_spec_cannot_run_pointnav(self):
        from objectnav_eval.bridge import _PoseFeed

        class _NoPose:
            _pose_spec = None

        with pytest.raises(ValueError, match="no modality spec"):
            _PoseFeed(_NoPose(), goal_placement="segment")

    def test_the_foreign_pose_ablation_is_refused_rather_than_half_applied(self):
        """Substituting the agent's trajectory would leave the goal rows anchored to a
        walk unrelated to them: a corruption of the geometry, not of the information."""
        from objectnav_eval.bridge import _PoseFeed

        class _Foreign:
            pass

        with pytest.raises(ValueError, match="not defined for a PointNav rollout"):
            _PoseFeed(_StubPolicy(_pose_spec(), "segment"), foreign=_Foreign(),
                      goal_placement="segment")

    def test_pointnav_gets_its_own_system_prompt_and_no_goal_text(self):
        from longnav.utils.vector_rollout import POINTNAV_SYSTEM_PROMPT

        from objectnav_eval.bridge import _PoseFeed

        feed = _PoseFeed(_StubPolicy(_pose_spec(), "segment"), goal_placement="segment")
        assert feed.reset_kwargs("chair") == {"system_prompt": POINTNAV_SYSTEM_PROMPT}

    def test_objectnav_reset_is_exactly_what_it_always_was(self):
        from objectnav_eval.bridge import _PoseFeed

        feed = _PoseFeed(_StubPolicy(_pose_spec(), None))
        assert feed.reset_kwargs("chair") == {"goal_text": "chair"}

    def test_an_objectnav_feed_installs_no_goal_marker(self):
        """Inertness: the PointNav additions must be invisible on the path that works."""
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), None)
        feed = _PoseFeed(policy)
        feed.on_episode_reset()
        assert policy.cfg.goal_marker is None
        assert policy.cfg.goal_placement is None
        assert feed.step_kwargs(policy, [1.0, 2.0, 0.5]) == {
            "obs_pose": [1.0, 2.0, 0.5]
        }
        assert policy.queued == []

    def test_the_injected_value_is_reported_for_the_trace(self):
        """`runner` records `pose_injected` from `last_stats["pose_value"]`, which
        `step(modality=...)` does not populate -- so the PointNav path has to put it
        back, or the one end-to-end proof that injection happened goes quiet exactly
        where there is most to get wrong."""
        from objectnav_eval.bridge import _PoseFeed

        policy = _StubPolicy(_pose_spec(), "segment")
        feed = _PoseFeed(policy, goal_placement="segment")
        feed.on_episode_reset()
        feed.step_kwargs(policy, [4.0, 5.0, 0.5], [9.0, 9.0, 0.0])
        stats = feed.annotate({})
        assert stats["pose_value"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        assert stats["goals_announced"] == 1


class TestTheFlagReachesTheModelInterpreter:
    """``--goal-placement`` is not inferable from the checkpoint, so if it silently stops
    being forwarded the rollout writes ObjectNav text for a PointNav model, the marker
    counts still add up, and the run prints a number. That is the whole reason the config
    travels to a worker as JSON rather than as a rebuilt command line -- but the *backend*
    args are still assembled by hand, so the assembly is pinned here."""

    def test_it_is_in_the_backend_args_when_set(self):
        from objectnav_eval.config import EvalConfig
        from objectnav_eval.policies import _stop_head_args

        assert _stop_head_args(EvalConfig(goal_placement="segment"))["goal_placement"] == (
            "segment"
        )

    def test_it_is_absent_when_unset(self):
        """So an ObjectNav run produces exactly the command line it always did --
        `PolicyBridge` turns each arg into a flag, and an extra one is a `SystemExit` in
        a policy server old enough not to know it."""
        from objectnav_eval.config import EvalConfig
        from objectnav_eval.policies import _stop_head_args

        assert "goal_placement" not in _stop_head_args(EvalConfig())

    def test_the_policy_server_accepts_the_flag_this_side_would_send(self):
        """The other end of the same flag, checked by name rather than assumed.

        ``PolicyBridge`` turns ``{"goal_placement": "segment"}`` into
        ``--goal-placement segment`` by mechanical ``_`` -> ``-`` substitution, so a
        rename on either side is a server that exits with a usage error before it ever
        connects -- which surfaces as "policy server exited before connecting", an
        unhelpful way to learn about a typo. ``--help`` is run in-process against the
        real parser, so the two names are compared rather than trusted.
        """
        import pathlib
        import subprocess
        import sys as _sys

        server = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "policy_server.py"
        out = subprocess.run(
            [_sys.executable, str(server), "--help"],
            capture_output=True, text=True, timeout=180,
        )
        assert out.returncode == 0, out.stderr
        assert "--goal-placement" in out.stdout


# ===========================================================================
# The source-kwargs seam
# ===========================================================================
class TestSourceSet:
    def test_numbers_and_booleans_are_converted(self):
        """`"5"` means something different to `range` than `5` does, and a source's
        factory signature expects the number."""
        assert _parse_source_set(["episodes_per_scene=5", "seed=7", "loop=true"]) == {
            "episodes_per_scene": 5, "seed": 7, "loop": True,
        }

    def test_floats_and_strings_pass_through(self):
        got = _parse_source_set(["min_geodesic_m=2.5", "path=/a/b*.glb"])
        assert got == {"min_geodesic_m": 2.5, "path": "/a/b*.glb"}

    def test_a_missing_equals_is_a_usage_error(self):
        with pytest.raises(ValueError, match="FIELD=VALUE"):
            _parse_source_set(["seed"])
