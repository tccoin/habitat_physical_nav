"""Rolling back a failed segment without losing the episode.

The property that matters, and the reason the failed excursion is *erased* rather than
relabelled under the new goal: `goal2d` stays piecewise-constant, one run per kept
segment. Relabelling would teach a policy that goals move when you fail.

The second property is bookkeeping honesty: a rolled-back segment still appears in the
reason histogram. If rollbacks were silent the corpus would read cleaner than the scenes
are, and every threshold re-cut from a pilot would be fit to a lie.
"""

import numpy as np
import pytest

from continuous_demos.pointnav import (
    OUTCOME_OK,
    OUTCOME_ROLLED_BACK,
    NOT_REACHED,
    REF_MISS,
    FrameBudget,
    RollbackBudget,
    SegmentRecord,
    segments_summary,
    truncate_parallel,
)


# ---------------------------------------------------------------------------
# truncate_parallel: the per-tick arrays move together or the corpus is broken
# ---------------------------------------------------------------------------
def make_arrays(n):
    """The four arrays the recorder keeps in lockstep."""
    frames = [f"frame{i}" for i in range(n)]
    poses = [[float(i), 0.0, 0.0] for i in range(n)]
    goal2d = [[9.0, 9.0, 0.0] for _ in range(n)]
    segment_index = [0] * n
    return [frames, poses, goal2d, segment_index]


def test_every_array_is_cut_to_the_same_length():
    arrays = make_arrays(10)
    truncate_parallel(arrays, 4)
    assert [len(a) for a in arrays] == [4, 4, 4, 4]
    assert arrays[0] == ["frame0", "frame1", "frame2", "frame3"]


def test_truncating_to_zero_empties_them():
    arrays = make_arrays(5)
    truncate_parallel(arrays, 0)
    assert [len(a) for a in arrays] == [0, 0, 0, 0]


def test_truncating_to_the_current_length_is_a_no_op():
    arrays = make_arrays(6)
    truncate_parallel(arrays, 6)
    assert [len(a) for a in arrays] == [6, 6, 6, 6]


def test_arrays_already_out_of_step_raise_rather_than_being_papered_over():
    """The arrow builder raises on a length mismatch, so catching it here turns a
    confusing build-time failure into an attributable one."""
    arrays = make_arrays(5)
    arrays[1] = arrays[1][:3]  # poses fell behind
    with pytest.raises(ValueError, match="out of step"):
        truncate_parallel(arrays, 5)


def test_a_negative_length_raises():
    with pytest.raises(ValueError, match="negative"):
        truncate_parallel(make_arrays(3), -1)


# ---------------------------------------------------------------------------
# goal2d stays piecewise-constant across a rollback
# ---------------------------------------------------------------------------
def runs(values):
    """Consecutive equal-value runs, as `[(value, length), ...]`."""
    out = []
    for v in values:
        key = tuple(v)
        if out and out[-1][0] == key:
            out[-1][1] += 1
        else:
            out.append([key, 1])
    return [(k, n) for k, n in out]


def test_an_erased_segment_leaves_no_run_behind():
    """The whole point. Segment 1 fails, its ticks go, and `goal2d` reads as if it never
    happened -- one run per kept segment, and none for the discarded one."""
    frames, poses, goal2d, seg_idx = [], [], [], []

    def drive(index, goal, n):
        for i in range(n):
            frames.append(f"s{index}f{i}")
            poses.append([float(i), 0.0, 0.0])
            goal2d.append(list(goal))
            seg_idx.append(index)

    drive(0, [1.0, 1.0, 0.0], 30)
    mark = len(frames)
    drive(1, [2.0, 2.0, 0.0], 20)  # this one fails
    truncate_parallel([frames, poses, goal2d, seg_idx], mark)
    drive(1, [3.0, 3.0, 0.0], 25)  # resampled goal, same segment index

    assert runs(goal2d) == [((1.0, 1.0, 0.0), 30), ((3.0, 3.0, 0.0), 25)]
    assert (2.0, 2.0, 0.0) not in [k for k, _ in runs(goal2d)]
    assert len(frames) == len(poses) == len(goal2d) == len(seg_idx) == 55
    assert seg_idx == [0] * 30 + [1] * 25


def test_the_pose_stream_stays_continuous_across_the_join():
    """The robot is physically returned to the last kept pose, so there is no jump."""
    # 2 cm a tick, which is what 0.5 m/s at 25 Hz looks like.
    poses = [[0.02 * i, 0.0, 0.0] for i in range(30)]
    mark = len(poses)
    poses += [[100.0, 100.0, 0.0] for _ in range(20)]  # the failed excursion
    truncate_parallel([poses], mark)

    # The recorder respawns at the last KEPT pose, so the next tick continues from there.
    resume = list(poses[-1])
    poses += [[resume[0] + 0.02 * (i + 1), resume[1], resume[2]] for i in range(10)]

    steps = [abs(poses[i + 1][0] - poses[i][0]) for i in range(len(poses) - 1)]
    assert max(steps) == pytest.approx(0.02), "no teleport anywhere in the kept stream"
    assert len(poses) == 40


# ---------------------------------------------------------------------------
# RollbackBudget
# ---------------------------------------------------------------------------
class TestRollbackBudget:
    def test_it_spends_down_to_its_limit(self):
        b = RollbackBudget(limit=3)
        assert [b.spend(NOT_REACHED) for _ in range(3)] == [True, True, True]
        assert b.exhausted
        assert b.spend(NOT_REACHED) is False

    def test_a_zero_limit_refuses_the_first_rollback(self):
        b = RollbackBudget(limit=0)
        assert b.spend(NOT_REACHED) is False
        assert b.exhausted

    def test_reasons_are_counted_so_the_histogram_sees_them(self):
        """Silent rollbacks make the corpus read cleaner than the scenes are."""
        b = RollbackBudget(limit=5)
        b.spend(NOT_REACHED)
        b.spend(NOT_REACHED)
        b.spend(REF_MISS)
        assert b.reasons == {NOT_REACHED: 2, REF_MISS: 1}
        assert b.used == 3


# ---------------------------------------------------------------------------
# The segment record, and what a rolled-back one contributes
# ---------------------------------------------------------------------------
def segment(index, outcome=OUTCOME_OK, geodesic=4.0, travelled=4.2, reason=None):
    return SegmentRecord(
        index=index, goal2d=[1.0, 2.0, 0.0], start_frame=0, end_frame=100,
        geodesic_m=geodesic, travelled_m=travelled, final_dist_m=0.12,
        turn_rad=1.0, island=0, outcome=outcome, reason=reason,
    )


def test_a_rolled_back_segment_scores_zero_and_is_not_reached():
    rolled = segment(1, OUTCOME_ROLLED_BACK, reason=NOT_REACHED)
    assert rolled.reached is False
    assert rolled.spl() == 0.0
    assert rolled.as_dict()["reason"] == NOT_REACHED


def test_rolled_back_segments_are_excluded_from_the_kept_totals_but_still_counted():
    records = [segment(0), segment(1, OUTCOME_ROLLED_BACK, reason=NOT_REACHED), segment(1)]
    summary = segments_summary(records)
    assert summary["num_segments"] == 2, "the two kept ones"
    assert summary["num_rolled_back"] == 1, "and the discarded one is still visible"
    assert summary["num_reached"] == 2
    assert summary["geodesic_m"] == pytest.approx(8.0), "no distance from the failed one"


def test_a_summary_over_nothing_does_not_divide_by_zero():
    summary = segments_summary([])
    assert summary["num_segments"] == 0
    assert summary["mean_spl"] is None and summary["mean_turn_deg"] is None


def test_num_frames_is_derived_not_stored():
    rec = SegmentRecord(index=0, goal2d=[0, 0, 0], start_frame=137, end_frame=265,
                        geodesic_m=1.0, travelled_m=1.0, final_dist_m=0.0,
                        turn_rad=0.0, island=0)
    assert rec.num_frames == 128
    assert rec.as_dict()["num_frames"] == 128


# ---------------------------------------------------------------------------
# The recorder's own rollback, against a stub simulator
# ---------------------------------------------------------------------------
def test_the_recorders_rollback_truncates_and_respawns_at_the_last_kept_pose():
    """`_rollback` is the one place the erase and the physical reset have to agree."""
    pytest.importorskip("habitat_sim")
    from continuous_demos.pointnav_episode import PointNavRecording, _rollback

    class FakeAABB:
        min = type("V", (), {"y": -0.25})()

    class FakeRobot:
        translation = [0.0, 0.3, 0.0]
        root_scene_node = type("N", (), {"cumulative_bb": FakeAABB()})()

    class FakeSim:
        robot = FakeRobot()
        reset_calls = []

        def get_joint_state(self, name, is_vel=False):
            return 0.0

        def reset(self, **kwargs):
            FakeSim.reset_calls.append(kwargs)

    rec = PointNavRecording(start_pose2d=[0.0, 0.0, 0.0])
    for i in range(30):
        rec.frames.append(i)
        rec.poses.append(np.array([float(i), 1.0, 0.5]))
        rec.goal2d.append([9.0, 9.0, 0.0])
        rec.segment_index.append(0)
    mark = len(rec.frames)
    for i in range(20):  # the failed segment
        rec.frames.append(100 + i)
        rec.poses.append(np.array([99.0, 99.0, 0.0]))
        rec.goal2d.append([7.0, 7.0, 0.0])
        rec.segment_index.append(1)

    budget = FrameBudget(target_frames=400)
    budget.add(50)
    sim = FakeSim()
    _rollback(sim, rec, budget, mark)

    assert len(rec.frames) == len(rec.poses) == len(rec.goal2d) == 30
    rec.check_alignment()  # must not raise
    assert budget.frames == 30
    assert [7.0, 7.0, 0.0] not in rec.goal2d

    spawn = FakeSim.reset_calls[-1]
    # Respawned at the last KEPT pose: planar [29, 1] -> habitat [29, y, -1].
    assert spawn["robot_translation"][0] == pytest.approx(29.0)
    assert spawn["robot_translation"][2] == pytest.approx(-1.0)
    assert spawn["snap_to_navmesh"] is False, (
        "the point came off the navmesh already; re-snapping can resolve elsewhere"
    )
