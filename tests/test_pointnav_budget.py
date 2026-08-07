"""The frame budget: when to open another segment, when to stop, what to keep.

Chaining exists so that every episode is roughly the same length -- a fixed-length row
wastes no training window. That property is entirely this class's arithmetic, and it has
already been wrong once: refusing to open a segment smaller than `min_segment_frames`
left an episode at 198 of 200 frames, which the exactness check then rejected. Both
halves of that fix are pinned below.
"""

import pytest

from continuous_demos.pointnav import TOO_FEW_FRAMES, FrameBudget


def test_a_fresh_budget_is_empty_and_wants_a_segment():
    b = FrameBudget(target_frames=400)
    assert b.frames == 0 and b.remaining() == 400
    assert not b.exhausted()
    assert b.wants_another_segment()


def test_adding_ticks_consumes_the_budget():
    b = FrameBudget(target_frames=100)
    b.add(40)
    assert (b.frames, b.remaining()) == (40, 60)
    b.add(60)
    assert b.exhausted() and b.remaining() == 0
    assert not b.wants_another_segment()


def test_remaining_never_goes_negative():
    """Truncation cuts on the tick the target is met, but a caller that overshoots by a
    tick must not get a negative radius out of `tail_radius_m`."""
    b = FrameBudget(target_frames=100)
    b.add(150)
    assert b.remaining() == 0
    assert b.tail_radius_m(fps=25, speed_mps=0.8) == 0.0


class TestWantsAnotherSegment:
    """The 198-of-200 bug, both directions."""

    def test_exact_keeps_opening_segments_right_up_to_the_target(self):
        """Refusing a short tail is what left an episode 2 frames short and binned it.
        The tail is over-aimed instead, so it overshoots and gets cut at the target."""
        b = FrameBudget(target_frames=200, min_segment_frames=50, exact=True)
        b.add(198)
        assert not b.room_for_a_segment()  # only 2 frames left
        assert b.wants_another_segment()  # ...and it opens one anyway

    def test_without_exact_the_minimum_segment_size_is_a_real_floor(self):
        """The length is free here, so a two-frame segment carrying its own goal buys
        nothing and the episode simply stops."""
        b = FrameBudget(target_frames=200, min_segment_frames=50, exact=False)
        b.add(198)
        assert not b.wants_another_segment()
        b2 = FrameBudget(target_frames=200, min_segment_frames=50, exact=False)
        b2.add(100)
        assert b2.wants_another_segment()

    def test_neither_mode_opens_a_segment_once_full(self):
        for exact in (True, False):
            b = FrameBudget(target_frames=100, exact=exact)
            b.add(100)
            assert not b.wants_another_segment()


class TestTailRadius:
    def test_it_aims_long_on_purpose(self):
        """Aiming at exactly the remaining frames systematically under-fills, because a
        segment also spends frames turning and decelerating into the goal."""
        b = FrameBudget(target_frames=400, tail_overshoot=1.5)
        b.add(300)  # 100 frames = 4 s left; at 0.8 m/s that is 3.2 m of driving
        assert b.tail_radius_m(fps=25, speed_mps=0.8) == pytest.approx(3.2 * 1.5)

    def test_the_floor_keeps_the_radius_above_the_minimum_geodesic(self):
        """A radius under `min_geodesic_m` draws nothing but `geodesic_too_short`
        rejections and burns the retry budget for no reason."""
        b = FrameBudget(target_frames=400)
        b.add(395)  # 5 frames left -> a radius of a few centimetres
        assert b.tail_radius_m(fps=25, speed_mps=0.8) < 0.5
        assert b.tail_radius_m(fps=25, speed_mps=0.8, floor_m=2.25) == pytest.approx(2.25)


class TestAccepts:
    def test_exactly_on_target_is_kept(self):
        b = FrameBudget(target_frames=400, exact=True)
        b.add(400)
        assert b.accepts() == (True, None)

    def test_a_couple_of_frames_short_is_still_kept(self):
        """Binning a clean episode for landing on 398 of 400 throws away good
        demonstration to buy a uniformity nothing needs to be exact: at
        `obs_stride_frames=5` a two-frame shortfall is not even one observation."""
        b = FrameBudget(target_frames=400, exact=True, tolerance_frames=25)
        b.add(398)
        assert b.accepts() == (True, None)

    def test_well_short_is_rejected_with_a_reason_code(self):
        b = FrameBudget(target_frames=400, exact=True, tolerance_frames=25)
        b.add(300)
        assert b.accepts() == (False, TOO_FEW_FRAMES)

    def test_the_tolerance_boundary_is_inclusive(self):
        b = FrameBudget(target_frames=400, exact=True, tolerance_frames=25)
        b.add(375)
        assert b.accepts()[0] is True
        b2 = FrameBudget(target_frames=400, exact=True, tolerance_frames=25)
        b2.add(374)
        assert b2.accepts()[0] is False

    def test_without_exact_the_floor_is_min_frames(self):
        b = FrameBudget(target_frames=400, min_frames=300, exact=False)
        b.add(320)
        assert b.accepts() == (True, None)
        b2 = FrameBudget(target_frames=400, min_frames=300, exact=False)
        b2.add(299)
        assert b2.accepts() == (False, TOO_FEW_FRAMES)

    def test_an_empty_episode_is_never_accepted(self):
        b = FrameBudget(target_frames=400, min_frames=0, exact=False)
        assert b.accepts() == (False, TOO_FEW_FRAMES)


class TestRewind:
    """A rollback discards a segment's ticks; the budget has to give them back."""

    def test_rewinding_restores_the_remaining_count(self):
        b = FrameBudget(target_frames=400)
        b.add(120)
        b.add(90)  # a segment that then fails
        b.rewind_to(120)
        assert b.frames == 120 and b.remaining() == 280
        assert b.wants_another_segment()

    def test_rewinding_to_zero_is_legal(self):
        b = FrameBudget(target_frames=400)
        b.add(80)
        b.rewind_to(0)
        assert b.frames == 0

    def test_rewinding_below_zero_raises(self):
        with pytest.raises(ValueError, match="negative"):
            FrameBudget(target_frames=400).rewind_to(-1)


class TestConstruction:
    def test_an_unreachable_floor_is_refused_at_construction(self):
        """Otherwise every episode is generated in full and then rejected, which reads as
        a broken scene rather than a bad flag."""
        with pytest.raises(ValueError, match="no episode could ever be accepted"):
            FrameBudget(target_frames=200, min_frames=300)

    def test_a_non_positive_target_is_refused(self):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="must be positive"):
                FrameBudget(target_frames=bad)
