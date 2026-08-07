"""PointNav episode design: screening, geometry, island choice, seeding.

Everything in `continuous_demos.pointnav` is arithmetic over numbers a simulator
produced, which is the point of the split -- "is a 1.4 m goal too close" and "does a
zero-length first leg break the turn calculation" are answerable without loading a scene.
No habitat import anywhere in this file.
"""

import math

import numpy as np
import pytest

from continuous_demos import pointnav as pn


CFG = pn.PointNavSampleConfig()


def screen(**overrides):
    """Screen a segment that passes by default, with the named field perturbed."""
    kwargs = dict(
        geodesic_m=4.0, euclid_m=3.5, height_delta_m=0.0, same_island=True, turn_rad=1.0
    )
    kwargs.update(overrides)
    return pn.screen_segment(CFG, **kwargs)


# ---------------------------------------------------------------------------
# screen_segment, at each boundary
# ---------------------------------------------------------------------------
def test_a_reasonable_segment_is_accepted():
    assert screen() is None


def test_an_unreachable_goal_is_no_path():
    assert screen(geodesic_m=float("inf")) == pn.NO_PATH
    assert screen(geodesic_m=float("nan")) == pn.NO_PATH


def test_a_different_island_is_caught_before_any_distance_test():
    """`inf` and `same_island=False` are the same physical fact; the island answer is
    the more specific one, so it must not be masked by the distance tests."""
    assert screen(same_island=False) == pn.GOAL_OTHER_ISLAND


@pytest.mark.parametrize("delta, expected", [(0.5, None), (0.501, pn.GOAL_OTHER_FLOOR),
                                             (-0.5, None), (-0.51, pn.GOAL_OTHER_FLOOR)])
def test_the_floor_test_is_symmetric_and_admits_a_ramp(delta, expected):
    """0.5 m admits a ramp and excludes a storey. Signed both ways: descending a
    staircase is as much a floor change as climbing one."""
    assert screen(height_delta_m=delta) == expected


@pytest.mark.parametrize("geo, expected", [
    (1.49, pn.GEODESIC_TOO_SHORT),
    (1.5, None),
    (8.0, None),
    (8.01, pn.GEODESIC_TOO_LONG),
])
def test_the_geodesic_band_is_inclusive_at_both_ends(geo, expected):
    assert screen(geodesic_m=geo, euclid_m=min(geo, 3.0)) == expected


def test_a_path_that_loops_back_is_rejected_even_when_it_is_long():
    """A 6 m path ending 0.4 m from where it started: the goal image is the start image."""
    assert screen(geodesic_m=6.0, euclid_m=0.4) == pn.EUCLID_TOO_SHORT
    assert screen(geodesic_m=6.0, euclid_m=1.0) is None


def test_the_turn_gate_is_off_by_default():
    """Gating on a minimum turn would delete straight corridors, which are real."""
    assert CFG.min_turn_deg == 0.0
    assert screen(turn_rad=0.0) is None


def test_the_turn_gate_works_when_switched_on():
    cfg = pn.PointNavSampleConfig(min_turn_deg=30.0)
    common = dict(geodesic_m=4.0, euclid_m=3.5, height_delta_m=0.0, same_island=True)
    assert pn.screen_segment(cfg, turn_rad=math.radians(29), **common) == pn.TURN_TOO_SMALL
    assert pn.screen_segment(cfg, turn_rad=math.radians(31), **common) is None
    # Signed: turning right by 40 degrees is as much a turn as turning left.
    assert pn.screen_segment(cfg, turn_rad=math.radians(-40), **common) is None


def test_every_code_screen_segment_can_return_has_prose():
    """The failure log renders `code: detail`, and `detail` falls back to this table."""
    for code in pn.SAMPLING_SCOPED | pn.SEGMENT_SCOPED:
        assert code in pn.REASON_TEXT and pn.REASON_TEXT[code]


def test_describe_puts_the_code_first_so_a_histogram_is_one_cut():
    line = pn.describe(pn.INEFFICIENT, "segment 2 spl 0.41 < 0.55")
    assert line.split(":")[0] == pn.INEFFICIENT
    assert pn.describe(pn.ESCAPED).startswith(pn.ESCAPED + ": ")


def test_the_two_scopes_do_not_overlap():
    """A code recoverable by rollback is not one raised before the robot moved."""
    assert not (pn.SAMPLING_SCOPED & pn.SEGMENT_SCOPED)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class TestTurnToFace:
    def test_facing_along_the_path_needs_no_turn(self):
        assert pn.turn_to_face(0.0, [[0.0, 0.0], [1.0, 0.0]]) == pytest.approx(0.0)

    def test_a_quarter_turn_left_is_positive(self):
        """+theta is CCW from +X in the planar frame, so a goal to the north is +pi/2."""
        assert pn.turn_to_face(0.0, [[0.0, 0.0], [0.0, 1.0]]) == pytest.approx(math.pi / 2)

    def test_a_quarter_turn_right_is_negative(self):
        assert pn.turn_to_face(0.0, [[0.0, 0.0], [0.0, -1.0]]) == pytest.approx(-math.pi / 2)

    def test_it_takes_the_short_way_round(self):
        """Unwrapped, a 10-degree correction across the seam reads as 350 and would size
        the turn phase's timeout from the wrong number."""
        turn = pn.turn_to_face(math.pi - 0.1, [[0.0, 0.0], [-1.0, -0.0001]])
        assert abs(turn) < 0.2

    def test_a_repeated_leading_point_is_skipped(self):
        """Planners emit duplicate points; a zero-length leg has no direction."""
        path = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 2.0]]
        assert pn.turn_to_face(0.0, path) == pytest.approx(math.pi / 2)

    def test_an_all_identical_path_raises_rather_than_facing_east(self):
        with pytest.raises(ValueError, match="coincide"):
            pn.turn_to_face(0.0, [[1.0, 1.0], [1.0, 1.0]])

    def test_a_one_point_path_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            pn.turn_to_face(0.0, [[1.0, 1.0]])


class TestPathTangentHeading:
    def test_start_and_end_differ_on_a_path_that_turns(self):
        path = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        assert pn.path_tangent_heading(path, "start") == pytest.approx(0.0)
        assert pn.path_tangent_heading(path, "end") == pytest.approx(math.pi / 2)

    def test_a_repeated_trailing_point_is_skipped(self):
        path = [[0.0, 0.0], [0.0, 3.0], [0.0, 3.0]]
        assert pn.path_tangent_heading(path, "end") == pytest.approx(math.pi / 2)


class TestSegmentSpl:
    def test_a_perfect_drive_is_one(self):
        assert pn.segment_spl(4.0, 4.0, True) == pytest.approx(1.0)

    def test_a_detour_is_the_ratio(self):
        assert pn.segment_spl(4.0, 8.0, True) == pytest.approx(0.5)

    def test_beating_the_geodesic_does_not_exceed_one(self):
        """Cutting a corner off the polyline is normal and is not super-optimal."""
        assert pn.segment_spl(4.0, 3.5, True) == pytest.approx(1.0)

    def test_not_reaching_is_zero_however_short_the_route(self):
        assert pn.segment_spl(4.0, 4.0, False) == 0.0

    def test_zero_over_zero_is_one_not_undefined(self):
        """Standing on the goal already is perfectly efficient. Mirrors
        `objectnav._path_ratio`'s guard."""
        assert pn.segment_spl(0.0, 0.0, True) == 1.0


# ---------------------------------------------------------------------------
# Island choice
# ---------------------------------------------------------------------------
class TestChooseIsland:
    #: The measured island structure of 17DRP5sb8fy on the robot navmesh: one floor and
    #: two closets. `(index, area_m2, radius_m)`.
    REAL = [(0, 24.2, 8.15), (1, 1.3, 1.01), (2, 2.9, 2.67)]

    def test_it_keeps_only_the_floor_on_a_real_scene(self):
        rng = np.random.default_rng(0)
        assert all(pn.choose_island(rng, self.REAL, CFG) == 0 for _ in range(20))

    def test_a_scene_with_no_room_returns_none_rather_than_raising(self):
        rng = np.random.default_rng(0)
        assert pn.choose_island(rng, [(0, 1.0, 0.5), (1, 2.0, 1.0)], CFG) is None
        assert pn.choose_island(rng, [], CFG) is None

    def test_both_floors_must_be_cleared(self):
        rng = np.random.default_rng(0)
        # Big but thin -- a long corridor of area 20 whose radius is 1 m.
        assert pn.choose_island(rng, [(0, 20.0, 1.0)], CFG) is None
        # Compact but tiny.
        assert pn.choose_island(rng, [(0, 4.0, 3.0)], CFG) is None

    def test_it_is_area_weighted(self):
        rng = np.random.default_rng(0)
        islands = [(0, 10.0, 3.0), (1, 30.0, 5.0)]
        draws = [pn.choose_island(rng, islands, CFG) for _ in range(10000)]
        share = sum(1 for d in draws if d == 1) / len(draws)
        assert 0.72 < share < 0.78  # 30 / 40

    def test_it_is_deterministic_for_a_seed(self):
        islands = [(0, 10.0, 3.0), (1, 30.0, 5.0), (2, 9.0, 2.5)]
        a = [pn.choose_island(np.random.default_rng(7), islands, CFG) for _ in range(5)]
        b = [pn.choose_island(np.random.default_rng(7), islands, CFG) for _ in range(5)]
        assert a == b


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
class TestEpisodeSeed:
    def test_it_is_stable(self):
        assert pn.episode_seed(0, "s", 7) == pn.episode_seed(0, "s", 7)

    def test_two_scenes_episode_seven_do_not_collide(self):
        """`corpus_seed + index` would give every scene the same sequence."""
        assert pn.episode_seed(0, "sceneA", 7) != pn.episode_seed(0, "sceneB", 7)

    def test_neighbouring_indices_are_not_neighbouring_seeds(self):
        seeds = [pn.episode_seed(0, "s", i) for i in range(4)]
        assert len(set(seeds)) == 4
        assert all(abs(seeds[i + 1] - seeds[i]) > 1000 for i in range(3))

    def test_the_corpus_seed_moves_everything(self):
        assert pn.episode_seed(0, "s", 7) != pn.episode_seed(1, "s", 7)

    def test_it_fits_in_a_numpy_seed(self):
        for i in range(200):
            assert 0 <= pn.episode_seed(12345, "some-scene-name", i) < 2 ** 32
