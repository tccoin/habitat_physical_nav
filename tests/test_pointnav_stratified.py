"""Stratified goal-distance sampling: reweight the draw, never relax the screening.

`sample_segment` draws a navigable point, screens it, and takes the first candidate that
passes. That yields the distance distribution the *scene* has. Measured over six HM3D
scenes, the geodesic between two random points on one island has median 5.6 m and reaches
30 m in 0.0% of draws -- so raising `max_geodesic_m` lengthens the tail without changing
the short-skewed shape. `stratified_pool` corrects that by choosing among candidates.

Two properties matter and are pinned here:

- **Off is off.** `stratified_pool = 0` must return the first screening-clean candidate,
  consuming draws in exactly the order it always did. Every corpus cut before this option
  existed depends on that, and the reproducibility contract in `sample_segment`'s docstring
  is only worth anything if the default path is untouched.
- **It reweights, never admits.** Candidates enter the pool only after passing the same
  `screen_segment` call an unstratified draw applies, so no setting of `stratified_pool`
  can produce a goal the unstratified sampler would have rejected. A version that widened
  the window to hit its target would silently change what the corpus means.

`plan_path` is monkeypatched: the geometry under test is the *choice among candidates*,
not habitat's shortest-path solver, and stubbing it makes the candidate distances exact
rather than whatever a scene happened to offer.
"""

import numpy as np
import pytest

pytest.importorskip("habitat_sim")

from continuous_demos import pointnav as pn
from continuous_demos import pointnav_episode as pe


class ScriptedPathfinder:
    """Hands out a fixed sequence of goals, each at a known distance along +x."""

    def __init__(self, distances):
        self.distances = list(distances)
        self.calls = 0

    def _next(self):
        d = self.distances[min(self.calls, len(self.distances) - 1)]
        self.calls += 1
        return np.array([float(d), 0.0, 0.0], dtype=np.float32)

    def get_random_navigable_point(self, island_index=None):
        return self._next()

    def get_random_navigable_point_near(self, circle_center=None, radius=None,
                                        island_index=None):
        return self._next()

    def get_island(self, point):
        return 0


def fake_plan_path(pathfinder, start, goal):
    """A straight path from start to goal; geodesic is the euclidean distance."""
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    return np.stack([start, goal]), float(np.linalg.norm(goal - start))


@pytest.fixture(autouse=True)
def _stub_plan_path(monkeypatch):
    monkeypatch.setattr(pe, "plan_path", fake_plan_path)


def draw(distances, *, pool=0, seed=0, **cfg_overrides):
    cfg = pn.PointNavSampleConfig(
        min_geodesic_m=2.0, max_geodesic_m=30.0, min_euclid_m=1.0,
        stratified_pool=pool, segment_retries=30, **cfg_overrides,
    )
    pf = ScriptedPathfinder(distances)
    segment, _rejections = pe.sample_segment(
        pf, np.random.default_rng(seed), cfg,
        start_3d=np.zeros(3, dtype=np.float32), theta_now=0.0, island=0,
    )
    return segment, pf


# ---------------------------------------------------------------------------
# Off is off
# ---------------------------------------------------------------------------
def test_disabled_takes_the_first_acceptable_candidate():
    segment, pf = draw([3.0, 25.0, 12.0], pool=0)
    assert segment is not None
    assert segment.geodesic_m == pytest.approx(3.0)
    assert pf.calls == 1, "an unstratified draw must stop at the first clean candidate"


def test_disabled_still_skips_candidates_that_fail_screening():
    """1.0 m is below min_geodesic_m, so it is rejected and the draw continues."""
    segment, pf = draw([1.0, 7.0], pool=0)
    assert segment.geodesic_m == pytest.approx(7.0)
    assert pf.calls == 2


# ---------------------------------------------------------------------------
# On: nearest to the sampled target
# ---------------------------------------------------------------------------
def test_stratified_returns_the_candidate_nearest_the_drawn_target():
    distances = [3.0, 25.0, 12.0, 8.0]
    pool = len(distances)
    # The target is drawn from the rng before any candidate, so it is reproducible.
    target = float(np.random.default_rng(0).uniform(2.0, 30.0))
    expected = min(distances, key=lambda d: abs(d - target))
    segment, pf = draw(distances, pool=pool, seed=0)
    assert segment.geodesic_m == pytest.approx(expected)
    assert pf.calls == pool, "the pool should be filled before choosing"


def test_stratified_never_returns_a_candidate_that_fails_screening():
    """A far target must not tempt the sampler past max_geodesic_m."""
    # 40 m exceeds max_geodesic_m and must be screened out however attractive it looks.
    for seed in range(25):
        segment, _ = draw([40.0, 40.0, 5.0, 40.0], pool=4, seed=seed)
        assert segment is not None
        assert segment.geodesic_m == pytest.approx(5.0)


def test_stratified_returns_none_when_nothing_screens_clean():
    segment, _ = draw([0.5, 0.7, 40.0], pool=8)
    assert segment is None


def test_stratified_spreads_the_distance_distribution():
    """The property the option exists for, against a scene-like candidate supply.

    Candidates are drawn from a right-skewed pool resembling real navmesh geometry. The
    unstratified sampler inherits that skew; the stratified one should reach materially
    further, because it gets to pick.
    """
    rng = np.random.default_rng(7)
    supply = np.clip(rng.gamma(shape=2.0, scale=2.8, size=(400, 24)), 0.3, 30.0)
    plain = [draw(row, pool=0, seed=i)[0] for i, row in enumerate(supply)]
    strat = [draw(row, pool=24, seed=i)[0] for i, row in enumerate(supply)]
    plain_d = np.array([s.geodesic_m for s in plain if s is not None])
    strat_d = np.array([s.geodesic_m for s in strat if s is not None])
    assert np.median(strat_d) > np.median(plain_d) + 1.0
    assert strat_d.std() > plain_d.std()
