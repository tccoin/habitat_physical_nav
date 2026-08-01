"""Correctness proof for the rotation canonicaliser.

No habitat_sim/magnum dependency -- ``rotation_canonicalize`` is pure
numpy/integer arithmetic, so these run in any env with the package installed.

The property under test. For a rotation run ``R`` with covered heading set
``S``, start ``theta_s`` and end ``theta_e``, the construction returns ``C``
with

  1. FEASIBILITY   ``C`` starts at ``theta_s``, ends at ``theta_e``, and covers
                   *exactly* ``S`` -- not a superset. A ``C`` covering more than
                   the human did would put views in the demonstration that the
                   human never collected.
  2. OPTIMALITY    ``|C| <= |X|`` for every feasible ``X``.

Optimality is settled here by exhaustive breadth-first search over the discrete
state space ``(heading, min_seen, max_seen)``. Because the action space is
discrete 30-degree primitives, that search is not a sample -- it is the answer,
for every reachable ``(range, start, end)`` target at once.

The complementary check, which cannot live in a unit test, is that the human's
own run is feasible and hence an upper bound: ``|C| <= |R|`` was verified over
all 2,353,082 rotation runs of the 56-scene corpus with zero violations
(``dump/rotation_blunders/verify_corpus.py``).
"""

import itertools
from collections import deque

import numpy as np
import pytest

from continuous_demos.rotation_canonicalize import (
    CELLS_PER_TURN,
    TURN_DEG,
    canonicalize_actions,
    canonicalize_demo,
    canonicalize_run,
    find_rotation_runs,
    optimal_length,
    optimal_signs,
)

H_BOUND = 26  # heading bound for the BFS lattice; > 2 full circles either way


def seq(s):
    """'LRL' -> action list."""
    return ["TURN_LEFT" if c == "L" else "TURN_RIGHT" for c in s]


def txt(actions):
    return "".join("L" if a == "TURN_LEFT" else "R" for a in actions)


def geometry(signs):
    h = lo = hi = 0
    for s in signs:
        h += s
        lo, hi = min(lo, h), max(hi, h)
    return lo, hi, h


def signs_of(actions):
    return [1 if a == "TURN_LEFT" else -1 for a in actions]


# --------------------------------------------------------------- known answers
KNOWN = [
    ("L", "L"),
    ("LLLL", "LLLL"),                 # monotone sweep: never shortenable
    ("R" * 11, "R" * 11),
    ("LR", "LR"),                     # a glance out and back is legitimate
    ("RL", "RL"),
    ("LRRL", "LRRL"),                 # centred scan, start == end
    ("LLRR", "LLRR"),
    ("LRLR", "LR"),                   # the canonical blunder
    ("LRLRLR", "LR"),
    ("LLRRLL", "LL"),
    ("LRL", "L"),
    ("L" * 12, "L" * 12),             # exact 360 spin
    ("L" * 24, "L" * 12),             # 720 double spin: half is a repeat
    ("L" * 13, "L" * 13),             # 390: already minimal for its end heading
    ("L" * 11 + "R" * 11, "L" * 11 + "R" * 11),  # an arc, not a circle
]


@pytest.mark.parametrize("src,want", KNOWN)
def test_known_answers(src, want):
    assert txt(canonicalize_run(seq(src))) == want


def test_badly_ordered_run_is_shortened():
    """0 -> +60 -> -30 -> +30: no repeated out-and-back, but not minimal."""
    out = canonicalize_run(seq("LLRRRLL"))
    assert len(out) == 5
    assert geometry(signs_of(out)) == geometry(signs_of(seq("LLRRRLL")))


# ------------------------------------------------------------ the full circle
FULL_CIRCLE = [
    # d degrees, expected cost degrees, expected signed sweep (None where tied)
    (0, 360, None),
    (30, 390, +390),
    (90, 450, +450),
    (150, 510, +510),
    (180, 540, None),
    (210, 510, -510),
    (270, 450, -450),
    (330, 390, -390),
]


@pytest.mark.parametrize("d_deg,want_cost,want_sweep", FULL_CIRCLE)
@pytest.mark.parametrize("prefer", [1, -1])
def test_full_circle_closed_form(d_deg, want_cost, want_sweep, prefer):
    """'Full circle' does NOT imply the run returns to its start heading.

    Assuming it does yields the right coverage and the wrong end heading, which
    would silently break the handoff into the following drive phase. For
    d > 180 the optimal sweep is *negative*.
    """
    d = d_deg // 30
    src = [1] * (CELLS_PER_TURN + d)
    lo, hi, end = geometry(src)
    assert hi - lo >= CELLS_PER_TURN

    out = optimal_signs(lo, hi, end, 0, prefer)
    assert len(out) * TURN_DEG == want_cost
    assert len(set(out)) == 1, "the full-circle optimum is monotone"
    clo, chi, cend = geometry(out)
    assert chi - clo >= CELLS_PER_TURN, "must still cover the whole circle"
    assert (cend - end) % CELLS_PER_TURN == 0, "must land on the end heading"
    if want_sweep is None:
        assert out[0] == prefer, "on a tie, keep the human's first direction"
    else:
        assert sum(out) * TURN_DEG == want_sweep


# ------------------------------------------------- exhaustive optimality proof
def bfs_distances():
    """Fewest primitives to reach every ``(heading, lo, hi)`` state from rest."""
    dist = {(0, 0, 0): 0}
    q = deque([(0, 0, 0)])
    while q:
        h, lo, hi = q.popleft()
        d = dist[(h, lo, hi)]
        for s in (1, -1):
            nh = h + s
            if abs(nh) > H_BOUND:
                continue
            st = (nh, min(lo, nh), max(hi, nh))
            if st not in dist:
                dist[st] = d + 1
                q.append(st)
    return dist


def test_optimal_length_matches_exhaustive_search():
    dist = bfs_distances()
    n_arc = 0
    circle_best = {}
    for (h, lo, hi), d in dist.items():
        if hi - lo >= CELLS_PER_TURN:
            k = h % CELLS_PER_TURN
            circle_best[k] = min(circle_best.get(k, d), d)
            continue
        n_arc += 1
        assert optimal_length(lo, hi, h) == d, (lo, hi, h)
    assert n_arc > 600, "the search space should be substantial"
    assert len(circle_best) == CELLS_PER_TURN
    for k, d in circle_best.items():
        assert optimal_length(-1, CELLS_PER_TURN, k) == d, k


@pytest.mark.parametrize("n", range(1, 13))
def test_invariants_over_every_sequence(n):
    """Feasibility, non-lengthening, and exact identity on optimal runs."""
    for bits in itertools.product(("L", "R"), repeat=n):
        src = seq("".join(bits))
        lo, hi, end = geometry(signs_of(src))
        out = canonicalize_run(src)
        clo, chi, cend = geometry(signs_of(out))

        assert len(out) <= n, "the human's run bounds the optimum"
        if hi - lo >= CELLS_PER_TURN:
            assert chi - clo >= CELLS_PER_TURN
            assert (cend - end) % CELLS_PER_TURN == 0
        else:
            assert (clo, chi, cend) == (lo, hi, end), "covers exactly S"
        if len(out) == n:
            assert out == src, "already-optimal runs are returned untouched"


# ----------------------------------------------------------------- sequencing
def test_runs_are_broken_by_every_non_turn_action():
    a = seq("LR") + ["LOOK_UP"] + seq("RL") + ["MOVE_FORWARD"] + seq("LL")
    assert find_rotation_runs(a) == [(0, 2), (3, 5), (6, 8)]


def test_non_turn_actions_are_never_touched():
    a = (
        ["MOVE_FORWARD"] * 3
        + seq("LRLR")
        + ["LOOK_DOWN", "MOVE_FORWARD"]
        + seq("LLL")
        + ["STOP"]
    )
    out, st = canonicalize_actions(a)
    assert out == ["MOVE_FORWARD"] * 3 + seq("LR") + [
        "LOOK_DOWN", "MOVE_FORWARD"
    ] + seq("LLL") + ["STOP"]
    assert st.actions_removed == 2
    assert st.runs == 2 and st.runs_altered == 1


def test_clean_episode_is_bit_identical():
    a = ["MOVE_FORWARD"] * 5 + seq("LLL") + ["MOVE_FORWARD"] * 4 + seq("LR") + ["STOP"]
    out, st = canonicalize_actions(a)
    assert out == a
    assert not st.touched and st.actions_removed == 0


# ------------------------------------------------------- the aligned triple
def _demo(actions, step=np.deg2rad(30.0)):
    """Build an (path_3d, path_headings, actions) triple consistent with actions."""
    pos = np.zeros(3, np.float32)
    p, h = [pos.copy()], [0.0]
    for a in actions:
        if a == "MOVE_FORWARD":
            pos = pos + np.array([0.25, 0, 0], np.float32)
        elif a == "TURN_LEFT":
            h.append(h[-1] + step)
            p.append(pos.copy())
            continue
        elif a == "TURN_RIGHT":
            h.append(h[-1] - step)
            p.append(pos.copy())
            continue
        p.append(pos.copy())
        h.append(h[-1])
    return np.array(p), np.array(h), actions


def test_canonicalize_demo_keeps_the_triple_aligned():
    actions = ["MOVE_FORWARD"] * 2 + seq("LRLR") + ["MOVE_FORWARD"] * 2 + ["STOP"]
    p, h, a = _demo(actions)
    p2, h2, a2, st = canonicalize_demo(p, h, a)
    assert len(p2) == len(a2) + 1 == len(h2)
    assert a2 == ["MOVE_FORWARD"] * 2 + seq("LR") + ["MOVE_FORWARD"] * 2 + ["STOP"]
    assert st.actions_removed == 2
    # the run's exit heading -- what the following drive phase inherits -- and
    # every pose outside the run must be untouched
    assert np.isclose(h2[4], h[6])
    assert np.allclose(p2[4:], p[6:])
    assert np.allclose(p2[:3], p[:3]) and np.allclose(h2[:3], h[:3])


def test_canonicalize_demo_handles_a_negated_heading_convention():
    """The per-primitive step is measured from the recording, not assumed."""
    actions = seq("LRLR") + ["STOP"]
    p, h, a = _demo(actions, step=-np.deg2rad(30.0))
    _, h2, a2, st = canonicalize_demo(p, h, a)
    assert txt(a2[:-1]) == "LR"
    assert np.isclose(h2[2], h[4])  # same exit heading despite the flipped sign


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_canonicalize_demo_is_the_identity_on_clean_data(dtype):
    actions = ["MOVE_FORWARD"] * 3 + seq("LLL") + ["MOVE_FORWARD"] + ["STOP"]
    p, h, a = _demo(actions)
    p, h = p.astype(dtype), h.astype(dtype)
    p2, h2, a2, st = canonicalize_demo(p, h, a)
    assert a2 == a and not st.touched
    assert p2 is p and h2 is h  # not copied, not upcast


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_canonicalize_demo_preserves_dtype(dtype):
    """The recorded arrays are float32; silently returning float64 would change
    every downstream pose by more than the edit itself does."""
    actions = seq("LRLR") + ["MOVE_FORWARD", "STOP"]
    p, h, a = _demo(actions)
    p2, h2, _, st = canonicalize_demo(p.astype(dtype), h.astype(dtype), a)
    assert st.touched
    assert p2.dtype == dtype and h2.dtype == dtype


def test_a_translating_run_is_left_exactly_as_recorded():
    """Real demos contain a handful of turn runs whose recorded position moves
    (staircase settling). The module's model does not cover those, so it
    declines to edit that one run rather than dropping the episode."""
    actions = seq("LRLR") + ["MOVE_FORWARD"] + seq("LRLR") + ["STOP"]
    p, h, a = _demo(actions)
    p[2] = p[2] + np.array([0.25, 0, 0], np.float32)  # break the first run only
    p2, h2, a2, st = canonicalize_demo(p, h, a)
    assert st.runs == 2
    assert st.runs_skipped_translating == 1
    assert st.runs_altered == 1
    # first run verbatim, second run canonicalised
    assert a2 == seq("LRLR") + ["MOVE_FORWARD"] + seq("LR") + ["STOP"]
    assert np.allclose(p2[:6], p[:6]) and np.allclose(h2[:6], h[:6])


def test_a_wholly_translating_episode_is_returned_untouched():
    actions = seq("LRLR")
    p, h, a = _demo(actions)
    p[2] = p[2] + np.array([0.25, 0, 0], np.float32)
    p2, h2, a2, st = canonicalize_demo(p, h, a)
    assert a2 == a and not st.touched and st.runs_skipped_translating == 1
    assert p2 is p and h2 is h
