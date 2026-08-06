"""Remove redundant rotation from a demo's action sequence, before replay.

------------------------------------------------------------------------
What this removes, and why it is free
------------------------------------------------------------------------
A *rotation run* is a maximal block of consecutive ``TURN_LEFT``/``TURN_RIGHT``
actions. Inside one the agent does not translate, so the position is fixed and
the set of headings the run faced is *exactly* the set of views it collected.
A run therefore contributes exactly three things:

  * the covered heading set ``S`` (an arc, or the whole circle),
  * the heading it started at, ``theta_s``,
  * the heading it left on, ``theta_e``.

Anything beyond the shortest sequence achieving those three is rotation that
collected nothing new -- a human re-glancing at a scene that has not changed
because their working memory has, not the world. Replacing every run with a
minimal equivalent is signal-preserving compression, not a filter with a
threshold: there is no tolerance parameter, no epsilon, and nothing to tune.

The human's primitives are exactly +/-30 degrees, so every heading in a run
lies on a 30-degree grid and the whole computation is integer arithmetic.

------------------------------------------------------------------------
The construction
------------------------------------------------------------------------
Units below are 30-degree cells. Headings are *unwrapped* within a run so the
arc arithmetic is well defined; coverage is judged on the circle, so an
unwrapped span of >= 12 cells has traversed all 12 residues and covers
everything.

**Arc case** (span < 12). Rotation moves continuously along the heading axis,
so covering an arc is exactly visiting its two extremes, and there are only two
orders::

    A:  s -> lo -> hi -> e      cost (s - lo) + (hi - lo) + (hi - e)
    B:  s -> hi -> lo -> e      cost (hi - s) + (hi - lo) + (e - lo)

Take the cheaper; on a tie take the one that turns the way the human turned.

**Full-circle case** (span >= 12). There are no extremes to visit. Note that
``theta_s`` and ``theta_e`` are still *independent* here -- "full circle" does
not mean the run came back to where it started, and assuming it did produces a
sequence with the right coverage and the **wrong end heading**, which would
silently corrupt the handoff into the following drive phase. With
``d = (e - s) mod 12``::

    cost = 12 + min(d, 12 - d)

swept **monotonically** the short way round. A reversal can only add
backtracking on top of a total variation that already has to reach 360 degrees,
so monotone is optimal and there is no two-route comparison. The turn direction
is opposite to the sign of ``d`` when ``d > 6``: ``d = 9`` (270 degrees) sweeps
*-450* degrees, not +450.

An already-minimal run is returned **unchanged, byte for byte** -- not
re-derived -- so this is an exact no-op on clean data. Measured over all 56
train scenes: 99.17% of the 2.35M rotation runs are already optimal.

------------------------------------------------------------------------
Why the action sequence and not the recorded frames
------------------------------------------------------------------------
``dump/data_diagnostics/croppability/FINDINGS.md`` measured what happens when
you cut a *recorded* trajectory: the seam puts a median 7.0 degrees (max 81)
into a single 0.05 s tick, in source data capped at 4.6 deg/tick, which is an
action-target corruption rather than a visual one. Editing the discrete action
sequence before replay has no seam at all -- the shortened run is re-rendered
along its own path by the same controller as everything else. That is the whole
reason this operates here.

------------------------------------------------------------------------
Correctness
------------------------------------------------------------------------
``tests/test_rotation_canonicalize.py`` proves the construction by exhaustive
breadth-first search over the discrete state space ``(heading, min, max)``,
which yields the true minimum for every reachable ``(range, start, end)``
target simultaneously, and checks it against every one -- including all eight
full-circle ``d`` cases and both tie directions. Separately, the human's own
run is feasible and therefore an upper bound on the optimum, so ``|C| <= |R|``
must hold for every real run; that was checked over all 2,353,082 rotation runs
in the corpus with zero violations, alongside a simulated feasibility check on
every altered run. See ``dump/rotation_blunders/FINDINGS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
TURN_ACTIONS = (TURN_LEFT, TURN_RIGHT)

CELLS_PER_TURN = 12  # 360 / 30
TURN_DEG = 30.0


# --------------------------------------------------------------------- geometry
def _geometry(signs: Sequence[int]) -> Tuple[int, int, int]:
    """``(lo, hi, end)`` in grid cells for a +/-1 sequence starting at 0."""
    h = lo = hi = 0
    for s in signs:
        h += s
        if h < lo:
            lo = h
        elif h > hi:
            hi = h
    return lo, hi, h


def optimal_length(lo: int, hi: int, end: int, start: int = 0) -> int:
    """Length, in turn primitives, of the shortest run with this geometry."""
    if hi - lo >= CELLS_PER_TURN:
        d = (end - start) % CELLS_PER_TURN
        return CELLS_PER_TURN + min(d, CELLS_PER_TURN - d)
    a = (start - lo) + (hi - lo) + (hi - end)
    b = (hi - start) + (hi - lo) + (end - lo)
    return min(a, b)


def _emit(legs) -> List[int]:
    out: List[int] = []
    for s, k in legs:
        out.extend([s] * k)
    return out


def _first_dir(legs) -> int:
    for s, k in legs:
        if k:
            return s
    return 0


def optimal_signs(
    lo: int, hi: int, end: int, start: int = 0, prefer: int = 1
) -> List[int]:
    """The shortest +/-1 sequence with this geometry.

    ``prefer`` is the human's own first turn direction. The optimum is not
    unique and the *ordering* of views is the only thing left to choose, so
    where the routes tie the tie goes to the human.
    """
    if hi - lo >= CELLS_PER_TURN:
        r = (end - start) % CELLS_PER_TURN
        back = CELLS_PER_TURN - r
        if r == 0:  # one spin, either way
            s, k = (1 if prefer >= 0 else -1), CELLS_PER_TURN
        elif r < back:
            s, k = 1, CELLS_PER_TURN + r
        elif back < r:
            s, k = -1, CELLS_PER_TURN + back
        else:  # d == 180: both directions tie, keep the human's
            s, k = (1 if prefer >= 0 else -1), CELLS_PER_TURN + r
        return _emit([(s, k)])

    a = (start - lo) + (hi - lo) + (hi - end)
    b = (hi - start) + (hi - lo) + (end - lo)
    legs_a = [(-1, start - lo), (1, hi - lo), (-1, hi - end)]
    legs_b = [(1, hi - start), (-1, hi - lo), (1, end - lo)]
    if a < b:
        return _emit(legs_a)
    if b < a:
        return _emit(legs_b)
    return _emit(legs_b if _first_dir(legs_b) == prefer else legs_a)


def canonicalize_run(actions: Sequence[str]) -> List[str]:
    """Canonicalise one run of pure turn actions. Identity if already minimal."""
    signs = [1 if a == TURN_LEFT else -1 for a in actions]
    lo, hi, end = _geometry(signs)
    if len(signs) == optimal_length(lo, hi, end):
        return list(actions)
    out = optimal_signs(lo, hi, end, 0, signs[0])
    return [TURN_LEFT if s > 0 else TURN_RIGHT for s in out]


# ------------------------------------------------------------------------ runs
def find_rotation_runs(actions: Sequence[str]) -> List[Tuple[int, int]]:
    """Maximal ``[start, end)`` blocks of consecutive turn actions.

    Broken by *anything* else -- ``MOVE_FORWARD``, ``LOOK_UP``, ``LOOK_DOWN``,
    ``STOP``. That is the strictest reading of "pure rotation": a run contains
    yaw and nothing else, so a LOOK in the middle splits one block into two
    rather than being silently treated as free. This is deliberately NOT
    ``episode_io.find_looking_slices``, which segments an episode into look and
    move phases for the replay controller and admits non-yaw actions; the two
    answer different questions and must not be conflated.
    """
    runs: List[Tuple[int, int]] = []
    i, n = 0, len(actions)
    while i < n:
        if actions[i] not in TURN_ACTIONS:
            i += 1
            continue
        j = i
        while j < n and actions[j] in TURN_ACTIONS:
            j += 1
        runs.append((i, j))
        i = j
    return runs


@dataclass
class CanonicalizeStats:
    """What canonicalisation did to one episode. All zero means untouched."""

    runs: int = 0
    runs_altered: int = 0
    runs_skipped_translating: int = 0
    actions_before: int = 0
    actions_removed: int = 0
    rotation_deg_before: float = 0.0
    rotation_deg_removed: float = 0.0
    edits: List[Tuple[int, int, int]] = field(default_factory=list)  # start, old, new

    @property
    def touched(self) -> bool:
        return self.runs_altered > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runs": self.runs,
            "runs_altered": self.runs_altered,
            "runs_skipped_translating": self.runs_skipped_translating,
            "actions_before": self.actions_before,
            "actions_removed": self.actions_removed,
            "rotation_deg_before": self.rotation_deg_before,
            "rotation_deg_removed": self.rotation_deg_removed,
        }


def canonicalize_actions(
    actions: Sequence[str],
) -> Tuple[List[str], CanonicalizeStats]:
    """Canonicalise every rotation run in an action sequence."""
    st = CanonicalizeStats(actions_before=len(actions))
    out: List[str] = []
    prev = 0
    for i, j in find_rotation_runs(actions):
        out.extend(actions[prev:i])
        run = actions[i:j]
        new = canonicalize_run(run)
        st.runs += 1
        st.rotation_deg_before += len(run) * TURN_DEG
        if len(new) != len(run):
            st.runs_altered += 1
            st.actions_removed += len(run) - len(new)
            st.rotation_deg_removed += (len(run) - len(new)) * TURN_DEG
            st.edits.append((i, len(run), len(new)))
        out.extend(new)
        prev = j
    out.extend(actions[prev:])
    return out, st


# ------------------------------------------------------------- the demo triple
def _wrap(a: np.ndarray | float):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def canonicalize_demo(
    path_3d: np.ndarray,
    path_headings: np.ndarray,
    actions: Sequence[str],
    *,
    position_atol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, List[str], CanonicalizeStats]:
    """Canonicalise an ``episode_io.extract_demo`` triple, keeping it aligned.

    ``actions[j]`` carries the robot from ``path_3d[j]`` to ``path_3d[j + 1]``,
    so a run occupying actions ``[i, j)`` owns path points ``i + 1 .. j``. A
    turn does not translate, so every position in that window is the same and
    only the headings need rebuilding.

    The per-primitive heading step is *measured from this episode's own
    recorded headings* rather than assumed to be +30 degrees, so the sign
    convention of ``hab_quat_to_heading`` cannot be got wrong here.

    A run whose recorded position is **not** constant is left exactly as the
    human recorded it and counted in ``runs_skipped_translating``. Those exist:
    over three scenes (138,211 runs) 8 of them move, all on staircases, where
    the recorded state carries a settling displacement of ~0.24 m into the
    frame after a forward move. The module's model does not cover that, so it
    declines rather than guesses -- and it declines for that one run, not the
    episode, because dropping the episode would cost more good data than the
    edit could ever return. Genuine turn runs are bit-exactly constant, so
    ``position_atol`` separates the two cases by two orders of magnitude.
    """
    path_3d = np.asarray(path_3d)
    path_headings = np.asarray(path_headings)
    if len(path_3d) != len(actions) + 1:
        raise ValueError(
            f"expected len(path_3d) == len(actions) + 1, got "
            f"{len(path_3d)} and {len(actions)}"
        )

    st = CanonicalizeStats(actions_before=len(actions))
    keep_p: List[np.ndarray] = [path_3d[0]]
    keep_h: List[float] = [float(path_headings[0])]
    out_actions: List[str] = []
    prev = 0
    for i, j in find_rotation_runs(actions):
        # untouched span before the run
        for k in range(prev, i):
            keep_p.append(path_3d[k + 1])
            keep_h.append(float(path_headings[k + 1]))
        out_actions.extend(actions[prev:i])
        prev = j

        run = list(actions[i:j])
        st.runs += 1
        st.rotation_deg_before += len(run) * TURN_DEG
        window = path_3d[i : j + 1]
        translating = not np.allclose(window, window[0], atol=position_atol)
        new = run if translating else canonicalize_run(run)
        if translating:
            st.runs_skipped_translating += 1
        if len(new) == len(run):  # unchanged: replay the recording verbatim
            for k in range(i, j):
                keep_p.append(path_3d[k + 1])
                keep_h.append(float(path_headings[k + 1]))
            out_actions.extend(run)
            continue

        st.runs_altered += 1
        st.actions_removed += len(run) - len(new)
        st.rotation_deg_removed += (len(run) - len(new)) * TURN_DEG
        st.edits.append((i, len(run), len(new)))
        # Signed heading change per TURN_LEFT, measured from the recording, so
        # the sign convention of hab_quat_to_heading cannot be got wrong here.
        # float64 for the arithmetic only; the output keeps the input dtype.
        head = path_headings[i : j + 1].astype(np.float64)
        signs = np.array([1.0 if a == TURN_LEFT else -1.0 for a in run])
        unit = float(np.median(_wrap(head[1:] - head[:-1]) * signs))
        h0, pos, acc = float(path_headings[i]), path_3d[i], 0.0
        for a in new:
            acc += unit if a == TURN_LEFT else -unit
            keep_p.append(pos)
            keep_h.append(float(_wrap(h0 + acc)))
        out_actions.extend(new)

    for k in range(prev, len(actions)):
        keep_p.append(path_3d[k + 1])
        keep_h.append(float(path_headings[k + 1]))
    out_actions.extend(actions[prev:])

    if not st.touched:  # nothing changed -> hand back the originals untouched
        return path_3d, path_headings, list(actions), st
    out_p = np.asarray(keep_p, dtype=path_3d.dtype)
    out_h = np.asarray(keep_h, dtype=path_headings.dtype)
    assert len(out_p) == len(out_actions) + 1 == len(out_h)
    return out_p, out_h, out_actions, st


def maybe_canonicalize(
    path_3d: np.ndarray,
    path_headings: np.ndarray,
    actions: Sequence[str],
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[CanonicalizeStats]]:
    """Call-site helper: a strict no-op when ``enabled`` is False."""
    if not enabled:
        return path_3d, path_headings, list(actions), None
    return canonicalize_demo(path_3d, path_headings, actions)
