"""How much of a *classic ObjectNav* episode budget a continuous agent has spent.

The benchmark this harness reports against caps an episode at 500 discrete
steps of ``MOVE_FORWARD`` (0.25 m) or ``TURN`` (30 degrees). A continuous agent
has neither, so every cap we have used so far measures the episode in a
currency the benchmark does not use:

* **policy steps** (``--max-steps``) are ``gap``-dependent. The same cap buys a
  1 Hz policy five times the motion it buys a 5 Hz one, so two checkpoints
  trained at different observation rates cannot be compared under it.
* **seconds** charge rotation about twice what a discrete step budget does --
  at the limits the demonstrations were recorded under, one second of driving
  is worth 8.00 budget units and one second of turning 3.82.

The unit here is the discrete step's *geometry*, taken literally:

    1 unit = 0.25 m translated  OR  pi/6 rad (30 degrees) rotated

accumulated per control tick from the planar pose the executor already
produces::

    budget += |d(x, y)| / 0.25 + |d(theta)| / (pi / 6)

That recovers the discrete step count with the no-ops removed: measured over
39,756 recorded episodes joined to their source ``reference_replay`` action
histograms, rotation units per ``TURN`` is 1.0000 (exact by construction --
every ``TURN`` is exactly 30 degrees) and translation units per
``MOVE_FORWARD`` is 0.8473, the shortfall being the 15.3% of ``MOVE_FORWARD``s
that are collision-blocked and displace the agent by nothing at all. (Those two
are corpus-wide figures; re-verified on one MP3D scene they come out 0.8074 and
16.4%. The rotation identity is exact and does not vary.)

**docs/GEOMETRIC_BUDGET.md** derives the conversions between discrete steps,
geometric units and seconds in both directions, states which of them are exact
and which are properties of the trajectory rather than of the robot, and gives
the measured numbers behind the paragraph above. Read it before using a
geometric cap to claim comparability with a published ObjectNav number.

## Why this class does not terminate anything by itself

A geometric budget is blind to exactly two behaviours, and both of them are
things a bad policy actually does:

* a **frozen** agent accrues nothing and would run forever;
* an agent **pushing into geometry** achieves no displacement, so it is free
  (measured ``blocked_drive_frac`` is 2.1-6.1% of ticks).

So the rule that uses this class always carries a sim-time backstop, and
:class:`~objectnav_eval.config.EvalConfig` refuses to be configured with the
geometric rule and no backstop. See :class:`BudgetRule`.

Mirrors ``continuous_demos.objectnav.PathTracker`` in spirit and in interface:
O(1) per sample, no navmesh, no rendering, no simulator query, so it is cheap
enough to feed every control tick. The caller owns the cadence, and sampling
coarser than the motion *under*-counts -- a chord is shorter than the arc it
subtends, and unobserved heading reversals cancel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

#: One discrete ``MOVE_FORWARD``.
UNIT_TRANSLATION_M = 0.25
#: One discrete ``TURN_LEFT`` / ``TURN_RIGHT``.
UNIT_ROTATION_RAD = math.pi / 6.0
#: The classic ObjectNav cap, in discrete steps and therefore in units.
DEFAULT_BUDGET_UNITS = 500.0
#: The mandatory sim-time backstop, seconds. Implies a floor of 1.67 units/s
#: for a 500-unit budget. Measured across 16 finished runs, per-episode rates
#: have p10 between 1.05 and 3.42 and the slowest single episode seen was 0.83
#: units/s: the backstop ends a genuinely stuck episode without touching a
#: healthy slow one, but on a badly-behaved policy it -- not the budget --
#: becomes the binding cap for the slowest tenth of episodes, which is why
#: `terminated_by` records which one fired.
DEFAULT_TIME_BACKSTOP_S = 300.0

#: Selectable termination rules. ``steps`` is the historical behaviour.
BUDGET_MODES = ("steps", "geometric", "both")


def wrap_angle(radians: float) -> float:
    """Signed angle difference in ``[-pi, pi]``.

    Without this a heading that crosses the pi/-pi seam reads as a 2*pi turn and
    charges 12 units for standing still.
    """
    return (float(radians) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class GeometricBudget:
    """Discrete-equivalent budget consumed, accumulated from planar poses.

    Translation and rotation are kept apart because the split is a finding in
    its own right and not merely bookkeeping: measured, human demonstrations
    spend ~36% of their budget rotating and our policies ~50%, which is the
    same statement as "the policies turn twice as much per metre travelled" in
    the currency the benchmark actually caps.
    """

    budget: float = DEFAULT_BUDGET_UNITS
    unit_translation_m: float = UNIT_TRANSLATION_M
    unit_rotation_rad: float = UNIT_ROTATION_RAD

    translation_units: float = 0.0
    rotation_units: float = 0.0
    translation_m: float = 0.0
    rotation_rad: float = 0.0
    num_samples: int = 0
    _previous: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.unit_translation_m <= 0 or self.unit_rotation_rad <= 0:
            raise ValueError("budget units must be positive lengths/angles")

    # ------------------------------------------------------------------
    def reset(self, pose2d: Optional[Sequence[float]] = None) -> None:
        self.translation_units = 0.0
        self.rotation_units = 0.0
        self.translation_m = 0.0
        self.rotation_rad = 0.0
        self.num_samples = 0
        self._previous = None
        if pose2d is not None:
            self.observe(pose2d)

    def observe(self, pose2d: Sequence[float]) -> float:
        """Charge the motion since the previous sample. Returns ``spent``.

        The first sample only seeds the reference pose; seed it with the spawn
        pose or the first chunk's motion is charged from wherever tick 1 landed.
        """
        pose = np.asarray(pose2d, dtype=float)
        if self._previous is not None:
            step_m = float(np.linalg.norm(pose[:2] - self._previous[:2]))
            step_rad = abs(wrap_angle(float(pose[2]) - float(self._previous[2])))
            self.translation_m += step_m
            self.rotation_rad += step_rad
            self.translation_units += step_m / self.unit_translation_m
            self.rotation_units += step_rad / self.unit_rotation_rad
        self._previous = pose[:3].copy()
        self.num_samples += 1
        return self.spent

    # ------------------------------------------------------------------
    @property
    def spent(self) -> float:
        return self.translation_units + self.rotation_units

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.budget

    @property
    def rotation_fraction(self) -> float:
        """Share of the budget spent turning. NaN before anything moves."""
        spent = self.spent
        return self.rotation_units / spent if spent > 0 else float("nan")

    def summary(self) -> Dict[str, Any]:
        """The per-episode record. Written whether or not the rule is active.

        Costs nothing next to a trajectory log and is what lets a finished run
        be re-scored under a different cap (``scripts/rescore_budget.py``)
        instead of re-run.
        """
        return {
            "units": self.spent,
            "translation_units": self.translation_units,
            "rotation_units": self.rotation_units,
            "rotation_fraction": self.rotation_fraction,
            "translation_m": self.translation_m,
            "rotation_rad": self.rotation_rad,
            "budget_units": self.budget,
            "exhausted": self.exhausted,
            "unit_translation_m": self.unit_translation_m,
            "unit_rotation_rad": self.unit_rotation_rad,
            "samples": self.num_samples,
        }


# ---------------------------------------------------------------------------
# The termination rule
# ---------------------------------------------------------------------------

#: Canonical termination codes, reported as ``terminated_by`` alongside the
#: human-readable ``terminated`` string (whose historical values are left
#: untouched so that finished runs and the scripts that read them still parse).
TERMINATED_BUDGET = "budget_exhausted"
TERMINATED_TIME = "time_limit"
TERMINATED_STEPS = "max_steps"
TERMINATED_ORACLE = "oracle_stop"
TERMINATED_POLICY = "policy_stop"
TERMINATED_TIPPED = "tipped_over"

#: ``terminated`` -> ``terminated_by``. Anything unlisted (the oracle's several
#: phrasings) falls back to ``oracle_stop`` via :func:`terminated_code`.
_TERMINATION_CODES = {
    "max steps": TERMINATED_STEPS,
    "policy called stop": TERMINATED_POLICY,
    "robot tipped over": TERMINATED_TIPPED,
    "budget exhausted": TERMINATED_BUDGET,
    "sim time limit": TERMINATED_TIME,
}


def terminated_code(terminated: Optional[str]) -> str:
    """Canonical code for a ``terminated`` string, old runs included."""
    if terminated is None:
        return TERMINATED_STEPS
    text = str(terminated)
    if text in _TERMINATION_CODES:
        return _TERMINATION_CODES[text]
    if text.startswith("auto-stop") or text.startswith("oracle stop"):
        return TERMINATED_ORACLE
    return text


@dataclass
class BudgetRule:
    """Geometric budget + its mandatory sim-time backstop, as one decision.

    Constructed only when the rule is active. It answers one question per
    control tick -- "is this episode over" -- and says which cap said so. The
    backstop is not optional and not defaulted-away: a rule with a non-positive
    or non-finite ``time_limit_s`` raises here, so the failure mode is a usage
    error rather than a run that hangs on a frozen policy.
    """

    budget: GeometricBudget
    time_limit_s: float = DEFAULT_TIME_BACKSTOP_S

    def __post_init__(self) -> None:
        if not (self.time_limit_s > 0 and math.isfinite(self.time_limit_s)):
            raise ValueError(
                "a geometric budget must carry a finite, positive sim-time "
                "backstop: it is blind to a frozen agent (which accrues nothing) "
                "and to one pushing into a wall (which achieves no displacement, "
                "and so spends nothing)"
            )
        if not (self.budget.budget > 0 and math.isfinite(self.budget.budget)):
            raise ValueError(f"budget must be positive and finite, got {self.budget.budget}")

    @property
    def units_per_second_floor(self) -> float:
        """The rate below which the backstop, not the budget, ends an episode."""
        return self.budget.budget / self.time_limit_s

    def exceeded(self, sim_time: float) -> Optional[str]:
        """Which cap has fired at this instant, or None."""
        if self.budget.exhausted:
            return TERMINATED_BUDGET
        if sim_time >= self.time_limit_s:
            return TERMINATED_TIME
        return None


# ---------------------------------------------------------------------------
# Offline: the same accumulation over a recorded trajectory
# ---------------------------------------------------------------------------


@dataclass
class BudgetTrace:
    """Per-tick budget over a recorded trajectory, and where a cap would cut.

    ``cumulative[i]`` is the budget spent by the end of tick ``i`` of
    ``poses``. The first pose seeds the accumulation and therefore costs zero,
    exactly as in the live rule.
    """

    cumulative: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    sim_time: Optional[np.ndarray] = None

    @property
    def total(self) -> float:
        return float(self.cumulative[-1]) if len(self.cumulative) else 0.0

    def truncation_tick(self, budget_units: float) -> Optional[int]:
        """Index of the first tick at or past ``budget_units``, or None."""
        hits = np.flatnonzero(self.cumulative >= float(budget_units))
        return int(hits[0]) if len(hits) else None

    def summary(self, budget_units: float = DEFAULT_BUDGET_UNITS) -> Dict[str, Any]:
        total = self.total
        cut = self.truncation_tick(budget_units)
        out: Dict[str, Any] = {
            "units": total,
            "translation_units": float(self.translation[-1]) if len(self.translation) else 0.0,
            "rotation_units": float(self.rotation[-1]) if len(self.rotation) else 0.0,
            "rotation_fraction": (
                float(self.rotation[-1]) / total if total > 0 else float("nan")
            ),
            "ticks": int(len(self.cumulative)),
            "budget_units": float(budget_units),
            "truncated": cut is not None,
            "truncated_at_tick": cut,
        }
        if self.sim_time is not None and len(self.sim_time):
            out["sim_seconds"] = float(self.sim_time[-1])
            out["units_per_sim_second"] = (
                total / float(self.sim_time[-1]) if self.sim_time[-1] > 0 else float("nan")
            )
            if cut is not None:
                out["truncated_at_sim_seconds"] = float(self.sim_time[cut])
        return out


def budget_trace(
    poses: Iterable[Sequence[float]],
    sim_time: Optional[Sequence[float]] = None,
    unit_translation_m: float = UNIT_TRANSLATION_M,
    unit_rotation_rad: float = UNIT_ROTATION_RAD,
) -> BudgetTrace:
    """Accumulate a budget over a recorded ``[X, Y, theta]`` sequence.

    The offline twin of :meth:`GeometricBudget.observe`, deliberately sharing
    its arithmetic rather than reimplementing it -- a second, "equivalent"
    accumulator would not raise when it drifted from the live one.
    """
    tracker = GeometricBudget(
        unit_translation_m=unit_translation_m, unit_rotation_rad=unit_rotation_rad
    )
    cumulative: List[float] = []
    translation: List[float] = []
    rotation: List[float] = []
    for pose in poses:
        tracker.observe(pose)
        cumulative.append(tracker.spent)
        translation.append(tracker.translation_units)
        rotation.append(tracker.rotation_units)
    times = None if sim_time is None else np.asarray(list(sim_time), dtype=float)
    return BudgetTrace(
        cumulative=np.asarray(cumulative, dtype=float),
        translation=np.asarray(translation, dtype=float),
        rotation=np.asarray(rotation, dtype=float),
        sim_time=times,
    )


def rescore_record(
    record: Dict[str, Any], budget_units: float = DEFAULT_BUDGET_UNITS, gap: int = 5
) -> Optional[Dict[str, Any]]:
    """What a ``budget_units`` cap would have done to one finished episode.

    Honest about what is and is not recoverable from a finished run:

    * **Recoverable exactly**: the budget spent, its translation/rotation split,
      the tick at which a cap would have cut, and the path length and sim time
      at that tick. All of it is arithmetic on the logged poses.
    * **Recoverable, with a one-chunk ambiguity**: ``oracle_success``. It latches
      the first time the agent comes within the success distance and never
      un-latches, so a truncated episode keeps it iff that moment preceded the
      cut. The log records the *chunk* it happened in, not the tick, so an
      episode whose arrival chunk straddles the cut is reported as ambiguous
      rather than guessed.
    * **NOT recoverable**: ``success``, ``spl``, ``soft_spl``, ``npwpl`` and the
      final distance. All of them are read at the episode's final pose, and a
      truncated episode has a different final pose. Re-scoring those needs a
      re-run; this function deliberately does not estimate them.
    """
    trace = trace_from_record(record)
    if trace is None:
        return None
    summary = trace.summary(budget_units)
    cut = summary["truncated_at_tick"]
    out: Dict[str, Any] = {
        "uid": record.get("uid"),
        "scene": record.get("scene"),
        "category": record.get("category"),
        "terminated": record.get("terminated"),
        "steps": record.get("steps"),
        "oracle_success": record.get("oracle_success"),
        "first_reach_step": record.get("first_reach_step"),
        "path_length_m": record.get("path_length_m"),
        **summary,
    }
    unit_m = UNIT_TRANSLATION_M
    if cut is None:
        out["oracle_success_under_cap"] = record.get("oracle_success")
        out["oracle_ambiguous"] = False
        return out
    out["truncated_at_step"] = int(cut // max(1, gap))
    out["path_at_truncation_m"] = float(trace.translation[cut]) * unit_m
    reach = record.get("first_reach_step")
    if reach is None:
        lo = hi = False
    else:
        # Ticks of chunk s are [s*gap, (s+1)*gap). The arrival tick is somewhere
        # inside chunk `reach`; `lo` assumes the last tick of it, `hi` the first.
        lo = (int(reach) + 1) * gap - 1 <= cut
        hi = int(reach) * gap <= cut
    out["oracle_success_under_cap"] = bool(hi)
    out["oracle_success_under_cap_conservative"] = bool(lo)
    out["oracle_ambiguous"] = bool(lo != hi)
    return out


def rescore_results(
    results: Sequence[Dict[str, Any]],
    budget_units: float = DEFAULT_BUDGET_UNITS,
    gap: int = 5,
) -> Dict[str, Any]:
    """:func:`rescore_record` over a whole run, plus the aggregate."""
    rows = []
    missing = 0
    for record in results:
        row = rescore_record(record, budget_units=budget_units, gap=gap)
        if row is None:
            missing += 1
            continue
        rows.append(row)
    units = np.asarray([row["units"] for row in rows], dtype=float) if rows else np.zeros(0)
    truncated = [row for row in rows if row["truncated"]]
    aggregate: Dict[str, Any] = {
        "budget_units": float(budget_units),
        "episodes": len(rows),
        # Episodes whose record carried no trajectory log: reported, never
        # silently counted as zero-budget episodes.
        "episodes_without_trajectory": missing,
        "truncated": len(truncated),
        "truncated_fraction": len(truncated) / len(rows) if rows else float("nan"),
        "units_mean": float(units.mean()) if len(units) else float("nan"),
        "units_p50": float(np.percentile(units, 50)) if len(units) else float("nan"),
        "units_p95": float(np.percentile(units, 95)) if len(units) else float("nan"),
        "units_max": float(units.max()) if len(units) else float("nan"),
        "rotation_fraction_mean": float(
            np.nanmean([row["rotation_fraction"] for row in rows])
        ) if rows else float("nan"),
        "units_per_sim_second_mean": float(
            np.nanmean([row.get("units_per_sim_second", np.nan) for row in rows])
        ) if rows else float("nan"),
        "oracle_success": float(
            np.mean([bool(row.get("oracle_success")) for row in rows])
        ) if rows else float("nan"),
        "oracle_success_under_cap": float(
            np.mean([bool(row.get("oracle_success_under_cap")) for row in rows])
        ) if rows else float("nan"),
        "oracle_ambiguous": sum(1 for row in rows if row.get("oracle_ambiguous")),
    }
    return {"aggregate": aggregate, "episodes": rows}


def trace_from_record(record: Dict[str, Any]) -> Optional[BudgetTrace]:
    """A :class:`BudgetTrace` from one episode record's trajectory log.

    Returns None when the run was made with ``--no-trajectory-log`` (or predates
    it), which is the one thing that makes an on-disk run un-rescorable. The
    caller reports that as "no trajectory", never as zero budget.

    One known, bounded discrepancy against the live accumulator: the trajectory
    log's first row is the pose *after* the first control tick, not the spawn
    pose, so this trace misses that first tick's motion. The live rule seeds
    from the spawn pose and does not. The gap is at most one tick -- under 0.4
    units against episode totals of 100-300 -- and it is one-sided (offline
    under-counts). ``record["budget"]["units"]`` is the exact figure where it is
    present; this function exists for the caps and the runs that number does not
    cover.
    """
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
        return None
    rows = trajectory.get("tick_pose")
    if not rows:
        return None
    #: ``[sim_time, X, Y, theta]`` per control tick -- see probes.TrajectoryLog.
    poses = [row[1:4] for row in rows]
    times = [row[0] for row in rows]
    return budget_trace(poses, sim_time=times)
