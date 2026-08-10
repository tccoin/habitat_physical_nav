"""PointNav episode design: what to sample, what to reject, and when to roll back.

Everything here is arithmetic over numbers a simulator produced -- no ``habitat_sim``
import, no ``robot_sim``, no rendering. The half that touches the simulator lives in
:mod:`continuous_demos.pointnav_episode`. Splitting them is what makes the design
decisions testable at all: "is a 1.4 m goal too close", "does a rollback leave the
recorded arrays consistent", and "does a chained episode land on exactly 400 frames" are
all answerable without loading a scene.

------------------------------------------------------------------------
The shape of an episode
------------------------------------------------------------------------
One episode is a chain of **segments**, physically continuous: segment ``k + 1`` starts
from wherever segment ``k`` finished, so the pose stream never jumps and the video never
cuts. Each segment is

    sample a goal -> plan a shortest path -> turn in place onto it -> drive it

and the episode ends when the frame target is reached. That is what lets every episode be
exactly the same length, which is the point: a fixed-length row wastes no window capacity
and makes batches uniform.

------------------------------------------------------------------------
Two rejection points, and why the second one rolls back
------------------------------------------------------------------------
**Sampling time** -- :func:`screen_segment` rejects a candidate goal before the robot has
moved. Costs nothing; just draw again. ``segment_retries`` of these per segment.

**Drive time** -- the robot tried and failed: it did not arrive, or it wandered, or it
fell through the floor. The naive responses are both bad. Aborting throws away the
``k - 1`` good segments already recorded. Truncating to the last good segment throws away
the episode anyway, because a short episode fails the exact-frame gate.

So the default is to **roll back**: discard segment ``k``'s frames, put the robot back
where segment ``k - 1`` left it, and draw a different goal. The recorded arrays are
truncated in memory (nothing is written until the whole episode passes), so the failed
excursion is *erased* rather than relabelled -- which is the property that matters.
Relabelling it under the new goal would make ``goal2d`` non-piecewise-constant, and the
model would have to learn "goals move when you fail".

Two consequences the rest of the design has to honour, both easy to get wrong:

* the health monitor is **per segment**, so a rolled-back segment's ticks cannot poison
  the episode's escape / ref-miss verdict. This also makes falling through an MP3D mesh
  hole a property of the route rather than of the episode.
* a rolled-back segment is still **counted**. If rollbacks were silent the corpus would
  read cleaner than the scenes are, and every threshold re-cut from the pilot would be
  fit to a lie.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from continuous_demos.pid_pose_controller import wrap_angle

# ===========================================================================
# Reason codes
# ===========================================================================
# Same shape as ``objectnav_eval.screening``: a machine-readable code plus prose, so the
# failure log's reason column is `code: detail` and `cut -d: -f1` over it is a histogram.

# -- sampling time: retried, not failures -----------------------------------
NO_PATH = "no_path"
GOAL_OFF_NAVMESH = "goal_off_navmesh"
GOAL_OTHER_ISLAND = "goal_other_island"
GOAL_OTHER_FLOOR = "goal_other_floor"
GEODESIC_TOO_SHORT = "geodesic_too_short"
GEODESIC_TOO_LONG = "geodesic_too_long"
EUCLID_TOO_SHORT = "euclid_too_short"
TURN_TOO_SMALL = "turn_too_small"

# -- episode setup: fatal ---------------------------------------------------
NO_ISLAND = "no_island"
START_OFF_NAVMESH = "start_off_navmesh"
SEGMENT_UNSAMPLED = "segment_unsampled"

# -- drive time, segment-scoped: recoverable by rollback --------------------
NOT_REACHED = "not_reached"
INEFFICIENT = "inefficient"
SEGMENT_TIMEOUT = "segment_timeout"
TURN_TIMEOUT = "turn_timeout"
ESCAPED = "escaped"
REF_MISS = "ref_miss"
TIPPED_OVER = "tipped_over"

# -- drive time, episode-scoped: fatal --------------------------------------
TOO_FEW_FRAMES = "too_few_frames"
BUDGET_EXHAUSTED = "budget_exhausted"
ROLLBACKS_EXHAUSTED = "rollbacks_exhausted"

#: Codes a rollback can recover from. Everything else ends the episode.
SEGMENT_SCOPED = frozenset(
    {NOT_REACHED, INEFFICIENT, SEGMENT_TIMEOUT, TURN_TIMEOUT, ESCAPED, REF_MISS, TIPPED_OVER}
)

#: Codes raised while drawing a candidate goal, before the robot has moved.
SAMPLING_SCOPED = frozenset(
    {NO_PATH, GOAL_OFF_NAVMESH, GOAL_OTHER_ISLAND, GOAL_OTHER_FLOOR,
     GEODESIC_TOO_SHORT, GEODESIC_TOO_LONG, EUCLID_TOO_SHORT, TURN_TOO_SMALL}
)

REASON_TEXT: Dict[str, str] = {
    NO_PATH: "the planner found no path between the sampled endpoints",
    GOAL_OFF_NAVMESH: "the sampled goal did not snap onto the robot navmesh",
    GOAL_OTHER_ISLAND: "start and goal are on different islands -- the robot does not fit through",
    GOAL_OTHER_FLOOR: "start and goal are on different storeys",
    GEODESIC_TOO_SHORT: "the path is too short to be more than an approach manoeuvre",
    GEODESIC_TOO_LONG: "the path would consume the whole episode frame budget",
    EUCLID_TOO_SHORT: "the path loops back, so the goal looks like the start",
    TURN_TOO_SMALL: "the robot is already facing the path",
    NO_ISLAND: "no navmesh island in this scene is big enough to hold an episode",
    START_OFF_NAVMESH: "the robot's own position did not snap onto the navmesh",
    SEGMENT_UNSAMPLED: "no acceptable goal was found within the retry budget",
    NOT_REACHED: "the robot stopped short of the goal",
    INEFFICIENT: "the robot reached the goal but took a much longer route than the plan",
    SEGMENT_TIMEOUT: "the drive phase ran out of local time",
    TURN_TIMEOUT: "the turn-in-place ran out of local time",
    ESCAPED: "the robot left the world (joint_z free-fall)",
    REF_MISS: "the robot left its own planned path -- it was fighting geometry",
    TIPPED_OVER: "the robot ended up on its side",
    TOO_FEW_FRAMES: "the episode could not be filled to the frame target",
    BUDGET_EXHAUSTED: "the episode spent its motion or sim-time budget",
    ROLLBACKS_EXHAUSTED: "too many segments failed to drive",
}


def describe(code: str, detail: str = "") -> str:
    """``"code: detail"``, the failure log's reason column.

    Code first and colon-separated so a histogram is one ``cut``. Falls back to the
    canonical prose when the caller has no specific detail.
    """
    return f"{code}: {detail or REASON_TEXT.get(code, code)}"


# ===========================================================================
# Sampling configuration
# ===========================================================================
@dataclass(frozen=True)
class PointNavSampleConfig:
    """What counts as an acceptable segment. Every default has a reason, below.

    Recorded verbatim into each episode's ``sampling`` block, so a corpus says what it
    was cut with rather than requiring the reader to find the run's command line.
    """

    # -- island ------------------------------------------------------------
    #: The robot navmesh (r=0.28, h=1.0) is strictly more fragmented than the shipped
    #: point-agent one, so a lot of its islands are closets and stair landings. Measured
    #: on 17DRP5sb8fy: 3 islands of area 24.2 / 1.3 / 2.9 m2, of which only the first is
    #: a floor. A 1.5 m geodesic does not fit in the other two.
    min_island_area_m2: float = 8.0
    min_island_radius_m: float = 2.0

    # -- per segment -------------------------------------------------------
    #: Below this the segment is mostly an approach manoeuvre rather than navigation:
    #: ``HolonomicControllerConfig`` has ``goal_tolerance_dist=0.15`` and
    #: ``approach_velocity_scaling_dist=0.5``, so a 1 m segment is a third approach.
    min_geodesic_m: float = 1.5
    #: Above this one segment eats the whole frame budget and the "episode" is a single
    #: goal, which defeats the point of chaining.
    max_geodesic_m: float = 8.0
    #: Rejects a long path that loops back to near its start -- the goal would be
    #: visually indistinguishable from where the robot began.
    min_euclid_m: float = 1.0
    #: Admits a ramp, excludes a storey (MP3D storeys are ~3 m). Deliberately NOT
    #: ``agent_max_climb`` (0.2), which is what ``sim_debug_drive.py`` used and which
    #: forbids ramps.
    max_height_delta_m: float = 0.5

    #: OFF by default, and it should stay off. Gating on a minimum turn biases the corpus
    #: against straight corridors, which are real and which a policy must handle. The
    #: behaviour we actually want in the data -- "the goal is not where you are facing" --
    #: is bought instead by randomising the spawn heading (see ``random_spawn_heading``),
    #: which makes segment 0 open with a turn of expected magnitude 90 degrees without
    #: deleting any geometry. Implemented, and reported per segment as ``turn_rad``, so
    #: the distribution can be looked at before anyone decides to gate on it.
    min_turn_deg: float = 0.0

    #: Draws per segment before giving up on it.
    segment_retries: int = 30

    #: OFF (0) by default, which keeps the draw order byte-identical to every corpus cut
    #: before this existed. When > 0, a segment draws a target distance uniformly from
    #: ``[min_geodesic_m, max_geodesic_m]``, collects up to this many candidates that pass
    #: the *same* screening, and keeps the one closest to the target.
    #:
    #: Why it exists: drawing a navigable point uniformly and accepting the first that
    #: passes gives the distribution the *scene* has, not the one asked for. Measured over
    #: six HM3D scenes, the geodesic between two random points on the same island has
    #: median 5.6 m and reaches 30 m in 0.0% of draws -- so widening ``max_geodesic_m``
    #: lengthens the tail but leaves the same short-skewed shape. Stratifying reweights
    #: toward a flat spread, as far as the geometry allows and no further.
    #:
    #: It only ever *reweights*: every candidate in the pool already passed
    #: ``screen_segment``, so this can never admit a goal an unstratified draw would have
    #: rejected. Costs up to ``stratified_pool`` extra path plans per segment.
    stratified_pool: int = 0
    #: Draws for the episode's spawn point.
    start_retries: int = 20
    #: Draw theta_0 uniformly rather than inheriting the navmesh sampler's arbitrary
    #: orientation. See ``min_turn_deg``.
    random_spawn_heading: bool = True

    #: Assumed traversal rate when sizing the tail segment so it roughly fills the
    #: remaining frames. A sampling **bias only, never a gate** -- an oversized tail is
    #: truncated, not rejected. Re-fit from a pilot's measured frames-per-metre.
    tail_speed_mps: float = 0.8

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Screening
# ===========================================================================
def screen_segment(
    config: PointNavSampleConfig,
    *,
    geodesic_m: float,
    euclid_m: float,
    height_delta_m: float,
    same_island: bool,
    turn_rad: float,
) -> Optional[str]:
    """Reason code for rejecting this candidate goal, or ``None`` to accept it.

    Order matters and is part of the reproducibility contract: the caller redraws on a
    rejection, so changing which test fires first changes which goals a given seed
    produces. Cheapest and most conclusive first.
    """
    if not np.isfinite(geodesic_m):
        return NO_PATH
    if not same_island:
        return GOAL_OTHER_ISLAND
    if abs(height_delta_m) > config.max_height_delta_m:
        return GOAL_OTHER_FLOOR
    if geodesic_m < config.min_geodesic_m:
        return GEODESIC_TOO_SHORT
    if geodesic_m > config.max_geodesic_m:
        return GEODESIC_TOO_LONG
    if euclid_m < config.min_euclid_m:
        return EUCLID_TOO_SHORT
    if config.min_turn_deg > 0 and abs(turn_rad) < math.radians(config.min_turn_deg):
        return TURN_TOO_SMALL
    return None


def choose_island(
    rng: np.random.Generator,
    islands: Sequence[Tuple[int, float, float]],
    config: PointNavSampleConfig,
) -> Optional[int]:
    """Pick an island index, weighted by area, from ``(index, area_m2, radius_m)``.

    Area-weighted rather than uniform so a scene's big open floor gets the episodes and
    its two-metre landing does not get an equal share. Making the weighting explicit and
    seeded also takes it out of the hands of ``get_random_navigable_point``'s own
    (undocumented) distribution, which is what a run's spatial coverage would otherwise
    silently depend on.

    Returns ``None`` when no island clears both floors -- the scene has no room for an
    episode, which is a legitimate answer for a corridor-sized scan.
    """
    eligible = [
        (idx, area)
        for idx, area, radius in islands
        if area >= config.min_island_area_m2 and radius >= config.min_island_radius_m
    ]
    if not eligible:
        return None
    indices = np.array([i for i, _ in eligible])
    weights = np.array([a for _, a in eligible], dtype=float)
    return int(rng.choice(indices, p=weights / weights.sum()))


# ===========================================================================
# Geometry
# ===========================================================================
def path_tangent_heading(path_xy: Any, at: str = "end", eps_m: float = 1e-3) -> float:
    """Heading of the path's first (``"start"``) or last (``"end"``) real leg, radians.

    "Real" meaning longer than ``eps_m``: a planner happily emits repeated points, and a
    zero-length leg has no direction. Raises rather than returning 0.0 if every point
    coincides -- silently facing east is worse than a caller that has to decide.
    """
    pts = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        raise ValueError(f"need at least 2 path points to take a tangent, got {pts.shape[0]}")
    order = range(pts.shape[0] - 1) if at == "start" else range(pts.shape[0] - 2, -1, -1)
    for i in order:
        delta = pts[i + 1] - pts[i]
        if float(np.hypot(delta[0], delta[1])) > eps_m:
            return float(math.atan2(delta[1], delta[0]))
    raise ValueError("every point on the path coincides; it has no tangent")


def turn_to_face(theta_now: float, path_xy: Any, eps_m: float = 1e-3) -> float:
    """Signed shortest turn from ``theta_now`` onto the path's first real leg, radians.

    Positive is anticlockwise, in the planar control frame (``X = habitat_x``,
    ``Y = -habitat_z``, theta CCW from +X). Wrapped, so a turn is never reported as the
    long way round -- an unwrapped difference reads a 10-degree correction as 350 degrees
    and would size the turn phase's timeout from the wrong number.
    """
    return wrap_angle(path_tangent_heading(path_xy, at="start", eps_m=eps_m) - theta_now)


def segment_spl(geodesic_m: float, travelled_m: float, reached: bool) -> float:
    """Success-weighted path length for one segment.

    Mirrors ``objectnav._path_ratio``'s guard: ``d / max(d, travelled)``, and 1.0 when
    both are zero (a goal the robot was already standing on is perfectly efficient, not
    undefined). Zero when the goal was not reached, by the definition of SPL.
    """
    if not reached:
        return 0.0
    if geodesic_m <= 0.0 and travelled_m <= 0.0:
        return 1.0
    return float(geodesic_m / max(geodesic_m, travelled_m))


def episode_seed(corpus_seed: int, scene: str, episode_index: int) -> int:
    """A stable 32-bit seed for one episode.

    A hash of the three identifiers rather than ``corpus_seed + index``, so that two
    scenes' episode 7 do not draw the same sequence, and so that generating scene B does
    not depend on whether scene A was generated first. ``episode_id`` is likewise a pure
    function of the same three, which is what makes resume exact.

    **The retry-loop structure is part of the reproducibility contract.** The seed fixes
    the *sequence* of draws; how many are consumed depends on how many candidates are
    rejected. Change the order of the tests in :func:`screen_segment`, or add one, and the
    same seed produces a different corpus.
    """
    key = f"{corpus_seed}|{scene}|{episode_index}".encode("utf-8")
    return int(zlib.crc32(key) & 0xFFFFFFFF)


# ===========================================================================
# Frame budget
# ===========================================================================
@dataclass
class FrameBudget:
    """Counts recorded ticks and decides when the episode is full.

    ``exact`` is the flagship setting: keep chaining until the episode is *exactly*
    ``target_frames`` long, cutting the final segment mid-drive. Every row then has the
    same length, so no training window is short. The cut segment is marked ``truncated``
    and is exempt from the reached-goal and efficiency gates -- it was never given the
    chance -- but not from the escape / tipped / ref-miss gates, which are about whether
    the frames are *usable* rather than whether the goal was met.

    With ``exact=False`` the episode stops at the last whole segment that fits and is
    kept if it cleared ``min_frames``.
    """

    target_frames: int
    min_frames: int = 0
    #: An episode will not open a segment it has less room for than this. Below it the
    #: segment would be pure approach and would carry a goal it never had time to face.
    min_segment_frames: int = 50
    exact: bool = True
    #: How far short of the target an ``exact`` episode may land and still be kept.
    #: Uniform-length rows are a convenience, not a correctness property, so this trades
    #: a few frames of raggedness for not discarding clean demonstration.
    tolerance_frames: int = 25
    frames: int = 0

    def __post_init__(self) -> None:
        if self.target_frames <= 0:
            raise ValueError(f"target_frames must be positive, got {self.target_frames}")
        if self.min_frames > self.target_frames:
            raise ValueError(
                f"min_frames ({self.min_frames}) exceeds target_frames "
                f"({self.target_frames}); no episode could ever be accepted"
            )

    def remaining(self) -> int:
        return max(0, self.target_frames - self.frames)

    def exhausted(self) -> bool:
        return self.frames >= self.target_frames

    def room_for_a_segment(self) -> bool:
        return self.remaining() >= self.min_segment_frames

    def wants_another_segment(self) -> bool:
        """Should the episode open one more segment?

        Under ``exact`` the answer is "yes, until the target is met" -- refusing to open a
        short tail is what leaves an episode at 198 of 200 frames and then rejects it for
        being short. The tail is deliberately over-aimed (:attr:`tail_overshoot`) so it
        overshoots and gets cut at the target, rather than opening a two-frame segment
        that carries a goal it never had time to face.

        Without ``exact`` the length is free, so ``min_segment_frames`` is a real floor
        and the episode simply stops at the last whole segment that fits.
        """
        if self.exhausted():
            return False
        return True if self.exact else self.room_for_a_segment()

    def add(self, n: int) -> None:
        self.frames += int(n)

    def rewind_to(self, n: int) -> None:
        """Set the count back to ``n`` ticks, after a rollback discarded frames."""
        if n < 0:
            raise ValueError(f"cannot rewind to a negative frame count: {n}")
        self.frames = int(n)

    #: The tail segment is aimed deliberately LONG by this factor. Aiming at exactly the
    #: remaining frames systematically under-fills, because a segment also spends frames
    #: turning in place and decelerating into the goal -- measured on the first smoke run,
    #: an episode landed on 198 of 200 frames and was then rejected for being short. An
    #: over-long tail costs nothing: it is truncated at the target, which is the mechanism
    #: that makes the length exact in the first place.
    tail_overshoot: float = 1.5

    def tail_radius_m(self, fps: float, speed_mps: float, floor_m: float = 0.0) -> float:
        """Roughly how far away the last goal should be to fill the remaining frames.

        A sampling bias, never a gate: the candidate is screened by the same rules
        whatever radius drew it. ``floor_m`` keeps the radius above the minimum acceptable
        geodesic -- a radius below it draws nothing but ``geodesic_too_short`` rejections
        and burns the retry budget for no reason.
        """
        want = self.remaining() / float(fps) * float(speed_mps) * self.tail_overshoot
        return max(float(floor_m), want)

    def accepts(self) -> Tuple[bool, Optional[str]]:
        """Is an episode of this length keepable? Returns ``(ok, reason_code)``.

        Under ``exact`` the episode is never *longer* than the target -- truncation caps
        it -- so this only asks whether it fell short, and by how much.
        :attr:`tolerance_frames` exists because binning an otherwise clean episode for
        landing on 198 of 200 frames throws away good demonstration to buy a uniformity
        nothing actually needs to be exact: at ``obs_stride_frames=5`` a two-frame
        shortfall is not even a whole observation.
        """
        if self.exact:
            short = self.target_frames - self.frames
            return (short <= self.tolerance_frames,
                    None if short <= self.tolerance_frames else TOO_FEW_FRAMES)
        if self.frames < max(self.min_frames, 1):
            return False, TOO_FEW_FRAMES
        return True, None


# ===========================================================================
# Segment records
# ===========================================================================
OUTCOME_OK = "ok"
OUTCOME_TRUNCATED = "truncated"
OUTCOME_ROLLED_BACK = "rolled_back"


@dataclass
class SegmentRecord:
    """One segment's provenance, written into the episode's ``segments`` list.

    Rolled-back segments are in this list too, with ``outcome="rolled_back"`` and the
    code that killed them. They cost frames in the video of nobody and appear nowhere in
    ``pose2d``, but leaving them out would make the corpus look cleaner than the scenes
    are -- and every threshold re-cut from a pilot would then be fit to a lie.
    """

    index: int
    goal2d: List[float]
    start_frame: int
    end_frame: int
    geodesic_m: float
    travelled_m: float
    final_dist_m: float
    turn_rad: float
    island: int
    outcome: str = OUTCOME_OK
    reason: Optional[str] = None
    #: Draws consumed before this segment's goal was accepted, and why they were rejected.
    sampling_rejections: Dict[str, int] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def reached(self) -> bool:
        return self.outcome == OUTCOME_OK

    def spl(self) -> float:
        return segment_spl(self.geodesic_m, self.travelled_m, self.reached)

    def as_dict(self) -> Dict[str, Any]:
        blob = asdict(self)
        blob["num_frames"] = self.num_frames
        blob["spl"] = self.spl()
        return blob


def gate_segment(
    *,
    final_dist_m: float,
    goal_threshold_m: float,
    geodesic_m: float,
    travelled_m: float,
    min_spl: float,
    truncated: bool = False,
) -> Optional[str]:
    """Did this segment produce usable demonstration? Reason code, or ``None``.

    A **truncated** segment is exempt: the episode's frame target cut it mid-drive, so it
    was never given the chance to arrive and judging it on arrival would reject every
    episode. The escape / tipped / ref-miss gates are applied separately by the caller and
    are *not* exempt -- those ask whether the frames are usable at all, which a truncation
    does not excuse.
    """
    if truncated:
        return None
    if final_dist_m > goal_threshold_m:
        return NOT_REACHED
    if segment_spl(geodesic_m, travelled_m, True) < min_spl:
        return INEFFICIENT
    return None


# ===========================================================================
# Rollback
# ===========================================================================
def truncate_parallel(arrays: Sequence[List[Any]], length: int) -> None:
    """Cut every per-tick array back to ``length`` entries, in place.

    They are parallel by construction (``rgb_paths[i]`` <-> ``pose2d[i]`` <->
    ``goal2d[i]`` <-> ``segment_index[i]``), and the arrow builder raises on a mismatch,
    so truncating them anywhere but together is a corpus that fails at build time. Doing
    it in one call is what stops a future fifth array from being forgotten at one of
    several call sites.
    """
    if length < 0:
        raise ValueError(f"cannot truncate to a negative length: {length}")
    for arr in arrays:
        if len(arr) < length:
            raise ValueError(
                f"cannot truncate a {len(arr)}-entry array up to {length}; the "
                "per-tick arrays are out of step with each other"
            )
        del arr[length:]


@dataclass
class RollbackBudget:
    """How many drive-time segment failures one episode may absorb.

    Bounded by count *and* by the caller's sim-time backstop, because a rollback spends
    simulator time while producing no frames -- on a pathological region an unbounded
    loop is otherwise possible, and it would look like a hang rather than a rejection.
    """

    limit: int = 3
    used: int = 0
    #: What each rollback was for, so the histogram sees them.
    reasons: Dict[str, int] = field(default_factory=dict)

    def spend(self, reason: str) -> bool:
        """Record a rollback. Returns False when the budget is already gone."""
        if self.used >= self.limit:
            return False
        self.used += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


def segments_summary(records: Sequence[SegmentRecord]) -> Dict[str, Any]:
    """Aggregate stats over an episode's kept segments, for the run log."""
    kept = [r for r in records if r.outcome != OUTCOME_ROLLED_BACK]
    reached = [r for r in kept if r.reached]
    return {
        "num_segments": len(kept),
        "num_rolled_back": len(records) - len(kept),
        "num_reached": len(reached),
        "geodesic_m": round(sum(r.geodesic_m for r in kept), 3),
        "travelled_m": round(sum(r.travelled_m for r in kept), 3),
        "mean_spl": round(float(np.mean([r.spl() for r in reached])), 4) if reached else None,
        "mean_turn_deg": (
            round(float(np.mean([abs(math.degrees(r.turn_rad)) for r in kept])), 2)
            if kept else None
        ),
    }
