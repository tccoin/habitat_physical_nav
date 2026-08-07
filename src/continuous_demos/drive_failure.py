"""Per-episode health monitoring for a replay, and the rejection rule that keeps
catastrophically failed demonstrations out of the corpus.

------------------------------------------------------------------------
Why the obvious signals are all blind
------------------------------------------------------------------------
The base is a **virtual gantry**, not a vehicle. ``tidybot_base_stabilized.urdf``
roots the robot at a ``world`` link, loaded with ``fixed_base=True``, and hangs
the machine off three actuated joints: prismatic ``joint_x``, prismatic
``joint_y``, revolute ``joint_th``. ``HabitatRobotSim.step`` writes a velocity
target onto each of those motors every substep, and each carries ``effort=1000``
(``500`` for yaw) against a 34 kg body. Below them sit three *passive* joints --
``joint_z`` (prismatic, ``effort=0``, no damping) and ``joint_pitch`` /
``joint_roll``.

So the planar channel is effectively open-loop. Driven into geometry the motors
can simply win, penetrate, and put the robot on the far side; and once it is
outside the world there is not even a contact left to perturb it, so it keeps
executing the commanded SE(2) trajectory *exactly*. Every planar quantity stays
pristine through a total failure. Measured on ``17DRP5sb8fy`` episode 11:
``joint_z`` reaches **-100 m**, a third of the recorded frames are pure black,
and the final pose lands **0.12 m** from the demonstration's endpoint -- inside
the 1 m goal check -- with a normal path length. That is how comparing endpoints
between two corpora concluded navigation was healthy while the robot was leaving
the world.

Reference-path tracking is blind to it too, and the reason is worth stating
because it is the opposite of what one expects. **The stuck case and the escape
case have opposite signatures on tracking error.** A robot obstructed by
geometry falls behind its reference path and the miss grows. A robot that has
left the world tracks *better than a healthy one* -- measured mean miss while
outside the world 0.06 m, against 0.07 m for a healthy episode's move ticks --
because nothing is pushing it off course any more. A detector built only on
tracking error catches the mild failure and misses the severe one.

------------------------------------------------------------------------
Two signals, reported separately, because they fail differently
------------------------------------------------------------------------
**1. Escape / noclip -- the unactuated vertical joint free-falls.**
``joint_z`` is the vertical suspension prismatic, ``effort=0`` and no damping in
the URDF. In the world the floor holds it, so its velocity sits at zero; outside
the world nothing holds it and it accelerates under gravity. That is a direct
physical consequence of the failure, not a proxy for it.

Validated in isolation rather than inferred from a failing episode -- spawn the
robot 30 m outside the scene at init and watch (``dump/drive_failure_detection/
isolation.py``, and ``tests/test_drive_failure_in_sim.py``):

| arm | ``joint_z`` velocity over 75 ticks |
|---|---|
| spawned on the navmesh, driving | -0.78 .. +0.11 m/s, zero after the spawn settle |
| spawned 30 m outside the world | -0.39, -1.94, -3.81, -7.14, -11.66 .. -14.5 m/s |

The escaped arm ramps at -9.7 m/s^2 through the first half second and holds
below -2 m/s for 66 consecutive ticks; the healthy arm never reaches -1 m/s at
all, and settles to exactly zero once the spawn transient is over. (The descent
does level off around -14.5 m/s rather than accelerating forever -- habitat
creates a velocity motor for every URDF joint including this one, and the
commented-out "remove motors for passive joints" block in
``sim._setup_robot_and_sensors`` is why it is still there. It does not matter:
the onset is what identifies the failure, and -14.5 m/s is still 18x anything a
supported robot produces.) Note also that ``robot.translation[1]`` is
**identical and constant** in both arms to 1e-9: the base link is pinned at
spawn, so the vertical is only visible through ``joint_z``.

The rule is therefore ``joint_z`` velocity at or below ``fall_speed_mps`` for
``fall_duration_s`` continuously. Legitimate descents cannot reach it: habitat's
navmesh is built with ``agent_max_climb = 0.2``, so a step down is at most 0.2 m
and tops out at sqrt(2*g*0.2) = 2.0 m/s *instantaneously*, and descending a ramp
in contact is bounded by the drive speed times the slope -- 1.4 m/s even at 45
degrees and the full 2 m/s cap.

Measured over 2,227 replayed episodes across all 56 scenes, against a ground
truth (navmesh snap failure plus a >5 m drop) that uses neither of the
detector's inputs:

| | healthy (2,204 eps) | escaped (23 eps) |
|---|---|---|
| peak descent speed | p50 1.17, p99 3.21, **max 6.50** m/s | **min 12.33** m/s |
| longest run at <= -2 m/s | **max 14 ticks** (0.56 s) | **min 40 ticks** (1.6 s) |
| total drop | max 6.40 m | min 13.09 m |

Every run length between 15 and 39 ticks is empty, so the 1.0 s default sits in
the middle of a gap two and a half windows wide: **23/23 true positives and 0
false positives in 2,227 episodes.** Seventeen of the 23 pass the goal check --
about 1% of everything the corpus would otherwise ship.

Instantaneous ``joint_z`` *position* is deliberately not the rule: real
staircases are large and common -- ``29hnd4uzFmX`` drops 3.13 m in 8 of 40
episodes, ``ULsKaCPVFJR`` 2.90 m -- and those episodes track their reference
path to 0.13 m throughout. It is the free-fall signature, not the depth, that is
unambiguous.

**2. Stuck -- sustained reference-path deviation.** Recorded on every episode
(``ref_miss_*``) but NOT rejecting by default. It has a real precision/recall
trade-off where the fall rule has none: at the cuts measured it deletes 1.5-5.2%
of otherwise-good episodes, and its false positives are real demonstrations.
``DriveFailureConfig.reject_ref_miss_s`` arms it once a cut has been chosen from
the measured distribution. The per-episode motion composition
(``frac_drive`` / ``frac_turn`` / ...) is recorded on the same terms.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Re-exported, not redefined: ``robot_pose`` is the single definition of where the robot
# is, and this module and ``objectnav_eval.navmesh`` each used to carry their own copy.
# Still importable from here because the escape detector below is its oldest caller.
from continuous_demos.robot_pose import robot_height_m  # noqa: F401

#: Bumped when the meaning of a field changes or a field consumers rely on is
#: added. Absent from a record means the record predates per-episode quality
#: metadata entirely -- such a corpus cannot be audited and must be regenerated.
QUALITY_SCHEMA_VERSION = 1

#: Top-level key the block lives under in each JSONL record.
QUALITY_KEY = "quality"

#: Tick taxonomy, as physical rates so 25 Hz and 30 Hz corpora are comparable.
#: Same classes and thresholds as dump/regen_readiness/tick_decomposition.py.
DRIVE_SPEED_MPS = 0.300
TURN_RATE_RADPS = 0.600
_MOTION_EPS = 1e-6

#: A tick slower than this counts as a NEAR-STOP (see :func:`near_stop_runs`).
#: Set just below the path controller's ``min_approach_linear_velocity`` of
#: 0.1 m/s, which is the slowest speed it will deliberately hold while still
#: making progress: anything under it is the base genuinely coming to rest
#: rather than crawling towards a goal tolerance.
NEAR_STOP_SPEED_MPS = 0.05


@dataclass(frozen=True)
class DriveFailureConfig:
    """Thresholds for :class:`EpisodeHealthMonitor`.

    Attributes:
        fall_speed_mps: downward ``joint_z`` velocity (positive number, m/s)
            that counts as falling. 2.0 is the instantaneous speed reached by a
            free fall of 0.2 m, which is habitat's ``agent_max_climb`` and
            therefore the tallest step the navmesh admits.
        fall_duration_s: how long that must hold continuously before the episode
            is condemned. Sustaining -2 m/s for a second takes about 5 m of free
            flight. Measured over 2,227 replayed episodes across all 56 scenes,
            the longest such run in a healthy episode is 14 ticks (0.56 s, a
            4.2 m fall down a stairwell that lands) and the shortest in an
            escaped one is 40 ticks (1.6 s); every value between 15 and 39 is
            empty. 1.0 s sits in the middle of that gap, and is also the window
            the owner proposed from first principles.
        ref_miss_flag_m: reference-path miss counted as "not following the path"
            when reporting ``ref_miss_worst_run_s``. Recording-only.
        reject_ref_miss_s: reject an episode whose reference miss stays above
            ``ref_miss_flag_m`` for this long. ``None`` (default) records the
            statistic without acting on it -- see the module docstring.
    """

    fall_speed_mps: float = 2.0
    fall_duration_s: float = 1.0
    ref_miss_flag_m: float = 1.0
    reject_ref_miss_s: Optional[float] = None


def vertical_velocity_mps(robot_sim) -> float:
    """``joint_z``'s velocity: negative is falling.

    This is the escape detector's entire input. ``joint_z`` is unactuated
    (``effort=0``, ``damping=0``), so in the world the floor pins this at zero
    and outside the world it is gravity. Returns 0.0 if the joint cannot be read,
    so a query that fails can never delete an episode.
    """
    try:
        return float(robot_sim.get_joint_state("joint_z", is_vel=True))
    except Exception:
        return 0.0


def _longest_run(flags: List[bool]) -> int:
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def motion_composition(poses, fps: float) -> Dict[str, float]:
    """Fraction of ticks spent driving / turning in place / settling / frozen.

    ``poses`` is the episode's ``[X, Y, theta]`` sequence (what lands in
    ``pose2d``). Classes are exclusive and exhaustive, in priority order:
    DRIVE (speed >= 0.300 m/s), TURN (yaw rate >= 0.600 rad/s and not driving),
    FROZEN (neither position nor heading changed at all), SETTLE (the rest --
    the controller crawling towards a deadband).
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or len(p) < 2:
        return {"frac_drive": 0.0, "frac_turn": 0.0, "frac_settle": 0.0,
                "frac_frozen": 0.0, "path_len_m": 0.0}
    dxy = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    dth = np.abs(_wrap(np.diff(p[:, 2])))
    drive = dxy * fps >= DRIVE_SPEED_MPS
    turn = (~drive) & (dth * fps >= TURN_RATE_RADPS)
    frozen = (dxy < _MOTION_EPS) & (dth < _MOTION_EPS)
    settle = (~drive) & (~turn) & (~frozen)
    return {
        "frac_drive": float(drive.mean()),
        "frac_turn": float(turn.mean()),
        "frac_settle": float(settle.mean()),
        "frac_frozen": float(frozen.mean()),
        "path_len_m": float(dxy.sum()),
    }


def near_stop_runs(
    poses,
    fps: float,
    speed_mps: float = NEAR_STOP_SPEED_MPS,
    min_duration_s: float = 0.0,
) -> List[float]:
    """Durations (seconds) of the maximal runs of near-stationary ticks.

    A tick is near-stationary when the base moved slower than ``speed_mps``;
    consecutive such ticks form one run, and the returned list holds each run's
    duration. Runs shorter than ``min_duration_s`` are dropped, which is how a
    single-tick sag in speed is kept from being reported as the robot stopping.

    Heading is deliberately ignored: this asks whether the base stopped
    TRANSLATING, which is a different question from :func:`motion_composition`'s
    (what is each tick *doing*) and is the one a phase-free replay raises. Such
    a replay slaves heading to the direction of travel, so wherever the
    demonstrated path doubles back the controller has to unwind through a
    near-stop -- an organic consequence of the reversal, not a phase, and not
    something anything special-cases. The claim that those stops are brief is
    exactly what this measures.

    The first pose contributes no tick (there is no interval before it), so a
    sequence of ``n`` poses yields at most ``n - 1`` near-stationary ticks.
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or len(p) < 2 or fps <= 0:
        return []
    dxy = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    slow = dxy * fps < speed_mps

    runs: List[float] = []
    length = 0
    for is_slow in slow:
        if is_slow:
            length += 1
            continue
        if length:
            runs.append(length / fps)
        length = 0
    if length:
        runs.append(length / fps)
    return [d for d in runs if d >= min_duration_s]


@dataclass
class EpisodeHealth:
    """Everything the generator knows about how one episode went.

    Written into the metadata under ``QUALITY_KEY`` so a corpus can be audited
    after the fact -- which shard, which episode, why it was kept.
    """

    schema_version: int = QUALITY_SCHEMA_VERSION
    #: Did the episode survive every rejection rule? The generator owns the goal
    #: check and sets this too; the monitor fills in everything else.
    success: bool = True
    #: ``"ok"``, or the rule that rejected it: ``"escaped"``, ``"goal"``,
    #: ``"ref_miss"``, ``"no_frames"``, ``"replay_error"``.
    outcome: str = "ok"
    # ---- goal check (the number the generator used to compute and discard) ----
    final_dist_m: Optional[float] = None
    goal_threshold_m: Optional[float] = None
    # ---- signal 1: the unactuated vertical joint in free fall ----
    escaped: bool = False
    #: Longest continuous run of ticks falling at or beyond ``fall_speed_mps``.
    fall_run_ticks: int = 0
    fall_run_s: float = 0.0
    #: Fastest descent seen, m/s, as a positive number. A supported robot never
    #: exceeds ~0.8 m/s; free fall passes 14 m/s inside three seconds.
    max_fall_speed_mps: float = 0.0
    #: Metres below the spawn height the robot ever reached (positive = below).
    max_drop_m: float = 0.0
    fall_speed_mps: Optional[float] = None
    fall_duration_s: Optional[float] = None
    # ---- signal 2: reference-path tracking, over move-phase ticks only ----
    ref_miss_mean_m: Optional[float] = None
    ref_miss_p95_m: Optional[float] = None
    ref_miss_max_m: Optional[float] = None
    ref_miss_worst_run_s: float = 0.0
    ref_miss_flag_m: Optional[float] = None
    # ---- composition ----
    n_ticks: int = 0
    duration_s: float = 0.0
    frac_drive: Optional[float] = None
    frac_turn: Optional[float] = None
    frac_settle: Optional[float] = None
    frac_frozen: Optional[float] = None
    path_len_m: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class EpisodeHealthMonitor:
    """Accumulates health signals over one episode, one call per recorded tick.

    Usage mirrors the recording loop::

        monitor = EpisodeHealthMonitor(robot_sim, fps)
        ...
        robot_sim.step(...)
        monitor.observe(move_controller=ctrl_or_None)   # once per recorded tick
        ...
        health = monitor.finish(poses)

    ``observe`` must be called *after* the physics step and once per frame
    appended, so the signals line up 1:1 with ``pose2d``. It reads three joint
    values and one array index; cost is under 10 us/tick, against ~9.8 ms/frame
    for writing the image.
    """

    def __init__(self, robot_sim, fps: float,
                 config: Optional[DriveFailureConfig] = None):
        self.robot_sim = robot_sim
        self.fps = float(fps)
        self.config = config or DriveFailureConfig()
        self._falling: List[bool] = []
        self._ref_miss: List[float] = []
        self._max_fall_speed = 0.0
        self._spawn_height = robot_height_m(robot_sim)
        self._min_height = self._spawn_height

    # -- per tick ----------------------------------------------------------
    def observe(self, move_controller=None) -> None:
        """Record one tick. Pass the active :class:`HolonomicRobotController`
        during a move phase, ``None`` during a look phase (looks do not
        translate, so they have no reference-path error to speak of)."""
        v_z = vertical_velocity_mps(self.robot_sim)
        self._falling.append(v_z <= -self.config.fall_speed_mps)
        if -v_z > self._max_fall_speed:
            self._max_fall_speed = -v_z

        height = robot_height_m(self.robot_sim)
        if height < self._min_height:
            self._min_height = height

        if move_controller is not None:
            miss = _reference_miss(move_controller, self.robot_sim.get_2d_pose())
            if miss == miss:  # drop NaN rather than poisoning the statistics
                self._ref_miss.append(miss)

    # -- end of episode ----------------------------------------------------
    def finish(self, poses=None) -> EpisodeHealth:
        cfg = self.config
        fps = self.fps
        run = _longest_run(self._falling)
        escaped = run >= cfg.fall_duration_s * fps

        health = EpisodeHealth(
            escaped=bool(escaped),
            fall_run_ticks=int(run),
            fall_run_s=float(run / fps),
            max_fall_speed_mps=float(self._max_fall_speed),
            max_drop_m=float(self._spawn_height - self._min_height),
            fall_speed_mps=float(cfg.fall_speed_mps),
            fall_duration_s=float(cfg.fall_duration_s),
            ref_miss_flag_m=float(cfg.ref_miss_flag_m),
            n_ticks=len(self._falling),
            duration_s=float(len(self._falling) / fps),
        )
        if self._ref_miss:
            miss = np.asarray(self._ref_miss, dtype=float)
            health.ref_miss_mean_m = float(miss.mean())
            health.ref_miss_p95_m = float(np.percentile(miss, 95))
            health.ref_miss_max_m = float(miss.max())
            health.ref_miss_worst_run_s = float(
                _longest_run([bool(m > cfg.ref_miss_flag_m) for m in miss]) / fps
            )
        if poses is not None:
            for k, v in motion_composition(poses, fps).items():
                setattr(health, k, v)

        if escaped:
            health.success = False
            health.outcome = "escaped"
        elif (cfg.reject_ref_miss_s is not None
              and health.ref_miss_worst_run_s >= cfg.reject_ref_miss_s):
            health.success = False
            health.outcome = "ref_miss"
        return health


def _reference_miss(move_controller, pose2d) -> float:
    """Distance from the robot to its reference path, as DWPP already knows it.

    ``PurePursuitController._calc_index`` maintains ``last_idx``, the densified
    path point the controller has projected the robot onto (searched forward
    only, within one look-ahead of the previous index, so a self-intersecting
    path cannot snap it backwards). The distance to that point is the
    controller's own view of how far off-path the robot is, and it is already
    computed every tick -- this only reads it back.
    """
    try:
        pp = move_controller.controller
        if pp.path is None:
            return float("nan")
        xy = np.asarray(pose2d[:2], dtype=float)
        return float(np.linalg.norm(pp.path[int(pp.last_idx)][:2] - xy))
    except Exception:
        return float("nan")
