"""Per-episode health monitoring for a replay, and the rejection rule that keeps
catastrophically failed demonstrations out of the corpus.

------------------------------------------------------------------------
The failure this exists for, and why nothing else can see it
------------------------------------------------------------------------
The base is a **virtual gantry**, not a vehicle. ``tidybot_base_stabilized.urdf``
roots the robot at a ``world`` link, loaded with ``fixed_base=True``, and hangs
the whole machine off three actuated joints: prismatic ``joint_x``, prismatic
``joint_y``, revolute ``joint_th``. ``HabitatRobotSim.step`` writes a velocity
target onto each of those three motors every substep, and each motor has
``effort=1000`` (``500`` for yaw) against a 34 kg body. Below them sit three
*passive* joints -- ``joint_z`` (prismatic, ``effort=0``, no damping) and
``joint_pitch`` / ``joint_roll`` -- which are the only suspension, and the only
mechanical coupling between the machine and the floor.

Two consequences follow, and together they are the whole problem:

1. **Planar motion is open-loop.** The X/Y/yaw motors are strong enough to track
   their velocity targets whatever the scene does, and once the robot is outside
   the geometry there is not even a contact left to perturb them. A robot that
   has fallen out of the world therefore keeps executing the commanded SE(2)
   trajectory *perfectly*.
2. **The vertical is a free variable.** ``joint_z`` carries no motor, so when the
   floor stops being there the machine simply falls, all the way to the joint's
   own -100 m limit, while X/Y carry on.

So every planar quantity stays pristine through a total failure. Measured on the
known-bad episode (``17DRP5sb8fy`` index 11): ``joint_z`` reaches **-100 m**, a
third of the recorded frames are pure black -- and the final pose lands **0.12 m**
from the demonstration's endpoint, well inside the 1 m goal check, with a path
length within a few percent of a healthy run. Comparing endpoints or path lengths
between corpora cannot see this; neither can ``pose2d``, which is planar.

Worse, the obvious "is it following its reference path" signal cannot see it
either, and for the same reason. Over the escaped episodes measured here the mean
DWPP reference-path miss *while outside the world* is ~0.08 m -- **better** than a
healthy robot's, because nothing is pushing it off course any more. Reference
tracking is a good detector of a robot that is stuck or being obstructed; it is
actively misleading about a robot that has left.

------------------------------------------------------------------------
What this detector keys on: support, not motion
------------------------------------------------------------------------
The one thing an escaped robot cannot fake is **contact**. Contact is the only
physical coupling the gantry has to the scene, so:

    a tick is UNSUPPORTED when no contact point involves the robot,
    and an episode ESCAPED when it is unsupported for a sustained window.

The window is what makes this safe. A robot dropping off a step is briefly
airborne, and that is normal: over 479 replayed episodes across 12 scenes the
longest contact-free run in any non-escaped episode was **25 ticks (1.0 s)**,
while the escaped ones ran 74 / 183 / 232 / 268 ticks. The default 2.0 s window
sits in that gap with a factor of two on either side, and 2.0 s of free fall is
19.6 m of descent -- more than any indoor scene has to give.

A single guarded second clause covers an escape that happens too close to the end
of the episode to accumulate 2 s: unsupported for 1.0 s *through the final tick*
**and** more than 1 m below the lowest height at which the robot was last
supported. Both halves are required; neither fires on its own.

``joint_z`` on its own is deliberately NOT the rule. Legitimate descents are large
and common -- ``29hnd4uzFmX`` drops 3.13 m down a staircase in 8 of 40 episodes,
``ULsKaCPVFJR`` 2.90 m -- and those episodes track their reference path to 0.13 m
and stay in contact throughout. A threshold on descent alone would delete them.
Height enters only as the guard on the terminal clause, where contact has already
established that the robot is unsupported.

------------------------------------------------------------------------
What is measured but not enforced
------------------------------------------------------------------------
Reference-path tracking (``ref_miss_*``) and the per-episode motion composition
(``frac_drive`` / ``frac_turn`` / ...) are recorded on every episode but do not
reject by default: they describe *quality*, not *validity*, and a threshold on
either deletes real demonstrations. ``DriveFailureConfig.reject_ref_miss_s``
turns the sustained-tracking-error rule on once a cut has been chosen from the
measured distribution.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

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


@dataclass(frozen=True)
class DriveFailureConfig:
    """Thresholds for :class:`EpisodeHealthMonitor`.

    Attributes:
        escape_unsupported_s: contact-free duration that condemns an episode
            outright. 2.0 s; the longest contact-free run measured in a healthy
            episode is 1.0 s and the shortest in an escaped one is 3.0 s.
        escape_terminal_unsupported_s: shorter contact-free window accepted when
            the run reaches the episode's final tick, so an escape in the last
            couple of seconds is still caught.
        escape_terminal_drop_m: descent below the last supported height that the
            terminal clause additionally requires. Guards the shorter window
            against a robot that merely finishes the episode mid-hop.
        ref_miss_flag_m: reference-path miss counted as "not following the path"
            when reporting ``ref_miss_worst_run_s``. Recording-only.
        reject_ref_miss_s: reject an episode whose reference miss stays above
            ``ref_miss_flag_m`` for this long. ``None`` (default) records the
            statistic without acting on it -- see the module docstring.
    """

    escape_unsupported_s: float = 2.0
    escape_terminal_unsupported_s: float = 1.0
    escape_terminal_drop_m: float = 1.0
    ref_miss_flag_m: float = 1.0
    reject_ref_miss_s: Optional[float] = None


def robot_height_m(robot_sim) -> float:
    """The robot's true habitat-frame height (metres).

    NOT ``robot.translation[1]``: the base link is pinned at the spawn point by
    ``fixed_base=True`` and the machine moves through its joints, so that
    component never changes during an episode. The live vertical is the base
    translation plus the passive suspension prismatic ``joint_z``. Same recipe as
    ``data_scripts/eval_objectnav_policy.py::robot_position_3d`` in the
    spatial_training repo; reading ``translation`` alone pins every height -- and
    every goal distance -- to the spawn.
    """
    height = float(robot_sim.robot.translation[1])
    try:
        height += float(robot_sim.get_joint_state("joint_z"))
    except (ValueError, KeyError, AttributeError):
        pass  # passive joint without a motor: the base height is what we have
    return height


def robot_contact_count(robot_sim) -> int:
    """Number of live contact points involving the robot.

    Filtered to the robot's own link object ids rather than counting every
    contact in the world, so a scene that ever gains another dynamic object
    cannot make an escaped robot look supported. Any failure to query is
    reported as ``-1`` ("unknown"), which the monitor treats as supported --
    a detector that cannot see must never delete data.
    """
    try:
        ids = set(robot_sim.robot.link_object_ids.keys())
        ids.add(robot_sim.robot.object_id)
        return sum(
            1
            for c in robot_sim.sim.get_physics_contact_points()
            if c.object_id_a in ids or c.object_id_b in ids
        )
    except Exception:
        return -1


def _longest_run(flags: List[bool]) -> int:
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def _terminal_run(flags: List[bool]) -> int:
    n = 0
    for v in reversed(flags):
        if not v:
            break
        n += 1
    return n


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


@dataclass
class EpisodeHealth:
    """Everything the generator knows about how one episode went.

    Written into the metadata under ``QUALITY_KEY`` so a corpus can be audited
    after the fact -- which shard, which episode, why it was kept.
    """

    schema_version: int = QUALITY_SCHEMA_VERSION
    #: Did the episode survive every rejection rule? Set by the generator, which
    #: owns the goal check; the monitor fills in everything else.
    success: bool = True
    #: ``"ok"``, or the rule that rejected it: ``"escaped"``, ``"goal"``,
    #: ``"ref_miss"``, ``"no_frames"``, ``"replay_error"``.
    outcome: str = "ok"
    # ---- goal check (the number the generator used to compute and discard) ----
    final_dist_m: Optional[float] = None
    goal_threshold_m: Optional[float] = None
    # ---- escape / support ----
    escaped: bool = False
    unsupported_run_ticks: int = 0
    unsupported_run_s: float = 0.0
    unsupported_terminal_ticks: int = 0
    #: Metres below the spawn height the robot ever reached (positive = below).
    max_drop_m: float = 0.0
    #: Metres below the last *supported* height at the final tick.
    final_drop_below_support_m: float = 0.0
    #: Ticks on which habitat's pathfinder could not snap the robot to the
    #: navmesh at all. Diagnostic; not part of any rejection rule.
    navmesh_snap_failures: int = 0
    # ---- reference-path tracking, over move-phase ticks only ----
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
    appended, so the signals line up 1:1 with ``pose2d``. Cost is ~10 us/tick,
    against ~9.8 ms/frame for writing the image.
    """

    def __init__(self, robot_sim, fps: float,
                 config: Optional[DriveFailureConfig] = None,
                 track_navmesh: bool = True):
        self.robot_sim = robot_sim
        self.fps = float(fps)
        self.config = config or DriveFailureConfig()
        self.track_navmesh = track_navmesh
        self._unsupported: List[bool] = []
        self._ref_miss: List[float] = []
        self._spawn_height = robot_height_m(robot_sim)
        self._min_height = self._spawn_height
        self._last_height = self._spawn_height
        #: Lowest height at which the robot was last known to be supported.
        self._last_supported_height = self._spawn_height
        self._navmesh_failures = 0

    # -- per tick ----------------------------------------------------------
    def observe(self, move_controller=None) -> None:
        """Record one tick. Pass the active :class:`HolonomicRobotController`
        during a move phase, ``None`` during a look phase (looks do not
        translate, so they have no reference-path error to speak of)."""
        n_contacts = robot_contact_count(self.robot_sim)
        # -1 means the query failed; treat as supported so a broken query can
        # never delete an episode.
        unsupported = n_contacts == 0
        self._unsupported.append(unsupported)

        height = robot_height_m(self.robot_sim)
        self._last_height = height
        if height < self._min_height:
            self._min_height = height
        if not unsupported:
            self._last_supported_height = height

        if move_controller is not None:
            miss = _reference_miss(move_controller, self.robot_sim.get_2d_pose())
            if miss == miss:  # drop NaN rather than poisoning the statistics
                self._ref_miss.append(miss)

        if self.track_navmesh and unsupported:
            # Only asked when the robot is already unsupported, so the cost is
            # paid on the ticks where it might mean something. Purely
            # diagnostic: navmesh membership is legitimately false near walls,
            # because the mesh is eroded by the agent radius.
            try:
                pf = self.robot_sim.sim.pathfinder
                if pf.is_loaded:
                    x, y, _th = self.robot_sim.get_2d_pose()
                    snapped = pf.snap_point(np.array([x, height, -y], dtype=np.float32))
                    if not (snapped[0] == snapped[0]):  # NaN
                        self._navmesh_failures += 1
            except Exception:
                pass

    # -- end of episode ----------------------------------------------------
    def finish(self, poses=None) -> EpisodeHealth:
        cfg = self.config
        fps = self.fps
        run = _longest_run(self._unsupported)
        terminal = _terminal_run(self._unsupported)
        drop_at_end = self._last_supported_height - self._last_height

        escaped = run >= cfg.escape_unsupported_s * fps
        if not escaped and terminal >= cfg.escape_terminal_unsupported_s * fps:
            escaped = drop_at_end >= cfg.escape_terminal_drop_m

        health = EpisodeHealth(
            escaped=bool(escaped),
            unsupported_run_ticks=int(run),
            unsupported_run_s=float(run / fps),
            unsupported_terminal_ticks=int(terminal),
            max_drop_m=float(self._spawn_height - self._min_height),
            final_drop_below_support_m=float(drop_at_end),
            navmesh_snap_failures=int(self._navmesh_failures),
            ref_miss_flag_m=float(cfg.ref_miss_flag_m),
            n_ticks=len(self._unsupported),
            duration_s=float(len(self._unsupported) / fps),
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
