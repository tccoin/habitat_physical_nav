"""Driving a chained PointNav episode through the simulator.

The half of PointNav that touches ``habitat_sim`` and ``HabitatRobotSim``; the decisions
it enforces -- what to sample, what to reject, when to roll back -- live in
:mod:`continuous_demos.pointnav` and are unit-tested without a scene.

------------------------------------------------------------------------
Why this is a separate tick loop and not a flag on ``record_segmented_demo``
------------------------------------------------------------------------
The two loops share about 35 lines. Reusing the ObjectNav one would mean six new optional
parameters and four new branches in a function whose documented value is that it has not
changed: ``replay.py`` is verified frame-for-frame against a pristine worktree, and that
property cannot be re-established from a unit test -- it costs a full regeneration diff.
Against 35 lines, that is not a trade worth making.

What must not drift between the two are the *constants*, and those are imported from
``replay`` rather than repeated: ``MOVE_STALL_TICKS``, ``MOVE_HEADING_GAIN``,
``MOVE_MAX_VW``, ``LOCAL_TIMEOUT_MPS``, ``TIPPED_OVER_DEG``, ``is_stalled``. Importing
them makes drift impossible rather than merely unlikely.

The tick ordering is deliberately identical to ``record_segmented_demo``'s, because the
corpora have to mean the same thing:

    obs = get_obs()          <- BEFORE the step: frame i is the observation the
    twist = controller.step()   controller acted on
    robot_sim.step(twist)
    poses.append(get_2d_pose())  <- AFTER the step: pose i is where that action landed
    health.observe()

------------------------------------------------------------------------
Segment structure
------------------------------------------------------------------------
Each segment is a turn-in-place onto the path's first leg followed by a pure-pursuit
drive along it. The turn is a real phase rather than something the path follower is left
to sort out: DWPP blends its heading toward the path tangent, so starting 180 degrees
wrong produces a long arcing departure that is neither the shortest path nor a
demonstration of anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import habitat_sim

from continuous_demos import pointnav as pn
from continuous_demos.controllers import (
    HeadingRotationController,
    HolonomicRobotController,
)
from continuous_demos.drive_failure import DriveFailureConfig, EpisodeHealthMonitor
from continuous_demos.objectnav import PathTracker
from continuous_demos.panoramic import rgb_view
from continuous_demos.replay import (
    LOCAL_TIMEOUT_MPS,
    MOVE_HEADING_GAIN,
    MOVE_MAX_VW,
    MOVE_STALL_EPS,
    MOVE_STALL_MAX_SPEED,
    MOVE_STALL_TICKS,
    RECORD_FPS,
    TIPPED_OVER_DEG,
    is_stalled,
)
from continuous_demos.viewport_overlay import camera_matrix
from continuous_demos.robot_pose import (
    navmesh_height,
    planar_heading_to_quat_xyzw,
    robot_navmesh_position,
)

#: Effective in-place yaw rate assumed when sizing a turn phase's timeout, rad/s. Well
#: under ``HeadingRotationConfig.w_max`` (2.0); a turn that cannot finish in this much
#: time is fighting something rather than merely being slow.
TURN_RATE_MPS = 0.5
TURN_TIMEOUT_SLACK_S = 2.0

#: Sensor whose world transform is handed to `on_tick` so a caller can project the path
#: into the frame it was rendered from. Against a rig this does not exist -- deliberately,
#: as `panoramic.py` argues -- and the caller names `color_sensor_0` instead.
OVERLAY_SENSOR_UUID = "color_sensor"


# ===========================================================================
# Results
# ===========================================================================
@dataclass
class PointNavRecording:
    """One episode, kept or rejected. Frames live in memory until it is kept.

    Nothing is written to disk here. A rejected episode therefore costs no inodes, which
    is what makes rollback cheap enough to be the default policy.
    """

    frames: List[Any] = field(default_factory=list)
    poses: List[Any] = field(default_factory=list)
    goal2d: List[List[float]] = field(default_factory=list)
    segment_index: List[int] = field(default_factory=list)
    segments: List[pn.SegmentRecord] = field(default_factory=list)
    start_pose2d: Optional[List[float]] = None
    island: int = -1
    #: ``None`` while the episode is still viable; a reason code once it is not.
    reason: Optional[str] = None
    detail: str = ""
    sampling: Dict[str, Any] = field(default_factory=dict)
    health: Any = None
    rollbacks: Optional[pn.RollbackBudget] = None
    sim_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.reason is None

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def kept_segments(self) -> List[pn.SegmentRecord]:
        return [s for s in self.segments if s.outcome != pn.OUTCOME_ROLLED_BACK]

    def check_alignment(self) -> None:
        """Every per-tick array is the same length. The arrow builder raises otherwise."""
        n = len(self.frames)
        for name in ("poses", "goal2d", "segment_index"):
            if len(getattr(self, name)) != n:
                raise AssertionError(
                    f"{name} has {len(getattr(self, name))} entries against "
                    f"{n} frames; the per-tick arrays are out of step"
                )


@dataclass
class _Segment:
    """A sampled, planned, not-yet-driven segment."""

    goal_habitat: np.ndarray
    goal2d: List[float]
    path_3d: np.ndarray
    path_xy: np.ndarray
    geodesic_m: float
    turn_rad: float
    rejections: Dict[str, int]


# ===========================================================================
# Sampling
# ===========================================================================
def island_table(pathfinder) -> List[Tuple[int, float, float]]:
    """``(index, area_m2, radius_m)`` for every island, for :func:`pointnav.choose_island`."""
    return [
        (i, float(pathfinder.island_area(i)), float(pathfinder.island_radius(i)))
        for i in range(int(pathfinder.num_islands))
    ]


def plan_path(pathfinder, start_3d, goal_3d) -> Tuple[Optional[np.ndarray], float]:
    """Shortest path between two habitat points. ``(points, geodesic)`` or ``(None, inf)``."""
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start_3d, dtype=np.float32)
    path.requested_end = np.asarray(goal_3d, dtype=np.float32)
    if not pathfinder.find_path(path):
        return None, float("inf")
    points = np.asarray(path.points, dtype=np.float32)
    if points.shape[0] < 2:
        return None, float("inf")
    return points, float(path.geodesic_distance)


def habitat_xy_to_planar(points_3d) -> np.ndarray:
    """Habitat ``(M, 3)`` -> planar ``(M, 2)``: ``X = habitat_x``, ``Y = -habitat_z``."""
    pts = np.asarray(points_3d, dtype=float).reshape(-1, 3)
    return np.stack([pts[:, 0], -pts[:, 2]], axis=1)


def sample_segment(
    pathfinder,
    rng: np.random.Generator,
    config: pn.PointNavSampleConfig,
    *,
    start_3d,
    theta_now: float,
    island: int,
    max_radius_m: Optional[float] = None,
) -> Tuple[Optional[_Segment], Dict[str, int]]:
    """Draw and screen goals until one is acceptable, or the retry budget runs out.

    ``max_radius_m`` biases the draw toward a goal that roughly fills the remaining frame
    budget (see ``FrameBudget.tail_radius_m``). It is a bias only: a goal drawn near is
    still screened by the same rules, and one drawn too far is rejected rather than
    silently accepted.

    **The order and count of draws is part of the reproducibility contract.** The seed
    fixes the sequence; how much of it is consumed depends on how many candidates are
    rejected, so changing a screening rule changes the corpus a seed produces.
    """
    rejections: Dict[str, int] = {}
    start = np.asarray(start_3d, dtype=np.float32)
    for _ in range(int(config.segment_retries)):
        if max_radius_m is not None and max_radius_m > 0:
            goal = pathfinder.get_random_navigable_point_near(
                circle_center=start, radius=float(max_radius_m), island_index=island
            )
        else:
            goal = pathfinder.get_random_navigable_point(island_index=island)
        goal = np.asarray(goal, dtype=np.float32)
        if not np.all(np.isfinite(goal)):
            rejections[pn.GOAL_OFF_NAVMESH] = rejections.get(pn.GOAL_OFF_NAVMESH, 0) + 1
            continue

        points, geodesic = plan_path(pathfinder, start, goal)
        same_island = int(pathfinder.get_island(goal)) == int(island)
        # A goal we cannot plan to has no tangent, so the turn is unknowable; screen on
        # the reachability facts first and only compute the turn when it can matter.
        turn = 0.0
        path_xy = None
        if points is not None:
            path_xy = habitat_xy_to_planar(points)
            try:
                turn = pn.turn_to_face(theta_now, path_xy)
            except ValueError:
                rejections[pn.NO_PATH] = rejections.get(pn.NO_PATH, 0) + 1
                continue

        code = pn.screen_segment(
            config,
            geodesic_m=geodesic,
            euclid_m=float(np.linalg.norm(goal[[0, 2]] - start[[0, 2]])),
            height_delta_m=float(goal[1] - start[1]),
            same_island=same_island,
            turn_rad=turn,
        )
        if code is not None:
            rejections[code] = rejections.get(code, 0) + 1
            continue

        goal_theta = pn.path_tangent_heading(path_xy, at="end")
        return (
            _Segment(
                goal_habitat=goal,
                goal2d=[float(goal[0]), float(-goal[2]), float(goal_theta)],
                path_3d=points,
                path_xy=path_xy,
                geodesic_m=geodesic,
                turn_rad=turn,
                rejections=dict(rejections),
            ),
            rejections,
        )
    return None, rejections


def sample_start(
    pathfinder, rng: np.random.Generator, config: pn.PointNavSampleConfig, island: int
) -> Optional[np.ndarray]:
    """A spawn point on ``island``, or None if the draws all came back non-finite."""
    for _ in range(int(config.start_retries)):
        point = np.asarray(
            pathfinder.get_random_navigable_point(island_index=island), dtype=np.float32
        )
        if np.all(np.isfinite(point)):
            return point
    return None


# ===========================================================================
# The recorder
# ===========================================================================
def record_pointnav_episode(
    robot_sim,
    *,
    scene: str,
    episode_index: int,
    corpus_seed: int = 0,
    config: Optional[pn.PointNavSampleConfig] = None,
    budget: Optional[pn.FrameBudget] = None,
    fps: int = RECORD_FPS,
    goal_threshold_m: float = 0.5,
    min_segment_spl: float = 0.55,
    failure_config: Optional[DriveFailureConfig] = None,
    rollback_limit: int = 3,
    sim_time_limit_s: float = 300.0,
    on_tick: Optional[Callable[[Dict[str, Any]], None]] = None,
    viewport_debug: bool = False,
    show_navmesh: bool = False,
    verbose: bool = False,
) -> PointNavRecording:
    """Drive one chained PointNav episode and return everything it produced.

    ``on_tick`` is called after every recorded tick with the live state (pose, goal,
    planned path, segment index, frame index). The debug renderer uses it to composite a
    top-down panel; the generator leaves it ``None``, so a generation run never pays for
    a map it will not look at.
    """
    config = config or pn.PointNavSampleConfig()
    budget = budget or pn.FrameBudget(target_frames=400)
    failure_config = failure_config or DriveFailureConfig()
    rec = PointNavRecording(rollbacks=pn.RollbackBudget(limit=rollback_limit))

    seed = pn.episode_seed(corpus_seed, scene, episode_index)
    rng = np.random.default_rng(seed)
    pathfinder = robot_sim.sim.pathfinder
    pathfinder.seed(seed)

    islands = island_table(pathfinder)
    island = pn.choose_island(rng, islands, config)
    rec.sampling = {
        "schema_version": 1,
        "seed": int(seed),
        "num_islands": len(islands),
        "island": island if island is not None else -1,
        "config": config.as_dict(),
        "fps": int(fps),
        "goal_threshold_m": float(goal_threshold_m),
        "min_segment_spl": float(min_segment_spl),
        "target_frames": int(budget.target_frames),
        "exact_frames": bool(budget.exact),
    }
    if island is None:
        rec.reason, rec.detail = pn.NO_ISLAND, (
            f"largest island is {max((a for _, a, _ in islands), default=0.0):.1f} m2, "
            f"below {config.min_island_area_m2} m2"
        )
        return rec
    rec.island = int(island)
    rec.sampling["island_area_m2"] = round(float(pathfinder.island_area(island)), 3)

    start_3d = sample_start(pathfinder, rng, config, island)
    if start_3d is None:
        rec.reason, rec.detail = pn.START_OFF_NAVMESH, "no navigable spawn point drawn"
        return rec

    theta0 = float(rng.uniform(-math.pi, math.pi)) if config.random_spawn_heading else 0.0
    _spawn(robot_sim, start_3d, theta0)
    rec.start_pose2d = [float(v) for v in robot_sim.get_2d_pose()]

    move_controller = HolonomicRobotController(robot_sim, pure_pursuit_config=None)
    turn_controller = HeadingRotationController(robot_sim)
    # Habitat's own navmesh wireframe, burned into the render. Shows what the planner
    # thinks is traversable next to what the robot is actually doing -- which is the
    # question behind every `not_reached` rejection.
    robot_sim.sim.navmesh_visualization = bool(show_navmesh)

    # ------------------------------------------------------------------
    # Chain segments until the frame target is met.
    # ------------------------------------------------------------------
    healths: List[Any] = []
    sim_time = 0.0
    while budget.wants_another_segment():
        segment_index = len(rec.kept_segments())
        pose_now = robot_sim.get_2d_pose()
        here = robot_navmesh_position(robot_sim)
        # Bias the draw once the episode is more than half full, so the last goal is
        # roughly the right distance away. Floored at the minimum acceptable geodesic:
        # a radius below it draws nothing but `geodesic_too_short` and burns the retries.
        radius = (
            budget.tail_radius_m(fps, config.tail_speed_mps,
                                 floor_m=config.min_geodesic_m * 1.5)
            if budget.remaining() < budget.target_frames // 2
            else None
        )
        segment, rejections = sample_segment(
            pathfinder, rng, config,
            start_3d=here, theta_now=float(pose_now[2]), island=island,
            max_radius_m=radius,
        )
        if segment is None:
            # Segment 0 unsampled is fatal; later, the frame gate decides.
            if segment_index == 0:
                rec.reason = pn.SEGMENT_UNSAMPLED
                rec.detail = f"rejections {rejections}"
                return rec
            break

        mark = len(rec.frames)
        monitor = EpisodeHealthMonitor(robot_sim, fps=fps, config=failure_config)
        code, detail, truncated, spent_s = _drive_segment(
            robot_sim, rec, segment, segment_index,
            move_controller=move_controller, turn_controller=turn_controller,
            monitor=monitor, budget=budget, fps=fps, on_tick=on_tick,
            viewport_debug=viewport_debug,
        )
        sim_time += spent_s
        health = monitor.finish(rec.poses[mark:])
        if code is None and not health.success:
            code = health.outcome  # "escaped" / "ref_miss" -- both are pn reason codes
            detail = _health_detail(health)
        if code is None:
            travelled = _path_length(rec.poses[mark:])
            final_dist = float(
                np.linalg.norm(np.asarray(rec.poses[-1][:2], dtype=float)
                               - np.asarray(segment.goal2d[:2], dtype=float))
            )
            code = pn.gate_segment(
                final_dist_m=final_dist, goal_threshold_m=goal_threshold_m,
                geodesic_m=segment.geodesic_m, travelled_m=travelled,
                min_spl=min_segment_spl, truncated=truncated,
            )
            detail = detail or (
                f"{final_dist:.2f}m from goal (> {goal_threshold_m}m)"
                if code == pn.NOT_REACHED else
                f"spl {pn.segment_spl(segment.geodesic_m, travelled, True):.2f} "
                f"< {min_segment_spl}" if code == pn.INEFFICIENT else ""
            )
        else:
            travelled = _path_length(rec.poses[mark:])
            final_dist = (
                float(np.linalg.norm(np.asarray(rec.poses[-1][:2], dtype=float)
                                     - np.asarray(segment.goal2d[:2], dtype=float)))
                if len(rec.poses) > mark else float("nan")
            )

        record = pn.SegmentRecord(
            index=segment_index, goal2d=list(segment.goal2d),
            start_frame=mark, end_frame=len(rec.frames),
            geodesic_m=round(segment.geodesic_m, 4),
            travelled_m=round(travelled, 4),
            final_dist_m=round(final_dist, 4) if final_dist == final_dist else float("nan"),
            turn_rad=round(segment.turn_rad, 5), island=int(island),
            outcome=pn.OUTCOME_TRUNCATED if truncated else pn.OUTCOME_OK,
            sampling_rejections=dict(rejections),
        )

        if code is None:
            rec.segments.append(record)
            healths.append(health)
            if verbose:
                print(f"    seg {segment_index}: {record.num_frames} frames, "
                      f"{segment.geodesic_m:.2f}m geodesic, spl {record.spl():.2f}, "
                      f"turn {math.degrees(segment.turn_rad):+.0f}deg"
                      + (" [truncated]" if truncated else ""))
            continue

        # ---- drive-time failure: roll back rather than lose the episode ----
        record.outcome, record.reason = pn.OUTCOME_ROLLED_BACK, code
        rec.segments.append(record)
        if verbose:
            print(f"    seg {segment_index}: ROLLBACK ({code}) {detail}")
        if not rec.rollbacks.spend(code):
            rec.reason = pn.ROLLBACKS_EXHAUSTED
            rec.detail = f"{rec.rollbacks.used} rollback(s): {rec.rollbacks.reasons}"
            return rec
        _rollback(robot_sim, rec, budget, mark)
        if sim_time >= sim_time_limit_s:
            rec.reason = pn.BUDGET_EXHAUSTED
            rec.detail = f"sim time {sim_time:.0f}s >= {sim_time_limit_s:.0f}s backstop"
            return rec

    rec.sim_time_s = sim_time
    rec.check_alignment()
    ok, code = budget.accepts()
    if not ok:
        rec.reason = code
        rec.detail = (
            f"{len(rec.frames)} frames against a target of {budget.target_frames}"
        )
        return rec
    rec.health = _merge_healths(healths, fps)
    return rec


# ===========================================================================
# Internals
# ===========================================================================
def _spawn(robot_sim, point_3d, theta: float) -> None:
    """Place the robot at a navmesh point facing ``theta``.

    ``snap_to_navmesh=False`` because the point already came off the navmesh; re-snapping
    would move it, and on a scene where the snap resolves differently across runs that is
    the reproducibility hazard ``docs/PID_TRACKING_REPORT.md`` documents.
    """
    robot_sim.reset(
        robot_translation=np.asarray(point_3d, dtype=np.float32).copy(),
        robot_rotation=planar_heading_to_quat_xyzw(theta),
        snap_to_navmesh=False,
        vertical_offset=None,
    )


def _rollback(robot_sim, rec: PointNavRecording, budget: pn.FrameBudget, mark: int) -> None:
    """Erase the failed segment's ticks and put the robot back where it started them.

    Erased, not relabelled: the frames go, the pose stream stays continuous across the
    join, and ``goal2d`` stays piecewise-constant. Relabelling the excursion under the
    next goal would teach a policy that goals move when you fail.

    The robot is physically returned to the last kept pose, so the resumed frame renders
    essentially identically to the discarded one -- and since a segment ends either
    goal-reached-and-stationary or at the spawn, velocity is continuous too.
    """
    resume = list(rec.poses[mark - 1]) if mark > 0 else list(rec.start_pose2d)
    pn.truncate_parallel([rec.frames, rec.poses, rec.goal2d, rec.segment_index], mark)
    budget.rewind_to(mark)
    height = navmesh_height(robot_sim)
    _spawn(robot_sim, [resume[0], height, -resume[1]], float(resume[2]))


def _drive_segment(
    robot_sim,
    rec: PointNavRecording,
    segment: _Segment,
    segment_index: int,
    *,
    move_controller,
    turn_controller,
    monitor,
    budget: pn.FrameBudget,
    fps: int,
    on_tick,
    viewport_debug: bool = False,
) -> Tuple[Optional[str], str, bool, float]:
    """Turn in place onto the path, then drive it. ``(code, detail, truncated, sim_s)``."""
    dt = 1.0 / fps
    ticks = 0

    # -- phase 1: turn in place -------------------------------------------
    turn_controller.reset()
    target = pn.path_tangent_heading(segment.path_xy, at="start")
    turn_controller.set_heading_trajectory([target], handoff_to_path=True)
    turn_timeout = int(math.ceil(abs(segment.turn_rad) / TURN_RATE_MPS) + TURN_TIMEOUT_SLACK_S)
    for _ in range(max(1, turn_timeout) * fps):
        if turn_controller.is_trajectory_complete():
            break
        if _record_tick(robot_sim, rec, segment, segment_index, turn_controller.step(dt=dt),
                        monitor, budget, on_tick, phase="turn", controller=None,
                        fps=fps, viewport_debug=viewport_debug) is False:
            return None, "", True, ticks * dt
        ticks += 1
    else:
        return (pn.TURN_TIMEOUT,
                f"{turn_timeout}s to turn {math.degrees(segment.turn_rad):+.0f}deg",
                False, ticks * dt)

    # -- phase 2: pure pursuit --------------------------------------------
    move_controller.reset()
    move_controller.set_hab_path3(segment.path_3d)
    local_timeout = int(np.ceil(
        move_controller.controller._path_distances[-1] / LOCAL_TIMEOUT_MPS + 0.5
    ))
    progress: List[float] = []
    speeds: List[float] = []
    for _ in range(max(1, local_timeout) * fps):
        step_args = move_controller.step(
            dt=dt, heading_alignment_gain=MOVE_HEADING_GAIN, max_vw=MOVE_MAX_VW
        )
        if _record_tick(robot_sim, rec, segment, segment_index, step_args, monitor, budget,
                        on_tick, phase="drive", controller=move_controller,
                        fps=fps, viewport_debug=viewport_debug) is False:
            return None, "", True, ticks * dt
        ticks += 1
        progress.append(
            float(move_controller.controller._path_distances[move_controller.controller.last_idx])
        )
        speeds.append(float(math.hypot(
            robot_sim.get_joint_state("joint_x", is_vel=True),
            robot_sim.get_joint_state("joint_y", is_vel=True),
        )))
        if robot_sim.is_tipped_over(angle_threshold_degrees=TIPPED_OVER_DEG):
            return pn.TIPPED_OVER, "roll/pitch past 45deg", False, ticks * dt
        if move_controller.is_goal_reached() and robot_sim.is_stationary(0.01, 0.01):
            break
        # A stall ends the phase; whether that is a failure is the goal check's call --
        # a robot that stalls 5 cm from the goal has still arrived.
        if is_stalled(progress, speeds, MOVE_STALL_TICKS, MOVE_STALL_EPS,
                      MOVE_STALL_MAX_SPEED):
            break
    else:
        return (pn.SEGMENT_TIMEOUT, f"{local_timeout}s for {segment.geodesic_m:.2f}m",
                False, ticks * dt)

    return None, "", False, ticks * dt


def _record_tick(
    robot_sim, rec: PointNavRecording, segment: _Segment, segment_index: int,
    step_args, monitor, budget: pn.FrameBudget, on_tick, phase: str, controller,
    fps: int, viewport_debug: bool = False,
) -> bool:
    """One tick, in ``record_segmented_demo``'s exact order. False once the budget is full.

    Observation BEFORE the step (frame ``i`` is what the controller acted on) and pose
    AFTER it (pose ``i`` is where that action landed) -- the same convention the ObjectNav
    corpus was recorded with, so an action chunk means the same thing in both.

    The viewport overlay is issued before ``get_obs()`` for the same reason: debug lines
    are consumed by the next render, so drawing after it puts them a tick late.
    """
    obs = robot_sim.get_obs()
    # Read the camera's world transform NOW, while it still matches the frame just
    # rendered. `on_tick` runs after the physics step, so a caller projecting the path
    # into the frame would otherwise use a camera pose one tick in the future.
    camera = camera_matrix(robot_sim, OVERLAY_SENSOR_UUID) if viewport_debug else None
    robot_sim.step(**step_args, dt=1.0 / fps)
    frame = rgb_view(obs)
    pose = robot_sim.get_2d_pose()

    rec.frames.append(frame)
    rec.poses.append(pose)
    rec.goal2d.append(list(segment.goal2d))
    rec.segment_index.append(int(segment_index))
    budget.add(1)
    monitor.observe(move_controller=controller)

    if on_tick is not None:
        on_tick({
            "pose2d": pose, "goal2d": segment.goal2d, "planned_path": segment.path_xy,
            "segment_index": segment_index, "frame_index": len(rec.frames) - 1,
            "frame": frame, "phase": phase, "geodesic_m": segment.geodesic_m,
            "sim_time": (len(rec.frames) - 1) / float(fps),
            "path_3d": segment.path_3d, "goal_3d": segment.goal_habitat,
            "camera": camera,
        })
    return not budget.exhausted()


def _path_length(poses: Sequence[Any]) -> float:
    """Planar path length over a run of poses, via the task layer's own accumulator."""
    tracker = PathTracker()
    for pose in poses:
        tracker.observe(pose)
    return float(tracker.length)


def _health_detail(health) -> str:
    if health.outcome == "escaped":
        return (f"joint_z free-falling for {health.fall_run_s:.2f}s at up to "
                f"{health.max_fall_speed_mps:.1f}m/s, {health.max_drop_m:.2f}m below spawn")
    if health.outcome == "ref_miss":
        return (f"off its planned path by > {health.ref_miss_flag_m:g}m for "
                f"{health.ref_miss_worst_run_s:.2f}s")
    return health.outcome


def _merge_healths(healths: Sequence[Any], fps: float):
    """One episode-level health block from the kept segments' blocks.

    Per-segment monitoring is what makes rollback sound -- a discarded excursion's ticks
    must not reach the episode's verdict -- so the episode's block is assembled here
    rather than by a monitor that saw everything.
    """
    from continuous_demos.drive_failure import EpisodeHealth

    if not healths:
        return EpisodeHealth()
    merged = EpisodeHealth()
    merged.n_ticks = sum(h.n_ticks for h in healths)
    merged.duration_s = merged.n_ticks / float(fps)
    merged.fall_run_ticks = max(h.fall_run_ticks for h in healths)
    merged.fall_run_s = max(h.fall_run_s for h in healths)
    merged.max_fall_speed_mps = max(h.max_fall_speed_mps for h in healths)
    merged.max_drop_m = max(h.max_drop_m for h in healths)
    merged.fall_speed_mps = healths[0].fall_speed_mps
    merged.fall_duration_s = healths[0].fall_duration_s
    merged.ref_miss_flag_m = healths[0].ref_miss_flag_m
    misses = [h.ref_miss_mean_m for h in healths if h.ref_miss_mean_m is not None]
    if misses:
        weights = [h.n_ticks for h in healths if h.ref_miss_mean_m is not None]
        merged.ref_miss_mean_m = float(np.average(misses, weights=weights))
        merged.ref_miss_p95_m = max(
            h.ref_miss_p95_m for h in healths if h.ref_miss_p95_m is not None
        )
        merged.ref_miss_max_m = max(
            h.ref_miss_max_m for h in healths if h.ref_miss_max_m is not None
        )
        merged.ref_miss_worst_run_s = max(
            h.ref_miss_worst_run_s for h in healths if h.ref_miss_worst_run_s is not None
        )
    for key in ("frac_drive", "frac_turn", "frac_settle", "frac_frozen"):
        vals = [getattr(h, key) for h in healths if getattr(h, key) is not None]
        if vals:
            weights = [h.n_ticks for h in healths if getattr(h, key) is not None]
            setattr(merged, key, round(float(np.average(vals, weights=weights)), 4))
    lens = [h.path_len_m for h in healths if h.path_len_m is not None]
    if lens:
        merged.path_len_m = round(float(sum(lens)), 4)
    return merged
