"""Watch a PointNav policy drive to goals, one after another, and record it.

The free-running counterpart of ``eval_objectnav_policy.py``: no split, no shards, no
aggregate. One scene, one continuous episode, and a **goal source** that decides where to
go next each time the robot arrives. That is exactly the chained-segment structure the
demonstrations have (``docs/POINTNAV_GENERATION.md``), so a policy trained on them is
being asked here for the thing it was trained to do, rather than for a single-goal
special case.

    # image-blind control: proves the plumbing without a checkpoint
    python scripts/pointnav_rollout.py --policy goal_pursuit \
        --scene hm3d/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb \
        --scene-root /Projects/data/scene_datasets \
        --max-goals 3 --layout both --output pointnav_rollout.mp4

    # a checkpoint, over the two-environment bridge
    python scripts/pointnav_rollout.py --policy model \
        --ckpt <dir> --goal-placement segment \
        --policy-python /workspace/conda/envs/longnav_vlm/bin/python \
        --policy-sys-path /Projects/spatial_training/src \
        --scene ... --scene-root ... --output pointnav_rollout.mp4

Everything except the loop is borrowed: :class:`objectnav_eval.executor.ChunkExecutor`
turns chunks into motion, :func:`objectnav_eval.policies.build_policy` loads the policy
(controls and bridged checkpoints alike), :class:`objectnav_eval.video.Recorder` and
:func:`~objectnav_eval.video.annotate` write the MP4, and
:class:`continuous_demos.topdown.TopDownMap` / :mod:`continuous_demos.viewport_overlay`
draw it -- the same two layouts ``pointnav_debug_episode.py`` offers, for the same reason:
the RGB alone cannot tell you whether a trajectory is inside the building.

What is *not* borrowed is the episode: there is no ``EvalEpisode``, no screening and no
step cap in policy steps. An interactive run ends when the goal source runs out, when the
budget does, or when the operator's ``--max-goals`` is reached.

Metrics here are deliberately per goal and deliberately few -- geodesic at issue, planar
distance, path travelled, and the segment SPL ``continuous_demos.pointnav.segment_spl``
already defines. The point of this script is to *watch*; ``eval_objectnav_policy.py
--episode-source pointnav`` is the one that produces numbers to report.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from typing import Optional

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from continuous_demos import pointnav as pn  # noqa: E402
from continuous_demos.episode_io import (  # noqa: E402
    DEFAULT_URDF_PATH,
    build_robot_sim,
    find_scene_dataset_config,
    resolve_scene_path,
)
from continuous_demos.objectnav import PathTracker  # noqa: E402
from continuous_demos.pointnav_episode import island_table, sample_start  # noqa: E402
from continuous_demos.robot_pose import (  # noqa: E402
    navmesh_height,
    planar_heading_to_quat_xyzw,
    robot_navmesh_position,
)
from continuous_demos.topdown import TopDownMap, TopDownState  # noqa: E402
from continuous_demos.viewport_overlay import (  # noqa: E402
    camera_matrix,
    chunk_to_world_xy,
    draw_commanded_chunk,
    draw_goal_in_view,
    draw_path_in_view,
)
from objectnav_eval.budget import GeometricBudget  # noqa: E402
from objectnav_eval.config import EvalConfig  # noqa: E402
from objectnav_eval.executor import ChunkExecutor  # noqa: E402
from objectnav_eval.probes import default_probes  # noqa: E402
from objectnav_eval.goal_sources import (  # noqa: E402
    NavmeshGoalSource,
    ScriptedGoalSource,
    planar_distance,
)
from objectnav_eval.navmesh import apply_navmesh_choice  # noqa: E402
from objectnav_eval.policies import Observation, build_policy  # noqa: E402

# Registers the `goal_pursuit` control. Imported for the side effect, exactly as
# `eval_objectnav_policy.py` does -- a registry entry is not worth a plugin loader.
import objectnav_eval.pointnav  # noqa: E402,F401
from objectnav_eval.video import OverlayState, Recorder, annotate  # noqa: E402

SERVER_SCRIPT = Path(__file__).resolve().parent / "policy_server.py"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class FrameComposer:
    """One tick -> one video frame, in whichever layout was asked for.

    Holds the top-down raster (built once per storey, cached by rounded navmesh height)
    and the executed trajectory, because both are episode-scoped rather than tick-scoped
    and rebuilding either per tick is what makes a debug renderer too slow to leave on.
    """

    def __init__(self, layout: str, meters_per_pixel: float = 0.05, map_scale: int = 3,
                 goal_radius_m: float = 0.5, sensor_uuid: str = "color_sensor",
                 path_lift_m: float = 0.15, chunk_lift_m: float = 0.85):
        self.layout = layout
        self.meters_per_pixel = float(meters_per_pixel)
        self.map_scale = int(map_scale)
        self.goal_radius_m = float(goal_radius_m)
        self.sensor_uuid = sensor_uuid
        # How far off the floor each overlay is drawn. NOT cosmetic: the camera sits at
        # 1.1 m with a 63.6 deg vertical FOV, so the floor first enters frame at
        # 1.1/tan(31.8 deg) = 1.77 m. A polyline drawn at floor level is therefore
        # invisible for everything nearer than that -- which is the whole of a commanded
        # chunk (0.8 s of motion, well under a metre) and the near half of any planned
        # path. Lifting trades positional literalness for being visible at all.
        self.path_lift_m = float(path_lift_m)
        self.chunk_lift_m = float(chunk_lift_m)
        self.executed: list = []
        self.past_goals: list = []
        #: ``(anchor_pose, chunk)`` from the last policy step, drawn on every frame
        #: until replaced. The top-down panel additionally needs it as planar points,
        #: which `chunk_to_world_xy` supplies.
        self.last_chunk = None
        self._maps: dict = {}

    @property
    def wants_topdown(self) -> bool:
        return self.layout in ("topdown", "both")

    @property
    def wants_viewport(self) -> bool:
        return self.layout in ("viewport", "both")

    def topdown(self, robot_sim):
        key = round(navmesh_height(robot_sim), 1)
        if key not in self._maps:
            self._maps[key] = TopDownMap.for_pathfinder(
                robot_sim.sim.pathfinder, key, meters_per_pixel=self.meters_per_pixel
            )
        return self._maps[key]

    def compose(self, robot_sim, rgb, state: dict):
        """``state`` carries the tick's pose, goal, planned path and captions."""
        pose = np.asarray(state["pose2d"], dtype=float)
        goal = state.get("goal2d")
        if self.wants_viewport:
            camera = camera_matrix(robot_sim, self.sensor_uuid)
            if camera is not None:
                cam_to_world, (_w, _h, hfov) = camera
                if state.get("path_3d") is not None:
                    rgb = draw_path_in_view(rgb, state["path_3d"], cam_to_world, hfov,
                                            lift_m=self.path_lift_m)
                if self.last_chunk is not None:
                    # The same call `objectnav_eval.runner` makes for `--overlay-chunk`;
                    # one implementation, so the two visualisations cannot disagree about
                    # what green means or where it is drawn.
                    rgb = draw_commanded_chunk(
                        rgb, robot_sim, self.last_chunk[0], self.last_chunk[1],
                        sensor_uuid=self.sensor_uuid, lift_m=self.chunk_lift_m,
                    )
                if state.get("goal_3d") is not None:
                    rgb = draw_goal_in_view(rgb, state["goal_3d"], cam_to_world, hfov,
                                            radius_m=self.goal_radius_m)
        frame = annotate(rgb, OverlayState(
            step=state["step"],
            sim_time=state["sim_time"],
            goal=f"g{state['goal_index']}",
            distance=(float("nan") if goal is None
                      else planar_distance(pose, goal)),
            start_distance=state.get("geodesic_m", float("nan")),
            reached=bool(state.get("reached")),
            is_observation=bool(state.get("is_observation")),
            extra=state.get("note", ""),
        ))
        if not self.wants_topdown:
            return frame
        panel = self.topdown(robot_sim).panel(
            TopDownState(
                pose2d=pose,
                goal2d=goal,
                planned_path=state.get("planned_path"),
                commanded=(None if self.last_chunk is None
                           else chunk_to_world_xy(*self.last_chunk)),
                executed=np.asarray(self.executed) if len(self.executed) > 1 else None,
                past_goals=self.past_goals,
                segment_index=state["goal_index"],
                frame_index=state["tick"],
                note=state.get("note", ""),
            ),
            side_px=frame.shape[0],
            scale=self.map_scale,
        )
        return np.hstack([frame, panel])


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------
def spawn(robot_sim, args, rng):
    """Put the robot somewhere navigable, and return its planar pose.

    ``--start`` is honoured verbatim (its height comes from the navmesh at that point);
    otherwise a point is drawn on an island big enough to hold an episode, with a random
    heading -- the same choice ``pointnav_episode._spawn`` makes, and for the same reason:
    inheriting the sampler's arbitrary orientation biases every episode's first turn.
    """
    pathfinder = robot_sim.sim.pathfinder
    config = pn.PointNavSampleConfig()
    if args.start:
        x, y, theta = (float(v) for v in args.start.split(","))
        point = np.array([x, navmesh_height(robot_sim), -y], dtype=np.float32)
        point = pathfinder.snap_point(point)
    else:
        island = pn.choose_island(rng, island_table(pathfinder), config)
        if island is None:
            raise SystemExit(
                f"no navmesh island in {args.scene} is big enough to hold an episode "
                f"(min area {config.min_island_area_m2} m2)"
            )
        point = sample_start(pathfinder, rng, config, island)
        if point is None:
            raise SystemExit("no navigable spawn point drawn")
        theta = float(rng.uniform(-np.pi, np.pi))
    robot_sim.reset(
        robot_translation=np.asarray(point, dtype=np.float32).copy(),
        robot_rotation=planar_heading_to_quat_xyzw(theta),
        snap_to_navmesh=False,
        vertical_offset=None,
    )
    return np.asarray(robot_sim.get_2d_pose(), dtype=float)


def geodesic_to(robot_sim, goal2d) -> float:
    """Navmesh distance from where the robot is to a planar goal.

    Both endpoints at navmesh height, via ``robot_navmesh_position`` -- issuing the query
    at the robot's own height is absorbed by the snap on a flat floor and resolves to the
    storey above in a stairwell.
    """
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = robot_navmesh_position(robot_sim)
    path.requested_end = np.array(
        [float(goal2d[0]), navmesh_height(robot_sim), -float(goal2d[1])], dtype=np.float32
    )
    if not robot_sim.sim.pathfinder.find_path(path):
        return float("inf")
    return float(path.geodesic_distance)


def run(robot_sim, policy, source, composer, recorder, args):
    """Drive until the goal source runs out, the budget does, or ``--max-goals`` is hit."""
    from continuous_demos.pid_pose_controller import PIDPoseController

    from objectnav_eval.control import build_pid_config

    # The SAME probes `eval_objectnav_policy.py` attaches, from the same factory, so a
    # PointNav rollout's motion numbers are comparable with an ObjectNav eval's rather
    # than being a second definition of "driving" that happens to agree today.
    # `--probe-contacts` gates the collision probe because it is the one that costs
    # simulator time (the others are a few flops on data the executor already produced).
    probes = default_probes(dt=args.dt, collect_contacts=args.probe_contacts)
    executor = ChunkExecutor(
        robot_sim, PIDPoseController(robot_sim, build_pid_config(args.pid_preset)),
        dt=args.dt, gap=args.gap, sensor_uuid=args.sensor_uuid,
        collect_contacts=args.probe_contacts,
    )
    for probe in probes:
        probe.reset(None)
    tracker = PathTracker()
    budget = GeometricBudget(budget=float(args.budget_units))

    pose = np.asarray(robot_sim.get_2d_pose(), dtype=float)
    tracker.reset(pose)
    budget.reset(pose)
    source.reset(robot_sim)
    policy.reset(None)

    goals: list = []
    goal = source.next_goal(robot_sim, pose)
    if goal is None:
        raise SystemExit("the goal source produced no first goal")
    record = _open_goal(robot_sim, goal, source, tracker, 0, 0)
    goals.append(record)

    step = 0
    reached_now = False
    trace: list = []
    terminated = "goal source exhausted"
    while step < args.max_steps:
        rgb = np.asarray(robot_sim.get_obs()[args.sensor_uuid])[..., :3]
        pose = np.asarray(robot_sim.get_2d_pose(), dtype=float)
        if recorder.enabled:
            recorder.add(composer.compose(robot_sim, rgb, _state(
                step, executor, record, pose, reached_now, True, args)))

        chunk = policy.act(Observation(
            rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
            goal=None, step=step, pose=pose, sim_time=executor.sim_time,
            # The point goal. A CHANGE of this value is what the model side reads as a
            # new goal announcement -- see bridge._PoseFeed -- so it is issued the moment
            # the source hands one over and not a tick later.
            goal_pose=np.asarray(record["goal2d"], dtype=float),
            # The route, for the image-blind control only -- it never crosses the
            # bridge, because a policy that could read the plan would not be solving
            # the task and no training conversation contains one.
            goal_path=record["planned_path"],
        ))

        # The chunk is relative to the pose it was emitted at; compose it back into
        # world poses so the overlay can draw what the model asked for next to where the
        # robot actually went. `relative_to_pose` is the exact inverse of the transform
        # the training targets were built with, so this is the model's own output and not
        # a re-derivation of it.
        composer.last_chunk = (pose, chunk)

        def on_tick(tick, _record=record):
            tracker.observe(tick.pose)
            # Per-tick trace, for the offline goal-directedness analysis. Poses only --
            # everything else it needs (geodesic distance, the descent direction) is a
            # pathfinder query that can be made later, off the critical path.
            trace.append([float(tick.pose[0]), float(tick.pose[1]), float(tick.pose[2]),
                          int(_record["index"])])
            for probe in probes:
                probe.on_tick(tick)
            budget.observe(tick.pose)
            composer.executed.append([float(tick.pose[0]), float(tick.pose[1])])
            if recorder.enabled and tick.tick_in_chunk < executor.gap - 1:
                frame = np.asarray(robot_sim.get_obs()[args.sensor_uuid])[..., :3]
                recorder.add(composer.compose(robot_sim, frame, _state(
                    tick.chunk_index, executor, _record, tick.pose,
                    reached_now, False, args)))

        executor.execute(chunk, step, on_tick=on_tick)
        step += 1
        pose = np.asarray(robot_sim.get_2d_pose(), dtype=float)

        if budget.exhausted:
            terminated = "budget exhausted"
            break
        if executor.sim_time >= args.time_limit_s:
            terminated = "sim time limit"
            break
        if robot_sim.is_tipped_over():
            terminated = "robot tipped over"
            break
        if chunk.stop:
            terminated = "policy called stop"
            break

        distance = planar_distance(pose, record["goal2d"])
        record["final_dist_m"] = distance
        if distance > args.goal_threshold:
            reached_now = False
            continue

        # Arrived. Close this goal out and ask for another; a source that has none left
        # ends the rollout rather than the loop inventing one.
        reached_now = True
        _close_goal(record, tracker, reached=True, steps=step)
        print(f"  goal {record['index']}: REACHED in {record['steps']} steps, "
              f"{record['travelled_m']:.2f} m travelled against a "
              f"{record['geodesic_m']:.2f} m geodesic (spl {record['spl']:.2f})",
              flush=True)
        if args.max_goals and len(goals) >= args.max_goals:
            terminated = "max goals"
            break
        goal = source.next_goal(robot_sim, pose)
        if goal is None:
            break
        composer.past_goals.append(list(record["goal2d"]))
        record = _open_goal(robot_sim, goal, source, tracker, len(goals), step)
        goals.append(record)
        print(f"  goal {record['index']}: [{record['goal2d'][0]:.2f}, "
              f"{record['goal2d'][1]:.2f}] at {record['geodesic_m']:.2f} m geodesic",
              flush=True)
    else:
        terminated = "max steps"

    if not goals[-1].get("closed"):
        _close_goal(goals[-1], tracker, reached=False, steps=step)
    return {
        "goals": goals,
        "reached": sum(1 for g in goals if g["reached"]),
        "steps": step,
        "ticks": executor.tick,
        "sim_seconds": executor.sim_time,
        "path_length_m": tracker.length,
        "budget": budget.summary(),
        "terminated": terminated,
        "probes": {p.name: p.summary() for p in probes},
        # [X, Y, theta, goal_index] per control tick. Written only when asked for, since
        # a 150-step episode is 1500 rows.
        "trace": trace,
    }


def plan_to(robot_sim, goal2d):
    """The navmesh route from here to ``goal2d``: planar ``(M, 2)`` and habitat ``(M, 3)``.

    For the **overlay and the image-blind control only** -- it never crosses the bridge,
    because a policy that could read the plan would not be solving the task and no
    training conversation contains one. ``NavmeshGoalSource`` already has this as a
    by-product of screening; a scripted or pushed goal does not, and without it the map
    shows no magenta route and the control drives the straight line into a wall.
    """
    from continuous_demos.pointnav_episode import habitat_xy_to_planar, plan_path

    points, _geodesic = plan_path(
        robot_sim.sim.pathfinder,
        robot_navmesh_position(robot_sim),
        np.array([float(goal2d[0]), navmesh_height(robot_sim), -float(goal2d[1])],
                 dtype=np.float32),
    )
    if points is None:
        return None, None
    return habitat_xy_to_planar(points), np.asarray(points, dtype=float)


def _open_goal(robot_sim, goal, source, tracker, index, step) -> dict:
    path_xy = getattr(source, "planned_path", None)
    path_3d = getattr(source, "planned_path_3d", None)
    if path_xy is None:
        path_xy, path_3d = plan_to(robot_sim, goal)
    return {
        "index": int(index),
        "goal2d": [float(v) for v in goal],
        "geodesic_m": geodesic_to(robot_sim, goal),
        "path_at_start_m": float(tracker.length),
        "planned_path": path_xy,
        "planned_path_3d": path_3d,
        "goal_3d": np.array(
            [float(goal[0]), navmesh_height(robot_sim), -float(goal[1])], dtype=float
        ),
        "reached": False,
        "closed": False,
        "final_dist_m": float("nan"),
        "steps": 0,
        "travelled_m": 0.0,
        "spl": 0.0,
        "step_opened": int(step),
    }


def _close_goal(record: dict, tracker: PathTracker, reached: bool, steps: int) -> None:
    """Freeze a goal's per-segment numbers.

    ``segment_spl`` is imported rather than restated: the generator scores a demonstration
    segment with it, so a rollout segment and a demonstration segment are efficient in the
    same sense and the two tables can be read side by side.
    """
    record["travelled_m"] = float(tracker.length) - float(record["path_at_start_m"])
    record["reached"] = bool(reached)
    record["spl"] = pn.segment_spl(record["geodesic_m"], record["travelled_m"], reached)
    record["steps"] = int(steps) - int(record["step_opened"])
    record["closed"] = True


def _state(step, executor, record, pose, reached, is_observation, args) -> dict:
    return {
        "step": step,
        "tick": executor.tick,
        "sim_time": executor.sim_time,
        "pose2d": pose,
        "goal2d": record["goal2d"],
        "goal_index": record["index"],
        "geodesic_m": record["geodesic_m"],
        "planned_path": record["planned_path"],
        "path_3d": record["planned_path_3d"],
        "goal_3d": record["goal_3d"],
        "reached": reached,
        "is_observation": is_observation,
        "note": f"goal {record['index']}  t={executor.sim_time:.1f}s",
    }


# ---------------------------------------------------------------------------
def build_source(args, rng):
    if args.goals:
        return ScriptedGoalSource(
            [[float(v) for v in leg.split(",")] for leg in args.goals.split(";") if leg],
            loop=args.loop_goals,
        )
    return NavmeshGoalSource(
        config=pn.PointNavSampleConfig(
            min_geodesic_m=args.min_geodesic, max_geodesic_m=args.max_geodesic,
        ),
        seed=args.seed,
        max_goals=args.max_goals,
    )


def build_eval_config(args) -> EvalConfig:
    """The subset of ``EvalConfig`` ``policies.build_policy`` reads.

    Reused rather than reimplemented so ``--policy model`` here launches exactly the
    policy server ``eval_objectnav_policy.py`` would, with the same flags forwarded the
    same way -- including ``--goal-placement``, which is the one thing a PointNav
    checkpoint cannot be run without.
    """
    return EvalConfig(
        policy=args.policy,
        policy_backend=args.policy_backend,
        ckpt=args.ckpt,
        device=args.device,
        policy_python=args.policy_python,
        policy_sys_path=args.policy_sys_path,
        policy_decode=args.policy_decode,
        policy_seed=args.policy_seed,
        goal_placement=args.goal_placement,
        anchor_offset=args.anchor_offset,
        control_chunk_len=args.control_chunk_len,
        control_speed=args.control_speed,
        dt=args.dt,
        gap=args.gap,
        output_dir=str(Path(args.output).parent),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene", required=True, help="scene id or .glb path")
    p.add_argument("--scene-root", default=None)
    p.add_argument("--output", default="pointnav_rollout.mp4")
    p.add_argument("--probe-contacts", action="store_true",
                   help="also run the collision probe (collisions_per_m, obstructed_frac). "
                        "Costs simulator time per tick, which is why it is opt-in")
    p.add_argument("--run-label", default=None,
                   help="groups this rollout in analyse_goal_directedness.py (default: "
                        "the checkpoint's run/checkpoint names)")
    p.add_argument("--metrics-json", default=None,
                   help="write the per-goal table here as well as printing it")
    p.add_argument("--no-video", dest="record_video", action="store_false",
                   help="run without writing an MP4 (much faster; the point of this "
                        "script is usually the video, so this is for smoke tests)")

    goals = p.add_argument_group("goals")
    goals.add_argument("--goals", default=None,
                       help="a scripted route: 'X,Y;X,Y[,theta];...' in the planar "
                            "control frame. Overrides the navmesh sampler, and is the "
                            "only way to compare two policies on the SAME route")
    goals.add_argument("--loop-goals", action="store_true",
                       help="restart a scripted route when it runs out")
    goals.add_argument("--max-goals", type=int, default=0, help="0 = until something else "
                                                                "ends the rollout")
    goals.add_argument("--goal-threshold", type=float, default=0.5,
                       help="planar distance at which a goal counts as reached and the "
                            "next one is issued. Position only, never heading")
    goals.add_argument("--min-geodesic", type=float, default=1.5)
    goals.add_argument("--max-geodesic", type=float, default=8.0)
    goals.add_argument("--start", default=None, help="spawn at 'X,Y,theta' instead of a "
                                                     "sampled navmesh point")
    goals.add_argument("--seed", type=int, default=0)

    run_g = p.add_argument_group("rollout")
    run_g.add_argument("--max-steps", type=int, default=400)
    run_g.add_argument("--dt", type=float, default=0.05)
    run_g.add_argument("--gap", type=int, default=5)
    run_g.add_argument("--budget-units", type=float, default=2000.0,
                       help="discrete-step equivalents of motion before the rollout ends "
                            "(0.25 m or 30 deg each). Generous by default: this is a "
                            "chained run, not one capped episode")
    run_g.add_argument("--time-limit-s", type=float, default=600.0,
                       help="MANDATORY sim-time backstop, for the reason "
                            "docs/GEOMETRIC_BUDGET.md gives: a frozen agent and one "
                            "pushing into a wall both spend no budget at all")
    run_g.add_argument("--pid-preset", default="baseline")
    run_g.add_argument("--navmesh", default="robot", choices=["robot", "dataset"],
                       help="which mesh the goals are sampled and measured on. 'robot' "
                            "(the default HERE, unlike the eval harness) is the honest "
                            "one for a free run: a sampled goal the robot cannot reach "
                            "is a wasted rollout rather than a screened episode")

    sim_g = p.add_argument_group("simulator")
    sim_g.add_argument("--width", type=int, default=640)
    sim_g.add_argument("--height", type=int, default=480)
    sim_g.add_argument("--panoramic", type=int, default=1)
    sim_g.add_argument("--sensor-uuid", default="color_sensor",
                       help="against a rig (--panoramic > 1) this must be "
                            "'color_sensor_0'; plain 'color_sensor' does not exist there")
    sim_g.add_argument("--urdf-path", default=str(DEFAULT_URDF_PATH))

    vid = p.add_argument_group("video")
    vid.add_argument("--layout", choices=("topdown", "viewport", "both"), default="both")
    vid.add_argument("--path-lift", type=float, default=0.15,
                     help="metres to lift the planned path off the floor when drawing it "
                          "into the RGB. The floor only enters frame beyond ~1.77 m "
                          "(camera 1.1 m, vertical FOV 63.6 deg), so a floor-level line is "
                          "invisible in the near field")
    vid.add_argument("--chunk-lift", type=float, default=0.85,
                     help="same, for the commanded chunk. Larger because a chunk is under "
                          "a metre long and would otherwise never be in frame at all")
    vid.add_argument("--meters-per-pixel", type=float, default=0.05)
    vid.add_argument("--map-scale", type=int, default=3)
    vid.add_argument("--video-scale", type=float, default=1.0)

    mdl = p.add_argument_group("policy")
    mdl.add_argument("--policy", default="goal_pursuit",
                     help="'model', or a control: 'goal_pursuit' drives at the goal, "
                          "'forward' drives straight, 'zero' stands still")
    mdl.add_argument("--policy-backend", default="vector_rollout")
    mdl.add_argument("--ckpt", default=None)
    mdl.add_argument("--device", default="cuda")
    mdl.add_argument("--policy-python", default=None)
    mdl.add_argument("--policy-sys-path", default=None)
    mdl.add_argument("--policy-decode", default=None)
    mdl.add_argument("--policy-seed", type=int, default=None)
    mdl.add_argument("--anchor-offset", default=None,
                     help="DIAGNOSTIC 'DX,DY[,DTHETA_RAD]'. Moves the origin "
                          "relative_se2 anchors on, WITHOUT changing the model's "
                          "context: the seeded row is never emitted, so the text, "
                          "images, marker count and injected-row count are all "
                          "identical and only the pose values move. Answers whether "
                          "the same task gets harder from a distant/rotated anchor")
    mdl.add_argument("--goal-placement", default="segment",
                     choices=["segment", "observation"],
                     help="where the goal's <pose> marker sits; must match the placement "
                          "the checkpoint's CORPUS was formatted with. Only read by "
                          "--policy model")
    mdl.add_argument("--control-chunk-len", type=int, default=10)
    mdl.add_argument("--control-speed", type=float, default=0.6)
    args = p.parse_args()

    scene_path = resolve_scene_path(args.scene, args.scene_root)
    robot_sim = build_robot_sim(
        scene_path=scene_path,
        scene_dataset_config=find_scene_dataset_config(scene_path, args.scene_root),
        width=args.width, height=args.height, urdf_path=args.urdf_path,
        num_cameras=args.panoramic,
    )
    print(f"scene: {scene_path}")
    print(f"navmesh: {apply_navmesh_choice(robot_sim, scene_path, args.navmesh)}")

    rng = np.random.default_rng(args.seed)
    robot_sim.sim.pathfinder.seed(args.seed)
    source = build_source(args, rng)
    print(f"goals: {source.describe()}")

    policy = build_policy(
        build_eval_config(args), str(SERVER_SCRIPT),
        str(Path(args.output).with_suffix(".policy.log")),
    )
    composer = FrameComposer(
        args.layout, meters_per_pixel=args.meters_per_pixel, map_scale=args.map_scale,
        path_lift_m=args.path_lift, chunk_lift_m=args.chunk_lift,
        goal_radius_m=args.goal_threshold, sensor_uuid=args.sensor_uuid,
    )
    recorder = Recorder(args.output, fps=max(1, round(1.0 / args.dt)),
                        scale=args.video_scale, enabled=args.record_video)
    try:
        spawn(robot_sim, args, rng)
        summary = run(robot_sim, policy, source, composer, recorder, args)
    finally:
        recorder.close()
        policy.close()
        robot_sim.sim.close(destroy=True)

    print(f"\n{summary['reached']}/{len(summary['goals'])} goal(s) reached in "
          f"{summary['steps']} policy steps ({summary['sim_seconds']:.1f} s sim, "
          f"{summary['path_length_m']:.1f} m travelled); ended: {summary['terminated']}")
    print("  idx   geodesic  travelled     spl   final   reached")
    for goal in summary["goals"]:
        print(f"  {goal['index']:3d}   {goal['geodesic_m']:8.2f}  {goal['travelled_m']:9.2f}"
              f"  {goal['spl']:6.3f}  {goal['final_dist_m']:6.2f}   {goal['reached']}")
    if recorder.enabled:
        print(f"\nWrote {args.output} ({recorder.frames} frames at "
              f"{max(1, round(1.0 / args.dt))} fps = real time)")
    # Provenance the offline analysis groups by. Written into the same blob so a
    # directory of runs is self-describing without a naming convention.
    summary["scene"] = args.scene
    summary["run"] = args.run_label or (Path(args.ckpt).parent.name + "/" + Path(args.ckpt).name
                                        if args.ckpt else args.policy)
    summary["seed"] = args.seed
    if args.metrics_json:
        Path(args.metrics_json).write_text(json.dumps(_jsonable(summary), indent=2))
        print(f"Wrote {args.metrics_json}")
    return 0 if summary["reached"] else 1


def _jsonable(value):
    """Drop the arrays the video needed and make the rest strict JSON."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()
                if k not in ("planned_path", "planned_path_3d", "goal_3d")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


if __name__ == "__main__":
    sys.exit(main())
