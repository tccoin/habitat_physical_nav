"""PointNav as an ObjectNav episode with one goal and no object.

The whole of the PointNav task layer, and it is deliberately thin: **a point goal is the
degenerate one-view-point ObjectNav episode**, so nothing here re-derives success, SPL,
soft-SPL, oracle success, oracle SPL or distance-to-goal. Those come from
``continuous_demos.objectnav`` unchanged -- which means a PointNav number and an ObjectNav
number are the same statistic computed by the same code, rather than two things that look
alike in a table.

What that reuse costs is one adapter, here:

* habitat's ``DistanceToGoal`` reads ``goal.position`` under ``distance_to="POINT"`` and
  ``goal.view_points[k].agent_state.position`` under ``"VIEW_POINTS"``. :class:`PointGoal`
  supplies both, pointing at the same point, so **either setting gives the identical
  number** and a run cannot be misconfigured into measuring something else.
* the episode is a plain dataclass rather than a habitat ``NavigationGoal``/``Episode``.
  The measures duck-type everything they touch and the task layer stashes a
  ``_shortest_path_cache`` on the episode; a habitat class would buy nothing and drag the
  ObjectNav-specific ``goals_by_category`` join in with it.

Frames, which is the one place this can be silently wrong: a goal is planar
``[X, Y, theta]`` (``X = habitat_x``, ``Y = -habitat_z``, theta CCW from +X) everywhere the
policy and the rollout loop see it, because that is the frame ``get_2d_pose()`` reports and
therefore the frame the injected value column is written in. It is converted to habitat
world exactly once, in :func:`build_point_episode`, via
``continuous_demos.robot_pose.planar_to_habitat``. Every geodesic query downstream is then
habitat-world like every ObjectNav one.

The heading is carried through but is **not** part of "reached": see
``goal_sources.planar_distance``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from continuous_demos import pointnav as pn
from continuous_demos.robot_pose import planar_heading_to_quat_xyzw, planar_to_habitat

from objectnav_eval.episodes import EvalEpisode, EpisodeSource, register_source, scene_name_of
from objectnav_eval.policies import ActionChunk, Observation, Policy, register_policy

#: What a PointNav episode's ``object_category`` is. Not ``None``: the reporting layer
#: groups by category and prints it, and a literal is more readable than "None" -- while
#: still never colliding with a real ObjectNav category, none of which contain a space.
POINT_CATEGORY = "point goal"


# ===========================================================================
# The episode
# ===========================================================================
@dataclass
class _AgentState:
    """The two fields habitat's ``ObjectViewLocation`` exposes that the measures read."""

    position: np.ndarray
    rotation: Optional[np.ndarray] = None


@dataclass
class _ViewPoint:
    agent_state: _AgentState
    iou: Optional[float] = None


@dataclass
class PointGoal:
    """One point goal, readable by habitat's measures under either ``distance_to``.

    ``view_points`` holds a single view point *at the goal itself*. That is the honest
    degenerate case rather than a trick: an ObjectNav view point is "somewhere the object
    is visible from", and for a point goal the only such place is the point.
    """

    position: np.ndarray
    radius: Optional[float] = None
    view_points: List[_ViewPoint] = field(default_factory=list)


@dataclass
class PointNavEpisode:
    """The episode object the task layer and habitat's measures consume.

    Duck-typed, not a habitat ``Episode`` subclass -- the measures touch ``goals``,
    ``start_position``, ``start_rotation``, ``info`` and the ``_shortest_path_cache`` the
    task layer stashes, and nothing else.
    """

    episode_id: str
    scene_id: str
    start_position: np.ndarray
    start_rotation: np.ndarray
    goals: List[PointGoal]
    info: Dict[str, Any] = field(default_factory=dict)
    object_category: Optional[str] = POINT_CATEGORY
    scene_dataset_config: Optional[str] = None
    #: Planar ``[X, Y, theta]`` of the goal -- the form the policy is fed. Kept beside the
    #: habitat-world ``goals`` rather than re-derived at each use, because the sign on z is
    #: exactly the kind of conversion that is right in three places and wrong in a fourth.
    goal2d: Optional[List[float]] = None
    _shortest_path_cache: Any = None


def build_point_episode(
    scene_id: str,
    start2d: Sequence[float],
    goal2d: Sequence[float],
    height_m: float,
    episode_id: str = "0",
    geodesic_m: float = float("nan"),
    scene_dataset_config: Optional[str] = None,
    goal_height_m: Optional[float] = None,
) -> PointNavEpisode:
    """A one-goal episode from planar start and goal poses.

    ``height_m`` is the **navmesh-level** habitat ``y`` the two points sit at -- what
    ``robot_pose.navmesh_height`` reports and what a pathfinder query is issued at, *not*
    the robot's own height. Using the robot's would put every goal ~0.25 m above the floor,
    which a snap absorbs on a flat floor and which resolves to the wrong storey in a
    stairwell.

    ``goal_height_m`` defaults to ``height_m``; pass it when the two are on different
    levels of a ramp.
    """
    start2d = [float(v) for v in start2d]
    goal2d = [float(v) for v in goal2d]
    if len(start2d) != 3 or len(goal2d) != 3:
        raise ValueError(
            f"start and goal are planar [X, Y, theta]; got {len(start2d)} and "
            f"{len(goal2d)} number(s)"
        )
    goal_position = planar_to_habitat(
        goal2d, height_m if goal_height_m is None else goal_height_m
    )
    return PointNavEpisode(
        episode_id=str(episode_id),
        scene_id=scene_id,
        start_position=planar_to_habitat(start2d, height_m),
        start_rotation=planar_heading_to_quat_xyzw(start2d[2]),
        goals=[
            PointGoal(
                position=goal_position,
                view_points=[_ViewPoint(agent_state=_AgentState(position=goal_position))],
            )
        ],
        info={"geodesic_distance": float(geodesic_m)},
        scene_dataset_config=scene_dataset_config,
        goal2d=list(goal2d),
    )


def to_eval_episode(episode: PointNavEpisode, uid: Optional[str] = None) -> EvalEpisode:
    """Wrap a :class:`PointNavEpisode` for the harness.

    ``extra["goal2d"]`` is what ``runner.EpisodeRunner`` reads to feed the policy its
    point goal, and ``view_points`` is the same single point the measures use -- so
    screening, the euclidean-distance diagnostic and ``DistanceToGoal`` all answer about
    one point rather than about three near-identical ones.
    """
    scene = scene_name_of(episode.scene_id)
    goal_position = np.asarray(episode.goals[0].position, dtype=np.float32).reshape(1, 3)
    return EvalEpisode(
        uid=f"{scene}:{episode.episode_id}",
        episode_id=str(episode.episode_id),
        scene_id=episode.scene_id,
        scene_name=scene,
        object_category=episode.object_category,
        start_position=np.asarray(episode.start_position, dtype=np.float64),
        start_rotation=np.asarray(episode.start_rotation, dtype=np.float64),
        view_points=goal_position,
        episode=episode,
        scene_dataset_config=episode.scene_dataset_config,
        dataset_geodesic=episode.info.get("geodesic_distance"),
        extra={"goal2d": list(episode.goal2d or [])},
    )


def score_trajectory(task, episode: PointNavEpisode, poses: Sequence[Sequence[float]],
                     height: Optional[float] = None, stop_index: Optional[int] = None,
                     evaluate_every: int = 1) -> Dict[str, float]:
    """Score a recorded planar trajectory against one point goal, without stepping.

    A one-line delegation to ``ObjectNavTask.evaluate_trajectory`` and that is the point:
    offline PointNav scoring and offline ObjectNav scoring are the same computation, so
    they are the same call. ``height`` defaults to the episode's own start height, as
    upstream.
    """
    return task.evaluate_trajectory(
        episode, poses, height=height, stop_index=stop_index,
        evaluate_every=evaluate_every,
    )


# ===========================================================================
# The image-blind control: drive at the goal
# ===========================================================================
def pursuit_chunk(
    pose: Sequence[float],
    goal: Sequence[float],
    chunk_len: int = 10,
    dt: float = 0.05,
    v_max: float = 0.6,
    w_max: float = 1.5,
    heading_gate_rad: float = 0.5,
    heading_gain: float = 2.0,
    stop_within_m: float = 0.05,
) -> np.ndarray:
    """``(T, 3)`` chunk of ``[dx, dy, dtheta]`` driving from ``pose`` toward ``goal``.

    The PointNav counterpart of ``policies.ConstantChunkPolicy``: a **control**, not a
    baseline. It is what proves the plumbing -- goal announcement, chunk decoding,
    anchoring, PID tracking, goal chaining -- without a checkpoint, and it is pure
    arithmetic so it is unit-testable with no simulator.

    A unicycle integrated forward in the **anchor's own frame**, because that is the frame
    the executor anchors on (``action_chunking.relative_to_pose``): row ``k`` is the pose
    after ``k + 1`` ticks, expressed relative to the pose the chunk was emitted at. Rows
    are cumulative, not per-tick deltas -- emitting the same offset in every row is the
    mistake that makes a control crawl at one tick's displacement per chunk and look like
    a broken executor.

    ``heading_gate_rad`` is why this is a *turn-then-drive* profile rather than an arc:
    translating while badly misaligned is how a controller ends up circling a goal it
    could have walked to. Below the gate it drives and steers at once.
    """
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
    pose = np.asarray(pose, dtype=float)
    goal = np.asarray(goal, dtype=float)
    # Everything is in the anchor's frame, so the anchor is the origin facing +x and the
    # goal's bearing is the only thing that has to be transported into it.
    dx, dy = float(goal[0] - pose[0]), float(goal[1] - pose[1])
    theta0 = float(pose[2])
    cos0, sin0 = math.cos(theta0), math.sin(theta0)
    gx = cos0 * dx + sin0 * dy
    gy = -sin0 * dx + cos0 * dy

    x = y = theta = 0.0
    rows = np.zeros((int(chunk_len), 3), dtype=np.float64)
    for k in range(int(chunk_len)):
        remaining_x, remaining_y = gx - x, gy - y
        distance = math.hypot(remaining_x, remaining_y)
        if distance <= stop_within_m:
            # Hold the last pose rather than drifting past: the rows are absolute
            # offsets, so repeating one is "stay here", which is exactly right.
            rows[k] = (x, y, theta)
            continue
        bearing = math.atan2(remaining_y, remaining_x)
        error = math.atan2(math.sin(bearing - theta), math.cos(bearing - theta))
        omega = max(-w_max, min(w_max, heading_gain * error))
        speed = v_max if abs(error) < heading_gate_rad else 0.0
        # Cap the step so the chunk cannot command a fly-past of the goal, which the PID
        # would faithfully execute and then have to come back from.
        speed = min(speed, distance / dt)
        theta += omega * dt
        x += speed * math.cos(theta) * dt
        y += speed * math.sin(theta) * dt
        rows[k] = (x, y, theta)
    return rows


def next_waypoint(pose: Sequence[float], path_xy: Any, goal: Sequence[float],
                  lookahead_m: float = 1.0) -> np.ndarray:
    """The point on ``path_xy`` to aim at from ``pose``, or ``goal`` when there is none.

    Classic pure-pursuit target selection, and the reason the control needs it at all:
    **a straight line to the goal goes through walls.** Measured on the first scene tried,
    a control aiming at the goal directly wedged itself against geometry 1 m in and stayed
    there for 190 policy steps -- which proves the executor works and demonstrates nothing
    else. Following the route the sampler already planned makes the control a useful
    stand-in for a policy that can see.

    The target is the first path point at least ``lookahead_m`` away, searched *forward*
    from the closest point on the path. Searching from the start instead makes a path that
    doubles back send the robot to the leg it has already driven; searching from the
    closest point is what makes progress monotone.
    """
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2) if path_xy is not None else None
    goal = np.asarray(goal, dtype=float)[:2]
    if path is None or len(path) == 0:
        return goal
    here = np.asarray(pose, dtype=float)[:2]
    distances = np.linalg.norm(path - here[None, :], axis=1)
    closest = int(np.argmin(distances))
    ahead = np.flatnonzero(distances[closest:] >= float(lookahead_m))
    if len(ahead) == 0:
        return path[-1]
    return path[closest + int(ahead[0])]


class PointGoalPursuitPolicy(Policy):
    """CONTROL: ignores the image and drives to ``Observation.goal_pose``.

    Registered as ``--policy goal_pursuit``. Like ``forward`` and ``zero`` it is how you
    find out the harness works, not a number to report: it is handed the goal -- and, when
    one was planned, the route -- so "it reaches the goal" says the goal chaining, the
    chunk decoding and the PID tracking are sound and says nothing at all about
    navigation. ``uses_observation`` is ``False`` and is written into every record, so a
    control run can never be mistaken for a model run.
    """

    name = "goal_pursuit"
    uses_observation = False

    def __init__(self, chunk_len: int = 10, dt: float = 0.05, speed: float = 0.6,
                 w_max: float = 1.5, lookahead_m: float = 1.0):
        self.chunk_len = int(chunk_len)
        self.dt = float(dt)
        self.speed = float(speed)
        self.w_max = float(w_max)
        self.lookahead_m = float(lookahead_m)

    def reset(self, goal: Optional[str]) -> None:
        pass

    def act(self, obs: Observation) -> ActionChunk:
        if obs.goal_pose is None:
            raise ValueError(
                "the goal_pursuit control needs Observation.goal_pose -- it is the only "
                "thing it reads. Use --episode-source pointnav, or a rollout that carries "
                "a goal source."
            )
        target = next_waypoint(obs.pose, obs.goal_path, obs.goal_pose,
                               lookahead_m=self.lookahead_m)
        return ActionChunk(
            pursuit_chunk(obs.pose, target, chunk_len=self.chunk_len, dt=self.dt,
                          v_max=self.speed, w_max=self.w_max),
            stats={"latency_s": 0.0},
        )

    def describe(self) -> str:
        return (
            f"CONTROL goal_pursuit (image-blind, follows the planned route to the point "
            f"goal when one is given and drives straight at it otherwise; chunk "
            f"{self.chunk_len}, {self.speed} m/s, {self.w_max} rad/s, lookahead "
            f"{self.lookahead_m} m)"
        )


register_policy(
    "goal_pursuit",
    lambda **kw: PointGoalPursuitPolicy(
        chunk_len=kw.get("chunk_len", 10), dt=kw.get("dt", 0.05),
        speed=kw.get("speed", 0.6),
    ),
)


# ===========================================================================
# A held-out episode set, generated from a seed
# ===========================================================================
class PointNavEpisodeSource(EpisodeSource):
    """Point-goal episodes sampled deterministically from a seed and a scene list.

    The demonstration generator makes *demonstrations*, not an evaluation split, so there
    is no file to read. Generating instead of storing is the better answer here rather
    than merely the cheaper one: ``(corpus_seed, scene, index)`` already determines an
    episode through ``continuous_demos.pointnav.episode_seed``, so the seed **is** the
    split -- reproducible without an artifact to lose, and extendable by raising
    ``episodes_per_scene`` without invalidating the episodes already in it.

    Two things it does deliberately:

    * **It samples on the navmesh shipped with the scene**, not on the robot's recomputed
      one. A source is loaded once in the launcher and once per worker, and recomputing a
      robot navmesh needs a whole ``Simulator`` per scene -- prohibitive, and a second
      place for the mesh to differ from the one the metrics use. The shipped mesh is also
      what ``--navmesh dataset`` (the default) measures on, so the episode and the metric
      agree by construction. Episodes the *robot* cannot reach are then screened, with a
      reason code, exactly as ObjectNav's are -- which is the honest outcome rather than a
      hidden one.
    * **It reuses the corpus's own screening.** ``PointNavSampleConfig`` /
      ``screen_segment`` / ``choose_island`` decide what a goal is, so an eval goal is the
      same kind of goal the policy was trained to reach. A separate idea of "too close"
      here would evaluate the policy on a distribution it never saw and the number would
      still look fine.

    ``path`` is a directory of scenes, a glob, or a single ``.glb``. Extra knobs come
    through ``--source-set FIELD=VALUE``::

        --episode-source pointnav \\
        --episodes '/Projects/data/scene_datasets/hm3d/val/*/*.basis.glb' \\
        --source-set episodes_per_scene=5 --source-set seed=101
    """

    name = "pointnav"

    def __init__(
        self,
        path: str,
        episodes_per_scene: int = 5,
        seed: int = 0,
        max_scenes: int = 0,
        min_geodesic_m: float = 3.0,
        max_geodesic_m: float = 15.0,
        start_retries: int = 40,
        segment_retries: int = 60,
    ):
        self.path = str(path)
        self.episodes_per_scene = int(episodes_per_scene)
        self.seed = int(seed)
        self.max_scenes = int(max_scenes)
        # Wider than the generator's 1.5-8.0 m band on purpose: a demonstration segment is
        # sized to fill part of a fixed-length row, while an evaluation episode should be
        # long enough that arriving is navigation rather than an approach manoeuvre. The
        # band is a flag so the two can be matched when that is what is wanted.
        self.config = pn.PointNavSampleConfig(
            min_geodesic_m=float(min_geodesic_m),
            max_geodesic_m=float(max_geodesic_m),
            start_retries=int(start_retries),
            segment_retries=int(segment_retries),
        )

    # ------------------------------------------------------------------
    def scene_paths(self) -> List[Path]:
        """Every scene ``.glb`` named by ``path``: a file, a glob, or a directory."""
        candidate = Path(self.path)
        if candidate.is_file():
            return [candidate]
        if candidate.is_dir():
            found = sorted(candidate.rglob("*.basis.glb")) or sorted(
                candidate.rglob("*.glb")
            )
        else:
            found = sorted(Path(p) for p in _glob(self.path))
        found = [p for p in found if p.suffix == ".glb"]
        if not found:
            raise FileNotFoundError(f"no scene .glb found for {self.path!r}")
        return found[: self.max_scenes] if self.max_scenes else found

    def load(self) -> List[EvalEpisode]:
        from objectnav_eval.navmesh import dataset_pathfinder

        out: List[EvalEpisode] = []
        for scene_path in self.scene_paths():
            pathfinder = dataset_pathfinder(scene_path)
            if pathfinder is None:
                # Loudly skipped rather than silently: a scene whose navmesh is missing
                # contributes zero episodes, and a split that quietly shrank is the
                # failure mode this whole package exists to avoid.
                print(f"[pointnav] no navmesh beside {scene_path.name}; skipping",
                      flush=True)
                continue
            out.extend(self._scene_episodes(scene_path, pathfinder))
        return out

    def _scene_episodes(self, scene_path: Path, pathfinder) -> List[EvalEpisode]:
        from continuous_demos.pointnav_episode import island_table, sample_start

        scene = scene_path.stem.replace(".basis", "")
        episodes: List[EvalEpisode] = []
        islands = island_table(pathfinder)
        for index in range(self.episodes_per_scene):
            episode_seed = pn.episode_seed(self.seed, scene, index)
            rng = np.random.default_rng(episode_seed)
            # Both RNGs, for the reason NavmeshGoalSource seeds both: the draw itself
            # happens inside the pathfinder, and seeding only ours leaves the sample to
            # habitat's global state.
            pathfinder.seed(episode_seed)
            island = pn.choose_island(rng, islands, self.config)
            if island is None:
                break  # this scene has no room for an episode at all
            start_3d = sample_start(pathfinder, rng, self.config, island)
            if start_3d is None:
                continue
            theta0 = float(rng.uniform(-math.pi, math.pi))
            segment, _rejections = _sample_goal(
                pathfinder, rng, self.config, start_3d, theta0, island
            )
            if segment is None:
                continue
            episode = build_point_episode(
                scene_id=str(scene_path),
                start2d=[float(start_3d[0]), -float(start_3d[2]), theta0],
                goal2d=list(segment.goal2d),
                height_m=float(start_3d[1]),
                goal_height_m=float(segment.goal_habitat[1]),
                episode_id=f"pn_{index:05d}",
                geodesic_m=float(segment.geodesic_m),
            )
            episodes.append(to_eval_episode(episode))
        return episodes


def _sample_goal(pathfinder, rng, config, start_3d, theta0, island):
    from continuous_demos.pointnav_episode import sample_segment

    return sample_segment(
        pathfinder, rng, config, start_3d=start_3d, theta_now=theta0, island=island
    )


def _glob(pattern: str) -> List[str]:
    from glob import glob

    return glob(pattern)


register_source("pointnav", PointNavEpisodeSource)
