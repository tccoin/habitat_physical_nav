"""ObjectNav task layer for :class:`~continuous_demos.sim.HabitatRobotSim`.

This sits **between a simulator and an RL ``Env``**. It adds the two things
``sim.py`` deliberately does not have -- episode setup (spawn the robot for an
ObjectNav episode) and ObjectNav metrics (success, SPL, soft-SPL, plus oracle
success / oracle SPL) -- while making **no assumptions about the action space**
and without wrapping stepping or rendering.

Why no ``step(action)``
-----------------------
Continuous control here means action chunking, real-time mixing of overlapping
trajectories, and deliberately simulated inference lag; none of that maps onto
one action per env step. So the caller keeps driving ``robot_sim.step(...)``
and ``robot_sim.get_obs()`` directly, and this task exposes *pull-based*
queries instead:

    task.reset(episode)     # spawn + reset measures            (once)
    task.observe_pose()     # O(1): feed the path-length tracker (per tick)
    task.call_stop()        # the agent declares termination     (when it wants)
    task.evaluate()         # navmesh queries + measures         (when it wants)

Cost is therefore decoupled from physics: stepping never triggers a geodesic
query or a render. Path length is the one quantity that must be integrated over
time, so it is tracked separately by the cheap :class:`PathTracker` rather than
inside SPL (upstream accumulates it inside ``update_metric``, which would tie
path length to metric-evaluation cadence).

Reuse
-----
``DistanceToGoal``, ``Success`` and ``SoftSPL``/``SPL`` come straight from
habitat-lab, so numbers are comparable to published ObjectNav results. They need
only three things from a "simulator" -- ``geodesic_distance``,
``get_agent_state``, ``previous_step_collided`` -- which :class:`NavSimAdapter`
provides on top of ``HabitatRobotSim`` (duck-typed; no ``Simulator`` subclass,
no hydra config, no ``Env``). habitat-lab has no oracle measures, so
:class:`OracleSuccess` / :class:`OracleSPL` are defined here.

Frames: ``HabitatRobotSim`` reports planar ``[X, Y, theta]`` while every
episode/goal coordinate is habitat world ``(x, y, z)``. Per ``sim.py``'s header,
``X = habitat_x`` and ``Y = -habitat_z``; :meth:`NavSimAdapter.get_agent_state`
is the one place that inversion happens.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import habitat_sim
from habitat.core.embodied_task import Measure, Measurements
from habitat.core.simulator import AgentState
from habitat.tasks.nav.nav import SPL, DistanceToGoal, SoftSPL, Success

from continuous_demos.sim import HabitatRobotSim


@dataclass
class ObjectNavConfig:
    """Task configuration.

    Defaults are the habitat ObjectNav leaderboard settings: geodesic distance
    to the nearest goal *view point*, success under 0.1 m.
    """

    success_distance: float = 0.1
    distance_to: str = "VIEW_POINTS"  # or "POINT"
    # The robot base does not sit exactly on the navmesh the way a point agent
    # does; snapping the reported position keeps geodesic queries well-posed.
    snap_agent_position: bool = True
    include_soft_spl: bool = True
    include_oracle: bool = True


# ---------------------------------------------------------------------------
# Simulator adapter: the habitat `Simulator` surface the nav measures require
# ---------------------------------------------------------------------------


def planar_to_habitat_position(pose2d: Sequence[float], height: float) -> np.ndarray:
    """``[X, Y, theta]`` (planar control frame) -> habitat world ``(x, y, z)``."""
    return np.array([pose2d[0], height, -pose2d[1]], dtype=np.float32)


def planar_heading_to_habitat_quat(theta: float):
    """Planar heading -> habitat rotation quaternion (about +y).

    Habitat's forward is local ``-z``; a rotation of ``phi`` about ``+y`` sends
    ``-z`` to ``(-sin phi, 0, -cos phi)``. We need that to equal the planar
    heading direction ``(cos theta, 0, -sin theta)``, giving ``phi = theta -
    pi/2`` (check: ``theta = 0`` -> ``phi = -pi/2`` -> ``+x``, which is planar
    ``+X``).
    """
    return habitat_sim.utils.common.quat_from_angle_axis(
        theta - math.pi / 2.0, np.array([0.0, 1.0, 0.0])
    )


class _NavAdapterBase:
    """Shared implementation of the three methods the nav measures touch."""

    def __init__(self, pathfinder, snap_agent_position: bool = True):
        self.pathfinder = pathfinder
        self.snap_agent_position = snap_agent_position

    # -- to be provided by subclasses -------------------------------------
    def _planar_pose(self) -> np.ndarray:
        raise NotImplementedError

    def _height(self) -> float:
        raise NotImplementedError

    # -- habitat Simulator surface ----------------------------------------
    def geodesic_distance(self, position_a, position_b, episode=None) -> float:
        """Copied from ``HabitatSim.geodesic_distance`` (habitat-lab 0.3.3).

        Kept verbatim (including the ``episode._shortest_path_cache`` reuse) so
        distances match habitat's own numbers exactly.
        """
        if episode is None or getattr(episode, "_shortest_path_cache", None) is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)
        self.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        pose = self._planar_pose()
        position = planar_to_habitat_position(pose, self._height())
        if self.snap_agent_position and self.pathfinder.is_loaded:
            snapped = self.pathfinder.snap_point(position)
            if not np.isnan(snapped[0]):
                position = np.array(snapped, dtype=np.float32)
        return AgentState(
            position=position, rotation=planar_heading_to_habitat_quat(pose[2])
        )

    @property
    def previous_step_collided(self) -> bool:
        # HabitatRobotSim exposes no collision flag; only habitat's `Collisions`
        # measure needs this, which the task does not enable.
        return False


class NavSimAdapter(_NavAdapterBase):
    """Live adapter reading the robot's pose out of a ``HabitatRobotSim``."""

    def __init__(self, robot_sim: HabitatRobotSim, snap_agent_position: bool = True):
        super().__init__(robot_sim.sim.pathfinder, snap_agent_position)
        self.robot_sim = robot_sim

    def _planar_pose(self) -> np.ndarray:
        return self.robot_sim.get_2d_pose()

    def _height(self) -> float:
        """Habitat-world y of the base *before* the URDF mounting offset.

        ``HabitatRobotSim.reset`` shifts the robot up by ``-cumulative_bb.min.y``
        so the model rests on the floor; recomputing the same quantity recovers
        the navmesh-level height that goals and view points are expressed at.
        """
        aabb = self.robot_sim.robot.root_scene_node.cumulative_bb
        vertical_offset = -aabb.min.y if aabb.min.y < 0 else 0.0
        return float(self.robot_sim.robot.translation[1]) - float(vertical_offset)


class StaticPoseNavAdapter(_NavAdapterBase):
    """Adapter over an externally supplied pose -- for scoring recorded
    trajectories with no simulation stepping (see
    :meth:`ObjectNavTask.evaluate_trajectory`)."""

    def __init__(self, pathfinder, height: float, snap_agent_position: bool = True):
        super().__init__(pathfinder, snap_agent_position)
        self.height = height
        self.pose = np.zeros(3, dtype=float)

    def set_pose(self, pose2d: Sequence[float]) -> None:
        self.pose = np.asarray(pose2d, dtype=float)

    def _planar_pose(self) -> np.ndarray:
        return self.pose

    def _height(self) -> float:
        return self.height


# ---------------------------------------------------------------------------
# Path length, tracked independently of metric evaluation
# ---------------------------------------------------------------------------


class PathTracker:
    """Accumulates travelled path length from planar poses.

    O(1) per sample, no navmesh and no rendering, so it is cheap enough to feed
    every physics tick. The caller owns the cadence: sampling *coarser* than the
    motion under-counts path length and therefore inflates SPL.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self, pose2d: Optional[Sequence[float]] = None) -> None:
        self.length = 0.0
        self.num_samples = 0
        self._previous = None
        if pose2d is not None:
            self.observe(pose2d)

    def observe(self, pose2d: Sequence[float]) -> float:
        xy = np.asarray(pose2d, dtype=float)[:2]
        if self._previous is not None:
            self.length += float(np.linalg.norm(xy - self._previous))
        self._previous = xy
        self.num_samples += 1
        return self.length


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


def _path_ratio(start_end_distance: float, path_length: float) -> float:
    """SPL's path-efficiency factor, guarding the degenerate 0/0 case."""
    denominator = max(start_end_distance, path_length)
    if denominator <= 0.0:
        return 1.0
    return start_end_distance / denominator


class ContinuousSPL(SPL):
    """habitat's SPL, with path length read from ``task.path_tracker``.

    Upstream accumulates path length inside ``update_metric``, which would make
    SPL depend on how often metrics are evaluated. The formula is unchanged.
    """

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._agent_episode_distance = task.path_tracker.length
        self._metric = ep_success * _path_ratio(
            self._start_end_episode_distance, self._agent_episode_distance
        )


class ContinuousSoftSPL(SoftSPL):
    """habitat's SoftSPL, with path length read from ``task.path_tracker``."""

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if self._start_end_episode_distance <= 0.0:
            ep_soft_success = 1.0
        else:
            ep_soft_success = max(
                0.0, 1.0 - distance_to_target / self._start_end_episode_distance
            )
        self._agent_episode_distance = task.path_tracker.length
        self._metric = ep_soft_success * _path_ratio(
            self._start_end_episode_distance, self._agent_episode_distance
        )


class OracleSuccess(Measure):
    """Success as decided by an oracle that stops the episode the moment the
    agent is close enough -- i.e. ignoring whether the agent called ``STOP``.

    Sticky: once the threshold is met it stays 1.0. Only samples at which the
    caller actually evaluates are seen, so this is a lower bound at the
    caller's evaluation cadence.
    """

    cls_uuid: str = "oracle_success"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._success_distance = config.success_distance
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if distance_to_target < self._success_distance:
            self._metric = 1.0


class OracleSPL(Measure):
    """SPL under oracle termination: path length is frozen at the first sample
    where the oracle would have stopped the episode."""

    cls_uuid: str = "oracle_spl"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._start_end_episode_distance: Optional[float] = None
        self._path_length_at_success: Optional[float] = None
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, OracleSuccess.cls_uuid]
        )
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._path_length_at_success = None
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        oracle_success = task.measurements.measures[
            OracleSuccess.cls_uuid
        ].get_metric()
        if oracle_success and self._path_length_at_success is None:
            self._path_length_at_success = task.path_tracker.length
        path_length = (
            self._path_length_at_success
            if self._path_length_at_success is not None
            else task.path_tracker.length
        )
        self._metric = oracle_success * _path_ratio(
            self._start_end_episode_distance, path_length
        )


def build_measurements(sim_adapter, config: ObjectNavConfig) -> Measurements:
    """Assemble the measure bundle in dependency order."""
    measures: List[Measure] = [
        DistanceToGoal(sim=sim_adapter, config=config),
        Success(sim=sim_adapter, config=config),
        ContinuousSPL(sim=sim_adapter, config=config),
    ]
    if config.include_soft_spl:
        measures.append(ContinuousSoftSPL(sim=sim_adapter, config=config))
    if config.include_oracle:
        measures.append(OracleSuccess(sim=sim_adapter, config=config))
        measures.append(OracleSPL(sim=sim_adapter, config=config))
    return Measurements(measures)


# ---------------------------------------------------------------------------
# The task layer
# ---------------------------------------------------------------------------


class ObjectNavTask:
    """Episode setup + ObjectNav metrics on top of an unmodified
    ``HabitatRobotSim``.

    Also *is* the ``task`` object habitat's measures expect: it exposes
    ``measurements``, ``is_stop_called`` and a no-op ``add_perf_timing``.

    The sim's scene is fixed at construction time (``HabitatRobotSim.reset``
    does not rebuild the robot when the scene changes), so one task/sim pair
    serves one scene; use
    :func:`continuous_demos.objectnav_dataset.episodes_by_scene` to drive the
    outer loop.
    """

    def __init__(
        self,
        robot_sim: HabitatRobotSim,
        config: Optional[ObjectNavConfig] = None,
        scene_id: Optional[str] = None,
    ):
        self.robot_sim = robot_sim
        self.config = config or ObjectNavConfig()
        self.sim_adapter = NavSimAdapter(
            robot_sim, snap_agent_position=self.config.snap_agent_position
        )
        self.path_tracker = PathTracker()
        self.measurements = build_measurements(self.sim_adapter, self.config)
        self.is_stop_called = False
        self.episode = None
        # Scene the sim was built for; used to reject mismatched episodes.
        self.scene_id = scene_id or robot_sim.cfg.sim_cfg.scene_id

    # -- habitat `task` protocol ------------------------------------------
    def add_perf_timing(self, *args: Any, **kwargs: Any) -> None:
        """No-op; habitat's ``Measurements.update_measures`` calls this."""

    # -- episode lifecycle -------------------------------------------------
    def reset(self, episode, snap_to_navmesh: bool = True) -> Dict[str, float]:
        """Spawn the robot for ``episode`` and reset all measures.

        Returns the metrics at t=0 (notably ``distance_to_goal``, which should
        match the episode's recorded ``info["geodesic_distance"]``).
        """
        self._check_scene(episode)
        if not getattr(episode, "goals", None):
            raise ValueError(
                f"Episode {episode.episode_id} has no goals; ObjectNav metrics "
                "are undefined. Check that goals_by_category was joined."
            )

        # Stale MultiGoalShortestPath caches carry the previous run's goal set.
        episode._shortest_path_cache = None

        # reset() mutates the translation in place (vertical offset), so copy.
        self.robot_sim.reset(
            robot_translation=np.array(episode.start_position, dtype=float),
            robot_rotation=list(episode.start_rotation),
            snap_to_navmesh=snap_to_navmesh,
        )

        self.episode = episode
        self.is_stop_called = False
        self.path_tracker.reset(self.robot_sim.get_2d_pose())
        self.measurements.reset_measures(episode=episode, task=self)
        return self.get_metrics()

    def _check_scene(self, episode) -> None:
        if os.path.basename(episode.scene_id) != os.path.basename(self.scene_id):
            raise ValueError(
                f"Episode {episode.episode_id} is in scene '{episode.scene_id}' "
                f"but this task's simulator was built for '{self.scene_id}'. "
                "Build one HabitatRobotSim (and task) per scene."
            )

    # -- pull-based hooks --------------------------------------------------
    def observe_pose(self, pose2d: Optional[Sequence[float]] = None) -> float:
        """Feed the path-length tracker. Cheap; call it every physics tick."""
        if pose2d is None:
            pose2d = self.robot_sim.get_2d_pose()
        return self.path_tracker.observe(pose2d)

    def call_stop(self) -> None:
        """The agent declares termination (what habitat's ``Success`` gates on)."""
        self.is_stop_called = True

    def evaluate(self) -> Dict[str, float]:
        """Update every measure and return the metrics. Runs navmesh queries."""
        if self.episode is None:
            raise RuntimeError("evaluate() called before reset().")
        self.measurements.update_measures(episode=self.episode, task=self)
        return self.get_metrics()

    def get_metrics(self) -> Dict[str, float]:
        """Last computed metrics; does not recompute anything."""
        metrics = dict(self.measurements.get_metrics())
        metrics["path_length"] = self.path_tracker.length
        return metrics

    def distance_to_goal(self) -> float:
        """Current geodesic distance to the goal set (single navmesh query)."""
        measure = self.measurements.measures[DistanceToGoal.cls_uuid]
        measure.update_metric(episode=self.episode, task=self)
        return float(measure.get_metric())

    # -- offline scoring ---------------------------------------------------
    def evaluate_trajectory(
        self,
        episode,
        poses: Sequence[Sequence[float]],
        height: Optional[float] = None,
        stop_index: Optional[int] = None,
        evaluate_every: int = 1,
    ) -> Dict[str, float]:
        """Score an already-recorded planar trajectory without stepping physics.

        ``poses`` is a ``pose2d`` sequence as stored in
        ``data/metadata/<scene>.jsonl``. ``stop_index`` is the sample at which
        the agent called stop (default: the last one); ``evaluate_every``
        controls how often the (expensive) measures are updated, while path
        length always uses every sample.
        """
        poses = np.asarray(poses, dtype=float)
        if len(poses) == 0:
            raise ValueError("evaluate_trajectory() needs at least one pose.")
        if height is None:
            height = float(episode.start_position[1])
        if stop_index is None:
            stop_index = len(poses) - 1

        episode._shortest_path_cache = None
        adapter = StaticPoseNavAdapter(
            self.robot_sim.sim.pathfinder,
            height,
            snap_agent_position=self.config.snap_agent_position,
        )
        scorer = _TrajectoryScorer(adapter, self.config)
        adapter.set_pose(poses[0])
        scorer.path_tracker.reset(poses[0])
        scorer.measurements.reset_measures(episode=episode, task=scorer)

        for index, pose in enumerate(poses[1:], start=1):
            adapter.set_pose(pose)
            scorer.path_tracker.observe(pose)
            if index >= stop_index:
                scorer.is_stop_called = True
            if index % evaluate_every == 0 or index == len(poses) - 1:
                scorer.measurements.update_measures(episode=episode, task=scorer)

        metrics = dict(scorer.measurements.get_metrics())
        metrics["path_length"] = scorer.path_tracker.length
        return metrics


class _TrajectoryScorer:
    """Minimal ``task`` protocol carrier for :meth:`ObjectNavTask.evaluate_trajectory`."""

    def __init__(self, sim_adapter, config: ObjectNavConfig):
        self.path_tracker = PathTracker()
        self.measurements = build_measurements(sim_adapter, config)
        self.is_stop_called = False

    def add_perf_timing(self, *args: Any, **kwargs: Any) -> None:
        pass
