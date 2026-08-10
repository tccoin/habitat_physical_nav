"""Where the next point goal comes from.

A PointNav rollout is a chain: drive to a goal, and when you get there be told another
one. *Who* decides the next one is the one thing that differs between watching an agent
run free, replaying a scripted route, and driving it from a GUI -- and none of those
differences belong in the rollout loop. So they are all one interface:

    goal = source.next_goal(robot_sim, current_pose)   # [X, Y, theta], or None to end

``None`` means "no more goals", which is how a scripted list ends and how a sampler
reports that the scene has run out of acceptable points. The loop treats it as the end of
the episode; it never invents a goal of its own.

Three implementations ship, and between them they cover what the seam is for:

* :class:`NavmeshGoalSource` -- sample on the robot's own navmesh, exactly as the
  demonstration generator does. This is the free-running mode.
* :class:`ScriptedGoalSource` -- a fixed list. Reproducible, and the only way to compare
  two policies on the *same* route.
* :class:`QueueGoalSource` -- whatever an outside caller has pushed. This is the GUI seam.

------------------------------------------------------------------------
What a GUI would have to implement
------------------------------------------------------------------------
Exactly one method, and it may block:

    class MyGuiGoals(GoalSource):
        def next_goal(self, robot_sim, pose) -> Optional[Sequence[float]]:
            '''Return [X, Y, theta] in the PLANAR control frame (X = habitat_x,
            Y = -habitat_z, theta CCW from +X), or None to end the episode.'''

The machine this was written on is headless, so no GUI is built here -- but the contract
it would have to meet is worth stating precisely, because two parts of it are easy to get
wrong from a click on a map:

1. **The frame is the planar control frame, not habitat world and not pixels.** A click
   on the top-down map is `(row, col)`; ``continuous_demos.topdown.pixel_to_world`` is the
   inverse that produces `(X, Y)`. Getting the sign of Y wrong yields a goal that is a
   perfect mirror image of the intended one, and the rollout looks like a policy that
   turns the wrong way rather than like a bad goal.
2. **theta is the goal's heading, and it is not ignored.** The corpus sets it to the
   planned path's final tangent (`docs/POINTNAV_GENERATION.md`), and it is injected as the
   third component of a value row. A constant -- 0.0 for every goal -- survives
   ``relative_se2`` as ``wrap(0 - theta_anchor)``, which *is* the scene's compass at the
   window origin: precisely the leak the frame conversion exists to remove. Prefer
   :func:`heading_toward`, or the tangent of a planned path.

:class:`QueueGoalSource` is the adapter that makes that a thread of the caller's choosing:
push goals from anywhere, and the rollout picks them up. It is deliberately non-blocking
by default (``timeout_s=0`` returns None immediately if nothing is queued, ending the
episode) because a rollout that hangs forever on an empty queue in a batch job is worse
than one that stops.
"""

from __future__ import annotations

import math
import queue
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from continuous_demos import pointnav as pn
from continuous_demos.pointnav_episode import island_table, sample_segment
from continuous_demos.robot_pose import robot_navmesh_position


def heading_toward(origin: Sequence[float], target: Sequence[float]) -> float:
    """Planar heading from ``origin`` to ``target``, radians CCW from +X.

    The cheap stand-in for a planned path's final tangent when there is no path -- a goal
    a caller simply named. Better than a constant for the reason in this module's header.
    """
    return float(math.atan2(float(target[1]) - float(origin[1]),
                            float(target[0]) - float(origin[0])))


def planar_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Planar distance between two ``[X, Y, ...]`` poses. Position only, never heading.

    "Reached" is a question about where the robot is, not which way it is facing: the
    demonstrations arrive at a goal on the path's final tangent, but a policy that arrives
    facing elsewhere has still arrived. Gating on heading too would end episodes the
    corpus itself would call successful.
    """
    return float(np.linalg.norm(np.asarray(a, dtype=float)[:2]
                                - np.asarray(b, dtype=float)[:2]))


class GoalSource:
    """Interface: produce the next point goal, or ``None`` to end the episode.

    Stateful and single-episode. :meth:`reset` is called once before the first goal.
    """

    name: str = "abstract"

    def reset(self, robot_sim: Any = None) -> None:
        """Start a fresh episode. Default: nothing to forget."""

    def next_goal(self, robot_sim: Any,
                  pose: Sequence[float]) -> Optional[Sequence[float]]:  # pragma: no cover
        """``[X, Y, theta]`` in the planar control frame, or ``None`` to stop."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class ScriptedGoalSource(GoalSource):
    """A fixed list of goals, in order. Reproducible by construction.

    The control for every other source: two policies compared on a sampled route are
    compared on two different routes unless the sampler's seed *and* the trajectory agree,
    and the trajectory is what decides where the next sample is drawn from. A script
    removes that entirely.

    Goals may be ``[X, Y]`` or ``[X, Y, theta]``. A missing heading is filled with the
    bearing from wherever the robot is when the goal is issued, which is a real heading
    rather than the constant this module's header warns about.
    """

    name = "scripted"

    def __init__(self, goals: Sequence[Sequence[float]], loop: bool = False):
        self.goals = [list(map(float, g)) for g in goals]
        for goal in self.goals:
            if len(goal) not in (2, 3):
                raise ValueError(
                    f"a scripted goal is [X, Y] or [X, Y, theta], got {len(goal)} "
                    f"number(s): {goal}"
                )
        self.loop = bool(loop)
        self.index = 0

    def reset(self, robot_sim: Any = None) -> None:
        self.index = 0

    def next_goal(self, robot_sim: Any, pose: Sequence[float]) -> Optional[List[float]]:
        if self.index >= len(self.goals):
            if not (self.loop and self.goals):
                return None
            self.index = 0
        goal = list(self.goals[self.index])
        self.index += 1
        if len(goal) == 2:
            goal.append(heading_toward(pose, goal))
        return goal

    def describe(self) -> str:
        return f"scripted ({len(self.goals)} goal(s){', looping' if self.loop else ''})"


class QueueGoalSource(GoalSource):
    """Goals pushed in from outside. **This is the seam a GUI drives.**

    ``push(goal)`` is thread-safe (it is a :class:`queue.Queue`), so a UI thread, a socket
    reader or a stdin reader can feed the rollout while it runs. Nothing here knows which.

    ``timeout_s`` is the whole policy decision this class makes:

    * ``0`` (default) -- take a goal if one is waiting, otherwise end the episode. Right
      for a batch job, where a blocked rollout is a hung worker.
    * ``> 0``          -- wait that long for one, then end. Right for a GUI, where the
      human needs a moment to click.
    * ``None``         -- wait forever. Only sane when a human is definitely there.

    ``fallback`` is what makes it composable rather than exclusive: an empty queue can
    hand off to a sampler, so an interactive session keeps moving when nobody is watching
    and takes direction the moment somebody is.
    """

    name = "queue"

    def __init__(self, timeout_s: Optional[float] = 0.0,
                 fallback: Optional[GoalSource] = None):
        self.queue: "queue.Queue[Optional[Sequence[float]]]" = queue.Queue()
        self.timeout_s = timeout_s
        self.fallback = fallback

    def push(self, goal: Optional[Sequence[float]]) -> None:
        """Offer a goal (or ``None`` to end the episode). Safe from any thread."""
        self.queue.put(None if goal is None else [float(v) for v in goal])

    def reset(self, robot_sim: Any = None) -> None:
        if self.fallback is not None:
            self.fallback.reset(robot_sim)

    def _take(self) -> Optional[Sequence[float]]:
        try:
            if self.timeout_s is None:
                return self.queue.get()
            if self.timeout_s <= 0:
                return self.queue.get_nowait()
            return self.queue.get(timeout=float(self.timeout_s))
        except queue.Empty:
            return None

    def next_goal(self, robot_sim: Any, pose: Sequence[float]) -> Optional[List[float]]:
        goal = self._take()
        if goal is None:
            if self.fallback is None:
                return None
            return self.fallback.next_goal(robot_sim, pose)
        goal = list(goal)
        if len(goal) == 2:
            goal.append(heading_toward(pose, goal))
        if len(goal) != 3:
            raise ValueError(f"a pushed goal is [X, Y] or [X, Y, theta], got {goal}")
        return goal

    def describe(self) -> str:
        wait = "blocking" if self.timeout_s is None else f"timeout {self.timeout_s}s"
        tail = f" -> {self.fallback.describe()}" if self.fallback is not None else ""
        return f"queue ({wait}){tail}"


class NavmeshGoalSource(GoalSource):
    """Sample the next goal on the navmesh, under the corpus's own screening rules.

    Every decision here is imported rather than restated: :func:`continuous_demos.pointnav
    .choose_island` picks the island (area-weighted, seeded),
    :func:`continuous_demos.pointnav_episode.sample_segment` draws and screens candidates
    against :class:`~continuous_demos.pointnav.PointNavSampleConfig`, and
    :func:`continuous_demos.pointnav.screen_segment` is what accepts one. That is what
    makes a goal a rollout is asked to reach the same *kind* of goal the demonstrations
    were cut on -- a sampler with its own idea of "too close" would be evaluating the
    policy on a distribution it was never trained for, and the number would still look
    fine.

    The heading is the planned path's final tangent, from ``sample_segment``, for the
    reason in this module's header: a constant heading is the scene's compass in
    disguise.

    Seeded per source, not per goal: the draw depends on where the robot *is*, which
    depends on what the policy did, so a seed fixes the sequence of random draws and not
    the sequence of goals. Two policies therefore see different routes, which is why
    :class:`ScriptedGoalSource` exists.
    """

    name = "navmesh"

    def __init__(self, config: Optional[pn.PointNavSampleConfig] = None, seed: int = 0,
                 max_goals: int = 0, island: Optional[int] = None):
        self.config = config or pn.PointNavSampleConfig()
        self.seed = int(seed)
        self.max_goals = int(max_goals)
        self.island = island
        self.rng = np.random.default_rng(self.seed)
        self.issued = 0
        self.rejections: dict = {}
        #: The last accepted candidate's planned path, planar `(M, 2)`. The debug
        #: overlays draw it; nothing else reads it.
        self.planned_path: Optional[np.ndarray] = None
        self.planned_path_3d: Optional[np.ndarray] = None
        self.geodesic_m: float = float("nan")

    def reset(self, robot_sim: Any = None) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.issued = 0
        self.rejections = {}
        self.planned_path = None
        self.planned_path_3d = None
        self.geodesic_m = float("nan")
        if robot_sim is not None:
            # The pathfinder's own RNG feeds `get_random_navigable_point`, and it is
            # separate from ours. Seeding both is what makes a run reproducible; seeding
            # only ours leaves the actual draw to habitat's global state.
            robot_sim.sim.pathfinder.seed(self.seed)
            self.island = None if self.island is None else int(self.island)

    def _resolve_island(self, pathfinder, here) -> Optional[int]:
        """The island to sample on: the robot's own if it is eligible, else a draw.

        Preferring the robot's own island is not an optimisation -- a goal on another
        island is unreachable by construction, and ``screen_segment`` would reject every
        candidate until the retry budget ran out.
        """
        if self.island is not None:
            return self.island
        current = int(pathfinder.get_island(np.asarray(here, dtype=np.float32)))
        islands = island_table(pathfinder)
        eligible = {
            index for index, area, radius in islands
            if area >= self.config.min_island_area_m2
            and radius >= self.config.min_island_radius_m
        }
        if current in eligible:
            return current
        return pn.choose_island(self.rng, islands, self.config)

    def next_goal(self, robot_sim: Any, pose: Sequence[float]) -> Optional[List[float]]:
        if self.max_goals and self.issued >= self.max_goals:
            return None
        pathfinder = robot_sim.sim.pathfinder
        # The query point is at NAVMESH height, not the robot's: the URDF lift is ~0.25 m
        # and resolves to the storey above in a stairwell. See `robot_pose.py`.
        here = robot_navmesh_position(robot_sim)
        island = self._resolve_island(pathfinder, here)
        if island is None:
            return None
        segment, rejections = sample_segment(
            pathfinder, self.rng, self.config,
            start_3d=here, theta_now=float(pose[2]), island=island,
        )
        for code, count in rejections.items():
            self.rejections[code] = self.rejections.get(code, 0) + count
        if segment is None:
            return None
        self.issued += 1
        self.planned_path = np.asarray(segment.path_xy, dtype=float)
        self.planned_path_3d = np.asarray(segment.path_3d, dtype=float)
        self.geodesic_m = float(segment.geodesic_m)
        return list(segment.goal2d)

    def describe(self) -> str:
        return (
            f"navmesh sampler (seed {self.seed}, geodesic "
            f"{self.config.min_geodesic_m}-{self.config.max_geodesic_m} m"
            + (f", max {self.max_goals} goals" if self.max_goals else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Same shape as `episodes.register_source` / `policies.register_policy`: a new way of
# choosing goals is an entry here, not a branch in the rollout loop.

_SOURCES: dict = {}


def register_goal_source(name: str, factory: Callable[..., GoalSource]) -> None:
    _SOURCES[name] = factory


def build_goal_source(kind: str, **kwargs: Any) -> GoalSource:
    if kind not in _SOURCES:
        raise KeyError(
            f"unknown goal source '{kind}'; registered: {sorted(_SOURCES)}"
        )
    return _SOURCES[kind](**kwargs)


def available_goal_sources() -> List[str]:
    return sorted(_SOURCES)


register_goal_source("navmesh", NavmeshGoalSource)
register_goal_source("scripted", ScriptedGoalSource)
register_goal_source("queue", QueueGoalSource)
