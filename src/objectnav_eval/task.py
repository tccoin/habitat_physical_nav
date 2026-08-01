"""The ObjectNav task, configured for this robot.

:class:`continuous_demos.objectnav.ObjectNavTask` already supplies episode spawn
and the habitat-lab measures (``success``, ``spl``, ``soft_spl``, plus the
oracle pair defined there). This module adds the two things a *robot* rollout
needs on top of it, without editing the task layer:

1. **A live query height.** ``NavSimAdapter._height()`` returns
   ``robot.translation[1]`` minus the URDF mounting lift, which is exactly the
   navmesh-level spawn height -- and is *frozen there for the whole episode*,
   because the base link never moves (the robot travels on its world prismatic
   joints). Adding ``joint_z`` makes the height track vertical travel, which
   matters as soon as an episode changes level. This is the same lesson as
   "position is not ``robot.translation``", applied to the third axis.

2. **A seam for new metrics.** ``extra_measures`` appends habitat ``Measure``
   objects after the standard bundle, so a new ObjectNav metric is a class plus
   one list entry -- no change to the task layer, no change to the runner.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from continuous_demos.objectnav import (
    NavSimAdapter,
    ObjectNavConfig,
    ObjectNavTask,
    build_measurements,
)

__all__ = ["ObjectNavConfig", "LiveHeightNavAdapter", "RobotObjectNavTask", "build_task"]


class LiveHeightNavAdapter(NavSimAdapter):
    """``NavSimAdapter`` whose reported height follows the robot's vertical joint.

    ``super()._height()`` is the navmesh-level height at spawn; ``joint_z`` is
    the prismatic vertical travel since then. Their sum is the height a geodesic
    query should be issued at. Falls back to the frozen spawn height when the
    joint has no motor (it is passive in some URDF variants).
    """

    def _height(self) -> float:
        spawn_height = super()._height()
        try:
            return spawn_height + float(self.robot_sim.get_joint_state("joint_z"))
        except (ValueError, KeyError):
            return spawn_height


class RobotObjectNavTask(ObjectNavTask):
    """``ObjectNavTask`` with the live-height adapter and pluggable measures."""

    def __init__(
        self,
        robot_sim,
        config: Optional[ObjectNavConfig] = None,
        scene_id: Optional[str] = None,
        extra_measures: Sequence[Callable[..., Any]] = (),
        live_height: bool = True,
    ):
        super().__init__(robot_sim, config, scene_id=scene_id)
        if live_height:
            # Rebuild rather than mutate: the measures hold a reference to the
            # adapter they were constructed with.
            self.sim_adapter = LiveHeightNavAdapter(
                robot_sim, snap_agent_position=self.config.snap_agent_position
            )
            self.measurements = build_measurements(self.sim_adapter, self.config)
        for factory in extra_measures:
            measure = factory(sim=self.sim_adapter, config=self.config)
            # Appended after the standard bundle, so dependency order holds.
            self.measurements.measures[measure.uuid] = measure


def build_task(
    robot_sim,
    success_distance: float = 1.0,
    distance_to: str = "VIEW_POINTS",
    scene_id: Optional[str] = None,
    scene_root: Optional[str] = None,
    extra_measures: Sequence[Callable[..., Any]] = (),
    live_height: bool = True,
) -> RobotObjectNavTask:
    """Build the task for a policy rollout.

    ``auto_switch_scene`` is off: this harness drives the outer loop scene by
    scene and builds one simulator per scene, so a mismatched episode is a
    sharding bug and should raise rather than silently pay a scene reload.
    """
    config = ObjectNavConfig(
        success_distance=success_distance,
        distance_to=distance_to,
        auto_switch_scene=False,
        scene_root=scene_root,
    )
    return RobotObjectNavTask(
        robot_sim,
        config,
        scene_id=scene_id,
        extra_measures=extra_measures,
        live_height=live_height,
    )
