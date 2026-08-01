"""Which episodes this robot can even attempt, and why the others cannot.

``build_robot_sim`` recomputes the navmesh for the *robot's* footprint (radius
0.28 m, height 1.0 m). The ObjectNav episodes were generated against the navmesh
that ships with the scene, cut for a much slimmer 0.1 m point agent. The robot's
mesh is therefore strictly smaller and more fragmented, and on some episodes the
goal ends up on a different navmesh island from the start -- the robot
physically cannot fit through whatever connects them.

Those episodes are unwinnable for this embodiment. Screening them out *before*
any compute is spent is the cheap part; the part that matters is that every
screened episode carries an explicit machine-readable ``reason``, is counted in
the summary, and appears in the per-episode JSONL with ``status: "skipped"``.
They are never silently dropped and never counted as failures.

Screening is a policy-independent property of (episode, robot navmesh), so it
runs once per episode per scene load, before the policy is ever called -- which
is also what makes ``--dry-run`` free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from objectnav_eval.episodes import EvalEpisode
from objectnav_eval.navmesh import geodesic, snap_points

# Machine-readable reason codes. The human-readable message is built per
# episode (it can quote a distance); grouping in the summary uses the code.
NO_VIEW_POINTS = "no_view_points"
START_OFF_NAVMESH = "start_off_navmesh"
GOALS_OFF_NAVMESH = "goals_off_navmesh"
GOAL_UNREACHABLE = "goal_unreachable"
TRIVIAL_SUCCESS = "trivially_successful"

REASON_TEXT = {
    NO_VIEW_POINTS: "no goal view points in the episode file",
    START_OFF_NAVMESH: "start position is off the robot navmesh",
    GOALS_OFF_NAVMESH: "no goal view point lands on the robot navmesh",
    GOAL_UNREACHABLE: (
        "goal unreachable on the robot navmesh (start and goal are on different "
        "islands -- the robot does not fit through)"
    ),
    TRIVIAL_SUCCESS: "starts inside the success radius (trivially successful)",
}


@dataclass
class ScreenResult:
    """Verdict for one episode, with both navmeshes' start distances recorded.

    Keeping ``geodesic_start_dataset_m`` alongside ``geodesic_start_m`` is what
    keeps the two-navmesh gap visible instead of mysterious: an episode skipped
    as unreachable still shows the distance the dataset thought it had.
    """

    ok: bool
    code: Optional[str] = None
    reason: str = ""
    geodesic_start_m: float = float("nan")
    geodesic_start_dataset_m: float = float("nan")
    num_goal_points: int = 0
    num_goal_points_on_navmesh: int = 0

    def record(self, episode: EvalEpisode) -> Dict[str, Any]:
        """The skipped-episode record written to the results file."""
        return {
            **episode.identity(),
            "status": "skipped",
            "reason_code": self.code,
            "reason": self.reason,
            "geodesic_start_m": self.geodesic_start_m,
            "geodesic_start_dataset_m": self.geodesic_start_dataset_m,
            "dataset_geodesic_m": episode.dataset_geodesic,
            "num_goal_points": self.num_goal_points,
            "num_goal_points_on_navmesh": self.num_goal_points_on_navmesh,
        }


@dataclass
class EpisodeScreener:
    """Screen episodes against a robot navmesh (and a reference navmesh).

    ``success_distance`` is the same threshold the metrics use: an episode whose
    start is already inside it would hand the policy a success it did not
    navigate to. The dataset guarantees a minimum start distance on *its*
    navmesh, not on the robot's, so a handful of episodes land inside.
    """

    robot_pathfinder: Any
    dataset_pathfinder: Any = None
    success_distance: float = 1.0
    # Screening out free successes is separable from screening out unwinnable
    # episodes; a run that wants them scored anyway can turn just this off.
    screen_trivial: bool = True

    def screen(self, episode: EvalEpisode) -> ScreenResult:
        start = np.asarray(episode.start_position, dtype=np.float32)
        view_points = np.asarray(episode.view_points, dtype=np.float32).reshape(-1, 3)
        d_dataset = (
            geodesic(self.dataset_pathfinder, start, view_points)
            if self.dataset_pathfinder is not None and len(view_points)
            else float("nan")
        )
        base = dict(
            geodesic_start_dataset_m=d_dataset, num_goal_points=int(len(view_points))
        )

        if len(view_points) == 0:
            return self._fail(NO_VIEW_POINTS, float("nan"), **base)

        snapped_start = self.robot_pathfinder.snap_point(start)
        if np.isnan(snapped_start[0]):
            return self._fail(START_OFF_NAVMESH, float("inf"), **base)

        goals = snap_points(self.robot_pathfinder, view_points)
        base["num_goal_points_on_navmesh"] = int(len(goals))
        if len(goals) == 0:
            return self._fail(GOALS_OFF_NAVMESH, float("inf"), **base)

        d_robot = geodesic(self.robot_pathfinder, snapped_start, goals)
        if not np.isfinite(d_robot):
            return self._fail(GOAL_UNREACHABLE, d_robot, **base)

        if self.screen_trivial and d_robot <= self.success_distance:
            return self._fail(
                TRIVIAL_SUCCESS,
                d_robot,
                reason=(
                    f"starts {d_robot:.2f} m from the goal, already within the "
                    f"{self.success_distance:.1f} m success radius "
                    "(trivially successful)"
                ),
                **base,
            )

        return ScreenResult(ok=True, geodesic_start_m=float(d_robot), **base)

    @staticmethod
    def _fail(code: str, distance: float, reason: str = "", **base: Any) -> ScreenResult:
        return ScreenResult(
            ok=False,
            code=code,
            reason=reason or REASON_TEXT[code],
            geodesic_start_m=float(distance),
            **base,
        )


@dataclass
class ScreeningLog:
    """Accumulates skipped-episode records for the summary."""

    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, episode: EvalEpisode, result: ScreenResult) -> Dict[str, Any]:
        record = result.record(episode)
        self.records.append(record)
        return record

    def counts_by_reason(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record["reason_code"]] = counts.get(record["reason_code"], 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.records)
