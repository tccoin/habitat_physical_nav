"""ObjectNav episode loading for the habitat-web human-demo shards.

The ``.json.gz`` files this repo replays (e.g. ``examples/17DRP5sb8fy.json.gz``)
are **stock ObjectNav-v1 datasets** -- same top-level keys
(``goals_by_category``, ``episodes``, ``category_to_task_category_id``,
``category_to_mp3d_category_id``), same goal/view-point schema -- with four
extra *episode* keys that habitat-lab's ``kw_only`` attrs episode class rejects:

    reference_replay, is_thda, scene_dataset, scene_state

That single incompatibility is the whole reason ``ObjectNavDatasetV1.from_json``
raises ``TypeError: __init__() got an unexpected keyword argument 'is_thda'``.
So rather than hand-rolling a parser, this module subclasses habitat's dataset
and widens the episode type. That buys the whole ``Dataset`` abstraction --
``scene_ids``, ``get_scene_episodes``, ``filter_episodes``, ``get_splits``,
``get_episode_iterator`` -- plus goal/view-point deserialization into real
``ObjectGoal`` / ``ObjectViewLocation`` objects, which is exactly what the
habitat nav measures in :mod:`continuous_demos.objectnav` consume.

``load_objectnav_episodes`` is the compatibility seam: it prefers the habitat
path and falls back to the handcrafted parse
(:func:`continuous_demos.episode_io.load_demo_objectnav` plus the light
dataclasses below) if habitat-lab is unavailable or a future shard trips
upstream's parser. Both paths expose the *same attribute surface*, so
downstream code is written once:

    episode.episode_id / .scene_id / .start_position / .start_rotation
    episode.object_category / .info / .reference_replay
    episode.goals[i].position
    episode.goals[i].view_points[j].agent_state.position
    episode._shortest_path_cache            (settable; geodesic-query cache)
"""

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import warnings

from continuous_demos.episode_io import load_demo_objectnav


# --------------------------------------------------------------------------
# Fallback episode types (no habitat-lab required)
# --------------------------------------------------------------------------


@dataclass
class RawAgentState:
    position: List[float]
    rotation: Optional[List[float]] = None


@dataclass
class RawViewPoint:
    agent_state: RawAgentState
    iou: Optional[float] = None


@dataclass
class RawGoal:
    position: List[float]
    view_points: List[RawViewPoint] = field(default_factory=list)
    object_id: Optional[int] = None
    object_category: Optional[str] = None
    radius: Optional[float] = None


@dataclass
class RawEpisode:
    """Minimal stand-in mirroring ``ObjectNavReplayEpisode``'s attribute surface."""

    episode_id: str
    scene_id: str
    start_position: List[float]
    start_rotation: List[float]
    object_category: Optional[str] = None
    info: Optional[Dict[str, Any]] = None
    goals: List[RawGoal] = field(default_factory=list)
    reference_replay: Optional[List[Dict[str, Any]]] = None
    episode_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    # Mirrors habitat's Episode attribute; MultiGoalShortestPath cache slot.
    _shortest_path_cache: Any = None

    @property
    def goals_key(self) -> str:
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


def load_raw_episodes(path: str) -> List[RawEpisode]:
    """Handcrafted parse -- the fallback used when the habitat path is unusable."""
    data = load_demo_objectnav(path)
    episodes = []
    for index, raw in enumerate(data["episodes"]):
        goals = [
            RawGoal(
                position=g["position"],
                view_points=[
                    RawViewPoint(
                        agent_state=RawAgentState(**vp["agent_state"]),
                        iou=vp.get("iou"),
                    )
                    for vp in g.get("view_points", [])
                ],
                object_id=g.get("object_id"),
                object_category=g.get("object_category"),
                radius=g.get("radius"),
            )
            for g in raw.get("resolved_goals", [])
        ]
        episodes.append(
            RawEpisode(
                episode_id=str(raw["episode_id"]),
                scene_id=raw["scene_id"],
                start_position=raw["start_position"],
                start_rotation=raw["start_rotation"],
                object_category=raw.get("object_category"),
                info=raw.get("info"),
                goals=goals,
                reference_replay=raw.get("reference_replay"),
                episode_index=index,
            )
        )
    return episodes


# --------------------------------------------------------------------------
# habitat-lab-backed dataset (preferred path)
# --------------------------------------------------------------------------

try:
    import attr

    from habitat.core.simulator import AgentState
    from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
    from habitat.datasets.pointnav.pointnav_dataset import (
        CONTENT_SCENES_PATH_FIELD,
        DEFAULT_SCENE_PATH_PREFIX,
    )
    from habitat.tasks.nav.nav import ShortestPathPoint
    from habitat.tasks.nav.object_nav_task import (
        ObjectGoal,
        ObjectGoalNavEpisode,
        ObjectViewLocation,
    )

    HABITAT_AVAILABLE = True
    HABITAT_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - exercised only without habitat-lab
    HABITAT_AVAILABLE = False
    HABITAT_IMPORT_ERROR = exc


if HABITAT_AVAILABLE:

    @attr.s(auto_attribs=True, kw_only=True)
    class ObjectNavReplayEpisode(ObjectGoalNavEpisode):
        """``ObjectGoalNavEpisode`` widened to hold the habitat-web extras.

        ``extra`` catches any *further* unknown key, so a new shard with an
        added field degrades to "stashed in a dict" instead of a hard parse
        failure.
        """

        reference_replay: Optional[List[Dict[str, Any]]] = None
        is_thda: Optional[bool] = None
        scene_dataset: Optional[str] = None
        scene_state: Optional[Any] = None
        extra: Dict[str, Any] = attr.Factory(dict)
        # Positional index in the shard. habitat's from_json *overwrites*
        # episode_id with str(i); we keep the real id and record the index here.
        episode_index: Optional[int] = None

    class ObjectNavReplayDataset(ObjectNavDatasetV1):
        """``ObjectNavDatasetV1`` that tolerates habitat-web episodes.

        ``from_json`` below is adapted from habitat-lab 0.3.3
        (``habitat/datasets/object_nav/object_nav_dataset.py::ObjectNavDatasetV1.from_json``)
        with exactly three deltas:

        1. episodes are :class:`ObjectNavReplayEpisode`, and keys unknown to it
           are routed into ``episode.extra`` instead of raising ``TypeError``;
        2. the original ``episode_id`` is **preserved** (upstream replaces it
           with the positional index, which would break the pipeline's
           resume-by-episode-id in ``generate_pose2d_dataset.py``); the index is
           kept as ``episode_index``;
        3. the ``category_to_*_category_id`` consistency assert is skipped when
           the shard omits those maps, instead of raising ``AttributeError``.
        """

        episodes: List["ObjectNavReplayEpisode"] = []  # type: ignore[assignment]

        @staticmethod
        def _deserialize_goal(serialized_goal: Dict[str, Any]) -> "ObjectGoal":
            goal = ObjectGoal(**serialized_goal)
            for index, view in enumerate(goal.view_points):
                view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
                view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
                goal.view_points[index] = view_location
            return goal

        @staticmethod
        def _make_episode(
            serialized_episode: Dict[str, Any], index: int
        ) -> "ObjectNavReplayEpisode":
            known = {
                f.name.lstrip("_")
                for f in attr.fields(ObjectNavReplayEpisode)
                if f.init
            }
            kwargs = {k: v for k, v in serialized_episode.items() if k in known}
            extra = {
                k: v
                for k, v in serialized_episode.items()
                if k not in known and k != "extra"
            }
            kwargs.setdefault("extra", {})
            kwargs["extra"] = {**kwargs["extra"], **extra}
            # `goals` carries a not-none validator but habitat-web shards keep
            # goals in `goals_by_category`; from_json fills them in right after.
            kwargs.setdefault("goals", [])
            kwargs["episode_id"] = str(serialized_episode["episode_id"])
            kwargs["episode_index"] = index
            return ObjectNavReplayEpisode(**kwargs)

        def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
            deserialized = json.loads(json_str)
            if CONTENT_SCENES_PATH_FIELD in deserialized:
                self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

            if "category_to_task_category_id" in deserialized:
                self.category_to_task_category_id = deserialized[
                    "category_to_task_category_id"
                ]
            if "category_to_scene_annotation_category_id" in deserialized:
                self.category_to_scene_annotation_category_id = deserialized[
                    "category_to_scene_annotation_category_id"
                ]
            if "category_to_mp3d_category_id" in deserialized:
                self.category_to_scene_annotation_category_id = deserialized[
                    "category_to_mp3d_category_id"
                ]

            has_category_maps = hasattr(
                self, "category_to_task_category_id"
            ) and hasattr(self, "category_to_scene_annotation_category_id")
            if has_category_maps:
                assert set(self.category_to_task_category_id.keys()) == set(
                    self.category_to_scene_annotation_category_id.keys()
                ), "category_to_task and category_to_mp3d must have the same keys"

            if len(deserialized["episodes"]) == 0:
                return

            if "goals_by_category" not in deserialized:
                raise ValueError(
                    "Dataset has no 'goals_by_category'; habitat's per-episode "
                    "goal de-duplication path is not supported for habitat-web "
                    "shards (it re-parses episodes with the stock episode class)."
                )

            for key, goals in deserialized["goals_by_category"].items():
                self.goals_by_category[key] = [
                    self._deserialize_goal(goal) for goal in goals
                ]

            for index, serialized_episode in enumerate(deserialized["episodes"]):
                episode = self._make_episode(serialized_episode, index)

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]
                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                if episode.goals_key not in self.goals_by_category:
                    raise KeyError(
                        f"Episode {episode.episode_id} references goals key "
                        f"'{episode.goals_key}' missing from goals_by_category."
                    )
                episode.goals = self.goals_by_category[episode.goals_key]

                if episode.shortest_paths is not None:
                    for path in episode.shortest_paths:
                        for p_index, point in enumerate(path):
                            if point is None or isinstance(point, (int, str)):
                                point = {
                                    "action": point,
                                    "rotation": None,
                                    "position": None,
                                }
                            path[p_index] = ShortestPathPoint(**point)

                self.episodes.append(episode)


def load_objectnav_dataset(path: str) -> "ObjectNavReplayDataset":
    """Load a (optionally gzipped) ObjectNav-v1 / habitat-web shard.

    Scene ids are left untouched (``scenes_dir=None``) -- this repo resolves
    them through :func:`continuous_demos.episode_io.resolve_scene_path`, which
    understands its own ``mp3d_example`` layouts.
    """
    if not HABITAT_AVAILABLE:
        raise ImportError(
            "habitat-lab is required for load_objectnav_dataset()"
        ) from HABITAT_IMPORT_ERROR

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        json_str = handle.read()

    dataset = ObjectNavReplayDataset(None)
    dataset.from_json(json_str)
    return dataset


def load_objectnav_episodes(path: str, allow_fallback: bool = True) -> Sequence[Any]:
    """Load episodes, preferring the habitat dataset and degrading gracefully.

    Returns :class:`ObjectNavReplayEpisode` objects normally, or
    :class:`RawEpisode` objects if the habitat path failed and
    ``allow_fallback``. Both satisfy the attribute surface documented at the top
    of this module.
    """
    try:
        return list(load_objectnav_dataset(path).episodes)
    except Exception as exc:
        if not allow_fallback:
            raise
        warnings.warn(
            f"habitat-lab ObjectNav dataset load failed for {path} ({exc!r}); "
            "falling back to the handcrafted parser. Goals and view points are "
            "still available, but habitat Dataset helpers are not.",
            RuntimeWarning,
            stacklevel=2,
        )
        return load_raw_episodes(path)


def episodes_by_scene(episodes: Sequence[Any]) -> Dict[str, List[Any]]:
    """Group episodes by ``scene_id``, preserving order.

    The one-sim-per-scene loop every script here uses: ``HabitatRobotSim`` fixes
    its scene at construction time (``reset(scene_id=...)`` does not rebuild the
    robot), so callers build one sim per scene and iterate that scene's episodes.
    """
    grouped: Dict[str, List[Any]] = {}
    for episode in episodes:
        grouped.setdefault(episode.scene_id, []).append(episode)
    return grouped
