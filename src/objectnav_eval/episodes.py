"""Where episodes come from.

An :class:`EpisodeSource` turns some on-disk representation into a list of
:class:`EvalEpisode`. Everything downstream -- screening, the runner, sharding,
reporting -- only ever sees :class:`EvalEpisode`, so supporting a new dataset
means writing one source and registering it, not touching the harness.

:class:`ObjectNavFileSource` covers the stock ObjectNav-v1 layout (HM3D, MP3D,
habitat-web replay shards), which is a split directory whose ``content/``
subdirectory holds one ``.json.gz`` per scene, or a single ``.json.gz``.

Two properties of that layout are easy to get wrong and are handled here:

* the top-level ``<split>.json.gz`` of a split directory usually holds only the
  category tables and **zero episodes** (this is exactly how ``val_mini`` is
  laid out), so the episodes have to be read from ``content/``;
* **MP3D reuses ``episode_id`` within a scene** (``x8F5xyUWy9e`` has two
  episodes numbered 3, for different goal categories), so ``scene:id`` is not a
  key. :attr:`EvalEpisode.uid` appends an occurrence suffix to make it one --
  without it two episodes share a video filename and a shard-membership test,
  and one silently overwrites the other.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# The unit of evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalEpisode:
    """One evaluable episode, decoupled from the file format it came from.

    ``episode`` is the underlying habitat episode object. It is carried through
    because :class:`continuous_demos.objectnav.ObjectNavTask` consumes it
    directly (it needs ``goals``/``view_points`` as habitat objects and stashes
    a ``_shortest_path_cache`` on it); everything else in this package uses the
    flattened fields above it, which any source can populate.
    """

    uid: str
    episode_id: str
    scene_id: str
    scene_name: str
    object_category: Optional[str]
    start_position: np.ndarray
    start_rotation: np.ndarray
    view_points: np.ndarray  # (N, 3), habitat world frame
    episode: Any = None
    scene_dataset_config: Optional[str] = None
    dataset_geodesic: Optional[float] = None  # episode.info["geodesic_distance"]
    shard: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def identity(self) -> Dict[str, Any]:
        """The fields every result record starts with."""
        return {
            "uid": self.uid,
            "episode_id": self.episode_id,
            "scene": self.scene_name,
            "category": self.object_category,
        }


def scene_name_of(scene_id: str) -> str:
    """``hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb`` -> ``TEEsavR23oF``."""
    return Path(scene_id).stem.replace(".basis", "")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class EpisodeSource:
    """Interface: produce the episodes of one dataset.

    Subclass and register with :func:`register_source` to add a dataset. The
    contract is exactly one method; everything a source must guarantee is on
    :class:`EvalEpisode` (notably: ``uid`` is unique across the whole return
    value, and ``view_points`` is an ``(N, 3)`` float array in the habitat world
    frame, possibly empty).
    """

    name: str = "abstract"

    def load(self) -> List[EvalEpisode]:  # pragma: no cover - interface
        raise NotImplementedError


_SOURCES: Dict[str, Callable[..., EpisodeSource]] = {}


def register_source(name: str, factory: Callable[..., EpisodeSource]) -> None:
    _SOURCES[name] = factory


def build_episode_source(kind: str, **kwargs: Any) -> EpisodeSource:
    if kind not in _SOURCES:
        raise KeyError(
            f"unknown episode source '{kind}'; registered: {sorted(_SOURCES)}"
        )
    return _SOURCES[kind](**kwargs)


def available_sources() -> List[str]:
    return sorted(_SOURCES)


def shard_paths(episodes_path: Path) -> List[Path]:
    """Every episode file under ``episodes_path``.

    Accepts a single ``.json.gz`` or a split directory. In the standard split
    layout the episodes live in ``content/*.json.gz`` and the top-level
    ``<split>.json.gz`` holds only the category tables, so ``content/`` wins
    when it exists.
    """
    episodes_path = Path(episodes_path)
    if episodes_path.is_file():
        return [episodes_path]
    content = episodes_path / "content"
    shards = sorted(content.glob("*.json.gz")) if content.is_dir() else []
    if not shards:
        shards = sorted(episodes_path.glob("*.json.gz"))
    if not shards:
        raise FileNotFoundError(f"no episode .json.gz found under {episodes_path}")
    return shards


class ObjectNavFileSource(EpisodeSource):
    """Stock ObjectNav-v1 ``.json.gz`` shards (HM3D / MP3D / habitat-web).

    Episodes are parsed by :func:`continuous_demos.objectnav_dataset
    .load_objectnav_episodes`, which prefers habitat-lab's own dataset class
    (real ``ObjectGoal``/``ObjectViewLocation`` objects, which the task layer's
    measures consume) and falls back to a handcrafted parse when habitat-lab is
    missing or a shard trips its parser.
    """

    name = "objectnav"

    def __init__(self, path: str, allow_fallback: bool = True):
        self.path = Path(path)
        self.allow_fallback = allow_fallback

    def load(self) -> List[EvalEpisode]:
        from continuous_demos.objectnav_dataset import load_objectnav_episodes

        out: List[EvalEpisode] = []
        seen: Dict[str, int] = defaultdict(int)
        for shard in shard_paths(self.path):
            for episode in load_objectnav_episodes(
                str(shard), allow_fallback=self.allow_fallback
            ):
                out.append(self._convert(episode, shard, seen))
        return out

    @staticmethod
    def _convert(episode: Any, shard: Path, seen: Dict[str, int]) -> EvalEpisode:
        scene = scene_name_of(episode.scene_id)
        base = f"{scene}:{episode.episode_id}"
        seen[base] += 1
        uid = base if seen[base] == 1 else f"{base}#{seen[base]}"

        view_points = np.asarray(
            [
                vp.agent_state.position
                for goal in (episode.goals or [])
                for vp in (getattr(goal, "view_points", None) or [])
            ],
            dtype=np.float32,
        ).reshape(-1, 3)

        info = getattr(episode, "info", None) or {}
        extra = getattr(episode, "extra", None) or {}
        return EvalEpisode(
            uid=uid,
            episode_id=str(episode.episode_id),
            scene_id=episode.scene_id,
            scene_name=scene,
            object_category=getattr(episode, "object_category", None),
            start_position=np.asarray(episode.start_position, dtype=np.float64),
            start_rotation=np.asarray(episode.start_rotation, dtype=np.float64),
            view_points=view_points,
            episode=episode,
            scene_dataset_config=(
                getattr(episode, "scene_dataset_config", None)
                or extra.get("scene_dataset_config")
            ),
            dataset_geodesic=info.get("geodesic_distance"),
            shard=shard.name,
        )


register_source("objectnav", ObjectNavFileSource)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFilter:
    """Declarative episode selection, reproducible from its fields alone."""

    scenes: Optional[Sequence[str]] = None
    categories: Optional[Sequence[str]] = None
    episode_ids: Optional[Sequence[str]] = None
    max_scenes: int = 0
    sample_per_scene: int = 0
    num_episodes: int = 0
    seed: int = 0

    def apply(self, episodes: Sequence[EvalEpisode]) -> List[EvalEpisode]:
        eps = list(episodes)
        if self.scenes:
            want = {s.strip() for s in self.scenes if s and s.strip()}
            eps = [e for e in eps if e.scene_name in want]
        if self.categories:
            want = {c.strip() for c in self.categories if c and c.strip()}
            eps = [e for e in eps if e.object_category in want]
        if self.episode_ids:
            want = {i.strip() for i in self.episode_ids if i and i.strip()}
            eps = [e for e in eps if e.episode_id in want or e.uid in want]

        # Stable, format-independent ordering. Sharding, uids and the JSONL are
        # all keyed off it, so runs of the same flags name the same episodes.
        eps.sort(key=lambda e: (e.scene_id, e.episode_id))

        # Subsetting for splits too big to run whole (MP3D val is 2195 episodes
        # over 11 scenes). Sample *per scene*: the episode files are grouped by
        # scene, so a head slice would evaluate one scene and call it a split.
        if self.max_scenes or self.sample_per_scene:
            eps = self._subsample(eps)
        return eps[: self.num_episodes] if self.num_episodes else eps

    def _subsample(self, eps: List[EvalEpisode]) -> List[EvalEpisode]:
        rng = np.random.default_rng(self.seed)
        by_scene: Dict[str, List[EvalEpisode]] = defaultdict(list)
        for e in eps:
            by_scene[e.scene_id].append(e)
        scenes = sorted(by_scene)
        if self.max_scenes and len(scenes) > self.max_scenes:
            scenes = sorted(
                rng.choice(scenes, size=self.max_scenes, replace=False).tolist()
            )
        picked: List[EvalEpisode] = []
        for scene in scenes:
            group = by_scene[scene]
            if self.sample_per_scene and len(group) > self.sample_per_scene:
                idx = rng.choice(len(group), size=self.sample_per_scene, replace=False)
                group = [group[i] for i in sorted(idx)]
            picked.extend(group)
        return picked


def group_by_scene(
    episodes: Iterable[EvalEpisode],
) -> "Dict[str, List[EvalEpisode]]":
    """Group episodes by ``scene_id``, preserving order.

    Scene load dominates a short episode, so every execution path here iterates
    scene by scene and builds one simulator per scene.
    """
    grouped: Dict[str, List[EvalEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.scene_id].append(episode)
    return dict(grouped)


def select_by_uid(
    episodes: Sequence[EvalEpisode], uids: Iterable[str]
) -> List[EvalEpisode]:
    """Resolve a shard's uid list back to episodes, in the canonical order.

    Raises if any uid is unresolvable: that means the episode list is not stable
    between the launcher and the worker, and a silently short shard would be
    reported as a completed one.
    """
    wanted = set(uids)
    found = [e for e in episodes if e.uid in wanted]
    if len(found) != len(wanted):
        missing = sorted(wanted - {e.uid for e in found})
        raise RuntimeError(
            f"resolved {len(found)} of {len(wanted)} episode uids; the episode "
            f"list is not stable. Missing: {missing[:10]}"
        )
    return found


def env_scene_root(scene_root: Optional[str]) -> Optional[str]:
    """``--scene-root`` with the usual environment fallback."""
    return scene_root or os.environ.get("HABITAT_SCENE_ROOT")
