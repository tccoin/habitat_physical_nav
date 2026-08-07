"""The on-disk shape of a raw corpus, and the two things a resumed run reads back.

Every generator in this repo writes the same layout, whatever it is generating:

    <output_dir>/
        images/<scene>/<episode>/frame_00000.<ext>   # one file per tick
        metadata/<scene>.jsonl                        # one JSON object per kept episode
        failed_episodes.log                           # TSV, one row per rejection
        corpus_params.json                            # the guard's marker (corpus_guard)

That layout is a contract between the generators and every consumer -- the arrow
builders, ``render_pose2d_videos.py``, the resume logic, and any shell one-liner someone
writes over the failure log. It used to live inside ``scripts/generate_pose2d_dataset.py``,
which meant a second generator either imported from a script (which CLAUDE.md forbids, for
good reasons) or reimplemented it and drifted. So it lives here.

Two functions carry more weight than they look like they do:

* :func:`load_done_episode_ids` **tolerates a torn final line.** A run killed mid-write
  leaves a partial JSON object; refusing to parse the shard would make the corpus
  un-resumable, and the partial episode is simply regenerated.
* :class:`FailureLog` owns the TSV format that :func:`load_failed_episode_ids` parses.
  They were previously written in one file and parsed in another, four functions apart.

Failures are appended, never truncated: a resumed run must not forget what an earlier one
rejected, or it re-attempts the same doomed episodes on every restart.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import imageio.v2 as imageio

#: Default basename of the rejection log inside an output directory.
FAILURE_LOG_NAME = "failed_episodes.log"

#: The TSV's columns, in order. Written into the header comment so the file is
#: self-describing, and consumed by :func:`load_failed_episode_ids` (column 1).
FAILURE_LOG_COLUMNS = ("episode_index", "episode_id", "scene", "reason")

#: Subdirectory names, so a consumer never spells them itself.
IMAGES_DIRNAME = "images"
METADATA_DIRNAME = "metadata"


def scene_name(scene_id: str) -> str:
    """Short, filesystem-safe scene name, e.g. ``mp3d/17DRP5sb8fy.glb`` -> ``17DRP5sb8fy``."""
    return os.path.splitext(os.path.basename(scene_id))[0]


def sanitize(name: str) -> str:
    """Make an arbitrary id safe to use as a single path component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def load_done_episode_ids(metadata_dir: Path, scene: str) -> Set[str]:
    """Episode ids already completed for ``scene``, read from its own JSONL only.

    Because the metadata is sharded per scene (``<scene>.jsonl``), resuming a run
    -- even one targeting a single scene -- never has to scan the other scenes'
    files: we open just the shard that matches the scene being generated.
    """
    path = Path(metadata_dir) / f"{scene}.jsonl"
    done: Set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
            ep = rec.get("episode_id")
            if ep is not None:
                done.add(ep)
    return done


def load_failed_episode_ids(log_file: Path) -> Set[str]:
    """Episode ids previously logged as skipped/failed (so we don't retry them)."""
    failed: Set[str] = set()
    log_file = Path(log_file)
    if not log_file.exists():
        return failed
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                failed.add(parts[1])  # episode_id column
    return failed


class FailureLog:
    """Append-only TSV of rejected episodes, and the set of ids already in it.

    Owns the format :func:`load_failed_episode_ids` parses, which is the whole point:
    the writer and the reader were previously four functions apart in one script.

    Opened in append mode, because a resumed run must not forget what an earlier one
    rejected -- otherwise every restart re-attempts the same doomed episodes. The header
    is written only when the file is new or empty. Every row is flushed immediately, so a
    run killed mid-scene still leaves a usable log.

    Use as a context manager::

        with FailureLog(output_dir / FAILURE_LOG_NAME, source=dataset_path) as log:
            if log.record(idx, ep_id, scene, "goal: 2.31m from goal (> 1.0m)"):
                n_failed += 1
    """

    def __init__(self, path: Path, source: str = "") -> None:
        self.path = Path(path)
        self.source = source
        #: Ids already recorded -- from previous runs at construction, plus this run's.
        self.known: Set[str] = load_failed_episode_ids(self.path)
        self._is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = None

    def open(self) -> "FailureLog":
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
            if self._is_new:
                self._fh.write(f"# Failed / skipped episodes for {self.source}\n")
                self._fh.write("# columns: " + "\t".join(FAILURE_LOG_COLUMNS) + "\n")
                self._fh.flush()
                self._is_new = False
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "FailureLog":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def record(self, episode_index, episode_id, scene: str, reason: str) -> bool:
        """Append one rejection. Returns False (and writes nothing) if already logged.

        ``scene`` is ``"-"`` for a rejection made before a scene was resolved. ``reason``
        should lead with a machine-readable code followed by ``": "`` and prose, so that
        ``cut -d: -f1`` over the file gives a histogram.
        """
        episode_id = str(episode_id)
        if episode_id in self.known:
            return False
        if self._fh is None:
            self.open()
        self._fh.write(f"{episode_index}\t{episode_id}\t{scene}\t{reason}\n")
        self._fh.flush()
        self.known.add(episode_id)
        return True


def save_episode_frames(
    image_root: Path,
    scene: str,
    episode_dir_name: str,
    frames: Sequence,
    image_ext: str,
) -> List[str]:
    """Write frames to ``<image_root>/<scene>/<episode>/`` and return relative paths.

    The returned paths are relative to ``image_root`` and are exactly what goes into a
    metadata record's ``rgb_paths``; the arrow builder is what makes them absolute.
    """
    rel_dir = Path(scene) / episode_dir_name
    abs_dir = Path(image_root) / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths: List[str] = []
    for i, frame in enumerate(frames):
        fname = f"frame_{i:05d}.{image_ext}"
        imageio.imwrite(abs_dir / fname, frame)
        rgb_paths.append(str(rel_dir / fname))
    return rgb_paths


def frame_filename(index: int, image_ext: str) -> str:
    """The one place the per-tick filename pattern is spelled."""
    return f"frame_{index:05d}.{image_ext}"


def append_metadata_record(fh, record: dict) -> None:
    """Write one JSONL record and flush.

    Flushed per record, deliberately: with ~24 workers sharing an output directory, a
    killed run should cost the episode in flight and nothing else.
    """
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def existing_scene_shards(metadata_dir: Path) -> List[Path]:
    """Every ``<scene>.jsonl`` under ``metadata_dir``, sorted."""
    metadata_dir = Path(metadata_dir)
    return sorted(metadata_dir.glob("*.jsonl")) if metadata_dir.exists() else []
