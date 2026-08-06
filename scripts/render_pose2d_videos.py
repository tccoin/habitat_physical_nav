"""Stitch the frames of a pose2d dataset back into videos, for debugging.

Consumes the output of ``generate_pose2d_dataset.py`` -- i.e. an ``--output-dir``
containing ``metadata/<scene>.jsonl`` records and an image root (default
``<output_dir>/images``) laid out as ``<scene>/<episode>/frame_*.png``.

Episodes can be selected by ``--episode-index``, by ``--episode-id`` (both accept
multiple values), or with ``--all`` to render everything present in the JSONLs.

Videos are written to::

    <output_dir>/videos/<scene>/<episode_index>_<episode_id>.mp4

(``<scene>`` is the short scene name used by the image layout, e.g.
``17DRP5sb8fy``, not the slash-bearing raw ``scene_id``.)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np

from continuous_demos.recording_metadata import resolve_corpus_params


def scene_name(scene_id: str) -> str:
    """Short, filesystem-safe scene name, e.g. ``mp3d/17DRP5sb8fy.glb`` -> ``17DRP5sb8fy``."""
    return os.path.splitext(os.path.basename(scene_id))[0]


def sanitize(name: str) -> str:
    """Make an arbitrary id safe to use as a single path component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def load_records(metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load every episode record from ``metadata/*.jsonl``."""
    jsonl_files = sorted(metadata_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {metadata_dir}")

    records: List[Dict[str, Any]] = []
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def select_records(
    records: List[Dict[str, Any]],
    episode_indices: Optional[List[int]],
    episode_ids: Optional[List[str]],
    render_all: bool,
) -> List[Dict[str, Any]]:
    """Filter records by index / id, or return all of them."""
    if render_all:
        return records

    wanted_idx = set(episode_indices or [])
    wanted_ids = set(episode_ids or [])
    selected = [
        r for r in records
        if r.get("episode_index") in wanted_idx or r.get("episode_id") in wanted_ids
    ]

    # Warn about anything requested but not found, to make debugging failures obvious.
    found_idx = {r.get("episode_index") for r in selected}
    found_ids = {r.get("episode_id") for r in selected}
    for i in wanted_idx - found_idx:
        print(f"  ! episode_index {i} not found in metadata")
    for eid in wanted_ids - found_ids:
        print(f"  ! episode_id {eid} not found in metadata")
    return selected


def render_video(
    record: Dict[str, Any],
    image_root: Path,
    videos_root: Path,
    fps: float,
) -> Optional[Path]:
    """Write one episode's frames to a video; return the output path (or None)."""
    scene = scene_name(record["scene_id"])
    ep_id = record.get("episode_id", "unknown")
    ep_idx = record.get("episode_index", "NA")

    frames: List[np.ndarray] = []
    missing = 0
    for rel in record["rgb_paths"]:
        path = image_root / rel
        try:
            frames.append(imageio.imread(path))
        except (FileNotFoundError, OSError):
            missing += 1

    if not frames:
        print(f"  [ep {ep_idx} / {ep_id}] no readable frames -- skipping")
        return None
    if missing:
        print(f"  [ep {ep_idx} / {ep_id}] warning: {missing} frame(s) missing/unreadable")

    out_dir = videos_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ep_idx}_{sanitize(ep_id)}.mp4"
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"  [ep {ep_idx} / {ep_id}] {len(frames)} frames -> {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render debug videos from a pose2d dataset's saved frames."
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Dataset directory produced by generate_pose2d_dataset.py "
             "(contains metadata/ and images/).",
    )
    parser.add_argument(
        "--image-root", type=Path, default=None,
        help="Root the JSONL rgb_paths are relative to (default: <output_dir>/images).",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="Directory holding the *.jsonl records (default: <output_dir>/metadata).",
    )
    parser.add_argument(
        "--episode-index", type=int, nargs="+", default=None,
        help="One or more episode indices to render.",
    )
    parser.add_argument(
        "--episode-id", type=str, nargs="+", default=None,
        help="One or more episode ids to render.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Render every episode present in the JSONLs.",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Output video FPS. Defaults to the rate the corpus records having been "
             "recorded at, so playback is real-time; a hardcoded default here plays "
             "a 25 Hz corpus 20%% fast. Pass 30 for pre-schema corpora (which store "
             "no rate), or any value to deliberately speed up / slow down playback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all and not args.episode_index and not args.episode_id:
        raise SystemExit(
            "Nothing selected: pass --all, --episode-index, and/or --episode-id."
        )

    image_root = args.image_root or (args.output_dir / "images")
    metadata_dir = args.metadata_dir or (args.output_dir / "metadata")
    videos_root = args.output_dir / "videos"

    # Play back at the rate the data was recorded at, read from the corpus
    # itself (see continuous_demos.recording_metadata); --fps overrides, and
    # is required for pre-schema corpora, which record no rate.
    fps = args.fps
    if fps is None:
        fps = resolve_corpus_params(metadata_dir, flag_name="--fps").params.native_fps
        print(f"Video fps: {fps:g} (from the corpus's recording block)")

    records = load_records(metadata_dir)
    selected = select_records(
        records, args.episode_index, args.episode_id, args.all
    )
    if not selected:
        print("No matching episodes to render.")
        return

    print(f"Rendering {len(selected)} episode(s) to {videos_root} ...")
    n_ok = 0
    for record in selected:
        if render_video(record, image_root, videos_root, fps) is not None:
            n_ok += 1
    print(f"\nDone. {n_ok}/{len(selected)} video(s) written under {videos_root}.")


if __name__ == "__main__":
    main()
