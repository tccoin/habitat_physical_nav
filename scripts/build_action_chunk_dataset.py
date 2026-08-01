"""Build a lower-frequency, action-chunked SFT dataset directly from the raw
``generate_pose2d_dataset.py`` output (``<data_dir>/images`` + ``<data_dir>/metadata/*.jsonl``).

Each output row is one *observation* (a single RGB frame, sampled at
``--obs-hz`` instead of the native recording fps) paired with an *action
chunk*: ``--action-chunk-len`` future poses, each expressed relative to that
observation's own pose. See ``continuous_demos.action_chunking`` for the full
frequency/index arithmetic and the relative-pose convention -- this script is
a thin driver over that module plus arrow-table assembly and the train/val
scene split.

------------------------------------------------------------------------
Arrow table assembly -- read before changing indexing
------------------------------------------------------------------------
For each scene's ``metadata/<scene>.jsonl``, each line is one episode record
with mutually-aligned ``pose2d`` / ``rgb_paths`` (native fps, per
``generate_pose2d_dataset.py``). For each episode:

  1. ``action_chunking.build_observation_records`` slides the observation
     window over ``pose2d``/``rgb_paths`` and returns one dict per observation
     (``obs_index``, ``frame_index``, ``image``, ``obs_pose``, ``action_chunk``)
     -- see that function's docstring for the exact drop/stride rule. By
     default consecutive observations are ``obs_stride_frames`` apart, every
     time (``--obs-spacing-mode fixed``); ``--obs-spacing-mode uniform`` or
     ``poisson`` instead draw each gap from a distribution centered on that
     same mean stride, simulating variable inference latency between
     observations -- see ``action_chunking.SpacingConfig`` for the exact
     distributions and why the alignment math doesn't otherwise change.
  2. Each dict becomes one output row, with the episode's ``episode_id`` /
     ``scene_id`` / ``goal_text`` copied onto every row it produces, the image
     path resolved to an absolute path under ``--image-root``, and the
     timing/limits block from ``chunk_build.stamped_row_fields`` repeated onto
     every row so a row is self-describing without needing the sidecar
     ``chunk_config.json`` this script also writes next to the dataset. Every
     derived value in that block (``dt_native``, ``obs_interval``,
     ``effective_obs_hz``, ``chunk_duration``, ``overlap_fraction``) is
     computed from the rate the *corpus itself records* -- never from a
     default -- which is also why changing any of them means a rebuild rather
     than a relabel. Note ``obs_stride_frames`` is always the *mean* stride,
     even under random spacing -- the realized per-row gap is whatever
     ``frame_index`` deltas between consecutive rows work out to.
  3. Rows across all episodes/scenes are collected into one flat list and
     handed to ``datasets.Dataset.from_list`` -- one arrow row per
     observation, NOT one row per episode. (Contrast with the per-frame
     ``pose2d`` column in ``metadata/<scene>.jsonl``, which is one row per
     episode holding the *whole* trajectory; this script flattens that out.)
  4. The train/val split is by *scene* (same rarest-scenes-to-val heuristic as
     ``build_pose2d_dataset.py``), applied by filtering the flat row list
     before it's split into a ``DatasetDict`` -- so all observations from a
     given episode always land in the same split (an episode never straddles
     train/val, since its scene doesn't).

No conversation/``messages`` formatting is applied here (unlike
``build_pose2d_dataset.py``'s ``to_convo``) -- this dataset only carries the
raw pose/image/action-chunk columns; formatting to a model-facing "predict
this chunk" example is left to the training pipeline.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict

from continuous_demos.action_chunking import (
    ChunkConfig,
    SpacingConfig,
    build_observation_records,
)
from continuous_demos.chunk_build import (
    add_chunking_args,
    describe_build,
    resolve_build_config,
    sidecar_dict,
    stamped_row_fields,
)
from continuous_demos.recording_metadata import ResolvedRecording, check_record_consistency


def load_scene_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_rows_for_episode(
    episode: Dict[str, Any],
    image_root: Path,
    chunk_config: ChunkConfig,
    spacing_config: SpacingConfig,
    rng: np.random.Generator,
    stamped: Dict[str, Any],
) -> List[Dict[str, Any]]:
    obs_records = build_observation_records(
        episode["pose2d"], episode["rgb_paths"], chunk_config, spacing_config, rng
    )
    rows = []
    for rec in obs_records:
        row = {
            "episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "goal_text": episode.get("goal_text", ""),
            "obs_index": rec["obs_index"],
            "frame_index": rec["frame_index"],
            "num_native_frames": len(episode["pose2d"]),
            "image": str(image_root / rec["image"]),
            "obs_pose": rec["obs_pose"],
            "action_chunk": rec["action_chunk"],
        }
        # Timing/limits block, in-band on every row: derived from the corpus's
        # own recorded rate, never from a default (see chunk_build).
        row.update(stamped)
        rows.append(row)
    return rows


def build_all_rows(
    metadata_dir: Path,
    image_root: Path,
    resolved: ResolvedRecording,
    chunk_config: ChunkConfig,
    spacing_config: SpacingConfig,
    max_length: Optional[int],
) -> List[Dict[str, Any]]:
    # One rng shared across every episode in the run (not reseeded per
    # episode) so the draws for episode 2 aren't a repeat of episode 1's --
    # still fully reproducible given --obs-spacing-seed plus this traversal
    # order (sorted scene files, then dataset order within each file).
    rng = np.random.default_rng(spacing_config.seed)
    stamped = stamped_row_fields(resolved, chunk_config, spacing_config)

    all_rows: List[Dict[str, Any]] = []
    n_episodes = 0
    n_dropped_short = 0
    for jsonl_path in sorted(metadata_dir.glob("*.jsonl")):
        episodes = load_scene_jsonl(jsonl_path)
        for i, episode in enumerate(episodes):
            # Catches a shard that was appended to across a parameter change:
            # resolve_corpus_params only ever saw its first record.
            check_record_consistency(episode, resolved, f"{jsonl_path.name} record {i}")
            if max_length is not None and len(episode["pose2d"]) > max_length:
                continue
            n_episodes += 1
            rows = build_rows_for_episode(
                episode, image_root, chunk_config, spacing_config, rng, stamped
            )
            if not rows:
                n_dropped_short += 1
                continue
            all_rows.extend(rows)
        print(f"  {jsonl_path.name}: {len(episodes)} episode(s)")

    print(
        f"Total: {n_episodes} episode(s) considered, {n_dropped_short} too short for "
        f"even one chunk, {len(all_rows)} observation row(s) produced."
    )
    return all_rows


def split_rows_by_scene(rows: List[Dict[str, Any]], val_scene_count: int):
    scene_counts = Counter(r["scene_id"] for r in rows)
    sorted_scenes = sorted(scene_counts.items(), key=lambda item: item[1])
    val_scenes = {scene for scene, _count in sorted_scenes[:val_scene_count]}
    print(f"Val scenes ({len(val_scenes)}): {val_scenes}")

    train_rows = [r for r in rows if r["scene_id"] not in val_scenes]
    val_rows = [r for r in rows if r["scene_id"] in val_scenes]
    return train_rows, val_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lower-frequency, action-chunked SFT dataset from "
        "generate_pose2d_dataset.py output."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Output dir of generate_pose2d_dataset.py (contains images/ and metadata/).",
    )
    parser.add_argument(
        "--image-root", type=Path, default=None,
        help="Root the rgb_paths in metadata/*.jsonl are relative to. "
             "Defaults to <data-dir>/images.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Where to save the arrow DatasetDict."
    )
    add_chunking_args(parser)
    parser.add_argument(
        "--val-scene-count", type=int, default=5,
        help="Number of (rarest) scenes held out for validation.",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Skip episodes whose native trajectory is longer than this many frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_root = args.image_root if args.image_root is not None else args.data_dir / "images"
    metadata_dir = args.data_dir / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"No metadata/ dir under {args.data_dir}")

    resolved, chunk_config, spacing_config = resolve_build_config(args, metadata_dir)
    print(describe_build(resolved, chunk_config, spacing_config))

    rows = build_all_rows(
        metadata_dir, image_root, resolved, chunk_config, spacing_config, args.max_length
    )
    if not rows:
        raise ValueError("No observation rows produced -- check --data-dir / --max-length.")

    train_rows, val_rows = split_rows_by_scene(rows, args.val_scene_count)
    print(f"Train rows: {len(train_rows)}  Val rows: {len(val_rows)}")
    if not train_rows:
        raise ValueError("Train split is empty -- lower --val-scene-count.")

    # An empty split's arrow table has no inferable schema and fails to
    # reload via datasets.load_from_disk, so only include "validation" when
    # there's actually something in it (e.g. --val-scene-count 0, or a
    # single-scene dataset).
    splits = {"train": Dataset.from_list(train_rows)}
    if val_rows:
        splits["validation"] = Dataset.from_list(val_rows)
    else:
        print("Val rows empty -- omitting 'validation' split from the saved DatasetDict.")
    dataset_dict = DatasetDict(splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)

    config_path = args.output_dir / "chunk_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            sidecar_dict(
                resolved,
                chunk_config,
                spacing_config,
                image_root,
                row_granularity="observation",
                num_rows={"train": len(train_rows), "validation": len(val_rows)},
            ),
            f,
            indent=2,
        )
    print(f"Saved dataset to {args.output_dir} ({config_path.name} alongside it).")


if __name__ == "__main__":
    main()
