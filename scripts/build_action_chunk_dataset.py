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
     -- see that function's docstring for the exact drop/stride rule.
  2. Each dict becomes one output row, with the episode's ``episode_id`` /
     ``scene_id`` / ``goal_text`` copied onto every row it produces, the image
     path resolved to an absolute path under ``--image-root``, and the
     resolved ``ChunkConfig`` fields (``native_fps``, ``obs_hz``,
     ``obs_stride_frames``, ``action_chunk_len``, ``dt_native``) repeated onto
     every row so a row is self-describing without needing the sidecar
     ``chunk_config.json`` this script also writes next to the dataset.
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

from datasets import Dataset, DatasetDict

from continuous_demos.action_chunking import ChunkConfig, build_observation_records, make_chunk_config


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
    episode: Dict[str, Any], image_root: Path, chunk_config: ChunkConfig
) -> List[Dict[str, Any]]:
    obs_records = build_observation_records(
        episode["pose2d"], episode["rgb_paths"], chunk_config
    )
    rows = []
    for rec in obs_records:
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "scene_id": episode["scene_id"],
                "goal_text": episode.get("goal_text", ""),
                "obs_index": rec["obs_index"],
                "frame_index": rec["frame_index"],
                "num_native_frames": len(episode["pose2d"]),
                "image": str(image_root / rec["image"]),
                "obs_pose": rec["obs_pose"],
                "action_chunk": rec["action_chunk"],
                "native_fps": chunk_config.native_fps,
                "obs_hz": chunk_config.obs_hz,
                "obs_stride_frames": chunk_config.obs_stride_frames,
                "action_chunk_len": chunk_config.action_chunk_len,
                "dt_native": chunk_config.dt_native,
            }
        )
    return rows


def build_all_rows(
    metadata_dir: Path, image_root: Path, chunk_config: ChunkConfig, max_length: Optional[int]
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    n_episodes = 0
    n_dropped_short = 0
    for jsonl_path in sorted(metadata_dir.glob("*.jsonl")):
        episodes = load_scene_jsonl(jsonl_path)
        for episode in episodes:
            if max_length is not None and len(episode["pose2d"]) > max_length:
                continue
            n_episodes += 1
            rows = build_rows_for_episode(episode, image_root, chunk_config)
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
    parser.add_argument("--obs-hz", type=float, default=4.0, help="Observation sampling rate.")
    parser.add_argument(
        "--action-chunk-len", type=int, default=10,
        help="Number of future (native-fps) poses per action chunk.",
    )
    parser.add_argument(
        "--native-fps", type=float, default=20.0,
        help="fps the pose2d/rgb_paths in metadata/*.jsonl were recorded at "
             "(the --fps passed to generate_pose2d_dataset.py -- not stored in the "
             "metadata, so this MUST be supplied correctly by the caller).",
    )
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

    chunk_config = make_chunk_config(
        native_fps=args.native_fps, obs_hz=args.obs_hz, action_chunk_len=args.action_chunk_len
    )
    print(
        f"ChunkConfig: native_fps={chunk_config.native_fps} obs_hz(requested)={chunk_config.obs_hz} "
        f"obs_hz(effective)={chunk_config.effective_obs_hz:.3f} "
        f"obs_stride_frames={chunk_config.obs_stride_frames} action_chunk_len={chunk_config.action_chunk_len} "
        f"chunk_duration={chunk_config.chunk_duration:.3f}s obs_interval={chunk_config.obs_interval:.3f}s "
        f"overlap_fraction={chunk_config.overlap_fraction:.3f}"
    )

    rows = build_all_rows(metadata_dir, image_root, chunk_config, args.max_length)
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
            {
                "native_fps": chunk_config.native_fps,
                "obs_hz_requested": chunk_config.obs_hz,
                "obs_hz_effective": chunk_config.effective_obs_hz,
                "obs_stride_frames": chunk_config.obs_stride_frames,
                "action_chunk_len": chunk_config.action_chunk_len,
                "dt_native": chunk_config.dt_native,
                "obs_interval": chunk_config.obs_interval,
                "chunk_duration": chunk_config.chunk_duration,
                "overlap_fraction": chunk_config.overlap_fraction,
                "image_root": str(image_root),
            },
            f,
            indent=2,
        )
    print(f"Saved dataset to {args.output_dir} ({config_path.name} alongside it).")


if __name__ == "__main__":
    main()
