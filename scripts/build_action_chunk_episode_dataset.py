"""Build an action-chunked SFT dataset from ``generate_pose2d_dataset.py`` output,
with **one row per episode** (contrast ``build_action_chunk_dataset.py``, which
emits one row per observation).

Same inputs and same per-episode chunking as ``build_action_chunk_dataset.py``
(see ``continuous_demos.action_chunking`` for the frequency/index arithmetic and
the relative-pose convention) -- the only difference is table shape. Here each
episode's stream of observations is gathered into list-valued columns on a
single row, so an episode never splits across rows:

  * ``obs_indices`` / ``frame_indices`` -- per-observation, one entry each
  * ``images``       -- absolute path per observation
  * ``obs_poses``    -- ``[x, y, theta]`` anchor per observation
  * ``action_chunks``-- one ``action_chunk_len``-long chunk per observation

plus episode-level ``episode_id`` / ``scene_id`` / ``goal_text`` /
``num_native_frames`` / ``num_observations``, and the resolved
``ChunkConfig``/``SpacingConfig`` scalars repeated onto every row so a row is
self-describing (the sidecar ``chunk_config.json`` is written too). The
train/val split is by scene (rarest scenes to val), same heuristic as
``build_action_chunk_dataset.py`` / ``build_pose2d_dataset.py`` -- and since an
episode is now a single row, scene-level splitting trivially keeps each episode
whole. No conversation/``messages`` formatting is applied.
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
    make_chunk_config,
)


def load_scene_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_row_for_episode(
    episode: Dict[str, Any],
    chunk_config: ChunkConfig,
    spacing_config: SpacingConfig,
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    """Turn one episode into a single dataset row, or ``None`` if the episode is
    too short to anchor even one full action chunk.

    ``images`` are left as the *relative* rgb_paths straight out of the
    metadata JSONL here; ``make_absolute`` (a ``.map`` step in ``main``, as in
    the reference ``build_uniform_dataset.py``) prepends ``--image-root``."""
    obs_records = build_observation_records(
        episode["pose2d"], episode["rgb_paths"], chunk_config, spacing_config, rng
    )
    if not obs_records:
        return None

    return {
        "episode_id": episode["episode_id"],
        "scene_id": episode["scene_id"],
        "goal_text": episode.get("goal_text", ""),
        "num_native_frames": len(episode["pose2d"]),
        "num_observations": len(obs_records),
        "obs_indices": [rec["obs_index"] for rec in obs_records],
        "frame_indices": [rec["frame_index"] for rec in obs_records],
        "images": [rec["image"] for rec in obs_records],
        "obs_poses": [rec["obs_pose"] for rec in obs_records],
        "action_chunks": [rec["action_chunk"] for rec in obs_records],
        "native_fps": chunk_config.native_fps,
        "obs_hz": chunk_config.obs_hz,
        "obs_stride_frames": chunk_config.obs_stride_frames,
        "action_chunk_len": chunk_config.action_chunk_len,
        "dt_native": chunk_config.dt_native,
        "obs_spacing_mode": spacing_config.mode,
    }


def build_all_rows(
    metadata_dir: Path,
    chunk_config: ChunkConfig,
    spacing_config: SpacingConfig,
    max_length: Optional[int],
) -> List[Dict[str, Any]]:
    # One rng shared across every episode in the run (not reseeded per
    # episode) so the draws for episode 2 aren't a repeat of episode 1's --
    # still fully reproducible given --obs-spacing-seed plus this traversal
    # order (sorted scene files, then dataset order within each file).
    rng = np.random.default_rng(spacing_config.seed)

    all_rows: List[Dict[str, Any]] = []
    n_episodes = 0
    n_dropped_short = 0
    for jsonl_path in sorted(metadata_dir.glob("*.jsonl")):
        episodes = load_scene_jsonl(jsonl_path)
        for episode in episodes:
            if max_length is not None and len(episode["pose2d"]) > max_length:
                continue
            n_episodes += 1
            row = build_row_for_episode(episode, chunk_config, spacing_config, rng)
            if row is None:
                n_dropped_short += 1
                continue
            all_rows.append(row)
        print(f"  {jsonl_path.name}: {len(episodes)} episode(s)")

    print(
        f"Total: {n_episodes} episode(s) considered, {n_dropped_short} too short for "
        f"even one chunk, {len(all_rows)} episode row(s) produced."
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
        description="Build a per-episode, action-chunked SFT dataset from "
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
        "--native-fps", type=float, default=25.0,
        help="fps the pose2d/rgb_paths in metadata/*.jsonl were recorded at "
             "(the --fps passed to generate_pose2d_dataset.py -- not stored in the "
             "metadata, so this MUST be supplied correctly by the caller). "
             "Default tracks generate_pose2d_dataset.py's 25 Hz. Data generated "
             "before the on-grid fix was recorded at 30 Hz, NOT the 20 Hz this "
             "default used to claim -- pass --native-fps 30 for those shards.",
    )
    parser.add_argument(
        "--val-scene-count", type=int, default=5,
        help="Number of (rarest) scenes held out for validation.",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Skip episodes whose native trajectory is longer than this many frames.",
    )
    parser.add_argument(
        "--num-proc", type=int, default=4,
        help="Number of CPU workers for the make_absolute path-fixing map step.",
    )
    parser.add_argument(
        "--obs-spacing-mode", default="fixed", choices=["fixed", "uniform", "poisson"],
        help="How far apart consecutive observations land. 'fixed': constant "
             "obs_stride_frames apart (default, matches --obs-hz exactly). 'uniform'/"
             "'poisson': each gap is sampled from a distribution centered on that same "
             "mean stride, simulating variable inference latency -- see "
             "continuous_demos.action_chunking.SpacingConfig.",
    )
    parser.add_argument(
        "--obs-spacing-jitter", type=float, default=0.5,
        help="Only used by --obs-spacing-mode uniform: half-width of the sampling window "
             "as a fraction of the mean stride (0.5 = +/-50%%).",
    )
    parser.add_argument(
        "--obs-spacing-seed", type=int, default=0,
        help="RNG seed for --obs-spacing-mode uniform/poisson (ignored for 'fixed'). "
             "One rng is shared across every episode in the run, so this seed alone "
             "(plus the fixed scene/episode traversal order) fully determines every "
             "sampled gap in the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_root = (args.image_root if args.image_root is not None else args.data_dir / "images").resolve()
    metadata_dir = args.data_dir / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"No metadata/ dir under {args.data_dir}")

    chunk_config = make_chunk_config(
        native_fps=args.native_fps, obs_hz=args.obs_hz, action_chunk_len=args.action_chunk_len
    )
    spacing_config = SpacingConfig(
        mode=args.obs_spacing_mode, jitter=args.obs_spacing_jitter, seed=args.obs_spacing_seed
    )
    print(
        f"ChunkConfig: native_fps={chunk_config.native_fps} obs_hz(requested)={chunk_config.obs_hz} "
        f"obs_hz(effective)={chunk_config.effective_obs_hz:.3f} "
        f"obs_stride_frames={chunk_config.obs_stride_frames} action_chunk_len={chunk_config.action_chunk_len} "
        f"chunk_duration={chunk_config.chunk_duration:.3f}s obs_interval={chunk_config.obs_interval:.3f}s "
        f"overlap_fraction={chunk_config.overlap_fraction:.3f}"
    )
    print(
        f"SpacingConfig: mode={spacing_config.mode!r} jitter={spacing_config.jitter} "
        f"seed={spacing_config.seed} "
        f"(mean stride {chunk_config.obs_stride_frames} frames; every draw is clipped to "
        f"[1, {chunk_config.action_chunk_len}])"
    )

    rows = build_all_rows(metadata_dir, chunk_config, spacing_config, args.max_length)
    if not rows:
        raise ValueError("No episode rows produced -- check --data-dir / --max-length.")

    train_rows, val_rows = split_rows_by_scene(rows, args.val_scene_count)
    print(f"Train rows (episodes): {len(train_rows)}  Val rows (episodes): {len(val_rows)}")
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

    # Make the per-observation image paths absolute (relative rgb_paths ->
    # --image-root), as a .map step -- matching build_uniform_dataset.py.
    print("Fixing image paths (make_absolute)...")

    def make_absolute(example):
        example["images"] = [str(image_root / p) for p in example["images"]]
        return example

    dataset_dict = dataset_dict.map(
        make_absolute, num_proc=args.num_proc, desc="Fixing Paths"
    )

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
                "obs_spacing_mode": spacing_config.mode,
                "obs_spacing_jitter": spacing_config.jitter,
                "obs_spacing_seed": spacing_config.seed,
                "image_root": str(image_root),
                "row_granularity": "episode",
            },
            f,
            indent=2,
        )
    print(f"Saved dataset to {args.output_dir} ({config_path.name} alongside it).")


if __name__ == "__main__":
    main()
