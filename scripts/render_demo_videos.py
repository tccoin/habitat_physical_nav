"""Watch the training demonstrations the policy is imitating.

Distinct from ``render_pose2d_videos.py``, which reads a
``generate_pose2d_dataset.py`` output directory. This one reads the **built
corpus** -- the HuggingFace dataset a run actually trained on -- so what you
see is what the model saw, including any filtering or resampling the build
applied. When a rollout looks wrong, this is how you check whether the
behaviour was learned or invented.

Two rates, and the difference matters:

* ``--rate native`` (default) plays every frame on disk at ``native_fps``
  (25 Hz), which is real-time and smooth. The corpus row only *indexes* every
  5th frame, but the intervening frames are still in the image directory, so
  this reconstructs continuous motion.
* ``--rate observation`` plays only the frames the model was given, at the
  observation rate (5 Hz). Jerkier, and that jerkiness is the point: it is the
  temporal resolution the policy actually perceives.

Episodes are chosen with a seeded RNG and the selection is printed, so a
sample can be reproduced and is not quietly curated:

    python scripts/render_demo_videos.py --num 6 --out /tmp/demos
    python scripts/render_demo_videos.py --episode-id <id> --rate observation
    python scripts/render_demo_videos.py --num 4 --seed 7 --overlay

``--overlay`` burns the per-observation pose and the commanded chunk into the
frame, which is what you want when the question is "is it turning because the
demo turned, or because the controller lagged".
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

DEFAULT_DATASET = "/Projects/spatial_training/data/v2_25hz/formatted_pose"


def episode_frames(row, rate):
    """Paths to render, in order.

    ``native`` walks the image directory between the first and last indexed
    frame rather than trusting a naming convention -- a missing frame should
    shorten the video, not raise on a formatted filename that does not exist.
    """
    indexed = list(row["images"])
    if rate == "observation":
        return indexed
    directory = Path(indexed[0]).parent
    first, last = int(row["frame_indices"][0]), int(row["frame_indices"][-1])
    frames = []
    for i in range(first, last + 1):
        candidate = directory / f"frame_{i:05d}.jpeg"
        if candidate.exists():
            frames.append(str(candidate))
    # If the directory is not laid out as expected, fall back rather than
    # silently emitting a 2-frame video.
    return frames if len(frames) >= len(indexed) else indexed


def annotate(image, text_lines):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    y = 4
    for line in text_lines:
        draw.text((6, y), line, fill=(255, 255, 0))
        y += 12
    return image


def render(row, out_path, rate, overlay):
    import imageio.v2 as imageio
    from PIL import Image

    frames = episode_frames(row, rate)
    fps = float(row["native_fps"]) if rate == "native" else float(row["effective_obs_hz"])
    stride = int(row["obs_stride_frames"])
    poses = np.asarray(row["obs_poses"], dtype=np.float64)
    chunks = np.asarray(row["action_chunks"], dtype=np.float64)
    first = int(row["frame_indices"][0])

    with imageio.get_writer(out_path, fps=max(fps, 1.0), macro_block_size=None) as writer:
        for k, path in enumerate(frames):
            image = Image.open(path).convert("RGB")
            if overlay:
                # Which observation is in force at this frame. In native mode a
                # frame sits between observations, so it takes the most recent
                # one -- the same value the policy would still be acting on.
                obs = min(len(poses) - 1,
                          (int(Path(path).stem.split("_")[-1]) - first) // stride
                          if rate == "native" else k)
                pose = poses[obs]
                chunk = chunks[obs]
                image = annotate(image, [
                    f"obs {obs}/{len(poses) - 1}  frame {k}/{len(frames) - 1}",
                    f"pose  X {pose[0]:+.2f}  Y {pose[1]:+.2f}  th {np.degrees(pose[2]):+7.1f}d",
                    f"chunk dxy {np.linalg.norm(chunk[-1][:2]):.3f} m  "
                    f"dth {np.degrees(chunk[-1][2]):+6.1f}d  (over {row['chunk_duration']}s)",
                    f"{row['goal_text']}",
                ])
            writer.append_data(np.asarray(image, dtype=np.uint8))
    return len(frames), fps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--num", type=int, default=4, help="how many episodes to sample")
    p.add_argument("--seed", type=int, default=0,
                   help="the sample is seeded and printed so it can be reproduced; "
                        "change it to draw a different sample, do not hand-pick")
    p.add_argument("--episode-id", default=None, help="render exactly this episode")
    p.add_argument("--rate", default="native", choices=["native", "observation"])
    p.add_argument("--overlay", action="store_true",
                   help="burn pose and commanded chunk into each frame")
    p.add_argument("--min-observations", type=int, default=0,
                   help="skip episodes shorter than this. Use it deliberately: a "
                        "length filter is a selection, and a corpus's short episodes "
                        "are not a different kind of episode")
    p.add_argument("--out", default="/tmp/demo_videos")
    args = p.parse_args()

    from datasets import load_from_disk

    data = load_from_disk(args.dataset)
    rows = data[args.split] if hasattr(data, "keys") else data
    os.makedirs(args.out, exist_ok=True)

    if args.episode_id is not None:
        picks = [i for i in range(len(rows))
                 if str(rows[i]["episode_id"]) == str(args.episode_id)]
        if not picks:
            raise SystemExit(f"episode_id {args.episode_id} not found")
    else:
        rng = random.Random(args.seed)
        if args.min_observations:
            lengths = np.asarray(rows["num_observations"])
            eligible = np.flatnonzero(lengths >= args.min_observations).tolist()
            print(f"{len(eligible)}/{len(rows)} episodes have >= "
                  f"{args.min_observations} observations")
        else:
            eligible = range(len(rows))
        picks = rng.sample(list(eligible), min(args.num, len(eligible)))

    print(f"dataset {args.dataset} split={args.split} n={len(rows)}  "
          f"seed={args.seed}  rate={args.rate}")
    manifest = []
    for index in picks:
        row = rows[index]
        name = f"{row['scene_id']}_{row['episode_id']}".replace("/", "_")[:120]
        out_path = os.path.join(args.out, f"{name}.mp4")
        n_frames, fps = render(row, out_path, args.rate, args.overlay)
        poses = np.asarray(row["obs_poses"], dtype=np.float64)
        travelled = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())
        gross = float(np.abs(np.diff(np.unwrap(poses[:, 2]))).sum())
        entry = {
            "index": index, "episode_id": row["episode_id"], "scene": row["scene_id"],
            "goal": row["goal_text"], "observations": int(row["num_observations"]),
            "frames": n_frames, "fps": fps, "path_m": round(travelled, 2),
            "gross_rotation_rad": round(gross, 2),
            "rad_per_m": round(gross / travelled, 2) if travelled > 0.05 else None,
            "video": out_path,
        }
        manifest.append(entry)
        print(f"  [{index:6d}] {row['goal_text']:12s} {row['num_observations']:3d} obs  "
              f"{travelled:5.1f} m  {gross:5.1f} rad  "
              f"{entry['rad_per_m'] if entry['rad_per_m'] is not None else '   -'} rad/m  "
              f"-> {out_path}")

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"seed": args.seed, "split": args.split, "rate": args.rate,
                   "episodes": manifest}, fh, indent=2)
    print(f"\nmanifest -> {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
