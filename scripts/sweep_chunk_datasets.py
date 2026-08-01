"""Build one action-chunk dataset per (``action_chunk_len``, ``obs_stride_frames``)
point of a grid, and tabulate them side by side.

Why a sweep is cheap: both of these are **build-time parameters of the arrow
table**, not properties of the raw corpus. Varying them re-reads
``metadata/*.jsonl`` and re-slices trajectories that already exist; nothing is
re-simulated and no frames are re-rendered. What they are *not* is relabelable
-- every derived value (``dt_native``, ``obs_interval``, ``effective_obs_hz``,
``chunk_duration``, ``overlap_fraction``) is stamped in-band onto every row, so
each grid point is a separate build.

The default grid is the one under discussion::

    action_chunk_len in {10, 12, 15}   x   obs_stride_frames in {5, 6}

At 25 Hz that is a 0.40 / 0.48 / 0.60 s action horizon against 5.0 Hz
(stride 5) or 4.167 Hz (stride 6) observations. The stride is swept rather
than an ``--obs-hz`` because the stride is what the indexing runs on and means
the same thing at every native rate, whereas a rate has to be quantized to
whole frames and lands differently at 25 vs 30 Hz.

The native rate is *not* part of the grid: it is read from the corpus's own
recording block (see ``continuous_demos.recording_metadata``). Sweeping it
would mean regenerating, which is the expensive thing this deliberately does
not do.

Usage::

    # what would be built, and every derived value, without building anything
    python sweep_chunk_datasets.py --data-dir output_25hz \\
        --output-root output_25hz/sweeps --dry-run

    # the real thing (resumable: already-built points are skipped)
    python sweep_chunk_datasets.py --data-dir output_25hz \\
        --output-root output_25hz/sweeps --val-scene-count 5

    # a pre-schema (30 Hz) corpus needs its rate stated once, as everywhere else
    python sweep_chunk_datasets.py --data-dir output --output-root /tmp/sweep \\
        --native-fps 30 --dry-run

Each point lands in ``<output-root>/clen<L>_stride<S>/`` with its usual
``chunk_config.json``; the comparison table is written to
``<output-root>/sweep_summary.json`` and ``sweep_summary.md`` and printed at
the end. The summary is rewritten after every point, so a killed run still
leaves the results of everything that finished.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from continuous_demos.action_chunking import make_chunk_config
from continuous_demos.chunk_build import (
    DEFAULT_SWEEP_CHUNK_LENS,
    DEFAULT_SWEEP_STRIDES,
    build_name,
)
from continuous_demos.recording_metadata import resolve_corpus_params

BUILDERS = {
    "observation": "build_action_chunk_dataset.py",
    "episode": "build_action_chunk_episode_dataset.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep action_chunk_len x obs_stride_frames and compare the builds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Output dir of generate_pose2d_dataset.py (contains images/ and metadata/).",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="One subdirectory per grid point is created under here.",
    )
    parser.add_argument(
        "--chunk-lens", type=int, nargs="+", default=list(DEFAULT_SWEEP_CHUNK_LENS),
        help="action_chunk_len values to sweep.",
    )
    parser.add_argument(
        "--obs-strides", type=int, nargs="+", default=list(DEFAULT_SWEEP_STRIDES),
        help="obs_stride_frames values to sweep.",
    )
    parser.add_argument(
        "--builder", choices=sorted(BUILDERS), default="observation",
        help="Which table shape to build at each point (default: one row per observation).",
    )
    parser.add_argument(
        "--native-fps", type=float, default=None,
        help="Only for corpora that predate the recording schema (all of which were "
             "recorded at 30 Hz). Passed straight through to the builder.",
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--val-scene-count", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--obs-spacing-mode", default="fixed",
                        choices=["fixed", "uniform", "poisson"])
    parser.add_argument("--obs-spacing-jitter", type=float, default=0.5)
    parser.add_argument("--obs-spacing-seed", type=int, default=0)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve the corpus, print the grid and every derived value, build nothing.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild grid points that already have a chunk_config.json (default: skip, "
             "so an interrupted sweep resumes).",
    )
    parser.add_argument(
        "--timeout", type=int, default=14400,
        help="Per-build timeout in seconds (default 4h). A point that times out is "
             "recorded as failed and the sweep continues.",
    )
    return parser.parse_args()


def builder_command(args: argparse.Namespace, chunk_len: int, stride: int, out_dir: Path) -> List[str]:
    script = Path(__file__).resolve().parent / BUILDERS[args.builder]
    cmd = [
        sys.executable, str(script),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(out_dir),
        "--action-chunk-len", str(chunk_len),
        "--obs-stride-frames", str(stride),
        "--val-scene-count", str(args.val_scene_count),
        "--obs-spacing-mode", args.obs_spacing_mode,
        "--obs-spacing-jitter", str(args.obs_spacing_jitter),
        "--obs-spacing-seed", str(args.obs_spacing_seed),
    ]
    if args.native_fps is not None:
        cmd += ["--native-fps", str(args.native_fps)]
    if args.image_root is not None:
        cmd += ["--image-root", str(args.image_root)]
    if args.max_length is not None:
        cmd += ["--max-length", str(args.max_length)]
    return cmd


def derived_table(native_fps: float, chunk_lens, strides) -> List[Dict[str, Any]]:
    """Every grid point's derived values, straight from the recorded rate."""
    table = []
    for chunk_len in chunk_lens:
        for stride in strides:
            entry: Dict[str, Any] = {
                "name": build_name(chunk_len, stride),
                "action_chunk_len": chunk_len,
                "obs_stride_frames": stride,
            }
            try:
                cfg = make_chunk_config(
                    native_fps=native_fps,
                    obs_stride_frames=stride,
                    action_chunk_len=chunk_len,
                )
            except ValueError as exc:
                # Degenerate: the chunk runs out before the next observation.
                entry["error"] = str(exc)
                table.append(entry)
                continue
            entry.update(
                native_fps=cfg.native_fps,
                dt_native=cfg.dt_native,
                obs_interval=cfg.obs_interval,
                effective_obs_hz=cfg.effective_obs_hz,
                chunk_duration=cfg.chunk_duration,
                overlap_fraction=cfg.overlap_fraction,
            )
            table.append(entry)
    return table


def markdown_table(rows: List[Dict[str, Any]], native_fps: float, source: str) -> str:
    header = (
        f"# Chunk sweep at native_fps = {native_fps:g} Hz (rate {source})\n\n"
        "| build | chunk_len | stride | chunk_duration (s) | obs_interval (s) | "
        "effective_obs_hz | overlap | rows | status |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        if "error" in r:
            lines.append(
                f"| {r['name']} | {r['action_chunk_len']} | {r['obs_stride_frames']} | "
                "- | - | - | - | - | rejected (chunk shorter than the observation "
                "interval) |"
            )
            continue
        rows_built = r.get("num_rows")
        lines.append(
            f"| {r['name']} | {r['action_chunk_len']} | {r['obs_stride_frames']} | "
            f"{r['chunk_duration']:.4f} | {r['obs_interval']:.4f} | "
            f"{r['effective_obs_hz']:.3f} | {r['overlap_fraction']:.3f} | "
            f"{rows_built if rows_built is not None else '-'} | {r.get('status', 'planned')} |"
        )
    return header + "\n".join(lines) + "\n"


def write_summary(output_root: Path, rows, native_fps: float, source: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"native_fps": native_fps, "native_fps_source": source, "builds": rows}, f, indent=2
        )
    with open(output_root / "sweep_summary.md", "w", encoding="utf-8") as f:
        f.write(markdown_table(rows, native_fps, source))


def existing_row_count(out_dir: Path) -> Optional[int]:
    config_path = out_dir / "chunk_config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            blob = json.load(f)
    except json.JSONDecodeError:
        return None
    counts = blob.get("num_rows") or {}
    return sum(v for v in counts.values() if isinstance(v, int)) or 0


def main() -> None:
    args = parse_args()
    metadata_dir = args.data_dir / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"No metadata/ dir under {args.data_dir}")

    # The one thing that is read, never chosen.
    resolved = resolve_corpus_params(metadata_dir, assumed_native_fps=args.native_fps)
    print(resolved.banner())
    native_fps = resolved.params.native_fps

    rows = derived_table(native_fps, args.chunk_lens, args.obs_strides)
    print(markdown_table(rows, native_fps, resolved.source))
    write_summary(args.output_root, rows, native_fps, resolved.source)

    if args.dry_run:
        print(f"--dry-run: nothing built. Plan written to {args.output_root}/sweep_summary.md")
        return

    for entry in rows:
        if "error" in entry:
            entry["status"] = "skipped (degenerate)"
            continue
        out_dir = args.output_root / entry["name"]
        already = existing_row_count(out_dir)
        if already is not None and not args.rebuild:
            entry["status"] = "already built"
            entry["num_rows"] = already
            print(f"[{entry['name']}] already built ({already} rows) -- skipping")
            write_summary(args.output_root, rows, native_fps, resolved.source)
            continue

        cmd = builder_command(args, entry["action_chunk_len"], entry["obs_stride_frames"], out_dir)
        print(f"\n[{entry['name']}] {' '.join(cmd)}")
        started = time.time()
        try:
            proc = subprocess.run(cmd, timeout=args.timeout)
            ok = proc.returncode == 0
            entry["status"] = "built" if ok else f"FAILED (exit {proc.returncode})"
        except subprocess.TimeoutExpired:
            entry["status"] = f"FAILED (timeout after {args.timeout}s)"
        entry["seconds"] = round(time.time() - started, 1)
        entry["num_rows"] = existing_row_count(out_dir)
        print(f"[{entry['name']}] {entry['status']} in {entry['seconds']}s")
        write_summary(args.output_root, rows, native_fps, resolved.source)

    print("\n" + markdown_table(rows, native_fps, resolved.source))
    print(f"Summary: {args.output_root}/sweep_summary.md")
    if any(str(r.get("status", "")).startswith("FAILED") for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
