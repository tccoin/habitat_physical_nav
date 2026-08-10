"""Shared CLI/config plumbing for the action-chunk dataset builders.

``build_action_chunk_dataset.py`` (one row per observation) and
``build_action_chunk_episode_dataset.py`` (one row per episode) differ only in
table shape, but each used to carry its own copy of the chunking arguments,
its own ``--native-fps`` default and its own idea of which derived values to
stamp onto a row. That duplication is how the two of them came to share a
default of ``20.0`` against a 30 Hz corpus. Everything they must agree on now
lives here:

* :func:`add_chunking_args` -- the argument group, defined once.
* :func:`resolve_build_config` -- read the corpus's own recording parameters
  (``continuous_demos.recording_metadata``), turn the build-time knobs into a
  validated :class:`~continuous_demos.action_chunking.ChunkConfig`, and refuse
  to proceed on a corpus whose rate is neither recorded nor explicitly stated.
* :func:`stamped_row_fields` -- the in-band block repeated onto every row.
* :func:`sidecar_dict` / :func:`build_name` -- the sidecar written next to a
  built dataset, and the directory name a sweep gives one.

------------------------------------------------------------------------
Recorded rate vs build-time knobs
------------------------------------------------------------------------
The distinction this module exists to enforce: ``native_fps`` is a **property
of the raw corpus** and is read, never chosen. ``action_chunk_len`` and
``obs_stride_frames`` are **build-time parameters of the arrow table** and are
chosen freely -- sweeping them is cheap and regenerates nothing. Every derived
quantity (``dt_native``, ``obs_interval``, ``effective_obs_hz``,
``chunk_duration``, ``overlap_fraction``) is a function of one recorded value
and two chosen ones, so all five are computed here from the corpus's own rate
and stamped per row. None of them may be assumed, and none of them can be
relabeled onto an existing dataset after the fact -- they are in-band, which
is why a change of any of them means a rebuild.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from continuous_demos.action_chunking import ChunkConfig, SpacingConfig, make_chunk_config
from continuous_demos.recording_metadata import (
    LEGACY_NATIVE_FPS,
    ResolvedRecording,
    resolve_corpus_params,
)

#: The stride/chunk-length grid under discussion, as a convenience for sweep
#: drivers and tests. At 25 Hz: stride 5 -> exactly 5 Hz observations,
#: stride 6 -> 4.167 Hz; chunk lengths 10/12/15 -> 0.40/0.48/0.60 s horizons.
DEFAULT_SWEEP_CHUNK_LENS: Tuple[int, ...] = (10, 12, 15)
DEFAULT_SWEEP_STRIDES: Tuple[int, ...] = (5, 6)


def add_chunking_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments both builders (and the sweep driver) share."""
    group = parser.add_argument_group("chunking (build-time parameters of the arrow table)")
    group.add_argument(
        "--action-chunk-len", type=int, default=10,
        help="Number of future native-fps poses per action chunk. Build-time knob: "
             "changing it costs a rebuild, not a regeneration. Default 10 "
             "(0.4s at 25 Hz).",
    )
    spacing = group.add_mutually_exclusive_group()
    spacing.add_argument(
        "--obs-stride-frames", type=int, default=None,
        help="Observation spacing in native frames -- the exact form, and the one to "
             "sweep, since it means the same thing at every native fps. The realized "
             "observation rate is native_fps / stride (at 25 Hz: 5 -> 5.0 Hz, "
             "6 -> 4.167 Hz). Mutually exclusive with --obs-hz; defaults to 5 if "
             "neither is given.",
    )
    spacing.add_argument(
        "--obs-hz", type=float, default=None,
        help="Observation spacing as a rate, quantized to a whole number of native "
             "frames (so the realized rate can differ from the request -- the build "
             "prints both). Mutually exclusive with --obs-stride-frames.",
    )
    group.add_argument(
        "--native-fps", type=float, default=None,
        help="NOT normally needed: the rate a corpus was recorded at is stored in its "
             "own metadata (the 'recording' block) and read from there. This flag is "
             "the explicit fallback for corpora generated before that schema existed, "
             f"all of which were recorded at {LEGACY_NATIVE_FPS:g} Hz -- pass "
             f"--native-fps {LEGACY_NATIVE_FPS:g} for those. Passing a value that "
             "contradicts what a corpus records is an error, not an override.",
    )

    group.add_argument(
        "--boundary-policy",
        choices=("keep", "clamp", "drop"),
        default="keep",
        help=(
            "what to do with an action chunk that spans a goal change in a chained "
            "PointNav episode -- its tail is motion toward the NEXT goal while the "
            "conditioning shows the current one. `keep` (default) leaves the chunk "
            "exactly as recorded, whose tail is a real turn-in-place toward the next "
            "goal (measured: 0.019 m translation, 37 deg rotation). `clamp` holds the "
            "last within-segment pose, which "
            "FABRICATES dead time -- it lifts exactly-zero ticks from 0.8%% to 11.7%% and "
            "freezes the second half of 11.7%% of chunks, and the policy then stands "
            "still. `drop` discards the row, removing every arrival from the corpus. "
            "Both were tried and both broke the policy. No effect on a "
            "corpus without segments."
        ),
    )

    latency = parser.add_argument_group("observation spacing distribution (inference latency)")
    latency.add_argument(
        "--obs-spacing-mode", default="fixed", choices=["fixed", "uniform", "poisson"],
        help="How far apart consecutive observations land. 'fixed': constant "
             "obs_stride_frames apart (default). 'uniform'/'poisson': each gap is "
             "sampled from a distribution centered on that same mean stride, "
             "simulating variable inference latency -- see "
             "continuous_demos.action_chunking.SpacingConfig.",
    )
    latency.add_argument(
        "--obs-spacing-jitter", type=float, default=0.5,
        help="Only used by --obs-spacing-mode uniform: half-width of the sampling "
             "window as a fraction of the mean stride (0.5 = +/-50%%).",
    )
    latency.add_argument(
        "--obs-spacing-seed", type=int, default=0,
        help="RNG seed for --obs-spacing-mode uniform/poisson (ignored for 'fixed'). "
             "One rng is shared across every episode in the run, so this seed alone "
             "(plus the fixed scene/episode traversal order) fully determines every "
             "sampled gap in the dataset.",
    )
    return parser


def resolve_build_config(
    args: argparse.Namespace,
    metadata_dir: Path,
    *,
    shard_paths: Optional[Sequence[Path]] = None,
) -> Tuple[ResolvedRecording, ChunkConfig, SpacingConfig]:
    """Read the corpus's recording parameters, then resolve the build knobs.

    Raises (via :func:`resolve_corpus_params`) rather than guessing when the
    corpus does not record its rate and none was supplied.
    """
    resolved = resolve_corpus_params(
        Path(metadata_dir), assumed_native_fps=args.native_fps, shard_paths=shard_paths
    )

    obs_stride_frames = getattr(args, "obs_stride_frames", None)
    obs_hz = getattr(args, "obs_hz", None)
    if obs_stride_frames is None and obs_hz is None:
        obs_stride_frames = 5  # see DEFAULT_SWEEP_STRIDES

    chunk_config = make_chunk_config(
        native_fps=resolved.params.native_fps,
        obs_hz=obs_hz,
        obs_stride_frames=obs_stride_frames,
        action_chunk_len=args.action_chunk_len,
    )
    spacing_config = SpacingConfig(
        mode=args.obs_spacing_mode, jitter=args.obs_spacing_jitter, seed=args.obs_spacing_seed
    )
    return resolved, chunk_config, spacing_config


def describe_build(
    resolved: ResolvedRecording, chunk_config: ChunkConfig, spacing_config: SpacingConfig
) -> str:
    """The banner both builders print before doing any work."""
    return "\n".join(
        [
            resolved.banner(),
            (
                f"ChunkConfig: native_fps={chunk_config.native_fps:g} "
                f"obs_stride_frames={chunk_config.obs_stride_frames} "
                f"action_chunk_len={chunk_config.action_chunk_len} "
                f"dt_native={chunk_config.dt_native:.4f}s "
                f"obs_interval={chunk_config.obs_interval:.4f}s "
                f"effective_obs_hz={chunk_config.effective_obs_hz:.3f} "
                f"chunk_duration={chunk_config.chunk_duration:.4f}s "
                f"overlap_fraction={chunk_config.overlap_fraction:.3f}"
            ),
            (
                f"SpacingConfig: mode={spacing_config.mode!r} jitter={spacing_config.jitter} "
                f"seed={spacing_config.seed} (mean stride "
                f"{chunk_config.obs_stride_frames} frames; every draw is clipped to "
                f"[1, {chunk_config.action_chunk_len}])"
            ),
        ]
    )


def stamped_row_fields(
    resolved: ResolvedRecording, chunk_config: ChunkConfig, spacing_config: SpacingConfig
) -> Dict[str, Any]:
    """The configuration block repeated in-band onto every output row.

    In-band rather than sidecar-only because rows get shuffled, concatenated
    and loaded far from the directory they were written in; it is also the
    reason a parameter change requires a rebuild rather than a relabel. Every
    derived field here is computed from the corpus's *recorded* rate.
    """
    params = resolved.params
    return {
        "native_fps": chunk_config.native_fps,
        "dt_native": chunk_config.dt_native,
        "obs_hz": chunk_config.obs_hz,
        "obs_stride_frames": chunk_config.obs_stride_frames,
        "action_chunk_len": chunk_config.action_chunk_len,
        "obs_interval": chunk_config.obs_interval,
        "effective_obs_hz": chunk_config.effective_obs_hz,
        "chunk_duration": chunk_config.chunk_duration,
        "overlap_fraction": chunk_config.overlap_fraction,
        "obs_spacing_mode": spacing_config.mode,
        # Straight from the corpus: what a saturated tick in this data hits,
        # and whether the rate was read or asserted.
        "v_max_mps": params.v_max_mps,
        "w_max_radps": params.w_max_radps,
        "recording_schema_version": params.schema_version,
        "native_fps_source": resolved.source,
        # Which task's demonstrations these are. `None` in the recording block means
        # objectnav (see RecordingParams), and it is resolved to the string HERE rather
        # than left null, because a consumer branching on the column should not also have
        # to know that history.
        "task": params.task_name,
    }


def sidecar_dict(
    resolved: ResolvedRecording,
    chunk_config: ChunkConfig,
    spacing_config: SpacingConfig,
    image_root: Path,
    **extra: Any,
) -> Dict[str, Any]:
    """``chunk_config.json``, written next to a built dataset.

    Carries the whole recording block (not just the fields worth repeating on
    every row) plus the resolved build knobs, so a built dataset can be
    audited without opening the arrow table.
    """
    blob: Dict[str, Any] = {
        "recording": resolved.params.as_dict(),
        "native_fps_source": resolved.source,
        "obs_hz_requested": getattr(chunk_config, "obs_hz", None),
        "obs_hz_effective": chunk_config.effective_obs_hz,
        "obs_stride_frames": chunk_config.obs_stride_frames,
        "action_chunk_len": chunk_config.action_chunk_len,
        "native_fps": chunk_config.native_fps,
        "dt_native": chunk_config.dt_native,
        "obs_interval": chunk_config.obs_interval,
        "chunk_duration": chunk_config.chunk_duration,
        "overlap_fraction": chunk_config.overlap_fraction,
        "obs_spacing_mode": spacing_config.mode,
        "obs_spacing_jitter": spacing_config.jitter,
        "obs_spacing_seed": spacing_config.seed,
        "task": resolved.params.task_name,
        "image_root": str(image_root),
    }
    blob.update(extra)
    return blob


def build_name(action_chunk_len: int, obs_stride_frames: int) -> str:
    """Directory name for one point of a (chunk_len x stride) sweep."""
    return f"clen{int(action_chunk_len)}_stride{int(obs_stride_frames)}"
