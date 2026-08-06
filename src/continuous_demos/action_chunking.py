"""Relative-pose action chunking: turns a per-frame ``pose2d`` trajectory (as
produced by ``generate_pose2d_dataset.py``, one pose per rendered RGB frame at
the replay's recording fps) into a lower-frequency stream of *observations*,
each paired with an *action chunk* -- a short run of future poses expressed
relative to that observation's own pose.

This is the format ``build_action_chunk_dataset.py`` assembles into an arrow
table, and that ``replay_chunked_actions.py`` re-expands back into an absolute
trajectory to sanity-check against a simulator. All of the index/frequency
arithmetic that both scripts need to agree on lives here so there is exactly
one place that can get the off-by-one wrong.

------------------------------------------------------------------------
Vocabulary
------------------------------------------------------------------------
* **native trajectory** -- the input: ``pose2d[0 .. N-1]``, one pose per
  frame, recorded at ``native_fps`` (whatever ``--fps`` was passed to
  ``generate_pose2d_dataset.py``). That rate is stored in the metadata
  JSONL's ``"recording"`` block and read back by
  ``continuous_demos.recording_metadata.resolve_corpus_params``; only corpora
  generated before that schema existed need it supplied by hand, and those
  were all recorded at 30 Hz.
* **observation** -- a single native-trajectory frame chosen to anchor a
  training example: its image, and its pose as the origin actions are made
  relative to. Observations are spaced ``obs_stride_frames`` native frames
  apart, i.e. at ``obs_hz = native_fps / obs_stride_frames``.
* **action chunk** -- ``action_chunk_len`` consecutive native-trajectory poses
  immediately following an observation, each expressed as ``[dx, dy, dtheta]``
  relative to that observation's own pose (not to each other -- every entry in
  a chunk is relative to the same anchor, so consuming the chunk does not
  require accumulating error across it).

------------------------------------------------------------------------
Why "chunk duration >= observation interval" is required
------------------------------------------------------------------------
A new action chunk is produced once per observation, i.e. once every
``obs_interval = obs_stride_frames * dt_native`` seconds. A chunk covers
``chunk_duration = action_chunk_len * dt_native`` seconds of future motion
starting at its observation. If ``chunk_duration < obs_interval``, the chunk
runs out of actions *before* the next observation's chunk becomes available --
an uncovered gap that makes synchronous chunk replay ill-defined. Requiring
``chunk_duration >= obs_interval`` guarantees every native frame is covered by
at least one chunk; when it is strictly greater, consecutive chunks overlap.
Equivalently, in frames rather than seconds: ``action_chunk_len >=
obs_stride_frames``, since both sides are just a count of native frames times
the same ``dt_native``. At 25 Hz with ``obs_stride_frames = 5`` and
``action_chunk_len = 10`` that is a 0.4 s chunk against a 0.2 s observation
interval, i.e. 2x coverage / 50% overlap.

------------------------------------------------------------------------
Relative-pose convention
------------------------------------------------------------------------
Matches the planar control frame used throughout this package (``X = habitat_x``,
``Y = -habitat_z``, ``theta`` CCW from ``+X`` -- see ``sim.py``'s module
docstring). A pose ``target`` expressed relative to ``anchor`` is ``target``
rotated into ``anchor``'s local frame (anchor's heading becomes local ``+X``):

    dx, dy   = target.xy - anchor.xy
    local_x  =  cos(anchor.theta) * dx + sin(anchor.theta) * dy
    local_y  = -sin(anchor.theta) * dx + cos(anchor.theta) * dy
    local_th = wrap_angle(target.theta - anchor.theta)

``relative_to_pose`` is the exact inverse (rotate back into the world frame
and add the anchor's translation); ``compose(anchor, relative(anchor, p)) == p``
for every ``p``, verified for random poses (including wrapped angles) in
``tests/test_action_chunking.py``.

------------------------------------------------------------------------
Variable observation spacing (simulated inference latency)
------------------------------------------------------------------------
By default observations land on a fixed stride (``obs_stride_frames`` native
frames apart, every time). :class:`SpacingConfig` / :func:`sample_gap_frames`
instead let the *gap* to the next observation be drawn per-observation from a
distribution -- standing in for a policy whose inference latency (hence how
long it takes to produce the next action chunk) varies run to run rather than
being a fixed clock tick.

The one hard constraint is unchanged and unrelated to which mode is active:
whatever gap is realized between two observations, the earlier one's chunk
must have enough actions to cover it (``gap <= action_chunk_len``), or replay
would run out of actions before the next chunk arrives. Under fixed spacing
this is checked once, up front, against the constant stride (see
``make_chunk_config``). Under variable spacing it's enforced per-sample by
clipping every draw to ``[1, action_chunk_len]`` (:func:`sample_gap_frames`) --
so it can never be violated by construction, at the cost of flattening the
tail of whatever distribution was requested into a point mass at
``action_chunk_len`` if the distribution's mean is set too close to (or past)
it. ``make_chunk_config`` still rejects a *mean* stride that exceeds
``action_chunk_len`` even in variable-spacing mode, since at that point
every draw would clip and "variable" spacing would silently degrade into
"fixed, at the chunk boundary" -- almost certainly not what was intended.

Downstream, nothing about chunk *construction* or *reconstruction* needs to
know which spacing mode produced a given dataset: an observation's chunk is
still built the same way regardless of how far the *previous* observation
was (each chunk is self-contained, anchored on its own pose -- spacing only
decides where the next ``idx`` lands). And on the replay side,
:func:`reconstruct_reference_trajectory` never reads a stride parameter at
all -- it derives each row's realized gap directly from consecutive
``frame_index`` values, which is exactly the fixed stride when spacing is
fixed and exactly the sampled-and-clipped gap when it isn't. This is why the
same replay code, and the same round-trip-exactness argument, cover both
modes without a fixed/variable branch anywhere in the reconstruction math.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from continuous_demos.pid_pose_controller import wrap_angle


@dataclass
class ChunkConfig:
    """Resolved, self-consistent chunking parameters.

    ``obs_stride_frames`` is the *quantized* observation spacing (native
    frames must fall on integer frame boundaries); ``obs_interval`` below is
    computed from that quantized stride, not from ``1 / obs_hz`` directly, so
    ``obs_hz`` may end up realized slightly differently than requested when
    ``native_fps`` isn't an exact multiple of it -- ``effective_obs_hz``
    reports what was actually achieved.
    """

    native_fps: float
    obs_hz: float
    action_chunk_len: int
    obs_stride_frames: int
    dt_native: float
    obs_interval: float
    chunk_duration: float

    @property
    def effective_obs_hz(self) -> float:
        return 1.0 / self.obs_interval

    @property
    def overlap_fraction(self) -> float:
        """Fraction of a chunk's duration re-covered by the *next* chunk.

        0.0 means chunks exactly touch (no gap, no overlap); this is only
        meaningful/well-defined because ``chunk_duration >= obs_interval`` is
        enforced at construction time.
        """
        return 1.0 - (self.obs_interval / self.chunk_duration)


def make_chunk_config(
    native_fps: float,
    obs_hz: Optional[float] = None,
    action_chunk_len: int = 10,
    obs_stride_frames: Optional[int] = None,
) -> ChunkConfig:
    """Build and validate a :class:`ChunkConfig` from the user-facing knobs.

    The observation spacing is given either as a rate (``obs_hz``, quantized to
    the nearest whole number of native frames) or, since the stride is what the
    arithmetic actually runs on, directly in frames (``obs_stride_frames``).
    Exactly one of the two is required. Specifying the stride is the exact
    form -- it is what makes a stride sweep mean the same thing at every
    ``native_fps``, where a rate would quantize differently at each -- and it
    makes ``obs_hz`` a reported consequence (``native_fps / stride``) rather
    than a request that may not be satisfiable.

    Raises ``ValueError`` if ``chunk_duration < obs_interval`` (see the module
    docstring for why that would leave a coverage gap between chunks) or if
    any input is non-positive.
    """
    if native_fps <= 0:
        raise ValueError(f"native_fps must be positive, got {native_fps}")
    if (obs_hz is None) == (obs_stride_frames is None):
        raise ValueError(
            "pass exactly one of obs_hz (a rate, quantized to whole native frames) or "
            f"obs_stride_frames (the exact spacing), got obs_hz={obs_hz!r} "
            f"obs_stride_frames={obs_stride_frames!r}"
        )
    if obs_hz is not None and obs_hz <= 0:
        raise ValueError(f"obs_hz must be positive, got {obs_hz}")
    if obs_stride_frames is not None and obs_stride_frames < 1:
        raise ValueError(f"obs_stride_frames must be >= 1, got {obs_stride_frames}")
    if action_chunk_len < 1:
        raise ValueError(f"action_chunk_len must be >= 1, got {action_chunk_len}")

    dt_native = 1.0 / native_fps
    if obs_stride_frames is None:
        obs_stride_frames = max(1, round(native_fps / obs_hz))
    else:
        obs_stride_frames = int(obs_stride_frames)
        obs_hz = native_fps / obs_stride_frames
    obs_interval = obs_stride_frames * dt_native
    chunk_duration = action_chunk_len * dt_native

    if chunk_duration < obs_interval - 1e-9:
        raise ValueError(
            f"chunk_duration ({chunk_duration:.4f}s = action_chunk_len={action_chunk_len} "
            f"* dt_native={dt_native:.4f}s) is shorter than obs_interval "
            f"({obs_interval:.4f}s = obs_stride_frames={obs_stride_frames} * dt_native); "
            "this leaves a gap between one chunk running out and the next observation's "
            "chunk becoming available. Increase --action-chunk-len or decrease --obs-hz."
        )

    return ChunkConfig(
        native_fps=float(native_fps),
        obs_hz=float(obs_hz),
        action_chunk_len=int(action_chunk_len),
        obs_stride_frames=int(obs_stride_frames),
        dt_native=dt_native,
        obs_interval=obs_interval,
        chunk_duration=chunk_duration,
    )


_SPACING_MODES = ("fixed", "uniform", "poisson")


@dataclass
class SpacingConfig:
    """How far the *next* observation lands from the current one, in native
    frames -- see the module docstring's "Variable observation spacing"
    section.

    ``mode="fixed"`` reproduces the original constant-stride behavior exactly
    (every gap equals ``ChunkConfig.obs_stride_frames``) and is the default,
    so existing callers are unaffected. ``"uniform"`` and ``"poisson"`` both
    center on that same mean stride -- ``obs_hz`` keeps its meaning as the
    *mean* observation rate in every mode -- and differ only in how spread
    out the realized gaps are around it:

    * ``"uniform"``: gap ~ ``Uniform[round(mean * (1 - jitter)),
      round(mean * (1 + jitter))]`` (inclusive), i.e. a symmetric window
      whose half-width is ``jitter`` as a fraction of the mean stride.
    * ``"poisson"``: gap ~ ``Poisson(mean)`` -- unbounded above in principle
      (clipped to ``action_chunk_len`` in practice, see
      :func:`sample_gap_frames`), asymmetric, more realistic for a latency
      that's usually close to its mean but occasionally spikes.

    ``seed`` makes a build reproducible: the same seed (plus the same
    episode traversal order) always draws the same sequence of gaps.
    """

    mode: str = "fixed"
    jitter: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in _SPACING_MODES:
            raise ValueError(f"mode must be one of {_SPACING_MODES}, got {self.mode!r}")
        if self.jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {self.jitter}")


def sample_gap_frames(
    rng: np.random.Generator, spacing_config: SpacingConfig, chunk_config: ChunkConfig
) -> int:
    """Draw one observation-to-observation gap, in native frames.

    Always returns an integer in ``[1, chunk_config.action_chunk_len]`` --
    the clip is what guarantees the "chunk covers the gap to the next
    observation" invariant holds for every draw, not just in expectation
    (see the module docstring).
    """
    mean = chunk_config.obs_stride_frames
    if spacing_config.mode == "fixed":
        raw = mean
    elif spacing_config.mode == "uniform":
        lo = max(1, round(mean * (1.0 - spacing_config.jitter)))
        hi = max(lo, round(mean * (1.0 + spacing_config.jitter)))
        raw = int(rng.integers(lo, hi + 1))  # rng.integers high is exclusive
    elif spacing_config.mode == "poisson":
        raw = int(rng.poisson(mean))
    else:  # pragma: no cover -- guarded by SpacingConfig.__post_init__
        raise ValueError(f"Unknown spacing mode {spacing_config.mode!r}")
    return int(np.clip(raw, 1, chunk_config.action_chunk_len))


def pose_to_relative(anchor: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """Express ``target`` (``[x, y, theta]``) relative to ``anchor``'s local frame."""
    x0, y0, th0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
    x1, y1, th1 = float(target[0]), float(target[1]), float(target[2])
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(th0), math.sin(th0)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    local_th = wrap_angle(th1 - th0)
    return np.array([local_x, local_y, local_th], dtype=float)


def relative_to_pose(anchor: Sequence[float], relative: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`pose_to_relative`: recover the absolute pose from
    ``anchor`` plus a ``[dx, dy, dtheta]`` relative to it."""
    x0, y0, th0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
    lx, ly, lth = float(relative[0]), float(relative[1]), float(relative[2])
    c, s = math.cos(th0), math.sin(th0)
    x1 = x0 + c * lx - s * ly
    y1 = y0 + s * lx + c * ly
    th1 = wrap_angle(th0 + lth)
    return np.array([x1, y1, th1], dtype=float)


def build_observation_records(
    pose2d: Sequence[Sequence[float]],
    rgb_paths: Sequence[str],
    chunk_config: ChunkConfig,
    spacing_config: Optional[SpacingConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Slide an observation window over one episode's native trajectory.

    ``pose2d`` and ``rgb_paths`` must be the same length ``N`` and mutually
    aligned frame-for-frame (exactly the arrays a ``metadata/<scene>.jsonl``
    record stores). Returns one record per observation, in ascending
    ``obs_index`` order:

        {
          "obs_index":    0, 1, 2, ...                 -- position in this episode's chunk stream
          "frame_index":  idx                            -- native-trajectory index this obs anchors on
          "image":        rgb_paths[idx]
          "obs_pose":     pose2d[idx]                    -- [x, y, theta], the chunk's anchor
          "action_chunk": [pose_to_relative(pose2d[idx], pose2d[idx+1]), ...,
                            pose_to_relative(pose2d[idx], pose2d[idx+action_chunk_len])]
        }

    An observation at native index ``idx`` is only emitted if the full chunk
    fits inside the recorded trajectory, i.e. ``idx + action_chunk_len <= N - 1``
    (0-indexed last valid frame is ``N - 1``). Trailing frames that can't
    anchor a full chunk are silently dropped -- there is no padding.

    ``spacing_config`` controls how far apart consecutive observations land
    (see the module docstring's "Variable observation spacing" section);
    defaults to fixed-stride spacing (``obs_stride_frames`` apart every time,
    the original behavior) when omitted. ``rng`` defaults to
    ``np.random.default_rng(spacing_config.seed)`` when omitted -- pass one
    explicitly to share a single draw sequence across multiple episodes in
    one build run rather than resetting the seed per episode.

    Note there is no closed-form observation count under variable spacing
    (unlike the fixed-stride case's
    ``floor((N - 1 - action_chunk_len) / obs_stride_frames) + 1``) since it
    depends on the realized gaps; the loop below simply runs until the next
    would-be chunk no longer fits.
    """
    n = len(pose2d)
    if len(rgb_paths) != n:
        raise ValueError(f"pose2d ({n}) and rgb_paths ({len(rgb_paths)}) length mismatch")

    if spacing_config is None:
        spacing_config = SpacingConfig(mode="fixed")
    if rng is None:
        rng = np.random.default_rng(spacing_config.seed)

    length = chunk_config.action_chunk_len

    records: List[Dict[str, Any]] = []
    obs_index = 0
    idx = 0
    while idx + length <= n - 1:
        anchor = pose2d[idx]
        chunk = [pose_to_relative(anchor, pose2d[idx + j]) for j in range(1, length + 1)]
        records.append(
            {
                "obs_index": obs_index,
                "frame_index": idx,
                "image": rgb_paths[idx],
                "obs_pose": [float(v) for v in anchor],
                "action_chunk": [c.tolist() for c in chunk],
            }
        )
        obs_index += 1
        idx += sample_gap_frames(rng, spacing_config, chunk_config)
    return records


def reconstruct_reference_trajectory(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Invert :func:`build_observation_records`: decode a sorted-by-``obs_index``
    run of a single episode's observation rows back into an absolute,
    native-fps pose trajectory -- the input this whole module started from.

    No spacing/stride parameter is taken or needed. Each row already records
    exactly where it landed (``frame_index``), so the realized gap to the
    next observation is simply
    ``rows[i + 1]["frame_index"] - rows[i]["frame_index"]`` -- this is
    ``obs_stride_frames`` under fixed spacing and the sampled-and-clipped
    draw under variable spacing, with no code-level distinction between the
    two: whichever mode built the dataset, the gap that actually happened is
    what's replayed. Only the first ``gap`` actions of row ``i``'s chunk are
    kept (de-relativized against that row's own ``obs_pose``) -- the rest
    would be superseded by row ``i + 1``'s chunk before ever being used. The
    last row, which nothing supersedes, is decoded in full.

    The returned array is prefixed with the first row's own ``obs_pose`` as
    its leading pose (matching ``eval_pid_tracking.py``'s ``ref_traj[0]`` --
    the pose the robot is already spawned at -- see
    ``replay_chunked_actions.py`` for why dropping that leading tick produces
    a real, non-decaying tracking discrepancy rather than a negligible one).

    Raises ``ValueError`` if any consecutive pair's gap isn't covered by the
    earlier row's chunk (``1 <= gap <= len(chunk)``) -- this should be
    unreachable for a dataset built by ``build_observation_records`` (which
    clips every gap at construction time), so tripping it means the rows
    passed in didn't actually come from one, in original order.
    """
    if not rows:
        return np.zeros((0, 3), dtype=float)

    targets = [np.asarray(rows[0]["obs_pose"], dtype=float)]
    n = len(rows)
    for i, row in enumerate(rows):
        anchor = row["obs_pose"]
        chunk = row["action_chunk"]
        if i == n - 1:
            gap = len(chunk)
        else:
            gap = rows[i + 1]["frame_index"] - row["frame_index"]
            if not (1 <= gap <= len(chunk)):
                raise ValueError(
                    f"row {i} (obs_index={row.get('obs_index')}) has a gap of {gap} frames "
                    f"to the next observation, outside its own chunk's coverage "
                    f"[1, {len(chunk)}] -- rows are not a valid, in-order single-episode "
                    "observation stream."
                )
        for j in range(gap):
            targets.append(relative_to_pose(anchor, chunk[j]))
    return np.asarray(targets, dtype=float)
