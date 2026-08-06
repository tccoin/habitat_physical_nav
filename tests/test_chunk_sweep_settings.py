"""Chunk construction, verified at every setting the sweep can build.

``chunk_len`` and ``obs_stride`` are build-time parameters of the arrow table,
so they are cheap to vary -- but every one of them has to satisfy the same
indexing contract, and the off-by-one in it has bitten before:

    chunk[j] = pose_to_relative(anchor, pose2d[idx + j + 1])

i.e. a chunk of length L covers ticks ``+1 .. +L`` relative to its
observation, **not** ``+0 .. +L-1``. The consequence that actually matters
downstream is that the next observation (``obs_stride`` frames later) decodes
from ``chunk[obs_stride - 1]`` and never from ``chunk[obs_stride]``.

These run the whole ``{10, 12, 15} x {5, 6}`` grid at both the new 25 Hz and
the legacy 30 Hz, plus the boundary and degenerate cases either side of it.
No simulator, no dataset.
"""

import itertools

import numpy as np
import pytest

from continuous_demos.action_chunking import (
    SpacingConfig,
    build_observation_records,
    make_chunk_config,
    pose_to_relative,
    reconstruct_reference_trajectory,
    relative_to_pose,
)
from continuous_demos.chunk_build import (
    DEFAULT_SWEEP_CHUNK_LENS,
    DEFAULT_SWEEP_STRIDES,
    build_name,
)

NATIVE_RATES = (25.0, 30.0)
GRID = list(itertools.product(DEFAULT_SWEEP_CHUNK_LENS, DEFAULT_SWEEP_STRIDES))
GRID_IDS = [build_name(c, s) for c, s in GRID]


def synthetic_trajectory(n: int = 200, seed: int = 7) -> np.ndarray:
    """A wiggly but deterministic pose2d trajectory, headings deliberately
    wrapped past +/-pi so the angle handling is exercised too."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = 0.05 * t + 0.3 * np.sin(0.1 * t)
    y = 0.2 * np.cos(0.07 * t) + 0.01 * rng.normal(size=n)
    theta = 0.15 * t - 3.0  # sweeps several turns, wraps repeatedly
    return np.stack([x, y, theta], axis=1)


@pytest.fixture(scope="module")
def pose2d():
    return synthetic_trajectory()


# ---------------------------------------------------------------------------
# Derived values: computed from the rate, correct at every setting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("native_fps", NATIVE_RATES)
@pytest.mark.parametrize("chunk_len,stride", GRID, ids=GRID_IDS)
def test_derived_values_match_closed_form(native_fps, chunk_len, stride):
    cfg = make_chunk_config(
        native_fps=native_fps, obs_stride_frames=stride, action_chunk_len=chunk_len
    )
    assert cfg.obs_stride_frames == stride
    assert cfg.action_chunk_len == chunk_len
    assert cfg.dt_native == pytest.approx(1.0 / native_fps)
    assert cfg.obs_interval == pytest.approx(stride / native_fps)
    assert cfg.effective_obs_hz == pytest.approx(native_fps / stride)
    assert cfg.chunk_duration == pytest.approx(chunk_len / native_fps)
    assert cfg.overlap_fraction == pytest.approx(1.0 - stride / chunk_len)
    # The stride form is exact: obs_hz is a consequence, not a request that
    # had to be quantized.
    assert cfg.obs_hz == pytest.approx(cfg.effective_obs_hz)


def test_the_two_configurations_under_discussion():
    # What the corpus is actually built at today: 30 Hz, chunk 10, stride 5.
    # 0.333s horizon and 6 Hz observations -- not the 0.5s / 4 Hz the docs
    # claimed before this change.
    now = make_chunk_config(native_fps=30.0, obs_stride_frames=5, action_chunk_len=10)
    assert now.chunk_duration == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert now.effective_obs_hz == pytest.approx(6.0)

    # And at 25 Hz, stride 5 is exactly 5 Hz -- the reason to prefer it.
    after = make_chunk_config(native_fps=25.0, obs_stride_frames=5, action_chunk_len=10)
    assert after.effective_obs_hz == pytest.approx(5.0)
    assert after.chunk_duration == pytest.approx(0.4)
    assert make_chunk_config(
        native_fps=25.0, obs_stride_frames=6, action_chunk_len=15
    ).effective_obs_hz == pytest.approx(25.0 / 6.0)


# ---------------------------------------------------------------------------
# The indexing contract itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("native_fps", NATIVE_RATES)
@pytest.mark.parametrize("chunk_len,stride", GRID, ids=GRID_IDS)
def test_chunk_covers_ticks_plus_one_through_plus_L(pose2d, native_fps, chunk_len, stride):
    cfg = make_chunk_config(
        native_fps=native_fps, obs_stride_frames=stride, action_chunk_len=chunk_len
    )
    rgb = [f"f{i}.png" for i in range(len(pose2d))]
    records = build_observation_records(pose2d, rgb, cfg, SpacingConfig(mode="fixed"))
    assert records, "grid point produced no observations"

    for rec in records:
        idx = rec["frame_index"]
        anchor = pose2d[idx]
        assert rec["obs_pose"] == pytest.approx(list(anchor))
        assert len(rec["action_chunk"]) == chunk_len
        for j, action in enumerate(rec["action_chunk"]):
            expected = pose_to_relative(anchor, pose2d[idx + j + 1])
            assert np.array_equal(np.asarray(action), expected), (
                f"chunk[{j}] is not tick +{j + 1}"
            )
        # The zeroth action is the *next* tick, never the observation itself.
        assert not np.allclose(rec["action_chunk"][0], [0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("native_fps", NATIVE_RATES)
@pytest.mark.parametrize("chunk_len,stride", GRID, ids=GRID_IDS)
def test_next_observation_decodes_from_chunk_stride_minus_one(
    pose2d, native_fps, chunk_len, stride
):
    """The off-by-one that has bitten before, stated directly."""
    cfg = make_chunk_config(
        native_fps=native_fps, obs_stride_frames=stride, action_chunk_len=chunk_len
    )
    rgb = [f"f{i}.png" for i in range(len(pose2d))]
    records = build_observation_records(pose2d, rgb, cfg, SpacingConfig(mode="fixed"))
    assert len(records) >= 2

    for cur, nxt in zip(records, records[1:]):
        assert nxt["frame_index"] - cur["frame_index"] == stride
        anchor = cur["obs_pose"]
        at_stride_minus_one = relative_to_pose(anchor, cur["action_chunk"][stride - 1])
        assert at_stride_minus_one[:2] == pytest.approx(nxt["obs_pose"][:2], abs=1e-12)

        # ...and chunk[stride] is one tick too far. (It exists here because
        # every grid point has chunk_len > stride; where it doesn't, indexing
        # it would be an IndexError, which is the same bug louder.)
        if stride < chunk_len:
            too_far = relative_to_pose(anchor, cur["action_chunk"][stride])
            assert not np.allclose(too_far[:2], nxt["obs_pose"][:2], atol=1e-6)


@pytest.mark.parametrize("native_fps", NATIVE_RATES)
@pytest.mark.parametrize("chunk_len,stride", GRID, ids=GRID_IDS)
def test_observations_land_on_the_stride(pose2d, native_fps, chunk_len, stride):
    cfg = make_chunk_config(
        native_fps=native_fps, obs_stride_frames=stride, action_chunk_len=chunk_len
    )
    rgb = [f"f{i}.png" for i in range(len(pose2d))]
    records = build_observation_records(pose2d, rgb, cfg, SpacingConfig(mode="fixed"))

    frame_indices = [r["frame_index"] for r in records]
    assert frame_indices == list(range(0, frame_indices[-1] + 1, stride))
    # Closed form for the fixed-stride case: an observation is emitted while
    # its whole chunk fits inside the trajectory.
    n = len(pose2d)
    assert len(records) == (n - 1 - chunk_len) // stride + 1
    assert frame_indices[-1] + chunk_len <= n - 1


@pytest.mark.parametrize("native_fps", NATIVE_RATES)
@pytest.mark.parametrize("chunk_len,stride", GRID, ids=GRID_IDS)
def test_round_trip_is_exact(pose2d, native_fps, chunk_len, stride):
    cfg = make_chunk_config(
        native_fps=native_fps, obs_stride_frames=stride, action_chunk_len=chunk_len
    )
    rgb = [f"f{i}.png" for i in range(len(pose2d))]
    records = build_observation_records(pose2d, rgb, cfg, SpacingConfig(mode="fixed"))

    recon = reconstruct_reference_trajectory(records)
    covered = records[-1]["frame_index"] + chunk_len + 1
    assert len(recon) == covered
    np.testing.assert_allclose(recon[:, :2], pose2d[:covered, :2], atol=1e-12)
    np.testing.assert_allclose(
        np.cos(recon[:, 2]), np.cos(pose2d[:covered, 2]), atol=1e-12
    )
    np.testing.assert_allclose(
        np.sin(recon[:, 2]), np.sin(pose2d[:covered, 2]), atol=1e-12
    )


# ---------------------------------------------------------------------------
# The boundary, and the degenerate side of it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("native_fps", NATIVE_RATES)
@pytest.mark.parametrize("stride", DEFAULT_SWEEP_STRIDES)
def test_stride_equal_to_chunk_len_is_the_exact_touch_boundary(pose2d, native_fps, stride):
    cfg = make_chunk_config(
        native_fps=native_fps, obs_stride_frames=stride, action_chunk_len=stride
    )
    assert cfg.overlap_fraction == pytest.approx(0.0)
    assert cfg.chunk_duration == pytest.approx(cfg.obs_interval)

    rgb = [f"f{i}.png" for i in range(len(pose2d))]
    records = build_observation_records(pose2d, rgb, cfg, SpacingConfig(mode="fixed"))
    # The last action of a chunk is exactly the next observation: chunks abut
    # with no overlap and no gap.
    nxt = relative_to_pose(records[0]["obs_pose"], records[0]["action_chunk"][stride - 1])
    assert nxt[:2] == pytest.approx(records[1]["obs_pose"][:2], abs=1e-12)


@pytest.mark.parametrize("native_fps", NATIVE_RATES)
@pytest.mark.parametrize("stride", DEFAULT_SWEEP_STRIDES)
def test_stride_beyond_chunk_len_is_rejected(native_fps, stride):
    # The degenerate corner: the chunk runs out before the next observation
    # arrives, so replay would have nothing to follow across the gap.
    with pytest.raises(ValueError, match="shorter than obs_interval"):
        make_chunk_config(
            native_fps=native_fps, obs_stride_frames=stride, action_chunk_len=stride - 1
        )


def test_obs_hz_and_stride_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        make_chunk_config(native_fps=25.0, obs_hz=5.0, obs_stride_frames=5, action_chunk_len=10)
    with pytest.raises(ValueError, match="exactly one"):
        make_chunk_config(native_fps=25.0, action_chunk_len=10)


def test_stride_form_and_rate_form_agree_when_the_rate_is_exact():
    by_rate = make_chunk_config(native_fps=25.0, obs_hz=5.0, action_chunk_len=12)
    by_stride = make_chunk_config(native_fps=25.0, obs_stride_frames=5, action_chunk_len=12)
    assert by_rate == by_stride


def test_the_rate_form_quantizes_where_the_stride_form_does_not():
    # 25 / 4 = 6.25 -> stride 6 -> 4.167 Hz, not the 4 Hz asked for. This is
    # why a sweep varies the stride.
    cfg = make_chunk_config(native_fps=25.0, obs_hz=4.0, action_chunk_len=10)
    assert cfg.obs_stride_frames == 6
    assert cfg.effective_obs_hz == pytest.approx(25.0 / 6.0)
    assert cfg.obs_hz == 4.0  # the request is preserved verbatim, unrealized
