"""Invariant checks on continuous_demos.recording_metadata -- no habitat_sim needed.

The point of the module under test is that a corpus stops being ambiguous
about how it was recorded. These tests pin the three cases that ambiguity used
to hide in: parameters present, parameters absent, and parameters that
disagree with themselves.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from continuous_demos.recording_metadata import (
    DEFAULT_V_MAX_MPS,
    LEGACY_NATIVE_FPS,
    RECORDING_KEY,
    REPLAY_MOVE_MAX_VW_RADPS,
    SCHEMA_VERSION,
    InconsistentRecordingParams,
    MissingRecordingParams,
    RecordingParams,
    check_record_consistency,
    legacy_params,
    params_for_generation,
    params_from_record,
    resolve_corpus_params,
    resolve_velocity_limits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "continuous_demos"

# A rotation config stand-in, so these tests never import controllers.py (and
# through it habitat_sim). Its w_max matches HeadingRotationConfig's default,
# which test_mirrored_controller_limits_still_match_source pins against source.
FAKE_ROTATION_CONFIG = SimpleNamespace(w_max=2.0)


def make_params(fps: float = 25.0, **kwargs) -> RecordingParams:
    return params_for_generation(
        fps, physics_timestep=0.008, rotation_config=FAKE_ROTATION_CONFIG, **kwargs
    )


def write_shard(path: Path, params: RecordingParams = None, n: int = 2) -> Path:
    """Write a minimal JSONL shard, with or without a recording block."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            record = {"episode_id": f"ep{i}", "pose2d": [[0.0, 0.0, 0.0]], "rgb_paths": ["a.png"]}
            if params is not None:
                record[RECORDING_KEY] = params.as_dict()
            f.write(json.dumps(record) + "\n")
    return path


# ---------------------------------------------------------------------------
# Building the block
# ---------------------------------------------------------------------------
def test_25hz_is_on_the_physics_grid_and_30hz_is_not():
    on = make_params(25.0)
    assert on.dt_native == pytest.approx(0.04)
    assert on.physics_steps_per_tick == pytest.approx(5.0)
    assert on.on_physics_grid

    off = make_params(30.0)
    assert off.physics_steps_per_tick == pytest.approx(1 / 30 / 0.008)
    assert not off.on_physics_grid


def test_velocity_caps_are_the_true_2_0_pair():
    params = make_params()
    assert params.v_max_mps == pytest.approx(2.0)
    assert params.w_max_radps == pytest.approx(2.0)


def test_angular_cap_takes_the_tighter_of_the_two_phases():
    # The move phase's hardcoded max_vw binds even though the pure-pursuit
    # config's own w_max is the inert 70 rad/s.
    v_max, w_max = resolve_velocity_limits(
        pure_pursuit_config=SimpleNamespace(v_max=2.0, w_max=70.0),
        rotation_config=SimpleNamespace(w_max=99.0),
    )
    assert v_max == pytest.approx(2.0)
    assert w_max == pytest.approx(REPLAY_MOVE_MAX_VW_RADPS)

    # ...and a tighter look-phase cap would bind instead.
    _v, w_max_tight = resolve_velocity_limits(rotation_config=SimpleNamespace(w_max=0.5))
    assert w_max_tight == pytest.approx(0.5)


def test_round_trips_through_json():
    params = make_params(25.0, image_width=640, image_height=480, image_ext="jpeg")
    restored = params_from_record({RECORDING_KEY: json.loads(json.dumps(params.as_dict()))})
    assert restored == params
    assert restored.schema_version == SCHEMA_VERSION


def test_unknown_future_fields_are_tolerated():
    blob = make_params().as_dict()
    blob["something_a_later_schema_added"] = 7
    assert params_from_record({RECORDING_KEY: blob}).native_fps == pytest.approx(25.0)


def test_optional_nones_are_not_written():
    assert "image_ext" not in make_params().as_dict()


# ---------------------------------------------------------------------------
# Case 1: the corpus records its parameters
# ---------------------------------------------------------------------------
def test_new_corpus_needs_no_flag(tmp_path):
    write_shard(tmp_path / "sceneA.jsonl", make_params(25.0))
    write_shard(tmp_path / "sceneB.jsonl", make_params(25.0))
    resolved = resolve_corpus_params(tmp_path)
    assert resolved.from_metadata
    assert resolved.params.native_fps == pytest.approx(25.0)
    assert resolved.params.dt_native == pytest.approx(0.04)


def test_agreeing_flag_is_accepted_and_disagreeing_flag_is_an_error(tmp_path):
    write_shard(tmp_path / "scene.jsonl", make_params(25.0))
    assert resolve_corpus_params(tmp_path, assumed_native_fps=25.0).from_metadata
    with pytest.raises(InconsistentRecordingParams, match="authoritative"):
        resolve_corpus_params(tmp_path, assumed_native_fps=30.0)


# ---------------------------------------------------------------------------
# Case 2: pre-schema corpus
# ---------------------------------------------------------------------------
def test_pre_schema_corpus_fails_loudly_without_a_rate(tmp_path):
    write_shard(tmp_path / "scene.jsonl", None)
    with pytest.raises(MissingRecordingParams) as excinfo:
        resolve_corpus_params(tmp_path)
    # The error must name the documented fallback, not just complain.
    assert "30" in str(excinfo.value)


def test_pre_schema_corpus_accepts_an_explicit_rate_and_marks_it_assumed(tmp_path):
    write_shard(tmp_path / "scene.jsonl", None)
    resolved = resolve_corpus_params(tmp_path, assumed_native_fps=LEGACY_NATIVE_FPS)
    assert not resolved.from_metadata
    assert resolved.params.native_fps == pytest.approx(30.0)
    assert resolved.params.schema_version == 0
    assert resolved.params.assumed_note
    assert "ASSUMED" in resolved.banner()


def test_legacy_params_are_flagged_off_grid():
    assert not legacy_params(30.0).on_physics_grid
    assert legacy_params(25.0).on_physics_grid


def test_empty_metadata_dir_is_an_error(tmp_path):
    with pytest.raises(Exception):
        resolve_corpus_params(tmp_path)


# ---------------------------------------------------------------------------
# Case 3: the corpus disagrees with itself
# ---------------------------------------------------------------------------
def test_mixed_pre_and_post_schema_shards_are_rejected(tmp_path):
    write_shard(tmp_path / "old.jsonl", None)
    write_shard(tmp_path / "new.jsonl", make_params(25.0))
    with pytest.raises(InconsistentRecordingParams, match="mixes"):
        resolve_corpus_params(tmp_path)


def test_shards_recorded_at_different_rates_are_rejected(tmp_path):
    write_shard(tmp_path / "a.jsonl", make_params(25.0))
    write_shard(tmp_path / "b.jsonl", make_params(30.0))
    with pytest.raises(InconsistentRecordingParams, match="disagree"):
        resolve_corpus_params(tmp_path)


def test_appended_shard_is_caught_per_record(tmp_path):
    # First record honest, later record from a resumed run at another rate --
    # invisible to a first-line-only resolve, caught by the per-record check.
    path = tmp_path / "scene.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"episode_id": "a", RECORDING_KEY: make_params(25.0).as_dict()}) + "\n")
        f.write(json.dumps({"episode_id": "b", RECORDING_KEY: make_params(30.0).as_dict()}) + "\n")

    resolved = resolve_corpus_params(tmp_path)
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    check_record_consistency(records[0], resolved, "record 0")
    with pytest.raises(InconsistentRecordingParams, match="mixes two generations"):
        check_record_consistency(records[1], resolved, "record 1")


def test_record_without_params_in_a_schema_corpus_is_caught(tmp_path):
    write_shard(tmp_path / "scene.jsonl", make_params(25.0))
    resolved = resolve_corpus_params(tmp_path)
    with pytest.raises(InconsistentRecordingParams, match="appended"):
        check_record_consistency({"episode_id": "x"}, resolved, "record x")


def test_record_with_params_in_an_assumed_corpus_is_caught(tmp_path):
    write_shard(tmp_path / "scene.jsonl", None)
    resolved = resolve_corpus_params(tmp_path, assumed_native_fps=30.0)
    with pytest.raises(InconsistentRecordingParams, match="appended"):
        check_record_consistency(
            {"episode_id": "x", RECORDING_KEY: make_params(25.0).as_dict()}, resolved, "record x"
        )


def test_provenance_differences_do_not_trip_consistency(tmp_path):
    # A resumed run on a newer commit, or at a different render size, is fine;
    # only the physics must match.
    a = make_params(25.0, image_width=640)
    b = make_params(25.0, image_width=160)
    assert a.agrees_with(b)


# ---------------------------------------------------------------------------
# Drift guards for the two constants that cannot be imported
# ---------------------------------------------------------------------------
def test_mirrored_controller_limits_still_match_source():
    """DEFAULT_V_MAX_MPS / REPLAY_MOVE_MAX_VW_RADPS mirror values that live as
    literals inside functions in files this task does not own, so they cannot
    be imported. If those literals move, this test is the alarm."""
    controllers_src = (SRC / "controllers.py").read_text(encoding="utf-8")
    replay_src = (SRC / "replay.py").read_text(encoding="utf-8")

    # A float literal spelled either "2" or "2.0".
    def literal(value: float) -> str:
        return re.escape(f"{value:g}") + r"(?:\.0+)?\b"

    # HolonomicRobotController's inline default config.
    assert re.search(r"\bv_max=%s" % literal(DEFAULT_V_MAX_MPS), controllers_src), (
        "controllers.py no longer sets v_max=%g; update DEFAULT_V_MAX_MPS." % DEFAULT_V_MAX_MPS
    )
    # HeadingRotationConfig's own cap, read live by resolve_velocity_limits.
    assert re.search(r"w_max: float = %s" % literal(FAKE_ROTATION_CONFIG.w_max), controllers_src)
    # The move phase's explicit max_vw in record_segmented_demo.
    assert re.search(
        r"\bmax_vw=%s" % literal(REPLAY_MOVE_MAX_VW_RADPS), replay_src
    ), "replay.py no longer passes max_vw=%g; update REPLAY_MOVE_MAX_VW_RADPS." % (
        REPLAY_MOVE_MAX_VW_RADPS
    )
