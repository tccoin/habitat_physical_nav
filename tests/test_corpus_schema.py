"""The guards that stop one corpus becoming two generations in a trench coat.

Schema version 2 records ``num_cameras`` and ``phase_free`` because both change
what a record MEANS -- one makes every frame a strip of N views, the other
changes what the trajectories are -- and neither is detectable downstream once
records are interleaved. These tests pin the three refusals that follow from
that (per-shard, per-directory, per-consumer), and the additive-ness that lets
a version-1 shard still be read and still be appended to.

No simulator needed: all of it is metadata and paths.
"""

import json

import pytest

from continuous_demos.corpus_guard import (
    CORPUS_MARKER_NAME,
    UnsafeOutputDir,
    assert_corpus_marker_agrees,
    assert_output_dir_is_safe,
    assert_shard_is_appendable,
    read_corpus_marker,
)
from continuous_demos.recording_metadata import (
    RECORDING_KEY,
    SCHEMA_VERSION,
    MultiViewCorpusError,
    RecordingParams,
    format_disagreements,
    params_for_generation,
    params_from_record,
    resolve_corpus_params,
)


def make_params(num_cameras=1, phase_free=False, native_fps=25.0, width=640, task=None):
    return params_for_generation(
        native_fps,
        image_width=width,
        image_height=480,
        image_ext="jpeg",
        num_cameras=num_cameras,
        phase_free=phase_free,
        physics_timestep=0.008,
        task=task,
    )


def write_shard(path, params, n=2):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "episode_id": f"ep{i}",
                "pose2d": [[0.0, 0.0, 0.0]],
                "rgb_paths": ["a.jpeg"],
                RECORDING_KEY: params.as_dict(),
            }) + "\n")


# ---------------------------------------------------------------------------
# the schema is additive: a version-1 shard still reads, with no migration
# ---------------------------------------------------------------------------
def test_schema_version_is_three():
    assert SCHEMA_VERSION == 3


def test_a_version_one_record_reads_as_single_camera_segmented():
    # Exactly the keys the old generator wrote -- no num_cameras, no phase_free.
    v1 = {
        "native_fps": 25.0, "dt_native": 0.04, "physics_timestep": 0.008,
        "physics_steps_per_tick": 5.0, "on_physics_grid": True,
        "v_max_mps": 2.0, "w_max_radps": 2.0, "image_width": 640,
        "image_height": 480, "image_ext": "jpeg", "schema_version": 1,
    }
    params = params_from_record({RECORDING_KEY: v1})
    assert params.schema_version == 1
    assert params.num_cameras == 1          # the default IS what v1 meant
    assert params.phase_free is False
    assert not params.is_multiview


def test_a_version_one_shard_is_still_appendable_by_a_default_run(tmp_path):
    # No migration step: the new defaults reproduce v1's meaning exactly, so a
    # default run may resume an old shard. Only a CHANGED knob is refused.
    v1 = RecordingParams(
        native_fps=25.0, dt_native=0.04, physics_timestep=0.008,
        physics_steps_per_tick=5.0, on_physics_grid=True,
        v_max_mps=2.0, w_max_radps=2.0, schema_version=1,
    )
    shard = tmp_path / "scene.jsonl"
    write_shard(shard, v1)
    assert_shard_is_appendable(shard, make_params())  # does not raise


def test_new_descriptive_fields_are_absent_for_a_single_camera_run():
    # So a modern single-camera record differs from a v1 one only in
    # schema_version, and nothing downstream sees a new key it must handle.
    blob = make_params().as_dict()
    assert "view_strip_order" not in blob
    assert "view_yaw_offsets_deg" not in blob
    assert blob["num_cameras"] == 1
    assert blob["phase_free"] is False


def test_multiview_params_round_trip_through_json():
    params = make_params(num_cameras=5, phase_free=True)
    restored = params_from_record(
        {RECORDING_KEY: json.loads(json.dumps(params.as_dict()))}
    )
    assert restored == params           # tuples survive the list round-trip
    assert restored.agrees_with(params)
    assert restored.view_strip_order == (3, 4, 0, 1, 2)
    assert restored.frame_width == 3200
    assert restored.image_width == 640  # still PER-CAMERA
    assert [round(y) for y in restored.view_yaw_offsets_deg] == [0, -72, -144, 144, 72]


def test_view_slice_cuts_the_strip_into_equal_views():
    params = make_params(num_cameras=5)
    assert [params.view_slice(i) for i in range(5)] == [
        (0, 640), (640, 1280), (1280, 1920), (1920, 2560), (2560, 3200)
    ]
    with pytest.raises(IndexError):
        params.view_slice(5)


def test_view_slice_refuses_on_a_single_camera_corpus():
    with pytest.raises(ValueError):
        make_params().view_slice(0)


# ---------------------------------------------------------------------------
# refusal 1: the per-shard appendability guard
# ---------------------------------------------------------------------------
def test_matching_params_are_appendable(tmp_path):
    shard = tmp_path / "scene.jsonl"
    write_shard(shard, make_params(num_cameras=5, phase_free=True))
    assert_shard_is_appendable(shard, make_params(num_cameras=5, phase_free=True))


def test_a_missing_shard_is_appendable(tmp_path):
    assert_shard_is_appendable(tmp_path / "nothing.jsonl", make_params())


@pytest.mark.parametrize("changed", [
    {"num_cameras": 5},
    {"phase_free": True},
    {"native_fps": 30.0},
])
def test_a_changed_knob_is_refused(tmp_path, changed):
    shard = tmp_path / "scene.jsonl"
    write_shard(shard, make_params())
    with pytest.raises(RuntimeError, match="REFUSING TO RESUME"):
        assert_shard_is_appendable(shard, make_params(**changed))


def test_the_refusal_names_the_field_and_both_values(tmp_path):
    shard = tmp_path / "scene.jsonl"
    write_shard(shard, make_params(num_cameras=1, phase_free=False))
    with pytest.raises(RuntimeError) as excinfo:
        assert_shard_is_appendable(shard, make_params(num_cameras=5, phase_free=True))
    message = str(excinfo.value)
    # the field names...
    assert "num_cameras" in message and "phase_free" in message
    # ...and both sides of each, not just "they differ"
    assert "already in shard" in message and "this run" in message
    assert "1" in message and "5" in message
    assert "True" in message and "False" in message


def test_a_preschema_shard_is_refused(tmp_path):
    shard = tmp_path / "scene.jsonl"
    with open(shard, "w", encoding="utf-8") as f:
        f.write(json.dumps({"episode_id": "old", "pose2d": [[0, 0, 0]]}) + "\n")
    with pytest.raises(RuntimeError, match="predates the schema"):
        assert_shard_is_appendable(shard, make_params())


def test_format_disagreements_is_one_line_per_field():
    diffs = make_params().disagreements(make_params(num_cameras=5, phase_free=True))
    rendered = format_disagreements(diffs)
    assert len(rendered.splitlines()) == len(diffs) == 2


# ---------------------------------------------------------------------------
# refusal 2: the per-directory corpus marker
# ---------------------------------------------------------------------------
def test_a_fresh_directory_gets_a_marker(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    params = make_params(num_cameras=5, phase_free=True)
    assert_corpus_marker_agrees(tmp_path, meta, params)
    assert (tmp_path / CORPUS_MARKER_NAME).exists()
    assert RecordingParams.from_dict(read_corpus_marker(tmp_path)) == params


def test_resuming_the_same_configuration_is_allowed(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    assert_corpus_marker_agrees(tmp_path, meta, make_params(num_cameras=5))
    assert_corpus_marker_agrees(tmp_path, meta, make_params(num_cameras=5))


def test_resuming_with_a_different_configuration_is_refused(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    assert_corpus_marker_agrees(tmp_path, meta, make_params(num_cameras=1))
    with pytest.raises(UnsafeOutputDir) as excinfo:
        assert_corpus_marker_agrees(tmp_path, meta, make_params(num_cameras=5))
    assert "num_cameras" in str(excinfo.value)


def test_an_existing_corpus_without_a_marker_is_refused(tmp_path):
    # The case the per-shard check cannot see: a NEW run pointed at a FINISHED
    # corpus whose parameters happen to match. Every corpus on disk today has
    # no marker, so this fails closed for all of them.
    meta = tmp_path / "metadata"
    meta.mkdir()
    write_shard(meta / "scene.jsonl", make_params())
    with pytest.raises(UnsafeOutputDir, match="REFUSING TO WRITE"):
        assert_corpus_marker_agrees(tmp_path, meta, make_params())


def test_adopting_an_existing_corpus_is_possible_but_explicit(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    write_shard(meta / "scene.jsonl", make_params())
    assert_corpus_marker_agrees(tmp_path, meta, make_params(), adopt_existing_corpus=True)
    assert (tmp_path / CORPUS_MARKER_NAME).exists()


# ---------------------------------------------------------------------------
# refusal 3: writing into the input, or into a protected root
# ---------------------------------------------------------------------------
def test_writing_beside_the_input_dataset_is_fine(tmp_path):
    (tmp_path / "in").mkdir()
    dataset = tmp_path / "in" / "scene.json.gz"
    dataset.touch()
    assert_output_dir_is_safe(tmp_path / "out", dataset)


def test_writing_into_the_input_directory_is_refused(tmp_path):
    source = tmp_path / "shards"
    source.mkdir()
    dataset = source / "scene.json.gz"
    dataset.touch()
    with pytest.raises(UnsafeOutputDir, match="overlaps"):
        assert_output_dir_is_safe(source, dataset)
    with pytest.raises(UnsafeOutputDir, match="overlaps"):
        assert_output_dir_is_safe(source / "generated", dataset)


def test_an_output_that_contains_the_input_directory_is_refused(tmp_path):
    source = tmp_path / "a" / "b"
    source.mkdir(parents=True)
    dataset = source / "scene.json.gz"
    dataset.touch()
    with pytest.raises(UnsafeOutputDir, match="overlaps"):
        assert_output_dir_is_safe(tmp_path / "a", dataset)


def test_the_image_root_is_checked_too(tmp_path):
    source = tmp_path / "shards"
    source.mkdir()
    dataset = source / "scene.json.gz"
    dataset.touch()
    with pytest.raises(UnsafeOutputDir, match="--image-root"):
        assert_output_dir_is_safe(tmp_path / "out", dataset, image_root=source / "imgs")


def test_a_protected_root_is_refused(tmp_path, monkeypatch):
    protected = tmp_path / "corpus"
    protected.mkdir()
    # The input lives in its own directory, so the overlap check below is not
    # what is being tested here.
    (tmp_path / "in").mkdir()
    dataset = tmp_path / "in" / "shard.json.gz"
    dataset.touch()
    monkeypatch.setenv("HABITAT_PROTECTED_ROOTS", str(protected))
    with pytest.raises(UnsafeOutputDir, match="protected root"):
        assert_output_dir_is_safe(protected / "new_run", dataset)
    assert_output_dir_is_safe(tmp_path / "elsewhere", dataset)  # not protected


# ---------------------------------------------------------------------------
# refusal 4: a single-view consumer must not read a strip as one image
# ---------------------------------------------------------------------------
def test_a_multiview_corpus_is_refused_by_default(tmp_path):
    write_shard(tmp_path / "scene.jsonl", make_params(num_cameras=5))
    with pytest.raises(MultiViewCorpusError) as excinfo:
        resolve_corpus_params(tmp_path)
    message = str(excinfo.value)
    assert "5 cameras" in message           # says what it is
    assert "3200" in message                # ...and how wide a frame really is
    assert "allow_multiview=True" in message  # ...and how to opt in


def test_a_multiview_corpus_is_readable_once_declared(tmp_path):
    write_shard(tmp_path / "scene.jsonl", make_params(num_cameras=5, phase_free=True))
    resolved = resolve_corpus_params(tmp_path, allow_multiview=True)
    assert resolved.params.num_cameras == 5
    assert resolved.params.phase_free is True
    assert resolved.params.native_fps == pytest.approx(25.0)


def test_a_single_camera_corpus_needs_no_declaration(tmp_path):
    write_shard(tmp_path / "scene.jsonl", make_params())
    assert resolve_corpus_params(tmp_path).params.num_cameras == 1


def test_shards_that_disagree_on_the_rig_are_refused(tmp_path):
    write_shard(tmp_path / "a.jsonl", make_params(num_cameras=1))
    write_shard(tmp_path / "b.jsonl", make_params(num_cameras=5))
    with pytest.raises(Exception) as excinfo:
        resolve_corpus_params(tmp_path, allow_multiview=True)
    assert "num_cameras" in str(excinfo.value)


def test_describe_rig_reads_as_english():
    assert make_params().describe_rig() == "objectnav, 1 camera, segmented"
    described = make_params(num_cameras=5, phase_free=True).describe_rig()
    assert "5 cameras" in described and "3200px strip" in described
    assert "phase-free" in described
    assert described.startswith("objectnav, ")
    assert make_params(task="pointnav").describe_rig() == "pointnav, 1 camera, segmented"


# ---------------------------------------------------------------------------
# `task`: None MEANS objectnav, and that is what keeps the existing corpora safe
# ---------------------------------------------------------------------------
# Schema 3 adds `task` to CONSISTENCY_KEYS so a PointNav corpus cannot be poured
# into an ObjectNav one. The whole design rests on ObjectNav continuing to stamp
# `None` rather than the string "objectnav": every corpus written before this
# field existed reads back as None, and `_values_agree(None, None)` is True, so
# they stay appendable by their own generator. Stamping the string instead would
# make the entire v2_25hz lineage unappendable overnight. These pin that.
def test_an_objectnav_run_writes_no_task_key_at_all(tmp_path):
    """`as_dict` drops None, so a new ObjectNav record is byte-identical to an old one."""
    assert "task" not in make_params().as_dict()
    assert make_params().task is None
    assert make_params().task_name == "objectnav"


def test_a_pre_task_record_reads_back_as_objectnav():
    v2 = {
        "native_fps": 25.0, "dt_native": 0.04, "physics_timestep": 0.008,
        "physics_steps_per_tick": 5.0, "on_physics_grid": True,
        "v_max_mps": 2.0, "w_max_radps": 2.0, "num_cameras": 1,
        "phase_free": False, "schema_version": 2,
    }
    params = params_from_record({RECORDING_KEY: v2})
    assert params.task is None
    assert params.task_name == "objectnav"


def test_a_pre_task_shard_is_still_appendable_by_a_default_run(tmp_path):
    """The v2_25hz-lineage guarantee. If this breaks, every shipped corpus is frozen."""
    shard = tmp_path / "scene.jsonl"
    legacy = RecordingParams(
        native_fps=25.0, dt_native=0.04, physics_timestep=0.008,
        physics_steps_per_tick=5.0, on_physics_grid=True,
        v_max_mps=2.0, w_max_radps=2.0, schema_version=2,
    )
    write_shard(shard, legacy)
    assert "task" not in legacy.as_dict()
    assert_shard_is_appendable(shard, make_params())  # must not raise


def test_a_pointnav_run_cannot_append_to_an_objectnav_shard(tmp_path):
    shard = tmp_path / "scene.jsonl"
    write_shard(shard, make_params())
    with pytest.raises(RuntimeError) as excinfo:
        assert_shard_is_appendable(shard, make_params(task="pointnav"))
    assert "task" in str(excinfo.value)


def test_an_objectnav_run_cannot_append_to_a_pointnav_shard(tmp_path):
    shard = tmp_path / "scene.jsonl"
    write_shard(shard, make_params(task="pointnav"))
    with pytest.raises(RuntimeError) as excinfo:
        assert_shard_is_appendable(shard, make_params())
    assert "task" in str(excinfo.value)


def test_a_pointnav_run_cannot_adopt_an_objectnav_output_dir(tmp_path):
    """The per-directory refusal, not just the per-shard one."""
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_shard(metadata / "scene.jsonl", make_params())
    assert_corpus_marker_agrees(tmp_path, metadata, make_params(), True)
    with pytest.raises(UnsafeOutputDir) as excinfo:
        assert_corpus_marker_agrees(tmp_path, metadata, make_params(task="pointnav"))
    assert "task" in str(excinfo.value)


def test_two_pointnav_runs_agree_with_each_other(tmp_path):
    shard = tmp_path / "scene.jsonl"
    write_shard(shard, make_params(task="pointnav"))
    assert_shard_is_appendable(shard, make_params(task="pointnav"))  # must not raise
    assert make_params(task="pointnav").as_dict()["task"] == "pointnav"


def test_shards_that_disagree_on_the_task_are_refused_at_read_time(tmp_path):
    write_shard(tmp_path / "a.jsonl", make_params())
    write_shard(tmp_path / "b.jsonl", make_params(task="pointnav"))
    with pytest.raises(Exception) as excinfo:
        resolve_corpus_params(tmp_path)
    assert "task" in str(excinfo.value)
