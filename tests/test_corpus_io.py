"""The raw-corpus on-disk contract: the failure log round-trips, resume reads back.

These are the two places where one function writes a file and a different function
parses it. The writer and the reader used to be four functions apart in one script; now
they are one module, and this is what keeps them agreeing.

No simulator, no habitat, no images -- everything here is file IO over a tmp_path.
"""

import json

import pytest

from continuous_demos.corpus_io import (
    FAILURE_LOG_COLUMNS,
    FailureLog,
    frame_filename,
    load_done_episode_ids,
    load_failed_episode_ids,
    sanitize,
    scene_name,
)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scene_id, expected",
    [
        ("mp3d/17DRP5sb8fy.glb", "17DRP5sb8fy"),
        ("/abs/path/to/00800-TEEsavR23oF.basis.glb", "00800-TEEsavR23oF.basis"),
        ("bare", "bare"),
    ],
)
def test_scene_name_strips_directory_and_one_suffix(scene_id, expected):
    assert scene_name(scene_id) == expected


def test_sanitize_keeps_a_path_component_a_single_component():
    assert "/" not in sanitize("a/b")
    assert sanitize("ep 12:3") == "ep_12_3"
    assert sanitize("keep.this-one_ok") == "keep.this-one_ok"


def test_frame_filename_is_zero_padded_to_five():
    assert frame_filename(0, "png") == "frame_00000.png"
    assert frame_filename(42, "jpeg") == "frame_00042.jpeg"
    # Past the padding width it keeps counting rather than wrapping -- an episode
    # longer than 100k ticks would still sort correctly by name.
    assert frame_filename(123456, "png") == "frame_123456.png"


# ---------------------------------------------------------------------------
# Failure log: the writer and the reader must agree
# ---------------------------------------------------------------------------
def test_failure_log_round_trips_through_its_own_reader(tmp_path):
    path = tmp_path / "failed_episodes.log"
    with FailureLog(path, source="some/dataset.json.gz") as log:
        assert log.record(3, "ep-a", "17DRP5sb8fy", "goal: 2.31m from goal (> 1.0m)")
        assert log.record(9, "ep-b", "-", "length 1200 > max_length 900")

    assert load_failed_episode_ids(path) == {"ep-a", "ep-b"}

    header = path.read_text().splitlines()[1]
    assert header == "# columns: " + "\t".join(FAILURE_LOG_COLUMNS)


def test_failure_log_appends_across_runs_and_never_forgets(tmp_path):
    """A resumed run must not re-attempt what an earlier one rejected."""
    path = tmp_path / "failed_episodes.log"
    with FailureLog(path, source="d") as log:
        log.record(0, "ep-a", "s", "escaped: left the world")

    with FailureLog(path, source="d") as log:
        assert log.known == {"ep-a"}
        # Already known -> writes nothing and says so, which is what the caller's
        # `n_failed` accounting keys on.
        assert log.record(0, "ep-a", "s", "escaped: left the world") is False
        assert log.record(1, "ep-b", "s", "not_reached: 1.2m") is True

    assert load_failed_episode_ids(path) == {"ep-a", "ep-b"}
    # Exactly one header, even though two runs opened it.
    assert path.read_text().count("# columns:") == 1
    assert len([l for l in path.read_text().splitlines() if not l.startswith("#")]) == 2


def test_failure_log_reason_leads_with_a_greppable_code(tmp_path):
    """`cut -d: -f1` over the reason column must give a histogram."""
    path = tmp_path / "log"
    with FailureLog(path, source="d") as log:
        log.record(0, "a", "s", "inefficient: segment 2 spl 0.41 < 0.55")
        log.record(1, "b", "s", "inefficient: segment 0 spl 0.30 < 0.55")
        log.record(2, "c", "s", "escaped: joint_z free-falling for 2.10s")

    codes = [
        line.split("\t")[3].split(":")[0]
        for line in path.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert codes == ["inefficient", "inefficient", "escaped"]


def test_failure_log_survives_a_reason_containing_a_colon(tmp_path):
    """Tab-separated, so prose with colons or spaces cannot shift a column."""
    path = tmp_path / "log"
    with FailureLog(path, source="d") as log:
        log.record(7, "ep:with:colons", "sc", "replay error: ValueError: bad: thing")

    assert load_failed_episode_ids(path) == {"ep:with:colons"}
    row = [l for l in path.read_text().splitlines() if not l.startswith("#")][0]
    idx, ep, scene, reason = row.split("\t")
    assert (idx, ep, scene) == ("7", "ep:with:colons", "sc")
    assert reason == "replay error: ValueError: bad: thing"


def test_load_failed_ids_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert load_failed_episode_ids(tmp_path / "nope.log") == set()


# ---------------------------------------------------------------------------
# Resume: the shard reader
# ---------------------------------------------------------------------------
def test_load_done_reads_only_the_matching_shard(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "sceneA.jsonl").write_text(
        json.dumps({"episode_id": "a1"}) + "\n" + json.dumps({"episode_id": "a2"}) + "\n"
    )
    (meta / "sceneB.jsonl").write_text(json.dumps({"episode_id": "b1"}) + "\n")

    assert load_done_episode_ids(meta, "sceneA") == {"a1", "a2"}
    assert load_done_episode_ids(meta, "sceneB") == {"b1"}
    assert load_done_episode_ids(meta, "sceneMissing") == set()


def test_load_done_tolerates_a_torn_final_line(tmp_path):
    """A run killed mid-write must leave a resumable shard, not an unreadable one.

    Refusing to parse would make the whole corpus un-resumable to save one episode
    that is simply regenerated.
    """
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "s.jsonl").write_text(
        json.dumps({"episode_id": "ok1"}) + "\n"
        + json.dumps({"episode_id": "ok2"}) + "\n"
        + '{"episode_id": "half-writ'
    )
    assert load_done_episode_ids(meta, "s") == {"ok1", "ok2"}


def test_load_done_skips_blank_lines_and_records_without_an_id(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "s.jsonl").write_text(
        json.dumps({"episode_id": "a"}) + "\n\n" + json.dumps({"scene_id": "x"}) + "\n"
    )
    assert load_done_episode_ids(meta, "s") == {"a"}
