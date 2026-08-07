"""The recorder's driving limits are named, and they agree with what gets stamped.

`replay.py` inlined these four values; a second recorder
(`continuous_demos.pointnav_episode`) now drives the same controllers and imports them
rather than repeating them. That only buys anything if the names keep their values, and
if the one that is *also* written into every corpus's metadata stays equal to the one
actually commanded -- a corpus whose recorded `w_max_radps` is not the limit it was
driven at is wrong in a way nothing downstream can detect.

Same guard style as `test_recording_metadata.py::test_mirrored_controller_limits_
still_match_source`. Pure imports; no simulator.
"""

from continuous_demos import recording_metadata, replay


def test_move_max_vw_is_what_the_corpus_records():
    """The stamped rotational limit must be the one the move phase commands.

    `recording_metadata.REPLAY_MOVE_MAX_VW_RADPS` is copied into `w_max_radps` on every
    record. If someone retunes the recorder without carrying it across, every corpus
    written afterwards claims a limit it was never driven at, and the arrow builder
    stamps that claim onto every row.
    """
    assert replay.MOVE_MAX_VW == recording_metadata.REPLAY_MOVE_MAX_VW_RADPS


def test_the_named_constants_still_hold_their_calibrated_values():
    """Pins the values the extraction preserved.

    Each of these was measured, not chosen (see the comments in `replay.py` and
    `dump/data_diagnostics/rotation_tuning/FINDINGS.md`). This test is not asserting
    that they can never change -- it is asserting that changing one is a deliberate act
    with a measurement behind it, rather than a side effect of an unrelated edit.
    """
    assert replay.MOVE_STALL_TICKS == 45
    assert replay.MOVE_STALL_EPS == 0.05
    assert replay.MOVE_STALL_MAX_SPEED == 0.02
    assert replay.PHASE_FREE_STALL_TICKS == 0
    assert replay.RECORD_FPS == 25
    assert replay.MOVE_HEADING_GAIN == 3.0
    assert replay.MOVE_MAX_VW == 2.0
    assert replay.LOCAL_TIMEOUT_MPS == 0.5
    assert replay.TIPPED_OVER_DEG == 45.0


def test_record_fps_lands_on_the_physics_grid():
    """25 Hz is the rate because it is 5 whole physics steps, not because it is round."""
    steps = (1.0 / replay.RECORD_FPS) / recording_metadata.DEFAULT_PHYSICS_TIMESTEP
    assert abs(steps - round(steps)) < 1e-9


def test_is_stalled_is_the_public_name_for_the_same_predicate():
    assert replay.is_stalled is replay._is_stalled


class TestIsStalled:
    """The rule itself, so a second recorder importing it knows what it bought."""

    def test_disabled_by_a_zero_window(self):
        assert replay.is_stalled([0.0] * 100, [0.0] * 100, 0, 0.05, 0.02) is False

    def test_false_until_the_window_has_filled(self):
        assert replay.is_stalled([0.0] * 10, [0.0] * 10, 45, 0.05, 0.02) is False

    def test_progress_along_the_path_clears_it(self):
        progress = [0.01 * i for i in range(100)]  # 1 cm/tick, well over eps in 45
        assert replay.is_stalled(progress, [0.0] * 100, 45, 0.05, 0.02) is False

    def test_the_speed_gate_is_the_load_bearing_part(self):
        """A robot crawling at the controller's minimum approach speed is not stalled.

        Same frozen along-path progress in both cases; only the speed differs. Without
        the gate this rule truncates a slow legitimate approach.
        """
        frozen = [1.0] * 100
        assert replay.is_stalled(frozen, [0.0] * 100, 45, 0.05, 0.02) is True
        assert replay.is_stalled(frozen, [0.1] * 100, 45, 0.05, 0.02) is False

    def test_without_a_speed_gate_frozen_progress_alone_is_enough(self):
        assert replay.is_stalled([1.0] * 100, [0.1] * 100, 45, 0.05, None) is True

    def test_only_the_last_window_of_speeds_counts(self):
        """Moving fast 60 ticks ago does not excuse being stopped now."""
        speeds = [1.0] * 50 + [0.0] * 50
        assert replay.is_stalled([1.0] * 100, speeds, 45, 0.05, 0.02) is True
