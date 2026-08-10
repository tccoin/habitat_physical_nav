"""``--anchor-offset`` must survive every hop to the policy server, or mean nothing.

This exists because it already failed once. The flag was defined on
``scripts/pointnav_rollout.py``, threaded through ``_PoseFeed``, the four backends and
``scripts/policy_server.py`` -- and then *not* copied in ``build_eval_config``, which
builds the ``EvalConfig`` that ``policies.build_policy`` actually reads. Every condition of
a 216-run anchor sweep therefore ran with the baseline frame, and nothing anywhere raised:
the runs completed, wrote metrics, and would have supported the confident conclusion that
moving the anchor has no effect.

That is the exact failure this package is organised against -- a plausible number rather
than an error -- so the flag gets a test at the seam that dropped it, plus the two either
side of it.
"""

import pytest

from objectnav_eval.config import EvalConfig
from objectnav_eval.policies import _stop_head_args as _pose_args


def test_eval_config_carries_the_offset():
    assert EvalConfig().anchor_offset is None
    assert EvalConfig(anchor_offset="30,0").anchor_offset == "30,0"


def test_pose_args_forwards_it_to_the_backend():
    assert "anchor_offset" not in _pose_args(EvalConfig()), "absent unless asked for"
    args = _pose_args(EvalConfig(anchor_offset="30,0,1.5"))
    assert args["anchor_offset"] == "30,0,1.5"


def test_the_rollout_script_copies_it_into_the_eval_config():
    """The hop that was missing. Parsing a flag and not forwarding it is a silent no-op."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "pointnav_rollout",
        Path(__file__).resolve().parents[1] / "scripts" / "pointnav_rollout.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import argparse
    import inspect

    # Every attribute `build_eval_config` reads, so this fails on a *missing copy* rather
    # than on an unrelated AttributeError.
    source = inspect.getsource(module.build_eval_config)
    fields = {
        line.split("=args.")[1].strip().rstrip(",")
        for line in source.splitlines() if "=args." in line
    }
    assert "anchor_offset" in fields, (
        "build_eval_config does not copy anchor_offset into EvalConfig, so the flag is "
        "parsed and silently discarded -- every condition would run the baseline frame"
    )
    args = argparse.Namespace(**{f: None for f in fields})
    args.anchor_offset = "30,0"
    args.output = "/tmp/x.mp4"
    assert module.build_eval_config(args).anchor_offset == "30,0"


def test_the_server_command_line_actually_contains_the_flag():
    """End of the chain: the dict `PolicyBridge` turns into argv."""
    args = _pose_args(EvalConfig(anchor_offset="60,0"))
    flags = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        flags += [flag] if value is True else [flag, str(value)]
    assert "--anchor-offset" in flags
    assert flags[flags.index("--anchor-offset") + 1] == "60,0"


# ---------------------------------------------------------------------------
# The row-count guard, which `anchor_offset` has to be discounted from
# ---------------------------------------------------------------------------
class _Stream:
    def __init__(self, n):
        self.rows = list(range(n))


class _Policy:
    def __init__(self, n):
        self._raw_poses = list(range(n))


def guard(feed_offset, policy_rows, stream_rows):
    """The arithmetic from `_PoseFeed.feed`, isolated."""
    seeded = 1 if feed_offset is not None else 0
    accumulated = len(_Policy(policy_rows)._raw_poses) - seeded
    return accumulated == len(_Stream(stream_rows).rows)


def test_the_guard_passes_when_the_seeded_anchor_is_the_only_difference():
    assert guard([60.0, 0.0], policy_rows=4, stream_rows=3)


def test_the_guard_still_fires_on_a_real_mismatch_with_an_offset_set():
    """The discount must not become a blanket off-by-one amnesty."""
    assert not guard([60.0, 0.0], policy_rows=5, stream_rows=3)
    assert not guard([60.0, 0.0], policy_rows=3, stream_rows=3)


def test_the_guard_is_unchanged_when_no_offset_is_set():
    assert guard(None, policy_rows=3, stream_rows=3)
    assert not guard(None, policy_rows=4, stream_rows=3)


# ---------------------------------------------------------------------------
# Negative values must survive the hop into the server's argv
# ---------------------------------------------------------------------------
def build_command(backend_args):
    """The argv construction from `PolicyBridge.__init__`, isolated."""
    command = ["python", "policy_server.py", "--socket", "/tmp/s"]
    for key, value in backend_args.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        command += [flag] if value is True else [f"{flag}={value}"]
    return command


def test_a_negative_offset_survives_as_one_argv_element():
    """`--flag value` breaks when the value starts with '-'; `--flag=value` does not.

    This is not hypothetical: an anchor at the midpoint of a westward path is a negative
    dx, and half of a 120-run sweep died on exactly this.
    """
    import argparse

    cmd = build_command({"anchor_offset": "-8.0859,0.6818", "goal_placement": "segment"})
    assert "--anchor-offset=-8.0859,0.6818" in cmd

    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-offset", default=None)
    parser.add_argument("--goal-placement", default=None)
    parsed = parser.parse_args(cmd[4:])
    assert parsed.anchor_offset == "-8.0859,0.6818"
    assert parsed.goal_placement == "segment"


def test_boolean_and_ordinary_values_are_unaffected():
    cmd = build_command({"stop_head": True, "seed": 7, "decode": "sample"})
    assert "--stop-head" in cmd and "--seed=7" in cmd and "--decode=sample" in cmd


# ---------------------------------------------------------------------------
# --no-pose-injection: the pose-free arm of a mixed-modality run
# ---------------------------------------------------------------------------
def test_no_pose_injection_reaches_the_backend_args():
    assert EvalConfig().no_pose_injection is False
    assert "no_pose_injection" not in _pose_args(EvalConfig())
    assert _pose_args(EvalConfig(no_pose_injection=True))["no_pose_injection"] is True


def test_no_pose_injection_survives_argv_construction():
    cmd = build_command(_pose_args(EvalConfig(no_pose_injection=True)))
    assert "--no-pose-injection" in cmd


def test_disable_pose_makes_the_feed_inert():
    """The mechanism: a checkpoint that DECLARES a pose spec must still render no marker.

    Feeding zeros would be a different input the model never trained on; the pose-free
    corpus has no marker in the text at all, so the feed has to be genuinely absent.
    """
    from objectnav_eval.bridge import _PoseFeed

    class Spec:
        token = "<pose>"
        key = "pose"
        n_features = 3
        column = "obs_poses"
        transform = "pose_relative_first"

    class Policy:
        _pose_spec = Spec()
        cfg = type("C", (), {"modality_marker": None, "goal_marker": None,
                             "goal_placement": None})()

    assert _PoseFeed(Policy()).spec is not None, "a pose checkpoint feeds pose by default"
    inert = _PoseFeed(Policy(), disable_pose=True)
    assert inert.spec is None
    assert not inert, "__bool__ must report inert, which is what silences the marker"
