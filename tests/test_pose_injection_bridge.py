"""Train-vs-rollout parity for the injected pose, and the wire that carries it.

The one failure this experiment could least afford is also the one that raises
nothing: the frame the collator computes at training time and the frame the
bridge produces at rollout time drifting apart. The model would read confident
nonsense, every metric would come out plausible, and no assertion anywhere in
either repo would fire. So the convention is pinned here, against the *real*
functions on both sides:

* the training side is ``ModalityEmbedSpec.apply_transform``, which is what
  ``TurnVectorCollator`` calls per example after windowing;
* the rollout side is ``VectorRolloutPolicy.pose_values``, called one raw pose
  at a time exactly as :class:`objectnav_eval.bridge._PoseFeed` drives it.

``pose_values`` is invoked as an unbound method on a stub carrying only
``_raw_poses``. That is deliberate rather than a shortcut: it runs the actual
production code, with no model to load and no reimplementation of the
convention to keep in sync -- a hand-written "equivalent" here would be the
second implementation the whole design exists to avoid.

Everything needing ``longnav`` skips where it is not importable; the wire and
the ``_PoseFeed`` gating tests run anywhere.
"""

import math
import os
import sys

import numpy as np
import pytest

from objectnav_eval.bridge import _PoseFeed, _StopHeadControl, _wire_safe

# The checkpoint's own code lives in the training checkout, which this
# environment need not have installed -- the same reason `--policy-sys-path`
# exists. Overridable, because a second checkout is exactly when a silently
# shadowed `longnav` would be worst.
_LONGNAV_SRC = os.environ.get("LONGNAV_SRC", "/Projects/spatial_training/src")
if os.path.isdir(_LONGNAV_SRC) and _LONGNAV_SRC not in sys.path:
    sys.path.insert(0, _LONGNAV_SRC)

longnav = pytest.importorskip(
    "longnav.utils.pose_frame",
    reason=f"longnav not importable (looked in {_LONGNAV_SRC}; set LONGNAV_SRC)",
)


def _pose_spec():
    """The spec as the pose run's checkpoint declares it, read from its own file.

    Read rather than written out here: the point of the test is agreement with
    what training did, and a spec literal copied into the test would agree with
    itself.
    """
    import json

    from longnav.utils.modality_embed import ModalityEmbedSpec

    path = os.environ.get(
        "POSE_SPEC_JSON", "/Projects/spatial_training/dump/pose_injection/pose_spec.json"
    )
    if not os.path.isfile(path):
        pytest.skip(f"pose spec not found at {path} (set POSE_SPEC_JSON)")
    with open(path) as fh:
        specs = json.load(fh)
    pose = [s for s in specs if s.get("transform") == "pose_relative_first"]
    assert len(pose) == 1, f"expected exactly one pose spec in {path}, got {len(pose)}"
    return ModalityEmbedSpec.from_dict(pose[0])


def _rollout_values(raw_poses):
    """What the bridge feeds the model, step by step, as `(N, 3)`.

    `_PoseFeed` decides *what to send*; `VectorRolloutPolicy.pose_values` turns
    the accumulated raw poses into the injected value. Both are the shipped
    implementations.
    """
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    class _Stub:
        pass

    spec = _pose_spec()
    policy = type("P", (), {"_pose_spec": spec, "cfg": _Config()})()
    feed = _PoseFeed(policy)
    stub = _Stub()
    stub._raw_poses = []  # what `VectorRolloutPolicy.reset()` sets

    out = []
    for pose in raw_poses:
        kwargs = feed.step_kwargs(policy, list(pose))
        value = VectorRolloutPolicy.pose_values(stub, kwargs["obs_pose"])
        assert tuple(value.shape) == (1, 3)
        out.append(value[0].tolist())
    return np.asarray(out, dtype=np.float64)


def _training_values(raw_poses):
    """What the collator puts on the wire for the same example.

    `dtype=torch.float32` is not a convenience here -- it is what
    `TurnVectorCollator.__call__` does to the raw column
    (`vector_sft.py`: `torch.as_tensor(ex[s.source_column], dtype=torch.float32)`)
    before handing it to `apply_transform`. See
    `test_only_input_precision_differs` for the one consequence.
    """
    import torch

    spec = _pose_spec()
    values = torch.as_tensor(np.asarray(raw_poses, dtype=np.float64), dtype=torch.float32)
    return spec.apply_transform(values).numpy().astype(np.float64)


def _as_float32(poses):
    """Round raw poses the way arrow -> collator does, before either path runs."""
    return np.asarray(poses, dtype=np.float32).astype(np.float64)


# ---------------------------------------------------------------------------
# the parity claim
# ---------------------------------------------------------------------------


def _synthetic_episode(n=97, seed=0):
    """A trajectory with a far-from-origin scene anchor and several wraps.

    Both properties are load-bearing. The distant origin is what makes the
    translate-then-rotate order observable (a translation-only transform still
    looks reasonable near the origin), and the accumulated heading is what makes
    an unwrapped difference show up -- on the real corpus that reaches 5.955 rad
    where the true rotation is 0.400.
    """
    rng = np.random.default_rng(seed)
    x, y, theta = 41.317, -18.902, 2.731
    poses = []
    for _ in range(n):
        theta += rng.normal(0.0, 0.9)
        x += math.cos(theta) * rng.uniform(0.0, 0.25)
        y += math.sin(theta) * rng.uniform(0.0, 0.25)
        poses.append([x, y, theta])
    return np.asarray(poses, dtype=np.float64)


def test_rollout_matches_collator_on_a_synthetic_episode():
    """Same raw poses in, same values out -- exactly, not approximately.

    Both sides are fed the float32-rounded raw poses, which is what training
    genuinely has (the collator casts the arrow column). With the input fixed,
    the two paths run the identical `relative_se2` in float64 and return
    float32, so any nonzero difference here is a divergence in the *transform*,
    which is the thing being pinned.
    """
    poses = _as_float32(_synthetic_episode())
    rollout = _rollout_values(poses)
    training = _training_values(poses)
    assert rollout.shape == training.shape == (len(poses), 3)
    np.testing.assert_array_equal(rollout, training)


def test_only_input_precision_differs_on_the_deployed_path():
    """In deployment the rollout gets full-precision poses; training did not.

    The simulator reports float64 and the bridge sends float64, while the
    training column was cast to float32 before the transform ran. So the two
    disagree by the float32 quantisation of a *scene* coordinate, which at ~40 m
    from the scene origin is a few microns. Bounded and asserted rather than
    hidden behind a loose tolerance, because "small" is not a property a test
    should take on trust -- and rounding the rollout to match would be matching
    an artefact of how the dataset was stored, not the convention.
    """
    poses = _synthetic_episode()
    diff = np.abs(_rollout_values(poses) - _training_values(poses))
    scale = np.abs(poses[:, :2]).max()
    bound = 4.0 * np.spacing(np.float32(scale))  # float32 ulp at that magnitude
    assert diff[:, :2].max() < bound, (diff[:, :2].max(), bound)
    assert diff[:, 2].max() < 1e-6


def test_first_row_is_the_origin():
    """Row 0 is exactly (0, 0, 0) -- the episode's first observation *is* the frame."""
    rollout = _rollout_values(_as_float32(_synthetic_episode()))
    np.testing.assert_allclose(rollout[0], [0.0, 0.0, 0.0], atol=0.0)


def test_heading_is_wrapped_not_differenced():
    """The trap, stated as a test.

    A naive `theta - theta0` on these two poses gives 5.955 rad; the true
    rotation is -0.328. Both are "the heading difference"; only one is the one
    the encoder's (cos, sin) expansion was trained on.
    """
    poses = _as_float32([[0.0, 0.0, -3.0], [0.0, 0.0, 2.955]])
    rollout = _rollout_values(poses)
    naive = poses[1, 2] - poses[0, 2]
    assert naive == pytest.approx(5.955, abs=1e-6)
    assert rollout[1, 2] == pytest.approx(-2 * math.pi + 5.955, abs=1e-6)
    assert abs(rollout[1, 2]) <= math.pi


def test_rotation_into_the_start_frame_actually_happens():
    """Translating without rotating is the plausible half-implementation.

    Start facing +y and step one metre along it: in the start frame that is one
    metre *ahead* (+x), not one metre to the left. A translate-only transform
    would report (0, 1).
    """
    poses = _as_float32([[10.0, 5.0, math.pi / 2], [10.0, 6.0, math.pi / 2]])
    rollout = _rollout_values(poses)
    np.testing.assert_allclose(rollout[1], [1.0, 0.0, 0.0], atol=1e-6)


def test_matches_collator_on_a_real_dataset_episode():
    """The same claim against real `obs_poses`, if the corpus is on this box."""
    datasets = pytest.importorskip("datasets", reason="HuggingFace datasets not installed")
    path = os.environ.get(
        "POSE_DATASET", "/Projects/spatial_training/data/v2_25hz/formatted_pose"
    )
    if not os.path.isdir(path):
        pytest.skip(f"pose corpus not found at {path} (set POSE_DATASET)")
    row = datasets.load_from_disk(path)["validation"][0]
    poses = _as_float32(np.asarray(row["obs_poses"], dtype=np.float64))
    assert poses.shape[1] == 3 and len(poses) == row["num_observations"]
    np.testing.assert_array_equal(_rollout_values(poses), _training_values(poses))


# ---------------------------------------------------------------------------
# what the bridge itself contributes
# ---------------------------------------------------------------------------


class _Config:
    """Stands in for `RolloutConfig` -- only the field `_PoseFeed` touches."""

    def __init__(self, modality_marker=None):
        self.modality_marker = modality_marker


def _policy_with_pose(marker=None):
    return type("P", (), {"_pose_spec": _pose_spec(), "cfg": _Config(marker)})()


class _NoPoseSpec:
    _pose_spec = None
    cfg = _Config()


def test_pose_feed_is_inert_without_a_spec():
    """A checkpoint that declares no pose gets the identical `step()` call."""
    policy = _NoPoseSpec()
    feed = _PoseFeed(policy)
    assert not feed
    assert feed.step_kwargs(policy, [1.0, 2.0, 3.0]) == {}
    assert feed.step_kwargs(policy, None) == {}
    # And no marker is written into the turn, so the rendered text is unchanged.
    assert policy.cfg.modality_marker is None


def test_pose_feed_installs_the_marker_into_every_turn():
    """Values alone inject nothing -- the marker has to be in the text.

    Regression test for a real failure: with `modality_marker` left at its
    `None` default the values were supplied with no occurrence to bind them to,
    and `ModalityEmbedder.pending` raised "values were provided but nothing was
    injected" on the first step of every episode.
    """
    policy = _policy_with_pose()
    feed = _PoseFeed(policy)
    assert policy.cfg.modality_marker == "<pose>"
    assert feed.marker == "<pose>"
    # Taken from the spec, never from a literal in the bridge.
    assert feed.marker == feed.spec.token


def test_pose_feed_refuses_to_override_a_different_marker():
    with pytest.raises(ValueError, match="refusing to guess"):
        _PoseFeed(_policy_with_pose("<vel>"))


def test_pose_feed_accepts_an_already_correct_marker():
    policy = _policy_with_pose("<pose>")
    assert _PoseFeed(policy).marker == "<pose>"


def test_pose_feed_rejects_a_config_that_cannot_render_a_marker():
    """An old longnav has no `modality_marker` field; say so, don't inject blind."""
    policy = type("P", (), {"_pose_spec": _pose_spec(), "cfg": object()})()
    with pytest.raises(RuntimeError, match="modality_marker"):
        _PoseFeed(policy)


def test_pose_feed_raises_when_the_checkpoint_needs_a_pose_and_gets_none():
    policy = _policy_with_pose()
    feed = _PoseFeed(policy)
    assert feed
    with pytest.raises(RuntimeError, match="pose_relative_first"):
        feed.step_kwargs(policy, None)


def test_pose_feed_rejects_a_wrong_width():
    policy = _policy_with_pose()
    feed = _PoseFeed(policy)
    with pytest.raises(ValueError, match=r"3 numbers"):
        feed.step_kwargs(policy, [1.0, 2.0])


def test_wire_safe_converts_the_injected_pose_tensor():
    """`last_stats['pose_value']` is a torch tensor, and the sim env has no torch.

    Left as one it fails at `conn.recv()` in the *other* process, an episode away
    from anything that would explain it.
    """
    import torch

    stats = {"stop_prob": 0.25, "pose_value": torch.tensor([1.5, -2.5, 0.75])}
    safe = _wire_safe(stats)
    assert safe["pose_value"] == [1.5, -2.5, 0.75]
    assert safe["stop_prob"] == 0.25
    for value in safe.values():
        assert not hasattr(value, "numpy")


def test_pose_survives_the_socket():
    """A real round trip through `serve()`, with a backend that only records.

    No model and no simulator: the claim is about the protocol. Without the
    `pose` key in `PolicyBridge.act`'s payload, or without `serve` forwarding it,
    every test above still passes and every rollout injects nothing.
    """
    import threading
    from multiprocessing.connection import Client, Listener

    from objectnav_eval.bridge import AUTHKEY, CMD_ACT, CMD_STOP, PolicyBackend, serve

    seen = []

    class _Recorder(PolicyBackend):
        def reset(self, goal_text=None):
            seen.append(("reset", goal_text))

        def act(self, rgb, pose=None):
            seen.append(("act", None if pose is None else list(pose)))
            return np.zeros((10, 3), dtype=np.float32)

    import tempfile
    from pathlib import Path

    socket_path = Path(tempfile.mkdtemp(prefix="pose_wire_")) / "sock"
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
    server = threading.Thread(
        target=serve, args=(str(socket_path), _Recorder()), daemon=True
    )
    server.start()
    conn = listener.accept()
    try:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        conn.send({
            "cmd": CMD_ACT, "shape": rgb.shape, "dtype": "uint8", "data": rgb.tobytes(),
            "pose": [3.25, -4.5, 1.125],
        })
        reply = conn.recv()
        assert "error" not in reply, reply
        assert reply["chunk_shape"] == [10, 3]
        # A sim side that predates pose: the key is simply absent, and a backend
        # whose checkpoint does not need one must still answer.
        conn.send({
            "cmd": CMD_ACT, "shape": rgb.shape, "dtype": "uint8", "data": rgb.tobytes(),
        })
        assert "error" not in conn.recv()
    finally:
        conn.send({"cmd": CMD_STOP})
        server.join(timeout=10)
        conn.close()
        listener.close()

    assert seen == [("act", [3.25, -4.5, 1.125]), ("act", None)]


def test_bridged_policy_sends_the_observation_pose():
    """`Observation.pose` is `get_2d_pose()`; it must reach `bridge.act` unaltered.

    The frame agreement between simulator and dataset rests on both sides
    reading `HabitatRobotSim.get_2d_pose()` -- `generate_pose2d_dataset.py`
    writes its output verbatim into `pose2d`, which becomes `obs_poses`. Any
    transform applied on the way to the bridge would break that identity, so the
    test is that nothing happens to it.
    """
    from objectnav_eval.policies import BridgedPolicy, Observation

    sent = {}

    class _FakeBridge:
        def reset(self, goal):
            pass

        def act(self, rgb, pose=None):
            sent["pose"] = pose
            return {"chunk": np.zeros((10, 3)), "stop": False, "stats": {}}

    pose = np.array([12.5, -3.25, 0.5])
    BridgedPolicy(_FakeBridge()).act(
        Observation(rgb=np.zeros((2, 2, 3), np.uint8), goal="chair", step=0,
                    pose=pose, sim_time=0.0)
    )
    np.testing.assert_array_equal(np.asarray(sent["pose"], dtype=float), pose)


# ---------------------------------------------------------------------------
# the stop head's knobs
# ---------------------------------------------------------------------------


class _FakeStopHead:
    def __init__(self, **kw):
        from longnav.utils.stop_head import StopHeadConfig

        self.cfg = StopHeadConfig(**kw)


def _policy_with_stop_head(**kw):
    head = _FakeStopHead(**kw)
    return type("P", (), {"model": type("M", (), {"stop_head": head})()})()


def test_stop_head_is_reported_but_does_not_act_unless_enabled():
    control = _StopHeadControl(_policy_with_stop_head())
    assert control  # present
    assert control.decided({"stop": True, "stop_prob": 0.99}) is False
    assert _StopHeadControl(_policy_with_stop_head(), enabled=True).decided(
        {"stop": True, "stop_prob": 0.99}
    ) is True


def test_stop_head_requested_on_a_checkpoint_without_one_raises():
    policy = type("P", (), {"model": type("M", (), {"stop_head": None})()})()
    with pytest.raises(ValueError, match="no stop head"):
        _StopHeadControl(policy, enabled=True)


def test_stop_knobs_are_validated_not_silently_written():
    """Assigning to a dataclass field after construction skips `__post_init__`."""
    with pytest.raises(ValueError, match="temperature"):
        _StopHeadControl(_policy_with_stop_head(), temperature=0.0)
    with pytest.raises(ValueError, match="threshold"):
        _StopHeadControl(_policy_with_stop_head(), threshold=1.5)
    with pytest.raises(ValueError, match="inference"):
        _StopHeadControl(_policy_with_stop_head(), inference="greedy")


def test_stop_logit_is_recovered_under_the_configured_temperature():
    """The recorded probability is temperature-scaled; a refit needs the logit."""
    control = _StopHeadControl(_policy_with_stop_head(temperature=2.0))
    logit = -4.0
    prob = 1.0 / (1.0 + math.exp(-logit / 2.0))
    stats = control.annotate({"stop_prob": prob})
    assert stats["stop_logit"] == pytest.approx(logit, abs=1e-6)


def test_config_rejects_both_terminations():
    from objectnav_eval.config import EvalConfig

    with pytest.raises(ValueError, match="pick one"):
        EvalConfig(episodes="x", stop_head=True, auto_stop=True)
    EvalConfig(episodes="x", stop_head=True)  # alone is fine
    EvalConfig(episodes="x", auto_stop=True)


# ---------------------------------------------------------------------------
# the pose-frame ablation
# ---------------------------------------------------------------------------


def _corrupt(rel_row, **kw):
    import torch

    from objectnav_eval.bridge import _PoseCorruption

    out = _PoseCorruption(**kw).apply(torch.tensor([rel_row], dtype=torch.float32))
    return [float(v) for v in out[0]]


def test_corruption_is_inert_by_default():
    from objectnav_eval.bridge import _PoseCorruption

    assert not _PoseCorruption()
    assert not _PoseCorruption(mirror=False, rotate_deg=0.0)
    assert not _PoseCorruption(rotate_deg=360.0)   # a full turn is the identity
    assert _PoseCorruption(mirror=True)
    assert _PoseCorruption(rotate_deg=90.0)
    row = [1.5, -2.5, 0.75]
    assert _corrupt(row) == pytest.approx(row)


def test_mirror_flips_the_strafe_axis_and_the_heading():
    """One metre ahead and one to the LEFT becomes one ahead and one to the RIGHT."""
    assert _corrupt([1.0, 1.0, 0.5], mirror=True) == pytest.approx([1.0, -1.0, -0.5])
    # Straight ahead is on the mirror line, so it is untouched.
    assert _corrupt([2.0, 0.0, 0.0], mirror=True) == pytest.approx([2.0, 0.0, 0.0])


def test_rotation_turns_ahead_into_left():
    assert _corrupt([1.0, 0.0, 0.0], rotate_deg=90.0) == pytest.approx(
        [0.0, 1.0, math.pi / 2], abs=1e-6
    )


def test_mirror_then_rotate_is_the_documented_order():
    """The two do not commute, so the order is part of the contract."""
    row = [1.0, 1.0, 0.0]
    both = _corrupt(row, mirror=True, rotate_deg=90.0)
    # mirror -> (1, -1, 0); then rotate 90 -> (1, 1, pi/2)
    assert both == pytest.approx([1.0, 1.0, math.pi / 2], abs=1e-6)
    # The other order would give (-1, 1) for the position; assert we did not do that.
    assert both[0] > 0


def test_corruption_preserves_geometry():
    """Self-consistent, which is what makes this an ablation and not just noise.

    Distances from the origin and pairwise distances are invariant, so the
    trajectory the model reads is a rigid (reflected) copy of the true one --
    only its orientation relative to the world is wrong.
    """
    rows = [[1.0, 0.0, 0.0], [3.0, 2.0, 0.4], [-1.0, 5.0, -2.9]]
    out = [_corrupt(r, mirror=True, rotate_deg=90.0) for r in rows]
    for r, o in zip(rows, out):
        assert math.hypot(*r[:2]) == pytest.approx(math.hypot(*o[:2]), abs=1e-5)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d_in = math.dist(rows[i][:2], rows[j][:2])
            d_out = math.dist(out[i][:2], out[j][:2])
            assert d_in == pytest.approx(d_out, abs=1e-5)


def test_corruption_keeps_the_origin_at_the_origin():
    """Row 0 stays at (0, 0) -- but its HEADING becomes the rotation angle.

    That is the consistent reading of a frame rotation, not a bug: re-expressing
    the episode in a frame turned by -phi means the agent started out facing phi.
    Leaving the heading at 0 while rotating the positions would be the
    inconsistent version, and would corrupt the geometry rather than the frame.
    """
    out = _corrupt([0.0, 0.0, 0.0], mirror=True, rotate_deg=90.0)
    assert out[:2] == pytest.approx([0.0, 0.0], abs=1e-7)
    assert out[2] == pytest.approx(math.pi / 2, abs=1e-6)
    # With no rotation the origin row is untouched entirely.
    assert _corrupt([0.0, 0.0, 0.0], mirror=True) == pytest.approx([0.0, 0.0, 0.0])


def test_heading_stays_wrapped_after_rotation():
    out = _corrupt([0.0, 0.0, 3.0], rotate_deg=90.0)
    assert abs(out[2]) <= math.pi


def test_rotating_the_RAW_pose_would_have_been_a_no_op():
    """Why the corruption lives on the relative pose. Stated as a test because the
    alternative implementation is the obvious one and fails silently.

    `relative_se2` re-anchors in the first observation's own frame, so rotating
    every raw pose is cancelled exactly. A raw-pose implementation of
    --pose-rotate-deg would have changed nothing at all.
    """
    poses = _synthetic_episode(n=20)
    rotated = poses.copy()
    phi = math.radians(90.0)
    cos, sin = math.cos(phi), math.sin(phi)
    rotated[:, 0] = cos * poses[:, 0] - sin * poses[:, 1]
    rotated[:, 1] = sin * poses[:, 0] + cos * poses[:, 1]
    rotated[:, 2] = poses[:, 2] + phi
    np.testing.assert_allclose(
        _rollout_values(_as_float32(poses)), _rollout_values(_as_float32(rotated)),
        atol=1e-5,
    )


def test_pose_feed_routes_a_corrupted_value_through_modality():
    """Corrupted values cannot go via `obs_pose=` -- that is the uncorrupted path."""
    from objectnav_eval.bridge import _PoseCorruption

    class _Policy:
        _pose_spec = None
        cfg = _Config()

        def pose_values(self, raw):
            import torch

            return torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    policy = _Policy()
    policy._pose_spec = _pose_spec()
    feed = _PoseFeed(policy, _PoseCorruption(mirror=True, rotate_deg=90.0))
    kwargs = feed.step_kwargs(policy, [5.0, 5.0, 1.0])
    assert "obs_pose" not in kwargs
    assert list(kwargs["modality"]) == ["pose"]
    value = kwargs["modality"]["pose"]
    assert [round(float(v), 5) for v in value[0]] == pytest.approx(
        [0.0, 1.0, round(math.pi / 2, 5)], abs=1e-4
    )


# ---------------------------------------------------------------------------
# the foreign-trajectory ablation
# ---------------------------------------------------------------------------


class _FakeForeign:
    """Stands in for `_ForeignPose` -- deterministic trajectories, no dataset."""

    def __init__(self, lengths=(4, 3)):
        self.lengths = list(lengths)
        self.drawn = 0

    def draw(self, rng):
        n = self.lengths[self.drawn % len(self.lengths)]
        self.drawn += 1
        # A straight walk along +x starting at a distinctive offset per draw.
        base = 100.0 * self.drawn
        return np.array([[base + i, 0.0, 0.0] for i in range(n)], dtype=np.float64)

    def walk(self, seed):
        from objectnav_eval.bridge import _ForeignWalk

        return _ForeignWalk(self, np.random.default_rng(int(seed)))


def test_foreign_walk_chains_trajectories_continuously():
    """Corpus episodes are shorter than an eval episode, so they must join up.

    The join is an SE(2) rebase, not a wrap or a freeze: wrapping teleports, and
    holding the last pose reads as 'stopped dead', which is a signal of its own.
    """
    walk = _FakeForeign(lengths=(4, 3)).walk(0)
    poses = np.array([walk.next() for _ in range(9)])
    assert walk.trajectories >= 3, "should have chained more than one trajectory"
    steps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    # Every consecutive pair is one unit apart -- including across the joins.
    np.testing.assert_allclose(steps, 1.0, atol=1e-9)


def test_foreign_walk_is_seeded_and_reproducible():
    a = [_FakeForeign().walk(7).next() for _ in range(5)]
    b = [_FakeForeign().walk(7).next() for _ in range(5)]
    assert a == b


def test_foreign_pose_replaces_the_robots_own():
    """The robot's real pose is discarded, and the substitution happens BEFORE
    the frame conversion so foreign values take the identical relative_se2."""
    policy = _policy_with_pose()
    feed = _PoseFeed(policy, foreign=_FakeForeign(lengths=(5,)), seed=3)
    feed.on_episode_reset()
    real = [999.0, -999.0, 3.0]
    got = [feed.step_kwargs(policy, real)["obs_pose"] for _ in range(3)]
    assert all(g != real for g in got), "the robot's own pose leaked through"
    xs = [g[0] for g in got]
    assert xs == sorted(xs), "foreign poses should advance along the drawn walk"


def test_foreign_redraws_every_episode():
    policy = _policy_with_pose()
    feed = _PoseFeed(policy, foreign=_FakeForeign(lengths=(5,)), seed=0)
    feed.on_episode_reset()
    first = feed.step_kwargs(policy, [0.0, 0.0, 0.0])["obs_pose"]
    feed.on_episode_reset()
    second = feed.step_kwargs(policy, [0.0, 0.0, 0.0])["obs_pose"]
    assert first != second, "each episode must draw its own trajectory"


def test_foreign_is_off_by_default():
    policy = _policy_with_pose()
    feed = _PoseFeed(policy)
    feed.on_episode_reset()
    assert feed.walk is None
    assert feed.step_kwargs(policy, [1.0, 2.0, 3.0])["obs_pose"] == [1.0, 2.0, 3.0]
