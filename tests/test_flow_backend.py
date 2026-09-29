"""The flow-matching policy backend: what it refuses, what it reports, and the frame.

``FlowRolloutBackend`` is the fourth registered backend and the only one for a head with
a single generation mode. Three claims are pinned here, in the order they would ruin a
closed-loop number:

* **the pose frame.** ``act()`` receives the agent's raw planar ``[X, Y, theta]`` and the
  model must be shown it relative to the episode's FIRST observation, in that
  observation's frame. A divergence between that and what the training collator computed
  raises nothing -- the model reads confident nonsense and the run produces a plausible
  success rate. ``test_backend_act_reproduces_the_collator_on_a_real_episode`` drives the
  real ``FlowRolloutBackend.act`` over a real episode's ``obs_poses`` and compares against
  ``ModalityEmbedSpec.apply_transform``, which is what ``TurnVectorCollator`` calls.
* **the decode rule.** ``FlowActionCodec.DECODES`` is ``("sample", "context")`` with no
  ``"mean"`` on purpose. ``"mean"`` must be rejected rather than approximated, and
  ``"context"`` rejected rather than sent down the wire as a 1024-long "chunk".
* **what the eval log says.** The same weights integrated with 4 Euler steps and with 10
  are different policies, and nothing in the checkpoint path records which one ran.

No VLM is loaded: ``__init__`` would instantiate a 2B-parameter backbone, so the parts
that are ordinary code are exercised against stubs -- but always the *shipped* functions
for anything that touches the convention (``VectorRolloutPolicy.pose_values``,
``_PoseFeed``), never a hand-written equivalent, which would be the second implementation
the design exists to avoid.

    /workspace/conda/envs/vln/bin/python -m pytest tests/test_flow_backend.py -q
"""

import json
import os
import sys

import numpy as np
import pytest

from objectnav_eval.bridge import FlowRolloutBackend, _PeriodicContextReset, _PoseFeed

_LONGNAV_SRC = os.environ.get("LONGNAV_SRC", "/Projects/spatial_training/src")
if os.path.isdir(_LONGNAV_SRC) and _LONGNAV_SRC not in sys.path:
    sys.path.insert(0, _LONGNAV_SRC)

#: The run this backend was written for. Its head config is the *self-describing*
#: checkpoint the modality spec is read from -- a spec literal in this file would only
#: agree with itself.
FLOW_CKPT = os.environ.get(
    "FLOW_CKPT",
    "/Projects/spatial_training/dump/pose_injection/run_v8_flow_planar_pose/checkpoint-400",
)


# ---------------------------------------------------------------------------
# what the backend refuses, before it ever touches a checkpoint
# ---------------------------------------------------------------------------


def test_backend_is_registered():
    from objectnav_eval.bridge import available_backends

    assert "flow_rollout" in available_backends()


def test_mean_decode_is_refused_and_says_why():
    """Averaging ODE samples IS the conditional mean, i.e. the creeping failure.

    The check runs before the backend's imports, so it fails identically in an
    interpreter that cannot load the checkpoint at all -- which is why this test needs
    neither torch nor a GPU.
    """
    with pytest.raises(ValueError, match="conditional mean"):
        FlowRolloutBackend(ckpt="/nonexistent", decode="mean")
    with pytest.raises(ValueError, match="conditional mean"):
        FlowRolloutBackend(ckpt="/nonexistent", decode="argmax")


def test_context_passthrough_is_refused():
    with pytest.raises(ValueError, match="offline passthrough"):
        FlowRolloutBackend(ckpt="/nonexistent", decode="context")


def test_sample_is_the_only_accepted_rule():
    """`sample` gets past the guard and fails later, on the missing checkpoint."""
    with pytest.raises(Exception) as excinfo:
        FlowRolloutBackend(ckpt="/nonexistent", decode="sample")
    assert "decode must be" not in str(excinfo.value)


def test_loader_rejects_a_mean_decode_on_the_model_side_too():
    """The model-side loader carries the same guard, for scripts that bypass the bridge."""
    flow_rollout = pytest.importorskip(
        "longnav.utils.flow_rollout",
        reason=f"longnav not importable (looked in {_LONGNAV_SRC}; set LONGNAV_SRC)",
    )
    with pytest.raises(ValueError, match="conditional mean"):
        flow_rollout.load_flow_policy("/nonexistent", decode="mean")
    with pytest.raises(ValueError, match="num_inference_steps"):
        flow_rollout.load_flow_policy("/nonexistent", num_inference_steps=0)


# ---------------------------------------------------------------------------
# the stubs: everything but the 2B backbone
# ---------------------------------------------------------------------------


class _Config:
    """Stands in for `RolloutConfig` -- only the field `_PoseFeed` touches."""

    def __init__(self, modality_marker=None):
        self.modality_marker = modality_marker


class _FakeCodec:
    def __init__(self, num_inference_steps=10, action_scales=(0.03, 0.03, 0.05)):
        import torch

        self.num_inference_steps = num_inference_steps
        self.action_scales = torch.tensor(list(action_scales), dtype=torch.float64)


class _FakeDecoder:
    def __init__(self, config, n_ticks=10):
        self._config = config
        self.n_ticks = n_ticks

    def to_config(self):
        return dict(self._config)


class _FakeFlowModel:
    def __init__(self, decoder_config, n_ticks=10, **codec_kwargs):
        self.decoder = _FakeDecoder(decoder_config, n_ticks)
        self.codec = _FakeCodec(**codec_kwargs)
        self.stop_head = None


class _FakeFlowPolicy:
    """A `VectorRolloutPolicy` with the real `pose_values` and no model.

    `pose_values` is taken unbound from the shipped class rather than reimplemented: the
    frame convention is the thing under test, and a local copy of it here would agree
    with itself and with nothing else.
    """

    def __init__(self, model, spec=None):
        from longnav.utils.vector_rollout import VectorRolloutPolicy

        self.model = model
        self.cfg = _Config()
        self._pose_spec = spec
        self._raw_poses = []
        self.injected = []
        self.steps = 0
        self.last_stats = {}
        self._pose_values = VectorRolloutPolicy.pose_values

    def pose_values(self, obs_pose):
        return self._pose_values(self, obs_pose)

    def describe(self):
        return "prefix='**' postfix='**'"

    def reset(self, goal_text=None):
        self._raw_poses = []

    def step(self, image, obs_pose=None, modality=None):
        """The shape of the real `step()`'s pose handling, and nothing else."""
        import torch

        self.steps += 1
        if obs_pose is not None:
            value = self.pose_values(obs_pose)
            self.injected.append([float(v) for v in value[0]])
            self.last_stats = {"pose_value": value[0].clone()}
        elif modality is not None:
            for value in modality.values():
                self.injected.append([float(v) for v in value[0]])
        return torch.zeros(self.model.decoder.n_ticks, 3)


def flow_backend(decoder_config=None, spec=None, decode="sample", seed=1000,
                 seed_by_episode_uid=False, num_inference_steps=10,
                 ode_rng_preconsume_size=0, context_reset_every=0, **model_kwargs):
    """A `FlowRolloutBackend` wearing a fake policy.

    `__init__` loads a multi-billion-parameter VLM, which is exactly what this suite does
    not do; everything below it -- `act`, `reset`, `describe` -- is ordinary code driving
    the shared `_PoseFeed`, and that is what is pinned here.
    """
    backend = FlowRolloutBackend.__new__(FlowRolloutBackend)
    backend.decode = decode
    backend.seed = seed
    backend.seed_by_episode_uid = seed_by_episode_uid
    backend.uses_episode_uid = seed_by_episode_uid
    backend.ode_rng_preconsume_size = ode_rng_preconsume_size
    backend._episode_count = 0
    backend.num_inference_steps = num_inference_steps
    backend._context = _PeriodicContextReset(context_reset_every)
    backend._seed_sampling = lambda policy, s: policy.__dict__.__setitem__("seeded", s)
    model = _FakeFlowModel(
        decoder_config or {"d_model": 128, "n_layers": 4, "n_heads": 4, "dim_ff": 512,
                           "dropout": 0.1, "n_context_tokens": 8},
        num_inference_steps=num_inference_steps, **model_kwargs,
    )
    backend.policy = _FakeFlowPolicy(model, spec)
    backend._pose = _PoseFeed(backend.policy, seed=seed)
    backend._stop = _StopStub()
    backend._trajectory_stop_threshold_m = None
    backend._trajectory_stop_yaw_threshold_rad = None
    backend._trajectory_stop_min_steps = 0
    backend._trajectory_stop_prefix_points = 10
    backend._trajectory_stop = False
    backend._trajectory_stop_stats = {}
    backend._decision_index = 0
    backend._stop_head_min_steps = 0
    return backend


class _StopStub:
    """`_StopHeadControl` on a checkpoint with no stop head, without the import dance."""

    def __bool__(self):
        return False

    def seed(self, policy, seed):
        pass

    def annotate(self, stats):
        return stats

    def decided(self, stats):
        return False

    def describe(self):
        return "stop head=none (oracle --auto-stop is the only termination)"


def _flow_pose_spec():
    """The pose spec as the v8 checkpoint declares it, read from the checkpoint."""
    from longnav.utils.modality_embed import ModalityEmbedSpec

    path = os.path.join(FLOW_CKPT, "turn_vector_head_config.json")
    if not os.path.isfile(path):
        pytest.skip(f"flow checkpoint not found at {FLOW_CKPT} (set FLOW_CKPT)")
    with open(path) as fh:
        meta = json.load(fh)
    specs = [s for s in meta["model"].get("modality_specs") or []
             if s.get("transform") == "pose_relative_first"]
    assert len(specs) == 1, f"expected one pose spec in {path}, got {len(specs)}"
    return ModalityEmbedSpec.from_dict(specs[0])


# ---------------------------------------------------------------------------
# the frame: the divergence that fails silently
# ---------------------------------------------------------------------------


def _rgb():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_trajectory_stop_requires_both_translation_and_yaw_to_be_small():
    import torch

    backend = flow_backend()
    backend._trajectory_stop_threshold_m = 0.01
    backend._trajectory_stop_yaw_threshold_rad = 0.01
    backend.policy.step = lambda *_args, **_kwargs: torch.tensor(
        [[0.0, 0.0, 0.02]], dtype=torch.float32
    )
    backend.act(_rgb())
    assert backend.wants_stop() is False
    assert backend.last_stats["action_yaw_path_rad"] == pytest.approx(0.02)

    backend.policy.step = lambda *_args, **_kwargs: torch.zeros((1, 3))
    backend.act(_rgb())
    assert backend.wants_stop() is True


def test_trajectory_stop_uses_the_executable_chunk_prefix():
    import torch

    backend = flow_backend()
    backend._trajectory_stop_threshold_m = 0.01
    backend._trajectory_stop_yaw_threshold_rad = 0.01
    backend._trajectory_stop_prefix_points = 10
    backend.policy.step = lambda *_args, **_kwargs: torch.tensor(
        [[0.0, 0.0, 0.0]] * 10 + [[1.0, 0.0, 0.0]], dtype=torch.float32
    )
    backend.act(_rgb())
    assert backend.wants_stop() is True
    assert backend.last_stats["action_path_length_m"] == pytest.approx(0.0)


def _training_values(raw_poses, spec):
    """What the collator puts on the wire for the same episode.

    `dtype=torch.float32` is what `TurnVectorCollator.__call__` does to the raw arrow
    column before handing it to `apply_transform`.
    """
    import torch

    values = torch.as_tensor(np.asarray(raw_poses, dtype=np.float64), dtype=torch.float32)
    return spec.apply_transform(values).numpy().astype(np.float64)


def _drive(backend, poses):
    """Feed an episode through the real `FlowRolloutBackend.act`, one pose per step."""
    for pose in poses:
        chunk = backend.act(_rgb(), pose=[float(v) for v in pose])
        assert chunk.shape == (backend.policy.model.decoder.n_ticks, 3)
        assert chunk.dtype == np.float32
    return np.asarray(backend.policy.injected, dtype=np.float64)


def test_backend_act_reproduces_the_collator_on_a_real_episode():
    """THE replay test: a real episode in, identical injected poses out.

    Same raw `obs_poses` the training corpus stores, once through
    `FlowRolloutBackend.act` -> `_PoseFeed` -> `VectorRolloutPolicy.pose_values` ->
    `relative_se2`, once through the collator's `apply_transform`. Exact equality, not
    approximate: with the float32-rounded input fixed (which is what training genuinely
    has), both sides run the identical transform in float64, so any difference at all is
    a divergence in the convention.
    """
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    datasets = pytest.importorskip("datasets", reason="HuggingFace datasets not installed")
    path = os.environ.get(
        "POSE_DATASET", "/Projects/spatial_training/data/v2_25hz/formatted_pose"
    )
    if not os.path.isdir(path):
        pytest.skip(f"pose corpus not found at {path} (set POSE_DATASET)")

    spec = _flow_pose_spec()
    row = datasets.load_from_disk(path)["validation"][0]
    poses = np.asarray(row["obs_poses"], dtype=np.float32).astype(np.float64)
    assert poses.shape[1] == 3 and len(poses) == row["num_observations"]

    backend = flow_backend(spec=spec)
    backend.reset(goal_text="chair")
    rollout = _drive(backend, poses)
    training = _training_values(poses, spec)

    assert rollout.shape == training.shape == (len(poses), 3)
    np.testing.assert_array_equal(rollout, training)
    # And the value is genuinely a change of frame, not a passthrough of the raw pose.
    np.testing.assert_allclose(rollout[0], [0.0, 0.0, 0.0], atol=0.0)
    assert np.abs(rollout[1:, :2]).max() < np.abs(poses[:, :2]).max()


def test_backend_reset_restarts_the_pose_origin():
    """A new episode is a new frame: `reset()` clears the accumulated raw poses.

    Without it, episode two's poses would be expressed against episode one's first
    observation -- in a different scene, with a different arbitrary origin.
    """
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    spec = _flow_pose_spec()
    backend = flow_backend(spec=spec)
    poses = [[10.0, 5.0, 0.5], [11.0, 5.0, 0.5]]

    backend.reset(goal_text="chair")
    _drive(backend, poses)
    backend.reset(goal_text="chair")
    backend.policy.injected.clear()
    second = _drive(backend, poses)

    np.testing.assert_allclose(second[0], [0.0, 0.0, 0.0], atol=0.0)


def test_backend_seeds_every_episode_unconditionally():
    """One generation mode, and it always draws noise -- so seeding is not gated on the
    decode rule the way it is for the bin and AR backends."""
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    backend = flow_backend(seed=1000)
    backend.reset(goal_text="chair")
    assert backend.policy.__dict__["seeded"] == 1000
    backend.reset(goal_text="chair")
    assert backend.policy.__dict__["seeded"] == 1000


def test_fixed_episode_seed_ignores_episode_uid():
    backend = flow_backend(seed=42)
    backend.reset(goal_text="sofa", episode_uid="wPLokgvCnuk:6")
    assert backend.policy.__dict__["seeded"] == 42


def test_uid_seed_option_is_rejected_for_fixed_episode_seeding():
    from objectnav_eval.config import EvalConfig

    with pytest.raises(ValueError, match="fixed per-episode"):
        EvalConfig(
            episodes="x", ckpt="/ckpt", policy_backend="flow_rollout",
            policy_python="/usr/bin/python", policy_seed_by_episode_uid=True,
        )


def test_ode_rng_compatibility_consumes_the_raw_sampler_permutation():
    import torch

    backend = flow_backend(ode_rng_preconsume_size=9)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(37)
    backend.policy.model.codec.generator = generator

    expected = torch.Generator(device="cpu")
    expected.manual_seed(37)
    torch.randperm(9, generator=expected, device="cpu")
    expected_next = torch.randn(4, generator=expected)

    backend.act(_rgb())
    actual_next = torch.randn(4, generator=generator)
    torch.testing.assert_close(actual_next, expected_next)


def test_backend_is_inert_on_a_checkpoint_with_no_pose_spec():
    """A flow checkpoint declaring no modality gets the identical `step()` call."""
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    backend = flow_backend(spec=None)
    backend.reset(goal_text="chair")
    assert not backend._pose
    assert backend.policy.cfg.modality_marker is None
    backend.act(_rgb(), pose=[1.0, 2.0, 3.0])
    assert backend.policy.injected == []
    assert backend.policy.steps == 1


def test_backend_installs_the_marker_from_the_checkpoints_own_spec():
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    spec = _flow_pose_spec()
    backend = flow_backend(spec=spec)
    assert backend.policy.cfg.modality_marker == spec.token
    assert backend._pose.marker == spec.token


def test_backend_raises_when_the_checkpoint_needs_a_pose_and_gets_none():
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    backend = flow_backend(spec=_flow_pose_spec())
    backend.reset(goal_text="chair")
    with pytest.raises(RuntimeError, match="pose_relative_first"):
        backend.act(_rgb(), pose=None)


# ---------------------------------------------------------------------------
# what the eval log records
# ---------------------------------------------------------------------------


def test_describe_reports_the_integration_that_actually_ran():
    """Euler steps and action scales are policy, not bookkeeping.

    The same weights at 4 steps and at 10 are different policies, and neither the
    checkpoint path nor the backend name records which one ran.
    """
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    described = flow_backend(num_inference_steps=4).describe()
    assert "euler_steps=4" in described
    assert "action_scales=(0.03, 0.03, 0.05)" in described
    assert "decode=sample" in described and "seed_reset_per_episode=1000" in described
    assert "n_ticks=10" in described
    assert "memory=unlimited" in described


def test_describe_surfaces_the_velocity_field_architecture():
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    described = flow_backend(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "dim_ff": 512, "dropout": 0.1,
         "n_context_tokens": 8, "use_incoming_motion": True}
    ).describe()
    assert "d_model=128" in described and "n_layers=4" in described
    assert "n_context_tokens=8" in described
    assert "use_incoming_motion=True" in described


def test_describe_fills_in_the_defaults_of_a_pre_option_checkpoint():
    """A checkpoint written before an option existed stores nothing for it; the log still
    has to state the architecture, or two runs' lines are not comparable by eye."""
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    described = flow_backend(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "dim_ff": 512, "dropout": 0.1}
    ).describe()
    assert "n_context_tokens=8" in described
    assert "use_incoming_motion=False" in described
    assert "n_dims=3" in described


def test_memory_ablation_is_reported_and_off_by_default():
    pytest.importorskip("longnav.utils.vector_rollout", reason="longnav not importable")
    assert "memory=unlimited" in flow_backend().describe()
    assert "memory=7 steps/block" in flow_backend(context_reset_every=7).describe()


# ---------------------------------------------------------------------------
# the CLI plumbing for the one knob that changes the policy
# ---------------------------------------------------------------------------


def test_inference_steps_reaches_the_backend_args():
    from objectnav_eval.config import EvalConfig

    config = EvalConfig(episodes="x", ckpt="/ckpt", policy_backend="flow_rollout",
                        policy_inference_steps=4, policy_python=None)
    assert config.policy_inference_steps == 4

    # The same field the bridged path turns into `--num-inference-steps 4`.
    from objectnav_eval.policies import build_policy

    class _Bridge:
        seen = {}

        def __init__(self, script, python, backend_args, log_path=None):
            _Bridge.seen = dict(backend_args)

    import objectnav_eval.bridge as bridge_module

    original = bridge_module.PolicyBridge
    bridge_module.PolicyBridge = _Bridge
    try:
        config.policy_python = "/usr/bin/python"
        build_policy(config, "scripts/policy_server.py")
    finally:
        bridge_module.PolicyBridge = original
    assert _Bridge.seen["num_inference_steps"] == 4
    assert _Bridge.seen["backend"] == "flow_rollout"


def test_inference_steps_is_absent_unless_asked_for():
    """Unset must mean the checkpoint's own value, i.e. no flag at all on the command
    line -- so a run that does not sweep the integration is byte-identical to one from
    before this option existed."""
    from objectnav_eval.config import EvalConfig
    from objectnav_eval.policies import build_policy

    config = EvalConfig(episodes="x", ckpt="/ckpt", policy_backend="flow_rollout",
                        policy_python="/usr/bin/python")

    class _Bridge:
        seen = {}

        def __init__(self, script, python, backend_args, log_path=None):
            _Bridge.seen = dict(backend_args)

    import objectnav_eval.bridge as bridge_module

    original = bridge_module.PolicyBridge
    bridge_module.PolicyBridge = _Bridge
    try:
        build_policy(config, "scripts/policy_server.py")
    finally:
        bridge_module.PolicyBridge = original
    assert "num_inference_steps" not in _Bridge.seen
