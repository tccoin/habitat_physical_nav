"""The two-environment bridge: run the policy in a second interpreter.

The simulator and the policy genuinely cannot share a Python environment here.
The sim env has ``habitat_sim`` and a ``transformers`` too old for current
checkpoints; the model env has ``transformers`` + flash-attn and no
``habitat_sim``. This is not incidental -- the dependency sets conflict -- so
the boundary is a first-class part of the design rather than a workaround.

The protocol is deliberately tiny: a unix socket, three message types
(``reset`` / ``act`` / ``stop``), frames crossing as raw bytes plus shape and
dtype rather than pickled arrays (the two envs also disagree about numpy, 1.26
vs 2.x, and unpickling arrays across that is asking for trouble).

Only this module and ``scripts/policy_server.py`` know the boundary exists, so
replacing the socket with something else later touches nothing else.

Imports numpy and the standard library only -- it has to be importable in
*both* environments.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

AUTHKEY = b"objectnav-eval-policy"

CMD_RESET = "reset"
CMD_ACT = "act"
CMD_STOP = "stop"


# ---------------------------------------------------------------------------
# Server side: what actually loads a checkpoint
# ---------------------------------------------------------------------------


class PolicyBackend:
    """Interface for the thing that turns an image into an action chunk.

    Lives in the *model* environment. New policy head types (a normalizing-flow
    head, a diffusion head, a discrete-action head) are new backends: implement
    these three methods, register the loader with :func:`register_backend`, and
    pass ``--policy-backend <name>``. Nothing on the simulator side changes,
    because the wire format is unchanged -- an ``(T, 3)`` chunk of relative
    poses, optionally with a stop flag.
    """

    def reset(self, goal_text: Optional[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def act(self, rgb: np.ndarray, pose: Optional[Sequence[float]] = None) -> np.ndarray:
        """Return an ``(T, 3)`` chunk of ``[dx, dy, dtheta]`` offsets.

        ``pose`` is the agent's **raw planar** ``[X, Y, theta]`` at this
        observation, exactly as ``HabitatRobotSim.get_2d_pose()`` returns it --
        scene frame, arbitrary per-scene origin, *not* relative to anything. It
        is offered to every backend and ignored by the ones whose checkpoint
        declares no use for it; see :class:`_PoseFeed` for why the raw value is
        what crosses the wire.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def last_stats(self) -> Dict[str, Any]:
        return {}

    def wants_stop(self) -> bool:
        """Policies with a termination head override this."""
        return False

    def describe(self) -> str:
        return type(self).__name__


_BACKENDS: Dict[str, Callable[..., PolicyBackend]] = {}


def register_backend(name: str, loader: Callable[..., PolicyBackend]) -> None:
    _BACKENDS[name] = loader


def build_backend(name: str, **kwargs: Any) -> PolicyBackend:
    if name not in _BACKENDS:
        raise KeyError(
            f"unknown policy backend '{name}'; registered: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name](**kwargs)


def available_backends() -> list:
    return sorted(_BACKENDS)


# ---------------------------------------------------------------------------
# Shared, checkpoint-driven capabilities
# ---------------------------------------------------------------------------
# Neither of the two below is a backend. They are the parts of a rollout that
# some checkpoints have and others do not, and which are therefore switched on
# by *reading the checkpoint*, not by picking a different `--policy-backend`.
# A pose-injecting AR head and a pose-injecting vector head are the same eval
# problem; making them two backends would duplicate the one piece that can be
# silently wrong.


class _PoseCorruption:
    """Mirror and/or rotate the frame the pose is expressed in. **Diagnostic only.**

    An ablation for the question the pose-injection run exists to answer: does the
    policy *use* the injected pose, or does it merely tolerate it? A corrupted
    frame is still perfectly self-consistent -- distances, pairwise separations
    and the shape of the travelled path are all preserved, and row 0 stays at the
    origin. What changes is which way "ahead at the start" points. (Row 0's
    *heading* does become the rotation angle, which is the consistent reading of
    a frame rotation: re-expressing the episode in a frame turned by -phi means
    the agent started out facing phi. Rotating the positions while pinning the
    heading at 0 would corrupt the geometry rather than the frame.) A policy navigating by the injected metric frame
    should degrade sharply; one that ignores it should not move at all.

    **The corruption applies to the RELATIVE pose, never to the raw scene pose,
    and that is not a detail.** ``relative_se2`` re-anchors every value in the
    first observation's own frame, so a rigid rotation of the raw poses is
    *algebraically cancelled*::

        rel' = R(-(theta0 + phi)) . R(phi) . (p - p0) = R(-theta0) . (p - p0) = rel

    Corrupting the input would therefore be an exact no-op for rotation, and the
    run would look like "rotating the frame changes nothing", which is a
    conclusion about the transform rather than about the model. (A mirror is
    *not* cancelled -- it conjugates through -- so a raw-pose implementation
    would have silently ablated half of what was asked for and no-opped the
    other half.)

    Order is **mirror, then rotate**, and it matters: reflections and rotations
    do not commute. ``mirror`` flips the strafe axis, `+y` (left) becoming `-y`,
    with the heading negated to match -- the world seen through a mirror placed
    along the start heading.
    """

    def __init__(self, mirror: bool = False, rotate_deg: float = 0.0):
        self.mirror = bool(mirror)
        self.rotate_deg = float(rotate_deg)

    def __bool__(self) -> bool:
        return self.mirror or bool(self.rotate_deg % 360.0)

    def apply(self, rel):
        """``(1, 3)`` relative pose -> corrupted ``(1, 3)``. Identity when inert."""
        if not self:
            return rel
        import math

        import torch

        from longnav.utils.pose_frame import wrap_to_pi

        x, y, theta = (float(v) for v in rel[0])
        if self.mirror:
            y, theta = -y, -theta
        if self.rotate_deg:
            phi = math.radians(self.rotate_deg)
            cos, sin = math.cos(phi), math.sin(phi)
            x, y = cos * x - sin * y, sin * x + cos * y
            theta = theta + phi
        theta = float(wrap_to_pi(torch.tensor(theta, dtype=torch.float64)))
        return torch.tensor([[x, y, theta]], dtype=torch.float32)

    def describe(self) -> str:
        if not self:
            return "pose frame=true"
        parts = []
        if self.mirror:
            parts.append("MIRRORED about the start heading (strafe axis flipped)")
        if self.rotate_deg:
            parts.append(f"ROTATED {self.rotate_deg:g} deg")
        return "pose frame=CORRUPTED: " + " then ".join(parts) + " -- diagnostic, not a policy"


class _ForeignPose:
    """Inject a **different episode's** pose sequence. The information ablation.

    The mirror/rotate corruption above is an isometry: it breaks the binding
    between the pose frame and world direction, but every intrinsic quantity
    survives -- distances travelled, revisit structure, how much ground has been
    covered. A policy using pose only for "have I been here before" is entirely
    unharmed by it, so a null result there cannot distinguish *unused* from
    *used, but only frame-intrinsically*.

    This closes that gap. The injected poses are real trajectories drawn from the
    training corpus, so they stay squarely on the distribution the encoder was
    fitted to -- unlike zeros or noise, which conflate "no information" with
    "input the model has never seen". What they carry is zero information about
    where the robot actually is or has been.

    **Chaining.** Corpus episodes top out at 311 observations while an eval
    episode may run 350 steps, so one trajectory is not always enough. Rather
    than wrapping (which teleports) or holding the last pose (which reads as
    "stopped dead", a signal of its own), the next trajectory is rebased by the
    SE(2) transform that puts its first pose exactly on the current last pose.
    The result is still one continuous, plausible walk.

    Sampling is seeded per episode from the same base as every other stochastic
    part of a rollout, so a run reproduces.
    """

    def __init__(self, path: str, split: str = "train", column: str = "obs_poses"):
        from datasets import load_from_disk

        data = load_from_disk(path)
        if hasattr(data, "keys"):
            if split not in data:
                raise KeyError(
                    f"split {split!r} not in {path}; have {sorted(data.keys())}"
                )
            data = data[split]
        if column not in data.column_names:
            raise KeyError(
                f"column {column!r} not in {path}:{split}; have {data.column_names}"
            )
        self.data = data
        self.column = column
        self.path = path
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def draw(self, rng) -> np.ndarray:
        row = self.data[int(rng.integers(len(self.data)))]
        return np.asarray(row[self.column], dtype=np.float64).reshape(-1, 3)

    def walk(self, seed: int) -> "_ForeignWalk":
        return _ForeignWalk(self, np.random.default_rng(int(seed)))

    def describe(self) -> str:
        return (f"pose source=FOREIGN {self.path}:{self.split}[{self.column}] "
                f"({len(self)} trajectories) -- diagnostic, not a policy")


class _ForeignWalk:
    """One episode's worth of foreign poses, extended on demand."""

    def __init__(self, source: _ForeignPose, rng):
        self.source = source
        self.rng = rng
        self.buffer = source.draw(rng)
        self.index = 0
        self.trajectories = 1

    def _extend(self) -> None:
        """Append another trajectory, rebased to continue from the current end."""
        import math

        nxt = self.source.draw(self.rng)
        if len(nxt) < 2:
            return
        end = self.buffer[-1]
        d_theta = end[2] - nxt[0, 2]
        cos, sin = math.cos(d_theta), math.sin(d_theta)
        rel = nxt[1:, :2] - nxt[0, :2]
        moved = np.empty_like(nxt[1:])
        moved[:, 0] = cos * rel[:, 0] - sin * rel[:, 1] + end[0]
        moved[:, 1] = sin * rel[:, 0] + cos * rel[:, 1] + end[1]
        moved[:, 2] = nxt[1:, 2] + d_theta
        self.buffer = np.concatenate([self.buffer, moved], axis=0)
        self.trajectories += 1

    def next(self) -> list:
        while self.index >= len(self.buffer):
            before = len(self.buffer)
            self._extend()
            if len(self.buffer) == before:  # degenerate rows only; avoid spinning
                return [float(v) for v in self.buffer[-1]]
        pose = self.buffer[self.index]
        self.index += 1
        return [float(v) for v in pose]


class _PoseFeed:
    """Per-turn agent pose, for a checkpoint whose config declares a pose modality.

    What crosses the wire is the **raw scene-frame** ``[X, Y, theta]`` the
    simulator reports, and the conversion to the frame the model was trained on
    happens inside ``VectorRolloutPolicy.pose_values`` -> ``pose_frame.
    relative_se2``. That is the whole point, and it is worth being explicit
    about why the conversion is not done here:

    the model is shown pose **relative to the first observation in context**,
    expressed in that observation's own frame. Training derives that at collate
    time from the raw ``obs_poses`` column via the spec's
    ``pose_relative_first`` transform, which is a thin binding to the same
    ``relative_se2``. So there is exactly one implementation of the convention
    and both paths call it. Reimplementing it here -- even correctly -- would
    create a second one, and a divergence between the two does not raise: the
    model reads confident nonsense and the run produces a number. Two ways it
    would go wrong that no assertion downstream would catch: a *previous-frame
    delta* instead of an episode-start anchor (plausible values, wrong
    semantics), and an unwrapped heading difference, which on this corpus
    reaches 5.955 rad where the true rotation is 0.400 rad.

    Detection is by **transform, not token name**: ``_resolve_pose_spec``
    already picks the single spec declaring ``pose_relative_first``, because
    that transform *is* the contract ``step(obs_pose=...)`` implements. A spec
    called ``<pose>`` that did not declare it would need its values supplied
    some other way, and would correctly not be fed from here.

    **Supplying the value is only half of it: the marker has to be in the
    text.** ``RolloutConfig.modality_marker`` defaults to ``None``, and nothing
    on the model side fills it in -- ``from_checkpoint`` does not, because the
    marker belongs to how a *turn is rendered*, not to the weights. Left unset,
    the values are supplied with no marker to bind them to and the embedder
    raises "modality values were provided but nothing was injected", on the very
    first step of every episode. So this class installs it, from
    ``spec.token``, never from a literal here: the token string is the spec's
    identity, and a copy of it in this file is a second place for it to be
    wrong. `format_action_chunk_dataset.py` and the rollout both build the turn
    through `vector_rollout.user_block_content`, so setting the marker
    reproduces the training message structure exactly -- position included,
    which matters because a marker directly after ``<|vision_start|>`` would
    silently corrupt every position id downstream.

    Inert on every checkpoint that declares no such spec: no marker is
    installed, ``step_kwargs`` returns an empty dict, and the ``step()`` call is
    character for character the one that ran before this existed.
    """

    def __init__(self, policy: Any, corruption: Optional["_PoseCorruption"] = None,
                 foreign: Optional["_ForeignPose"] = None, seed: int = 0):
        # `_pose_spec` is None on any checkpoint without the modality mechanism,
        # and the attribute itself is absent on a longnav old enough to predate
        # it -- both mean "this checkpoint does not want pose".
        self.spec = getattr(policy, "_pose_spec", None)
        self.fed = 0
        self.corruption = corruption or _PoseCorruption()
        self.foreign = foreign
        self.seed = int(seed)
        self.walk: Optional[_ForeignWalk] = None
        self.episodes = 0
        self.marker: Optional[str] = None
        if self.spec is not None:
            self._install_marker(policy)

    def _install_marker(self, policy: Any) -> None:
        """Write this spec's token into every turn's user block."""
        config = getattr(policy, "cfg", None)
        if config is None or not hasattr(config, "modality_marker"):
            raise RuntimeError(
                f"this checkpoint declares modality {self.spec.token}, but its "
                "RolloutConfig has no `modality_marker` field -- the longnav checkout on "
                "--policy-sys-path predates the modality mechanism and cannot render the "
                "marker into a turn. Point it at one that has it."
            )
        existing = config.modality_marker
        if existing is None:
            config.modality_marker = self.spec.token
        elif existing != self.spec.token:
            raise ValueError(
                f"the rollout config already writes marker {existing!r} into each turn, "
                f"but the checkpoint's spec is {self.spec.token!r}. One of them is not "
                "what the model was trained on; refusing to guess which."
            )
        self.marker = config.modality_marker

    def __bool__(self) -> bool:
        return self.spec is not None

    def on_episode_reset(self) -> None:
        self.fed = 0
        # A fresh draw per episode, seeded like every other stochastic part of a
        # rollout so the run reproduces.
        if self.foreign is not None:
            self.walk = self.foreign.walk(self.seed + self.episodes)
        self.episodes += 1

    def step_kwargs(self, policy: Any,
                    pose: Optional[Sequence[float]]) -> Dict[str, Any]:
        if self.spec is None:
            return {}
        if pose is None:
            raise RuntimeError(
                f"this checkpoint declares modality {self.spec.token} with the "
                "'pose_relative_first' transform, so every turn needs the agent's raw "
                "planar [X, Y, theta] -- but the caller sent none. The marker is in the "
                "text either way, so continuing would inject nothing and the model would "
                "read a meaningless embedding. (Sim side: Observation.pose must reach "
                "PolicyBridge.act; in-process: pass act(rgb, pose=...).)"
            )
        values = [float(v) for v in pose]
        if len(values) != 3:
            raise ValueError(
                f"pose must be 3 numbers [X, Y, theta], got {len(values)}: {values}"
            )
        if self.walk is not None:
            # Substituted BEFORE the frame conversion, so the foreign poses go
            # through the identical relative_se2 the real ones would have. The
            # robot's actual pose is simply discarded.
            values = self.walk.next()
        self.fed += 1
        if not self.corruption:
            # Production path: hand the raw pose over and let the model's own
            # `pose_values` -> `relative_se2` do the conversion, untouched.
            return {"obs_pose": values}
        # Corrupting: take the relative value from the SAME function, then map
        # it. `pose_values` also records the raw pose, so the episode-start
        # anchor stays correct. `obs_pose=` and `modality[key]` are mutually
        # exclusive by design, so the corrupted value goes through `modality`.
        return {"modality": {self.spec.key: self.corruption.apply(policy.pose_values(values))}}

    def describe(self) -> str:
        if self.spec is None:
            return "pose=none (no spec declares 'pose_relative_first')"
        return (
            f"pose={self.spec.token} raw [X, Y, theta] -> relative_se2 vs this "
            f"episode's first observation (transform={self.spec.transform}, "
            f"column={self.spec.source_column}, marker in user block={self.marker!r})"
            + (f"\n  [{self.corruption.describe()}]" if self.corruption else "")
            + (f"\n  [{self.foreign.describe()}]" if self.foreign is not None else "")
        )


class _StopHeadControl:
    """The checkpoint's stop classifier, and the runtime knobs on it.

    Off unless asked for, and that default is the point rather than caution.
    Every evaluation in this project so far ran with the harness's oracle
    ``--auto-stop``, which ends an episode on arrival; a stop head firing under
    that regime would be indistinguishable from the oracle in the metrics, and
    the two would silently share credit for the same episode. So ``enabled``
    gates only whether :meth:`decided` can return True -- the probability is
    computed and reported either way, which is what makes a threshold fittable
    after the fact rather than by re-running.

    ``temperature`` / ``threshold`` / ``inference`` are written onto the loaded
    ``StopHeadConfig``. They are runtime knobs by design (``stop_head.py``:
    temperature "is not trained and does not change AP or AUC"), so a
    miscalibrated head is a scalar to fit on saved scores, not a failed
    experiment. They are validated here because assigning to a dataclass field
    after construction bypasses its ``__post_init__``.
    """

    MODES = ("sample", "threshold", "argmax")

    def __init__(
        self,
        policy: Any,
        enabled: bool = False,
        temperature: Optional[float] = None,
        threshold: Optional[float] = None,
        inference: Optional[str] = None,
    ):
        self.head = getattr(getattr(policy, "model", None), "stop_head", None)
        self.enabled = bool(enabled)
        if self.enabled and self.head is None:
            raise ValueError(
                "--stop-head was requested but this checkpoint has no stop head "
                "(model.stop_head is None). Nothing would ever end the episode; use "
                "--auto-stop for the oracle stop instead."
            )
        if self.head is None:
            return
        cfg = self.head.cfg
        if inference is not None:
            if inference not in self.MODES:
                raise ValueError(
                    f"stop inference must be one of {self.MODES}, got {inference!r}"
                )
            cfg.inference = inference
        if temperature is not None:
            if float(temperature) <= 0:
                raise ValueError(f"stop temperature must be > 0, got {temperature}")
            cfg.temperature = float(temperature)
        if threshold is not None:
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError(f"stop threshold must be in [0, 1], got {threshold}")
            cfg.threshold = float(threshold)

    def __bool__(self) -> bool:
        return self.head is not None

    def seed(self, policy: Any, seed: int) -> None:
        """Pin the sampled decision for this episode.

        Only meaningful for ``inference == 'sample'``, and harmless otherwise --
        but only called when sampling, so a threshold run does not allocate a
        generator it will never read.
        """
        if self.head is not None and self.head.cfg.inference == "sample":
            policy.seed_stop(int(seed))

    def annotate(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Add ``stop_logit`` -- the *uncalibrated* score -- next to ``stop_prob``.

        The recorded probability is already temperature-scaled, and the episode
        record rounds floats to 3 decimals, so a head whose probabilities all sit
        in (0.000, 0.005) would be saved as a column of zeros and no threshold
        could be fitted on it afterwards. The logit has none of that dynamic-range
        problem, and it is the quantity a post-hoc fit actually wants: temperature
        divides it, so refitting T offline is only possible from the logit.

        Recovered by inverting the reported probability rather than by a second
        forward pass, which makes it exactly the number the head produced.
        Saturated probabilities (|logit| beyond about 17 in float32) are clamped
        and reported at the clamp; that is a limit of the recorded value, not of
        the head, and it only bites where the decision is not in doubt anyway.
        """
        if self.head is None:
            return stats
        prob = stats.get("stop_prob")
        if prob is None:
            return stats
        import math

        p = min(max(float(prob), 1e-7), 1.0 - 1e-7)
        stats["stop_logit"] = float(self.head.cfg.temperature) * math.log(p / (1.0 - p))
        return stats

    def decided(self, stats: Dict[str, Any]) -> bool:
        """The policy's own termination decision, or False when the head is off.

        Reads ``last_stats['stop']``, which ``VectorRolloutPolicy.step`` already
        computed under the configured inference rule. Recomputing it here would
        be a second decision rule for the same logit.
        """
        return bool(self.enabled and stats.get("stop"))

    def describe(self) -> str:
        if self.head is None:
            return "stop head=none (oracle --auto-stop is the only termination)"
        cfg = self.head.cfg
        state = "ACTIVE" if self.enabled else "reported only (not acting)"
        return (
            f"stop head={state} inference={cfg.inference} "
            f"temperature={cfg.temperature} threshold={cfg.threshold}"
        )


def _wire_safe(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Make ``last_stats`` survive the socket.

    ``VectorRolloutPolicy.last_stats['pose_value']`` is a **torch tensor** (the
    value actually injected, kept so a decodability probe does not have to
    re-derive the frame). The sim environment has no torch, and the two
    environments also disagree about numpy, so anything but plain Python here
    fails at ``conn.recv()`` -- as an unpickling error a whole episode away from
    its cause. Converted rather than dropped: it is the one number that proves
    what the model was shown.
    """
    out: Dict[str, Any] = {}
    for key, value in stats.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


class VectorRolloutBackend(PolicyBackend):
    """The turn-vector action-chunk head (``longnav.utils.vector_rollout``).

    The import is inside ``__init__`` so this module stays importable in the sim
    environment, where ``longnav``/``torch`` do not exist.
    """

    def __init__(
        self,
        ckpt: str,
        device: str = "cuda",
        merge_lora: bool = True,
        max_context_tokens: int = 0,
        placeholder: Optional[str] = None,
        seed: int = 0,
        stop_head: bool = False,
        stop_temperature: Optional[float] = None,
        stop_threshold: Optional[float] = None,
        stop_inference: Optional[str] = None,
        pose_mirror: bool = False,
        pose_rotate_deg: float = 0.0,
        pose_foreign: Optional[str] = None,
        pose_foreign_split: str = "train",
    ):
        from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy

        config = RolloutConfig(
            device=device, merge_lora=merge_lora, max_context_tokens=max_context_tokens
        )
        if placeholder:
            config.placeholder = placeholder
        self.policy = VectorRolloutPolicy.from_checkpoint(ckpt, config)
        self.seed = int(seed)
        self._episode_count = 0
        # Both read the checkpoint and are inert on one that declares neither, so
        # a plain vector head loads and runs exactly as it did before.
        self._pose = _PoseFeed(
            self.policy, _PoseCorruption(pose_mirror, pose_rotate_deg),
            _ForeignPose(pose_foreign, pose_foreign_split) if pose_foreign else None,
            seed=int(seed),
        )
        self._stop = _StopHeadControl(
            self.policy, enabled=stop_head, temperature=stop_temperature,
            threshold=stop_threshold, inference=stop_inference,
        )

    def reset(self, goal_text: Optional[str]) -> None:
        self.policy.reset(goal_text=goal_text)
        self._pose.on_episode_reset()
        self._stop.seed(self.policy, self.seed + self._episode_count)
        self._episode_count += 1

    def act(self, rgb: np.ndarray, pose: Optional[Sequence[float]] = None) -> np.ndarray:
        from PIL import Image

        chunk = self.policy.step(
            Image.fromarray(rgb), **self._pose.step_kwargs(self.policy, pose)
        )
        return np.asarray(chunk.numpy(), dtype=np.float32)

    @property
    def last_stats(self) -> Dict[str, Any]:
        return self._stop.annotate(_wire_safe(self.policy.last_stats))

    def wants_stop(self) -> bool:
        return self._stop.decided(self.policy.last_stats)

    def describe(self) -> str:
        return (
            f"{self.policy.describe()}\n"
            f"  [{self._pose.describe()}]\n"
            f"  [{self._stop.describe()}]"
        )


register_backend("vector_rollout", VectorRolloutBackend)


class BinRolloutBackend(PolicyBackend):
    """The discrete-bin control head (``longnav.utils.bin_rollout``), decode configurable.

    Reuses ``VectorRolloutPolicy`` unmodified -- see ``bin_rollout.load_bin_policy``: a
    ``TurnBinClassifier``'s ``normalizer`` slot is a ``BinCodec``, whose ``denormalize``
    already turns logits into a real ``(T, 3)`` chunk under whichever decode rule
    (``argmax`` / ``sample`` / ``mean``) it is set to. The only thing this backend adds is
    fixed-per-episode seeding for ``sample``: ``reset()`` reseeds the codec's fallback
    generator from ``seed + <episode count>``, so repeat runs of the same episode sequence
    under ``decode="sample"`` are reproducible, and successive episodes in one process do
    not share one continuously-mutating RNG stream.
    """

    def __init__(
        self,
        ckpt: str,
        device: str = "cuda",
        merge_lora: bool = True,
        max_context_tokens: int = 0,
        placeholder: Optional[str] = None,
        decode: str = "argmax",
        seed: int = 0,
        stop_head: bool = False,
        stop_temperature: Optional[float] = None,
        stop_threshold: Optional[float] = None,
        stop_inference: Optional[str] = None,
        pose_mirror: bool = False,
        pose_rotate_deg: float = 0.0,
        pose_foreign: Optional[str] = None,
        pose_foreign_split: str = "train",
    ):
        from longnav.utils.bin_rollout import load_bin_policy, seed_sampling
        from longnav.utils.vector_rollout import RolloutConfig

        self._seed_sampling = seed_sampling
        config = RolloutConfig(
            device=device, merge_lora=merge_lora, max_context_tokens=max_context_tokens
        )
        if placeholder:
            config.placeholder = placeholder
        self.decode = decode
        self.seed = int(seed)
        self._episode_count = 0
        self.policy = load_bin_policy(ckpt, config, decode=decode)
        self._pose = _PoseFeed(
            self.policy, _PoseCorruption(pose_mirror, pose_rotate_deg),
            _ForeignPose(pose_foreign, pose_foreign_split) if pose_foreign else None,
            seed=int(seed),
        )
        self._stop = _StopHeadControl(
            self.policy, enabled=stop_head, temperature=stop_temperature,
            threshold=stop_threshold, inference=stop_inference,
        )

    def reset(self, goal_text: Optional[str]) -> None:
        self.policy.reset(goal_text=goal_text)
        if self.decode == "sample":
            self._seed_sampling(self.policy, self.seed + self._episode_count)
        self._pose.on_episode_reset()
        self._stop.seed(self.policy, self.seed + self._episode_count)
        self._episode_count += 1

    def act(self, rgb: np.ndarray, pose: Optional[Sequence[float]] = None) -> np.ndarray:
        from PIL import Image

        chunk = self.policy.step(Image.fromarray(rgb), **self._pose.step_kwargs(self.policy, pose))
        return np.asarray(chunk.numpy(), dtype=np.float32)

    @property
    def last_stats(self) -> Dict[str, Any]:
        return self._stop.annotate(_wire_safe(self.policy.last_stats))

    def wants_stop(self) -> bool:
        return self._stop.decided(self.policy.last_stats)

    def describe(self) -> str:
        return (
            f"{self.policy.describe()} [bin decode={self.decode} seed_base={self.seed}]\n"
            f"  [{self._pose.describe()}]\n"
            f"  [{self._stop.describe()}]"
        )


register_backend("bin_rollout", BinRolloutBackend)


class _PeriodicContextReset:
    """Memory-length ablation: drop rollout history every ``every`` policy steps.

    ``VectorRolloutPolicy.reset()`` clears ``past_key_values`` **and**
    ``past_image_embeds``, zeroes ``rope_offset`` and ``step_index``, and
    re-stakes the tokenised system prompt. So the step after a reset is
    byte-for-byte the shape of an episode's first step: prompt retained, history
    gone, positions restarting at 0. No KV surgery, no RoPE hole.

    Memory is a **sawtooth, not a window**: within a block the available history
    runs 0 -> ``every``-1 steps, so the arm measures an average memory of about
    half the block, not a constant one. Sufficient to answer "does history
    matter at all", imprecise for "how much".

    Two things it deliberately does not do:

    * It calls ``policy.reset()``, **never the backend's** ``reset()``. The
      latter reseeds the per-episode sampling generator; re-running it
      mid-episode would restart the RNG stream and correlate the chunks either
      side of a block boundary, corrupting the very thing being measured.
    * With ``every == 0`` (the default) ``before_act`` returns before touching
      anything, so the unlimited-memory path is exactly what it was.

    Known confounds, recorded because they are reasons a result could move that
    have nothing to do with image memory: ``step_index`` restarts, so the policy
    always sees "Observation 0, 1, 2..." and cannot know how long it has been
    searching; and a policy that opens an episode with a sweep will repeat that
    sweep every block.

    On a pose-injecting checkpoint ``reset()`` also clears ``_raw_poses``, so
    each block gets a fresh pose origin. That is the *correct* reading rather
    than a third confound -- ``pose_frame.py`` defines the value as relative to
    the first observation **in context**, and a block is exactly a context --
    but it does mean this ablation removes metric history along with visual
    history, so the two cannot be separated by running it.
    """

    def __init__(self, every: int = 0):
        self.every = max(0, int(every))
        self.goal: Optional[str] = None
        self.step = 0
        self.resets = 0

    def on_episode_reset(self, goal_text: Optional[str]) -> None:
        self.goal, self.step, self.resets = goal_text, 0, 0

    def before_act(self, policy) -> None:
        if not self.every:
            return  # production path: unlimited memory, untouched
        if self.step and self.step % self.every == 0:
            policy.reset(goal_text=self.goal)
            self.resets += 1
        self.step += 1


class ARRolloutBackend(PolicyBackend):
    """The autoregressive-over-ticks action head (``longnav.utils.ar_action_head``).

    Structurally the same backend as :class:`BinRolloutBackend` -- load the checkpoint,
    fix a decode rule, reseed once per episode -- for the same reason: a
    ``TurnARActionClassifier`` also occupies ``VectorRolloutPolicy``'s ``normalizer``
    slot, so ``step()`` needs no override. Its last line calls
    ``normalizer.denormalize(...)``, and ``ARActionCodec`` runs the sequential decode
    there (T forward passes through the tiny causal decoder, each tick's chosen code fed
    back in) and composes a real ``(T, 3)`` chunk. The wire format is therefore unchanged
    and nothing on the simulator side knows this head exists.

    There is no ``longnav.utils.ar_rollout`` to import the way ``bin_rollout`` provides
    ``load_bin_policy``, so the four lines that build the policy live here.

    Three differences from the bin head, all of which fail *silently* if copied across:

    * **The decode rule lives on the normalizer, not on ``model.codec``.** For a
      ``TurnBinClassifier`` those are one object; here ``model.codec`` is the
      ``ChunkVQCodec`` lookup table *inside* the ``ARActionCodec``. Writing
      ``model.codec.decode`` sets an attribute nothing reads, and the head then decodes
      under its own default rather than the requested rule. ``ChunkVQCodec.DECODES`` is a
      different tuple as well, so validating against it would admit ``"logits"``, which
      this head does not implement.
    * **``"context"`` is not a policy.** It is ``ARActionCodec``'s passthrough for offline
      analysis (``dump/autoregressive_head/predict_ar.py``), returning the raw
      ``(context_dim,)`` trunk vector. Rejected here rather than sent down the wire as a
      256-long "chunk" that the executor would happily try to drive.
    * **The default decode is ``"sample"``**, not the bin backend's ``"argmax"``. It is
      this codec's own default and the only rule the head's design treats as an executable
      sequential policy. ``"mean"`` is accepted because it is the control-within-the-
      control -- if creeping returns under mean decode from these same weights, averaging
      is the failure mode independent of architecture -- but it is computed from a
      greedy-decoded prefix and is reporting-only, not a deployable rule.

    Temperature stays at the codec's default of 1.0; there is no CLI flag for it.

    **Decoder architecture is checkpoint-carried, and deliberately not this backend's
    business.** ``CausalActionDecoder`` has grown options -- multi-token context
    (``n_context_tokens``), a per-tick pose/velocity state (``use_prefix_pose``), a
    ``markov`` attention mode, and ``direct_context``, which sets ``context_proj`` to
    ``None`` outright -- all defaulting OFF so older checkpoints load unchanged. None of
    them reaches this class: ``from_pretrained`` rebuilds the decoder from the
    checkpoint's own ``ar_decoder_kwargs``, and the only decoder attribute written here
    is ``normalizer.decode``. ``use_prefix_pose`` is the one with an external
    requirement -- it needs ``centroids`` handed to ``decoder.forward``/``decode`` -- and
    ``ARActionCodec.denormalize`` already supplies them, on the ``"mean"`` path as well
    as the sequential one. So the backend is architecture-agnostic by construction, and
    the only thing it owes a new architecture is *saying which one it loaded*: see
    :meth:`describe`.
    """

    def __init__(
        self,
        ckpt: str,
        device: str = "cuda",
        merge_lora: bool = True,
        max_context_tokens: int = 0,
        placeholder: Optional[str] = None,
        decode: str = "sample",
        seed: int = 0,
        context_reset_every: int = 0,
        stop_head: bool = False,
        stop_temperature: Optional[float] = None,
        stop_threshold: Optional[float] = None,
        stop_inference: Optional[str] = None,
        pose_mirror: bool = False,
        pose_rotate_deg: float = 0.0,
        pose_foreign: Optional[str] = None,
        pose_foreign_split: str = "train",
    ):
        # Checked before the imports: a bad decode rule should fail the same way in any
        # interpreter, not only in one that can already load the checkpoint.
        if decode not in ("argmax", "sample", "mean"):
            raise ValueError(
                f"decode must be one of ('argmax', 'sample', 'mean'), got {decode!r}"
                + (" -- 'context' is the offline passthrough, not an executable policy"
                   if decode == "context" else "")
            )

        from transformers import AutoProcessor

        from longnav.utils.ar_action_head import TurnARActionClassifier
        from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy

        config = RolloutConfig(
            device=device, merge_lora=merge_lora, max_context_tokens=max_context_tokens
        )
        if placeholder:
            config.placeholder = placeholder
        self.decode = decode
        self.seed = int(seed)
        self._episode_count = 0
        # 0 = unlimited memory (production default); see _PeriodicContextReset.
        self._context = _PeriodicContextReset(context_reset_every)

        processor = AutoProcessor.from_pretrained(str(ckpt))
        model = TurnARActionClassifier.from_pretrained(
            ckpt, processor, dtype=config.dtype, device=config.device
        )
        model.normalizer.decode = decode
        self.policy = VectorRolloutPolicy(model, processor, config)
        # Read off the checkpoint, inert when it declares neither -- so every AR
        # checkpoint evaluated before these existed runs the identical path.
        self._pose = _PoseFeed(
            self.policy, _PoseCorruption(pose_mirror, pose_rotate_deg),
            _ForeignPose(pose_foreign, pose_foreign_split) if pose_foreign else None,
            seed=int(seed),
        )
        self._stop = _StopHeadControl(
            self.policy, enabled=stop_head, temperature=stop_temperature,
            threshold=stop_threshold, inference=stop_inference,
        )

    def _reseed(self, seed: int) -> None:
        """Reseed ``ARActionCodec.generator`` -- the sticky fallback ``denormalize`` hands
        to the decoder's ``torch.multinomial``. The generator's device must match the
        logits' (they are produced by the decoder, which lives wherever the model does),
        or ``multinomial`` refuses the pair outright."""
        import torch

        device = next(self.policy.model.decoder.parameters()).device
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(int(seed))
        self.policy.model.normalizer.generator = generator

    def reset(self, goal_text: Optional[str]) -> None:
        self.policy.reset(goal_text=goal_text)
        if self.decode == "sample":
            self._reseed(self.seed + self._episode_count)
        self._pose.on_episode_reset()
        self._stop.seed(self.policy, self.seed + self._episode_count)
        self._episode_count += 1
        self._context.on_episode_reset(goal_text)

    def act(self, rgb: np.ndarray, pose: Optional[Sequence[float]] = None) -> np.ndarray:
        from PIL import Image

        # No-op unless --context-reset-every was passed; see _PeriodicContextReset.
        self._context.before_act(self.policy)
        chunk = self.policy.step(Image.fromarray(rgb), **self._pose.step_kwargs(self.policy, pose))
        return np.asarray(chunk.numpy(), dtype=np.float32)

    @property
    def last_stats(self) -> Dict[str, Any]:
        stats = self._stop.annotate(_wire_safe(self.policy.last_stats))
        context = getattr(self, "_context", None)
        if context is not None and context.every:
            stats["context_resets"] = context.resets
        return stats

    def wants_stop(self) -> bool:
        return self._stop.decided(self.policy.last_stats)

    #: Decoder options that default OFF, so a checkpoint predating them looks exactly like
    #: one that declined them. Printed with their defaults filled in precisely because of
    #: that: the eval log has to say which architecture ran, and "absent" and "off" are
    #: indistinguishable in the stored config.
    AR_DECODER_KEYS = (
        ("n_context_tokens", 1),
        ("use_prefix_pose", False),
        ("attn_mode", "causal"),
        ("direct_context", False),
    )

    def describe(self) -> str:
        """One line of decode policy, one of decoder architecture.

        The architecture line reads ``CausalActionDecoder.to_config()`` -- the same dict
        ``save_pretrained`` stores as ``ar_decoder_kwargs`` and ``from_pretrained``
        rebuilds the decoder from -- so it reports what was actually loaded rather than
        restating the flags this backend was called with. Without it a v2 run and a v3 run
        leave identical eval logs (``n_codes``/``n_ticks`` are the same across them), and
        comparing the two afterwards is guesswork.
        """
        model = self.policy.model
        cfg = model.decoder.to_config()
        arch = " ".join(f"{key}={cfg.get(key, default)}"
                        for key, default in self.AR_DECODER_KEYS)
        # getattr: the test suite builds a partial backend with __new__ to avoid
        # loading a multi-billion-parameter VLM, and describe() has to keep
        # working on it -- that is the point of pinning it there.
        context = getattr(self, "_context", None)
        memory = (f"memory={context.every} steps/block"
                  if context is not None and context.every else "memory=unlimited")
        pose = getattr(self, "_pose", None)
        stop = getattr(self, "_stop", None)
        return (
            f"{self.policy.describe()}\n"
            f"  [ar decode={self.decode} seed_base={self.seed} "
            f"n_codes={model.n_codes} n_ticks={model.n_ticks} {memory}]\n"
            f"  [ar decoder d_model={cfg.get('d_model')} n_layers={cfg.get('n_layers')} "
            f"n_heads={cfg.get('n_heads')} {arch}]"
            + (f"\n  [{pose.describe()}]" if pose is not None else "")
            + (f"\n  [{stop.describe()}]" if stop is not None else "")
        )


register_backend("ar_rollout", ARRolloutBackend)


def serve(socket_path: str, backend: PolicyBackend) -> None:
    """Answer reset/act over ``socket_path`` until told to stop.

    Errors are reported back over the socket rather than killed silently, so the
    simulator side raises with the real traceback instead of hanging or seeing a
    dead pipe with no explanation.
    """
    from multiprocessing.connection import Client

    conn = Client(socket_path, family="AF_UNIX", authkey=AUTHKEY)
    print("connected to sim process", flush=True)
    try:
        while True:
            try:
                message = conn.recv()
            except EOFError:
                break
            try:
                command = message["cmd"]
                if command == CMD_STOP:
                    break
                if command == CMD_RESET:
                    backend.reset(message.get("goal_text"))
                    conn.send({"ok": True})
                    continue
                rgb = np.frombuffer(message["data"], dtype=message["dtype"]).reshape(
                    message["shape"]
                )
                # A plain list of three floats, not an array: the two environments
                # disagree about numpy (1.26 vs 2.x) and this has to unpickle in
                # both. `.get` rather than `[...]`, so a sim side predating pose
                # still talks to a server that has it -- the backend raises with
                # a real explanation if its checkpoint actually needed one.
                chunk = np.asarray(
                    backend.act(rgb, pose=message.get("pose")), dtype=np.float32
                )
                conn.send(
                    {
                        "chunk": chunk.tobytes(),
                        "chunk_shape": list(chunk.shape),
                        "stop": bool(backend.wants_stop()),
                        "stats": dict(backend.last_stats),
                    }
                )
            except Exception as exc:  # report, don't die silently on the sim side
                import traceback

                traceback.print_exc()
                conn.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()
    print("policy server exiting", flush=True)


# ---------------------------------------------------------------------------
# Client side: the handle the simulator process holds
# ---------------------------------------------------------------------------


class PolicyBridge:
    """Launches ``policy_python scripts/policy_server.py`` and talks to it.

    The listener is created *before* the subprocess starts and ``accept()``
    blocks until the model has loaded and connected, so a checkpoint that fails
    to load surfaces as a startup error with the server log attached rather than
    as a timeout mid-episode.
    """

    def __init__(
        self,
        server_script: str,
        policy_python: str,
        backend_args: Dict[str, Any],
        log_path: Optional[str] = None,
    ):
        from multiprocessing.connection import Listener

        self.socket_path = Path(tempfile.mkdtemp(prefix="objectnav_policy_")) / "sock"
        self.log_path = (
            Path(log_path) if log_path else self.socket_path.with_suffix(".log")
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        listener = Listener(str(self.socket_path), family="AF_UNIX", authkey=AUTHKEY)

        command = [str(policy_python), str(server_script), "--socket", str(self.socket_path)]
        for key, value in backend_args.items():
            if value is None or value is False:
                continue
            flag = "--" + key.replace("_", "-")
            command += [flag] if value is True else [flag, str(value)]
        print(f"[bridge] launching policy server: {' '.join(command)}", flush=True)
        print(f"[bridge] server log -> {self.log_path}", flush=True)
        self._log = open(self.log_path, "w")
        self.process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)
        try:
            self.conn = self._accept(listener)
        except BaseException:
            self.process.kill()
            raise
        finally:
            listener.close()
        print("[bridge] policy server connected", flush=True)

    def _accept(self, listener):
        """``listener.accept()``, but noticing if the server dies first.

        A bare ``accept()`` blocks forever when the model-side interpreter
        crashes on startup (a missing import, a checkpoint it cannot load), and
        the run then hangs with no output at all. Accepting on a thread lets the
        main thread poll the subprocess and turn that into the real error, with
        the server's traceback attached.
        """
        import threading

        outcome: Dict[str, Any] = {}

        def _run() -> None:
            try:
                outcome["conn"] = listener.accept()
            except BaseException as exc:  # noqa: BLE001 - reported below
                outcome["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        while True:
            thread.join(1.0)
            if not thread.is_alive():
                break
            code = self.process.poll()
            if code is not None:
                tail = ""
                try:
                    self._log.flush()
                    tail = Path(self.log_path).read_text()[-3000:]
                except OSError:
                    pass
                raise RuntimeError(
                    f"policy server exited with code {code} before connecting; "
                    f"see {self.log_path}\n{tail}"
                )
        if "error" in outcome:
            raise outcome["error"]
        return outcome["conn"]

    def _rpc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.conn.send(payload)
            reply = self.conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            tail = ""
            try:
                tail = Path(self.log_path).read_text()[-2000:]
            except OSError:
                pass
            raise RuntimeError(
                f"policy server died; see {self.log_path}\n{tail}"
            ) from exc
        if reply.get("error"):
            raise RuntimeError(f"policy server error: {reply['error']}")
        return reply

    def reset(self, goal_text: Optional[str]) -> None:
        self._rpc({"cmd": CMD_RESET, "goal_text": goal_text})

    def act(self, rgb: np.ndarray, pose: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        reply = self._rpc(
            {
                "cmd": CMD_ACT, "shape": rgb.shape, "dtype": "uint8", "data": rgb.tobytes(),
                # Raw planar [X, Y, theta] straight from `get_2d_pose()`. Sent
                # unconditionally and cheaply (three floats); what to do with it is
                # the checkpoint's business, not the simulator's -- see _PoseFeed.
                "pose": None if pose is None else [float(v) for v in pose],
            }
        )
        chunk = (
            np.frombuffer(reply["chunk"], dtype=np.float32)
            .reshape(reply["chunk_shape"])
            .astype(np.float64)
        )
        return {"chunk": chunk, "stop": bool(reply.get("stop", False)),
                "stats": reply.get("stats", {})}

    def close(self) -> None:
        try:
            self.conn.send({"cmd": CMD_STOP})
        except Exception:
            pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self._log.close()
