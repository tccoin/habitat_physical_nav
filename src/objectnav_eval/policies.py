"""What drives the robot: the policy interface and the harness's own policies.

A policy is asked for an :class:`ActionChunk` given an :class:`Observation`, and
that is the whole contract. Deliberately:

* the chunk is a block of **relative** poses, so a policy never needs to know
  where it is in the world (the executor anchors on the robot's live pose);
* ``ActionChunk.stop`` lets a policy with a termination head end the episode,
  without every other policy having to have one;
* ``ActionChunk.extra`` carries head-specific outputs (a flow head's log-prob or
  sample set, a diffusion head's denoising trace) into the per-episode record
  without widening the interface.

The image-blind controls are not baselines worth reporting -- they are how you
find out the harness itself is sound. ``forward`` drives the base straight
ahead; if that does not reduce the distance to a goal in front of it, chunk
decoding, anchoring or PID tracking is broken and no model number measured
through this loop means anything. ``zero`` stands still, which bounds what a
policy that has learned nothing scores and pins path length at ~0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class Observation:
    """Everything the harness offers a policy at one decision point."""

    rgb: np.ndarray
    goal: Optional[str]
    step: int
    pose: np.ndarray  # planar [X, Y, theta]; proprioceptive policies may use it
    sim_time: float


@dataclass
class ActionChunk:
    """A block of future poses, relative to the pose at emission time.

    ``poses`` is ``(T, 3)`` of ``[dx, dy, dtheta]`` in the emitting pose's own
    frame -- the same encoding the training data uses (see
    ``continuous_demos.action_chunking``). The executor consumes the first
    ``gap`` rows and discards the rest, exactly as in training, because the
    remainder would be superseded by the next observation.
    """

    poses: np.ndarray
    stop: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.poses = np.asarray(self.poses, dtype=np.float64).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.poses)


class Policy:
    """Interface every policy implements."""

    name: str = "abstract"
    #: False for image-blind controls; recorded in the results so a control run
    #: can never be mistaken for a model run.
    uses_observation: bool = True

    def reset(self, goal: Optional[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def act(self, obs: Observation) -> ActionChunk:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Harness controls
# ---------------------------------------------------------------------------


class ConstantChunkPolicy(Policy):
    """CONTROL: ignores the image and emits a fixed chunk.

    Each row is an offset from the **same** anchor (the pose at chunk emission),
    so a constant-velocity chunk ramps linearly rather than repeating one
    offset. Getting that wrong makes ``forward`` crawl at one tick's worth of
    displacement per chunk and look like a broken executor.
    """

    uses_observation = False

    def __init__(self, kind: str = "zero", chunk_len: int = 10, dt: float = 0.05,
                 speed: float = 1.0):
        if kind not in ("forward", "zero"):
            raise ValueError(f"unknown control policy '{kind}'")
        self.name = kind
        self.kind = kind
        step = speed * dt
        chunk = np.zeros((chunk_len, 3), dtype=np.float64)
        if kind == "forward":
            chunk[:, 0] = step * np.arange(1, chunk_len + 1)
        self._chunk = chunk
        self._speed = speed

    def reset(self, goal: Optional[str]) -> None:
        pass

    def act(self, obs: Observation) -> ActionChunk:
        return ActionChunk(self._chunk.copy(), stats={"latency_s": 0.0})

    def describe(self) -> str:
        return (
            f"CONTROL {self.kind} (image-blind, chunk {self._chunk.shape}, "
            f"{self._speed} m/s)"
        )


# ---------------------------------------------------------------------------
# Model policies
# ---------------------------------------------------------------------------


class BridgedPolicy(Policy):
    """A checkpoint running in a second interpreter (see :mod:`.bridge`)."""

    name = "model"

    def __init__(self, bridge: Any):
        self.bridge = bridge
        self._last_stats: Dict[str, Any] = {}

    def reset(self, goal: Optional[str]) -> None:
        self.bridge.reset(goal)

    def act(self, obs: Observation) -> ActionChunk:
        reply = self.bridge.act(obs.rgb)
        self._last_stats = reply["stats"]
        return ActionChunk(reply["chunk"], stop=reply["stop"], stats=reply["stats"])

    def describe(self) -> str:
        return "model (bridged to a second interpreter)"

    def close(self) -> None:
        self.bridge.close()


class InProcessPolicy(Policy):
    """A checkpoint loaded in this interpreter, via a :mod:`.bridge` backend.

    Only usable when one environment has both ``habitat_sim`` and the model
    stack; the bridged path is the normal one here.
    """

    name = "model"

    def __init__(self, backend: Any):
        self.backend = backend

    def reset(self, goal: Optional[str]) -> None:
        self.backend.reset(goal)

    def act(self, obs: Observation) -> ActionChunk:
        chunk = self.backend.act(np.ascontiguousarray(obs.rgb, dtype=np.uint8))
        return ActionChunk(
            chunk, stop=self.backend.wants_stop(), stats=dict(self.backend.last_stats)
        )

    def describe(self) -> str:
        return f"model in-process: {self.backend.describe()}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_POLICIES: Dict[str, Callable[..., Policy]] = {
    "zero": lambda **kw: ConstantChunkPolicy("zero", **_control_kwargs(kw)),
    "forward": lambda **kw: ConstantChunkPolicy("forward", **_control_kwargs(kw)),
}


def _control_kwargs(kw: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("chunk_len", "dt", "speed")
    return {k: v for k, v in kw.items() if k in keep}


def register_policy(name: str, factory: Callable[..., Policy]) -> None:
    """Add a policy kind, e.g. a new scripted control or a new bridge flavour."""
    _POLICIES[name] = factory


def available_policies() -> list:
    return sorted(set(_POLICIES) | {"model"})


def build_policy(config: Any, server_script: str, log_path: Optional[str] = None) -> Policy:
    """Instantiate the policy named by ``config.policy``.

    ``model`` is special-cased because it is the only kind that needs the
    two-environment bridge; every other kind comes from the registry.
    """
    if config.policy != "model":
        if config.policy not in _POLICIES:
            raise KeyError(
                f"unknown policy '{config.policy}'; available: {available_policies()}"
            )
        policy = _POLICIES[config.policy](
            chunk_len=config.control_chunk_len, dt=config.dt, speed=config.control_speed
        )
        print(f"{policy.describe()}", flush=True)
        return policy

    if not config.ckpt:
        raise ValueError("--ckpt is required for --policy model")

    backend_args = {
        "ckpt": config.ckpt,
        "device": config.device,
        "backend": config.policy_backend,
        "max_context_tokens": config.max_context_tokens,
    }
    if config.placeholder:
        backend_args["placeholder"] = config.placeholder
    if config.policy_sys_path:
        backend_args["sys_path"] = config.policy_sys_path
    if config.policy_decode is not None:
        backend_args["decode"] = config.policy_decode
    if config.policy_seed is not None:
        backend_args["seed"] = config.policy_seed

    if config.policy_python:
        from objectnav_eval.bridge import PolicyBridge

        return BridgedPolicy(
            PolicyBridge(server_script, config.policy_python, backend_args, log_path)
        )

    from objectnav_eval.bridge import build_backend

    in_process_kwargs = dict(
        ckpt=config.ckpt,
        device=config.device,
        max_context_tokens=config.max_context_tokens,
        placeholder=config.placeholder,
    )
    if config.policy_decode is not None:
        in_process_kwargs["decode"] = config.policy_decode
    if config.policy_seed is not None:
        in_process_kwargs["seed"] = config.policy_seed
    backend = build_backend(config.policy_backend, **in_process_kwargs)
    return InProcessPolicy(backend)
