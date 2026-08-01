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
from typing import Any, Callable, Dict, Optional

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

    def act(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Return an ``(T, 3)`` chunk of ``[dx, dy, dtheta]`` offsets."""
        raise NotImplementedError

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
    ):
        from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy

        config = RolloutConfig(
            device=device, merge_lora=merge_lora, max_context_tokens=max_context_tokens
        )
        if placeholder:
            config.placeholder = placeholder
        self.policy = VectorRolloutPolicy.from_checkpoint(ckpt, config)

    def reset(self, goal_text: Optional[str]) -> None:
        self.policy.reset(goal_text=goal_text)

    def act(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        return np.asarray(self.policy.step(Image.fromarray(rgb)).numpy(), dtype=np.float32)

    @property
    def last_stats(self) -> Dict[str, Any]:
        return dict(self.policy.last_stats)

    def describe(self) -> str:
        return self.policy.describe()


register_backend("vector_rollout", VectorRolloutBackend)


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
                chunk = np.asarray(backend.act(rgb), dtype=np.float32)
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

    def act(self, rgb: np.ndarray) -> Dict[str, Any]:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        reply = self._rpc(
            {"cmd": CMD_ACT, "shape": rgb.shape, "dtype": "uint8", "data": rgb.tobytes()}
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
