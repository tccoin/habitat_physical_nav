"""One configuration object for a whole evaluation run.

Every knob lives here, in one dataclass, for a specific reason: a parallel run
hands its configuration to worker processes, and rebuilding a worker's command
line flag by flag is how a flag silently stops being forwarded. The config is
serialised into the shard file as JSON and the worker reconstructs it, so a new
option is forwarded correctly by construction.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

# Execution cadence defaults, from the chunk config the current checkpoints were
# trained on: 20 Hz control, an observation every 5 ticks (so 4 Hz observations).
DEFAULT_DT = 0.05
DEFAULT_GAP = 5
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


@dataclass
class EvalConfig:
    # -- data ---------------------------------------------------------------
    episodes: str = ""
    episode_source: str = "objectnav"
    scene_root: str = "data/scene_datasets"
    output_dir: str = "dump/objectnav_eval"

    # -- episode selection --------------------------------------------------
    num_episodes: int = 0
    scenes: Optional[str] = None
    categories: Optional[str] = None
    episode_ids: Optional[str] = None
    max_scenes: int = 0
    sample_per_scene: int = 0
    seed: int = 0

    # -- rollout ------------------------------------------------------------
    max_steps: int = 200
    gap: int = DEFAULT_GAP
    dt: float = DEFAULT_DT
    success_distance: float = 1.0
    distance_to: str = "VIEW_POINTS"
    evaluate_every: int = 1
    auto_stop: bool = False
    snap_start: bool = False
    screen_trivial: bool = True
    live_height: bool = True
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    sensor_uuid: str = "color_sensor"

    # -- video --------------------------------------------------------------
    record_video: bool = False
    video_scale: float = 1.0

    # -- policy -------------------------------------------------------------
    policy: str = "model"
    policy_backend: str = "vector_rollout"
    ckpt: Optional[str] = None
    device: str = "cuda"
    max_context_tokens: int = 110000
    placeholder: Optional[str] = None
    policy_python: Optional[str] = None
    # Colon-separated paths prepended to sys.path *in the model interpreter*.
    # The checkpoint's own code usually lives in a source checkout that the
    # model env does not have installed -- and may have a same-named namespace
    # package from a different checkout ahead of it, which fails as a missing
    # submodule rather than a missing package.
    policy_sys_path: Optional[str] = None
    control_chunk_len: int = 10
    control_speed: float = 1.0

    # -- diagnostics --------------------------------------------------------
    keep_tracking_series: bool = False

    # -- parallelism --------------------------------------------------------
    gpus: Optional[str] = None
    workers_per_gpu: int = 1

    # -- worker plumbing (not user-facing) ----------------------------------
    shard_index: int = 0

    # ------------------------------------------------------------------
    @property
    def tag(self) -> str:
        """Filename suffix separating control runs from model runs."""
        return "" if self.policy == "model" else f"_{self.policy}"

    @property
    def fps(self) -> int:
        return max(1, round(1.0 / self.dt))

    def selection_lists(self) -> Dict[str, Optional[List[str]]]:
        return {
            "scenes": _split(self.scenes),
            "categories": _split(self.categories),
            "episode_ids": _split(self.episode_ids),
        }

    def gpu_slots(self) -> List[Optional[str]]:
        gpus = [g.strip() for g in self.gpus.split(",") if g.strip()] if self.gpus else [None]
        return [g for g in gpus for _ in range(max(1, self.workers_per_gpu))]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            # A worker reading a shard written by a different version of the
            # code should say so rather than run with silently missing options.
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


def _split(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Not `required=True`: a worker is launched with only --worker-shard and
    # reads its whole configuration out of the shard file. Requiring it here
    # would make every worker die at argument parsing, which the launcher
    # reports as a failed shard rather than as the CLI bug it is. main()
    # enforces it on the paths that actually need it.
    parser.add_argument("--episodes", default="",
                        help="ObjectNav split dir (with content/*.json.gz) or a .json.gz")
    parser.add_argument("--episode-source", default="objectnav",
                        help="episode source kind (see objectnav_eval.episodes)")
    parser.add_argument("--scene-root", default="data/scene_datasets")
    parser.add_argument("--output-dir", default="dump/objectnav_eval")

    sel = parser.add_argument_group("episode selection")
    sel.add_argument("--num-episodes", type=int, default=0, help="0 = all")
    sel.add_argument("--scenes", default=None, help="comma-separated scene names")
    sel.add_argument("--categories", default=None, help="comma-separated goal categories")
    sel.add_argument("--episode-ids", default=None,
                     help="comma-separated episode ids, or scene:id uids")
    sel.add_argument("--max-scenes", type=int, default=0,
                     help="keep at most this many scenes, chosen at random under --seed. "
                          "Scene load is the fixed cost per worker, so this is the knob "
                          "that actually shrinks a run")
    sel.add_argument("--sample-per-scene", type=int, default=0,
                     help="sample at most this many episodes from each kept scene; with "
                          "--max-scenes this gives a stratified subset of a big split")
    sel.add_argument("--seed", type=int, default=0,
                     help="seed for --max-scenes/--sample-per-scene, so a subset is "
                          "reproducible from the flags alone")

    run = parser.add_argument_group("rollout")
    run.add_argument("--max-steps", type=int, default=200, help="policy steps per episode")
    run.add_argument("--gap", type=int, default=DEFAULT_GAP,
                     help="control ticks executed per chunk, = the training observation "
                          "stride (5 ticks at 20 Hz = 4 Hz observations)")
    run.add_argument("--dt", type=float, default=DEFAULT_DT, help="control tick, seconds")
    run.add_argument("--success-distance", type=float, default=1.0,
                     help="geodesic distance to a goal view point that counts as reaching it")
    run.add_argument("--distance-to", default="VIEW_POINTS", choices=["VIEW_POINTS", "POINT"])
    run.add_argument("--evaluate-every", type=int, default=1,
                     help="update the task measures every N control ticks. 1 matches the "
                          "reference harness; higher is cheaper but makes the oracle "
                          "metrics lower bounds")
    run.add_argument("--auto-stop", action="store_true",
                     help="end the episode on arrival; makes success == oracle_success")
    run.add_argument("--snap-start", action="store_true",
                     help="snap the spawn onto the robot navmesh instead of using the "
                          "episode's start position verbatim")
    run.add_argument("--keep-trivial", dest="screen_trivial", action="store_false",
                     help="score episodes that start inside the success radius instead of "
                          "screening them out (they are free successes)")
    run.add_argument("--frozen-height", dest="live_height", action="store_false",
                     help="query goal distances at the spawn height for the whole episode "
                          "(the task layer's own behaviour) instead of tracking joint_z")
    run.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    run.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    run.add_argument("--sensor-uuid", default="color_sensor")

    vid = parser.add_argument_group("video")
    vid.add_argument("--record-video", action="store_true",
                     help="one MP4 per episode, one frame per control tick, so playback at "
                          "1/--dt fps is real time")
    vid.add_argument("--video-scale", type=float, default=1.0)

    mdl = parser.add_argument_group("policy")
    mdl.add_argument("--policy", default="model",
                     help="'model', or an image-blind control ('forward' drives straight "
                          "ahead, 'zero' stands still). Controls verify the harness and "
                          "bound a policy that learned nothing; they are not baselines")
    mdl.add_argument("--policy-backend", default="vector_rollout",
                     help="policy head implementation loaded in the model environment")
    mdl.add_argument("--ckpt", default=None, help="trained checkpoint directory")
    mdl.add_argument("--device", default="cuda")
    mdl.add_argument("--max-context-tokens", type=int, default=110000)
    mdl.add_argument("--placeholder", default=None)
    mdl.add_argument("--policy-python", default=os.environ.get("VECTOR_POLICY_PYTHON"),
                     help="interpreter that can load the checkpoint, when this env cannot. "
                          "Defaults to $VECTOR_POLICY_PYTHON")
    mdl.add_argument("--policy-sys-path", default=os.environ.get("VECTOR_POLICY_SYS_PATH"),
                     help="colon-separated paths prepended to sys.path in the model "
                          "interpreter, e.g. the src/ of the checkout the checkpoint's "
                          "code lives in. Defaults to $VECTOR_POLICY_SYS_PATH")
    mdl.add_argument("--control-chunk-len", type=int, default=10)
    mdl.add_argument("--control-speed", type=float, default=1.0,
                     help="m/s for the 'forward' control")

    diag = parser.add_argument_group("diagnostics")
    diag.add_argument("--keep-tracking-series", action="store_true",
                      help="store the per-tick commanded/achieved/lag series in each "
                           "episode record (large; the summary statistics usually suffice)")

    par = parser.add_argument_group("parallelism")
    par.add_argument("--gpus", default=None,
                     help="comma-separated GPU ids to shard over, e.g. '4,5,6,7'. "
                          "Default: one worker on the current CUDA_VISIBLE_DEVICES")
    par.add_argument("--workers-per-gpu", type=int, default=1)
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    known = {f.name for f in fields(EvalConfig)}
    return EvalConfig(**{k: v for k, v in vars(args).items() if k in known})
