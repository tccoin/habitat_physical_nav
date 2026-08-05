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
    # Distance at which --auto-stop actually ends the episode. None means "the
    # same as success_distance", i.e. terminate the moment the episode is won.
    # Setting it *tighter* than success_distance decouples the two: the oracle
    # measures still latch at success_distance (OracleSuccess on ever reaching
    # it, OracleSPL freezing path length there), so oracle_success and
    # oracle_spl are unchanged, while the robot keeps driving -- which is how
    # you get full-length footage of an episode that has already been won.
    # It does move `success` and `spl`, because path length keeps accruing and
    # the episode now ends wherever the robot got to.
    auto_stop_distance: Optional[float] = None
    # Policy steps to keep running AFTER --auto-stop's condition first holds.
    # 0 = terminate immediately, the historical behaviour. A positive value opens
    # a window in which a gated stop classifier (see `stop_requires_oracle`) can
    # declare for itself, while guaranteeing the episode ends soon after arrival
    # rather than running to the step cap. Because the window opens only once the
    # agent has already arrived, `oracle_success`/`oracle_spl` are untouched by
    # it -- they latched at the moment the window opened.
    auto_stop_delay: int = 0
    # Which navmesh every distance is measured on: 'robot' (recomputed for
    # r=0.28 h=1.0) or 'dataset' (the point-agent mesh shipped with the
    # scene, what published ObjectNav numbers use). Changes what the metrics
    # mean, not how the robot moves. See navmesh.apply_navmesh_choice.
    navmesh: str = "robot"
    snap_start: bool = False
    screen_trivial: bool = True
    live_height: bool = True
    # Rebuild the simulator before every episode. HabitatRobotSim.reset() does
    # not fully restore the physics state, so an episode's trajectory depends on
    # what ran before it in the same process (~1.4% of path length, measured
    # with a stateless policy). Costs a scene load per episode.
    fresh_sim_per_episode: bool = False
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
    # Distributional-head knobs (e.g. --policy-backend bin_rollout). None means "don't
    # forward this flag" so backends that don't accept it (vector_rollout) are unaffected.
    policy_decode: Optional[str] = None
    policy_seed: Optional[int] = None
    # Memory-length ablation. Expressed in SECONDS because that is the quantity
    # that stays meaningful: one policy step is gap*dt seconds, so a step count
    # silently changes duration the moment --dt or --gap moves -- the same trap
    # the corpus dt mismatch set. None = unlimited memory (production).
    policy_memory_seconds: Optional[float] = None
    # -- the checkpoint's own stop classifier -------------------------------
    # OFF by default, and that is a semantic choice rather than caution: with
    # --auto-stop the oracle ends the episode on arrival, so a stop head firing
    # underneath it would share credit for the same episode and neither could be
    # read separately. When it *is* on, run without --auto-stop -- the oracle
    # measures latch independently (OracleSuccess on ever reaching
    # success_distance, OracleSPL freezing path length there), so oracle_success
    # and oracle_spl stay comparable to every auto-stop run, while `success` and
    # `spl` finally measure the policy's own termination instead of being pinned
    # equal to their oracle counterparts.
    #
    # The stop *probability* is recorded every step regardless (see
    # `stop_prob` in the runner's record), because temperature and threshold are
    # post-hoc scalars fitted on saved scores -- a miscalibrated head is a
    # calibration problem, and re-running the rollout to try another threshold
    # would be paying for the same footage twice.
    stop_head: bool = False
    # Demote the classifier: it may end an episode ONLY at a moment the oracle
    # agrees the agent is within --success-distance. Its false positives then
    # cost nothing -- they are counted and discarded -- and what remains being
    # measured is the question worth asking of a head this miscalibrated: *given
    # that you are at the goal, do you say so?* Without the gate a head with a
    # bad tail destroys the navigation metrics before they can be read, which is
    # what happened at threshold 0.95 on run_v5_ar_pose/checkpoint-4400.
    stop_requires_oracle: bool = False
    # POSE-FRAME ABLATION. Mirror and/or rotate the frame the injected pose is
    # expressed in, keeping it perfectly self-consistent but pointing the wrong
    # way. The test of whether a pose-injecting policy actually *uses* the metric
    # frame: if it navigates by noiseless GPS+compass this should wreck it, and
    # if it ignores the injection the numbers should not move at all. Diagnostic
    # only -- never a configuration to report a headline number under.
    pose_mirror: bool = False
    pose_rotate_deg: float = 0.0
    # POSE-INFORMATION ABLATION. Inject another episode's pose sequence, drawn
    # from a training-corpus dataset. Complements the mirror/rotate corruption,
    # which is an isometry and therefore leaves every intrinsic quantity intact
    # (distance travelled, revisit structure, coverage) -- a policy using pose
    # only for "have I been here" survives that untouched. Foreign poses stay on
    # the encoder's training distribution, unlike zeros or noise, while carrying
    # no information about where the robot actually is.
    pose_foreign: Optional[str] = None
    pose_foreign_split: str = "train"
    stop_temperature: Optional[float] = None
    stop_threshold: Optional[float] = None
    stop_inference: Optional[str] = None
    # Pose injection has no flag on purpose: whether the policy is shown its
    # pose is a property of the checkpoint (a `<pose>` modality spec with the
    # `pose_relative_first` transform), and a flag would let the command line
    # disagree with the weights. See bridge._PoseFeed.

    # -- diagnostics --------------------------------------------------------
    keep_tracking_series: bool = False
    # The motion-taxonomy and coherence probes are unconditional: they are a few
    # floating-point operations on ticks the executor already emits. The
    # collision probe is the one that costs *simulator* time -- it samples the
    # physics contact manifold once per control tick -- so it gets a switch.
    collision_probe: bool = True

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

    def __post_init__(self) -> None:
        # Both terminations at once would end each episode at whichever fired
        # first, and the run could not say which -- the oracle's arrival and the
        # policy's decision would share one `success` column. Rejected rather
        # than resolved by precedence, because a precedence rule is exactly the
        # kind of thing a later reader assumes went the other way.
        if self.stop_head and self.auto_stop and not self.stop_requires_oracle:
            raise ValueError(
                "--stop-head and --auto-stop both terminate the episode; pick one, or "
                "add --stop-requires-oracle. Alone, --stop-head measures the policy's "
                "own stopping (oracle_success/oracle_spl still latch on ever reaching "
                "--success-distance). Gated, the classifier can only fire where the "
                "oracle already agrees, so the two cannot claim the same episode and "
                "running them together is well defined."
            )
        if self.stop_requires_oracle and not self.stop_head:
            raise ValueError(
                "--stop-requires-oracle gates the classifier, but --stop-head is off so "
                "the classifier cannot end an episode anyway"
            )
        if self.auto_stop_delay and not self.auto_stop:
            raise ValueError(
                "--auto-stop-delay without --auto-stop: nothing would open the window "
                "it delays"
            )
        if self.auto_stop_delay < 0:
            raise ValueError(f"--auto-stop-delay must be >= 0, got {self.auto_stop_delay}")

    @property
    def stop_distance(self) -> float:
        """Where ``--auto-stop`` terminates. Explicitly not ``or``-defaulted: a
        legitimate 0.0 must not silently fall back to ``success_distance``."""
        return (
            self.success_distance
            if self.auto_stop_distance is None
            else self.auto_stop_distance
        )

    @property
    def context_reset_every(self) -> Optional[int]:
        """``policy_memory_seconds`` in policy steps, or None for unlimited.

        One policy step advances ``gap`` control ticks of ``dt`` seconds each.
        Rounded to at least 1: a request below one step's duration cannot mean
        less than one step, and 0 would silently mean 'unlimited'.
        """
        if self.policy_memory_seconds is None:
            return None
        return max(1, round(self.policy_memory_seconds / (self.gap * self.dt)))

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
                     help="oracle termination: end the episode on the control tick the "
                          "agent first comes within --success-distance, for policies with "
                          "no termination head of their own. Makes success == "
                          "oracle_success at --evaluate-every 1")
    run.add_argument("--auto-stop-distance", type=float, default=None,
                     help="terminate --auto-stop at this distance instead of "
                          "--success-distance. Tighter values keep the robot driving "
                          "after the episode is already won, for footage: oracle_success "
                          "and oracle_spl are unaffected (they latch at "
                          "--success-distance), but success and spl are, since path "
                          "length keeps accruing")
    run.add_argument("--auto-stop-delay", type=int, default=0,
                     help="policy steps to keep running after --auto-stop's condition "
                          "first holds, instead of terminating on the arriving tick. "
                          "Opens a fixed window in which a gated stop classifier (see "
                          "--stop-requires-oracle) can declare for itself. The window "
                          "opens only once the agent has arrived, so oracle_success and "
                          "oracle_spl already latched and are unaffected")
    run.add_argument("--navmesh", default="robot", choices=["robot", "dataset"],
                     help="WHICH NAVMESH THE METRICS ARE MEASURED ON. 'robot' "
                          "(default) recomputes it for this robot's footprint "
                          "(r=0.28 m, h=1.0 m) -- honest about where the robot fits, "
                          "but fragmented: it screens out 28%% of HM3D val as "
                          "goal-unreachable, and where it routes around something the "
                          "physics body can drive through it overstates the distance "
                          "and can pin SPL at 1.000. 'dataset' uses the point-agent "
                          "mesh shipped with the scene -- connected, comparable with "
                          "published ObjectNav numbers, no absurd jumps, but it "
                          "understates difficulty by assuming a robot that fits "
                          "through gaps this one does not. Does NOT change how the "
                          "robot moves; it is a physics body either way")
    run.add_argument("--snap-start", action="store_true",
                     help="snap the spawn onto the robot navmesh instead of using the "
                          "episode's start position verbatim")
    run.add_argument("--keep-trivial", dest="screen_trivial", action="store_false",
                     help="score episodes that start inside the success radius instead of "
                          "screening them out (they are free successes)")
    run.add_argument("--fresh-sim-per-episode", action="store_true",
                     help="rebuild the simulator before every episode. HabitatRobotSim's "
                          "reset() leaves physics state behind, so without this an "
                          "episode's trajectory depends on what ran before it in the same "
                          "worker. Costs one scene load per episode; aggregate metrics are "
                          "stable without it, per-episode ones are not")
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
    mdl.add_argument("--policy-decode", default=None,
                     choices=["argmax", "sample", "mean", "logits"],
                     help="decode rule for a distributional policy head (bin_rollout, "
                          "ar_rollout); ignored by backends that don't accept it. Unset "
                          "means the backend's own default, which differs between them: "
                          "argmax for bin_rollout, sample for ar_rollout")
    mdl.add_argument("--policy-memory-seconds", type=float, default=None,
                     help="MEMORY-LENGTH ABLATION. Drop rollout history (KV cache and "
                          "visual dedup DB) every this many seconds of simulated time, "
                          "keeping the system prompt. Unset = unlimited memory, which is "
                          "the production default and is byte-identical to not passing "
                          "this. Converted to policy steps with gap*dt, so it means the "
                          "same duration at any control rate. NB the memory is a SAWTOOTH "
                          "not a window: within a block, history runs from 0 up to this "
                          "value, averaging about half it")
    mdl.add_argument("--policy-seed", type=int, default=None,
                     help="base seed for a backend's per-episode seeded sampling "
                          "(bin_rollout / ar_rollout with --policy-decode sample); "
                          "combined with an episode counter so each episode gets its own "
                          "fixed seed")
    mdl.add_argument("--stop-head", action="store_true",
                     help="let the checkpoint's own stop classifier end the episode. "
                          "Requires a checkpoint that has one. Run this INSTEAD OF "
                          "--auto-stop, not alongside it: with both, an episode ends at "
                          "whichever fires first and neither can be read on its own. "
                          "oracle_success/oracle_spl stay comparable to auto-stop runs "
                          "either way (they latch on ever reaching --success-distance); "
                          "success/spl become the policy's own termination. The stop "
                          "probability is recorded per step regardless of this flag")
    mdl.add_argument("--pose-mirror", action="store_true",
                     help="POSE-FRAME ABLATION: mirror the injected pose about the "
                          "start heading, flipping the strafe axis (+y left becomes "
                          "-y) and negating the heading. Applied to the RELATIVE pose, "
                          "so it is exactly the frame the model reads")
    mdl.add_argument("--pose-rotate-deg", type=float, default=0.0,
                     help="POSE-FRAME ABLATION: rotate the injected pose frame by this "
                          "many degrees. Applied AFTER --pose-mirror (they do not "
                          "commute). Note this must act on the relative pose: rotating "
                          "the raw scene pose is algebraically cancelled by "
                          "relative_se2's re-anchoring and would be a silent no-op")
    mdl.add_argument("--pose-foreign", default=None,
                     help="POSE-INFORMATION ABLATION: inject another episode's pose "
                          "sequence instead of the robot's, sampled per episode from "
                          "this load_from_disk() dataset (needs an `obs_poses` column, "
                          "e.g. data/v2_25hz/formatted_pose). Stays on the encoder's "
                          "training distribution -- unlike zeros or noise, which "
                          "conflate 'no information' with 'input never seen' -- while "
                          "carrying nothing about where the robot is. Trajectories are "
                          "chained with an SE(2) rebase when one runs short of "
                          "--max-steps, so the walk stays continuous")
    mdl.add_argument("--pose-foreign-split", default="train",
                     help="split of --pose-foreign to draw from")
    mdl.add_argument("--stop-requires-oracle", action="store_true",
                     help="the classifier may only end an episode at a step where the "
                          "agent is ALREADY within --success-distance. Its false "
                          "positives are then counted and discarded instead of ending "
                          "the run, so what gets measured is 'given that you are at the "
                          "goal, do you say so'. Makes --stop-head and --auto-stop "
                          "compatible: gated, they cannot claim the same episode")
    mdl.add_argument("--stop-inference", default=None,
                     choices=["sample", "threshold", "argmax"],
                     help="how the stop logit becomes a decision. Unset = the "
                          "checkpoint's own setting (the training default is 'sample'). "
                          "'threshold' uses --stop-threshold; 'argmax' is the same rule "
                          "pinned at 0.5")
    mdl.add_argument("--stop-threshold", type=float, default=None,
                     help="stop probability at/above which --stop-inference threshold "
                          "terminates. Unset = the checkpoint's own value")
    mdl.add_argument("--stop-temperature", type=float, default=None,
                     help="divides the stop logit before the sigmoid. A post-hoc "
                          "calibration knob -- it is not trained and does not change the "
                          "head's AP or AUC, only where its operating point sits")

    diag = parser.add_argument_group("diagnostics")
    diag.add_argument("--keep-tracking-series", action="store_true",
                      help="store the per-tick commanded/achieved/lag series in each "
                           "episode record (large; the summary statistics usually suffice)")
    diag.add_argument("--no-collision-probe", dest="collision_probe",
                      action="store_false",
                      help="skip the per-tick physics contact sample. The collision "
                           "diagnostic ('contact' in each record) then reports nothing; "
                           "the motion and coherence diagnostics are unaffected")

    par = parser.add_argument_group("parallelism")
    par.add_argument("--gpus", default=None,
                     help="comma-separated GPU ids to shard over, e.g. '4,5,6,7'. "
                          "Default: one worker on the current CUDA_VISIBLE_DEVICES")
    par.add_argument("--workers-per-gpu", type=int, default=1)
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    known = {f.name for f in fields(EvalConfig)}
    return EvalConfig(**{k: v for k, v in vars(args).items() if k in known})
