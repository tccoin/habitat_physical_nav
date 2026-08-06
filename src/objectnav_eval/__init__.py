"""ObjectNav policy evaluation system.

A composable evaluation harness for closed-loop ObjectNav policies driven
through :class:`continuous_demos.sim.HabitatRobotSim`. It layers on top of
:mod:`continuous_demos.objectnav` (the ObjectNav *task*: spawn + habitat-lab
success/SPL/soft-SPL/oracle measures) and adds everything an evaluation *run*
needs that a task does not:

    episodes    -- where episodes come from                (:mod:`.episodes`)
    screening   -- which of them this robot can even win   (:mod:`.screening`)
    policies    -- what drives the robot                   (:mod:`.policies`)
    executor    -- how an action chunk becomes motion      (:mod:`.executor`)
    probes      -- diagnostics riding along the rollout    (:mod:`.probes`)
    budget      -- what an episode is allowed to cost      (:mod:`.budget`)
    runner      -- one episode, start to metrics           (:mod:`.runner`)
    parallel    -- many episodes over many GPUs            (:mod:`.parallel`)
    reporting   -- JSONL records + the aggregate summary   (:mod:`.reporting`)

Each of those is a separate module with a small interface, so a new dataset, a
new policy head, a new metric or a new diagnostic is an addition rather than an
edit. See ``docs/OBJECTNAV_EVAL.md``.

Nothing here imports ``habitat_sim`` or ``habitat`` at module import time except
:mod:`.navmesh`, :mod:`.task` and :mod:`.runner`; the pure logic (sharding,
aggregation, tracking statistics, the policy bridge protocol) is importable in
any environment, which is what lets the policy run in a second interpreter and
lets the test suite run without a simulator.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
