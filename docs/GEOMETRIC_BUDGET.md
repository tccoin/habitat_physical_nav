# The geometric episode budget

**How much motion is one ObjectNav episode allowed, and in what currency?**

Classic ObjectNav caps an episode at **500 discrete steps**, each either a
`MOVE_FORWARD` of 0.25 m or a `TURN` of 30 degrees. A continuous agent has
neither, so every cap this harness has used measures the episode in a currency
the benchmark does not use. The geometric budget takes the discrete step's
*geometry* literally:

> **1 unit = 0.25 m translated OR pi/6 rad (30 degrees) rotated**, accumulated
> per control tick:  `budget += |d(x,y)| / 0.25 + |d(theta)| / (pi/6)`

---

## The one number people will quote

> **500 geometric units is about 95 s of human-demonstration motion and about
> 130-150 s of current-policy motion — against the 70 s cap that every result
> before 2026-08-06 was produced under.**

Adopting a 500-unit cap is therefore roughly a **1.4x** increase in effective
episode budget measured at demonstration speed, and **1.8-2.1x** measured at the
speed our policies actually move. Under the 70 s cap our policies spent a median
of **234 units — 47% of a 500-unit budget**. Every "the policy ran out of steps"
result so far was produced with roughly half the budget a classic ObjectNav
agent gets, and any comparison of old numbers to new ones has to carry that.

There is **no scalar "1 second = X units" conversion.** The ratio is a property
of the trajectory, not of the robot; the rest of this document is mostly about
why, and how much it varies.

---

## Contents

1. [Why not steps, and why not seconds](#1-why-not-steps-and-why-not-seconds)
2. [The kinematic ceilings, from this repo's own limits](#2-the-kinematic-ceilings)
3. [Forward: geometric units -> seconds (exact, with conditions)](#3-forward-geometric-units---seconds)
4. [Backward: seconds -> geometric units (not a single number)](#4-backward-seconds---geometric-units)
5. [The mixed-tick caveat](#5-the-mixed-tick-caveat)
6. [Geometric units vs discrete steps: the exactness result](#6-geometric-units-vs-discrete-steps)
7. [Worked examples](#7-worked-examples)
8. [Conversion table for a 500-unit cap](#8-conversion-table-for-a-500-unit-cap)
9. [What each currency is blind to](#9-what-each-currency-is-blind-to)
10. [Using it](#10-using-it)
11. [Provenance, and what was verified vs cited](#11-provenance)

---

## 1. Why not steps, and why not seconds

**Policy steps** (`--max-steps`) are `gap`-dependent. One policy step is `gap`
control ticks of `dt` seconds, so the same cap buys a 1 Hz policy five times the
motion it buys a 5 Hz one. Two checkpoints trained at different observation
rates cannot be compared under a step cap, and the harness has already shipped
runs at gap 5, 10 and 25.

**Seconds** are rate-independent but charge rotation about twice what a discrete
step budget does. At the limits the demonstrations were recorded under, one
second of driving buys 8.00 units and one second of turning 3.82 — a factor of
**N = 2.09**. A policy that turns more than the reference is therefore *charged
less* per unit of progress under a time cap than under the benchmark's own cap.

**Geometric units** are what the benchmark's cap actually buys, and they are
invariant to `gap`, to `dt`, and to how the controller splits its time.

## 2. The kinematic ceilings

Two limits set the extremes. Both come from this repo, not from a convention:

| where | limit | source |
| --- | --- | --- |
| data generation, driving | `v_max = 2.0` m/s | `HolonomicControllerConfig` in `src/continuous_demos/controllers.py` |
| data generation, turning | `w_max = 2.0` rad/s | `HeadingRotationConfig.w_max`, same file |
| inference (eval), driving | `v_max = 2.0` m/s | `PID_BASELINE` in `src/objectnav_eval/control.py` |
| inference (eval), turning | `w_max = pi` rad/s, `aw_max = 400 deg/s^2` | same |

Each recording also carries its own generation limits in
`recording.v_max_mps` / `recording.w_max_radps`, so a corpus states them rather
than inheriting them from the source tree.

In units per second:

```
driving at v_max :  2.0 / 0.25       = 8.000 units/s
turning at w_max :  2.0 / (pi/6)     = 3.820 units/s        (data generation)
turning at w_max :  pi  / (pi/6)     = 6.000 units/s        (inference PID)

asymmetry  N = 8.000 / 3.820 = 2.094  = (pi/6)/0.25 * (v_max/w_max)
```

**These ceilings are reached.** Measured over 600 demonstration episodes
(293,159 control ticks) the recorded motion is essentially bang-bang:

* linear speed sits at **exactly 2.000 m/s** from p90 upward (drive-classified
  ticks: p75 = 2.000); only 0.05% of ticks exceed 2.05 m/s.
* angular rate sits at **exactly 2.000 rad/s** from p90 upward (turn-classified
  ticks: p75 = 2.000); only 0.3% of ticks exceed 2.001 rad/s.

So for demonstration-quality motion the conversion factors **are** the kinematic
limits — that is a property of the recording controller, not of the data. It is
what makes the demo-side conversion tight, and it is exactly what our policies
do *not* do: their drive ticks reach 4.5-5.3 units/s on average against the
demos' 6.2, and their turn ticks 2.7-3.2 against 3.1 with a p50 at the cap.
The inference PID's higher angular ceiling (6.0 units/s) is never approached —
`aw_max = 6.98 rad/s^2` needs 0.45 s to reach pi rad/s from rest, and a chunk at
gap 5 / dt 0.04 lasts 0.2 s.

## 3. Forward: geometric units -> seconds

**Exact, when the motion decomposes into pure phases** (drive, then turn, each
saturating its limit):

```
seconds = translation_units / (v_max / 0.25)  +  rotation_units / (w_max / (pi/6))
        = translation_units / 8.000           +  rotation_units / 3.820
```

The two conditions are worth stating plainly, because both fail in practice:

* every tick saturates its limit — true for the demonstrations, false for our
  policies;
* no tick does both at once — false for both (see section 5), and when it is
  false this formula is an **upper bound** on the time.

## 4. Backward: seconds -> geometric units

There is no single conversion. The rate is a weighted average over the modes the
trajectory happens to use:

```
units_per_second = f_drive * r_drive + f_turn * r_turn + f_mixed * r_mixed
```

where `f_*` are *time* fractions and `r_*` the per-mode rates. Measured, pooling
control ticks (drive = translating above 0.20 m/s, turn = rotating above
0.40 rad/s, mixed = both, still = neither):

**Human demonstrations** (1200 episodes, 6 MP3D scenes, 567,682 ticks):

| mode | share of time | share of budget | units/s mean | p50 | p75 |
| --- | --- | --- | --- | --- | --- |
| drive | 30.5% | 35.9% | 6.20 | 7.27 | 8.10 |
| turn | 44.2% | 26.1% | 3.11 | **3.820** | **3.820** |
| mixed | 21.2% | 37.8% | 9.34 | 9.55 | 10.78 |
| still | 4.1% | 0.2% | 0.22 | 0.00 | 0.28 |
| **overall** | | | **5.26** | | |

**Policy rollouts** (16 finished runs on disk, 70 s cap, gap 5/10/25):

| | range over runs | median |
| --- | --- | --- |
| overall units/s | 1.80 - 4.05 | **3.35** |
| rotation share of budget | 0.37 - 0.54 | 0.50 |
| mixed share of time | 8.4% - 24.6% | ~19% |
| mixed share of budget | 18.5% - 43.7% | ~35% |

The healthy recent checkpoints (v8/v9/v10) sit at **3.3-4.1 units/s**; the low
end of the range is the deliberately-crippled pose ablations (`MIRROR90`), which
move slowly because they navigate badly.

**Policies convert time into budget ~30-35% less efficiently than demos** (3.35
vs 5.26 units/s). They neither drive at cruise nor hold the turn cap, and they
spend 10-20% of their time in `still` against the demos' 4%.

Consequently:

```
500 units at the demo rate   (5.26 u/s)  =  95 s
500 units at the policy rate (3.3-4.1)   =  123-152 s   (median run: 149 s)
70 s (the cap used to date)  at 3.35 u/s =  234 units   =  47% of 500
```

**In the reverse direction, which is what a run config actually needs** — one
policy step is `gap * dt` seconds:

| gap | dt | s/step | 500 units at 3.9 u/s | at 3.35 u/s |
| --- | --- | --- | --- | --- |
| 5 | 0.04 | 0.20 s | 128 s = 641 steps | 149 s = 746 steps |
| 10 | 0.04 | 0.40 s | 128 s = 320 steps | 149 s = 373 steps |
| 25 | 0.04 | 1.00 s | 128 s = 128 steps | 149 s = 149 steps |
| 5 | 0.05 | 0.25 s | 128 s = 513 steps | 149 s = 597 steps |

This table is exactly why the budget is not expressed in steps: the same amount
of motion is 128 or 641 policy steps depending only on the observation rate.

## 5. The mixed-tick caveat

Ticks that translate **and** rotate at the same time reach up to
`8.000 + 3.820 = 11.82` units/s. Measured on the demonstrations, mixed ticks
have p90 = 11.81 and p99 = 11.82 units/s — the sum of the two ceilings, hit
almost exactly — and only 0.43% exceed it.

They account for **21% of demonstration time but 38% of demonstration budget**,
and 8-25% of policy time but 19-44% of policy budget.

**A discrete agent cannot do this at all.** `MOVE_FORWARD` and `TURN` are
mutually exclusive actions; there is no discrete step that both translates and
rotates. So a continuous agent earns budget in a mode the discrete step count
has no analogue for, at up to 3.1x the rate of the discrete agent's best move.

Two consequences:

1. **Any single scalar seconds<->units conversion hides this.** The mixed
   fraction varies by a factor of three across our own runs, and it is the
   largest single term in the budget for several of them.
2. **The 0.83 units-per-source-action ratio must not be inverted naively.** It
   is a statement about matched trajectories (same episode, same route), not a
   general exchange rate. A continuous agent that drives arcs where the human
   drove straight-then-turned spends its budget in a different mode mix, and the
   ratio moves.

## 6. Geometric units vs discrete steps

Measured on the discrete `reference_replay` *states* — the ground truth the
demonstrations were recorded from:

| quantity | corpus-wide (cited) | re-verified here |
| --- | --- | --- |
| rotation_units / #TURN | **1.0000** | **1.0000000** (exact) |
| translation_units / #MOVE_FORWARD | 0.8473 | 0.8074 |
| units / #source actions | 0.8346 (per-episode median 0.856) | 0.8652 (median 0.889) |

**Rotation is exact by construction**: every `TURN` is exactly 30 degrees, so
its geometric cost is exactly 1 unit and the total rotation of a replay divided
by 30 degrees equals its `TURN` count to eight significant figures. This is the
part that does not depend on data.

**Translation is data-dependent**, and the shortfall is actions that achieved
(almost) nothing. On the re-verification scene, of 89,561 `MOVE_FORWARD`s:

* **16.4% displace the agent by nothing at all** — collision-blocked;
* the remainder average 0.2416 m rather than the nominal 0.25 m, because a
  partially-blocked forward slides along the obstacle.

Of the 0.193 shortfall, 0.165 (85%) is fully-blocked forwards and 0.028 (15%)
is partial sliding. So the geometric budget is **not approximating the discrete
step count — it is measuring the same quantity with the no-ops removed.** An
agent charged in geometric units is not charged for walking into a wall; an
agent charged in discrete steps is. (That is also why the budget needs a
separate backstop; see section 9.)

### A separate fact: the continuous replay does not preserve rotation exactly

Joining the same 997 episodes to their *continuous* recordings rather than their
discrete states:

| quantity | continuous replay | discrete states |
| --- | --- | --- |
| rotation_units / #TURN | 0.916 | 1.0000 |
| translation_units / #MOVE_FORWARD | 0.767 | 0.807 |
| units / #source actions | 0.802 | 0.865 |

The continuous replay realises about **92% of the source's rotation and 95% of
its translation**: the path controller cuts corners and settles within a heading
tolerance rather than landing on each 30-degree grid point. Neither is a bug,
but it means the "1.0000, exact" result is a statement about the **discrete
states**, and a continuous trajectory replaying the same episode spends slightly
less. Quote the right one.

## 7. Worked examples

**A 1 m-diameter circle.** Circumference pi m, one full revolution:

```
translation  pi / 0.25          = 12.566 units
rotation     2*pi / (pi/6)      = 12.000 units
total                           = 24.566 units
forward formula (pure phases)   = 12.566/8.000 + 12.000/3.820 = 1.571 + 3.142 = 4.712 s
```

But driving a circle *is* mixed motion: both limits can be saturated at once, so
a robot that actually drives it takes `max(1.571, 3.142) = 3.14 s`, not 4.71 s.
The forward formula is an upper bound, and this example is the mixed-tick
caveat in one line. (`tests/test_geometric_budget.py` pins the 24.566.)

**A 10 m straight corridor plus a 90-degree corner:**

```
10 / 0.25 = 40.0 units  +  (pi/2)/(pi/6) = 3.0 units  =  43.0 units
seconds (pure phases)   = 40/8.000 + 3/3.820 = 5.000 + 0.785 = 5.79 s
in discrete steps       = 40 MOVE_FORWARD + 3 TURN = 43 steps    (identical)
```

Straight-line navigation is where the two currencies agree exactly.

**A typical human demonstration episode:** median 98 units, p95 223.5 (corpus);
median 85, p95 213 on the 1200-episode re-verification sample. At the demo rate
that is ~18 s of motion, and it is 20% of a 500-unit budget.

## 8. Conversion table for a 500-unit cap

**Idealised** — every tick saturates a kinematic limit, pure phases only, no
mixed ticks. `f` is the fraction of *time* spent turning:

| f (turn time) | units/s | 500 units = | rotation's share of the budget |
| --- | --- | --- | --- |
| 0.00 | 8.000 | **62.5 s** | 0.000 |
| 0.25 | 6.955 | **71.9 s** | 0.137 |
| 0.36 | 6.495 | **77.0 s** | 0.212 |
| 0.50 | 5.910 | **84.6 s** | 0.323 |
| 0.75 | 4.865 | **102.8 s** | 0.589 |
| 1.00 | 3.820 | **130.9 s** | 1.000 |

> **Do not confuse the two rightmost columns.** "Humans spend 36% of their
> budget on rotation, our policies 50%" is the *last* column, not `f`. In the
> idealised model a 36% budget share corresponds to a turn *time* fraction of
> about 0.62, and a 50% budget share to about 0.68 — turning is cheap per
> second, so it takes much more than half the time to consume half the budget.

**Measured**, which is what to actually use:

| | units/s | 500 units = |
| --- | --- | --- |
| human demonstrations | 5.26 | **95 s** |
| policy rollouts (median of 16 runs) | 3.35 | **149 s** |
| policy rollouts (healthy checkpoints) | 3.3 - 4.1 | **123 - 152 s** |

The measured column is **worse than the idealised column at any comparable mix**
because not every tick saturates. Feeding each distribution's own mode mix
through the idealised rates gives the rate it *would* reach if every tick were
at a limit:

| | idealised at its own mix | measured | achieved |
| --- | --- | --- | --- |
| human demonstrations | 6.64 units/s | 5.26 | **79%** |
| policy rollouts (median run) | 5.50 units/s | 3.35 | **61%** |

*The last column is a readout of control quality*, not of navigation strategy:
it says the controller spends much of each chunk accelerating, decelerating and
correcting rather than at cruise. The demonstrations lose 21% to that; our
policies lose 39%.

## 9. What each currency is blind to

| currency | blind to |
| --- | --- |
| **policy steps** | `gap`. The same number buys 5x the motion at gap 25 that it does at gap 5. |
| **seconds** | *what the robot is doing*. Rotation costs 2.09x more per unit of budget than driving, so a spinning policy is undercharged. |
| **geometric units** | *standing still, and pushing into geometry*. A frozen agent accrues nothing; an agent driving into a wall achieves no displacement and so spends nothing. Measured `blocked_drive_frac` is 2.1-6.1% of ticks and the `still` share of policy time is 10-20%. |

The third is why **a geometric budget always carries a sim-time backstop**, and
why `EvalConfig` refuses to construct one without it (`BudgetRule.__post_init__`
raises). The default backstop of 300 s against a 500-unit budget implies a floor
of 1.67 units/s. Measured per-episode rates have p10 between 1.05 and 3.42 across
runs, and the slowest single episode observed was **0.83 units/s**.

The backstop is therefore not neutral, and this is worth stating: for a healthy
checkpoint it never binds (their p10 is 2.7-3.4 units/s, so even their slowest
episodes reach 500 units in well under 300 s), but for a badly-behaved one — the
`MIRROR90` ablations run at 1.8-2.2 units/s overall and have p10 near 1.1 — the
*clock*, not the budget, becomes the binding cap on the slowest tenth of
episodes, which then end having spent 315-410 units rather than 500. That is the
intended behaviour (a policy that cannot convert time into motion should not be
given unbounded time to do it), but it means the two rules must be read
together. Which one fired is recorded per episode as `terminated_by`
(`budget_exhausted` | `time_limit` | `max_steps` | `oracle_stop` |
`policy_stop` | `tipped_over`) and counted in the summary.

## 10. Using it

```bash
# the classic cap, in the currency the benchmark actually uses
python scripts/eval_objectnav_policy.py --ckpt ... --episodes ... \
    --episode-budget geometric --budget-units 500 --budget-time-limit 300

# keep the step cap as well; whichever fires first ends the episode
python scripts/eval_objectnav_policy.py ... --episode-budget both --max-steps 350

# what a 500-unit cap would have done to runs already on disk
python scripts/rescore_budget.py dump/eval_a dump/eval_b --budget-units 250,500,750
```

`--episode-budget steps` is the default and is byte-identical to the historical
behaviour. The budget spent is recorded in every episode record regardless
(`record["budget"]`, and `budget_units` / `budget_turn_frac` in the JSONL), the
same way `path_length_m` always is — it costs one subtraction per tick and it is
what lets a finished run be re-scored instead of re-run.

Termination is at the **tick** the budget runs out, not the next chunk boundary,
reusing the executor's existing `stop_when` seam (the same one the oracle stop
uses). Under `--episode-budget geometric` the step cap is deliberately not a
termination rule at all: the loop bound is derived from the backstop instead, so
a stale `--max-steps 200` cannot silently bind and leave the run reporting a
geometric cap it never applied.

**Re-scoring is honest about what survives truncation.** The budget, the
truncation tick, and the path and sim time at that tick are exact.
`oracle_success` survives, because it latches and never un-latches — with a
one-chunk ambiguity where the cut lands inside the arrival chunk, which is
reported rather than guessed. `success`, `spl`, `soft_spl` and `npwpl` do
**not** survive: they are read at the final pose, and a truncated episode has a
different one. Nothing estimates them.

### Recomputing the factors for a different robot

Everything in sections 2-4 is `v_max`, `w_max` and the two unit sizes. For a
different robot:

```
r_drive = v_max / 0.25              units/s
r_turn  = w_max / (pi/6)            units/s
N       = r_drive / r_turn = (pi/6)/0.25 * (v_max/w_max) = 2.0944 * v_max/w_max
```

The unit sizes are properties of the *benchmark* (0.25 m and 30 degrees are
ObjectNav's discrete action), not of the robot, and should not be changed to
match a different base — that would make the budget non-comparable with the
published 500-step cap, which is the entire point of it. `GeometricBudget`
accepts `unit_translation_m` / `unit_rotation_rad` anyway, for the case where a
different benchmark is being targeted.

## 11. Provenance

**Verified in this session, from data on disk:**

| result | source | n |
| --- | --- | --- |
| `rotation_units / #TURN = 1.0000000` on discrete states | `examples/17DRP5sb8fy.json.gz` | 1140 episodes |
| `translation_units / #MOVE_FORWARD = 0.8074`; 16.4% of forwards blocked | same | 89,561 forwards |
| `units / #source actions = 0.8652`, per-episode median 0.889 | same | 1140 episodes |
| continuous replay realises 0.916 of source rotation, 0.767 per forward | `output_25hz/metadata/17DRP5sb8fy.jsonl` joined to the above | 997 matched |
| demo phase table, 5.26 units/s overall, 37.3% rotation share | `output_25hz/metadata/*.jsonl`, first 6 scenes | 1200 episodes, 567,682 ticks |
| speed pins at 2.000 m/s and 2.000 rad/s from p90 | same, first 3 scenes | 600 episodes |
| mixed ticks p90 = 11.81, p99 = 11.82 units/s | same | 1200 episodes |
| policy rates 1.80-4.05 units/s, rotation share 0.37-0.54 | `/Projects/spatial_training/dump/eval_system/pose_v5/*/objectnav_eval_results.json` | 16 runs, 70-1370 episodes |
| the 70 s cap | the same runs' own `config` (`max_steps * gap * dt`) | 15 of 16 runs |
| kinematic limits | `controllers.py`, `objectnav_eval/control.py` | source |

**Cited from a prior measurement pass, not re-derived here** (the numbers in the
task that produced this work, measured over 39,756 recorded episodes joined to
their source action histograms):

* `translation_units / #MOVE_FORWARD = 0.8473`, 15.3% of forwards blocked;
* `units / #source actions = 0.8346`, per-episode median 0.856, p5 0.634,
  p95 0.961;
* human demos consume median 98 units, p95 223.5;
* 87.9% of uncensored human demos fit in 500 discrete steps, 90.4% in 500
  geometric units.

The re-verification above is a **single MP3D scene** (`17DRP5sb8fy`) and six
scenes for the trajectory statistics, so it is expected to differ from
corpus-wide figures by a few points — and it does, in both directions
(0.807 vs 0.847 for forwards, 0.865 vs 0.835 overall). The *exact* result agrees
to eight figures, as it must. Where the two disagree, prefer the corpus-wide
number and treat the difference as scene-to-scene variance, not as a
contradiction.

**Not verified, and worth knowing:**

* the 87.9% / 90.4% coverage figures need the uncensored full corpus; all 1140
  episodes of the re-verification scene fit in both caps, so it says nothing.
* `blocked_drive_frac` 2.1-6.1% is quoted from the collision probe's own
  summaries, not recomputed here.
* no evaluation was run to produce this document. The geometric rule's effect on
  `success`/`spl` at a 500-unit cap is therefore **unmeasured** — the re-scoring
  script deliberately refuses to estimate it.

## Scope

This is a conversion between *our* setup and the classic discrete ObjectNav
budget, using *our* `v_max` and `w_max` and *our* recorded trajectory
distributions. The unit definitions are the benchmark's and are fixed; every
rate in this document is not. A different robot, a different controller, or a
policy with a different turn/move mix gets different factors — section 10 says
how to recompute the kinematic ones, and `scripts/rescore_budget.py` measures
the empirical ones from any finished run.

Implementation: `src/objectnav_eval/budget.py`, wired in
`src/objectnav_eval/runner.py` and `config.py`, pinned by
`tests/test_geometric_budget.py`.
