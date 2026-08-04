"""Sweep the stop classifier's operating point over an already-finished run.

A stop head that ranks well but is scaled badly is a **calibration** problem:
temperature and threshold are one scalar each, and both are runtime knobs the
weights never saw (`stop_head.py` reports AP and AUC precisely because they are
invariant to any monotone rescaling). So the fix is a sweep over saved scores,
not another pass over the split.

That is possible because every episode record carries two per-step traces:

    record["stop"]["trace"]          [(prob, logit, path_m), ...]  one per policy step
    record["goal_distance"]["trace"] [distance_to_goal_m, ...]     the same steps

so "would a threshold of 0.02 have stopped this episode within
--success-distance" is a lookup, not a rollout.

**The sweep is exact, not an estimate.** The policy never sees the stop
decision, so stopping at step k does not change what the trajectory was up to
step k. Success and SPL at any threshold are therefore recoverable from a run
that did not stop there -- which is the whole reason to run the split once with
no termination and calibrate afterwards.

WHAT THIS CAN AND CANNOT ANSWER
-------------------------------
* **Downward is exact.** A *lower* threshold fires no later than the recorded
  run did, so every step it could fire on is in the trace.
* **Upward is censored.** A *higher* threshold fires later or never. Where the
  recorded episode ended because the head fired, there is no data past that
  step, and those episodes are reported separately as `censored` rather than
  folded into the rate. An episode that ran to the step cap is not censored --
  its trace covers everything that happened.
* **SPL needs the third trace column.** Runs recorded before `path_m` was added
  to the trace report success only; the SPL column reads `-` for them rather
  than being computed from the final path length, which would be wrong in the
  flattering direction.

    python scripts/sweep_stop_threshold.py <results.json> [--success-distance 1.0]
    python scripts/sweep_stop_threshold.py <results.json> --temperature 2.0
"""

import argparse
import json
import math
from typing import List, Optional, Tuple


def load(path: str):
    with open(path) as fh:
        payload = json.load(fh)
    records = [r for r in payload.get("results", []) if r.get("status") == "evaluated"]
    config = payload.get("config", {})
    with_traces = [r for r in records if (r.get("stop") or {}).get("trace")]
    return records, with_traces, config


def rescale(logit: Optional[float], temperature: float) -> Optional[float]:
    """Probability under a *different* temperature than the run used."""
    if logit is None or not math.isfinite(logit):
        return None
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def first_crossing(probs: List[Optional[float]], threshold: float) -> Optional[int]:
    for i, p in enumerate(probs):
        if p is not None and p >= threshold:
            return i
    return None


def spl_at(record, step: int, success: bool) -> Optional[float]:
    """habitat's SPL, using the path length *at the moment of stopping*.

    `shortest / max(shortest, taken)`, zero when the episode failed. Returns None
    when the trace predates the `path_m` column, so a missing value is visible
    rather than silently substituted.
    """
    trace = record["stop"]["trace"]
    if step >= len(trace) or len(trace[step]) < 3:
        return None
    taken = trace[step][2]
    shortest = record.get("geodesic_start_m")
    if taken is None or shortest is None or not math.isfinite(float(taken)):
        return None
    if not success:
        return 0.0
    shortest = float(shortest)
    return shortest / max(shortest, float(taken)) if max(shortest, float(taken)) > 0 else 0.0


def evaluate(record, threshold: float, temperature: Optional[float],
             success_distance: float):
    """-> (outcome, step, distance, spl). Outcome: success|fail|never|censored."""
    trace = record["stop"]["trace"]
    probs = [
        row[0] if temperature is None else rescale(row[1] if len(row) > 1 else None, temperature)
        for row in trace
    ]
    distances = (record.get("goal_distance") or {}).get("trace") or []
    step = first_crossing(probs, threshold)
    if step is None:
        # Nothing fired inside what we can see. If the recorded episode was cut
        # short *by the head*, a higher threshold's outcome is unknown -- the
        # trajectory past that step never happened.
        if record.get("terminated") == "policy called stop":
            return "censored", None, None, None
        return "never", None, None, None
    if step >= len(distances) or distances[step] is None:
        return "censored", step, None, None
    distance = float(distances[step])
    success = distance < success_distance
    return ("success" if success else "fail"), step, distance, spl_at(record, step, success)


def ranking_report(traced, success_distance: float,
                   temperature: Optional[float]) -> None:
    """AP and AUC of the stop score against "the agent is actually at the goal".

    The one number that decides whether calibration can rescue this head at all.
    A threshold and a temperature are monotone transforms, so they cannot change
    an ordering -- if the head scores *arrived* steps above *not arrived* steps,
    some operating point works and finding it is arithmetic. If it does not, no
    threshold exists and the head needs retraining, not tuning.

    Labelled per policy step, not per episode, because that is the decision the
    head actually makes. The base rate is low, which is why AP is reported next
    to AUC: `stop_head.py` makes the same argument about accuracy.
    """
    from longnav.utils.stop_head import average_precision, roc_auc

    scores, labels = [], []
    for record in traced:
        distances = (record.get("goal_distance") or {}).get("trace") or []
        for i, row in enumerate(record["stop"]["trace"]):
            if i >= len(distances) or distances[i] is None:
                continue
            score = row[0] if temperature is None else rescale(
                row[1] if len(row) > 1 else None, temperature)
            if score is None:
                continue
            scores.append(score)
            labels.append(1.0 if float(distances[i]) < success_distance else 0.0)
    if not scores:
        return
    positives = int(sum(labels))
    print(f"\nranking vs 'within {success_distance} m of the goal at that step', "
          f"over {len(scores)} step(s), {positives} positive:")
    if positives == 0:
        print("  no positive steps in this run -- the agent never reached the goal, "
              "so ranking is undefined here rather than bad")
        return
    print(f"  average precision {average_precision(scores, labels):.4f}  "
          f"(chance = base rate {positives / len(scores):.4f})")
    print(f"  ROC AUC           {roc_auc(scores, labels):.4f}  (chance = 0.5)")
    print("  AP and AUC are invariant to threshold and temperature: if these are at "
          "chance,\n  no operating point exists and the head needs retraining, not tuning.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="objectnav_eval_results.json from a run")
    p.add_argument("--success-distance", type=float, default=None,
                   help="default: the value the run used")
    p.add_argument("--temperature", type=float, default=None,
                   help="recompute probabilities from the saved logits under this "
                        "temperature instead of the one the run used")
    p.add_argument("--thresholds", default=None,
                   help="comma-separated; default is a decade sweep plus 0.5")
    args = p.parse_args()

    records, traced, config = load(args.results)
    if not traced:
        raise SystemExit(
            f"no stop traces in {args.results} -- this run's checkpoint has no stop head, "
            "or it predates the trace being recorded"
        )
    success_distance = (
        args.success_distance if args.success_distance is not None
        else float(config.get("success_distance", 1.0))
    )
    ran_with = "--stop-head" if config.get("stop_head") else (
        "--auto-stop" if config.get("auto_stop") else "no termination")
    print(f"{len(traced)}/{len(records)} evaluated episode(s) carry a stop trace")
    print(f"run terminated with: {ran_with}   success_distance {success_distance} m")
    if args.temperature is not None:
        print(f"probabilities recomputed at temperature {args.temperature} "
              f"(run used {config.get('stop_temperature') or 'the checkpoint value'})")

    every = [p for r in traced for p in [row[0] for row in r["stop"]["trace"]]]
    every.sort()
    def pct(q):
        return every[min(len(every) - 1, int(q * len(every)))]
    print(f"\nstop probability over all {len(every)} steps: "
          f"min {every[0]:.4f}  p50 {pct(0.5):.4f}  p90 {pct(0.9):.4f}  "
          f"p99 {pct(0.99):.4f}  max {every[-1]:.4f}")

    ranking_report(traced, success_distance, args.temperature)

    thresholds = (
        [float(t) for t in args.thresholds.split(",")] if args.thresholds
        else [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
    print(f"\n{'thresh':>8} {'fired':>7} {'success':>8} {'spl':>7} {'fail':>6} "
          f"{'never':>6} {'cens':>5} {'med step':>9} {'med dist':>9}")
    best = None
    for threshold in thresholds:
        outcomes = [evaluate(r, threshold, args.temperature, success_distance)
                    for r in traced]
        kinds = [o[0] for o in outcomes]
        steps = sorted(o[1] for o in outcomes if o[1] is not None)
        dists = sorted(o[2] for o in outcomes if o[2] is not None)
        spls = [o[3] for o in outcomes if o[3] is not None]
        fired = kinds.count("success") + kinds.count("fail")
        rate = kinds.count("success") / len(traced)
        med_step = f"{steps[len(steps) // 2]:d}" if steps else "-"
        med_dist = f"{dists[len(dists) // 2]:.2f}" if dists else "-"
        # Mean over ALL traced episodes: an episode that never fires scores 0,
        # which is what "the policy failed to declare success" is worth.
        spl_text = f"{sum(spls) / len(traced):>7.3f}" if spls else f"{'-':>7}"
        print(f"{threshold:>8.3f} {fired:>7d} {rate:>8.3f} {spl_text} "
              f"{kinds.count('fail'):>6d} {kinds.count('never'):>6d} "
              f"{kinds.count('censored'):>5d} {med_step:>9} {med_dist:>9}")
        if best is None or rate > best[1]:
            best = (threshold, rate)
    if best:
        print(f"\nbest success in this sweep: {best[1]:.3f} at threshold {best[0]:.3f}")
    print("\n'success' is the fraction of ALL traced episodes where the threshold fires "
          "within success_distance;\n'spl' averages over the same denominator, using the "
          "path length at the moment of stopping.\n'cens' = the recorded episode was ended "
          "by the head, so this threshold's outcome is unknown -- read\nany row with cens > 0 "
          "as a lower bound, and re-run the split with no termination to remove it.")


if __name__ == "__main__":
    main()
