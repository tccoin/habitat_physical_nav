"""Which PID limits the evaluation runs under, and why that is a choice.

The harness used to build a bare ``PIDPoseControllerConfig()``, so the
inference controller was whatever ``pid_pose_controller.py``'s dataclass
defaults happened to be. That is fine until you want to *change* them --
at which point there is no record of what the published numbers ran under,
and editing the dataclass silently rewrites the history of every prior run.

So the baseline is pinned here as data (``PID_BASELINE``), a test asserts the
dataclass still agrees with it, and runs select a named preset that is written
into the results JSON.

## Why there is a preset other than the baseline

The demonstrations this corpus is distilled from were *not* produced by this
controller. ``generate_pose2d_dataset.py`` drives the base through
``HolonomicRobotController.step`` (``controllers.py``), which computes

    w_body = clip(heading_error * heading_alignment_gain, -max_vw, +max_vw)

and writes it straight into ``chassis_twist`` -- **no angular acceleration
limit is applied to the body command at all**. (``HolonomicControllerConfig``
does carry ``aw_max=100.0``, but that rate-limits pure-pursuit's *virtual
heading* ``w_ref``, which never becomes the body command; likewise
``HeadingRotationController`` sets ``w_cmd`` instantly.) The inference PID, by
contrast, ramps at ``aw_max * dt``.

Measured over 58,520 setpoints from 60 ``v2_25hz`` validation episodes, the
angular acceleration implied by the demonstrations is:

    p50 0.79   p90 12.00   p99 41.51   max 100.14  rad/s^2

against a baseline ``aw_max`` of 6.98 rad/s^2 (400 deg/s^2) -- **exceeded by
23.8% of demonstration ticks**. The linear side is not mismatched: only 2.0%
of ticks exceed ``a_max = 8.0``. Demonstrated |w| averages 1.112 rad/s, which
the baseline needs 4 of a chunk's 5 ticks to even reach from rest.

``demo_matched`` therefore raises the angular limits to what data generation
effectively allowed, and leaves everything else alone. It is a hypothesis
about why decoder-level heading improvements do not reach navigation metrics,
not an established fix -- compare it against ``baseline`` on the same episode
set before believing it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

# The limits every result up to 2026-08-06 was produced under. Pinned as data
# so that changing the dataclass defaults cannot silently redefine "baseline";
# tests/test_objectnav_eval.py asserts the two still agree.
PID_BASELINE: Dict[str, float] = {
    "kp_pos": 2.5,
    "ki_pos": 0.0,
    "kd_pos": 0.3,
    "kp_th": 4.0,
    "ki_th": 0.0,
    "kd_th": 0.2,
    "v_max": 2.0,
    "a_max": 8.0,
    "w_max": math.radians(180.0),
    "aw_max": math.radians(400.0),
    "integral_limit_pos": 0.5,
    "integral_limit_th": math.radians(30.0),
    "feedforward_gain": 1.0,
}

PID_PRESETS: Dict[str, Dict[str, float]] = {
    # Exactly PID_BASELINE. Named so a run can state it rather than imply it.
    "baseline": {},
    # Angular limits raised to what data generation effectively permitted:
    # aw_max to the demonstrations' observed ceiling (100 rad/s^2), w_max to
    # the data-gen cap plus the margin the recorded trajectories actually use
    # (max |w| measured 2.254 rad/s). Linear limits untouched -- they are not
    # the mismatch.
    "demo_matched": {
        "aw_max": 100.0,
        "w_max": 2.5,
    },
    # Half way, to tell "the limit binds" from "the limit is the whole story".
    "demo_half": {
        "aw_max": 25.0,
        "w_max": 2.5,
    },
    # No limits at all -- the PID gains alone decide the twist.
    #
    # The argument for it: an action chunk is already a *ramped* trajectory.
    # The model was trained on setpoints that encode their own acceleration
    # profile, so a rate limiter downstream is not enforcing physics the chunk
    # ignored, it is re-imposing a ramp on top of one that is already there,
    # and the second ramp is the one the model never saw. Under that reading
    # the limits are not a safety feature but a distortion, and the cleanest
    # test is to remove them entirely.
    #
    # The risk it carries: the limits are also what bounds the response to a
    # *large* tracking error. When the robot is wedged against geometry the
    # setpoint keeps advancing while the pose does not, so `error_xy` grows
    # without bound and `kp_pos * error_xy` with it. Nothing then caps the
    # commanded speed. Watch `blocked_drive_frac` and the contact probe on any
    # run using this, and treat a large jump in path length as suspect rather
    # than as progress.
    "unlimited": {
        "v_max": float("inf"),
        "a_max": float("inf"),
        "w_max": float("inf"),
        "aw_max": float("inf"),
    },
    # `unlimited` minus the failure mode above: the rate limits go, but the
    # velocity ceilings stay at what the demonstrations actually reached, so a
    # wedged robot cannot command an arbitrarily large twist. This is the one
    # to prefer if `unlimited` tracks well but behaves badly in contact.
    "unramped": {
        "a_max": float("inf"),
        "aw_max": float("inf"),
        "v_max": 2.0,
        "w_max": 2.5,
    },
}


def _coerce(name: str, raw: Any) -> float:
    """Accept ``aw_max=100`` and ``aw_max=deg:400`` alike.

    The dataclass writes two of its angular defaults as ``math.radians(...)``,
    so a reader who copies a number off the source is as likely to reach for
    degrees as radians. Making the unit sayable is cheaper than a silent 57x.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("deg:"):
            return math.radians(float(text[4:]))
        if text.startswith("rad:"):
            return float(text[4:])
        return float(text)
    return float(raw)


def resolve_pid_settings(
    preset: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Baseline <- preset <- explicit overrides, as a plain dict.

    Returned rather than applied so the caller can record it: the settings
    belong in the results file next to the numbers they produced.
    """
    name = preset or "baseline"
    if name not in PID_PRESETS:
        raise ValueError(
            f"unknown PID preset {name!r}; have {sorted(PID_PRESETS)}"
        )
    settings = dict(PID_BASELINE)
    settings.update(PID_PRESETS[name])
    for key, raw in (overrides or {}).items():
        if key not in PID_BASELINE:
            raise ValueError(
                f"unknown PID field {key!r}; have {sorted(PID_BASELINE)}"
            )
        settings[key] = _coerce(key, raw)
    return settings


def build_pid_config(
    preset: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
):
    """The resolved settings as a ``PIDPoseControllerConfig``.

    Imported lazily: this module is also read by the reporting side, which
    runs in environments that need not have ``continuous_demos`` importable.
    """
    from continuous_demos.pid_pose_controller import PIDPoseControllerConfig

    return PIDPoseControllerConfig(**resolve_pid_settings(preset, overrides))


def describe_pid(settings: Mapping[str, float]) -> str:
    """One line for the run banner -- only what differs from the baseline."""
    changed = [
        f"{k}={settings[k]:.4g}"
        for k in sorted(settings)
        if abs(settings[k] - PID_BASELINE[k]) > 1e-12
    ]
    return "baseline" if not changed else ", ".join(changed)
