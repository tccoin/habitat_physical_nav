"""Round-trip / known-value checks for hab_quat_to_heading.

hab_quat_to_heading is pure math (no simulator instance needed), but it
currently lives in continuous_demos.replay, which imports continuous_demos.sim
(habitat_sim) and continuous_demos.controllers (magnum) at module level just
for type hints -- so importing it transitively requires the full `habitat`
conda env even though the function itself doesn't. Skip cleanly rather than
fail where those aren't installed, so the rest of the suite still runs
anywhere; this is a real dependency-layering wart worth revisiting if these
heavy imports ever become annoying (e.g. by moving pure math like this into
a dependency-free module), but that's out of scope here.

Expected values below were derived by hand from the function's own formula
(rotating habitat local -z by the quaternion) and then cross-checked
numerically against the real implementation before being written down here
-- see the derivation: for a quaternion representing rotation by `yaw` about
habitat's +y (up) axis, hab_quat_to_heading returns wrap(yaw + pi/2). The
+pi/2 offset is real, not a bug: the identity quaternion's forward (habitat
local -z) maps to the planar frame's +Y axis, not its +X (theta=0) axis.
"""

import math

import numpy as np
import pytest

hab_quat_to_heading = pytest.importorskip(
    "continuous_demos.replay",
    reason="continuous_demos.replay transitively requires habitat_sim/magnum",
).hab_quat_to_heading


def quat_from_yaw(yaw: float):
    """xyzw quaternion for a rotation of `yaw` about the habitat +y (up) axis."""
    half = yaw / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class TestHabQuatToHeading:
    def test_identity_rotation_faces_planar_plus_y(self):
        # Identity quaternion's forward (habitat local -z) maps to the planar
        # frame's +Y axis (heading = +90 deg), not its +X/theta=0 axis.
        heading = hab_quat_to_heading((0.0, 0.0, 0.0, 1.0))
        assert heading == pytest.approx(math.pi / 2, abs=1e-6)

    @pytest.mark.parametrize("yaw_deg", [30, 90, 180, -90, -45])
    def test_matches_yaw_plus_90_degrees(self, yaw_deg):
        yaw = math.radians(yaw_deg)
        heading = hab_quat_to_heading(quat_from_yaw(yaw))
        expected = wrap(yaw + math.pi / 2)
        # Compare on the circle to sidestep the +-180 deg boundary ambiguity.
        diff = wrap(heading - expected)
        assert diff == pytest.approx(0.0, abs=1e-6)

    def test_full_turn_returns_to_start(self):
        heading = hab_quat_to_heading(quat_from_yaw(2 * math.pi))
        assert heading == pytest.approx(math.pi / 2, abs=1e-6)
