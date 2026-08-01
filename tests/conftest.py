"""Make ``src/`` importable when the package has not been pip-installed.

``pip install -e .`` is the documented setup, but the evaluation system is also
run out of a checkout from environments that must not have it installed (the
model environment, which the policy bridge launches). Keeping the tests
independent of the install state means the same suite covers both.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
