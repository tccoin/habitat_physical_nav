"""Regression test for the reference_replay action/state ordering invariant.

No habitat_sim/magnum dependency -- verify_replay_format.py only needs gzip/
json/numpy, so this runs in any env. This is a real category-1 invariant
check (see REPLAY_DATA_FORMAT.md): entry i's action produced entry i's state,
verified over every transition of every episode, not a hand-picked example.

Invoked as a subprocess against the actual script exactly as a user would run
it (`python scripts/verify_replay_format.py <dataset>`), rather than importing
its internals -- scripts/ is intentionally not an importable package (see
CLAUDE.md: scripts are CLI entry points, not library code), so this is the
lowest-friction way to regression-test it without restructuring the script.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "examples" / "17DRP5sb8fy.json.gz"
SCRIPT = REPO_ROOT / "scripts" / "verify_replay_format.py"


def test_no_ordering_violations_on_bundled_dataset():
    assert DATASET.exists(), f"bundled example dataset missing: {DATASET}"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(DATASET)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"verify_replay_format.py exited {result.returncode} "
        f"(a hard invariant assertion failed):\n{result.stdout}\n{result.stderr}"
    )
    assert "ordering violations (total): 0" in result.stdout, (
        "expected zero action/state ordering violations on the bundled dataset:\n"
        f"{result.stdout}"
    )
    assert "CONFIRMED" in result.stdout
