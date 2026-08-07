"""Guards that stop a generation run from writing where it must not.

``generate_pose2d_dataset.py`` is resumable by design: it appends to whatever
is already under ``--output-dir`` and skips episode ids it finds there. That is
the right behaviour for resuming an interrupted run and a dangerous one for a
mistyped path, because appending is silent and succeeds. The recording-params
check (``recording_metadata``) catches a resume whose *parameters* differ; it
cannot catch a resume that never should have been a resume at all -- a new run
pointed at somebody else's finished corpus, with parameters that happen to
match, quietly interleaving its episodes into it.

Nothing downstream can undo that. The built corpora reference their source
recordings by absolute path, so a shard that gains lines makes every dataset
built from it inconsistent with itself, and there is no marker to say which
lines are new. Hence two checks here, both hard errors:

1. :func:`assert_output_dir_is_safe` -- the output must not be, contain, or sit
   inside the directory the input dataset was read from, and must not sit under
   anything named in ``HABITAT_PROTECTED_ROOTS``. A run that writes into its own
   input corrupts the thing it is reading, mid-read.

2. The corpus marker (:func:`read_corpus_marker` / :func:`write_corpus_marker`)
   -- a small JSON file this generator drops in an output directory the first
   time it uses it. On a later run: no marker but shards present means the
   directory belongs to a generation that predates the marker, i.e. is somebody
   else's finished corpus, and is refused. A marker that disagrees with the
   current parameters is refused with the field named. Only a matching marker,
   or a genuinely fresh directory, is allowed to proceed.

The marker deliberately fails CLOSED for every corpus that already exists: they
have no marker, so no run can resume into one by accident. ``--adopt-existing-
corpus`` is the explicit escape hatch, and exists so that "I really do mean this
directory" is a thing an operator has to type rather than a thing a typo can do.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Name of the marker file, at the top level of ``--output-dir``.
CORPUS_MARKER_NAME = "corpus_params.json"

#: Colon-separated absolute paths that no run may write into, read from the
#: environment so that deployment-specific paths stay out of this standalone
#: package. Set it to the roots holding corpora and training data, e.g.
#: ``export HABITAT_PROTECTED_ROOTS=/data/output_25hz:/data/built``.
PROTECTED_ROOTS_ENV = "HABITAT_PROTECTED_ROOTS"


class UnsafeOutputDir(RuntimeError):
    """The requested ``--output-dir`` would write over data it must not."""


def _resolve(path) -> Path:
    return Path(path).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    """Is ``child`` equal to, or nested inside, ``parent``? (Both resolved.)"""
    return child == parent or parent in child.parents


def protected_roots() -> List[Path]:
    """Absolute roots named by :data:`PROTECTED_ROOTS_ENV`, resolved."""
    raw = os.environ.get(PROTECTED_ROOTS_ENV, "")
    return [_resolve(p) for p in raw.split(os.pathsep) if p.strip()]


def assert_output_dir_is_safe(
    output_dir,
    dataset_path,
    image_root=None,
    extra_protected=(),
) -> None:
    """Refuse an output location that overlaps the input or a protected root.

    ``dataset_path`` is the ``.json.gz`` being replayed; its *directory* is what
    is protected, since that is the shard set a batch run reads from. Both the
    output directory and the image root are checked, because ``--image-root``
    can point somewhere else entirely.
    """
    out = _resolve(output_dir)
    source_dir = _resolve(dataset_path).parent
    targets = [("--output-dir", out)]
    if image_root is not None:
        targets.append(("--image-root", _resolve(image_root)))

    for flag, target in targets:
        if _is_within(target, source_dir) or _is_within(source_dir, target):
            raise UnsafeOutputDir(
                f"{flag} {target} overlaps the directory the input dataset is read "
                f"from ({source_dir}). Generation appends, so this would write into "
                "its own input while reading it. Choose an output directory outside "
                "the dataset tree."
            )
        for root in list(protected_roots()) + [_resolve(p) for p in extra_protected]:
            if _is_within(target, root):
                raise UnsafeOutputDir(
                    f"{flag} {target} is inside the protected root {root} (named by "
                    f"${PROTECTED_ROOTS_ENV}). Generation appends silently, and these "
                    "roots hold corpora that other things reference by absolute path. "
                    "Choose an output directory outside them."
                )


def marker_path(output_dir) -> Path:
    return Path(output_dir) / CORPUS_MARKER_NAME


def read_corpus_marker(output_dir) -> Optional[Dict[str, Any]]:
    """The marker this generator left in ``output_dir``, or ``None``."""
    path = marker_path(output_dir)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return blob if isinstance(blob, dict) else None


def write_corpus_marker(output_dir, params_blob: Dict[str, Any]) -> None:
    """Drop (or refresh) the marker, atomically.

    Atomically because a batch run fans this generator out over ~24 concurrent
    worker processes sharing one ``--output-dir``, every one of which will try
    to write it at about the same moment. ``os.replace`` makes the losers
    harmless instead of leaving a half-written file for the next run to choke
    on.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(directory), prefix=".corpus_params.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(params_blob, f, indent=2, sort_keys=True)
        os.replace(tmp, marker_path(directory))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def existing_shards(metadata_dir) -> List[Path]:
    directory = Path(metadata_dir)
    return sorted(directory.glob("*.jsonl")) if directory.exists() else []


# ---------------------------------------------------------------------------
# The two refusals themselves
# ---------------------------------------------------------------------------
def assert_shard_is_appendable(jsonl_path, params) -> None:
    """Refuse to append records to a shard that was recorded differently.

    Generation is resumable and appends, so without this check a rerun after a
    parameter change would leave one shard holding two generations with nothing
    to mark the boundary. Aborting is deliberate: it is an operator error about
    the whole ``--output-dir``, not a per-episode failure, and the fix is a
    fresh output dir rather than anything the generator can do.

    Three parameter changes reach here, and none of them is detectable
    downstream once the records are interleaved:

      * the tick rate (the 30 Hz -> 25 Hz move, which motivated the block);
      * ``--panoramic`` -- half the shard's frames would be one image and half
        a strip of N views, with ``rgb_paths`` giving no hint which is which;
      * ``--no-phases`` -- half the trajectories would be human-segmented
        replays and half continuous ones, in a corpus whose entire purpose is
        to be imitated.

    This is a hard error and must stay one. A warning would be worth nothing:
    it scrolls past in a 24-hour run fanned out over ~24 concurrent workers,
    and the corpus is then poisoned in a way that yields plausible numbers
    rather than a crash.
    """
    from continuous_demos.recording_metadata import format_disagreements, shard_params

    jsonl_path = Path(jsonl_path)
    existing = shard_params(jsonl_path) if jsonl_path.exists() else None
    if existing is None:
        if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
            raise RuntimeError(
                f"{jsonl_path} already holds records with no 'recording' block, i.e. from a "
                f"generation that predates the schema (those were all recorded at 30 Hz). "
                f"Appending {params.native_fps:g} Hz records to it would produce a shard with "
                "two time bases and no way to tell which record is which. Write to a fresh "
                "--output-dir."
            )
        return
    diffs = existing.disagreements(params)
    if diffs:
        raise RuntimeError(
            f"REFUSING TO RESUME: {jsonl_path} was not recorded the way this run is "
            f"recording.\n"
            + format_disagreements(diffs, mine="already in shard", theirs="this run")
            + f"\n  already in shard: {existing.describe_rig()} @ "
            f"{existing.native_fps:g} Hz (schema {existing.schema_version})"
            + f"\n  this run:         {params.describe_rig()} @ "
            f"{params.native_fps:g} Hz (schema {params.schema_version})"
            + "\nResuming would interleave two generations in one shard, which nothing "
            "downstream can separate. Write to a fresh --output-dir."
        )


def assert_corpus_marker_agrees(
    output_dir, metadata_dir, params, adopt_existing_corpus: bool = False
) -> None:
    """Refuse an ``--output-dir`` that belongs to a different generation.

    :func:`assert_shard_is_appendable` guards one scene's shard; this guards the
    whole directory, including scenes this run has not reached yet, and catches
    the case that check cannot see: a *new* run pointed at a *finished* corpus
    whose parameters happen to match, which would interleave its episodes into
    it with nothing to mark which lines are new.

    The marker fails closed for every corpus written before it existed -- they
    have none, so shards-without-marker is refused. That is the point: no run
    can accidentally resume into the corpora already on disk.
    """
    from continuous_demos.recording_metadata import RecordingParams, format_disagreements

    marker = read_corpus_marker(output_dir)
    if marker is None:
        if existing_shards(metadata_dir) and not adopt_existing_corpus:
            raise UnsafeOutputDir(
                f"REFUSING TO WRITE: {output_dir} already contains metadata shards but no "
                f"{CORPUS_MARKER_NAME}, so it was written by a generation that predates "
                "the corpus marker -- i.e. it is an existing corpus, not this run's "
                "output. Appending to it would interleave two generations with nothing "
                "to separate them, and anything built from it references it by absolute "
                "path. Use a fresh --output-dir. If you genuinely intend to extend this "
                "corpus, pass --adopt-existing-corpus."
            )
        write_corpus_marker(output_dir, params.as_dict())
        return

    existing = RecordingParams.from_dict(marker)
    diffs = existing.disagreements(params)
    if diffs:
        raise UnsafeOutputDir(
            f"REFUSING TO RESUME: {output_dir} was created by a different generation.\n"
            + format_disagreements(diffs, mine="corpus marker", theirs="this run")
            + f"\n  corpus marker: {existing.describe_rig()} @ {existing.native_fps:g} Hz"
            + f"\n  this run:      {params.describe_rig()} @ {params.native_fps:g} Hz"
            + f"\nSee {marker_path(output_dir)}. Write to a fresh --output-dir."
        )
