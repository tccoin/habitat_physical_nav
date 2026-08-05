"""Translate a `longnav.constants.episode_labels_table` subset into harness uids.

The two repos mean different things by ``episode_id``, and the difference is an
off-by-one that evaluates the wrong episodes rather than failing:

* **longnav** loads through habitat-lab's stock ``ObjectNavDatasetV1.from_json``,
  which ends with ``for i, episode in enumerate(...): episode.episode_id = str(i)``.
  Its label ``mv2HUxq3B53_55`` therefore means *index 55 in that scene's JSON*.
* **this harness** uses ``ObjectNavReplayDataset``, which deliberately preserves
  the real id -- and in HM3D val every episode carries ``episode_id == "0"``. Uids
  are disambiguated by occurrence instead: index ``i`` is ``scene:0`` when ``i == 0``
  and ``scene:0#{i+1}`` after that.

So ``mv2HUxq3B53_55`` is ``mv2HUxq3B53:0#56``. Both orderings are file order, which
is what makes the translation well defined -- and which this script *checks*
rather than assumes, by comparing ``start_position`` against the raw
``content/*.json.gz`` at every index of every scene it touches.

    python scripts/episode_set_from_longnav.py sample101 \\
        --episodes /Projects/spatial_training/data/datasets/objectnav/hm3d/v1/val \\
        --out .../episode_sets/hm3d_val_sample101.txt
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_LONGNAV_SRC = "/Projects/spatial_training/src"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subset", help="key in longnav.constants.episode_labels_table")
    p.add_argument("--episodes", required=True, help="ObjectNav dataset dir")
    p.add_argument("--out", default=None, help="write the comma-separated uids here")
    p.add_argument("--longnav-src", default=DEFAULT_LONGNAV_SRC)
    p.add_argument("--no-verify", action="store_true",
                   help="skip the per-index start_position check (do not)")
    args = p.parse_args()

    if Path(args.longnav_src).is_dir() and args.longnav_src not in sys.path:
        sys.path.insert(0, args.longnav_src)

    from longnav.constants import episode_labels_table

    from objectnav_eval.episodes import build_episode_source

    if args.subset not in episode_labels_table:
        raise SystemExit(
            f"unknown subset {args.subset!r}; known: {sorted(episode_labels_table)}"
        )
    labels = episode_labels_table[args.subset]

    # `load()` returns file order; the sort happens later, in EpisodeFilter.
    episodes = build_episode_source("objectnav", path=args.episodes).load()
    by_scene: dict = {}
    for episode in episodes:
        by_scene.setdefault(episode.scene_name, []).append(episode)

    uids, missing = [], []
    for label in labels:
        scene, _, index = label.rpartition("_")
        i = int(index)
        if scene in by_scene and i < len(by_scene[scene]):
            uids.append(by_scene[scene][i].uid)
        else:
            missing.append(label)

    if not args.no_verify:
        content = Path(args.episodes) / "content"
        for scene in sorted({u.split(":")[0] for u in uids}):
            path = content / f"{scene}.json.gz"
            if not path.is_file():
                print(f"  ! {scene}: no {path.name}, index alignment unverified")
                continue
            with gzip.open(path) as fh:
                raw = json.load(fh)["episodes"]
            ours = by_scene[scene]
            if len(raw) != len(ours) or not all(
                np.allclose(raw[i]["start_position"], ours[i].start_position)
                for i in range(len(raw))
            ):
                raise SystemExit(
                    f"FAIL {scene}: raw JSON order and harness load order disagree, so "
                    "index-to-uid translation is not valid for this dataset"
                )
        print(f"verified index alignment on {len({u.split(':')[0] for u in uids})} scene(s)")

    print(f"{args.subset}: {len(labels)} label(s) -> {len(uids)} uid(s), "
          f"{len(set(uids))} distinct, {len(missing)} unresolved")
    if missing:
        raise SystemExit(f"unresolved labels: {missing}")

    text = ",".join(uids)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
