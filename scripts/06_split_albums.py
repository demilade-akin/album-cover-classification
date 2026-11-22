"""
Album-level split for the final experiments.

  python scripts/06_split_albums.py

What it does
  1. Reads data/manifest_all.csv (one row per image) and data/metadata/albums_10.csv.
  2. Puts the 10 albums of albums_10.csv in train. They are forced into the training pool so that the
     breadth vs depth experiment (A: 10 albums, B: same image count over many albums, C: all training albums)
     tests on the same test albums.
  3. Sorts every other album by image count and cuts the list into 4 size bands (quartiles).
     Each band is shuffled with seed 42 and split about 70/15/15 into train, val and test, so large and
     small albums are spread evenly over the three splits.
  4. Writes data/split_albums.csv (album, split). This file is committed and fixes the split for every
     final experiment. Test albums stay untouched until the final evaluation.

Console summary per split: albums, releases, images, images per part, and how many albums have at least
2 releases (only those albums take part in verification, because a query's own release is removed from
its gallery).
"""
import csv, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
DATA = ROOT / "data"
MANIFEST = DATA / "manifest_all.csv"
ALBUMS_10 = DATA / "metadata/albums_10.csv"
OUT = DATA / "split_albums.csv"
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
N_BANDS = 4
SPLITS = ["train", "val", "test"]
PARTS = ["front", "back", "medium", "booklet"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_csv(MANIFEST)
    forced = {r["folder"] for r in read_csv(ALBUMS_10)}   # folder name = album key in the manifest

    images = defaultdict(int)                              # album -> image count
    for r in rows:
        images[r["album"]] += 1
    missing = forced - set(images)
    if missing:
        raise SystemExit(f"albums_10.csv albums missing from the manifest: {sorted(missing)}")

    # 1. The 10 pilot albums go to train first
    assign = {a: "train" for a in forced}

    # 2. Size bands over the remaining albums. Ties in image count are broken by name so the order
    #    (and the split) does not depend on dictionary order.
    rest = sorted((a for a in images if a not in forced), key=lambda a: (images[a], a))
    rng = random.Random(SEED)
    for b in range(N_BANDS):
        band = rest[b * len(rest) // N_BANDS:(b + 1) * len(rest) // N_BANDS]   # equal-size quartile bands
        band = sorted(band)
        rng.shuffle(band)
        n_test = round(len(band) * TEST_FRAC)
        n_val = round(len(band) * VAL_FRAC)
        for i, a in enumerate(band):
            assign[a] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        print(f"Band {b + 1}: {len(band)} albums, {images[min(band, key=images.get)]}"
              f"-{images[max(band, key=images.get)]} images each, val {n_val}, test {n_test}")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["album", "split"])
        for a in sorted(assign):
            w.writerow([a, assign[a]])

    # 3. Summary per split
    releases = defaultdict(set)                            # album -> release IDs
    part_count = defaultdict(lambda: defaultdict(int))    # split -> part -> images
    for r in rows:
        releases[r["album"]].add(r["release_id"])
        part_count[assign[r["album"]]][r["type"]] += 1

    print(f"\n{'split':<6}{'albums':>7}{'releases':>10}{'images':>8}"
          + "".join(f"{p:>9}" for p in PARTS) + f"{'>=2 rel':>9}")
    for s in SPLITS:
        albums = [a for a in assign if assign[a] == s]
        n_rel = sum(len(releases[a]) for a in albums)
        n_img = sum(images[a] for a in albums)
        multi = sum(len(releases[a]) >= 2 for a in albums)
        print(f"{s:<6}{len(albums):>7}{n_rel:>10}{n_img:>8}"
              + "".join(f"{part_count[s][p]:>9}" for p in PARTS) + f"{multi:>9}")
    print(f"total {len(assign):>7}{sum(len(v) for v in releases.values()):>10}{len(rows):>8}")
    print("\n'>=2 rel' = albums with at least 2 releases. Only val and test albums of this kind take part "
          "in verification.")
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
