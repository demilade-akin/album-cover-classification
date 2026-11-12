"""
Split the cleaned recipe images into train / val / test folders for Keras.

Input
  data/clean.csv, data/clean/<class>/<file>.jpg       output of 02_clean.py

Split
  70 / 15 / 15, seed 42, stratified by class and source site together. Each site photographs food in its
  own style, and some sites supply most of one class (sugarspunrun: half of cake and cookies,
  omnivorescookbook: most of stir fry). Stratifying by site as well keeps every site's share the same in
  train, val and test, so the test set measures the same mix the model trained on. split.csv keeps the
  site column, so the notebook can report accuracy per site and check for a site shortcut.
  Strata too small to split three ways (under MIN_STRATUM images) are merged into one "other sites" stratum
  per class. Near-duplicates were already removed by 02_clean.py, so no photo can sit on both sides.

Output
  data/split/<train|val|test>/<class>/<file>.jpg   copies (never moves), rebuilt from scratch on every run
  data/split.csv                                   filename, class, source_site, split

Run from anywhere:  python assignment2/scripts/03_split.py
"""
import csv, shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # assignment2 folder
DATA = ROOT / "data"
CLEAN_DIR = DATA / "clean"
CLEAN_CSV = DATA / "clean.csv"
SPLIT_DIR = DATA / "split"
SPLIT_CSV = DATA / "split.csv"

CLASSES = ["pasta", "soup", "salad", "cake", "cookies", "stir_fry"]
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
MIN_STRATUM = 10        # a class-site group smaller than this joins its class's "other sites" group


def strata(rows):
    """Stratum label per row: class|site, or class|other for small class-site groups."""
    size = Counter((r["class"], r["source_site"]) for r in rows)
    return [f"{r['class']}|{r['source_site'] if size[(r['class'], r['source_site'])] >= MIN_STRATUM else 'other'}"
            for r in rows]


def main():
    with open(CLEAN_CSV, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (r["class"], r["filename"]))   # fixed order: same split every run

    # Two-step split: first take test off, then split the rest into train and val.
    labels = strata(rows)
    rest, test = train_test_split(rows, test_size=TEST_FRAC, stratify=labels, random_state=SEED)
    train, val = train_test_split(rest, test_size=VAL_FRAC / (1 - TEST_FRAC), stratify=strata(rest),
                                  random_state=SEED)
    assignment = {r["filename"]: "train" for r in train} | {r["filename"]: "val" for r in val} | \
                 {r["filename"]: "test" for r in test}
    assert len(assignment) == len(rows), "a filename appears twice in clean.csv"

    # Rebuild the split folders from scratch so no file from an older split lingers.
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for r in rows:
        dest = SPLIT_DIR / assignment[r["filename"]] / r["class"] / r["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CLEAN_DIR / r["class"] / r["filename"], dest)

    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "class", "source_site", "split"])
        w.writeheader()
        for r in rows:
            w.writerow({"filename": r["filename"], "class": r["class"], "source_site": r["source_site"],
                        "split": assignment[r["filename"]]})

    # Tables for the write-up
    splits = ["train", "val", "test"]
    by_class = Counter((r["class"], assignment[r["filename"]]) for r in rows)
    print(f"{'class':<10}" + "".join(f"{s:>7}" for s in splits) + f"{'total':>7}")
    for c in CLASSES:
        print(f"{c:<10}" + "".join(f"{by_class[(c, s)]:>7}" for s in splits) + f"{sum(by_class[(c, s)] for s in splits):>7}")
    print(f"{'total':<10}" + "".join(f"{sum(by_class[(c, s)] for c in CLASSES):>7}" for s in splits) + f"{len(rows):>7}")

    by_site = Counter((r["source_site"], assignment[r["filename"]]) for r in rows)
    sites = sorted({r["source_site"] for r in rows})
    print(f"\n{'site':<24}" + "".join(f"{s:>7}" for s in splits))
    for site in sites:
        print(f"{site:<24}" + "".join(f"{by_site[(site, s)]:>7}" for s in splits))
    print(f"\nSaved {SPLIT_DIR.relative_to(ROOT)}/ and {SPLIT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
