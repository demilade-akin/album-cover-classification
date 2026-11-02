"""
Split the cleaned album images into train / val / test for Keras.

- Groups near-identical images (same scan resized or recompressed) so they never straddle splits.
- Stratifies by album, and by "look" if looks.csv exists (e.g. GKMC minivan, Thriller alternates),
  so every tagged look reaches train first, then test, then val.
- Copies (never moves) images from data/assignment2/dataset/ into data/assignment2/split/<train|val|test>/<album>/
  and saves data/assignment2/split.csv.

Needs:  pip install imagehash pandas pillow
Run from anywhere:  python scripts/03_split_assignment2.py   (or paste into a notebook cell run from the project folder)
"""
import random, shutil
from itertools import combinations
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # project folder
A2 = ROOT / "data/assignment2"
DATA_DIR = A2 / "dataset"
OUT_DIR = A2 / "split"            # rebuilt on every run, keep nothing else in it
LOOKS_CSV = A2 / "looks.csv"      # optional: columns file,look
HASH_THRESHOLD = 4                 # pHash bits (of 64) counted as "same image"; keep strict
VAL_FRAC, TEST_FRAC = 0.15, 0.15
SEED = 42

assert DATA_DIR.is_dir(), f"{DATA_DIR} not found"
print(f"Reading albums from {DATA_DIR}/\n")

# 1. Hash every image, and flag files the sanity check should have converted
rows, unconverted = [], []
for path in sorted(DATA_DIR.glob("*/*.jpg")):
    with Image.open(path) as im:
        if im.mode != "RGB" or im.format != "JPEG":
            unconverted.append(f"{path} ({im.format} {im.mode})")
        rows.append({"file": path.name, "cls": path.parent.name,
                     "path": str(path), "hash": imagehash.phash(im)})
df = pd.DataFrame(rows)
if unconverted:
    print("WARNING: run the sanity check with FIX = True first. Not yet RGB JPEG:")
    for u in unconverted:
        print("  ", u)
    print()

# 2. Attach alternate-look tags (everything untagged is "standard")
df["look"] = "standard"
if LOOKS_CSV.exists():
    looks = pd.read_csv(LOOKS_CSV).set_index("file")["look"]
    df["look"] = df["file"].map(looks).fillna("standard")
    missing = set(looks.index) - set(df["file"])
    if missing:
        print("WARNING: looks.csv names files that don't exist:", missing, "\n")

# 3. Group near-identical images within each album (union-find)
parent = list(range(len(df)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


hashes, classes = df["hash"].tolist(), df["cls"].tolist()
cross_class = []
for i, j in combinations(range(len(df)), 2):
    if hashes[i] - hashes[j] <= HASH_THRESHOLD:
        if classes[i] == classes[j]:
            parent[find(i)] = find(j)
        else:
            cross_class.append((df.at[i, "path"], df.at[j, "path"]))
df["group"] = [find(i) for i in range(len(df))]

# 4. Report the groups so you can check them by eye
sizes = df.groupby("group").size()
print(f"{len(df)} images, {len(sizes)} groups, {int((sizes > 1).sum())} groups with duplicates")
for gid in sizes[sizes > 1].sort_values(ascending=False).index:
    print(f"  Group of {sizes[gid]}:", ", ".join(df.loc[df["group"] == gid, "path"]))
if cross_class:
    print("\nSame image under two different albums (likely mislabels):")
    for a, b in cross_class:
        print("  ", a, "<->", b)

# 5. Assign whole groups to splits, stratified by album and look
rng = random.Random(SEED)
groups = (df.groupby("group")
            .agg(cls=("cls", "first"), look=("look", "first"), n=("file", "size"))
            .reset_index())
assign = {}
for _, g in groups.groupby(["cls", "look"], sort=True):
    ids = g["group"].tolist()
    rng.shuffle(ids)
    n_of = dict(zip(g["group"], g["n"]))
    total = g["n"].sum()
    target = {"test": round(TEST_FRAC * total), "val": round(VAL_FRAC * total)}
    count = {"train": 0, "val": 0, "test": 0}
    for k, gid in enumerate(ids):
        if k == 0:
            s = "train"                          # every look reaches train first
        elif k == 1:
            s = "test"                           # then test
        elif k == 2 and len(ids) >= 4:
            s = "val"                            # then val, if there's enough
        elif count["test"] < target["test"]:
            s = "test"
        elif count["val"] < target["val"]:
            s = "val"
        else:
            s = "train"
        assign[gid] = s
        count[s] += n_of[gid]
df["split"] = df["group"].map(assign)

# 6. Checks
assert df.groupby("group")["split"].nunique().max() == 1, "a group spans two splits"
table = pd.crosstab(df["cls"], df["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
print("\nImages per album and split:")
print(table)
print("Totals:", table.sum().to_dict())
if (table == 0).any().any():
    print("WARNING: an album has no images in at least one split")
alt = df[df["look"] != "standard"]
if len(alt):
    print("\nAlternate looks:")
    print(pd.crosstab([alt["cls"], alt["look"]], alt["split"]))

# 7. Copy into split/<split>/<album>/ for Keras, and save the record
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
for r in df.itertuples():
    dest = OUT_DIR / r.split / r.cls
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(r.path, dest / r.file)
df["hash"] = df["hash"].astype(str)
df.drop(columns="path").to_csv(A2 / "split.csv", index=False)

class_names = sorted(df["cls"].unique())
print(f"\nCopied to {OUT_DIR}/ and saved split.csv")
print("class_names for Keras:", class_names)