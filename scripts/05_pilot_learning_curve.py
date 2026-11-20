"""
Pilot learning curve for the data report: is the dataset large enough for transfer learning?

  python scripts/05_pilot_learning_curve.py 10     reads data/manifest_10.csv
  python scripts/05_pilot_learning_curve.py all    reads data/manifest_all.csv

What it does
  1. Splits the manifest by release (70/15/15, seed 42, stratified by album), so every image of a release
     lands in the same split. Each album is shuffled with its own seed, so an album gets the same split in
     every dataset that contains it (the 10 albums split identically in "10" and "all").
     Albums with too few releases for a val or test release keep them all in training. They are listed.
  2. Extracts features with a frozen Keras ResNet-50 pre-trained on ImageNet (images squashed to 224x224,
     Keras ResNet-50 preprocessing). Cached per image file (path + SHA-256) in data/features_resnet50.npz,
     shared by both datasets, so an image is never processed twice.
  3. Trains a logistic regression on those features with 25/50/75/100% of the training releases
     (3 repeats per size, class weights balanced), for both labels: album and packaging part (4 classes).
  4. Reports VALIDATION macro F1 and accuracy only. The test split is never touched here.

Output
  data/split_<name>.csv                          release_id, album, split
  results/pilot_results_<name>.csv               one row per label, fraction and repeat
  results/figures/pilot_learning_curve_<name>.png  macro F1 vs training size, both labels
  console: mean results, plus album accuracy per packaging part at 100%
  (reports/data_report/ holds the submitted report and its figure. This script never writes there.)

Needs:  uv add tensorflow scikit-learn
Run time: a few minutes for "10". For "all", feature extraction takes roughly 10 to 20 minutes on Apple Silicon
CPU, and logistic regression with 430 classes is much slower than with 10.
"""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]   # repo root
DATA = ROOT / "data"
FEATURES = DATA / "features_resnet50.npz"    # shared cache for every dataset, keyed per image file
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
REPEATS = 3
PARTS = ["front", "back", "medium", "booklet"]


# 1. Split by release, stratified by album -------------------------------------------------------
def load_rows(manifest):
    with open(manifest, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_by_release(rows, split_path):
    releases = defaultdict(set)                      # album -> release IDs
    for r in rows:
        releases[r["album"]].add(r["release_id"])
    assign, small = {}, []
    for album in sorted(releases):
        rel = sorted(releases[album])
        random.Random(f"{SEED}-{album}").shuffle(rel)
        n_test = round(len(rel) * TEST_FRAC)
        n_val = round(len(rel) * VAL_FRAC)
        if n_test == 0 or n_val == 0:
            # Fewer than about 4 releases: rounding leaves no val or test release, so the album sits in
            # training only and is never evaluated. Kept, not dropped, and reported below.
            small.append((album, len(rel), n_val, n_test))
        for i, rid in enumerate(rel):
            assign[rid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    if small:
        print(f"WARNING: {len(small)} of {len(releases)} albums have no val or no test release "
              f"(they stay in training and are never evaluated):")
        for album, n, n_val, n_test in small[:25]:
            print(f"  {album:<50} releases {n}, val {n_val}, test {n_test}")
        if len(small) > 25:
            print(f"  ... and {len(small) - 25} more (see the split file)")
    with open(split_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["release_id", "album", "split"])
        album_of = {r["release_id"]: r["album"] for r in rows}
        for rid in sorted(assign):
            w.writerow([rid, album_of[rid], assign[rid]])
    return assign


# 2. Frozen Keras ResNet-50 features, cached ------------------------------------------------------------
def load_cache(rows):
    """{(file, sha256): feature vector} from the shared cache.
    The first cache format stored only file paths (no SHA-256). Those entries take the SHA-256 from the
    current manifest, which is safe because images on disk are never downloaded again."""
    if not FEATURES.exists():
        return {}
    cached = np.load(FEATURES, allow_pickle=False)
    files, X = list(cached["files"]), cached["X"]
    if "sha256" in cached.files:
        shas = list(cached["sha256"])
    else:
        sha_of = {r["file"]: r["sha256"] for r in rows}
        shas = [sha_of.get(f, "") for f in files]
    return {(f, h): X[i] for i, (f, h) in enumerate(zip(files, shas)) if h}


def save_cache(cache):
    keys = sorted(cache)
    tmp = FEATURES.with_name(FEATURES.name + ".part")
    with open(tmp, "wb") as fh:                 # file handle, so numpy does not append its own .npz suffix
        np.savez_compressed(fh, X=np.stack([cache[k] for k in keys]).astype(np.float32),
                            files=np.array([k[0] for k in keys]), sha256=np.array([k[1] for k in keys]))
    tmp.replace(FEATURES)


def extract_features(rows):
    cache = load_cache(rows)
    keys = [(r["file"], r["sha256"]) for r in rows]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"Features: {len(rows) - len(todo)} cached, {len(todo)} to extract")

    if todo:
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

        # Frozen ResNet-50 without its classifier head; global average pooling gives a 2048-d vector
        model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
        model.trainable = False

        def load(path):
            img = Image.open(path).convert("RGB").resize((224, 224))   # squash, same as the training plan
            return np.asarray(img, dtype=np.float32)

        try:
            for start in tqdm(range(0, len(todo), 32), desc="ResNet-50 features", unit="batch",
                              dynamic_ncols=True):
                idx = todo[start:start + 32]
                x = preprocess_input(np.stack([load(DATA / rows[i]["file"]) for i in idx]))
                for i, vec in zip(idx, model.predict(x, verbose=0)):
                    cache[keys[i]] = vec
                if (start // 32) % 100 == 99:
                    save_cache(cache)                                  # checkpoint every 3,200 images
        finally:
            save_cache(cache)                                          # keeps finished batches on Ctrl-C
    return np.stack([cache[k] for k in keys]).astype(np.float32)


# 3. Learning curve -------------------------------------------------------------------------------
def fit_eval(X_tr, y_tr, X_va, y_va):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, accuracy_score
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    return f1_score(y_va, pred, average="macro"), accuracy_score(y_va, pred), pred


def main():
    ap = argparse.ArgumentParser(description="Pilot learning curve on one dataset.")
    ap.add_argument("dataset", choices=["10", "all"], help="reads data/manifest_<dataset>.csv")
    name = ap.parse_args().dataset
    manifest = DATA / f"manifest_{name}.csv"
    split_path = DATA / f"split_{name}.csv"
    results_path = ROOT / f"results/pilot_results_{name}.csv"
    figure = ROOT / f"results/figures/pilot_learning_curve_{name}.png"

    rows = load_rows(manifest)
    assign = split_by_release(rows, split_path)
    split = np.array([assign[r["release_id"]] for r in rows])
    X = extract_features(rows)
    labels = {"album": np.array([r["album"] for r in rows]),
              "part": np.array([r["type"] for r in rows])}
    release = np.array([r["release_id"] for r in rows])
    album = labels["album"]

    va = split == "val"
    album_of = {r["release_id"]: r["album"] for r in rows}
    train_rel = defaultdict(list)                    # album -> training release IDs
    for rid, s in assign.items():
        if s == "train":
            train_rel[album_of[rid]].append(rid)
    for a in train_rel:
        train_rel[a].sort()

    print(f"\nImages per split: " + ", ".join(f"{s} {int((split == s).sum())}" for s in ["train", "val", "test"]))

    out, last_pred = [], {}
    for frac in FRACTIONS:
        for rep in range(REPEATS if frac < 1.0 else 1):
            rng = random.Random(f"{SEED}-{frac}-{rep}")
            keep = set()
            for a, rel in train_rel.items():       # same fraction of releases from every album
                keep.update(rng.sample(rel, max(1, round(len(rel) * frac))))
            tr = np.isin(release, list(keep))
            for name, y in labels.items():
                f1, acc, pred = fit_eval(X[tr], y[tr], X[va], y[va])
                out.append({"label": name, "fraction": frac, "repeat": rep,
                            "train_images": int(tr.sum()), "val_macro_f1": round(f1, 4),
                            "val_accuracy": round(acc, 4)})
                if frac == 1.0:
                    last_pred[name] = pred
            print(f"  {int(frac * 100)}% of training releases, repeat {rep + 1}: done")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    # Summary table: mean over repeats
    print(f"\n{'label':<7}{'fraction':>9}{'train imgs':>12}{'macro F1':>10}{'accuracy':>10}")
    curves = defaultdict(list)
    for name in labels:
        for frac in FRACTIONS:
            sel = [o for o in out if o["label"] == name and o["fraction"] == frac]
            f1 = np.mean([o["val_macro_f1"] for o in sel]); sd = np.std([o["val_macro_f1"] for o in sel])
            acc = np.mean([o["val_accuracy"] for o in sel]); n = int(np.mean([o["train_images"] for o in sel]))
            curves[name].append((n, f1, sd))
            print(f"{name:<7}{int(frac * 100):>8}%{n:>12}{f1:>10.3f}{acc:>10.3f}")

    # Album accuracy on validation, broken down by packaging part (100% training data)
    part_va = labels["part"][va]
    hit = last_pred["album"] == album[va]
    print("\nAlbum accuracy on validation by packaging part (100% of training data):")
    for p in PARTS:
        m = part_va == p
        acc = f"{hit[m].mean():.3f}" if m.any() else "  n/a"
        print(f"  {p:<8} {acc}  (n={int(m.sum())})")

    # Plot
    n_albums = len(set(album))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, pts in curves.items():
        n, f1, sd = zip(*pts)
        ax.errorbar(n, f1, yerr=sd, marker="o", capsize=3,
                    label=f"Album ({n_albums} classes)" if name == "album" else "Packaging part (4 classes)")
    ax.set_xlabel("Training images")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Pilot learning curve: frozen ResNet-50 + logistic regression")
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    print(f"\nSaved {split_path.relative_to(ROOT)}, {results_path.relative_to(ROOT)} and "
          f"{figure.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
