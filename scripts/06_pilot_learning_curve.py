"""
Pilot learning curve for the data report: is the dataset large enough for transfer learning?

What it does
  1. Splits data/project/manifest.csv by release (70/15/15, seed 42, stratified by album), so every image
     of a release lands in the same split. Saves the split to data/project/split.csv for later reuse.
  2. Extracts features once with a frozen Keras ResNet-50 pre-trained on ImageNet (images squashed to
     224x224, Keras ResNet-50 preprocessing). Cached in data/project/features_resnet50.npz, so reruns skip this step.
  3. Trains a logistic regression on those features with 25/50/75/100% of the training releases
     (3 repeats per size, class weights balanced), for both labels: album (10 classes) and part (4 classes).
  4. Reports VALIDATION macro F1 and accuracy only. The test split is never touched here.

Output
  data/project/pilot_results.csv            one row per label, fraction and repeat
  reports/figures/pilot_learning_curve.png  macro F1 vs training size, both labels
  console: mean results, plus album accuracy per packaging part at 100%

Needs:  uv add tensorflow scikit-learn
Run from anywhere:  python scripts/06_pilot_learning_curve.py   (a few minutes on Apple Silicon)
"""
import csv, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # project folder
DATA = ROOT / "data/project"
MANIFEST = DATA / "manifest.csv"
SPLIT_PATH = DATA / "split.csv"
FEATURES = DATA / "features_resnet50.npz"
RESULTS = DATA / "pilot_results.csv"
FIGURE = ROOT / "reports/figures/pilot_learning_curve.png"
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
REPEATS = 3
PARTS = ["front", "back", "medium", "booklet"]


# 1. Split by release, stratified by album -------------------------------------------------------
def load_rows():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_by_release(rows):
    releases = defaultdict(set)                      # album -> release IDs
    for r in rows:
        releases[r["album"]].add(r["release_id"])
    assign = {}
    for album in sorted(releases):
        rel = sorted(releases[album])
        random.Random(f"{SEED}-{album}").shuffle(rel)
        n_test = round(len(rel) * TEST_FRAC)
        n_val = round(len(rel) * VAL_FRAC)
        for i, rid in enumerate(rel):
            assign[rid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    with open(SPLIT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["release_id", "album", "split"])
        album_of = {r["release_id"]: r["album"] for r in rows}
        for rid in sorted(assign):
            w.writerow([rid, album_of[rid], assign[rid]])
    return assign


# 2. Frozen Keras ResNet-50 features, cached ------------------------------------------------------------
def extract_features(rows):
    if FEATURES.exists():
        cached = np.load(FEATURES, allow_pickle=True)
        if list(cached["files"]) == [r["file"] for r in rows]:
            print(f"Using cached features from {FEATURES.relative_to(ROOT)}")
            return cached["X"]

    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

    # Frozen ResNet-50 without its classifier head; global average pooling gives a 2048-d vector
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    model.trainable = False

    def load(path):
        img = Image.open(path).convert("RGB").resize((224, 224))   # squash, same as the training plan
        return np.asarray(img, dtype=np.float32)

    feats, batch = [], []
    print(f"Extracting features for {len(rows)} images ...")
    for i, r in enumerate(rows):
        batch.append(load(DATA / r["file"]))
        if len(batch) == 32 or i == len(rows) - 1:
            x = preprocess_input(np.stack(batch))                   # Keras ResNet50's own ImageNet preprocessing
            feats.append(model.predict(x, verbose=0))
            batch = []
            print(f"  {i + 1}/{len(rows)}", end="\r")
    X = np.concatenate(feats).astype(np.float32)
    np.savez_compressed(FEATURES, X=X, files=np.array([r["file"] for r in rows]))
    print()
    return X


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
    rows = load_rows()
    assign = split_by_release(rows)
    split = np.array([assign[r["release_id"]] for r in rows])
    X = extract_features(rows)
    labels = {"album": np.array([r["album"] for r in rows]),
              "part": np.array([r["type"] for r in rows])}
    release = np.array([r["release_id"] for r in rows])
    album = labels["album"]

    va = split == "val"
    train_rel = defaultdict(list)                    # album -> training release IDs
    for rid, s in assign.items():
        if s == "train":
            train_rel[album[release == rid][0]].append(rid)
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

    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
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
        print(f"  {p:<8} {hit[m].mean():.3f}  (n={int(m.sum())})")

    # Plot
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, pts in curves.items():
        n, f1, sd = zip(*pts)
        ax.errorbar(n, f1, yerr=sd, marker="o", capsize=3,
                    label="Album (10 classes)" if name == "album" else "Packaging part (4 classes)")
    ax.set_xlabel("Training images")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Pilot learning curve: frozen ResNet-50 + logistic regression")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=200)
    print(f"\nSaved {RESULTS.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
