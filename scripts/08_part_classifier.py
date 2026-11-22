"""
Experiment 1: part classifier (front, back, medium, booklet) on albums never seen in training.

  python scripts/08_part_classifier.py

What it does
  1. Loads the frozen ResNet-50 features (data/features_resnet50.npz) for every image in data/manifest_all.csv
     and the album-level split (data/split_albums.csv).
  2. Tunes a logistic regression step by step on VALIDATION albums only. A step is kept only if validation
     macro F1 goes up.
       Step 0  no regularization, no class weights
       Step 1  add class_weight="balanced"
       Step 2  add L2 regularization, C in C_GRID (0.000001 to 10)
       Step 3  class_weight="balanced" with C in STEP3_C (0.0001, 0.001)
     Features are standardised with a scaler fitted on training albums only.
  3. Refits the chosen settings on training albums and evaluates ONCE on test albums.

Note on "no regularization": scikit-learn 1.8 deprecated penalty=None. C=np.inf is its replacement and fits the
same unpenalised model (C is the inverse of the penalty strength, so an infinite C means zero penalty).

Output (results/part_classifier/)
  tuning.csv             one row per tried setting: validation accuracy, macro F1, F1 per part
  metrics.json           test numbers, test set size and the chosen settings
  predictions.csv        one row per test image: true part, predicted part, probability per part
  confusion_matrix.png   test confusion matrix, counts and row percentages
  model.joblib           the fitted scaler + logistic regression (one sklearn Pipeline)

Reuse (experiment 3): load_data() and train_part_classifier() take any subset of training rows and the
settings in metrics.json. The file name starts with a digit, so import it with importlib:
  mod = importlib.import_module("08_part_classifier")   # with scripts/ on sys.path
"""
import csv, json, warnings
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest_all.csv"
SPLIT = DATA / "split_albums.csv"
FEATURES = DATA / "features_resnet50.npz"
OUT = ROOT / "results" / "part_classifier"

PARTS = ["front", "back", "medium", "booklet"]   # fixed class order for every table and matrix
MAX_ITER = 5000                                  # high enough that lbfgs should converge, warnings are recorded
# The first run used [0.01, 0.1, 1, 10] and picked 0.01, the edge of the grid, with validation F1 still rising
# as C fell. The grid was extended downwards on that validation evidence. The test set had already been scored
# once with C=0.01, so this is the second test evaluation. metrics.json records both.
# The second run picked 0.0001, again the edge, so the grid went down to 0.000001 for a third run.
C_GRID = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# Step 3: balanced class weights retried with strong L2. Step 1 only tried them without regularization, and the
# second run showed strong L2 pushing booklets towards front, which class weights might counter.
STEP3_C = [0.0001, 0.001]
# Earlier test evaluations, copied from each run's metrics.json, kept for honest reporting.
TEST_HISTORY = [
    {"run": 1, "C_grid": [0.01, 0.1, 1, 10], "chosen_C": 0.01, "class_weight": "none",
     "test_accuracy": 0.833, "test_macro_f1": 0.813},
    {"run": 2, "C_grid": [0.0001, 0.001, 0.01, 0.1, 1, 10], "chosen_C": 0.0001, "class_weight": "none",
     "test_accuracy": 0.858, "test_macro_f1": 0.833},
]


# ---------------------------------------------------------------- data
def load_data():
    """Return (rows, X, y). rows: manifest rows with an added "split" key. X: (n, 2048) features. y: part names.

    Features are looked up by (file, sha256), the cache key shared with 05 and 07.
    """
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(SPLIT, newline="", encoding="utf-8") as f:
        split_of = {r["album"]: r["split"] for r in csv.DictReader(f)}

    cached = np.load(FEATURES, allow_pickle=False)
    index = {(f, h): i for i, (f, h) in enumerate(zip(cached["files"], cached["sha256"]))}
    missing = [r["file"] for r in rows if (r["file"], r["sha256"]) not in index]
    if missing:
        raise SystemExit(f"{len(missing)} images have no cached feature. Run scripts/07_extract_features.py first.")
    unsplit = sorted({r["album"] for r in rows} - split_of.keys())
    if unsplit:
        raise SystemExit(f"{len(unsplit)} albums are missing from {SPLIT.name}, e.g. {unsplit[:3]}")

    X = cached["X"][[index[(r["file"], r["sha256"])] for r in rows]].astype(np.float32)
    for r in rows:
        r["split"] = split_of[r["album"]]
    y = np.array([r["type"] for r in rows])
    return rows, X, y


# ---------------------------------------------------------------- model
def train_part_classifier(X, y, C=np.inf, class_weight=None, max_iter=MAX_ITER):
    """Fit scaler + logistic regression on (X, y). Returns (pipeline, convergence warning messages).

    C=np.inf means no regularization. The scaler is fitted on X only, so pass training rows only.
    """
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X, y)
    msgs = [str(w.message).splitlines()[0] for w in caught if issubclass(w.category, ConvergenceWarning)]
    return pipe, msgs


def scores(y_true, y_pred):
    """Accuracy, macro F1 and F1 per part, in PARTS order."""
    per_part = f1_score(y_true, y_pred, labels=PARTS, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=PARTS, average="macro", zero_division=0),
        **{f"f1_{p}": v for p, v in zip(PARTS, per_part)},
    }


def describe(C, class_weight):
    reg = "none" if np.isinf(C) else f"L2 C={C:g}"
    return f"reg={reg}, class_weight={class_weight or 'none'}"


# ---------------------------------------------------------------- tuning (validation only)
def tune(X_tr, y_tr, X_va, y_va):
    """Step-by-step search on validation albums. Returns (tuning rows, chosen settings dict)."""
    log = []

    def trial(step, C, cw):
        pipe, warn = train_part_classifier(X_tr, y_tr, C=C, class_weight=cw)
        s = scores(y_va, pipe.predict(X_va))
        log.append({"step": step, "settings": describe(C, cw), "C": C, "class_weight": cw or "none",
                    "converged": not warn, **s})
        return s["macro_f1"]

    # Step 0: unpenalised, unweighted
    best_C, best_cw = np.inf, None
    best = trial(0, best_C, best_cw)

    # Step 1: balanced class weights
    f1 = trial(1, np.inf, "balanced")
    if f1 > best:
        best, best_cw = f1, "balanced"

    # Step 2: L2 strength, on top of whatever step 1 kept
    step2 = [(trial(2, C, best_cw), C) for C in C_GRID]
    f1, C = max(step2)
    if f1 > best:
        best, best_C = f1, C

    # Step 3: balanced class weights with strong L2, kept only if it beats everything so far
    step3 = [(trial(3, C, "balanced"), C) for C in STEP3_C]
    f1, C = max(step3)
    if f1 > best:
        best, best_C, best_cw = f1, C, "balanced"

    for r in log:
        r["kept"] = (r["C"] == best_C or (np.isinf(r["C"]) and np.isinf(best_C))) and r["class_weight"] == (best_cw or "none")
    return log, {"C": best_C, "class_weight": best_cw, "max_iter": MAX_ITER, "val_macro_f1": best}


def print_tuning(log):
    head = f"{'step':>4}  {'settings':<38}{'conv':>5}{'acc':>7}{'macroF1':>9}" + "".join(f"{p:>9}" for p in PARTS)
    print(head)
    for r in log:
        mark = " *" if r["kept"] else ""
        print(f"{r['step']:>4}  {r['settings']:<38}{'yes' if r['converged'] else 'NO':>5}{r['accuracy']:>7.3f}"
              f"{r['macro_f1']:>9.3f}" + "".join(f"{r['f1_' + p]:>9.3f}" for p in PARTS) + mark)
    print("* = chosen settings (validation macro F1)")


def save_tuning(log):
    cols = ["step", "settings", "C", "class_weight", "converged", "accuracy", "macro_f1",
            *[f"f1_{p}" for p in PARTS], "kept"]
    with open(OUT / "tuning.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in log:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) and not np.isinf(r[k]) else r[k]) for k in cols})


# ---------------------------------------------------------------- test outputs
def plot_confusion(cm, path):
    """Counts in each cell, row percentage underneath. Colour follows the row percentage."""
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(PARTS)):
        for j in range(len(PARTS)):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(PARTS)), PARTS)
    ax.set_yticks(range(len(PARTS)), PARTS)
    ax.set_xlabel("Predicted part")
    ax.set_ylabel("True part (uploader tag)")
    ax.set_title("Part classifier, test albums")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, X, y = load_data()
    split = np.array([r["split"] for r in rows])
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    for name, m in [("train", tr), ("val", va), ("test", te)]:
        print(f"{name:<5} {m.sum():>6} images, {len({rows[i]['album'] for i in np.flatnonzero(m)}):>3} albums")

    # ---- tuning on validation albums
    print("\nTuning on validation albums")
    log, chosen = tune(X[tr], y[tr], X[va], y[va])
    print_tuning(log)
    save_tuning(log)

    # ---- refit with chosen settings on training albums, test once
    pipe, warn = train_part_classifier(X[tr], y[tr], C=chosen["C"], class_weight=chosen["class_weight"])
    proba = pipe.predict_proba(X[te])
    classes = list(pipe.classes_)
    pred = np.array(classes)[proba.argmax(axis=1)]
    y_te = y[te]

    prec, rec, f1, support = precision_recall_fscore_support(y_te, pred, labels=PARTS, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=PARTS)
    test_rows = [rows[i] for i in np.flatnonzero(te)]
    metrics = {
        "settings": {"C": "inf (no regularization)" if np.isinf(chosen["C"]) else chosen["C"],
                     "class_weight": chosen["class_weight"] or "none", "max_iter": MAX_ITER,
                     "scaler": "StandardScaler fitted on training albums", "converged": not warn},
        "val_macro_f1": chosen["val_macro_f1"],
        "C_grid": C_GRID,
        "test_evaluation_number": len(TEST_HISTORY) + 1,
        "earlier_test_evaluations": TEST_HISTORY,
        "test_size": {"images": int(te.sum()), "albums": len({r["album"] for r in test_rows}),
                      "releases": len({r["release_id"] for r in test_rows})},
        "train_size": {"images": int(tr.sum()), "albums": len({rows[i]["album"] for i in np.flatnonzero(tr)})},
        "accuracy": accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, labels=PARTS, average="macro", zero_division=0),
        "per_class": {p: {"precision": prec[k], "recall": rec[k], "f1": f1[k], "support": int(support[k])}
                      for k, p in enumerate(PARTS)},
        "confusion_matrix": {"labels": PARTS, "rows_true_cols_pred": cm.tolist()},
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    with open(OUT / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "album", "release_id", "true_part", "pred_part", *[f"p_{p}" for p in PARTS]])
        for r, pr, p in zip(test_rows, pred, proba):
            w.writerow([r["file"], r["album"], r["release_id"], r["type"], pr,
                        *[f"{p[classes.index(c)]:.4f}" for c in PARTS]])

    plot_confusion(cm, OUT / "confusion_matrix.png")
    joblib.dump(pipe, OUT / "model.joblib")

    # ---- console summary
    print(f"\nTest albums: {metrics['test_size']['images']} images, {metrics['test_size']['albums']} albums, "
          f"{metrics['test_size']['releases']} releases")
    print(f"Settings: {describe(chosen['C'], chosen['class_weight'])}, converged: {not warn}")
    print(f"Accuracy {metrics['accuracy']:.3f}   macro F1 {metrics['macro_f1']:.3f}")
    print(f"{'part':<9}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>6}")
    for p in PARTS:
        c = metrics["per_class"][p]
        print(f"{p:<9}{c['precision']:>7.3f}{c['recall']:>8.3f}{c['f1']:>7.3f}{c['support']:>6}")
    print("\nConfusion matrix (rows true, cols predicted):", PARTS)
    for p, line in zip(PARTS, cm):
        print(f"{p:<9}" + "".join(f"{v:>7}" for v in line))
    print(f"\nSaved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
