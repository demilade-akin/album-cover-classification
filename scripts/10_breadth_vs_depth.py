"""
Experiment 3: breadth vs depth. For a part classifier that must work on unseen albums, do more albums beat more
editions of a few albums?

  python scripts/10_breadth_vs_depth.py

Training sets (train albums only)
  A  depth       every image of the 10 albums in albums_10.csv (2,223 images)
  B  breadth     the same number of images as A, spread over all 302 train albums: round-robin over albums
                 (album order shuffled), a random unused image of the album each turn. Seeds 42 to 46.
  C  everything  every train album image (16,590)
A vs B holds the image count fixed and changes only how the images are spread. C shows what extra data adds.

Model and tuning
  Same model as experiment 1 (scripts/08_part_classifier.py): standardised features + logistic regression, no
  class weights. The scaler is fitted on each training set. C is tuned separately for every run on validation
  macro F1, because the best C can change with training set size.

Evaluation
  Every run is scored once on the same test albums: accuracy, macro F1, F1 per part. B is summarised as mean and
  standard deviation over the 5 seeds.

Output (results/breadth_depth/)
  results.csv            one row per run: training set size, chosen C, validation and test scores
  summary.csv            A, B (mean and sd), C
  breadth_vs_depth.png   grouped bars of test macro F1 and F1 per part, B with error bars (sd over seeds)
"""
import csv, importlib, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pc = importlib.import_module("08_part_classifier")       # reuse data loading, training and scoring from exp 1

ALBUMS_10 = ROOT / "data" / "metadata" / "albums_10.csv"
OUT = ROOT / "results" / "breadth_depth"
C_GRID = [0.00001, 0.0001, 0.001, 0.01, 0.1]
B_SEEDS = [42, 43, 44, 45, 46]
METRICS = ["accuracy", "macro_f1"] + [f"f1_{p}" for p in pc.PARTS]


def breadth_sample(idx_by_album, n, seed):
    """n image indices spread as evenly as possible over albums: round-robin, random image within each album."""
    rng = np.random.default_rng(seed)
    albums = sorted(idx_by_album)
    rng.shuffle(albums)                                   # random album order, so the last partial round is random
    queues = {a: list(rng.permutation(idx_by_album[a])) for a in albums}
    picked = []
    while len(picked) < n:
        for a in albums:
            if queues[a]:                                 # small albums run out, the rest keep going
                picked.append(queues[a].pop())
                if len(picked) == n:
                    break
    return np.array(sorted(picked))


def run(name, seed, train_idx, X, y, rows, va, te):
    """Tune C on validation, refit with the best C, score once on test."""
    val = []
    for C in C_GRID:
        pipe, warn = pc.train_part_classifier(X[train_idx], y[train_idx], C=C)
        val.append((pc.scores(y[va], pipe.predict(X[va]))["macro_f1"], C, pipe, warn))
    best_f1, best_C, pipe, warn = max(val, key=lambda v: v[0])
    test = pc.scores(y[te], pipe.predict(X[te]))
    out = {"set": name, "seed": seed if seed is not None else "-", "n_images": len(train_idx),
           "n_albums": len({rows[i]["album"] for i in train_idx}), "chosen_C": best_C,
           "converged": not warn, "val_macro_f1": best_f1,
           **{f"val_macro_f1_C={C:g}": f1 for f1, C, _, _ in val}, **{f"test_{m}": test[m] for m in METRICS}}
    print(f"{name}{'' if seed is None else f' seed {seed}':<9} n={out['n_images']:>5} albums={out['n_albums']:>3} "
          f"C={best_C:<7g} val F1 {best_f1:.3f}  test acc {test['accuracy']:.3f}  test F1 {test['macro_f1']:.3f}")
    return out


def plot(summary, path):
    groups = ["macro_f1"] + [f"f1_{p}" for p in pc.PARTS]
    labels = ["macro F1"] + pc.PARTS
    sets = [("A", "A: 10 albums"), ("B", "B: same size, 302 albums"), ("C", "C: all 302 albums")]
    colors = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}
    x = np.arange(len(groups))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (s, lab) in enumerate(sets):
        r = summary[s]
        means = [r[f"test_{g}"] for g in groups]
        err = [r[f"test_{g}_sd"] for g in groups] if s == "B" else None
        ax.bar(x + (k - 1) * w, means, w, yerr=err, capsize=3, label=lab, color=colors[s])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Breadth vs depth: part classifier on test albums")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, X, y = pc.load_data()
    split = np.array([r["split"] for r in rows])
    tr, va, te = np.flatnonzero(split == "train"), split == "val", split == "test"

    with open(ALBUMS_10, newline="", encoding="utf-8") as f:
        ten = {r["folder"] for r in csv.DictReader(f)}
    a_idx = np.array([i for i in tr if rows[i]["album"] in ten])
    if len({rows[i]["album"] for i in a_idx}) != 10:
        raise SystemExit("Not all 10 albums of albums_10.csv are in the train split")
    idx_by_album = {}
    for i in tr:
        idx_by_album.setdefault(rows[i]["album"], []).append(i)

    results = [run("A", None, a_idx, X, y, rows, va, te)]
    for seed in B_SEEDS:
        results.append(run("B", seed, breadth_sample(idx_by_album, len(a_idx), seed), X, y, rows, va, te))
    results.append(run("C", None, tr, X, y, rows, va, te))

    cols = list(results[0].keys())
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: fmt(v) for k, v in r.items()})

    # ---- summary: A, B mean and sd over seeds (sample sd, ddof=1), C
    summary = {}
    for s in ["A", "B", "C"]:
        rs = [r for r in results if r["set"] == s]
        row = {"set": s, "runs": len(rs), "n_images": rs[0]["n_images"], "n_albums": rs[0]["n_albums"],
               "chosen_C": "/".join(sorted({f"{r['chosen_C']:g}" for r in rs}, key=float))}
        for m in METRICS:
            v = np.array([r[f"test_{m}"] for r in rs])
            row[f"test_{m}"] = float(v.mean())
            row[f"test_{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        summary[s] = row
    scols = ["set", "runs", "n_images", "n_albums", "chosen_C"] + [c for m in METRICS for c in (f"test_{m}", f"test_{m}_sd")]
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scols)
        w.writeheader()
        for s in summary.values():
            w.writerow({k: fmt(s[k]) for k in scols})
    plot(summary, OUT / "breadth_vs_depth.png")

    print(f"\nTest albums: {int(te.sum())} images. B: mean ± sd over {len(B_SEEDS)} seeds.")
    print(f"{'set':<4}{'images':>7}{'albums':>7}  {'C':<12}" + "".join(f"{m.replace('f1_', ''):>15}" for m in METRICS))
    for s in summary.values():
        cells = "".join(f"{s[f'test_{m}']:>8.3f} ± {s[f'test_{m}_sd']:.3f}" if s["set"] == "B"
                        else f"{s[f'test_{m}']:>15.3f}" for m in METRICS)
        print(f"{s['set']:<4}{s['n_images']:>7}{s['n_albums']:>7}  {s['chosen_C']:<12}{cells}")
    print(f"\nSaved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
