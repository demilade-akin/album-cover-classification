"""
Experiment 3b: breadth vs depth for the album verifier. Same question as experiment 3, asked of the second check:
for a verifier that must work on unseen albums, do more albums beat more editions of a few albums?

  python scripts/10b_breadth_vs_depth_verifier.py

Training sets (train albums only), identical to scripts/10_breadth_vs_depth.py
  A  depth       every image of the 10 albums in albums_10.csv (2,223 images)
  B  breadth     the same number of images as A, spread over all 302 train albums. Drawn by breadth_sample()
                 from experiment 3 with seeds 42 to 46, so each seed picks the identical images.
  C  everything  every train album image (16,590)

Model, fixed for every run (training code imported from scripts/09_verifier.py)
  standardise (fitted on the run's training images) -> Dense(2048 -> 512) -> L2 norm, trained as a softmax
  classifier over the run's training albums with balanced class weights, Adam lr 1e-3, batch 256. Network
  initialisation uses seed 42 in every run, so within B only the sampled images change. Early stopping on
  validation AUC (all checks, top-1 scorer), patience 5, best weights kept. d=512 and top-1 are the settings
  experiment 2 chose on validation. They are not retuned per set.

Checks
  Validation and test checks and galleries come from build_checks() in experiment 2 and are the same for every
  run: genuine, random impostor, hard impostor. Gallery excludes the query's own release and pHash near-duplicates.

Evaluation
  Threshold per run maximises balanced accuracy on validation checks. Test checks are scored once per run.
  C runs first. If its test AUC is more than SANITY_TOL from results/verifier/metrics.json, the script stops
  before training A and B.

Output (results/breadth_depth_verifier/)
  results.csv                     one row per run
  summary.csv                     A, B (mean and sample sd over seeds), C
  epochs.csv                      validation AUC after every epoch of every run
  breadth_vs_depth_verifier.png   grouped bars of test AUC and balanced accuracy, B with error bars
  metrics.json                    settings, check counts, test scoring count, sanity check
"""
import csv, importlib, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
vf = importlib.import_module("09_verifier")              # data, checks, training, scoring from experiment 2
bd = importlib.import_module("10_breadth_vs_depth")      # training sets from experiment 3

OUT = ROOT / "results" / "breadth_depth_verifier"
VERIFIER_METRICS = ROOT / "results" / "verifier" / "metrics.json"
EMB = 512
SCORER = "top1"
SANITY_TOL = 0.005
PARTS = vf.PARTS
METRICS = ["auc", "auc_random", "auc_hard", "balanced_accuracy", "accuracy", "genuine_accept",
           "random_reject", "hard_reject"] + [f"genuine_accept_{p}" for p in PARTS]


def run(name, seed, train_idx, X, rows, val, test, epoch_log, counter):
    """Train one projection on train_idx, pick the threshold on validation, score test once."""
    val_checks, val_gal = val
    test_checks, test_gal = test
    print(f"\n{name}{'' if seed is None else f' seed {seed}'}: {len(train_idx)} images")
    log = []
    embedder, val_auc, best_ep, _ = vf.train_projection(EMB, X, rows, val_checks, val_gal, log,
                                                        train_idx=train_idx, scorers=[SCORER])
    for r in log:
        epoch_log.append({"set": name, "seed": seed if seed is not None else "-", **r})

    E = embedder.predict(X, batch_size=1024, verbose=0)
    _, y_val = vf.subset_masks(val_checks)
    thr, val_bacc = vf.best_threshold(y_val, vf.album_scores(E, val_checks, val_gal)[SCORER])

    s = vf.album_scores(E, test_checks, test_gal)[SCORER]   # the only test scoring for this run
    counter[name] = counter.get(name, 0) + 1
    res = vf.evaluate(test_checks, s, thr)
    a = res["all"]
    out = {"set": name, "seed": seed if seed is not None else "-", "n_images": len(train_idx),
           "n_albums": len({rows[i]["album"] for i in train_idx}), "best_epoch": best_ep, "epochs_run": len(log),
           "val_auc": val_auc, "threshold": thr, "val_balanced_accuracy": val_bacc,
           "test_auc": a["auc"], "test_auc_random": res["random"]["auc"], "test_auc_hard": res["hard"]["auc"],
           "test_balanced_accuracy": (a["genuine_accept_rate"] + a["impostor_reject_rate"]) / 2,
           "test_accuracy": a["accuracy"], "test_genuine_accept": a["genuine_accept_rate"],
           "test_random_reject": res["random"]["impostor_reject_rate"],
           "test_hard_reject": res["hard"]["impostor_reject_rate"],
           **{f"test_genuine_accept_{p}": res["by_part"][p]["accuracy_genuine"] for p in PARTS}}
    print(f"  best epoch {best_ep} of {len(log)}  val AUC {val_auc:.4f}  threshold {thr:.4f}  "
          f"test AUC {out['test_auc']:.4f}  bal acc {out['test_balanced_accuracy']:.3f}")
    return out


def plot(summary, baseline_auc, path):
    groups = ["auc", "auc_random", "auc_hard", "balanced_accuracy"]
    labels = ["AUC, all checks", "AUC, random impostor", "AUC, hard impostor", "balanced accuracy"]
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
    ax.axhline(baseline_auc, color="grey", ls="--", lw=1)
    ax.text(len(groups) - 0.5, baseline_auc, f"untrained baseline AUC {baseline_auc:.3f}", ha="right",
            va="bottom", fontsize=7, color="grey")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Test score")
    ax.set_ylim(0.5, 1)                                   # 0.5 = chance for AUC and balanced accuracy
    ax.set_title("Breadth vs depth: album verifier on test albums")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: bd.fmt(r[k]) for k in cols})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, X = vf.load_data()
    ph = vf.load_phashes(rows)
    val = vf.build_checks(rows, X, ph, "val")
    test = vf.build_checks(rows, X, ph, "test")
    print(f"val : {val[2]}\ntest: {test[2]}")

    # ---- training sets, built exactly as in scripts/10_breadth_vs_depth.py (same manifest order)
    split = np.array([r["split"] for r in rows])
    tr = np.flatnonzero(split == "train")
    with open(bd.ALBUMS_10, newline="", encoding="utf-8") as f:
        ten = {r["folder"] for r in csv.DictReader(f)}
    a_idx = np.array([i for i in tr if rows[i]["album"] in ten])
    if len({rows[i]["album"] for i in a_idx}) != 10:
        raise SystemExit("Not all 10 albums of albums_10.csv are in the train split")
    idx_by_album = {}
    for i in tr:
        idx_by_album.setdefault(rows[i]["album"], []).append(i)

    with open(VERIFIER_METRICS, encoding="utf-8") as f:
        ref = json.load(f)
    ref_auc = ref["test"]["trained"]["all"]["auc"]
    ref_epoch = ref["settings"]["trained"]["epochs"]
    baseline_auc = ref["test"]["baseline"]["all"]["auc"]

    epoch_log, counter = [], {}
    args = (X, rows, val[:2], test[:2], epoch_log, counter)

    # ---- C first: sanity check against experiment 2
    results = [run("C", None, tr, *args)]
    c = results[0]
    sanity = {"reference_test_auc": ref_auc, "reference_best_epoch": ref_epoch, "set_c_test_auc": c["test_auc"],
              "set_c_best_epoch": c["best_epoch"], "difference": c["test_auc"] - ref_auc, "tolerance": SANITY_TOL,
              "passed": abs(c["test_auc"] - ref_auc) <= SANITY_TOL}
    print(f"\nSanity: C test AUC {c['test_auc']:.4f} vs experiment 2 {ref_auc:.4f} "
          f"(epoch {c['best_epoch']} vs {ref_epoch}): {'ok' if sanity['passed'] else 'FAILED'}")
    if not sanity["passed"]:
        with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"sanity_check": sanity, "test_scorings": counter, "stopped": True}, f, indent=2, default=float)
        raise SystemExit("Set C does not reproduce experiment 2. Stopping before A and B.")

    results.insert(0, run("A", None, a_idx, *args))
    for seed in bd.B_SEEDS:
        results.insert(-1, run("B", seed, bd.breadth_sample(idx_by_album, len(a_idx), seed), *args))

    cols = list(results[0].keys())
    write_csv(OUT / "results.csv", results, cols)
    write_csv(OUT / "epochs.csv", epoch_log, ["set", "seed", "emb_size", "epoch", "train_loss",
                                              "auc_all_top1", "auc_random_top1", "auc_hard_top1"])

    # ---- summary: mean and sample sd (ddof=1) over runs
    summary = {}
    for s in ["A", "B", "C"]:
        rs = [r for r in results if r["set"] == s]
        row = {"set": s, "runs": len(rs), "n_images": rs[0]["n_images"], "n_albums": rs[0]["n_albums"]}
        for m in ["best_epoch", "threshold"] + [f"test_{m}" for m in METRICS]:
            v = np.array([r[m] for r in rs], dtype=float)
            row[m] = float(v.mean())
            row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        summary[s] = row
    scols = ["set", "runs", "n_images", "n_albums"] + [c for m in ["best_epoch", "threshold"] +
                                                       [f"test_{m}" for m in METRICS] for c in (m, f"{m}_sd")]
    write_csv(OUT / "summary.csv", summary.values(), scols)
    plot(summary, baseline_auc, OUT / "breadth_vs_depth_verifier.png")

    metrics = {
        "settings": {"emb_size": EMB, "scorer": SCORER, "lr": vf.LR, "batch": vf.BATCH, "patience": vf.PATIENCE,
                     "max_epochs": vf.MAX_EPOCHS, "class_weight": "balanced", "init_seed": vf.SEED,
                     "b_seeds": bd.B_SEEDS, "near_duplicate_phash_distance": vf.NEAR_DUP,
                     "threshold_rule": "max balanced accuracy on validation checks"},
        "val_checks": val[2], "test_checks": test[2],
        "test_scorings": {**counter, "total": sum(counter.values()), "runs": len(results)},
        "sanity_check": sanity,
        "untrained_baseline_test_auc": baseline_auc,
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"\nB: mean ± sd over {len(bd.B_SEEDS)} seeds. Test scored {sum(counter.values())} times "
          f"({len(results)} runs).")
    show = ["best_epoch", "test_auc", "test_auc_hard", "test_balanced_accuracy", "test_genuine_accept"]
    print(f"{'set':<4}{'images':>7}{'albums':>7}" + "".join(f"{m.replace('test_', ''):>20}" for m in show))
    for s in summary.values():
        cells = "".join(f"{s[m]:>11.3f} ± {s[f'{m}_sd']:.3f}" if s["set"] == "B" else f"{s[m]:>20.3f}"
                        for m in show)
        print(f"{s['set']:<4}{s['n_images']:>7}{s['n_albums']:>7}{cells}")
    print(f"\nSaved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
