"""
Experiment 2: album verifier. Given an image and a claimed album, does the image belong to that album?

  python scripts/09_verifier.py

What it does
  1. Perceptual hash (pHash) of every image, cached in data/phash.csv. Hash distance <= 2 counts as the same scan.
  2. Builds checks separately for validation and test albums. Only albums with at least 2 releases take part.
     Every image of those albums is a query and gets three checks:
       genuine          claimed album = its own album
       random impostor  another album from the same split, drawn per query with seed 42
       hard impostor    an album by the same artist in the same split (the most similar one if several),
                        otherwise the album in the same split whose mean raw ResNet-50 vector is most similar
     Impostor albums are drawn from the eligible albums (>= 2 releases) of the same split. They are fixed from
     the raw features, so both models are scored on exactly the same checks.
  3. Gallery for a check = claimed album's images, minus the query's own release, minus near-duplicates of the
     query. A query whose own gallery ends up empty is dropped with all three of its checks (counted in
     metrics.json), so the 1:1:1 balance holds.
  4. Album score = top-1 cosine similarity, or mean of the top-3, against the gallery. Chosen per model on
     validation AUC (all checks).
  5. Models
       baseline   raw ResNet-50 features
       trained    Keras projection: standardise -> Dense(2048 -> d) -> L2 norm, trained on the 302 TRAIN albums
                  as an album classifier (softmax head, balanced class weights). The head is discarded and the
                  L2-normalised embedding is used for similarity. d in 128, 256, 512 and the number of epochs are
                  chosen on validation AUC (early stopping, patience PATIENCE, best weights kept).
  6. Each model's threshold maximises balanced accuracy (genuine vs all impostors) on validation checks.
  7. Evaluates ONCE on test checks: AUC and accuracy at the validation threshold for genuine vs random,
     genuine vs hard, and all checks, plus accuracy by the query's part.

Nothing from validation or test albums is used for training. Validation albums set the scorer, d, epochs and
threshold. Test albums are only scored at the end.

Output (results/verifier/)
  tuning.csv          validation AUC per model, embedding size, epochs and scorer
  tuning_epochs.csv   validation AUC after every epoch of every projection run
  metrics.json        test numbers, test sizes, chosen settings and thresholds
  checks_test.csv     one row per test check: scores and decisions for both models
  roc_curves.png      test ROC, both models, random vs hard impostors
  projection.keras    trained embedding model (input 2048 raw features, output L2-normalised embedding)

Needs:  uv add imagehash
"""
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest_all.csv"
SPLIT = DATA / "split_albums.csv"
FEATURES = DATA / "features_resnet50.npz"
ALBUMS = DATA / "metadata" / "albums_all.csv"
PHASH = DATA / "phash.csv"
OUT = ROOT / "results" / "verifier"

SEED = 42
NEAR_DUP = 2                     # pHash Hamming distance at or below this = same scan
PARTS = ["front", "back", "medium", "booklet"]
CHECK_TYPES = ["genuine", "random", "hard"]
SCORERS = ["top1", "top3"]
EMB_SIZES = [128, 256, 512]
MAX_EPOCHS = 60
PATIENCE = 5                     # epochs without a better validation AUC before stopping
BATCH = 256
LR = 1e-3


# ---------------------------------------------------------------- data
def load_data():
    """rows: manifest rows with split and artist added. X: (n, 2048) raw ResNet-50 features, manifest order."""
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(SPLIT, newline="", encoding="utf-8") as f:
        split_of = {r["album"]: r["split"] for r in csv.DictReader(f)}
    with open(ALBUMS, newline="", encoding="utf-8") as f:
        artist_of = {r["folder"]: r["artist"] for r in csv.DictReader(f)}

    cached = np.load(FEATURES, allow_pickle=False)
    index = {(f, h): i for i, (f, h) in enumerate(zip(cached["files"], cached["sha256"]))}
    missing = [r["file"] for r in rows if (r["file"], r["sha256"]) not in index]
    if missing:
        raise SystemExit(f"{len(missing)} images have no cached feature. Run scripts/07_extract_features.py first.")
    X = cached["X"][[index[(r["file"], r["sha256"])] for r in rows]].astype(np.float32)
    for r in rows:
        r["split"] = split_of[r["album"]]
        r["artist"] = artist_of[r["album"]]
    return rows, X


def load_phashes(rows):
    """64-bit pHash per image as uint64, manifest order. Cached in data/phash.csv keyed by (file, sha256)."""
    import imagehash
    cache = {}
    if PHASH.exists():
        with open(PHASH, newline="", encoding="utf-8") as f:
            cache = {(r["file"], r["sha256"]): r["phash"] for r in csv.DictReader(f)}
    todo = [r for r in rows if (r["file"], r["sha256"]) not in cache]
    if todo:
        for r in tqdm(todo, desc="pHash", unit="img"):
            with Image.open(DATA / r["file"]) as img:
                cache[(r["file"], r["sha256"])] = str(imagehash.phash(img))   # 16 hex chars = 64 bits
        tmp = PHASH.with_name(PHASH.name + ".part")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "sha256", "phash"])
            for (file, sha), h in sorted(cache.items()):
                w.writerow([file, sha, h])
        tmp.replace(PHASH)
    return np.array([int(cache[(r["file"], r["sha256"])], 16) for r in rows], dtype=np.uint64)


def l2n(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)


# ---------------------------------------------------------------- checks
def build_checks(rows, X, ph, split):
    """Checks and galleries for one split. Returns (checks, galleries, info).

    checks: list of dicts (q, album, claimed, type, part). galleries: list of index arrays, same order.
    """
    rng = np.random.default_rng(SEED)            # own generator per split, so test checks never depend on val
    album_idx = defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] == split:
            album_idx[r["album"]].append(i)
    releases = {a: {rows[i]["release_id"] for i in idx} for a, idx in album_idx.items()}
    eligible = sorted(a for a in album_idx if len(releases[a]) >= 2)
    album_idx = {a: np.array(album_idx[a]) for a in eligible}

    # Hard impostor per album, from mean L2-normalised raw features (fixed for both models)
    Xn = l2n(X)
    means = l2n(np.stack([Xn[album_idx[a]].mean(axis=0) for a in eligible]))
    sim = means @ means.T
    np.fill_diagonal(sim, -np.inf)
    artist = {a: rows[album_idx[a][0]]["artist"] for a in eligible}
    hard_of, hard_src = {}, {}
    for k, a in enumerate(eligible):
        same = [j for j, b in enumerate(eligible) if b != a and artist[b] == artist[a]]
        pool = same if same else range(len(eligible))
        j = max(pool, key=lambda j: sim[k, j])
        hard_of[a], hard_src[a] = eligible[j], ("same_artist" if same else "most_similar")

    def gallery(q, claimed):
        g = album_idx[claimed]
        g = g[np.array([rows[j]["release_id"] != rows[q]["release_id"] for j in g])]
        if len(g):
            g = g[np.bitwise_count(ph[g] ^ ph[q]) > NEAR_DUP]      # drop near-duplicate scans of the query
        return g

    checks, galleries, dropped = [], [], 0
    for a in eligible:
        others = [b for b in eligible if b != a]
        for q in album_idx[a]:
            g_gen = gallery(q, a)
            rand = others[rng.integers(len(others))]               # drawn for every query, even if dropped,
            if len(g_gen) == 0:                                      # so the draws do not shift
                dropped += 1
                continue
            for t, claimed in [("genuine", a), ("random", rand), ("hard", hard_of[a])]:
                checks.append({"q": int(q), "album": a, "claimed": claimed, "type": t, "part": rows[q]["type"]})
                galleries.append(g_gen if t == "genuine" else gallery(q, claimed))
    empty = sum(len(g) == 0 for g in galleries)
    if empty:
        raise SystemExit(f"{empty} impostor checks in {split} have an empty gallery")
    info = {"albums": len(eligible), "queries": len(checks) // 3, "checks": len(checks),
            "dropped_queries_empty_gallery": dropped,
            "hard_same_artist_albums": sum(v == "same_artist" for v in hard_src.values()),
            "hard_most_similar_albums": sum(v == "most_similar" for v in hard_src.values())}
    return checks, galleries, info


# ---------------------------------------------------------------- scoring
def album_scores(E, checks, galleries):
    """{"top1": array, "top3": array} of album scores. E: L2-normalised embeddings, manifest order."""
    top1, top3 = np.empty(len(checks)), np.empty(len(checks))
    for k, (c, g) in enumerate(zip(checks, galleries)):
        s = E[g] @ E[c["q"]]
        top = np.sort(s)[::-1][:3]                                  # fewer than 3 gallery images: mean of what exists
        top1[k], top3[k] = top[0], top.mean()
    return {"top1": top1, "top3": top3}


def subset_masks(checks):
    t = np.array([c["type"] for c in checks])
    return {"random": np.isin(t, ["genuine", "random"]), "hard": np.isin(t, ["genuine", "hard"]),
            "all": np.ones(len(t), bool)}, (t == "genuine").astype(int)


def aucs(checks, s):
    masks, y = subset_masks(checks)
    return {k: roc_auc_score(y[m], s[m]) for k, m in masks.items()}


def best_threshold(y, s):
    """Threshold on s that maximises balanced accuracy. Accept (match) when score >= threshold."""
    fpr, tpr, thr = roc_curve(y, s)
    k = np.argmax((tpr + 1 - fpr) / 2)
    return float(thr[k]), float((tpr[k] + 1 - fpr[k]) / 2)


# ---------------------------------------------------------------- trained projection
def train_projection(d, X, rows, val_checks, val_gal, log, train_idx=None, scorers=SCORERS):
    """Train one projection of size d. Returns (embedding model, best val AUC, best epoch, best scorer).

    train_idx: row indices to train on (default: every train album image). The softmax head covers the albums
    in train_idx. scorers: scorers considered for early stopping (default: top1 and top3).
    """
    import keras
    from keras import layers
    keras.utils.set_random_seed(SEED)

    tr = np.flatnonzero([r["split"] == "train" for r in rows]) if train_idx is None else np.asarray(train_idx)
    albums = sorted({rows[i]["album"] for i in tr})
    cls = {a: k for k, a in enumerate(albums)}
    y_tr = np.array([cls[rows[i]["album"]] for i in tr])
    counts = np.bincount(y_tr, minlength=len(albums))
    class_weight = {k: len(y_tr) / (len(albums) * c) for k, c in enumerate(counts)}   # "balanced"

    norm = layers.Normalization(name="standardise")
    norm.adapt(X[tr])                                               # mean and variance from train albums only
    inp = keras.Input((X.shape[1],), name="resnet50_features")
    emb = layers.UnitNormalization(name="l2")(layers.Dense(d, name="embedding")(norm(inp)))
    head = layers.Dense(len(albums), activation="softmax", name="album_head")(emb)
    clf = keras.Model(inp, head)
    embedder = keras.Model(inp, emb, name=f"projection_{d}")
    clf.compile(optimizer=keras.optimizers.Adam(LR), loss="sparse_categorical_crossentropy")

    val_idx = np.flatnonzero([r["split"] == "val" for r in rows])
    E = np.zeros((len(rows), d), np.float32)                        # only validation rows are filled
    best = (-1.0, 0, None, None)                                     # (auc, epoch, scorer, weights)
    for epoch in range(1, MAX_EPOCHS + 1):
        h = clf.fit(X[tr], y_tr, batch_size=BATCH, epochs=1, class_weight=class_weight, shuffle=True, verbose=0)
        E[val_idx] = embedder.predict(X[val_idx], batch_size=1024, verbose=0)
        sc = album_scores(E, val_checks, val_gal)
        row = {"emb_size": d, "epoch": epoch, "train_loss": h.history["loss"][0]}
        for s in scorers:
            for k, v in aucs(val_checks, sc[s]).items():
                row[f"auc_{k}_{s}"] = v
        log.append(row)
        top = max(scorers, key=lambda s: row[f"auc_all_{s}"])
        if row[f"auc_all_{top}"] > best[0]:
            best = (row[f"auc_all_{top}"], epoch, top, embedder.get_weights())
        elif epoch - best[1] >= PATIENCE:
            break
        print(f"  d={d} epoch {epoch:>2}  loss {row['train_loss']:.3f}  val AUC "
              + "  ".join(f"{s} {row[f'auc_all_{s}']:.4f}" for s in scorers))
    embedder.set_weights(best[3])
    return embedder, best[0], best[1], best[2]


# ---------------------------------------------------------------- reporting
def evaluate(checks, s, thr):
    """AUC, accuracy, genuine acceptance and impostor rejection per subset, plus accuracy by query part."""
    masks, y = subset_masks(checks)
    dec = (s >= thr).astype(int)
    ok = dec == y
    out = {}
    for k, m in masks.items():
        out[k] = {"auc": roc_auc_score(y[m], s[m]), "accuracy": ok[m].mean(), "n_checks": int(m.sum()),
                  "genuine_accept_rate": dec[m & (y == 1)].mean(), "impostor_reject_rate": 1 - dec[m & (y == 0)].mean()}
    part = np.array([c["part"] for c in checks])
    t = np.array([c["type"] for c in checks])
    out["by_part"] = {p: {"accuracy_all": ok[part == p].mean(), "n_queries": int((part == p).sum() // 3),
                          **{f"accuracy_{ct}": ok[(part == p) & (t == ct)].mean() for ct in CHECK_TYPES}}
                      for p in PARTS}
    return out


def plot_roc(checks, scores, path):
    masks, y = subset_masks(checks)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    colors = {"baseline": "#4C72B0", "trained": "#DD8452"}
    for model, s in scores.items():
        for sub, ls in [("random", "-"), ("hard", "--")]:
            m = masks[sub]
            fpr, tpr, _ = roc_curve(y[m], s[m])
            ax.plot(fpr, tpr, ls, color=colors[model], lw=1.6,
                    label=f"{model}, {sub} impostor (AUC {roc_auc_score(y[m], s[m]):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set_xlabel("False accept rate (impostor accepted)")
    ax.set_ylabel("True accept rate (genuine accepted)")
    ax.set_title("Album verifier, test albums")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in cols})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, X = load_data()
    ph = load_phashes(rows)

    val_checks, val_gal, val_info = build_checks(rows, X, ph, "val")
    test_checks, test_gal, test_info = build_checks(rows, X, ph, "test")
    print(f"val : {val_info}\ntest: {test_info}")

    # ---- tuning on validation checks
    tuning = []
    Xn = l2n(X)
    base_val = album_scores(Xn, val_checks, val_gal)
    for s in SCORERS:
        tuning.append({"model": "baseline", "emb_size": "-", "epochs": "-", "scorer": s,
                       **{f"val_auc_{k}": v for k, v in aucs(val_checks, base_val[s]).items()}})
    base_scorer = max(SCORERS, key=lambda s: aucs(val_checks, base_val[s])["all"])

    epoch_log, runs = [], {}
    for d in EMB_SIZES:
        print(f"\nProjection d={d}")
        embedder, auc, ep, scorer = train_projection(d, X, rows, val_checks, val_gal, epoch_log)
        runs[d] = (embedder, auc, ep, scorer)
        best_row = next(r for r in epoch_log if r["emb_size"] == d and r["epoch"] == ep)
        for s in SCORERS:
            tuning.append({"model": "trained", "emb_size": d, "epochs": ep, "scorer": s,
                           **{f"val_auc_{k}": best_row[f"auc_{k}_{s}"] for k in ["all", "random", "hard"]}})
    best_d = max(EMB_SIZES, key=lambda d: runs[d][1])
    embedder, _, best_ep, trained_scorer = runs[best_d]

    for r in tuning:
        r["chosen"] = ((r["model"] == "baseline" and r["scorer"] == base_scorer) or
                       (r["model"] == "trained" and r["emb_size"] == best_d and r["scorer"] == trained_scorer))
    cols = ["model", "emb_size", "epochs", "scorer", "val_auc_all", "val_auc_random", "val_auc_hard", "chosen"]
    write_csv(OUT / "tuning.csv", tuning, cols)
    ecols = ["emb_size", "epoch", "train_loss"] + [f"auc_{k}_{s}" for s in SCORERS for k in ["all", "random", "hard"]]
    write_csv(OUT / "tuning_epochs.csv", epoch_log, ecols)

    print(f"\n{'model':<9}{'d':>5}{'epochs':>8}{'scorer':>8}{'AUC all':>9}{'random':>8}{'hard':>8}")
    for r in tuning:
        print(f"{r['model']:<9}{r['emb_size']:>5}{r['epochs']:>8}{r['scorer']:>8}{r['val_auc_all']:>9.4f}"
              f"{r['val_auc_random']:>8.4f}{r['val_auc_hard']:>8.4f}" + (" *" if r["chosen"] else ""))
    print("* = chosen on validation AUC (all checks)")

    # ---- thresholds on validation
    E_all = embedder.predict(X, batch_size=1024, verbose=0)          # embeddings are L2-normalised by the model
    trained_val = album_scores(E_all, val_checks, val_gal)
    _, y_val = subset_masks(val_checks)
    thr = {"baseline": best_threshold(y_val, base_val[base_scorer]),
           "trained": best_threshold(y_val, trained_val[trained_scorer])}

    # ---- test, once
    test_scores = {"baseline": album_scores(Xn, test_checks, test_gal)[base_scorer],
                   "trained": album_scores(E_all, test_checks, test_gal)[trained_scorer]}
    results = {m: evaluate(test_checks, s, thr[m][0]) for m, s in test_scores.items()}

    metrics = {
        "settings": {
            "baseline": {"features": "raw ResNet-50 (2048)", "scorer": base_scorer,
                         "threshold": thr["baseline"][0], "val_balanced_accuracy": thr["baseline"][1]},
            "trained": {"emb_size": best_d, "epochs": best_ep, "scorer": trained_scorer, "lr": LR, "batch": BATCH,
                        "class_weight": "balanced", "patience": PATIENCE, "threshold": thr["trained"][0],
                        "val_balanced_accuracy": thr["trained"][1]},
            "near_duplicate_phash_distance": NEAR_DUP, "seed": SEED,
        },
        "val_checks": val_info,
        "test_checks": test_info,
        "test": results,
    }
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    with open(OUT / "checks_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_file", "query_album", "claimed_album", "check_type", "query_part", "gallery_size",
                    "score_baseline", "score_trained", "match_baseline", "match_trained"])
        for k, (c, g) in enumerate(zip(test_checks, test_gal)):
            sb, st = test_scores["baseline"][k], test_scores["trained"][k]
            w.writerow([rows[c["q"]]["file"], c["album"], c["claimed"], c["type"], c["part"], len(g),
                        f"{sb:.4f}", f"{st:.4f}", int(sb >= thr["baseline"][0]), int(st >= thr["trained"][0])])

    plot_roc(test_checks, test_scores, OUT / "roc_curves.png")
    embedder.save(OUT / "projection.keras")

    # ---- console summary
    print(f"\nTest: {test_info['albums']} albums, {test_info['queries']} queries, {test_info['checks']} checks")
    for m in ["baseline", "trained"]:
        r = results[m]
        print(f"\n{m} (scorer {metrics['settings'][m]['scorer']}, threshold {thr[m][0]:.4f})")
        print(f"  {'subset':<8}{'AUC':>8}{'acc':>8}{'gen acc':>9}{'imp rej':>9}{'n':>7}")
        for k in ["random", "hard", "all"]:
            v = r[k]
            print(f"  {k:<8}{v['auc']:>8.4f}{v['accuracy']:>8.3f}{v['genuine_accept_rate']:>9.3f}"
                  f"{v['impostor_reject_rate']:>9.3f}{v['n_checks']:>7}")
        print(f"  {'part':<8}{'acc all':>8}{'genuine':>9}{'random':>8}{'hard':>8}{'queries':>9}")
        for p in PARTS:
            v = r["by_part"][p]
            print(f"  {p:<8}{v['accuracy_all']:>8.3f}{v['accuracy_genuine']:>9.3f}{v['accuracy_random']:>8.3f}"
                  f"{v['accuracy_hard']:>8.3f}{v['n_queries']:>9}")
    print(f"\nSaved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
