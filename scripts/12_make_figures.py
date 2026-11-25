"""
Report figures and tables, generated only from committed files in results/ and data/. No number is typed by hand.

  python scripts/12_make_figures.py             write everything and copy PDFs and .tex tables into the report
  python scripts/12_make_figures.py --no-copy   write to results/ only, leave reports/final_report/ untouched

Needs no images and no feature cache, so it runs on a fresh clone. Labels use the report's terms: "type" for the
packaging part and "disc" for MusicBrainz's "medium". One-column figures are drawn 3.5 in wide (the IEEE column),
so their fonts print at their set size.

Figures (results/figures/, each as .pdf and .png; the .pdf files are copied to reports/final_report/figures/)
  pipeline.pdf                     image -> frozen ResNet-50 -> 2,048 features -> type classifier / projection -> verifier
  exp1_tuning_C.pdf                type classifier: validation macro F1 against C (log scale)
  exp1_confusion_matrix.pdf        type classifier: test confusion matrix, counts and row percentages
  exp2_roc.pdf                     verifier: test ROC, baseline vs trained, random vs hard impostors
  exp2_accuracy_by_part.pdf        verifier: test accuracy by query type, baseline vs trained, per check type
  exp3_breadth_depth.pdf           breadth vs depth, type classifier: test macro F1 and F1 per type
  exp3_breadth_depth_verifier.pdf  breadth vs depth, verifier: test AUC per impostor kind, untrained baseline, A, B, C

Tables (results/tables/)
  CSV, one per result:
    dataset_split.csv          albums, releases, images and images per type for each split
    part_classifier_test.csv   test precision, recall, F1 and support per type, plus accuracy and macro F1
    verifier_test.csv          test AUC, accuracy, genuine accept and impostor reject rates per model and subset
    verifier_by_part.csv       test accuracy by query type per model and check type
    breadth_depth.csv          A, B (mean and sd over seeds), C
  LaTeX, one per report table (copied to reports/final_report/tables/):
    dataset_split.tex, exp1_type_metrics.tex, exp2_verifier_metrics.tex, exp2_verifier_by_type.tex,
    exp3_breadth_depth.tex, exp3_verifier.tex

TODO (audit, not built yet): add the audit figure (top flagged images) and the hand-check table once
scripts/11_audit.py exists.
"""
import csv, json, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
REPORT_FIG = ROOT / "reports" / "final_report" / "figures"
REPORT_TAB = ROOT / "reports" / "final_report" / "tables"
PARTS = ["front", "back", "medium", "booklet"]                 # names in the data (MusicBrainz terms)
LABEL = {"front": "front", "back": "back", "medium": "disc", "booklet": "booklet"}   # names in the report
FEATURE_DIM = 2048                          # ResNet-50 global average pooling output, fixed by the architecture
COLUMN_W = 3.5                              # IEEE column width in inches: one-column figures are drawn at this size

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"baseline": "#4C72B0", "trained": "#DD8452", "A": "#4C72B0", "B": "#DD8452", "C": "#55A868",
          "untrained": "#9E9E9E"}
COPY_TO_REPORT = "--no-copy" not in sys.argv


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if COPY_TO_REPORT:
        shutil.copy(FIG / f"{name}.pdf", REPORT_FIG / f"{name}.pdf")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------- pipeline diagram
def fig_pipeline(emb_size):
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#555555", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))

    box(1, 16, 11, 8, "Uploaded\nimage")
    box(15.5, 16, 17, 8, "Frozen ResNet-50\n(ImageNet)\n224×224 input", fc="#DDE6F2")
    box(36, 16, 10, 8, f"{FEATURE_DIM:,}\nfeatures")
    box(50.5, 26, 20, 10, "Type classifier\nstandardise +\nlogistic regression", fc="#F8E3D3")
    box(50.5, 3, 20, 10, f"Projection\nDense {FEATURE_DIM:,}→{emb_size}\n+ L2 normalise", fc="#F8E3D3")
    box(74.5, 26, 24.5, 10, "front / back /\ndisc / booklet\nvs uploader tag")
    box(74.5, 3, 24.5, 10, "Verifier: top-1 cosine\nvs claimed album's\ngallery → match / no match")
    arrow(12.8, 20, 14.8, 20)
    arrow(33.4, 20, 35.3, 20)
    arrow(46.8, 21.5, 49.8, 30)
    arrow(46.8, 18.5, 49.8, 8)
    arrow(71.4, 31, 73.8, 31)
    arrow(71.4, 8, 73.8, 8)
    save(fig, "pipeline")


# ---------------------------------------------------------------- experiment 1
def fig_tuning_C():
    rows = read_csv(RES / "part_classifier" / "tuning.csv")
    l2 = sorted([r for r in rows if r["step"] == "2"], key=lambda r: float(r["C"]))
    bal = sorted([r for r in rows if r["step"] == "3"], key=lambda r: float(r["C"]))
    none = next(r for r in rows if r["step"] == "0")
    chosen = next(r for r in rows if r["kept"] == "True")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot([float(r["C"]) for r in l2], [float(r["macro_f1"]) for r in l2], "o-", color=COLORS["baseline"],
            label="L2, no class weights")
    ax.plot([float(r["C"]) for r in bal], [float(r["macro_f1"]) for r in bal], "s--", color=COLORS["trained"],
            label="L2, balanced class weights")
    ax.axhline(float(none["macro_f1"]), color="grey", ls=":", lw=1, label="no regularization")
    ax.plot(float(chosen["C"]), float(chosen["macro_f1"]), "o", ms=11, mfc="none", mec="black", label="chosen")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Validation macro F1")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    save(fig, "exp1_tuning_C")


def fig_confusion(metrics):
    cm = np.array(metrics["confusion_matrix"]["rows_true_cols_pred"])
    labels = [LABEL[p] for p in metrics["confusion_matrix"]["labels"]]
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("True type (uploader tag)")
    ax.spines[:].set_visible(False)
    save(fig, "exp1_confusion_matrix")


# ---------------------------------------------------------------- experiment 2
def fig_roc():
    rows = read_csv(RES / "verifier" / "checks_test.csv")
    t = np.array([r["check_type"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    for model in ["baseline", "trained"]:
        s = np.array([float(r[f"score_{model}"]) for r in rows])
        for sub, ls in [("random", "-"), ("hard", "--")]:
            m = np.isin(t, ["genuine", sub])
            y = (t[m] == "genuine").astype(int)
            fpr, tpr, _ = roc_curve(y, s[m])
            ax.plot(fpr, tpr, ls, color=COLORS[model], lw=1.5,
                    label=f"{model}, {sub} impostor (AUC {roc_auc_score(y, s[m]):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    ax.set_xlabel("False accept rate")
    ax.set_ylabel("True accept rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    save(fig, "exp2_roc")


def fig_accuracy_by_part(vm):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), sharey=True)
    x = np.arange(len(PARTS))
    w = 0.38
    for ax, ct, title in zip(axes, ["genuine", "random", "hard"],
                             ["Genuine checks (accepted)", "Random impostors (rejected)", "Hard impostors (rejected)"]):
        for k, model in enumerate(["baseline", "trained"]):
            vals = [vm["test"][model]["by_part"][p][f"accuracy_{ct}"] for p in PARTS]
            ax.bar(x + (k - 0.5) * w, vals, w, color=COLORS[model], label=model)
        ax.set_xticks(x, [LABEL[p] for p in PARTS], fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test accuracy")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "exp2_accuracy_by_part")


# ---------------------------------------------------------------- experiment 3
SET_NAMES = {"A": "A: 10 albums", "B": "B: 302 albums", "C": "C: 302 albums"}


def grouped_bars(series, labels, ylabel, ylim, name, xlabel=None):
    """Column-wide grouped bar chart shared by both breadth vs depth figures.

    series: list of (legend label, colour, means per group, sd per group or None), drawn left to right in a group.
    """
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.6))
    for k, (lab, color, means, err) in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, means, w, yerr=err, capsize=2, error_kw={"lw": 0.8},
               color=color, label=lab)
    ax.set_xticks(x, labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24 if xlabel else -0.13), ncol=2, fontsize=7, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    save(fig, name)


def fig_breadth_depth(summary):
    groups = ["macro_f1"] + [f"f1_{p}" for p in PARTS]
    labels = ["macro F1"] + [LABEL[p] for p in PARTS]
    series = []
    for s in ["A", "B", "C"]:
        r = summary[s]
        series.append((f"{SET_NAMES[s]}, {int(r['n_images']):,} images", COLORS[s],
                       [float(r[f"test_{g}"]) for g in groups],
                       [float(r[f"test_{g}_sd"]) for g in groups] if s == "B" else None))
    grouped_bars(series, labels, "Test F1", (0, 1), "exp3_breadth_depth")


VERIFIER_YLIM = (0.85, 1.0)                 # every bar lies inside this range, so no bar is cut off


def fig_breadth_depth_verifier(summary_v, vm):
    subsets = ["all", "random", "hard"]
    labels = ["all", "random", "hard"]
    series = [("untrained baseline", COLORS["untrained"], [vm["test"]["baseline"][s]["auc"] for s in subsets], None)]
    for s in ["A", "B", "C"]:
        r = summary_v[s]
        cols = ["test_auc"] + [f"test_auc_{sub}" for sub in subsets[1:]]
        series.append((f"{SET_NAMES[s]}, {int(r['n_images']):,} images", COLORS[s],
                       [float(r[c]) for c in cols], [float(r[f"{c}_sd"]) for c in cols] if s == "B" else None))
    lo, hi = VERIFIER_YLIM
    vals = [v for _, _, m, _ in series for v in m]
    if min(vals) < lo or max(vals) > hi:
        raise SystemExit(f"Verifier AUCs {min(vals):.3f} to {max(vals):.3f} fall outside the axis {lo} to {hi}")
    grouped_bars(series, labels, "Test AUC", VERIFIER_YLIM, "exp3_breadth_depth_verifier", xlabel="Impostors")


# ---------------------------------------------------------------- tables
def table_dataset():
    rows = read_csv(ROOT / "data" / "manifest_all.csv")
    split_of = {r["album"]: r["split"] for r in read_csv(ROOT / "data" / "split_albums.csv")}
    out = []
    for sp in ["train", "val", "test", "total"]:
        rs = [r for r in rows if sp == "total" or split_of[r["album"]] == sp]
        rel = defaultdict(set)
        for r in rs:
            rel[r["album"]].add(r["release_id"])
        out.append({"split": sp, "albums": len(rel), "releases": len({r["release_id"] for r in rs}),
                    "images": len(rs), **{p: sum(r["type"] == p for r in rs) for p in PARTS},
                    "albums_2plus_releases": sum(len(v) >= 2 for v in rel.values())})
    write_csv(TAB / "dataset_split.csv", out)


def table_part(metrics):
    out = [{"row": p, "precision": c["precision"], "recall": c["recall"], "f1": c["f1"], "support": c["support"]}
           for p, c in metrics["per_class"].items()]
    n = metrics["test_size"]["images"]
    out.append({"row": "macro", "precision": float(np.mean([r["precision"] for r in out])),
                "recall": float(np.mean([r["recall"] for r in out])), "f1": metrics["macro_f1"], "support": n})
    out.append({"row": "accuracy", "precision": "", "recall": "", "f1": metrics["accuracy"], "support": n})
    write_csv(TAB / "part_classifier_test.csv", out)


def table_verifier(vm):
    out, by_part = [], []
    for model in ["baseline", "trained"]:
        st = vm["settings"][model]
        for sub in ["random", "hard", "all"]:
            r = vm["test"][model][sub]
            out.append({"model": model, "subset": sub, "scorer": st["scorer"], "threshold": st["threshold"],
                        "auc": r["auc"], "accuracy": r["accuracy"], "genuine_accept_rate": r["genuine_accept_rate"],
                        "impostor_reject_rate": r["impostor_reject_rate"], "n_checks": r["n_checks"]})
        for p in PARTS:
            r = vm["test"][model]["by_part"][p]
            by_part.append({"model": model, "part": p, "n_queries": r["n_queries"], "accuracy_all": r["accuracy_all"],
                            "accuracy_genuine": r["accuracy_genuine"], "accuracy_random": r["accuracy_random"],
                            "accuracy_hard": r["accuracy_hard"]})
    write_csv(TAB / "verifier_test.csv", out)
    write_csv(TAB / "verifier_by_part.csv", by_part)


def table_breadth(summary):
    out = []
    for s in ["A", "B", "C"]:
        r = summary[s]
        row = {"set": s, "runs": int(r["runs"]), "images": int(r["n_images"]), "albums": int(r["n_albums"]),
               "chosen_C": r["chosen_C"]}
        for m in ["accuracy", "macro_f1"] + [f"f1_{p}" for p in PARTS]:
            row[m] = float(r[f"test_{m}"])
            row[f"{m}_sd"] = float(r[f"test_{m}_sd"])
        out.append(row)
    write_csv(TAB / "breadth_depth.csv", out)


# ---------------------------------------------------------------- LaTeX tables for the report
def n(v):
    return f"{int(v):,}"


def f3(v):
    return f"{float(v):.3f}"


def pct(v):
    return f"{100 * float(v):.1f}\\%"


def pm(mean, sd):
    return f"{f3(mean)} $\\pm$ {f3(sd)}"


def write_tex(name, source, lines):
    """results/tables/<name>.tex, also copied to reports/final_report/tables/ unless --no-copy."""
    text = f"% Generated from {source}\n" + "\n".join(lines) + "\n"
    (TAB / f"{name}.tex").write_text(text, encoding="utf-8")
    if COPY_TO_REPORT:
        shutil.copy(TAB / f"{name}.tex", REPORT_TAB / f"{name}.tex")


SET_ROWS = {"A": "A: depth", "B": "B: breadth", "C": "C: all data"}


def tex_dataset():
    rows = {r["split"]: r for r in read_csv(TAB / "dataset_split.csv")}
    names = {"train": "Train", "val": "Validation", "test": "Test", "total": "Total"}

    def line(sp):
        r = rows[sp]
        return " & ".join([names[sp]] + [n(r[c]) for c in ["albums", "releases", "images"] + PARTS]) + " \\\\"
    write_tex("dataset_split", "results/tables/dataset_split.csv", [
        "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
        "Set & Albums & Releases & Images & " + " & ".join(LABEL[p].capitalize() for p in PARTS) + " \\\\",
        "\\midrule", line("train"), line("val"), line("test"), "\\midrule", line("total"),
        "\\bottomrule", "\\end{tabular}"])


def tex_part():
    rows = {r["row"]: r for r in read_csv(TAB / "part_classifier_test.csv")}
    body = [f"{LABEL[p].capitalize()} & {f3(rows[p]['precision'])} & {f3(rows[p]['recall'])} & {f3(rows[p]['f1'])}"
            f" & {n(rows[p]['support'])} \\\\" for p in PARTS]
    m, a = rows["macro"], rows["accuracy"]
    write_tex("exp1_type_metrics", "results/tables/part_classifier_test.csv", [
        "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Type & Precision & Recall & F1 & Images \\\\", "\\midrule",
        *body, "\\midrule",
        f"Macro average & {f3(m['precision'])} & {f3(m['recall'])} & {f3(m['f1'])} & {n(m['support'])} \\\\",
        f"Accuracy & & & {f3(a['f1'])} & {n(a['support'])} \\\\", "\\bottomrule", "\\end{tabular}"])


def tex_verifier():
    rows = read_csv(TAB / "verifier_test.csv")
    body = []
    for model in ["baseline", "trained"]:
        if body:
            body.append("\\midrule")
        for k, sub in enumerate(["random", "hard", "all"]):
            r = next(r for r in rows if r["model"] == model and r["subset"] == sub)
            body.append(f"{model.capitalize() if k == 0 else ''} & {sub.capitalize()} & {f3(r['auc'])} & "
                        f"{pct(r['accuracy'])} & {pct(r['genuine_accept_rate'])} & {pct(r['impostor_reject_rate'])} \\\\")
    write_tex("exp2_verifier_metrics", "results/tables/verifier_test.csv", [
        "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
        " & & & & Genuine & Impostors \\\\", "Model & Impostors & AUC & Accuracy & accepted & rejected \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}"])

    rows = read_csv(TAB / "verifier_by_part.csv")
    get = lambda model, p: next(r for r in rows if r["model"] == model and r["part"] == p)
    body = [f"{LABEL[p].capitalize()} & {n(get('trained', p)['n_queries'])} & "
            f"{pct(get('baseline', p)['accuracy_genuine'])} & {pct(get('trained', p)['accuracy_genuine'])} & "
            f"{pct(get('trained', p)['accuracy_random'])} & {pct(get('trained', p)['accuracy_hard'])} \\\\"
            for p in PARTS]
    write_tex("exp2_verifier_by_type", "results/tables/verifier_by_part.csv", [
        "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
        " & & \\multicolumn{2}{c}{Genuine accepted} & \\multicolumn{2}{c}{Impostors rejected (trained)} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}", "Type & Images & Baseline & Trained & Random & Hard \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}"])


def tex_breadth():
    rows = {r["set"]: r for r in read_csv(TAB / "breadth_depth.csv")}
    metrics = ["accuracy", "macro_f1"] + [f"f1_{p}" for p in PARTS]
    body = []
    for s in ["A", "B", "C"]:
        r = rows[s]
        cells = [pm(r[m], r[f"{m}_sd"]) if s == "B" else f3(r[m]) for m in metrics]
        body.append(" & ".join([SET_ROWS[s], n(r["albums"]), n(r["images"])] + cells) + " \\\\")
    write_tex("exp3_breadth_depth", "results/tables/breadth_depth.csv", [
        "\\begin{tabular}{@{}lrrcccccc@{}}", "\\toprule",
        "Training set & Albums & Images & Accuracy & Macro F1 & "
        + " & ".join(LABEL[p].capitalize() for p in PARTS) + " \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}"])


def tex_breadth_verifier(summary_v, vm):
    body = [f"No training & -- & -- & {f3(vm['test']['baseline']['all']['auc'])} & -- & -- & -- \\\\"]
    for s in ["A", "B", "C"]:
        r = summary_v[s]
        body.append(" & ".join([SET_ROWS[s], n(r["n_albums"]), n(r["n_images"])] + [f3(r[c]) for c in [
            "test_auc", "test_random_reject", "test_hard_reject", "test_genuine_accept"]]) + " \\\\")
    write_tex("exp3_verifier", "results/breadth_depth_verifier/summary.csv", [
        "\\begin{tabular}{@{}lrrcccc@{}}", "\\toprule",
        " & & & & \\multicolumn{2}{c}{Impostors rejected} & Genuine \\\\", "\\cmidrule(lr){5-6}",
        "Training set & Albums & Images & AUC & Random & Hard & accepted \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}"])


def main():
    for d in (FIG, TAB, REPORT_FIG, REPORT_TAB):
        d.mkdir(parents=True, exist_ok=True)
    pcm = json.load(open(RES / "part_classifier" / "metrics.json"))
    vm = json.load(open(RES / "verifier" / "metrics.json"))
    summary = {r["set"]: r for r in read_csv(RES / "breadth_depth" / "summary.csv")}
    summary_v = {r["set"]: r for r in read_csv(RES / "breadth_depth_verifier" / "summary.csv")}

    fig_pipeline(vm["settings"]["trained"]["emb_size"])
    fig_tuning_C()
    fig_confusion(pcm)
    fig_roc()
    fig_accuracy_by_part(vm)
    fig_breadth_depth(summary)
    fig_breadth_depth_verifier(summary_v, vm)

    table_dataset()
    table_part(pcm)
    table_verifier(vm)
    table_breadth(summary)

    tex_dataset()
    tex_part()
    tex_verifier()
    tex_breadth()
    tex_breadth_verifier(summary_v, vm)
    where = f" (copied to {REPORT_FIG.relative_to(ROOT)}/ and {REPORT_TAB.relative_to(ROOT)}/)" if COPY_TO_REPORT else ""
    print(f"Figures in {FIG.relative_to(ROOT)}/, tables in {TAB.relative_to(ROOT)}/{where}")


if __name__ == "__main__":
    main()
