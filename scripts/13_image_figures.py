"""
Report figures built from the images themselves (data/images/), not from results/.

  python scripts/13_image_figures.py             write the figures and copy the PDFs into the report
  python scripts/13_image_figures.py --no-copy   write to results/figures/ only

Needs the images (rebuild them with scripts/04_download_project.py). Unlike scripts/12_make_figures.py, this
script cannot run on a fresh clone. The images belong to their uploaders, so each figure shows small thumbnails
and the report credits the Cover Art Archive.

Figures (results/figures/, each as .pdf and .png; the .pdf files are copied to reports/final_report/figures/)
  image_types.pdf          the four image types from one release, in a 2 by 2 grid: front, back, disc, booklet
  audit_dark_side.pdf      three images of The Dark Side of the Moon in one row: a front, a back the type classifier
                           called a front (same release), and another back of the album that it called a back
  audit_black_sabbath.pdf  the same for Black Sabbath
  audit_bitches_brew.pdf   the same for Bitches Brew

The audit figures show why the type classifier is most confident when it is wrong: backs that carry clean artwork
look like fronts to it, while a typical back carries text, a barcode and small print. Each label reads the
prediction and its probability from results/part_classifier/predictions.csv, so no number is typed by hand.

The releases are chosen by hand from the audit (results/audit/hand_check.csv). The image paths come from
data/manifest_all.csv, so the script fails clearly if an image is missing.
"""
import csv, shutil, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"
REPORT_FIG = ROOT / "reports" / "final_report" / "figures"
COPY_TO_REPORT = "--no-copy" not in sys.argv

PARTS = ["front", "back", "medium", "booklet"]                 # names in the data (MusicBrainz terms)
LABEL = {"front": "Front", "back": "Back", "medium": "Disc", "booklet": "Booklet"}   # names in the report

# Image types figure: Miles Davis, Kind of Blue, 1997 US CD in a jewel case
TYPES_ALBUM = "miles_davis__kind_of_blue"
TYPES_RELEASE = "7d7ebace-0e5a-43c2-bd6d-18b5f35bed0b"

# Audit figures: for each album, the release whose back the classifier called a front (its front is shown too),
# and a release whose back it called a back. Both backs come from the test albums' predictions.
AUDIT_ROWS = {
    #  figure name            album                                    back called front                       back called back
    "audit_dark_side": ("pink_floyd__the_dark_side_of_the_moon", "b84ee12a-09ef-421b-82de-0441a926375b", "10600476-3b48-4709-a91d-3fdf6d2455e6"),
    "audit_black_sabbath": ("black_sabbath__black_sabbath", "9fd196b4-0437-4ce2-9e7e-9966dfa86180", "a55102aa-9658-454b-83d2-22f8bb13c5d6"),
    "audit_bitches_brew": ("miles_davis__bitches_brew", "79bd4086-dc50-47a6-a317-dc73338ed8fe", "e20153d0-4bef-4f4f-963e-71967da294b9"),
}

THUMB_PX = 900        # longest side of each embedded image, large enough to print clearly at full width
FIG_W = 6.5           # inches, the text width of the single-column report

plt.rcParams.update({"font.size": 12})


def image_paths(album, release, needed=PARTS):
    """Map each part to its image file for one release, from the manifest. Stops if a needed part is missing."""
    paths = {}
    with open(DATA / "manifest_all.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["album"] == album and row["release_id"] == release:
                paths[row["type"]] = DATA / row["file"]
    missing = [p for p in needed if p not in paths]
    if missing:
        sys.exit(f"Release {release} of {album} has no image for: {', '.join(missing)}")
    for part, path in paths.items():
        if not path.exists():
            sys.exit(f"Missing image file for {part}: {path}\nRebuild the images with scripts/04_download_project.py")
    return paths


def thumbnail(path):
    """Resize an image so its longest side is THUMB_PX, keeping its proportions, and centre it on a white
    square. Every cell then has the same size, so the labels under the images line up."""
    img = Image.open(path).convert("RGB")
    scale = THUMB_PX / max(img.size)              # scale up small scans too, so every image fills its cell
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    cell = Image.new("RGB", (THUMB_PX, THUMB_PX), "white")
    cell.paste(img, ((THUMB_PX - img.width) // 2, (THUMB_PX - img.height) // 2))
    return cell


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if COPY_TO_REPORT:
        shutil.copy(FIG / f"{name}.pdf", REPORT_FIG / f"{name}.pdf")
    print(f"wrote {FIG.relative_to(ROOT)}/{name}.pdf")


def image_types():
    """A 2 by 2 grid of the four images with the type written under each, large enough to read details."""
    paths = image_paths(TYPES_ALBUM, TYPES_RELEASE)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_W + 0.5))
    for ax, part in zip(axes.flat, PARTS):
        ax.imshow(thumbnail(paths[part]))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(LABEL[part], labelpad=6)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    save(fig, "image_types")


def load_predictions():
    """Type classifier predictions on the test albums, keyed by image file."""
    with open(ROOT / "results" / "part_classifier" / "predictions.csv", newline="") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def prediction_label(pred, path):
    """Short two-line label: the uploader's tag, then the model's prediction and its probability,
    e.g. "Back" over "called front (98%)"."""
    row = pred[str(path.relative_to(DATA))]
    p = float(row[f"p_{row['pred_part']}"])
    return f"{LABEL[row['true_part']]}\ncalled {LABEL[row['pred_part']].lower()} ({p:.0%})"


def row_figure(name, panels):
    """Images side by side, each with a label under it. panels is a list of (path, label)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(FIG_W, FIG_W / n + 0.7))
    for ax, (path, label) in zip(axes, panels):
        ax.imshow(thumbnail(path))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(label, labelpad=8)
    fig.tight_layout(w_pad=1.5)
    save(fig, name)


def audit_figures():
    """One figure per album, in one row: the front, the back the classifier called a front (same release), and a
    back of another release that it called a back."""
    pred = load_predictions()
    for name, (album, wrong_release, right_release) in AUDIT_ROWS.items():
        wrong = image_paths(album, wrong_release, needed=["back", "front"])
        right = image_paths(album, right_release, needed=["back"])["back"]
        row_figure(name, [(wrong["front"], "Front\n"),
                          (wrong["back"], prediction_label(pred, wrong["back"])),
                          (right, prediction_label(pred, right))])


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    if COPY_TO_REPORT:
        REPORT_FIG.mkdir(parents=True, exist_ok=True)
    image_types()
    audit_figures()
