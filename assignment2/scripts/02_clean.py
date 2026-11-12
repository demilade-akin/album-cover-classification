"""
Clean the downloaded recipe images so Keras can load them, and drop images that would teach the wrong label.

Input
  data/metadata.csv, data/raw/<class>/<file>          output of 01_download_recipes.py
  data/exclude.csv (optional)                         filename,reason for files you want dropped by hand

Removal rules, applied in this order (each removed image is logged with its reason)
  1. EXCLUDE_RULES below: title patterns that mark a photo as not one dish of that class. Roundup posts
     ("20 Fresh Summer Pasta Recipes") show collages, spaghetti squash is a vegetable, ice cream is not a
     cookie, a Taiwanese pineapple cake is a pastry.
  2. data/exclude.csv, if it exists.
  3. Near-duplicates: perceptual hash (pHash) distance <= HASH_THRESHOLD. The first copy is kept. If the
     copies carry different classes, all of them are dropped as a label conflict. A copy on both sides of
     the split would leak test images into training.
  4. Files that can't be opened.

Conversion for every kept image
  EXIF rotation applied, converted to RGB (fixes WebP, PNG and palette images, which
  image_dataset_from_directory can't read or reads inconsistently), shorter side reduced to MAX_SIDE px
  if larger, aspect ratio kept, saved as JPEG quality 95. The resize to 224x224 happens in the notebook,
  so the squash-versus-crop choice stays a modelling decision.

Output
  data/clean/<class>/<site>_<id>.jpg    rebuilt from scratch on every run, so it always matches the rules
  data/clean.csv                        one row per kept image (with width, height and pHash)
  data/removed.csv                      one row per dropped image, with the reason

Run from anywhere:  python assignment2/scripts/02_clean.py
"""
import csv, re, shutil
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # assignment2 folder
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
CLEAN_DIR = DATA / "clean"
META_PATH = DATA / "metadata.csv"
EXCLUDE_PATH = DATA / "exclude.csv"
CLEAN_CSV = DATA / "clean.csv"
REMOVED_CSV = DATA / "removed.csv"

CLASSES = ["pasta", "soup", "salad", "cake", "cookies", "stir_fry"]
MAX_SIDE = 512          # shorter side cap in px. Enough headroom for random crops before the 224 resize.
JPEG_QUALITY = 95
HASH_THRESHOLD = 4      # pHash bits that may differ for two files to count as the same photo

# (reason, title regex, class or None for any class). Matching is case-insensitive.
EXCLUDE_RULES = [
    ("roundup post (collage photo)", r"^\s*\d+\+?\s.*\b(recipes|ideas)\b", None),
    ("spaghetti squash is not pasta", r"\bspaghetti squash\b", "pasta"),
    ("ice cream is not a cookie", r"\bice cream\b", "cookies"),
    ("pineapple cake is a pastry", r"\bpineapple cakes?\b", "cake"),
]
_RULES = [(reason, re.compile(pat, re.I), cls) for reason, pat, cls in EXCLUDE_RULES]


def rule_reason(row):
    """Return the reason of the first matching EXCLUDE_RULES entry, or None."""
    for reason, pat, cls in _RULES:
        if (cls is None or cls == row["class"]) and pat.search(row["recipe_title"]):
            return reason
    return None


def load_exclude():
    if not EXCLUDE_PATH.exists():
        return {}
    with open(EXCLUDE_PATH, newline="", encoding="utf-8") as f:
        return {r["filename"]: r.get("reason") or "excluded by hand" for r in csv.DictReader(f)}


def open_rgb(path):
    """Open an image upright and in RGB. Transparent areas become white rather than black."""
    with Image.open(path) as src:             # closes the file; the converted image lives in memory
        im = ImageOps.exif_transpose(src)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            return bg
        return im.convert("RGB")


def shrink(im):
    """Reduce so the shorter side is at most MAX_SIDE, keeping the aspect ratio. Never enlarges."""
    w, h = im.size
    scale = MAX_SIDE / min(w, h)
    if scale >= 1:
        return im
    return im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def main():
    with open(META_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    exclude = load_exclude()
    print(f"{len(rows)} images in {META_PATH.relative_to(ROOT)}")

    removed, candidates = [], []          # candidates: (row, phash) that passed rules 1, 2 and 4
    for row in rows:
        reason = rule_reason(row) or exclude.get(row["filename"])
        if reason:
            removed.append((row, reason))
            continue
        try:
            with open_rgb(RAW_DIR / row["class"] / row["filename"]) as im:
                h = imagehash.phash(im)   # only the hash is kept, so memory stays small
        except Exception as e:            # rule 4: unreadable file
            removed.append((row, f"unreadable ({e.__class__.__name__})"))
            continue
        candidates.append((row, h))

    # Rule 3: near-duplicates. Group every image with any earlier image within the threshold.
    groups = []                           # list of lists of candidate indices
    for i, (_, h) in enumerate(candidates):
        for g in groups:
            if any(h - candidates[j][1] <= HASH_THRESHOLD for j in g):
                g.append(i)
                break
        else:
            groups.append([i])
    keep = set()
    for g in groups:
        classes = {candidates[j][0]["class"] for j in g}
        if len(classes) > 1:              # same photo, different labels: no way to tell which is right
            for j in g:
                removed.append((candidates[j][0], "near-duplicate with a different class"))
            continue
        keep.add(g[0])
        for j in g[1:]:
            removed.append((candidates[j][0], f"near-duplicate of {candidates[g[0]][0]['filename']}"))

    # Rebuild the clean folder from scratch so it never holds files from an older rule set.
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    kept_rows = []
    for i, (row, h) in enumerate(candidates):
        if i not in keep:
            continue
        with open_rgb(RAW_DIR / row["class"] / row["filename"]) as original:
            im = shrink(original).copy()  # copy, so closing the original can't empty the image we save
        name = Path(row["filename"]).stem + ".jpg"
        dest = CLEAN_DIR / row["class"] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=JPEG_QUALITY)
        kept_rows.append({"filename": name, "class": row["class"], "source_site": row["source_site"],
                          "recipe_title": row["recipe_title"], "recipe_url": row["recipe_url"],
                          "width": im.size[0], "height": im.size[1], "phash": str(h)})

    with open(CLEAN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kept_rows[0]))
        w.writeheader()
        w.writerows(kept_rows)
    with open(REMOVED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "class", "source_site", "recipe_title", "reason"])
        w.writeheader()
        for row, reason in sorted(removed, key=lambda x: (x[0]["class"], x[0]["filename"])):
            w.writerow({k: row[k] for k in ("filename", "class", "source_site", "recipe_title")} | {"reason": reason})

    before = Counter(r["class"] for r in rows)
    after = Counter(r["class"] for r in kept_rows)
    print(f"\n{'class':<10}{'before':>8}{'after':>8}")
    for c in CLASSES:
        print(f"{c:<10}{before[c]:>8}{after[c]:>8}")
    print(f"{'total':<10}{sum(before.values()):>8}{sum(after.values()):>8}")
    print("\nRemoved, by reason")
    for reason, n in Counter(r for _, r in removed).most_common():
        print(f"  {n:>4}  {reason}")
    print(f"\nSaved {CLEAN_DIR.relative_to(ROOT)}/, {CLEAN_CSV.relative_to(ROOT)}, {REMOVED_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
