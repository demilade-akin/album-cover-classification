"""
Check one uploaded image: predict its packaging part and, optionally, verify it against a claimed album.

  python scripts/predict.py IMAGE [--album FOLDER]

  IMAGE    any image file (JPEG, PNG, ...), inside the dataset or not
  FOLDER   an album folder name from data/metadata/albums_all.csv, e.g. michael_jackson__thriller

What it does
  1. Extracts frozen ResNet-50 features with exactly the preprocessing of 07_extract_features.py
     (RGB, squashed to 224x224, Keras resnet50.preprocess_input). load_image is imported from 07 itself.
  2. Part: the trained part classifier (results/part_classifier/model.joblib) prints all four probabilities.
  3. Album (with --album): embeds the image with the trained projection (results/verifier/projection.keras) and
     takes the top-1 cosine similarity against the claimed album's gallery. Compares it with the threshold
     chosen on validation albums (results/verifier/metrics.json) and prints match or no match.

Gallery rules (same as experiment 2, so a scan cannot match itself)
  - Gallery = every image of the claimed album in data/manifest_all.csv.
  - If IMAGE is itself in the dataset (same SHA-256), its whole release is left out.
  - Images within pHash distance 2 of IMAGE (near-duplicate scans, data/phash.csv) are left out.
Gallery features come from data/features_resnet50.npz. If the cache is missing, they are computed from the images
in data/images/ (slower). One of the two must exist.
"""
import argparse, csv, hashlib, importlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data"
RES = ROOT / "results"
PARTS = ["front", "back", "medium", "booklet"]
NEAR_DUP = 2

fx = importlib.import_module("07_extract_features")     # same load_image as the feature cache


def resnet_features(paths):
    """(n, 2048) frozen ResNet-50 features for a list of image paths."""
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    out = []
    for start in range(0, len(paths), 32):
        x = preprocess_input(np.stack([fx.load_image(p) for p in paths[start:start + 32]]))
        out.append(model.predict(x, verbose=0))
    return np.concatenate(out).astype(np.float32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def predict_part(feat):
    import joblib
    model = joblib.load(RES / "part_classifier" / "model.joblib")
    proba = model.predict_proba(feat[None])[0]
    by_part = dict(zip(model.classes_, proba))
    best = max(by_part, key=by_part.get)
    print("\nPart classifier")
    for p in PARTS:
        print(f"  {p:<8} {by_part[p]:6.1%}{'  <- predicted' if p == best else ''}")


def verify_album(image, feat, folder):
    import imagehash, keras
    from PIL import Image

    rows = read_csv(DATA / "manifest_all.csv")
    albums = {r["folder"]: r for r in read_csv(DATA / "metadata" / "albums_all.csv")}
    if folder not in albums:
        raise SystemExit(f"Unknown album folder '{folder}'. See the folder column of data/metadata/albums_all.csv")
    split = {r["album"]: r["split"] for r in read_csv(DATA / "split_albums.csv")}[folder]
    gallery = [r for r in rows if r["album"] == folder]

    # Leave out the query's own release (if the query is a dataset image) and near-duplicate scans
    own = {r["release_id"] for r in rows if r["sha256"] == sha256(image)}
    notes = []
    if own:
        before = len(gallery)
        gallery = [r for r in gallery if r["release_id"] not in own]
        if before - len(gallery):                       # only when the query's release belongs to this album
            notes.append(f"query is a dataset image: left out its release ({before - len(gallery)} images)")
    phash = {(r["file"], r["sha256"]): int(r["phash"], 16) for r in read_csv(DATA / "phash.csv")}
    with Image.open(image) as img:
        q_hash = int(str(imagehash.phash(img)), 16)
    before = len(gallery)
    gallery = [r for r in gallery if bin(phash[(r["file"], r["sha256"])] ^ q_hash).count("1") > NEAR_DUP]
    if before - len(gallery):
        notes.append(f"left out {before - len(gallery)} near-duplicate scan(s) of the query")
    if not gallery:
        raise SystemExit("The claimed album has no gallery images left after the exclusions")

    # Gallery features: from the cache if present, otherwise from the images on disk
    cache_path = DATA / "features_resnet50.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        index = {(f, h): i for i, (f, h) in enumerate(zip(cached["files"], cached["sha256"]))}
        G = cached["X"][[index[(r["file"], r["sha256"])] for r in gallery]]
    else:
        missing = [r["file"] for r in gallery if not (DATA / r["file"]).exists()]
        if missing:
            raise SystemExit("Need data/features_resnet50.npz or the album's images in data/images/ "
                             "(run 04_download_project.py all, then 07_extract_features.py)")
        G = resnet_features([DATA / r["file"] for r in gallery])

    settings = json.load(open(RES / "verifier" / "metrics.json"))["settings"]["trained"]
    projection = keras.saving.load_model(RES / "verifier" / "projection.keras")
    E = projection.predict(np.vstack([feat[None], G]), verbose=0)     # L2-normalised by the model
    sims = E[1:] @ E[0]
    best = int(np.argmax(sims))
    score, thr = float(sims[best]), settings["threshold"]
    if settings["scorer"] != "top1":
        raise SystemExit(f"predict.py implements top-1 only, metrics.json says {settings['scorer']}")

    a = albums[folder]
    print(f"\nAlbum verifier: claimed album {a['artist']}: {a['album']} ({folder}, a {split} album)")
    for n in notes:
        print(f"  note: {n}")
    if split == "train":
        print("  note: the projection was trained on this album, so this is not an unseen-album check")
    print(f"  gallery    {len(gallery)} images")
    print(f"  score      {score:.4f}  (top-1 cosine, best match {gallery[best]['file']})")
    print(f"  threshold  {thr:.4f}  (chosen on validation albums)")
    print(f"  decision   {'MATCH' if score >= thr else 'NO MATCH: flag for review'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--album", help="claimed album folder, e.g. michael_jackson__thriller")
    args = ap.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"No such file: {args.image}")
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(args.image) as img:
            img.verify()                                # fail early and clearly on non-image files
    except UnidentifiedImageError:
        raise SystemExit(f"Not an image file: {args.image}")

    feat = resnet_features([args.image])[0]
    print(f"Image: {args.image}")
    predict_part(feat)
    if args.album:
        verify_album(args.image, feat, args.album)


if __name__ == "__main__":
    main()
