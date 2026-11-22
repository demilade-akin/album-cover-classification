"""
Frozen ResNet-50 features for every image in data/manifest_all.csv.

  python scripts/07_extract_features.py

What it does
  1. Loads the shared cache data/features_resnet50.npz and computes only the images it lacks. The 2,223
     features of the "10" dataset are reused, never recomputed.
  2. Uses exactly the settings of scripts/05_pilot_learning_curve.py, so old and new features are
     comparable: Keras ResNet-50 with ImageNet weights, include_top=False, pooling="avg" (2048-d),
     images converted to RGB and squashed to 224x224, Keras resnet50.preprocess_input, batches of 32.
  3. Saves progress every 1,000 new images. Each save writes a temp file and then renames it over the
     cache, so a crash or Ctrl-C mid-save cannot corrupt the cache. Rerun the same command to resume.
  4. Checks at the end that every manifest image has a 2048-d feature and that none contain NaN.

Cache format (same as 05's save_cache, which 05's load_cache reads):
  X       float32 array, one 2048-d row per image
  files   image path relative to data/, e.g. images/<album>/<part>/<release_id>.jpg
  sha256  SHA-256 of the image file, from the manifest
The first cache version stored only X and files. Those entries take their SHA-256 from the manifest,
the same rule 05 uses, and are written back in the full format.

Run time: about 15 to 25 minutes on Apple Silicon CPU for the ~21,000 missing images.
"""
import csv, time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]   # repo root
DATA = ROOT / "data"
MANIFEST = DATA / "manifest_all.csv"
FEATURES = DATA / "features_resnet50.npz"    # shared cache for every dataset, keyed per image file
BATCH = 32
SAVE_EVERY = 1000                            # new images between checkpoints
DIM = 2048


def load_rows():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cache(rows):
    """{(file, sha256): feature vector}. Same logic as load_cache in 05_pilot_learning_curve.py."""
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
    """Write to <cache>.part, then rename over the cache. The rename is atomic on the same disk."""
    keys = sorted(cache)
    tmp = FEATURES.with_name(FEATURES.name + ".part")
    with open(tmp, "wb") as fh:                 # file handle, so numpy does not append its own .npz suffix
        np.savez_compressed(fh, X=np.stack([cache[k] for k in keys]).astype(np.float32),
                            files=np.array([k[0] for k in keys]), sha256=np.array([k[1] for k in keys]))
    tmp.replace(FEATURES)


def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))   # squash, no crop, same as 05
    return np.asarray(img, dtype=np.float32)


def extract(rows, cache):
    keys = [(r["file"], r["sha256"]) for r in rows]
    todo = [k for k in dict.fromkeys(keys) if k not in cache]  # dict.fromkeys drops duplicates, keeps order
    print(f"Features: {len(keys) - len(todo)} of {len(keys)} cached, {len(todo)} to extract")
    if not todo:
        return

    import tensorflow as tf  # noqa: F401  (imported here so the check-only path starts fast)
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

    # Frozen ResNet-50 without its classifier head; global average pooling gives a 2048-d vector
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    model.trainable = False

    done_since_save, t0 = 0, time.time()
    try:
        # tqdm counts images, so the rate and ETA read in images per second
        with tqdm(total=len(todo), desc="ResNet-50 features", unit="img", dynamic_ncols=True) as bar:
            for start in range(0, len(todo), BATCH):
                batch = todo[start:start + BATCH]
                x = preprocess_input(np.stack([load_image(DATA / f) for f, _ in batch]))
                for k, vec in zip(batch, model.predict(x, verbose=0)):
                    cache[k] = vec
                bar.update(len(batch))
                done_since_save += len(batch)
                if done_since_save >= SAVE_EVERY:
                    save_cache(cache)
                    done_since_save = 0
    finally:
        save_cache(cache)                       # keeps finished batches after a crash or Ctrl-C
        print(f"Saved {len(cache)} features to {FEATURES.relative_to(ROOT)} "
              f"({(time.time() - t0) / 60:.1f} min this run)")


def check(rows):
    """Reload the cache from disk and confirm full, finite, 2048-d coverage of the manifest."""
    cache = load_cache(rows)
    keys = [(r["file"], r["sha256"]) for r in rows]
    missing = [k for k in keys if k not in cache]
    if missing:
        raise SystemExit(f"CHECK FAILED: {len(missing)} manifest images have no feature, e.g. {missing[0][0]}")
    X = np.stack([cache[k] for k in keys])
    if X.shape[1] != DIM:
        raise SystemExit(f"CHECK FAILED: features are {X.shape[1]}-d, expected {DIM}")
    n_nan = int(np.isnan(X).any(axis=1).sum())
    if n_nan:
        raise SystemExit(f"CHECK FAILED: {n_nan} features contain NaN")
    print(f"Check passed: {X.shape[0]} images, {X.shape[1]}-d, no NaN. Cache holds {len(cache)} entries.")


def main():
    rows = load_rows()
    extract(rows, load_cache(rows))
    check(rows)


if __name__ == "__main__":
    main()
