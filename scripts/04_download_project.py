"""
Build a project dataset: one image per type per release for the albums in one album list.

  python scripts/04_download_project.py 10     albums_10.csv  -> data/manifest_10.csv
  python scripts/04_download_project.py all    albums_all.csv -> data/manifest_all.csv

Both datasets share one snapshot, one release-metadata file and one image folder. An image already on disk is
reused, never downloaded again, so the 10 albums are the same files in both datasets.

Inputs (committed, so a fresh clone rebuilds the same dataset while the Cover Art Archive still serves the images)
  data/metadata/image_types_raw.csv   pinned image snapshot from 03_count_image_types.py
  data/metadata/releases.csv          pinned release metadata (date, country, format). Albums missing from it are
                                      fetched from MusicBrainz once and appended. Albums already in it are never
                                      refetched.

Selection rules (deterministic: each pick depends only on the release ID and type, not on the album list)
  - Types kept: Front, Back, Medium, Booklet. "Back;Spine" counts as Back and "Medium;Matrix/Runout" as Medium.
    Every other tag or combination is dropped (Tray, Other, Untyped, Front;Sticker, ...).
  - Unapproved images are dropped.
  - Picks listed in excluded_images.csv are dropped after selection (the slot stays empty, no other candidate
    takes it). These are the 8 images that failed to download in the original 10-album build. Pinning them
    out keeps the 10-album dataset at the 2,223 images the data report describes, in both datasets.
  - One image per type per release. Single-tag images win over the mapped combinations. When a release has
    several candidates (booklet pages, multiple discs), the pick is seeded by release ID and type.

Output
  data/images/<album>/<type>/<release_id>.jpg   500px thumbnails (250px fallback), saved as RGB JPEG
  data/manifest_<name>.csv                      one row per selected image on disk, with release metadata, SHA-256
  data/logs/04_failures_<name>.csv              downloads that failed this run (rerun to retry)

Resumable: files are written to .part and renamed, so Ctrl-C or a crash never leaves a broken image.
Failures never stop the run: an album whose MusicBrainz metadata keeps failing and images that keep failing are
skipped, listed at the end, and fetched by the next run.
Needs ALBUM_PROJECT_CONTACT set (see http_client.py) only when something must be fetched.
"""
import argparse, csv, hashlib, io, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from http_client import get, user_agent

ROOT = Path(__file__).resolve().parents[1]   # repo root
META = ROOT / "data/metadata"
DATASETS = {"10": META / "albums_10.csv", "all": META / "albums_all.csv"}
RAW_PATH = META / "image_types_raw.csv"
RELEASES_PATH = META / "releases.csv"
EXCLUDED_PATH = META / "excluded_images.csv"   # picks that are pinned out of every dataset (see below)
IMG_DIR = ROOT / "data/images"
SEED = 42
WORKERS = 4                  # parallel image downloads (MusicBrainz calls stay sequential at 1 per second)
SIZES = ["500", "250"]       # thumbnail sizes to try, in order

# Tag set on the Cover Art Archive -> class folder. Anything not listed is dropped.
TYPE_MAP = {
    frozenset({"Front"}): "front",
    frozenset({"Back"}): "back",
    frozenset({"Medium"}): "medium",
    frozenset({"Booklet"}): "booklet",
    frozenset({"Back", "Spine"}): "back",
    frozenset({"Medium", "Matrix/Runout"}): "medium",
}
TYPES = ["front", "back", "medium", "booklet"]
RELEASE_COLS = ["date", "year", "country", "formats", "packaging", "status"]
RELEASE_FIELDS = ["album", "release_id", "title"] + RELEASE_COLS
MANIFEST_COLS = ["album", "type", "release_id", "image_id", "tags", "candidates", "file", "sha256"] + RELEASE_COLS


def load_albums(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["folder"], r["rg_id"]) for r in csv.DictReader(f)]


def write_csv_atomic(path, rows, cols):
    """Write to a .part file, then rename, so a crash never leaves a half-written file. LF endings, as committed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


# 1. Pick one image per (release, type) from the snapshot -----------------------------------------
def select_images(folders):
    groups = defaultdict(list)    # (album, release_id, type) -> [(single_tag, image_id, tags)]
    with open(RAW_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["album"] not in folders or not r["image_id"] or r["approved"] != "True":
                continue
            tags = frozenset(r["types"].split(";"))
            kind = TYPE_MAP.get(tags)
            if kind:
                groups[(r["album"], r["release_id"], kind)].append((len(tags) == 1, r["image_id"], r["types"]))

    with open(EXCLUDED_PATH, newline="", encoding="utf-8") as f:
        excluded = {(r["release_id"], r["type"]) for r in csv.DictReader(f)}

    picks = []
    for (album, rid, kind), cands in sorted(groups.items()):
        pool = sorted([c for c in cands if c[0]] or cands, key=lambda c: int(c[1]))
        _, image_id, tags = random.Random(f"{SEED}-{rid}-{kind}").choice(pool)
        if (rid, kind) in excluded:
            continue                       # pick first, then drop, so no other image replaces it
        picks.append({"album": album, "release_id": rid, "type": kind, "image_id": image_id,
                      "tags": tags, "candidates": len(cands)})
    return picks


# 2. Release metadata, fetched once per album and pinned ------------------------------------------
def fetch_album_releases(folder, rg_id):
    rows, offset = [], 0
    while True:
        data = get("https://musicbrainz.org/ws/2/release",
                   {"release-group": rg_id, "inc": "media", "limit": 100, "offset": offset, "fmt": "json"}).json()
        for rel in data["releases"]:
            formats = sorted({m.get("format") or "Unknown" for m in rel.get("media", [])})
            date = rel.get("date") or ""
            rows.append({"album": folder, "release_id": rel["id"], "title": rel.get("title", ""),
                         "date": date, "year": date[:4], "country": rel.get("country") or "",
                         "formats": "+".join(formats) or "Unknown",
                         "packaging": rel.get("packaging") or "", "status": rel.get("status") or ""})
        offset += 100
        if offset >= data["release-count"]:
            return rows


def load_releases(albums):
    """Release metadata keyed by release ID. Fetches only albums missing from releases.csv and appends them.
    The file is rewritten atomically after each album, so a crash loses at most the album in progress."""
    rows = []
    if RELEASES_PATH.exists():
        with open(RELEASES_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    have = {r["album"] for r in rows}
    missing = [(a, rg) for a, rg in albums if a not in have]
    failed = []
    if missing:
        user_agent()                       # fail early if ALBUM_PROJECT_CONTACT is unset
        print(f"Release metadata: {len(albums) - len(missing)} albums pinned, fetching {len(missing)}")
        try:
            for folder, rg_id in tqdm(missing, desc="release metadata", unit="album", dynamic_ncols=True):
                try:
                    rows += fetch_album_releases(folder, rg_id)
                except (requests.RequestException, ValueError, KeyError):
                    # MusicBrainz kept failing: skip the album, the next run fetches it again.
                    # Its images still download. Its manifest rows have blank metadata until then.
                    failed.append(folder)
                    continue
                write_csv_atomic(RELEASES_PATH, rows, RELEASE_FIELDS)
        except KeyboardInterrupt:
            sys.exit("\nStopped. Albums fetched so far are saved in releases.csv. Rerun to continue.")
    return {r["release_id"]: r for r in rows}, failed


# 3. Download one image, saved as RGB JPEG --------------------------------------------------------
def dest_of(p):
    return IMG_DIR / p["album"] / p["type"] / f"{p['release_id']}.jpg"


def download(p):
    dest = dest_of(p)
    if dest.exists():
        return p, "kept"
    for size in SIZES:
        url = f"https://coverartarchive.org/release/{p['release_id']}/{p['image_id']}-{size}.jpg"
        try:
            r = get(url, pause=0.3)
        except requests.RequestException:
            return p, "failed"
        if r is None:
            continue                       # this size doesn't exist, try the next
        try:
            img = Image.open(io.BytesIO(r.content))
            img.load()
        except Exception:
            return p, "failed"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        if img.format == "JPEG" and img.mode == "RGB":
            tmp.write_bytes(r.content)
        else:                              # PNG, grayscale or CMYK: convert so Keras reads every file the same way
            img.convert("RGB").save(tmp, "JPEG", quality=95)
        tmp.replace(dest)
        return p, "ok"
    return p, "failed"


def download_all(picks):
    """Downloads only images not on disk. Progress per album and overall (with ETA over real downloads)."""
    todo = [p for p in picks if not dest_of(p).exists()]
    print(f"{len(picks) - len(todo)} images already on disk, {len(todo)} to download")
    if todo:
        user_agent()                       # fail early if ALBUM_PROJECT_CONTACT is unset
    by_album = defaultdict(list)
    for p in todo:
        by_album[p["album"]].append(p)

    failed = []
    overall = tqdm(total=len(todo), desc="overall", unit="img", position=0, dynamic_ncols=True)
    try:
        for i, (album, items) in enumerate(sorted(by_album.items()), 1):
            overall.set_postfix_str(f"album {i}/{len(by_album)} {album[:28]}, failed {len(failed)}")
            pool = ThreadPoolExecutor(max_workers=WORKERS)
            try:
                futures = [pool.submit(download, p) for p in items]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  {album[:30]}",
                                unit="img", position=1, leave=False, dynamic_ncols=True):
                    p, status = fut.result()
                    overall.update(1)
                    if status == "failed":
                        failed.append(p)
            finally:
                # On Ctrl-C, drop queued downloads. Running ones finish and rename their .part file.
                pool.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        overall.close()
        print("\nStopped. Images on disk are complete. Rerun to continue (the manifest was not rewritten).")
        sys.exit(130)
    overall.close()
    return failed


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Download one image per type per release for an album list.")
    ap.add_argument("dataset", choices=sorted(DATASETS), help="10 (albums_10.csv) or all (albums_all.csv)")
    name = ap.parse_args().dataset
    manifest_path = ROOT / f"data/manifest_{name}.csv"
    fail_path = ROOT / f"data/logs/04_failures_{name}.csv"

    albums = load_albums(DATASETS[name])
    picks = select_images({folder for folder, _ in albums})
    print(f"Dataset {name}: {len(albums)} albums, {len(picks)} images selected from {RAW_PATH.name}")
    releases, meta_failed = load_releases(albums)
    failed = download_all(picks)

    # Manifest is rebuilt from disk every run, so it always matches the folder
    counts, rows = defaultdict(Counter), []
    for p in tqdm(picks, desc="hashing", unit="img", leave=False, dynamic_ncols=True):
        dest = dest_of(p)
        if not dest.exists():
            continue
        counts[p["album"]][p["type"]] += 1
        meta = releases.get(p["release_id"], {})
        rows.append({**p, "file": dest.relative_to(ROOT / "data").as_posix(), "sha256": sha256(dest),
                     **{c: meta.get(c, "") for c in RELEASE_COLS}})
    write_csv_atomic(manifest_path, rows, MANIFEST_COLS)

    # Totals only: a per-album table is unreadable at 430 albums (the manifest has the detail)
    total = Counter()
    for c in counts.values():
        total.update(c)
    per_album = sorted(sum(c.values()) for c in counts.values())
    print(f"\n{'':<10}" + "".join(f"{t:>9}" for t in TYPES) + f"{'total':>9}")
    print(f"{'images':<10}" + "".join(f"{total[t]:>9}" for t in TYPES) + f"{sum(total.values()):>9}")
    print(f"{'albums>0':<10}" + "".join(f"{sum(1 for c in counts.values() if c[t]):>9}" for t in TYPES)
          + f"{len(counts):>9}")
    if per_album:
        print(f"\nImages per album: min {per_album[0]}, median {per_album[len(per_album) // 2]}, "
              f"max {per_album[-1]}. Releases: {len({r['release_id'] for r in rows})}")
    no_images = sorted({a for a, _ in albums} - set(counts))
    if no_images:
        print(f"{len(no_images)} albums have no usable image: {', '.join(no_images[:10])}"
              + (" ..." if len(no_images) > 10 else ""))
    print(f"\nSaved {manifest_path.relative_to(ROOT)}")
    if meta_failed:
        print(f"\nRelease metadata failed for {len(meta_failed)} albums (MusicBrainz errors). Their manifest rows "
              f"have blank date, country and format. Rerun to fetch them: {', '.join(meta_failed[:10])}"
              + (" ..." if len(meta_failed) > 10 else ""))
    if failed:
        write_csv_atomic(fail_path, failed, ["album", "type", "release_id", "image_id", "tags", "candidates"])
        print(f"\n{len(failed)} images failed (archive.org errors or missing thumbnails). Rerun to retry. "
              f"List: {fail_path.relative_to(ROOT)}")
        for p in failed[:10]:
            print(f"  {p['album']}  {p['type']}  {p['release_id']}")


if __name__ == "__main__":
    main()
