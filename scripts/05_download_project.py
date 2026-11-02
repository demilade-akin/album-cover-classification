"""
Build the final project dataset: one image per type per release for the albums in final_albums.csv.

Reads the pinned snapshot data/metadata/image_types_raw.csv (made by 04_count_image_types.py), so every
machine picks the same images. Release metadata (date, country, format) comes from MusicBrainz once and is
pinned in data/metadata/releases.csv. Both files are committed, so a fresh clone rebuilds the same dataset
as long as the Cover Art Archive still serves the images.

Selection rules
  - Types kept: Front, Back, Medium, Booklet. "Back;Spine" counts as Back and "Medium;Matrix/Runout" as Medium.
    Every other tag or combination is dropped (Tray, Other, Untyped, Front;Sticker, ...).
  - Unapproved images are dropped.
  - One image per type per release. Single-tag images win over the mapped combinations. When a release has
    several candidates (booklet pages, multiple discs), the pick is seeded, so it is the same on every run.

Output
  data/project/images/<album>/<type>/<release_id>.jpg   500px thumbnails (250px fallback), saved as RGB JPEG
  data/project/manifest.csv                            one row per selected image, with release metadata and SHA-256
  data/metadata/releases.csv                           pinned release metadata for every release of each album

Resumable: images already on disk are kept. Failed downloads are listed at the end: rerun to retry them.
Run from anywhere:  python scripts/05_download_project.py   (about 2,100 images, roughly 20 to 40 minutes)
"""
import csv, hashlib, io, random, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # project folder
HEADERS = {"User-Agent": "AlbumCoverProject/0.1 (akinbobola71@gmail.com)"}
ALBUMS_PATH = ROOT / "data/metadata/final_albums.csv"
RAW_PATH = ROOT / "data/metadata/image_types_raw.csv"
RELEASES_PATH = ROOT / "data/metadata/releases.csv"
IMG_DIR = ROOT / "data/project/images"
MANIFEST_PATH = ROOT / "data/project/manifest.csv"
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

_local = threading.local()


def session():
    # One requests.Session per thread, since sessions are not guaranteed thread-safe
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def get(url, params=None, pause=1.1, retries=4):
    """GET with a polite pause. Returns the response, or None on 404. Retries 5xx and timeouts."""
    for attempt in range(retries):
        time.sleep(pause)
        try:
            r = session().get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def load_albums():
    with open(ALBUMS_PATH, newline="", encoding="utf-8") as f:
        return [(r["folder"], r["rg_id"]) for r in csv.DictReader(f)]


# 1. Pick one image per (release, type) from the snapshot
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

    picks = []
    for (album, rid, kind), cands in sorted(groups.items()):
        pool = sorted([c for c in cands if c[0]] or cands, key=lambda c: int(c[1]))
        _, image_id, tags = random.Random(f"{SEED}-{rid}-{kind}").choice(pool)
        picks.append({"album": album, "release_id": rid, "type": kind, "image_id": image_id,
                      "tags": tags, "candidates": len(cands)})
    return picks


# 2. Release metadata, fetched once and pinned
def load_releases(albums):
    if not RELEASES_PATH.exists():
        rows = []
        for folder, rg_id in tqdm(albums, desc="release metadata", unit="album"):
            offset = 0
            while True:
                data = get("https://musicbrainz.org/ws/2/release",
                           {"release-group": rg_id, "inc": "media", "limit": 100,
                            "offset": offset, "fmt": "json"}).json()
                for rel in data["releases"]:
                    formats = sorted({m.get("format") or "Unknown" for m in rel.get("media", [])})
                    date = rel.get("date") or ""
                    rows.append({"album": folder, "release_id": rel["id"], "title": rel.get("title", ""),
                                 "date": date, "year": date[:4], "country": rel.get("country") or "",
                                 "formats": "+".join(formats) or "Unknown",
                                 "packaging": rel.get("packaging") or "", "status": rel.get("status") or ""})
                offset += 100
                if offset >= data["release-count"]:
                    break
        with open(RELEASES_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["album", "release_id", "title"] + RELEASE_COLS)
            w.writeheader()
            w.writerows(rows)
    with open(RELEASES_PATH, newline="", encoding="utf-8") as f:
        return {r["release_id"]: r for r in csv.DictReader(f)}


# 3. Download one image, saved as RGB JPEG
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    albums = load_albums()
    picks = select_images({folder for folder, _ in albums})
    print(f"Selected {len(picks)} images from {RAW_PATH.name}")
    releases = load_releases(albums)

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download, p) for p in picks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="images", unit="img"):
            p, status = fut.result()
            if status == "failed":
                failed.append(p)

    # Manifest is rebuilt from disk every run, so it always matches the folder
    cols = ["album", "type", "release_id", "image_id", "tags", "candidates", "file", "sha256"] + RELEASE_COLS
    counts = defaultdict(Counter)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in picks:
            dest = dest_of(p)
            if not dest.exists():
                continue
            counts[p["album"]][p["type"]] += 1
            meta = releases.get(p["release_id"], {})
            w.writerow({**p, "file": dest.relative_to(ROOT / "data/project").as_posix(), "sha256": sha256(dest),
                        **{c: meta.get(c, "") for c in RELEASE_COLS}})

    print(f"\n{'album':<34}" + "".join(f"{t:>9}" for t in TYPES) + f"{'total':>9}")
    total = Counter()
    for album in sorted(counts):
        c = counts[album]
        total.update(c)
        print(f"{album[:33]:<34}" + "".join(f"{c[t]:>9}" for t in TYPES) + f"{sum(c.values()):>9}")
    print(f"{'TOTAL':<34}" + "".join(f"{total[t]:>9}" for t in TYPES) + f"{sum(total.values()):>9}")
    print(f"\nSaved {MANIFEST_PATH.relative_to(ROOT)}")
    if failed:
        print(f"\n{len(failed)} images failed (archive.org errors or missing thumbnails). Rerun to retry:")
        for p in failed[:20]:
            print(f"  {p['album']}  {p['type']}  {p['release_id']}")


if __name__ == "__main__":
    main()
