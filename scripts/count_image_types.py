"""
Count the image types the Cover Art Archive holds for the 10 albums in data/assignment2/dataset/.
Downloads no images: it reads only each release's JSON index on the Cover Art Archive.

Answers two questions:
  1. Which image types exist (Front, Back, Medium, Booklet, Tray, ...) and how common each is.
  2. How many non-front images each album has (is "identify the album from any image" viable?).

Output (data/metadata/):
  image_types_raw.csv  one row per image: album, release_id, image_id, types, approved
  image_types.csv      per album and type: images tagged with that type / tagged with that type only

Resumable: releases already in image_types_raw.csv are skipped on a rerun.
Run from anywhere:  python scripts/count_image_types.py   (roughly 40 to 60 minutes for 10 albums)
"""
import csv, re, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # project folder
HEADERS = {"User-Agent": "AlbumCoverProject/0.1 (akinbobola71@gmail.com)"}
SUMMARY_PATH = ROOT / "data/raw/covers/summary.csv"   # has the release-group ID (rg_id) of each downloaded album
DATASET_DIR = ROOT / "data/assignment2/dataset"               # only albums with a folder here are counted
RAW_PATH = ROOT / "data/metadata/image_types_raw.csv"
OUT_PATH = ROOT / "data/metadata/image_types.csv"
CORE_TYPES = ["Front", "Back", "Medium", "Booklet", "Tray"]   # printed first in the table
WORKERS = 4        # parallel Cover Art Archive requests (MusicBrainz calls stay sequential at 1 per second)

_local = threading.local()


def session():
    # One requests.Session per thread, since sessions are not guaranteed thread-safe
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def slug(text):
    # Same rule as download_covers.py, so folder names match
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def get_json(url, params=None, pause=1.1, retries=4):
    """GET with a polite pause. Returns None on 404 (release has no art). Retries 5xx and timeouts."""
    for attempt in range(retries):
        time.sleep(pause)
        try:
            r = session().get(url, params=params, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def album_list():
    """(folder, rg_id) for every album in summary.csv that is also in dataset/."""
    keep = {p.name for p in DATASET_DIR.iterdir() if p.is_dir()}
    with open(SUMMARY_PATH, newline="", encoding="utf-8") as f:
        rows = [(f"{slug(r['artist'])}__{slug(r['album'])}", r["rg_id"]) for r in csv.DictReader(f)]
    albums = [(folder, rg) for folder, rg in rows if folder in keep]
    missing = keep - {folder for folder, _ in albums}
    if missing:
        print("Warning: no rg_id found for", sorted(missing))
    return albums


def releases_with_art(rg_id):
    """IDs of every release in the album that MusicBrainz says has any cover art (not just a front)."""
    ids, offset = [], 0
    while True:
        data = get_json("https://musicbrainz.org/ws/2/release",
                        {"release-group": rg_id, "limit": 100, "offset": offset, "fmt": "json"})
        for rel in data["releases"]:
            caa = rel.get("cover-art-archive")
            if caa is None or caa.get("artwork"):
                ids.append(rel["id"])
        offset += 100
        if offset >= data["release-count"]:
            return ids


def fetch_images(rid):
    """Image list for one release. Returns (rid, images) or (rid, None) if archive.org keeps failing."""
    try:
        data = get_json(f"https://coverartarchive.org/release/{rid}", pause=0.5)
        return rid, (data or {}).get("images", [])
    except requests.RequestException:
        return rid, None


def collect():
    """Fetch the Cover Art Archive index for each release and append one row per image.
    Releases that keep failing (archive.org 500s) are skipped and not written, so a rerun retries them."""
    done = set()
    if RAW_PATH.exists():
        with open(RAW_PATH, newline="", encoding="utf-8") as f:
            done = {r["release_id"] for r in csv.DictReader(f)}
    new_file = not RAW_PATH.exists()
    failed = []

    with open(RAW_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["album", "release_id", "image_id", "types", "approved"])
        for folder, rg_id in tqdm(album_list(), desc="albums", unit="album", position=0):
            todo = [rid for rid in releases_with_art(rg_id) if rid not in done]
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(fetch_images, rid) for rid in todo]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"  {folder[:30]}", unit="rel", position=1, leave=False):
                    rid, images = fut.result()
                    if images is None:
                        failed.append((folder, rid))
                        continue
                    for img in images:
                        types = ";".join(img.get("types") or []) or "Untyped"
                        w.writerow([folder, rid, img["id"], types, img.get("approved")])
                    if not images:
                        w.writerow([folder, rid, "", "", ""])   # marks the release as checked
                    f.flush()

    if failed:
        print(f"\n{len(failed)} releases failed after retries (archive.org errors). Rerun to retry them:")
        for folder, rid in failed[:20]:
            print(f"  {folder}  {rid}")


def summarise():
    """Per album and type: images carrying the tag (any) and images carrying only that tag (single)."""
    any_count, single_count = defaultdict(Counter), defaultdict(Counter)
    releases = defaultdict(set)
    with open(RAW_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            releases[r["album"]].add(r["release_id"])
            if not r["image_id"]:
                continue
            types = r["types"].split(";")
            for t in types:
                any_count[r["album"]][t] += 1
            if len(types) == 1:
                single_count[r["album"]][types[0]] += 1

    totals = Counter()
    for c in any_count.values():
        totals.update(c)
    others = sorted((t for t in totals if t not in CORE_TYPES), key=lambda t: -totals[t])
    types = [t for t in CORE_TYPES if t in totals] + others

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["album", "type", "images_any", "images_single"])
        for album in sorted(any_count):
            for t in types:
                w.writerow([album, t, any_count[album][t], single_count[album][t]])

    # Console table: single-type counts, which is what a training set could actually use
    print("\nSingle-type images per album (releases checked in brackets)")
    print(f"{'album':<22}" + "".join(f"{t[:8]:>9}" for t in types))
    col_sum = Counter()
    for album in sorted(any_count):
        name = f"{album.split('__')[1][:14]} ({len(releases[album])})"
        print(f"{name:<22}" + "".join(f"{single_count[album][t]:>9}" for t in types))
        col_sum.update(single_count[album])
    print(f"{'TOTAL':<22}" + "".join(f"{col_sum[t]:>9}" for t in types))
    print(f"{'(any tag)':<22}" + "".join(f"{totals[t]:>9}" for t in types))
    print(f"\nSaved {OUT_PATH} and {RAW_PATH}")


if __name__ == "__main__":
    collect()
    summarise()
