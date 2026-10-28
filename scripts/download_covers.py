"""
Step 2: download front covers for albums that passed the screen.
Reads screen.csv (made by screen_albums.py). Falls back to albums.csv lookup if screen.csv is missing.
Output: covers/<artist>__<album>/<release-id>.jpg, per-album manifest.csv, covers/summary.csv

Needs tqdm for the progress bars:  pip install tqdm
Run from the folder that contains screen.csv:  python download_covers.py
"""
import requests, time, hashlib, csv, re, random
from pathlib import Path
from tqdm import tqdm

HEADERS = {"User-Agent": "AlbumCoverProject/0.1 (akinbobola71@gmail.com)"}
CSV_PATH = "../albums.csv"  # fallback: columns artist,album (with a header row)
SCREEN_PATH = Path("../screen.csv")   # output of screen_albums.py
MIN_FRONT = 50                     # keep albums with at least this many front-art releases
TOP_N = 20                         # download at most this many albums (None = every album that passes)
MAX_PER_ARTIST = 1                 # at most this many albums per artist (None = no limit)
OUT_DIR = Path("../covers")
SUMMARY_PATH = OUT_DIR / "summary.csv"
MAX_IMAGES = 40                    # stop an album once it has this many saved images
SEED = 42

OUT_DIR.mkdir(exist_ok=True)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def mb_get(url, params, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(1.1)    # MusicBrainz rate limit: about 1 request per second
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5)


def find_release_group(artist, album):
    q = f'releasegroup:"{album}" AND artist:"{artist}" AND primarytype:album'
    data = mb_get("https://musicbrainz.org/ws/2/release-group",
                  {"query": q, "limit": 10, "fmt": "json"})
    for rg in data["release-groups"]:
        if not rg.get("secondary-types"):   # skips live, compilation, remix, etc.
            return rg
    return None


def download_album(rg_id, folder):
    out = OUT_DIR / folder
    out.mkdir(parents=True, exist_ok=True)
    # Clear leftovers from an earlier failed run so the folder matches the new manifest
    for old in list(out.glob("*.jpg")) + [out / "manifest.csv"]:
        old.unlink(missing_ok=True)

    releases, offset = [], 0
    while True:
        data = mb_get("https://musicbrainz.org/ws/2/release",
                      {"release-group": rg_id, "limit": 100, "offset": offset, "fmt": "json"})
        releases += data["releases"]
        offset += 100
        if offset >= data["release-count"]:
            break
    # Sort first (API order isn't guaranteed), then shuffle with a per-album seed,
    # so each album's sample is the same no matter which albums ran before it
    releases.sort(key=lambda rel: rel["id"])
    random.Random(f"{SEED}-{rg_id}").shuffle(releases)

    # Releases worth checking: MusicBrainz says they have front art (or doesn't say)
    candidates = [rel for rel in releases
                  if rel.get("cover-art-archive") is None or rel["cover-art-archive"].get("front")]
    # Inner bar counts saved images toward the cap; checked/dupes/failed show beside it
    bar = tqdm(total=min(MAX_IMAGES, len(candidates)), desc="  images", unit="img",
               position=1, leave=False)
    checked = dupes = failed = 0

    seen, rows = {}, []
    for rel in releases:
        if len(seen) >= MAX_IMAGES:
            break
        rid = rel["id"]
        row = {"release_id": rid, "date": rel.get("date", ""), "country": rel.get("country", ""),
               "has_front": False, "file": "", "duplicate_of": ""}
        caa = rel.get("cover-art-archive")
        if caa is None or caa.get("front"):
            checked += 1
            try:
                time.sleep(1.1)
                r = requests.get(f"https://coverartarchive.org/release/{rid}",
                                 headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    front = next((i for i in r.json()["images"] if i.get("front")), None)
                    if front:
                        url = front["thumbnails"].get("500") or front["image"]
                        img_resp = requests.get(url, headers=HEADERS, timeout=60)
                        img_resp.raise_for_status()   # don't save an error page as a .jpg
                        img = img_resp.content
                        h = hashlib.sha256(img).hexdigest()
                        row["has_front"] = True
                        if h in seen:
                            row["duplicate_of"] = seen[h]
                            dupes += 1
                        else:
                            (out / f"{rid}.jpg").write_bytes(img)
                            seen[h] = f"{rid}.jpg"
                            row["file"] = f"{rid}.jpg"
                            bar.update(1)
            except requests.RequestException as e:
                failed += 1
                tqdm.write(f"   skipped release {rid}: {e}")   # one bad image shouldn't kill the album
            bar.set_postfix(checked=f"{checked}/{len(candidates)}", dupes=dupes, failed=failed)
        rows.append(row)
    bar.close()

    if rows:
        with open(out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    return len(releases), sum(r["has_front"] for r in rows), len(seen)


# ---- Load the album list: (artist, album, rg_id or None) ----
ALBUMS = []
if SCREEN_PATH.exists():
    with open(SCREEN_PATH, newline="", encoding="utf-8-sig") as f:
        passed = [r for r in csv.DictReader(f)
                  if r["status"] == "ok"
                  and r["title_ok"].strip().lower() == "true"
                  and int(r["front_art"] or 0) >= MIN_FRONT]
    passed.sort(key=lambda r: int(r["front_art"]), reverse=True)   # best albums first
    picked, per_artist = [], {}
    for r in passed:
        if MAX_PER_ARTIST and per_artist.get(r["artist"], 0) >= MAX_PER_ARTIST:
            continue                                               # artist already at its limit
        per_artist[r["artist"]] = per_artist.get(r["artist"], 0) + 1
        picked.append(r)
    if TOP_N:
        picked = picked[:TOP_N]
    ALBUMS = [(r["artist"], r["album"], r["rg_id"]) for r in picked]
    print(f"{len(passed)} albums have front_art >= {MIN_FRONT}; "
          f"downloading {len(ALBUMS)} after the per-artist limit and top-N")
else:
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            artist, album = row["artist"].strip(), row["album"].strip()
            if artist and album:
                ALBUMS.append((artist, album, None))
    print(f"Loaded {len(ALBUMS)} albums from {CSV_PATH}")

# ---- Load previous results so finished albums are skipped on rerun ----
FIELDS = ["artist", "album", "matched", "rg_id", "releases", "with_front", "saved", "status"]
done = {}
if SUMMARY_PATH.exists():
    with open(SUMMARY_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done[(r["artist"], r["album"])] = r


def save_summary():
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(done.values())


# Only the albums still to do, so the progress bar and its ETA reflect real remaining work
pending = [a for a in ALBUMS if not (done.get((a[0], a[1]), {}).get("status") == "ok")]
print(f"{len(ALBUMS) - len(pending)} already done, {len(pending)} to download\n")

# ---- Run every pending album ----
albums_bar = tqdm(pending, desc="Albums", unit="album", position=0)
for artist, album, rg_id in albums_bar:
    key = (artist, album)
    albums_bar.set_postfix_str(f"{artist} - {album}"[:45])

    result = {"artist": artist, "album": album, "matched": "", "rg_id": "",
              "releases": 0, "with_front": 0, "saved": 0, "status": ""}
    try:
        if rg_id is None:                       # fallback mode: look the album up
            rg = find_release_group(artist, album)
            if rg is None:
                result["status"] = "no match"
                tqdm.write(f"NO MATCH: {artist} - {album}")
                done[key] = result
                save_summary()
                continue
            rg_id = rg["id"]
            matched = f'{rg["artist-credit"][0]["name"]} - {rg["title"]}'
            tqdm.write(f"Matched '{artist} - {album}' -> {matched} (score {rg['score']})")
        else:                                   # screened mode: ID already verified
            matched = f"{artist} - {album}"
        result["matched"], result["rg_id"] = matched, rg_id
        total, with_front, saved = download_album(rg_id, f"{slug(artist)}__{slug(album)}")
        result.update(releases=total, with_front=with_front, saved=saved, status="ok")
        tqdm.write(f"{matched}: {total} releases, {with_front} with front art, {saved} saved")
    except Exception as e:
        result["status"] = f"error: {e}"
        tqdm.write(f"ERROR on {artist} - {album}: {e}")

    done[key] = result
    save_summary()   # written after every album, so a crash loses nothing
albums_bar.close()

# ---- Summary table ----
print("\n{:<40}{:>10}{:>12}{:>8}".format("Album", "Releases", "With front", "Saved"))
for r in done.values():
    print("{:<40}{:>10}{:>12}{:>8}  {}".format(
        f'{r["artist"]} - {r["album"]}'[:39], r["releases"], r["with_front"], r["saved"],
        "" if r["status"] == "ok" else r["status"]))