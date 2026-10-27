import requests, time, hashlib, csv, re, random
from pathlib import Path

HEADERS = {"User-Agent": "AlbumCoverProject/0.1 (akinbobola71@gmail.com)"}
CSV_PATH = "albums.csv"        # columns: artist,album (with a header row)
OUT_DIR = Path("covers")
SUMMARY_PATH = OUT_DIR / "summary.csv"
MAX_IMAGES = 40                # stop an album once it has this many saved images
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

    seen, rows = {}, []
    for rel in releases:
        if len(seen) >= MAX_IMAGES:
            break
        rid = rel["id"]
        row = {"release_id": rid, "date": rel.get("date", ""), "country": rel.get("country", ""),
               "has_front": False, "file": "", "duplicate_of": ""}
        caa = rel.get("cover-art-archive")
        if caa is None or caa.get("front"):
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
                        else:
                            (out / f"{rid}.jpg").write_bytes(img)
                            seen[h] = f"{rid}.jpg"
                            row["file"] = f"{rid}.jpg"
            except requests.RequestException as e:
                print(f"   skipped release {rid}: {e}")   # one bad image shouldn't kill the album
        rows.append(row)

    if rows:
        with open(out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    return len(releases), sum(r["has_front"] for r in rows), len(seen)

# ---- Load the album list ----
ALBUMS = []
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        artist, album = row["artist"].strip(), row["album"].strip()
        if artist and album:
            ALBUMS.append((artist, album))
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

# ---- Run every album ----
for artist, album in ALBUMS:
    key = (artist, album)
    if key in done and done[key]["status"] == "ok":
        print(f"Skipping (already done): {artist} - {album}")
        continue

    result = {"artist": artist, "album": album, "matched": "", "rg_id": "",
              "releases": 0, "with_front": 0, "saved": 0, "status": ""}
    try:
        rg = find_release_group(artist, album)
        if rg is None:
            result["status"] = "no match"
            print(f"NO MATCH: {artist} - {album}")
        else:
            matched = f'{rg["artist-credit"][0]["name"]} - {rg["title"]}'
            result["matched"], result["rg_id"] = matched, rg["id"]
            print(f"Matched '{artist} - {album}' -> {matched} (score {rg['score']})")
            total, with_front, saved = download_album(rg["id"], f"{slug(artist)}__{slug(album)}")
            result.update(releases=total, with_front=with_front, saved=saved, status="ok")
            print(f"   {total} releases, {with_front} with front art, {saved} saved")
    except Exception as e:
        result["status"] = f"error: {e}"
        print(f"ERROR on {artist} - {album}: {e}")

    done[key] = result
    save_summary()   # written after every album, so a crash loses nothing

# ---- Summary table ----
print("\n{:<40}{:>10}{:>12}{:>8}".format("Album", "Releases", "With front", "Saved"))
for r in done.values():
    print("{:<40}{:>10}{:>12}{:>8}  {}".format(
        f'{r["artist"]} - {r["album"]}'[:39], r["releases"], r["with_front"], r["saved"],
        "" if r["status"] == "ok" else r["status"]))