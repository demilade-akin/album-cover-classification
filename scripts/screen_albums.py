"""
Step 1: screen every album in albums.csv.
Looks each album up on MusicBrainz and counts releases with front cover art.
No images are downloaded. Results go to data/metadata/screen.csv (resumes if interrupted).

Run from anywhere:  python scripts/screen_albums.py
"""
import requests, time, csv, re
from pathlib import Path

HEADERS = {"User-Agent": "AlbumCoverProject/0.1 (akinbobola71@gmail.com)"}
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # project folder
CSV_PATH = ROOT / "data/metadata/albums.csv"
OUT_PATH = ROOT / "data/metadata/screen.csv"
FIELDS = ["rank", "year", "artist", "album", "matched_title", "matched_artist",
          "title_ok", "rg_id", "releases", "front_art", "status"]


def mb_get(url, params, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(1.1)   # MusicBrainz rate limit: about 1 request per second
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5)


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# Load albums and any previous results
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    albums = list(csv.DictReader(f))

done = {}
if OUT_PATH.exists():
    with open(OUT_PATH, newline="", encoding="utf-8") as f:
        done = {r["rank"]: r for r in csv.DictReader(f) if r["status"] == "ok"}


def save():
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(done.values())


for i, a in enumerate(albums, 1):
    if a["rank"] in done:
        continue
    row = {k: "" for k in FIELDS}
    row.update(rank=a["rank"], year=a["year"], artist=a["artist"], album=a["album"])
    try:
        q = f'releasegroup:"{a["album"]}" AND artist:"{a["artist"]}" AND primarytype:album'
        hits = mb_get("https://musicbrainz.org/ws/2/release-group",
                      {"query": q, "limit": 10, "fmt": "json"})["release-groups"]
        rg = next((h for h in hits if not h.get("secondary-types")), None)
        if rg is None:
            row["status"] = "no match"
        else:
            row.update(matched_title=rg["title"],
                       matched_artist=rg["artist-credit"][0]["name"],
                       title_ok=norm(rg["title"]) == norm(a["album"]),
                       rg_id=rg["id"])
            data = mb_get("https://musicbrainz.org/ws/2/release",
                          {"release-group": rg["id"], "limit": 100, "fmt": "json"})
            row["releases"] = data["release-count"]
            row["front_art"] = sum(1 for r in data["releases"]
                                   if (r.get("cover-art-archive") or {}).get("front"))
            row["status"] = "ok"
    except Exception as e:
        row["status"] = f"error: {e}"
    done[a["rank"]] = row
    save()   # written after every album, so a crash loses nothing
    print(f'[{i}/{len(albums)}] {a["artist"]} - {a["album"]}: '
          f'{row["front_art"]} front / {row["releases"]} releases ({row["status"]})')

print("\nDone. Open screen.csv and sort by front_art.")