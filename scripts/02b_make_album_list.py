"""
Build data/metadata/albums_all.csv from screen.csv: every album with status "ok" and title_ok True.

No size filter: albums with few releases stay in. Any cutoff happens later, at training time.
Folder names follow the same rule as before, slug(artist) + "__" + slug(album), using the artist and album
spelling from albums.csv (not the MusicBrainz match), so the 10 albums in albums_10.csv keep their folders.

Checks, and stops with an error if one fails:
  - no two albums share a folder name or a release-group ID
  - every album in albums_10.csv appears in albums_all.csv with the same folder and rg_id

Output: data/metadata/albums_all.csv (artist, album, folder, rg_id), in Rolling Stone rank order.
Reads only local files.  Run from anywhere:  python scripts/02b_make_album_list.py
"""
import csv, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
SCREEN_PATH = ROOT / "data/metadata/screen.csv"
TEN_PATH = ROOT / "data/metadata/albums_10.csv"
OUT_PATH = ROOT / "data/metadata/albums_all.csv"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main():
    with open(SCREEN_PATH, newline="", encoding="utf-8") as f:
        screen = [r for r in csv.DictReader(f) if r["status"] == "ok" and r["title_ok"] == "True"]
    screen.sort(key=lambda r: int(r["rank"]))
    rows = [{"artist": r["artist"], "album": r["album"],
             "folder": f"{slug(r['artist'])}__{slug(r['album'])}", "rg_id": r["rg_id"]} for r in screen]

    # Collisions would merge two albums into one class (folder) or duplicate one album as two classes (rg_id)
    for key in ("folder", "rg_id"):
        dupes = [v for v, n in Counter(r[key] for r in rows).items() if n > 1]
        if dupes:
            raise SystemExit(f"Duplicate {key}: {dupes}")

    # The 10-album dataset must be a strict subset with identical names
    by_rg = {r["rg_id"]: r for r in rows}
    with open(TEN_PATH, newline="", encoding="utf-8") as f:
        for t in csv.DictReader(f):
            match = by_rg.get(t["rg_id"])
            if match is None or match["folder"] != t["folder"]:
                raise SystemExit(f"{t['folder']} from {TEN_PATH.name} has no identical entry in the new list")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["artist", "album", "folder", "rg_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}: {len(rows)} albums, "
          f"{sum(int(r['releases']) for r in screen)} releases listed at screening")


if __name__ == "__main__":
    main()
