"""
Snapshot every Cover Art Archive image and its type tags for the albums in an album list.
Downloads no images: it reads only each release's JSON index on the Cover Art Archive.

One shared snapshot covers every album list (albums_10.csv is a subset of albums_all.csv):
  data/metadata/image_types_raw.csv    one row per image: album, release_id, image_id, types, approved.
                                       A release with no images gets one row with empty fields (marks it checked).
  data/metadata/snapshot_albums.csv    one row per finished album: album, rg_id, releases_with_art,
                                       snapshot_date. An album is finished only when every release succeeded.
Summaries, rebuilt from the whole snapshot on every run:
  data/metadata/image_types.csv          per album and type: images with that tag (any) / with only that tag
  data/metadata/image_types_by_album.csv per album: releases checked, single-tag Front/Back/Medium/Booklet counts
  console                                totals for the chosen album list (not one line per album)

Pinned snapshot rules
  - Without --resume nothing is fetched once image_types_raw.csv exists, so the dataset never changes silently.
  - With --resume, finished albums are skipped entirely (no new releases get added to them). Unfinished albums
    are listed on MusicBrainz and only releases missing from image_types_raw.csv are fetched.
  - A finished album whose rg_id differs from the album list stops the run: its rows belong to another release
    group. Fix it with --refetch, which removes the album's rows and fetches it again.
  - The original 10 albums predate snapshot_albums.csv. On first use they are recorded as finished with
    LEGACY_DATE, so they are never refetched.

Failures: releases that keep failing (archive.org 500s) are not written, so the next --resume retries them.
An album whose MusicBrainz release list keeps failing is skipped and stays unfinished, and the run carries on.
The full list goes to data/logs/03_failures.csv. Ctrl-C is safe: each release is written in one go.

Needs ALBUM_PROJECT_CONTACT set (see http_client.py) when anything is fetched.
Run from anywhere:
  python scripts/03_count_image_types.py                       summaries only
  python scripts/03_count_image_types.py --resume              fetch unfinished albums of albums_all.csv
  python scripts/03_count_image_types.py --resume --limit 3    only the first 3 unfinished albums
  python scripts/03_count_image_types.py --albums albums_10.csv
  python scripts/03_count_image_types.py --resume --refetch FOLDER [FOLDER ...]
                                         drop these albums from the snapshot and fetch them again (for an album
                                         whose rg_id was corrected in rg_overrides.csv). Never the 10 report albums.
"""
import argparse, csv, io, statistics, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import requests
from tqdm import tqdm

from http_client import get, user_agent

ROOT = Path(__file__).resolve().parents[1]   # repo root
META = ROOT / "data/metadata"
RAW_PATH = META / "image_types_raw.csv"
PROGRESS_PATH = META / "snapshot_albums.csv"
SCREEN_PATH = META / "screen.csv"            # release counts at screening, used only to size the progress bar
OUT_PATH = META / "image_types.csv"
BY_ALBUM_PATH = META / "image_types_by_album.csv"
FAIL_PATH = ROOT / "data/logs/03_failures.csv"
LEGACY_DATE = "2026-11-02"                   # snapshot date of the original 10 albums
CORE_TYPES = ["Front", "Back", "Medium", "Booklet", "Tray"]   # listed first in image_types.csv
WORKERS = 4        # parallel Cover Art Archive requests (MusicBrainz calls stay sequential at 1 per second)
RAW_COLS = ["album", "release_id", "image_id", "types", "approved"]
PROGRESS_COLS = ["album", "rg_id", "releases_with_art", "snapshot_date"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path, rows, cols):
    """Write to a .part file, then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def album_list(name):
    """(folder, rg_id) for each album in the list. A bare file name is looked up in data/metadata/."""
    path = Path(name) if Path(name).is_absolute() or Path(name).exists() else META / name
    return [(r["folder"], r["rg_id"]) for r in read_csv(path)], path


# Progress file: which albums are finished ---------------------------------------------------------
def load_progress():
    """album -> row. Creates the file on first use and records the albums already in the snapshot."""
    if PROGRESS_PATH.exists():
        return {r["album"]: r for r in read_csv(PROGRESS_PATH)}
    if not RAW_PATH.exists():
        return {}
    releases = defaultdict(set)
    for r in read_csv(RAW_PATH):
        releases[r["album"]].add(r["release_id"])
    with open(META / "albums_all.csv", newline="", encoding="utf-8") as f:
        rg_of = {r["folder"]: r["rg_id"] for r in csv.DictReader(f)}
    progress = {a: {"album": a, "rg_id": rg_of.get(a, ""), "releases_with_art": len(rels),
                    "snapshot_date": LEGACY_DATE} for a, rels in sorted(releases.items())}
    write_csv_atomic(PROGRESS_PATH, progress.values(), PROGRESS_COLS)
    print(f"Created {PROGRESS_PATH.name}: {len(progress)} albums already in the snapshot, dated {LEGACY_DATE}")
    return progress


def repair_tail():
    """If a hard kill cut the last line short, drop that line and any earlier lines of the same release,
    so the release counts as unchecked and gets fetched again."""
    if not RAW_PATH.exists():
        return
    data = RAW_PATH.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    lines = data.split(b"\n")
    partial = lines.pop().split(b",")
    rid = partial[1] if len(partial) > 1 and len(partial[1]) == 36 else None
    while rid and lines and lines[-1].split(b",")[1:2] == [rid]:
        lines.pop()
    RAW_PATH.write_bytes(b"\n".join(lines) + b"\n")
    print(f"Repaired a cut-off last line in {RAW_PATH.name}")


def drop_albums(folders):
    """Remove every snapshot row and the progress entry of these albums, so the next fetch starts clean."""
    with open(META / "albums_10.csv", newline="", encoding="utf-8") as f:
        protected = {r["folder"] for r in csv.DictReader(f)} & set(folders)
    if protected:
        raise SystemExit(f"Refusing to refetch albums of the 10-album dataset: {sorted(protected)}")
    rows = read_csv(RAW_PATH)
    kept = [r for r in rows if r["album"] not in folders]
    write_csv_atomic(RAW_PATH, kept, RAW_COLS)
    progress = {a: r for a, r in load_progress().items() if a not in folders}
    write_csv_atomic(PROGRESS_PATH, progress.values(), PROGRESS_COLS)
    print(f"Dropped {len(rows) - len(kept)} snapshot rows of {len(folders)} albums: {', '.join(folders)}")


def check_rg_ids(albums, progress):
    """Stop if a finished album was snapshotted under a different release group than the list now says."""
    stale = [a for a, rg in albums if a in progress and progress[a]["rg_id"] and progress[a]["rg_id"] != rg]
    if stale:
        raise SystemExit("These finished albums have a different rg_id in the album list than in the snapshot:\n  "
                         + "\n  ".join(stale) + "\nRerun with --resume --refetch " + " ".join(stale))


# Fetching ----------------------------------------------------------------------------------------
def releases_with_art(rg_id):
    """IDs of every release in the album that MusicBrainz says has any cover art (not just a front)."""
    ids, offset = [], 0
    while True:
        data = get("https://musicbrainz.org/ws/2/release",
                   {"release-group": rg_id, "limit": 100, "offset": offset, "fmt": "json"}).json()
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
        r = get(f"https://coverartarchive.org/release/{rid}", pause=0.5)
        return rid, (r.json() if r is not None else {}).get("images", [])
    except (requests.RequestException, ValueError):
        return rid, None


def release_lines(folder, rid, images):
    """All CSV lines for one release as a single string, so it is written in one call."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")   # the committed file uses LF
    for img in images:
        w.writerow([folder, rid, img["id"], ";".join(img.get("types") or []) or "Untyped", img.get("approved")])
    if not images:
        w.writerow([folder, rid, "", "", ""])        # marks the release as checked
    return buf.getvalue()


def collect(albums, limit, refetch=()):
    repair_tail()
    if refetch:
        unknown = set(refetch) - {a for a, _ in albums}
        if unknown:
            raise SystemExit(f"--refetch names albums not in the list: {sorted(unknown)}")
        drop_albums(list(refetch))
    progress = load_progress()
    check_rg_ids(albums, progress)
    todo_albums = [(a, rg) for a, rg in albums if a not in progress]
    if limit:
        todo_albums = todo_albums[:limit]
    print(f"{len(albums) - len([a for a, _ in albums if a not in progress])} of {len(albums)} albums already "
          f"finished. Fetching {len(todo_albums)}.")
    if not todo_albums:
        return

    user_agent()                                 # fail early if ALBUM_PROJECT_CONTACT is unset
    done = {r["release_id"] for r in read_csv(RAW_PATH)} if RAW_PATH.exists() else set()
    screen_n = {r["rg_id"]: int(r["releases"] or 0) for r in read_csv(SCREEN_PATH)}
    new_file = not RAW_PATH.exists()
    failed = []

    # Overall bar in releases, sized from the screening counts, so the ETA tracks real work across albums
    overall = tqdm(total=sum(screen_n.get(rg, 0) for _, rg in todo_albums), desc="overall", unit="rel",
                   position=0, dynamic_ncols=True)
    try:
        with open(RAW_PATH, "a", newline="", encoding="utf-8") as f:
            if new_file:
                f.write(",".join(RAW_COLS) + "\n")
            for i, (folder, rg_id) in enumerate(todo_albums, 1):
                overall.set_postfix_str(f"album {i}/{len(todo_albums)} {folder[:28]}, failed {len(failed)}")
                est = screen_n.get(rg_id, 0)
                try:
                    listed = releases_with_art(rg_id)
                except (requests.RequestException, ValueError, KeyError):
                    # MusicBrainz kept failing for this album: leave it unfinished and carry on.
                    # The next --resume lists it again.
                    failed.append({"album": folder, "release_id": "(MusicBrainz release list)"})
                    overall.update(est)
                    continue
                share = max(est, len(listed))
                overall.total += share - est
                todo = [rid for rid in listed if rid not in done]
                album_failed = 0
                pool = ThreadPoolExecutor(max_workers=WORKERS)
                try:
                    futures = [pool.submit(fetch_images, rid) for rid in todo]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  {folder[:30]}",
                                    unit="rel", position=1, leave=False, dynamic_ncols=True):
                        rid, images = fut.result()
                        overall.update(1)
                        if images is None:
                            failed.append({"album": folder, "release_id": rid})
                            album_failed += 1
                            continue
                        f.write(release_lines(folder, rid, images))
                        f.flush()
                        done.add(rid)
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
                overall.update(share - len(todo))   # releases without art or already in the snapshot
                if album_failed == 0:
                    progress[folder] = {"album": folder, "rg_id": rg_id, "releases_with_art": len(listed),
                                        "snapshot_date": date.today().isoformat()}
                    write_csv_atomic(PROGRESS_PATH, progress.values(), PROGRESS_COLS)
    except KeyboardInterrupt:
        overall.close()
        print("\nStopped. Everything written so far is kept. Rerun with --resume to continue.")
        write_failures(failed)
        sys.exit(130)
    overall.close()
    write_failures(failed)


def write_failures(failed):
    if not failed:
        return
    write_csv_atomic(FAIL_PATH, failed, ["album", "release_id"])
    print(f"\n{len(failed)} requests failed after retries (archive.org or MusicBrainz). Their albums stay unfinished. "
          f"Rerun with --resume to retry. Full list: {FAIL_PATH.relative_to(ROOT)}")
    for r in failed[:10]:
        print(f"  {r['album']}  {r['release_id']}")


# Summaries ---------------------------------------------------------------------------------------
def summarise(albums, list_path):
    """Per album and type: images carrying the tag (any) and images carrying only that tag (single)."""
    any_count, single_count = defaultdict(Counter), defaultdict(Counter)
    releases, img_total = defaultdict(set), Counter()
    for r in read_csv(RAW_PATH):
        releases[r["album"]].add(r["release_id"])
        if not r["image_id"]:
            continue
        img_total[r["album"]] += 1                   # every image, whatever its tags
        types = r["types"].split(";")
        for t in types:
            any_count[r["album"]][t] += 1
        if len(types) == 1:
            single_count[r["album"]][types[0]] += 1

    totals = Counter()
    for c in any_count.values():
        totals.update(c)
    others = sorted((t for t in totals if t not in CORE_TYPES), key=lambda t: (-totals[t], t))
    types = [t for t in CORE_TYPES if t in totals] + others

    rows = [{"album": a, "type": t, "images_any": any_count[a][t], "images_single": single_count[a][t]}
            for a in sorted(any_count) for t in types]
    write_csv_atomic(OUT_PATH, rows, ["album", "type", "images_any", "images_single"])
    core = CORE_TYPES[:4]
    by_album = [{"album": a, "releases_checked": len(releases[a]),
                 **{t.lower(): single_count[a][t] for t in core}, "images_total": img_total[a]}
                for a in sorted(releases)]
    write_csv_atomic(BY_ALBUM_PATH, by_album, ["album", "releases_checked"] + [t.lower() for t in core]
                     + ["images_total"])

    # Console: totals for the chosen list only
    names = [a for a, _ in albums]
    present = [a for a in names if a in releases]
    finished = {r["album"] for r in read_csv(PROGRESS_PATH)} if PROGRESS_PATH.exists() else set(releases)
    no_art = [a for a in names if a in finished and a not in releases]
    unfinished = [a for a in names if a not in finished]
    print(f"\nAlbum list {list_path.name}: {len(names) - len(unfinished)} of {len(names)} albums finished, "
          f"{len(present)} with images")
    if no_art:
        print(f"  Finished with no releases with artwork: {', '.join(no_art)}")
    if unfinished:
        print(f"  Not finished (rerun with --resume): {len(unfinished)}")
    if not present:
        return
    n_rel = [len(releases[a]) for a in present]
    print(f"Releases checked: {sum(n_rel)} (per album min {min(n_rel)}, median {statistics.median(n_rel):g}, "
          f"max {max(n_rel)})")
    print(f"Images: {sum(img_total[a] for a in present)}")
    print(f"\n{'type':<10}{'single tag':>12}{'any tag':>10}{'albums with 0 single':>22}")
    for t in core:
        single = sum(single_count[a][t] for a in present)
        anyt = sum(any_count[a][t] for a in present)
        zero = sum(1 for a in present if single_count[a][t] == 0)
        print(f"{t:<10}{single:>12}{anyt:>10}{zero:>22}")
    print(f"\nSaved {OUT_PATH.relative_to(ROOT)} and {BY_ALBUM_PATH.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--albums", default="albums_all.csv", help="album list in data/metadata/ (or a path)")
    ap.add_argument("--resume", action="store_true", help="fetch albums not yet finished in the snapshot")
    ap.add_argument("--limit", type=int, default=0, help="fetch at most this many unfinished albums")
    ap.add_argument("--refetch", nargs="+", default=[], metavar="FOLDER",
                    help="with --resume: drop these albums from the snapshot and fetch them again")
    args = ap.parse_args()
    albums, list_path = album_list(args.albums)

    if args.refetch and not args.resume:
        raise SystemExit("--refetch needs --resume")
    if RAW_PATH.exists() and not args.resume:
        # Keep the pinned snapshot: refetching would add releases uploaded since, and change the dataset
        print(f"{RAW_PATH.name} exists (pinned snapshot), so nothing is fetched. "
              "Use --resume to fetch albums that are not finished yet.")
    else:
        collect(albums, args.limit, args.refetch)
    summarise(albums, list_path)


if __name__ == "__main__":
    main()
