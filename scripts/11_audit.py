"""
Downstream audit: run both models' test outputs as an upload checker and list suspicious uploads.

  python scripts/11_audit.py            keeps hand_check.csv if it already holds a verdict or note
  python scripts/11_audit.py --force    overwrites hand_check.csv even then

Inputs (all committed, no images or feature cache needed except for the thumbnails in the HTML sheet)
  results/part_classifier/predictions.csv   part classifier on every test image
  results/verifier/checks_test.csv          verifier checks on test albums (genuine checks, trained model)
  results/verifier/metrics.json             the trained verifier's threshold, chosen on validation albums
  data/manifest_all.csv, data/metadata/albums_all.csv, data/metadata/releases.csv

Two flag types
  part   the predicted part differs from the uploader's tag. Confidence = predicted probability of that part.
  album  a genuine check scores below the threshold: the image does not match the other releases of its own
         album. Confidence = threshold minus score (how far below the threshold).
Each type is ranked by confidence, most confident first.

Output (results/audit/)
  flags.csv          every flag
  hand_check.csv     top 50 part flags and top 20 album flags, with empty verdict and note columns
                     (verdict: real mislabel, model error or unclear). Not overwritten if any verdict or
                     note is filled in, unless --force.
  hand_check.html    the same flags with image thumbnails, for judging by eye. Thumbnails point at
                     data/images/, so the page needs the images on disk (rebuild with 04_download_project.py all).
                     Always regenerated. Its text uses the report's terms ("type", "disc"); the CSV columns keep
                     the code's terms ("part", "medium").
"""
import argparse, csv, html, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "audit"
TOP_PART, TOP_ALBUM = 50, 20
PARTS = ["front", "back", "medium", "booklet"]
REF_IMAGES = 3            # reference images of the album shown next to each album flag

COLS = ["flag_type", "rank", "confidence", "file", "album", "release_id", "uploader_tag", "pred_part",
        "part_confidence", "album_score", "threshold"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})


def build_flags():
    preds = read_csv(RES / "part_classifier" / "predictions.csv")
    checks = [c for c in read_csv(RES / "verifier" / "checks_test.csv") if c["check_type"] == "genuine"]
    thr = json.load(open(RES / "verifier" / "metrics.json"))["settings"]["trained"]["threshold"]
    pred_of = {p["file"]: p for p in preds}
    score_of = {c["query_file"]: float(c["score_trained"]) for c in checks}

    def base(p, flag_type, confidence):
        return {"flag_type": flag_type, "confidence": confidence, "file": p["file"], "album": p["album"],
                "release_id": p["release_id"], "uploader_tag": p["true_part"], "pred_part": p["pred_part"],
                "part_confidence": float(p[f"p_{p['pred_part']}"]),
                "album_score": score_of.get(p["file"], ""), "threshold": thr}

    part = [base(p, "part", float(p[f"p_{p['pred_part']}"])) for p in preds if p["pred_part"] != p["true_part"]]
    album = [base(pred_of[f], "album", thr - s) for f, s in score_of.items() if s < thr]
    for flags in (part, album):
        flags.sort(key=lambda r: -r["confidence"])
        for k, r in enumerate(flags, 1):
            r["rank"] = k
    return preds, checks, part, album, thr


def album_info():
    """Per album folder: display name and MusicBrainz release-group ID."""
    rows = read_csv(DATA / "metadata" / "albums_all.csv")
    return ({r["folder"]: f"{r['artist']}: {r['album']}" for r in rows},
            {r["folder"]: r["rg_id"] for r in rows})


def release_details():
    """Per release ID: year, country and format joined with " · ", skipping empty fields."""
    return {r["release_id"]: " · ".join(v for v in (r["year"], r["country"], r["formats"]) if v.strip())
            for r in read_csv(DATA / "metadata" / "releases.csv")}


def term(part):
    """The report says "disc" where the data says "medium"."""
    return "disc" if part == "medium" else part


def has_verdicts(path):
    """True if an existing hand-check sheet holds any non-empty verdict or note."""
    if not path.exists():
        return False
    return any((r.get("verdict") or "").strip() or (r.get("note") or "").strip() for r in read_csv(path))


def reference_images(rows):
    """Per album, a few front images from different releases, to compare an album flag against."""
    fronts = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["release_id"]):
        if r["type"] == "front":
            fronts[r["album"]].append(r)
    return fronts


def write_html(sheet, names, rg_ids, details, fronts, path):
    def img(file, size=160):
        src = html.escape(f"../../data/{file}")
        return f'<a href="{src}" target="_blank"><img src="{src}" loading="lazy" style="max-width:{size}px;max-height:{size}px"></a>'

    rows_html = []
    for i, r in enumerate(sheet, 1):
        if r["flag_type"] == "part":
            what = (f"Uploader tag <b>{term(r['uploader_tag'])}</b><br>Predicted type <b>{term(r['pred_part'])}</b> "
                    f"({r['part_confidence']:.1%})")
            refs = ""
        else:
            what = (f"Uploader tag <b>{term(r['uploader_tag'])}</b><br>Album score <b>{r['album_score']:.3f}</b> "
                    f"&lt; threshold {r['threshold']:.3f}<br>Predicted type {term(r['pred_part'])} "
                    f"({r['part_confidence']:.1%})")
            others = [f for f in fronts[r["album"]] if f["release_id"] != r["release_id"]][:REF_IMAGES]
            refs = "".join(img(f["file"], 90) for f in others)
        label = "type" if r["flag_type"] == "part" else r["flag_type"]
        detail = details.get(r["release_id"], "")
        detail = f"<br><span class=m>{html.escape(detail, quote=False)}</span>" if detail else ""
        links = (f'<br><a href="https://musicbrainz.org/release/{r["release_id"]}/cover-art" target="_blank">'
                 f'Release images ↗</a><br><a href="https://musicbrainz.org/release-group/{rg_ids[r["album"]]}" '
                 f'target="_blank">Album on MusicBrainz ↗</a>')
        rows_html.append(
            f"<tr><td>{i}</td><td>{label} #{r['rank']}</td><td>{img(r['file'])}</td>"
            f"<td><b>{html.escape(names.get(r['album'], r['album']))}</b><br>"
            f"<span class=m>release {r['release_id']}</span>{detail}{links}</td><td>{what}</td><td>{refs}</td>"
            f"<td class=v>real mislabel / model error / unclear</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Audit hand-check</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #ddd;padding:8px;vertical-align:top;text-align:left;font-size:14px}}
th{{background:#f4f4f4;position:sticky;top:0}} .m{{color:#777;font-size:12px}} .v{{color:#999;font-size:12px}}
</style></head><body>
<h1>Audit hand-check: top {TOP_PART} type flags and top {TOP_ALBUM} album flags (test albums)</h1>
<p>Record each verdict in <code>results/audit/hand_check.csv</code>. Click a thumbnail for the full image. The Album column links to all images of that release and to the album on MusicBrainz.
For album flags, the right-hand column shows front covers of other releases of the claimed album.</p>
<table><tr><th>#</th><th>Flag</th><th>Image</th><th>Album</th><th>Flag details</th><th>Album reference</th><th>Verdict</th></tr>
{''.join(rows_html)}
</table></body></html>"""
    path.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite hand_check.csv even if it holds verdicts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    preds, checks, part, album, thr = build_flags()
    write_csv(OUT / "flags.csv", part + album, COLS)

    sheet = part[:TOP_PART] + album[:TOP_ALBUM]
    for r in sheet:
        r["verdict"], r["note"] = "", ""
    hand_check = OUT / "hand_check.csv"
    if has_verdicts(hand_check) and not args.force:
        print(f"WARNING: {hand_check.relative_to(ROOT)} already holds verdicts or notes. Not overwritten "
              f"(use --force to overwrite). The HTML page is still regenerated.\n")
    else:
        write_csv(hand_check, sheet, COLS + ["verdict", "note"])
    rows = read_csv(DATA / "manifest_all.csv")
    names, rg_ids = album_info()
    write_html(sheet, names, rg_ids, release_details(), reference_images(rows), OUT / "hand_check.html")

    flagged = {r["file"] for r in part + album}
    both = {r["file"] for r in part} & {r["file"] for r in album}
    n = len(preds)
    print(f"Images checked: {n} (test albums). Genuine verifier checks: {len(checks)}. "
          f"Verifier threshold: {thr:.4f}")
    print(f"Part flags : {len(part):>5} ({len(part) / n:.1%} of images)")
    print(f"Album flags: {len(album):>5} ({len(album) / len(checks):.1%} of genuine checks)")
    print(f"Flagged by both: {len(both)}. Images with at least one flag: {len(flagged)} ({len(flagged) / n:.1%})")
    print("\nPart flags by uploader tag -> predicted part:")
    pairs = defaultdict(int)
    for r in part:
        pairs[(r["uploader_tag"], r["pred_part"])] += 1
    for (t, p), c in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<8} -> {p:<8} {c:>4}")
    print("\nAlbum flags by uploader tag:", {p: sum(r["uploader_tag"] == p for r in album) for p in PARTS})
    print(f"\nSaved to {OUT.relative_to(ROOT)}/ (hand-check sheet: {TOP_PART} part + {TOP_ALBUM} album flags)")


if __name__ == "__main__":
    main()
