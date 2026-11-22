# TME 6015 final project: album packaging classifier

Recognise an album from any packaging image (front, back, disc, booklet). The main result is accuracy per
image type. Assignment 2 (the food image classifier) has its own repo, `tme6015-food-classifier`.

## Setup

```
uv sync
export ALBUM_PROJECT_CONTACT="you@example.com"     # MusicBrainz requires a contact in the User-Agent
```

`uv sync` creates `.venv` (Python 3.12) from `pyproject.toml` and `uv.lock`. Only network steps need
`ALBUM_PROJECT_CONTACT`. Scripts run from anywhere, for example `python scripts/04_download_project.py 10`.

## Layout

```
scripts/        data pipeline, run in order (see below). http_client.py holds the shared HTTP code.
data/
  metadata/     committed inputs and snapshots
                  albums.csv            Rolling Stone 500 candidates (467 after exclusions)
                  screen.csv            MusicBrainz screen: front-art releases per album
                  albums_10.csv         the 10 albums of the data report
                  albums_all.csv        the 430 screened albums (built by 02b_make_album_list.py)
                  image_types_raw.csv   Cover Art Archive images and type tags (one shared snapshot)
                  snapshot_albums.csv   albums finished in the snapshot, with the date of each
                  image_types.csv, image_types_by_album.csv   summaries of the snapshot
                  releases.csv          release metadata (one shared snapshot)
                  excluded_images.csv   picks kept out of every dataset
                  rg_overrides.csv      corrected release groups for albums the screen matched wrongly
  manifest_<name>.csv, split_<name>.csv   committed, one per dataset
  images/       <album>/<type>/<release_id>.jpg, shared by both datasets. Not in git.
  logs/         failure lists from download runs. Not in git.
  features_resnet50.npz   ResNet-50 features cached per image file. Not in git.
results/        generated outputs: pilot_results_<name>.csv, figures/, model metrics later
notebooks/      one notebook per model (added with the first model)
reports/
  data_report/  submitted data report: LaTeX source, bibliography, figures, PDF. Frozen.
  final_report/ later
```

## Two datasets

| name  | album list       | albums | manifest              | snapshot date of its albums |
|-------|------------------|--------|-----------------------|-----------------------------|
| `10`  | `albums_10.csv`  | 10     | `manifest_10.csv`     | November 2, 2026            |
| `all` | `albums_all.csv` | 430    | `manifest_all.csv`    | 10 albums as above, 420 later (per album in `snapshot_albums.csv`) |

`10` is the dataset in the submitted data report (2,223 images, 899 releases). `all` is every album in
`screen.csv` with status `ok` and `title_ok` True, with no size filter. Any cutoff happens at training time.
`all` contains `10` unchanged: both share one snapshot, one image folder and one release-metadata file, and the
image pick and the split are seeded per release and per album, so the 10 albums get the same images and the same
split in both datasets.

The new albums were snapshotted later than the original 10. The Cover Art Archive grows over time, so an album
snapshotted later can hold uploads the original 10 never saw. `snapshot_albums.csv` records the date of each album.

`rg_overrides.csv` corrects 3 albums where `01_screen_albums.py` matched a stray release group instead of the
album: The Beatles (White Album), Who's Next and Reach Out. An audit of all 430 albums compared each release
group's first release year with the Rolling Stone year. Only these 3 were wrong matches. Six more differ by one year
because MusicBrainz records the earliest release anywhere (a Japanese edition, a promo). The audit cannot catch a
wrong group from the same year.

`excluded_images.csv` lists 8 images of the 10 albums that failed to download in the original build. They stay
out of both datasets, so `10` matches the report exactly.

## Reproduce a dataset

```
python scripts/04_download_project.py 10    # or: all
```

A fresh clone needs only this. The script reads the committed snapshots, so it picks the same images on every
machine. Compare the `sha256` column of the manifest to confirm. Images deleted from the Cover Art Archive since
are the only source of difference.

## Full pipeline

1. `01_screen_albums.py`: albums.csv to screen.csv (front-art count per album)
2. `02_download_covers.py`: front covers of the top 20 candidates, used to choose the 10 albums (Assignment 2 era).
   Its output folder was deleted. The script stays as a record of how the 10 were chosen.
3. `02b_make_album_list.py`: screen.csv plus rg_overrides.csv to albums_all.csv. Checks folder and release-group
   collisions and that albums_10.csv is an identical subset.
4. `03_count_image_types.py [--albums albums_all.csv] [--resume] [--limit N]`: snapshot every image and type tag.
   Without `--resume` it fetches nothing once the snapshot exists. With `--resume` it skips finished albums
   entirely and fetches only missing releases of unfinished ones. If an album's rg_id in the list no longer matches
   the snapshot (after an override), it stops and names the album. `--refetch FOLDER ...` drops that album's rows
   and fetches it again. It refuses the 10 report albums.
5. `04_download_project.py {10,all}`: one image per type per release into data/images/, writes the manifest.
   Fetches release metadata only for albums missing from releases.csv. Reuses images on disk.
6. `05_pilot_learning_curve.py {10,all}`: release-level split, pilot learning curve into results/.

Steps 1 to 6 have run for `10`. To build `all`:

```
python scripts/03_count_image_types.py --resume      # several hours, resumable
python scripts/04_download_project.py all            # several hours, resumable
python scripts/05_pilot_learning_curve.py all
```

Both downloads respect MusicBrainz (about 1 request per second) and use at most 4 parallel Cover Art Archive
requests. archive.org returns transient 500s: failed items are retried with backoff, then skipped and listed in
`data/logs/`. Rerun the same command to retry them. Ctrl-C is safe at any point.
