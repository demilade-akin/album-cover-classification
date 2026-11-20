# Final project: album packaging classifier

Recognise an album from any packaging image (front, back, disc, booklet). The main result is accuracy per
image type. Paths below are relative to this `project/` folder.

## Two datasets

| name  | album list                      | albums | manifest                 | snapshot date of its albums |
|-------|---------------------------------|--------|--------------------------|-----------------------------|
| `10`  | `data/metadata/albums_10.csv`   | 10     | `data/manifest_10.csv`   | November 2, 2026            |
| `all` | `data/metadata/albums_all.csv`  | 430    | `data/manifest_all.csv`  | 10 albums as above, 420 later (per album in `snapshot_albums.csv`) |

`10` is the dataset in the submitted data report (2,223 images, 899 releases). `all` is every album in
`screen.csv` with status `ok` and `title_ok` True, with no size filter. Any cutoff happens at training time.
`all` contains `10` unchanged: both share one snapshot, one image folder and one release-metadata file, and
the image pick and the split are seeded per release and per album, so the 10 albums get the same images and the
same split in both datasets.

The new albums were snapshotted later than the original 10. The Cover Art Archive grows over time, so an album
snapshotted later can hold uploads the original 10 never saw. `snapshot_albums.csv` records the date of each album.

## Reproduce a dataset

```
uv sync
export ALBUM_PROJECT_CONTACT="you@example.com"     # MusicBrainz requires a contact in the User-Agent
python project/scripts/04_download_project.py 10    # or: all
```

A fresh clone needs only this. The script reads the committed snapshots, `data/metadata/image_types_raw.csv`
(which images exist) and `data/metadata/releases.csv` (release date, country, format), so it picks the same images on
every machine. Compare the `sha256` column of the manifest to confirm. Images deleted from the Cover Art Archive
since are the only source of difference. Steps that only read local files do not need `ALBUM_PROJECT_CONTACT`.

`data/metadata/excluded_images.csv` lists 8 images of the 10 albums that failed to download in the original build.
They stay out of both datasets, so `10` matches the report exactly.

## Layout

```
data/
  metadata/      albums.csv          Rolling Stone 500 candidates (467 after exclusions)
                 screen.csv          MusicBrainz screen: front-art releases per album
                 albums_10.csv       the 10 albums of the data report (folder name, release-group ID)
                 albums_all.csv      the 430 screened albums, built from screen.csv by 02b_make_album_list.py
                 image_types_raw.csv Cover Art Archive images and type tags (one shared snapshot)
                 snapshot_albums.csv albums finished in the snapshot, with the date of each
                 image_types.csv, image_types_by_album.csv   summaries of the snapshot
                 releases.csv        release metadata (one shared snapshot)
                 excluded_images.csv picks kept out of every dataset
  raw/           candidate_fronts/ (40 front covers each for the top 20 candidates). Not in git.
  images/        <album>/<type>/<release_id>.jpg, shared by both datasets. Not in git.
  logs/          failure lists from download runs. Not in git.
  manifest_<name>.csv, split_<name>.csv, pilot_results_<name>.csv
  features_resnet50.npz   ResNet-50 features cached per image file. Not in git.
reports/         data report (LaTeX source, bibliography, figures, submitted PDF)
scripts/         run from anywhere: python project/scripts/<name>.py
```

## Full pipeline

1. `01_screen_albums.py`: albums.csv to screen.csv (front-art count per album)
2. `02_download_covers.py`: front covers of the top 20 into data/raw/candidate_fronts/ (Assignment 2 era)
3. `02b_make_album_list.py`: screen.csv to albums_all.csv. Checks folder and release-group collisions and that
   albums_10.csv is an identical subset.
4. `03_count_image_types.py [--albums albums_all.csv] [--resume] [--limit N]`: snapshot every image and type tag.
   Without `--resume` it fetches nothing once the snapshot exists. With `--resume` it skips finished albums
   entirely and fetches only missing releases of unfinished ones.
5. `04_download_project.py {10,all}`: one image per type per release into data/images/, writes the manifest.
   Fetches release metadata only for albums missing from releases.csv. Reuses images on disk.
6. `05_pilot_learning_curve.py {10,all}`: release-level split and pilot learning curve.
   `reports/figures/pilot_learning_curve.png` is the report figure. The script writes `pilot_learning_curve_<name>.png`.

Steps 1 to 6 have run for `10`. To build `all`:

```
python project/scripts/03_count_image_types.py --resume      # several hours, resumable
python project/scripts/04_download_project.py all            # several hours, resumable
python project/scripts/05_pilot_learning_curve.py all
```

Both downloads respect MusicBrainz (about 1 request per second) and use at most 4 parallel Cover Art Archive
requests. archive.org returns transient 500s: failed items are retried with backoff, then skipped and listed in
`data/logs/`. Rerun the same command to retry them. Ctrl-C is safe at any point.
