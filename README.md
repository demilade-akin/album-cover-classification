# Album Edition Classifier (TME 6015)

Assignment 2: recognise an album from front covers of editions not seen in training.
Final project: recognise an album from any packaging image (front, back, disc, booklet).

## Reproduce the project dataset

```
uv sync
python scripts/05_download_project.py
```

That is all a fresh clone needs. The script reads two committed snapshots, `data/metadata/image_types_raw.csv`
(which images exist) and `data/metadata/releases.csv` (release date, country, format), so it picks the same
images on every machine. Compare the `sha256` column of `data/project/manifest.csv` to confirm.
Images missing from the snapshot or deleted from the Cover Art Archive since are the only source of difference.

## Layout

```
data/
  metadata/      albums.csv        Rolling Stone 500 candidates (467 after exclusions)
                 screen.csv        MusicBrainz screen: front-art releases per album
                 final_albums.csv  the 10 albums used (folder name, release-group ID)
                 image_types_raw.csv / image_types.csv  Cover Art Archive images and type tags (snapshot)
                 releases.csv      release metadata (snapshot)
  raw/           candidate_fronts/ (40 front covers each for the top 20 candidates),
                 remaining_fronts/ (old notebook top-up, replaced by data/project fronts). Not in git.
  assignment2/   dataset/ (cleaned front covers, Dataset A), split/ + split.csv, looks.csv (to add)
  project/       images/<album>/<type>/ (not in git), manifest.csv
models/          saved models
scripts/         run from anywhere: python scripts/<name>.py
sample.ipynb     Assignment 2 notebook (run from the project folder)
```

## Full pipeline

1. `01_screen_albums.py`: albums.csv to screen.csv (front-art count per album)
2. `02_download_covers.py`: front covers of the top 20 into data/raw/candidate_fronts/
3. Manual: pick the final 10, clean them into data/assignment2/dataset/, list them in final_albums.csv
4. `03_split_assignment2.py`: grouped, stratified train/val/test split for Assignment 2
5. `04_count_image_types.py`: snapshot every image and type tag for the final albums
6. `05_download_project.py`: one image per type per release into data/project/

Steps 1 to 5 have already run and their outputs are committed. Step 4 skips the fetch when the snapshot
exists: delete `image_types_raw.csv` (and `releases.csv`) to take a new one, which changes the dataset.
