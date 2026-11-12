# Final project: album packaging classifier

Recognise an album from any packaging image (front, back, disc, booklet). The main result is accuracy per
image type. Paths below are relative to this `project/` folder.

## Reproduce the dataset

```
uv sync
python project/scripts/04_download_project.py
```

That is all a fresh clone needs. The script reads two committed snapshots, `data/metadata/image_types_raw.csv`
(which images exist) and `data/metadata/releases.csv` (release date, country, format), so it picks the same
images on every machine. Compare the `sha256` column of `data/manifest.csv` to confirm.
Images missing from the snapshot or deleted from the Cover Art Archive since are the only source of difference.

## Layout

```
data/
  metadata/      albums.csv        Rolling Stone 500 candidates (467 after exclusions)
                 screen.csv        MusicBrainz screen: front-art releases per album
                 final_albums.csv  the 10 albums used (folder name, release-group ID)
                 image_types_raw.csv / image_types.csv  Cover Art Archive images and type tags (snapshot)
                 releases.csv      release metadata (snapshot)
  raw/           candidate_fronts/ (40 front covers each for the top 20 candidates). Not in git.
  images/        <album>/<type>/<release_id>.jpg. Not in git.
  manifest.csv   one row per selected image
  split.csv, pilot_results.csv     outputs of the pilot learning curve
reports/         data report (LaTeX source, bibliography, figures, submitted PDF)
  claude_outputs/  earlier generated drafts of the report
scripts/         run from anywhere: python project/scripts/<name>.py
```

## Full pipeline

1. `01_screen_albums.py`: albums.csv to screen.csv (front-art count per album)
2. `02_download_covers.py`: front covers of the top 20 into data/raw/candidate_fronts/
3. Manual: pick the final 10 and list them in final_albums.csv
4. `03_count_image_types.py`: snapshot every image and type tag for the final albums
5. `04_download_project.py`: one image per type per release into data/images/
6. `05_pilot_learning_curve.py`: pilot learning curve for the data report

Steps 1 to 5 have already run and their outputs are committed. Step 4 skips the fetch when the snapshot
exists: delete `image_types_raw.csv` (and `releases.csv`) to take a new one, which changes the dataset.
