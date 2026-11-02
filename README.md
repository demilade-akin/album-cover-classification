# Album Edition Classifier (TME 6015)

Assignment 2: recognise an album from front covers of editions not seen in training.
Final project: recognise an album from any packaging image (front, back, disc, booklet).

## Layout

```
data/
  metadata/      albums.csv (candidates), screen.csv (MusicBrainz screen),
                 image_types.csv / image_types_raw.csv (Cover Art Archive type counts)
  raw/           covers/ (original downloads, 20 albums), covers_extra/ (top-up, unreviewed)
                 not in git: rebuild with the scripts
  assignment2/   dataset/ (cleaned 10 albums, Dataset A), split/ + split.csv, looks.csv (to add)
models/          saved models
scripts/         run from anywhere: python scripts/<name>.py
sample.ipynb     Assignment 2 notebook (run from the project folder)
```

## Pipeline

1. `scripts/screen_albums.py`: albums.csv to screen.csv
2. `scripts/download_covers.py`: front covers into data/raw/covers/
3. Manual review, copy the kept 10 albums into data/assignment2/dataset/
4. `scripts/split_dataset.py`: grouped, stratified train/val/test split
5. `scripts/count_image_types.py`: image type counts per album (project feasibility)
