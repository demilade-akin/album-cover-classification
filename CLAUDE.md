# TME 6015 final project: album packaging classifier

All work lives in `project/`. Assignment 2 (food image classifier) moved to its own repo,
`~/Developer/tme6015-food-classifier`. Nothing here reads or writes that repo.

## Environment
- macOS on Apple Silicon, PyCharm, `uv`-managed `.venv` on Python 3.12. `pyproject.toml` and `uv.lock` at the root.
- **Keras / TensorFlow.** CPU is fine at this data size. `tensorflow-metal` optional. Colab is an option.
- Load splits with `image_dataset_from_directory`, passing `class_names` explicitly so label indices match across splits.
- Every script sets `ROOT = Path(__file__).resolve().parents[1]`, which is the `project/` folder. Data paths are
  relative to it. Scripts run from anywhere: `python project/scripts/<name>.py`.

## Working style
- Direct and concise. Push back when something has a problem.
- Don't write code unless I ask for it.
- Explain the reasoning behind choices so I can write it up. The interpretation must be my own, and I verify anything generated.
- Report prose: active voice, plain concrete words, no filler or AI phrases, no dashes or semicolons.
- Comment code clearly and mark each model's section in notebooks.

## Modelling rules
- Augment only the training set, on the fly as Keras layers.
- Tune on validation only. Touch the test set once per model at the end. Report test size honestly.
- Metrics: accuracy plus macro F1, per-class precision and recall, confusion matrix.

## Project

Classify any packaging image (front, back, disc, booklet) into 10 albums. Question: does a model recognise an album
from parts of the package that look nothing like the front? The main result is accuracy per image type.

### Deadlines
- Data report: Thursday, November 5, 2026.

### Data report requirements (15 marks)
Problem statement, number of data points, how I know it is enough (3), how the data is representative (3), steps to
get enough data (3), at least 3 biases using lecture terms (3), how to remove them (3). Justify with explanations,
preliminary training results and related literature.

### Layout (relative to `project/`)
```
data/metadata/   albums.csv, screen.csv, final_albums.csv (the 10 albums), image_types_raw.csv +
                 releases.csv (pinned snapshots the download reads), image_types.csv
data/raw/        candidate_fronts/ (40 fronts each for the top 20 candidates). Not in git.
data/images/     <album>/<type>/<release_id>.jpg. Not in git.
data/            manifest.csv, split.csv, pilot_results.csv, features_resnet50.npz
reports/         data_report.tex, references.bib, figures/, submitted PDF
scripts/         01_screen_albums, 02_download_covers, 03_count_image_types, 04_download_project,
                 05_pilot_learning_curve
```
- Reproducibility: a fresh clone rebuilds the dataset with `04_download_project.py` from the committed snapshots. Never
  refetch the snapshots silently: `03_count_image_types.py` skips the fetch unless run with `--resume` or the file is deleted.

### Data source
- MusicBrainz (release group = album, release = edition) and the Cover Art Archive.
- Requests need the identifying User-Agent header. MusicBrainz: about 1 request per second. Cover Art Archive
  (archive.org) returns transient 500s: retry, skip, rerun.
- The release ID in each filename is the edition identifier.
- SHA-256 dedup only catches byte-identical files. Do not call images "distinct" beyond that.

### Plan
- Image types: Front, Back, Medium, Booklet. Map "Back;Spine" to Back and "Medium;Matrix/Runout" to Medium. Drop Tray,
  Other, Untyped and other multi-tag images.
- One image per type per release (booklets have many pages per release).
- Split by release: all images of a release in one split, stratified by album and type.
- Album imbalance (Darkness and Tapestry far larger than Paranoid): class weights plus macro F1.
- Thin backs for Paranoid, Tommy and Joshua Tree. Report it.
- Hand-check about 30 images per type and report the mislabel rate.
- Record country, year and format per release to show representativeness.
- Pending: instructor approval to reuse the albums from the earlier Assignment 2 version.

### Rules that must hold
- **Split by edition (release), never by image.** The same image (same scan resized or recompressed) on both sides is
  leakage. A different scan of the same artwork in test is the task, not leakage.
- **Augmentation:** random resized crop (scale about 0.7 to 1.0), small rotation (about ±10°), brightness and contrast
  jitter, random erasing. **No flips** (covers contain text). Little or no blur.
- **Preprocessing:** resize to 224×224 by squashing, ImageNet mean/std normalisation.
- **Metrics:** the shared set plus per-type accuracy.
- Cleaning rule for covers: keep the front of any packaged edition (stickers, banners, obi, box sets, photos of physical
  copies). Delete bare discs, back covers, booklet pages, inner sleeves and spreads where the front is under about half
  the image.
