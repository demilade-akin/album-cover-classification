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

Classify any packaging image (front, back, disc, booklet) into albums. Question: does a model recognise an album
from parts of the package that look nothing like the front? The main result is accuracy per image type.

### Two datasets
- `10`: the 10 albums in `albums_10.csv`, as in the submitted data report (2,223 images, 899 releases). Must stay
  reproducible: `manifest_10.csv`, `split_10.csv`, `pilot_results_10.csv` must not change.
- `all`: the 430 albums in `albums_all.csv` (screen.csv with status ok and title_ok True, no size filter). Built by
  `02b_make_album_list.py`. Any size cutoff happens at training time, not download time.
- `all` contains `10` unchanged: one shared snapshot, image folder, releases.csv and feature cache. Image picks are
  seeded per release and type, splits per album, so the 10 albums match in both.
- Snapshot dates: the original 10 albums on November 2, 2026. The other 420 later, dated per album in
  `snapshot_albums.csv`. Mention the gap when comparing the datasets.
- `excluded_images.csv` keeps out 8 picks that failed to download in the original build, so `10` matches the report.
- Network scripts read the User-Agent contact from `ALBUM_PROJECT_CONTACT`. Never hardcode it.

### Deadlines
- Data report: Thursday, November 5, 2026.

### Data report requirements (15 marks)
Problem statement, number of data points, how I know it is enough (3), how the data is representative (3), steps to
get enough data (3), at least 3 biases using lecture terms (3), how to remove them (3). Justify with explanations,
preliminary training results and related literature.

### Layout (relative to `project/`)
```
data/metadata/   albums.csv, screen.csv, albums_10.csv, albums_all.csv, image_types_raw.csv + releases.csv (pinned
                 snapshots the download reads), snapshot_albums.csv, excluded_images.csv, image_types*.csv
data/raw/        candidate_fronts/ (40 fronts each for the top 20 candidates). Not in git.
data/images/     <album>/<type>/<release_id>.jpg, shared by both datasets. Not in git.
data/logs/       failure lists from download runs. Not in git.
data/            manifest_<name>.csv, split_<name>.csv, pilot_results_<name>.csv, features_resnet50.npz (not in git)
reports/         data_report.tex, references.bib, figures/, submitted PDF
scripts/         01_screen_albums, 02_download_covers, 02b_make_album_list, 03_count_image_types,
                 04_download_project {10,all}, 05_pilot_learning_curve {10,all}, http_client (shared)
```
- Run order for `all`: `02b_make_album_list.py`, `03_count_image_types.py --resume`, `04_download_project.py all`,
  `05_pilot_learning_curve.py all`. 03 and 04 are resumable: rerun the same command after a crash or failures.
- Reproducibility: a fresh clone rebuilds either dataset with `04_download_project.py {10,all}` from the committed
  snapshots. Never refetch the snapshots silently: `03_count_image_types.py` fetches nothing without `--resume`, and
  with `--resume` skips albums already finished in `snapshot_albums.csv`.

### Data source
- MusicBrainz (release group = album, release = edition) and the Cover Art Archive.
- Requests need the identifying User-Agent header (contact from `ALBUM_PROJECT_CONTACT`). MusicBrainz: about 1 request per second. Cover Art Archive
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
