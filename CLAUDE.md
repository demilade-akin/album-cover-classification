# Project: Album Edition Classifier (TME 6015)

Two deliverables share this repo and the same 10 albums.

- **Assignment 2:** classify front covers into 10 albums. Each class holds front covers of different editions (reissues, regional pressings, deluxe and anniversary versions). Question: does a model recognise an album across editions it has not seen in training?
- **Final project:** classify any packaging image (front, back, disc, booklet) into the same 10 albums. Question: does a model recognise an album from parts of the package that look nothing like the front? The main result is accuracy per image type.

## Deadlines
- Data report for the final project: Thursday, November 5, 2026.
- Assignment 2: extended (date not recorded here).

## Assignment 2 requirements (15 marks)
1. Own multi-class dataset (5)
2. CNN from scratch: accuracy, explain performance, reasoning for architecture and hyperparameters (5)
3. Transfer learning with a pre-trained model on the same data (2)
4. Compare the two models using architecture, hyperparameters and training plots (3)
5. Reflection: did I use gen AI, based on my Assignment 1 experience

Marks reward analysis and explanation, not novelty or raw accuracy. Deliverable: notebook with code, comments, plots and written interpretation.

## Data report requirements (15 marks)
Problem statement; number of data points; how I know it is enough (3); how the data is representative (3); steps to get enough data (3); at least 3 biases using lecture terms (3); how to remove them (3). Justify with explanations, preliminary training results and related literature.

## Layout
```
data/metadata/      albums.csv, screen.csv, final_albums.csv (the 10 albums), image_types_raw.csv +
                    releases.csv (pinned snapshots the project download reads), image_types.csv
data/raw/           candidate_fronts/ (40 fronts each for the top 20 candidates), remaining_fronts/ (old top-up from the
                    notebook, replaced by the project fronts, delete once Dataset B is built). Not in git.
data/assignment2/   dataset/ (cleaned, Dataset A), split/ + split.csv, looks.csv (to add)
data/project/       images/<album>/<type>/<release_id>.jpg (not in git), manifest.csv
models/             saved models
scripts/            01_screen_albums, 02_download_covers, 03_split_assignment2, 04_count_image_types,
                    05_download_project. Run from anywhere: python scripts/<name>.py
sample.ipynb        Assignment 2 notebook, run from the project folder
```
- Reproducibility: a fresh clone rebuilds the project dataset with `05_download_project.py` from the committed snapshots. Never refetch the snapshots silently: `04_count_image_types.py` skips the fetch unless run with `--resume` or the file is deleted.

## Environment
- macOS on Apple Silicon, PyCharm, `uv`-managed `.venv` on Python 3.12.
- **Keras / TensorFlow.** CPU is fine at this data size; `tensorflow-metal` optional. Colab is an option.
- Load splits with `image_dataset_from_directory`, passing `class_names` explicitly so label indices match across splits.

## Data source
- MusicBrainz (release group = album, release = edition) and the Cover Art Archive.
- Requests need the identifying User-Agent header. MusicBrainz: about 1 request per second. Cover Art Archive (archive.org) returns transient 500s: retry, skip, rerun.
- The release ID in each filename is the edition identifier.
- SHA-256 dedup only catches byte-identical files. Do not call images "distinct" beyond that.

## Assignment 2 plan
- **Dataset A (balanced):** `data/assignment2/dataset/`, 393 images, 38 to 40 per album, no class weights.
- **Dataset B (uncapped):** A's training images plus the reviewed project fronts in `data/project/images/<album>/front/`, excluding A's val and test releases (by release ID and near-duplicate check), trained with class weights. Extras go into training only. Val and test are identical to A's.
- A vs B isolates one variable: training data volume. Four runs: from-scratch CNN and pre-trained model, each on A and B.
- **Split:** about 70/15/15, stratified by album and look, near-duplicate groups kept together, seed 42. `03_split_assignment2.py` must use `HASH_THRESHOLD = 2` (the current split at 4 is invalid and must be rerun).
- **Looks:** `looks.csv` (`file,look`) tags GKMC minivan covers and Thriller alternates so each look reaches train and test.

## Final project plan
- Image types: Front, Back, Medium, Booklet. Map "Back;Spine" to Back and "Medium;Matrix/Runout" to Medium. Drop Tray, Other, Untyped and other multi-tag images.
- One image per type per release (booklets have many pages per release).
- Split by release: all images of a release in one split, stratified by album and type.
- Album imbalance (Darkness and Tapestry far larger than Paranoid): class weights plus macro F1.
- Thin backs for Paranoid, Tommy and Joshua Tree. Report it.
- Hand-check about 30 images per type and report the mislabel rate.
- Record country, year and format per release to show representativeness.
- Pending: instructor approval to reuse the Assignment 2 albums.

## Rules that must hold
- **Split by edition (release), never by image.** The same image (same scan resized or recompressed) on both sides is leakage. A different scan of the same artwork in test is the task, not leakage.
- **Augment only the training set, on the fly as Keras layers:** random resized crop (scale about 0.7 to 1.0), small rotation (about ±10°), brightness and contrast jitter, random erasing. **No flips** (covers contain text). Little or no blur.
- **Preprocessing:** resize to 224×224 by squashing, ImageNet mean/std normalisation for both models.
- **Metrics:** accuracy plus macro F1, per-album precision and recall, confusion matrix, per-look (A2) or per-type (project) accuracy.
- Tune on validation only. Touch the test set once per model at the end. Report test size honestly.
- Cleaning rule for covers: keep the front of any packaged edition (stickers, banners, obi, box sets, photos of physical copies). Delete bare discs, back covers, booklet pages, inner sleeves and spreads where the front is under about half the image.

## Working style
- Direct and concise. Push back when something has a problem.
- Don't write code unless I ask for it.
- Explain the reasoning behind choices so I can write it up. The interpretation must be my own, and I verify anything generated.
- Report prose: active voice, plain concrete words, no filler or AI phrases, no dashes or semicolons.
- Comment code clearly and mark each model's section in the notebook.
- Keep a running note of where gen AI helped, for item 5 of Assignment 2.
