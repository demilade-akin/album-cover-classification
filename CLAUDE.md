# TME 6015 final project: album packaging classifier

Assignment 2 (food image classifier) moved to its own repo,
`~/Developer/tme6015-food-classifier`. Nothing here reads or writes that repo.

## Environment
- macOS on Apple Silicon, PyCharm, `uv`-managed `.venv` on Python 3.12. `pyproject.toml` and `uv.lock` at the root.
- **Keras / TensorFlow.** CPU is fine at this data size. `tensorflow-metal` optional. Colab is an option.
- Load splits with `image_dataset_from_directory`, passing `class_names` explicitly so label indices match across splits.
- Every script sets `ROOT = Path(__file__).resolve().parents[1]`, which is the repo root. Data paths are
  relative to it. Scripts run from anywhere: `python scripts/<name>.py`.

## Working style
- Direct and concise. Push back when something has a problem.
- Don't write code unless I ask for it.
- Explain the reasoning behind choices so I can write it up. The interpretation must be my own, and I verify anything generated.
- Report prose: active voice, plain concrete words, no filler or AI phrases, no dashes or semicolons.
- Comment code clearly and mark each model's section in notebooks.

## Modelling rules
- Augment only the training set, on the fly as Keras layers.
- Tune on validation only. Touch the test set once per model at the end. Report test size honestly.
- Metrics: accuracy plus macro F1, per-class precision and recall, confusion matrix. Verification: ROC curve, AUC,
  accuracy at a threshold chosen on validation.

## Project (final definition, set November 22, 2026)

### Problem
Music databases and marketplaces (MusicBrainz, the Cover Art Archive, Discogs) rely on uploaders to tag packaging
images by hand. Tags are often wrong and images get attached to the wrong album. Nobody can check millions of uploads
by hand.

### Solution: an upload checker with two models
1. **Part classifier:** image -> front, back, disc (medium) or booklet. 4 classes.
2. **Album verifier:** image + claimed album -> match or no match. Score = mean of the top 3 similarities between the
   image and that album's gallery images. Threshold chosen on validation albums.

Both must work on **albums never seen in training**. That is the central claim.

### Research question
For generalising to unseen albums, do more albums beat more editions of a few albums? (Instructor feedback: more data
over more variation of the same albums.)

### Deadline and deliverables
- Final project due **Thursday, November 26, 2026** (25 marks): report, presentation, public GitHub repo with dataset,
  code and trained models.
- Report: CVPR or IEEE two-column template, sections following the final project rubric (introduction and business
  value, literature review, model selection, data preparation, training, downstream integration, results, discussion).
- The report must explain the change from the data report: following instructor feedback, the dataset grew from 10
  to 430 albums and the task changed so it works on unseen albums.
- Images stay out of the public repo unless the instructor says otherwise. Commit manifests, splits, models and the
  rebuild scripts.

### Split (final experiments)
- Dataset `all`, split **by album**: about 70/15/15 train/validation/test albums, fixed seed, saved to a committed
  split file. Test albums are untouched until final evaluation.
- The 10 albums of `albums_10.csv` are forced into the training pool, so breadth vs depth uses the same test albums.
- The release-level split (`split_10.csv`, `05_pilot_learning_curve.py`) belongs to the data report pilot only.

### Models
- All models use frozen Keras ResNet-50 features (ImageNet weights, Keras `preprocess_input`), cached per image in
  `data/features_resnet50.npz`.
- Part classifier: logistic regression on the features, class weights (fronts are about 40% of images).
- Verifier baseline: cosine similarity on the raw features.
- Verifier trained: a small projection layer on the frozen features, trained on training albums so images of the same
  album sit close together. Compare against the baseline.
- Optional, only if everything else is done: one fine-tuning run on Colab.

### Experiments (exactly three)
1. **Part classifier on test albums:** accuracy, macro F1 (headline), per-class precision and recall, confusion matrix.
2. **Verifier on test albums:** each test image gets one genuine check (own album), one random impostor check and one
   hard impostor check (an album by the same artist, otherwise the most similar-looking album by features). Impostors
   come only from other test albums. Report ROC, AUC and accuracy, split into random vs hard and by the query's part,
   for both the baseline and the trained verifier.
3. **Breadth vs depth:** part classifier trained on (A) the 10 albums (~2,200 images), (B) an equal number of images
   sampled across all training albums, (C) all training albums. All tested on the same test albums. A vs B isolates
   how the images are spread. C shows what extra data adds.

### Verification rules
- A query's gallery is its claimed album's images **minus the query's own release** and minus near-duplicates of the
  query (imagehash), so a scan uploaded to several releases cannot match itself.
- Only test albums with at least 2 releases take part in verification. The part classifier uses every image.
- Pick the gallery score (top-1 vs mean of top 3) on validation, not test.
- Albums with more than one cover design (good kid, m.A.A.d city, Thriller anniversary editions) are known hard
  positives. Discuss them as a limitation with examples. Linking different designs needs text (OCR): future work.

### Downstream integration
- An audit script runs both models over the test albums' images and writes a report of tag mismatches (predicted part
  differs from the uploader tag) and album mismatches, sorted by confidence. This is the presentation demo.
- Hand-check the top 50 flagged images and report how many are real mislabels. This replaces the label check promised
  in the data report.

### Out of scope
320-class album classification, retrieval or search, predicting format or decade, extra data collection, more than
one fine-tuning run.

## Data

### Two datasets
- `10`: the 10 albums in `albums_10.csv`, as in the submitted data report (2,223 images, 899 releases). Must stay
  reproducible: `data/manifest_10.csv`, `data/split_10.csv`, `results/pilot_results_10.csv` must not change.
- `all`: the 430 albums in `albums_all.csv` (screen.csv with status ok and title_ok True, no size filter). Built by
  `02b_make_album_list.py`. Any size cutoff happens at training time, not download time.
- `all` contains `10` unchanged: one shared snapshot, image folder, releases.csv and feature cache. Image picks are
  seeded per release and type, splits per album, so the 10 albums match in both.
- Snapshot dates: the original 10 albums on November 2, 2026. The other 420 later, dated per album in
  `snapshot_albums.csv`. Mention the gap when comparing the datasets.
- `rg_overrides.csv` fixes 3 albums the screen matched to a stray release group (The Beatles, Who's Next, Reach
  Out). 02b applies it. 03 stops when a finished album's rg_id changed. Use `--resume --refetch FOLDER`.
- `excluded_images.csv` keeps out 8 picks that failed to download in the original build, so `10` matches the report.
- Network scripts read the User-Agent contact from `ALBUM_PROJECT_CONTACT`. Never hardcode it.

### Layout
```
scripts/         data pipeline: 01_screen_albums, 02_download_covers (record only), 02b_make_album_list,
                 03_count_image_types, 04_download_project {10,all}, 05_pilot_learning_curve {10,all}, http_client
data/metadata/   albums.csv, screen.csv, albums_10.csv, albums_all.csv, image_types_raw.csv + releases.csv (pinned
                 snapshots the download reads), snapshot_albums.csv, excluded_images.csv, rg_overrides.csv,
                 image_types*.csv
data/            manifest_<name>.csv, split_<name>.csv (committed). images/, logs/, features_resnet50.npz (not in git)
results/         generated outputs: pilot_results_<name>.csv, figures/, model metrics
notebooks/       one notebook per model (create with the first model)
reports/         data_report/ (submitted, frozen: never write into it), final_report/ (later)
```
- Inputs go in `data/`, anything a script or notebook generates goes in `results/`, report sources in `reports/`.
- Run order for `all`: `02b_make_album_list.py`, `03_count_image_types.py --resume`, `04_download_project.py all`,
  `05_pilot_learning_curve.py all`. 03 and 04 are resumable: rerun the same command after a crash or failures.
- Reproducibility: a fresh clone rebuilds either dataset with `04_download_project.py {10,all}` from the committed
  snapshots. Never refetch the snapshots silently: `03_count_image_types.py` fetches nothing without `--resume`, and
  with `--resume` skips albums already finished in `snapshot_albums.csv`.

### Data source
- MusicBrainz (release group = album, release = edition) and the Cover Art Archive.
- Requests need the identifying User-Agent header (contact from `ALBUM_PROJECT_CONTACT`). MusicBrainz: about 1
  request per second. Cover Art Archive (archive.org) returns transient 500s: retry, skip, rerun.
- The release ID in each filename is the edition identifier.
- SHA-256 dedup only catches byte-identical files. Do not call images "distinct" beyond that.

### Rules that must hold
- **No leakage across albums:** final experiments split by album. Nothing from validation or test albums is used for
  training, impostor selection or thresholds (thresholds come from validation albums only).
- Image types: Front, Back, Medium, Booklet. "Back;Spine" counts as Back, "Medium;Matrix/Runout" as Medium. Other tags
  are dropped. One image per type per release.
- Augmentation (only if fine-tuning): random resized crop (scale about 0.7 to 1.0), small rotation (about ±10°),
  brightness and contrast jitter, random erasing. **No flips** (covers contain text). Little or no blur.
- Preprocessing: resize to 224×224 by squashing, then the preprocessing that matches the pre-trained model.
- Data report (`reports/data_report/`) is submitted and frozen. Never write into it.
