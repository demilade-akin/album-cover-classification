# An upload checker for album packaging images

TME 6015 (AI/ML Workflow Design) final project. Akinbobola Akin.

Music databases and marketplaces such as MusicBrainz, the Cover Art Archive and Discogs depend on uploaders to tag
packaging images by hand. Tags are often wrong, and images end up attached to the wrong album. This project builds
two models that check an upload automatically:

1. **Part classifier.** Image to front, back, medium (disc) or booklet. It flags images whose uploader tag
   disagrees with the prediction.
2. **Album verifier.** Image plus claimed album to match or no match. It compares the image with the album's other
   images and flags uploads that probably belong to a different album.

Both models are evaluated on **albums never seen in training**. The dataset holds 23,430 images from 430 albums,
split by album into 302 train, 64 validation and 64 test albums. A third experiment asks whether more albums or
more editions of a few albums help the part classifier generalise to unseen albums.

## Headline results (test albums)

Taken from `results/tables/`. The test set has 64 albums and 3,629 images.

| Experiment | Result |
|---|---|
| 1. Part classifier | accuracy 0.858, macro F1 0.833 (F1: front 0.877, back 0.782, medium 0.985, booklet 0.687) |
| 2. Album verifier, AUC (baseline → trained) | random impostors 0.943 → 0.961, hard impostors 0.906 → 0.917, all checks 0.924 → 0.939 |
| 2. Album verifier, accuracy at validation threshold | baseline 0.867, trained 0.912 (all checks) |
| 3. Breadth vs depth, macro F1 | A: 10 albums 0.747. B: same image count over 302 albums 0.817 ± 0.007 (5 seeds). C: all 302 albums 0.833 |

Spreading the same number of images over many albums beats piling them onto a few. Adding seven times more images
on top of that adds only a little.

## Images are not in this repo

The repo holds the manifests, snapshots, splits, results and trained models, but no images. They come from the
Cover Art Archive and keep their uploaders' rights. Rebuild them with:

```bash
export ALBUM_PROJECT_CONTACT="you@example.com"   # MusicBrainz asks for a contact in the User-Agent
python scripts/04_download_project.py all          # several hours, resumable
```

The script reads the committed snapshots, so it picks the same images on every machine. The `sha256` column of
`data/manifest_all.csv` lets you confirm each file. The only difference can come from images deleted from the
Cover Art Archive since the snapshot.

**You do not need the images to check the results.** `scripts/12_make_figures.py` rebuilds every chart and table
from the committed files in `results/`. Only the figures that show album images (`scripts/13_image_figures.py`)
need the images.

## Setup

```bash
git clone https://github.com/demilade-akin/album-upload-checker.git
cd album-upload-checker
uv sync
```

`uv sync` creates `.venv` with Python 3.12 from `pyproject.toml` and `uv.lock`. Activate it with
`source .venv/bin/activate`, or prefix commands with `uv run`. Every script finds the repo root by itself, so you
can run it from any folder. TensorFlow runs on CPU, which is fast enough at this data size.

## Run the experiments

Run the steps in order. Step 07 needs the images. Steps 08 to 10 need the feature cache that step 07 writes. Step 09 reads the committed `data/phash.csv` and hashes images only if that file is missing. Step 12 needs only committed files. Step 13 needs the images.

| Step | Command | What it does | Output | Time |
|---|---|---|---|---|
| 06 | `python scripts/06_split_albums.py` | Splits albums 70/15/15 by size band, seed 42. The 10 data report albums go to train. | `data/split_albums.csv` | seconds |
| 07 | `python scripts/07_extract_features.py` | Frozen ResNet-50 (ImageNet) features for every image. Resumable. | `data/features_resnet50.npz` (not in git) | 15 to 25 min on CPU |
| 08 | `python scripts/08_part_classifier.py` | Experiment 1. Tunes logistic regression on validation, tests once. | `results/part_classifier/` | under a minute |
| 09 | `python scripts/09_verifier.py` | Experiment 2. pHash near-duplicates, checks, baseline and trained projection, thresholds on validation, tests once. | `results/verifier/`, `data/phash.csv` | several minutes |
| 10 | `python scripts/10_breadth_vs_depth.py` | Experiment 3. Training sets A, B (5 seeds) and C, each tuned on validation. | `results/breadth_depth/` | under a minute |
| 11 | `python scripts/11_audit.py` | Downstream audit. Flags test images whose predicted part differs from the uploader tag, and images that fail the verifier against their own album, ranked by confidence. Writes a hand-check sheet of the top 50 part flags and top 20 album flags. | `results/audit/` | seconds |
| 12 | `python scripts/12_make_figures.py` | Every report chart and table, from `results/` only. | `results/figures/`, `results/tables/` | seconds |
| 13 | `python scripts/13_image_figures.py` | Report figures built from album images: the four image types (Kind of Blue) and the audit examples (The Dark Side of the Moon, Black Sabbath, Bitches Brew), each showing a front, a back the part classifier called a front and a back it called a back, and two figures of album flags on discs and booklet pages, each next to a front of its album. Labels read predictions from `results/part_classifier/predictions.csv` and album scores from `results/audit/flags.csv`. | `results/figures/` | seconds |

Validation albums set every hyperparameter and threshold. Test albums are scored only at the end of each
experiment. `results/part_classifier/metrics.json` records each time the part classifier's test set was scored
during tuning.

## Inference

Check any image file, inside the dataset or not:

```bash
python scripts/predict.py path/to/image.jpg                                    # predicted part, all 4 probabilities
python scripts/predict.py path/to/image.jpg --album michael_jackson__thriller  # plus match / no match
```

`--album` takes a folder name from `data/metadata/albums_all.csv`. The verifier compares the image with that
album's gallery using the same rules as experiment 2: if the image is in the dataset, its own release is left out,
and near-duplicate scans (pHash distance 2 or less) are left out too. Gallery features come from
`data/features_resnet50.npz`, or from the images in `data/images/` if the cache is missing. The script warns when
the claimed album is a training album, because then the check is not on an unseen album.

To use the models in your own code, load them like this. Both take the 2,048 ResNet-50 features as input.

```python
import joblib, keras
part_model = joblib.load("results/part_classifier/model.joblib")       # scaler + logistic regression
projection = keras.saving.load_model("results/verifier/projection.keras")  # 2048 -> 512, L2-normalised
part = part_model.predict(features)                  # features: (n, 2048) array
embedding = projection.predict(features)             # cosine similarity = dot product of embeddings
```

The verifier's decision threshold and scorer (top-1 similarity against the claimed album's gallery) are stored in
`results/verifier/metrics.json`.

## Where results land

```
results/
  part_classifier/  tuning.csv, metrics.json, predictions.csv, confusion_matrix.png, model.joblib
  verifier/         tuning.csv, tuning_epochs.csv, metrics.json, checks_test.csv, roc_curves.png, projection.keras
  breadth_depth/    results.csv, summary.csv, breadth_vs_depth.png
  audit/            flags.csv, hand_check.csv (verdict column filled by hand), hand_check.html
  figures/          report figures (.pdf and .png): charts from step 12, album image figures from step 13,
                    plus the data report pilot curve
  tables/           one CSV per report table
  pilot_results_10.csv   data report pilot (frozen)
```

## Data pipeline (steps 01 to 05)

These steps built the dataset. A fresh clone only needs step 04 (see above).

1. `01_screen_albums.py`: Rolling Stone 500 candidates (`albums.csv`) to `screen.csv`, the number of releases
   with front art per album on MusicBrainz.
2. `02_download_covers.py`: front covers of the top 20 candidates, used to choose the original 10 albums. Kept as a
   record. Its output was deleted.
3. `02b_make_album_list.py`: `screen.csv` plus `rg_overrides.csv` to `albums_all.csv` (430 albums).
4. `03_count_image_types.py --resume`: snapshots every image and type tag from the Cover Art Archive. Without
   `--resume` it fetches nothing.
5. `04_download_project.py {10,all}`: one image per type per release into `data/images/`, and the manifest.
6. `05_pilot_learning_curve.py {10,all}`: the data report's release-level pilot. Not used by the final
   experiments.

Type rules: Front, Back, Medium and Booklet are kept. "Back;Spine" counts as Back and "Medium;Matrix/Runout" as
Medium. Other tags are dropped. Each release adds at most one image per type.

### Two datasets

| name | album list | albums | manifest | snapshot |
|---|---|---|---|---|
| `10` | `albums_10.csv` | 10 | `manifest_10.csv` | November 2, 2026 |
| `all` | `albums_all.csv` | 430 | `manifest_all.csv` | the 10 as above, the other 420 later (dated per album in `snapshot_albums.csv`) |

`10` is the dataset of the submitted data report (2,223 images, 899 releases). Following instructor feedback, the
final project grew it to `all` and changed the task so it works on unseen albums. `all` contains `10` unchanged.

- `rg_overrides.csv` fixes 3 albums that the screen matched to a stray release group (The Beatles, Who's Next,
  Reach Out).
- `excluded_images.csv` keeps out 8 images that failed to download in the original build, so `10` matches the
  report exactly.

Downloads keep to about 1 MusicBrainz request per second and at most 4 parallel Cover Art Archive requests.
archive.org returns transient 500 errors. Failed items are retried, then listed in `data/logs/`. Rerun the same
command to retry them.

## Layout

```
scripts/        pipeline and experiments, numbered in run order. http_client.py holds the shared HTTP code.
data/
  metadata/     committed inputs and snapshots (album lists, screen, image tags, releases, overrides)
  manifest_*.csv, split_*.csv, split_albums.csv, phash.csv   committed
  images/, logs/, features_resnet50.npz                       not in git, rebuilt by scripts 04 and 07
results/        generated outputs (see above)
```

Assignment 2 (the food image classifier) lives in its own repo, `tme6015-food-classifier`.
