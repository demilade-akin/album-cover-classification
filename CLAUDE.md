# TME 6015 final project: album packaging upload checker

Due **Thursday, November 26, 2026**: report, presentation, public GitHub repo (code, models, results).
Assignment 2 lives in a separate repo (`~/Developer/tme6015-food-classifier`). Nothing here touches it.

## Working style
- Direct and concise. Push back when something has a problem.
- Don't write code unless I ask for it.
- Explain reasoning so I can write it up. The interpretation must be my own, and I verify anything generated.
- Report prose: active voice, plain concrete words, no filler or AI phrases, no dashes or semicolons.

## Environment
- macOS, Apple Silicon, PyCharm, `uv` (`uv sync`), Python 3.12, Keras/TensorFlow and scikit-learn on CPU.
- Scripts set `ROOT = Path(__file__).resolve().parents[1]` (repo root) and run from anywhere: `python scripts/<name>.py`.
- Network scripts read the User-Agent contact from `ALBUM_PROJECT_CONTACT`. Never hardcode it.

## The project
Music databases and marketplaces rely on uploaders to tag packaging images by hand, and tags are often wrong.
Two models check an upload, and both are evaluated on **albums never seen in training**:
1. **Part classifier:** image to front, back, disc or booklet. Logistic regression on frozen ResNet-50 features.
2. **Album verifier:** image plus claimed album to match or no match. Top-1 cosine similarity against the album's
   gallery, in a trained 512-d projection of the frozen features. Threshold picked on validation.

Research question: for unseen albums, do more albums beat more editions of a few albums?

Terminology: the data calls discs `medium` (MusicBrainz's term). Say "disc" in the report and slides, with one line
explaining it.

## Data
- 430 albums, 23,430 images, 10,189 releases from the Cover Art Archive. One image per part per release.
- Split by album (`data/split_albums.csv`): 302 train, 64 val, 64 test albums, stratified by album size. The 10
  original albums are forced into train.
- Features: frozen Keras ResNet-50, cached per image in `data/features_resnet50.npz` (not in git).
- Images are not in git. Rebuild: `python scripts/04_download_project.py all` (reads the committed snapshots).
- Never refetch snapshots: `03_count_image_types.py` fetches nothing without `--resume`.

## Results (test albums, from `results/`)
- Part classifier (C=0.0001, no class weights): accuracy 0.858, macro F1 0.833. Disc 0.985, booklet 0.687.
- Verifier, trained vs baseline: AUC 0.939 vs 0.924, accuracy 0.912 vs 0.867. Genuine discs accepted 67%,
  genuine booklets 51%.
- Breadth vs depth (macro F1): 10 albums 0.747, same images over 302 albums 0.817 ± 0.007, all 302 albums 0.833.

## Layout
```
scripts/   01-05 data pipeline and data report pilot, 06 split, 07 features, 08 part classifier, 09 verifier,
           10 breadth vs depth, 11 audit, 12 figures and tables, predict.py (single-image inference)
data/      manifests, splits, metadata snapshots (committed). images/, logs/, feature cache (ignored)
results/   one folder per experiment, figures/, tables/, audit/
reports/   data_report/ (submitted, frozen: never edit), final_report/ (NeurIPS style, single column)
```

## Rules
- Tune on validation albums only. Test albums are evaluated once per model.
- Every number in the report comes from a file in `results/`. Never type numbers by hand.
- Cite each source once, at first mention.
- Course documents (instructions, rubrics) stay out of the public repo.

## Still to do
1. Report: draft every section from `results/` (currently a skeleton). Table slots need `.tex` tables in
   `reports/final_report/tables/`. Remove or generate the missing `dataset_overview.pdf` slot.
2. Remove `reports/final_report/instructions.pdf` from git and ignore it. Push.
3. Hand-check `results/audit/hand_check.csv` (70 flags, verdict: real mislabel, model error, unclear). Then fill the
   demo section with the counts.
4. Check `predict.py` runs on one image.
5. Slides.
6. Optional, only if time: one fine-tuning run on Colab, compared with the frozen-feature part classifier.
