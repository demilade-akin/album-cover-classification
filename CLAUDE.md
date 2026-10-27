# Project: Album Edition Classifier (TME 6015, Assignment 2)

## Goal
Multi-class image classification of album covers. Each class is one album. The images in a class are different editions (reissues, regional, deluxe, anniversary) of that album. The research question: does a model learn the album's identity and generalize to editions it has never seen?

## Assignment requirements (15 marks)
1. Build my own dataset for multi-class classification (5 marks)
2. CNN from scratch with my own architecture. Report accuracy, explain performance and the reasoning behind architecture and hyperparameters (5 marks)
3. Transfer learning with a pre-trained model on the same dataset (2 marks)
4. Compare the two models using architecture, hyperparameters and training plots (3 marks)
5. Short reflection on whether I used gen AI, based on my Assignment 1 experience

Marks reward analysis and explanation, not novelty or raw accuracy. Deliverable is a notebook with code, comments, plots and written interpretation.

## Deadline and scope
Due in about 4 days from Oct 27, 2026. Keep scope small. Do not add features beyond what is listed here unless I ask.

## Data
- Source: MusicBrainz API (release group = album, release = edition) and Cover Art Archive. Front covers only.
- Album list lives in `albums.csv` (columns: `artist,album`). Albums come from a larger candidate pool, then I keep the ones with enough images and visible edition variation. This selection step goes in the bias section of the write-up.
- Downloader writes to `covers/<artist>__<album>/<release-id>.jpg`, plus a per-album `manifest.csv` and a global `covers/summary.csv`.
- The release ID in each filename is the edition identifier. It is what the train/val/test split is based on.
- Rate limit: about 1 request per second. Requests need the identifying User-Agent header.
- Cap of 40 saved images per album, taken from a shuffled release list (seed 42).
- Images were saved as 500px thumbnails where available, but some are full size. Resize everything to one fixed size (for example 224x224) in preprocessing.
- SHA-256 dedup only catches byte-identical files. It does not detect near-duplicates. Do not describe the dataset as "distinct" beyond that.

## Rules that must hold
- **Split by edition, never by image.** All images from one release stay in the same split. Hold out whole editions for validation and test.
- **Augment only the training set, after the split.** Validation and test stay untouched. Apply augmentation on the fly during training, not as saved copies.
- **No horizontal or vertical flips.** Covers contain text and fixed layouts. Use random resized crop, small rotation, color jitter, slight blur and random erasing.
- Handle class imbalance deliberately (cap per class or class weights) and state which.
- Keep the test set untouched until final evaluation. No tuning on it.
- Report test size honestly. With a small dataset, one misclassification moves a class score a lot.

## Modeling plan
- Model A: small CNN built from scratch. Run with and without augmentation to compare curves.
- Model B: pre-trained ResNet (ResNet18 to start) with the final layer replaced. Try frozen backbone vs. fine-tuning only if time allows.
- Plot train and validation loss and accuracy for every run. Add a confusion matrix on the test set.
- Error analysis matters: look at which editions are misclassified (obi strips, anniversary banners, altered art) and explain why.

## Environment
- macOS on Apple Silicon, PyCharm, Python 3.9 virtualenv at `.venv`, notebook `sample.ipynb`.
- Use PyTorch. Use the `mps` device when available, otherwise `cpu`.
- A urllib3 LibreSSL warning appears on import. It is harmless.

## Working style
- Be direct and concise. Say when something I propose has a problem.
- Explain the reasoning behind modeling choices so I can write it up. The interpretation must be my own understanding, and I must verify anything generated.
- Comment code clearly and mark each model's section in the notebook.
- Keep a running note of where gen AI helped, for item 5 of the assignment.