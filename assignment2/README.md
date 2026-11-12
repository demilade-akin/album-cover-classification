# Assignment 2: food image classifier

Classify recipe photos into 6 classes: pasta, soup, salad, cake, cookies, stir_fry. The dataset is built from
the main photo of recipes on seven recipe blogs. Paths below are relative to this `assignment2/` folder.

## Pipeline

```
python assignment2/scripts/01_download_recipes.py --strict   # download and label (resumable)
python assignment2/scripts/02_clean.py                       # convert to RGB JPEG, drop bad and duplicate images
python assignment2/scripts/03_split.py                       # 70/15/15 split into train/val/test folders
```

**01 download.** Sources: recipetineats.com, budgetbytes.com, cafedelites.com, loveandlemons.com,
omnivorescookbook.com, sugarspunrun.com, natashaskitchen.com. Uses the WP Recipe Maker REST API, falling back to
the schema.org Recipe data on each page. 1 to 2 s between requests, follows robots.txt. Caps: `--max-per-class`
(150 per class per site) and `--max-total-per-class` (300 per class across sites). See the script docstring for
the labelling rules.

**02 clean.** Drops images whose title shows the photo isn't one dish of the class (roundup posts, spaghetti
squash as pasta, ice cream as cookies, pineapple cake as cake), any file listed in `data/exclude.csv`, and
near-duplicates (pHash distance at most 4). Converts every kept image to RGB JPEG with the shorter side at most
512 px. Every removal is logged with its reason in `data/removed.csv`. No manual review pass was done.

**03 split.** Stratified by class and source site, seed 42. Copies files, never moves them. Both 02 and 03
rebuild their output folders from scratch and give the same result on every run.

## Current dataset

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| pasta | 204 | 44 | 43 | 291 |
| soup | 210 | 45 | 45 | 300 |
| salad | 210 | 45 | 45 | 300 |
| cake | 210 | 44 | 45 | 299 |
| cookies | 202 | 43 | 43 | 288 |
| stir_fry | 139 | 31 | 31 | 201 |
| total | 1175 | 252 | 252 | 1679 |

1,694 downloaded, 15 removed by the cleaning rules.

## Loading in the notebook

```python
from pathlib import Path
import tensorflow as tf

CLASS_NAMES = ["pasta", "soup", "salad", "cake", "cookies", "stir_fry"]   # fixed order for every split
SPLIT = Path("assignment2/data/split")
train_ds = tf.keras.utils.image_dataset_from_directory(
    SPLIT / "train", class_names=CLASS_NAMES, image_size=(224, 224), seed=42)

# Class weights for the stir_fry gap: n_total / (n_classes * n_class), computed from the train folder
counts = [len(list((SPLIT / "train" / c).glob("*.jpg"))) for c in CLASS_NAMES]
class_weight = {i: sum(counts) / (len(counts) * n) for i, n in enumerate(counts)}
# model.fit(train_ds, validation_data=val_ds, class_weight=class_weight, ...)
```

`image_size` squashes by default. Pass `crop_to_aspect_ratio=True` to centre-crop instead: most photos are
portrait (width / height about 0.7), so squashing distorts them.

## Layout

```
data/
  raw/<class>/       downloaded images (jpg, png, webp). Not in git.
  clean/<class>/     cleaned RGB JPEGs. Not in git, rebuilt by 02.
  split/<train|val|test>/<class>/   Keras folders. Not in git, rebuilt by 03.
  metadata.csv       one row per downloaded image (title, recipe URL, image URL)
  processed.csv      every recipe the downloader looked at, with its outcome
  clean.csv          one row per kept image (size, pHash)
  removed.csv        one row per dropped image, with the reason
  split.csv          filename, class, source_site, split
scripts/
  01_download_recipes.py, 02_clean.py, 03_split.py
```

## Things to report

- **Site and class are tangled.** sugarspunrun supplies half of cake and cookies, and omnivorescookbook supplies
  most of stir_fry. A model could learn a site's photo style instead of the food. The split keeps each site's
  share equal across train, val and test. Use the `source_site` column in `split.csv` to report accuracy per site.
- **Label noise.** Labels come from site categories and titles, and sites sometimes attach the wrong photo.
  Cleaning removed known patterns only.
- **Newest-first sampling.** The WPRM API lists newest recipes first, so capped sites give mostly recent photos.
