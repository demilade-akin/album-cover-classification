# Assignment 2: food image classifier

Classify recipe photos into 6 classes: pasta, soup, salad, cake, cookies, stir_fry. The dataset is built from
the main photo of recipes on six recipe blogs. Paths below are relative to this `assignment2/` folder.

## Build the raw dataset

```
python assignment2/scripts/01_download_recipes.py            # category first, title as fallback
python assignment2/scripts/01_download_recipes.py --strict   # category label must be confirmed by the title
```

Sources: recipetineats.com, budgetbytes.com, seriouseats.com, cafedelites.com, loveandlemons.com,
sallysbakingaddiction.com. The script uses the WP Recipe Maker REST API where it exists and falls back to the
schema.org Recipe data on each page. It waits 1 to 2 s between requests and follows robots.txt. It is
resumable: rerun the same command to continue. See the docstring at the top of the script for the full rules.

This step only downloads and labels. Cleaning, deduplication and splitting come later.

## Layout

```
data/
  raw/<class>/<site>_<recipe id>.<ext>   downloaded images. Not in git.
  metadata.csv     filename, class, source_site, recipe_title, recipe_url, image_url
  processed.csv    every recipe looked at, with its outcome (downloaded or skip reason)
scripts/
  01_download_recipes.py
```

## Known issues

- Serious Eats' robots.txt blocks AI and ML crawlers by name. Decide whether to keep it as a source.
- Some images may arrive as WebP, which `image_dataset_from_directory` cannot read. Convert them when cleaning.
- stir_fry is the thinnest class. Check the counts after the full run.
