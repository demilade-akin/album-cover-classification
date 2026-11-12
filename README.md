# TME 6015 coursework

This repo holds two separate pieces of work that share one Python environment. Each folder is self-contained:
its scripts only read and write inside that folder.

| Folder | What it is |
| --- | --- |
| [`assignment2/`](assignment2/README.md) | Assignment 2: 6-class food image classifier (pasta, soup, salad, cake, cookies, stir fry) built from recipe blog photos |
| [`project/`](project/README.md) | Final project: classify any album packaging image (front, back, disc, booklet) into 10 albums. Includes the data report |

## Setup

```
uv sync
```

Shared at the root: `pyproject.toml`, `uv.lock`, `.venv/`. Scripts run from anywhere, for example
`python assignment2/scripts/01_download_recipes.py` or `python project/scripts/04_download_project.py`.
