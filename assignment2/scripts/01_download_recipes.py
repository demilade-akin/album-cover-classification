"""
Download recipe hero images for a 6-class food dataset: pasta, soup, salad, cake, cookies, stir_fry.

This step only downloads and labels. It does no cleaning, deduplication, format conversion or splitting.

Collection, per site
  1. WP Recipe Maker REST API (/wp-json/wp/v2/wprm_recipe), 100 recipes per page. The parent posts of each
     page are fetched in one batched request (posts?include=...), and their featured images in one more
     (media?include=...). That is 3 requests per 100 recipes instead of 1 + 100.
  2. If the WPRM API is unavailable on a site, the script reads recipe URLs from the site's sitemaps and
     parses the schema.org Recipe JSON-LD on each page (1 request per recipe).
  3. If a single WPRM recipe has no usable image, the script reads the JSON-LD on its parent page.

Hero image
  WPRM: the parent post's featured image, else the recipe card image (resize query removed).
  JSON-LD: Recipe.image, the original upload rather than a crop. Step-by-step photos are never fetched.

Labelling (keyword lists in CLASS_PATTERNS below)
  1. Match the recipe's categories (WP post categories + WPRM course tags, or JSON-LD recipeCategory).
  2. If no category matches, match the title.
  3. Exactly one class matched: keep it. Zero, or more than one: skip it and log the reason.
  4. Title veto: if the title names something that is not the class (banana bread filed under "Cakes"),
     skip it. Sites file loosely, so a category match alone is not trusted for these cases.
  5. Strict mode (--strict): a category label is kept only if the title also names that class. Budget
     Bytes files Chicken Stroganoff and a turkey skillet under "Pasta", for example. Strict mode trades
     volume for cleaner labels. Title-only labels are unaffected, since the title already confirms them.
  Category wins over title. A "Salads" post called "Pasta Salad" is labelled salad.

Politeness
  One request at a time, 1 to 2 s random gap between any two requests (longer if robots.txt sets a
  Crawl-delay). robots.txt is checked for every host, including image CDNs, with RFC 9309 rules (4xx on
  robots.txt = no rules, 5xx or no connection = disallowed). A site whose homepage or API refuses us (for
  example a Cloudflare 403 challenge) is skipped. The script never tries to get past bot protection.

Output (under assignment2/data/)
  raw/<class>/<site>_<recipe id>.<ext>   original bytes as served (jpg, png or webp)
  metadata.csv                           filename, class, source_site, recipe_title, recipe_url (+ image_url)
  processed.csv                          every recipe looked at, with status (downloaded / skip reason)

Resumable: recipes listed in processed.csv are skipped on the next run, so a rerun only does new work.
Failed downloads are not logged as processed, so they are retried. Delete processed.csv to re-evaluate
skipped recipes (after editing the keyword lists, for example).

Usage (from anywhere):
  python assignment2/scripts/01_download_recipes.py              all sites, up to 150 images per class per site
  python assignment2/scripts/01_download_recipes.py --strict     cleaner labels, fewer images
  python assignment2/scripts/01_download_recipes.py --sites cafedelites.com --max-per-class 20
  python assignment2/scripts/01_download_recipes.py --force-jsonld --sites loveandlemons.com   (test the fallback)
"""
import argparse, csv, html, json, os, random, re, time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()   # assignment2 folder
OUT_DIR = ROOT / "data"
RAW_DIR = OUT_DIR / "raw"
META_PATH = OUT_DIR / "metadata.csv"
LOG_PATH = OUT_DIR / "processed.csv"

# Descriptive user agent. Put a contact address in the RECIPE_CONTACT environment variable if you want
# site owners to be able to reach you (good practice for scrapers).
CONTACT = os.environ.get("RECIPE_CONTACT", "no contact given")
USER_AGENT = f"TME6015-RecipeDataset/0.1 (university coursework image classifier; {CONTACT})"
ROBOTS_AGENT = "TME6015-RecipeDataset"   # the token matched against User-agent lines in robots.txt

SITES = ["recipetineats.com", "budgetbytes.com", "seriouseats.com",
         "cafedelites.com", "loveandlemons.com", "sallysbakingaddiction.com"]

MIN_GAP, MAX_GAP = 1.0, 2.0   # seconds between requests
TIMEOUT = 30
MAX_RETRIES = 3

# ---------------------------------------------------------------------------------------------------------
# Class keywords. Each class has patterns that count as a match and patterns that cancel a match
# (a crab cake is not a cake). Matching is case-insensitive on whole words. Edit these to change the labels,
# and report the final lists in the write-up, because they define what each class means.
# ---------------------------------------------------------------------------------------------------------
CLASS_PATTERNS = {
    "pasta": {
        "include": [r"pastas?", r"spaghetti", r"penne", r"lasagnas?", r"lasagnes?", r"fettuccin[ie]",
                    r"linguine", r"macaroni", r"mac (and|&|n) cheese", r"rigatoni", r"orzo", r"ravioli",
                    r"tortellini", r"tagliatelle", r"pappardelle", r"bucatini", r"carbonara", r"bolognese"],
        "exclude": [r"pasta sauce", r"pasta dough"],
        "veto": [r"sauce", r"dough"],
    },
    "soup": {
        "include": [r"soups?", r"chowders?", r"bisque", r"minestrone", r"gazpacho"],
        "exclude": [r"soup dumplings?"],
        "veto": [r"dumplings?"],
    },
    "salad": {
        "include": [r"salads?", r"slaw", r"coleslaw"],
        "exclude": [r"salad dressings?", r"dressings?", r"vinaigrettes?"],
        "veto": [r"dressings?", r"vinaigrettes?"],
    },
    "cake": {
        "include": [r"cakes?", r"cheesecakes?", r"cupcakes?", r"bundt"],
        "exclude": [r"(crab|fish|rice|potato|salmon|corn|tuna|zucchini|veggie|vegetable) cakes?",
                    r"pancakes?", r"cake pops?", r"cake mix cookies", r"cake (batter|flour|mix)",
                    r"funnel cakes?"],
        "veto": [r"breads?", r"muffins?", r"brownies?", r"bars?", r"scones?", r"pancakes?", r"frosting",
                 r"icing", r"glaze", r"donuts?", r"doughnuts?", r"cinnamon rolls?", r"treats",
                 r"substitute", r"bouquet"],
    },
    "cookies": {
        "include": [r"cookies?", r"biscotti", r"snickerdoodles?", r"shortbread", r"macarons?"],
        "exclude": [r"cookie dough dip", r"edible cookie dough", r"cookie butter", r"cookie bars?"],
        "veto": [r"bars?", r"dough", r"dip"],
    },
    "stir_fry": {
        "include": [r"stir[- ]?fr(y|ies|ied)", r"lo mein", r"chow mein", r"fried rice", r"pad thai"],
        "exclude": [],
        "veto": [r"sauce", r"marinade"],
    },
}
CLASSES = list(CLASS_PATTERNS)
_COMPILED = {c: ([re.compile(rf"\b{p}\b", re.I) for p in v["include"]],
                 [re.compile(rf"\b{p}\b", re.I) for p in v["exclude"]])
             for c, v in CLASS_PATTERNS.items()}
_VETO = {c: [re.compile(rf"\b{p}\b", re.I) for p in v.get("veto", [])] for c, v in CLASS_PATTERNS.items()}

META_FIELDS = ["filename", "class", "source_site", "recipe_title", "recipe_url", "image_url"]
LOG_FIELDS = ["source_site", "recipe_key", "recipe_url", "status", "detail"]


# ======================================================================== labelling

def classes_in(text):
    """Return the set of classes whose patterns match text, after removing excluded phrases."""
    text = html.unescape(text or "")
    found = set()
    for cls, (inc, exc) in _COMPILED.items():
        cleaned = text
        for pat in exc:                       # blank out excluded phrases first, then look for includes
            cleaned = pat.sub(" ", cleaned)
        if any(p.search(cleaned) for p in inc):
            found.add(cls)
    return found


STRICT = False   # set by --strict


def assign_label(categories, title):
    """Categories first, title as fallback, then the title veto (and in strict mode, title confirmation).
    Returns (class or None, status, detail)."""
    cls, status, detail = _match(categories, title)
    if cls and STRICT and detail == "category" and cls not in classes_in(title):
        return None, "skip_unconfirmed", f"{cls} from category, title does not name it"
    if cls:
        hit = next((p.pattern for p in _VETO[cls] if p.search(html.unescape(title or ""))), None)
        if hit:
            return None, "skip_veto", f"{cls} vetoed by title ({hit})"
    return cls, status, detail


def _match(categories, title):
    from_cats = set()
    for cat in categories:
        from_cats |= classes_in(cat)
    if len(from_cats) == 1:
        return from_cats.pop(), "ok", "category"
    if len(from_cats) > 1:
        return None, "skip_ambiguous", "categories: " + ",".join(sorted(from_cats))
    from_title = classes_in(title)
    if len(from_title) == 1:
        return from_title.pop(), "ok", "title"
    if len(from_title) > 1:
        return None, "skip_ambiguous", "title: " + ",".join(sorted(from_title))
    return None, "skip_no_match", ""


# ======================================================================== polite HTTP

class PoliteSession:
    """One requests session with a global 1-2 s gap between calls and a robots.txt check per host."""

    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.last = 0.0
        self.robots = {}        # host -> RobotFileParser
        self.delays = {}        # host -> crawl delay in seconds (0 if none)

    def _wait(self, host):
        gap = max(random.uniform(MIN_GAP, MAX_GAP), self.delays.get(host, 0))
        sleep_for = self.last + gap - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        self.last = time.time()

    def allowed(self, url):
        """Read robots.txt for the URL's host once, following RFC 9309: a 4xx (missing or forbidden)
        means no rules apply, a 5xx or network failure means stay out completely."""
        p = urlparse(url)
        host = p.netloc
        if host not in self.robots:
            rp = RobotFileParser()
            robots_url = f"{p.scheme}://{host}/robots.txt"
            self._wait(host)
            try:
                r = self.s.get(robots_url, timeout=TIMEOUT)
                if r.status_code >= 500:
                    rp.disallow_all = True     # server error: assume complete disallow (RFC 9309 2.3.1.4)
                elif r.status_code >= 400:
                    rp.allow_all = True        # unavailable: crawler may access (RFC 9309 2.3.1.3)
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp.disallow_all = True         # cannot check, so do not crawl
            self.robots[host] = rp
            self.delays[host] = float(rp.crawl_delay(ROBOTS_AGENT) or 0)
        return self.robots[host].can_fetch(ROBOTS_AGENT, url)

    def get(self, url, **kw):
        """GET with robots check, rate limit and retries on 429/5xx. Returns a Response or None."""
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        for attempt in range(MAX_RETRIES):
            self._wait(host)
            try:
                r = self.s.get(url, timeout=TIMEOUT, **kw)
            except requests.RequestException as e:
                print(f"    network error ({e.__class__.__name__}), attempt {attempt + 1}")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = int(r.headers.get("Retry-After", 0) or 0) or 2 ** (attempt + 1)
                print(f"    HTTP {r.status_code}, waiting {wait}s")
                time.sleep(min(wait, 60))
                continue
            return r
        return None


# ======================================================================== resumable state

def load_state():
    """Read processed.csv and metadata.csv so a rerun skips finished recipes."""
    done = set()
    if LOG_PATH.exists():
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            done = {(row["source_site"], row["recipe_key"]) for row in csv.DictReader(f)}
    counts = Counter()                       # (site, class) -> images already downloaded
    if META_PATH.exists():
        with open(META_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (RAW_DIR / row["class"] / row["filename"]).exists():
                    counts[(row["source_site"], row["class"])] += 1
    return done, counts


def append_row(path, fields, row):
    """Append one row and flush, so an interrupted run loses at most the current recipe."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


# ======================================================================== image download

EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def full_size(url):
    """Drop resize queries such as ?fit=150,188 or ?w=300 so the original upload is fetched."""
    p = urlparse(url)
    return urlunparse(p._replace(query="")) if p.query else url


def download_image(sess, url, cls, stem):
    """Save the image to raw/<cls>/<stem>.<ext>. Returns the filename or raises."""
    # Ask for JPEG or PNG first: image CDNs often serve AVIF otherwise, which Keras cannot decode.
    r = sess.get(url, headers={"Accept": "image/jpeg,image/png;q=0.9,image/webp;q=0.5"})
    if r is None:
        raise IOError("no response after retries")
    if r.status_code != 200:
        raise IOError(f"HTTP {r.status_code}")
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if ctype not in EXT:
        raise IOError(f"unsupported content type {ctype or 'none'}")
    name = stem + EXT[ctype]
    dest = RAW_DIR / cls / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)                         # atomic: no half-written image survives a crash
    return name


# ======================================================================== JSON-LD

# Quotes around the type attribute are optional: minified pages drop them.
LDJSON_RE = re.compile(r'<script[^>]+type=["\']?application/ld\+json["\']?[^>]*>(.*?)</script>', re.S | re.I)


def find_recipe_ld(page_html):
    """Return the first schema.org Recipe object on the page, or None. Handles @graph and lists."""
    def walk(node):
        if isinstance(node, list):
            for n in node:
                hit = walk(n)
                if hit:
                    return hit
        elif isinstance(node, dict):
            t = node.get("@type")
            types = t if isinstance(t, list) else [t]
            if "Recipe" in types:
                return node
            if "@graph" in node:
                return walk(node["@graph"])
        return None

    for block in LDJSON_RE.findall(page_html):
        try:
            hit = walk(json.loads(block.strip()))
        except json.JSONDecodeError:
            continue
        if hit:
            return hit
    return None


SIZE_SUFFIX = re.compile(r"-(\d+)x(\d+)\.\w+$")


def ld_image(recipe):
    """The hero image from Recipe.image (a string, a list, or ImageObjects). The list usually holds crops
    of the same photo (500x500, 480x270, ...), so take the original upload, else the largest crop."""
    img = recipe.get("image")
    items = img if isinstance(img, list) else [img]
    urls = []
    for it in items:
        if isinstance(it, dict):
            it = it.get("url") or it.get("contentUrl")
        if isinstance(it, str):
            urls.append(it)
    if not urls:
        return None
    originals = [u for u in urls if not SIZE_SUFFIX.search(urlparse(u).path)]
    if originals:
        return originals[0]
    def area(u):
        m = SIZE_SUFFIX.search(urlparse(u).path)
        return int(m.group(1)) * int(m.group(2))
    return max(urls, key=area)


def ld_list(value):
    """recipeCategory can be a string, a comma-separated string, or a list."""
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def parse_recipe_page(sess, url):
    """Fetch a page and return (title, categories, image_url) from its Recipe JSON-LD, or None."""
    r = sess.get(url)
    if r is not None and r.status_code in (401, 403, 429):
        return "blocked"
    if r is None or r.status_code != 200:
        return None
    recipe = find_recipe_ld(r.text)
    if not recipe:
        return None
    return html.unescape(str(recipe.get("name", ""))), ld_list(recipe.get("recipeCategory")), ld_image(recipe)


# ======================================================================== per-recipe processing

class Run:
    def __init__(self, sess, max_per_class):
        self.sess = sess
        self.max_per_class = max_per_class
        self.done, self.counts = load_state()
        self.stats = Counter()               # (site, status) -> recipes

    def full(self, site, cls):
        return self.max_per_class and self.counts[(site, cls)] >= self.max_per_class

    def site_full(self, site):
        return self.max_per_class and all(self.full(site, c) for c in CLASSES)

    def log(self, site, key, url, status, detail=""):
        self.stats[(site, status)] += 1
        # A failed download is not recorded, so the next run retries it.
        if status != "fail_download":
            append_row(LOG_PATH, LOG_FIELDS, {"source_site": site, "recipe_key": key, "recipe_url": url,
                                              "status": status, "detail": detail})
            self.done.add((site, key))

    def handle(self, site, key, url, title, categories, image_url):
        """Label one recipe and download its hero image if it gets exactly one class."""
        cls, status, detail = assign_label(categories, title)
        if cls is None:
            self.log(site, key, url, status, detail)
            return
        if self.full(site, cls):
            self.stats[(site, "skip_class_full")] += 1   # not logged: a later run with a higher cap can use it
            return
        if not image_url and url:
            try:
                parsed = parse_recipe_page(self.sess, url)    # last resort: JSON-LD on the recipe page
            except PermissionError:
                parsed = None
            image_url = parsed[2] if isinstance(parsed, tuple) else None
        if not image_url:
            self.log(site, key, url, "skip_no_image")
            return
        stem = f"{site.split('.')[0]}_{key}"
        try:
            name = download_image(self.sess, image_url, cls, stem)
        except (IOError, PermissionError) as e:
            print(f"    image failed for {url}: {e}")
            self.log(site, key, url, "fail_download", str(e))
            return
        append_row(META_PATH, META_FIELDS, {"filename": name, "class": cls, "source_site": site,
                                            "recipe_title": title, "recipe_url": url, "image_url": image_url})
        self.counts[(site, cls)] += 1
        self.log(site, key, url, "downloaded", f"{cls} ({detail})")


# ======================================================================== WPRM collection

def wp_get_json(sess, url):
    try:
        r = sess.get(url)
    except PermissionError:
        return None, None
    if r is None or r.status_code != 200:
        return None, r
    try:
        return r.json(), r
    except ValueError:
        return None, r


def wp_categories(sess, base):
    """All post category names for a site, id -> name. One request per 100 categories."""
    names, page = {}, 1
    while True:
        data, r = wp_get_json(sess, f"{base}/wp-json/wp/v2/categories?per_page=100&page={page}&_fields=id,name")
        if not data:
            break
        names.update({c["id"]: html.unescape(c["name"]) for c in data})
        if page >= int(r.headers.get("X-WP-TotalPages", 1)):
            break
        page += 1
    return names


def collect_wprm(run, site, base, max_pages):
    """Walk the wprm_recipe endpoint. Returns False if the endpoint is unusable on this site."""
    sess = run.sess
    fields = "id,link,recipe.name,recipe.image_url,recipe.parent_post_id,recipe.type,recipe.tags.course"
    first, r = wp_get_json(sess, f"{base}/wp-json/wp/v2/wprm_recipe?per_page=100&page=1&_fields={fields}")
    if first is None:
        print(f"  WPRM API unavailable (HTTP {r.status_code if r is not None else 'none'})")
        return False
    total_pages = int(r.headers.get("X-WP-TotalPages", 1))
    print(f"  WPRM API: {r.headers.get('X-WP-Total', '?')} recipes on {total_pages} pages")
    cat_names = wp_categories(sess, base)

    page, data = 1, first
    while data:
        print(f"  page {page}/{total_pages}  counts: " +
              ", ".join(f"{c}={run.counts[(site, c)]}" for c in CLASSES))
        recipes = [d for d in data if (d.get("recipe") or {}).get("type", "food") == "food"
                   and (site, str(d["id"])) not in run.done]

        # Batch 1: parent posts of this page (categories, link, title, featured image id)
        parent_ids = sorted({str(d["recipe"].get("parent_post_id")) for d in recipes
                             if d["recipe"].get("parent_post_id") not in (None, "", "0", 0)})
        posts = {}
        if parent_ids:
            pdata, _ = wp_get_json(sess, f"{base}/wp-json/wp/v2/posts?include={','.join(parent_ids)}"
                                         f"&per_page=100&_fields=id,link,title,categories,featured_media")
            posts = {str(p["id"]): p for p in (pdata or [])}

        # Batch 2: featured images of those posts (the hero image)
        media_ids = sorted({str(p["featured_media"]) for p in posts.values() if p.get("featured_media")})
        media = {}
        if media_ids:
            mdata, _ = wp_get_json(sess, f"{base}/wp-json/wp/v2/media?include={','.join(media_ids)}"
                                         f"&per_page=100&_fields=id,source_url")
            media = {str(m["id"]): m.get("source_url") for m in (mdata or [])}

        for d in recipes:
            rec = d["recipe"]
            post = posts.get(str(rec.get("parent_post_id")))
            cats = [t["name"] for t in (rec.get("tags") or {}).get("course") or []]
            if post:
                cats += [cat_names.get(cid, "") for cid in post.get("categories", [])]
                url = post["link"]
                title = html.unescape(post["title"]["rendered"]) or rec.get("name", "")
                image = media.get(str(post.get("featured_media")))
            else:   # parent not public (draft, private, or a page): no real recipe URL to record
                run.log(site, str(d["id"]), d.get("link", ""), "skip_no_parent")
                continue
            if not image and rec.get("image_url"):
                image = full_size(rec["image_url"])
            run.handle(site, str(d["id"]), url, title, cats, image)
            if run.site_full(site):
                print("  every class reached the cap for this site")
                return True

        page += 1
        if page > total_pages or (max_pages and page > max_pages):
            break
        data, r = wp_get_json(sess, f"{base}/wp-json/wp/v2/wprm_recipe?per_page=100&page={page}&_fields={fields}")
    return True


# ======================================================================== JSON-LD collection (fallback)

LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.S)
SKIP_SITEMAP = re.compile(r"(category|tag|author|page-sitemap|image|video|wprm|web-story|attachment)", re.I)


def sitemap_urls(sess, base, limit_sitemaps=60):
    """Collect page URLs from the sitemaps named in robots.txt (or the usual default paths)."""
    roots = []
    r = sess.get(f"{base}/robots.txt")
    if r is not None and r.status_code == 200:
        roots = re.findall(r"(?im)^sitemap:\s*(\S+)", r.text)
    roots = roots or [f"{base}/sitemap_index.xml", f"{base}/sitemap.xml"]
    queue, seen, pages = list(roots), set(), []
    while queue and len(seen) < limit_sitemaps:
        sm = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        try:
            r = sess.get(sm)
        except PermissionError:
            continue
        if r is None or r.status_code != 200:
            continue
        for loc in LOC_RE.findall(r.text):
            loc = html.unescape(loc)
            if loc.split("?")[0].endswith(".xml"):
                if not SKIP_SITEMAP.search(loc):
                    queue.append(loc)
            else:
                pages.append(loc)
    return list(dict.fromkeys(pages))


def collect_jsonld(run, site, base, max_pages):
    """Fallback: sitemap URLs, then Recipe JSON-LD on each page. One request per recipe."""
    urls = sitemap_urls(run.sess, base)
    print(f"  sitemaps: {len(urls)} page URLs")
    if not urls:
        return
    # Fetch URLs whose slug already names a class first. This changes only the order, not the labels,
    # so the per-class cap fills with fewer wasted page fetches.
    urls.sort(key=lambda u: 0 if classes_in(urlparse(u).path.replace("-", " ")) else 1)
    fetched, blocked_in_a_row = 0, 0
    for url in urls:
        key = urlparse(url).path.strip("/").replace("/", "_")[:120] or "home"
        if (site, key) in run.done:
            continue
        if max_pages and fetched >= max_pages * 100:     # same budget scale as WPRM pages
            print("  page budget reached")
            break
        if run.site_full(site):
            print("  every class reached the cap for this site")
            break
        fetched += 1
        try:
            parsed = parse_recipe_page(run.sess, url)
        except PermissionError:
            run.log(site, key, url, "skip_robots")
            continue
        if parsed == "blocked":
            blocked_in_a_row += 1
            if blocked_in_a_row >= 5:
                print("  5 refusals (401/403/429) in a row: the site is blocking us, stopping it")
                break
            continue                         # not logged, so a later run retries it
        blocked_in_a_row = 0
        if parsed is None:
            run.log(site, key, url, "skip_not_recipe")
            continue
        title, cats, image = parsed
        run.handle(site, key, url, title, cats, image)
        if fetched % 25 == 0:
            print(f"  {fetched} pages  counts: " + ", ".join(f"{c}={run.counts[(site, c)]}" for c in CLASSES))


# ======================================================================== main

def resolve_base(sess, site):
    """Find the canonical https base URL (with or without www) by following the homepage redirect."""
    for candidate in (f"https://www.{site}", f"https://{site}"):
        try:
            if not sess.allowed(candidate + "/"):
                continue
            r = sess.get(candidate + "/", allow_redirects=True)
        except PermissionError:
            continue
        if r is not None and r.status_code == 200:
            p = urlparse(r.url)
            return f"{p.scheme}://{p.netloc}"
    return None


def print_summary(run):
    counts = Counter()
    if META_PATH.exists():
        with open(META_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[(row["class"], row["source_site"])] += 1
    sites = sorted({s for _, s in counts})
    print("\nImages per class per source (all runs, from metadata.csv)")
    w = max([len(s) for s in sites] + [10])
    print("class".ljust(10) + "".join(s.split(".")[0][:w].rjust(w + 2) for s in sites) + "total".rjust(8))
    for c in CLASSES:
        row = [counts[(c, s)] for s in sites]
        print(c.ljust(10) + "".join(str(n).rjust(w + 2) for n in row) + str(sum(row)).rjust(8))
    totals = [sum(counts[(c, s)] for c in CLASSES) for s in sites]
    print("total".ljust(10) + "".join(str(n).rjust(w + 2) for n in totals) + str(sum(totals)).rjust(8))

    if run.stats:
        print("\nRecipes handled in this run, by outcome")
        by_site = defaultdict(dict)
        for (site, status), n in run.stats.items():
            by_site[site][status] = n
        for site, d in sorted(by_site.items()):
            print(f"  {site}: " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))


def main():
    global OUT_DIR, RAW_DIR, META_PATH, LOG_PATH, STRICT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=SITES, help="domains to collect from")
    ap.add_argument("--max-per-class", type=int, default=150,
                    help="cap per class per site, 0 for no cap (default 150)")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="WPRM pages of 100 recipes per site, 0 for all (JSON-LD: 100 page fetches per unit)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output folder (default assignment2/data)")
    ap.add_argument("--force-jsonld", action="store_true", help="skip the WPRM API (to test the fallback)")
    ap.add_argument("--strict", action="store_true",
                    help="keep a category label only if the title also names the class")
    args = ap.parse_args()

    OUT_DIR = args.out
    RAW_DIR, META_PATH, LOG_PATH = OUT_DIR / "raw", OUT_DIR / "metadata.csv", OUT_DIR / "processed.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sess = PoliteSession()
    STRICT = args.strict
    run = Run(sess, args.max_per_class)
    print(f"User-Agent: {USER_AGENT}")
    print(f"Labelling: {'strict (category must be confirmed by title)' if STRICT else 'category first, title fallback'}")
    print(f"Resuming with {len(run.done)} recipes already processed\n")

    try:
        for site in args.sites:
            print(f"== {site}")
            if run.site_full(site):
                print("  already at the cap for every class")
                continue
            base = resolve_base(sess, site)
            if base is None:
                print("  skipped: robots.txt unreadable or disallows us, or the site blocks automated "
                      "requests (bot protection is not bypassed)")
                continue
            if args.force_jsonld or not collect_wprm(run, site, base, args.max_pages):
                print("  falling back to sitemap + JSON-LD")
                collect_jsonld(run, site, base, args.max_pages)
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to continue where it stopped.")
    print_summary(run)


if __name__ == "__main__":
    main()
