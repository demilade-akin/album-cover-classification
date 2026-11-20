"""
Shared HTTP helper for the download scripts (03 and 04).

MusicBrainz asks every client to identify itself with a contact in the User-Agent. The contact is read from the
ALBUM_PROJECT_CONTACT environment variable (an email or URL), so it is not hardcoded in the repo:
    export ALBUM_PROJECT_CONTACT="you@example.com"
Scripts that only read local files never need it. The first network request fails with a clear message if unset.
"""
import os, threading, time

import requests

_local = threading.local()


def user_agent():
    contact = os.environ.get("ALBUM_PROJECT_CONTACT", "akinbobola71@gmail.com").strip()
    if not contact:
        raise SystemExit("Set ALBUM_PROJECT_CONTACT (your email or URL) before any network step. "
                         "MusicBrainz requires a contact in the User-Agent.")
    return f"AlbumCoverProject/0.2 ({contact})"


def session():
    # One requests.Session per thread, since sessions are not guaranteed thread-safe
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update({"User-Agent": user_agent()})
    return _local.s


def get(url, params=None, pause=1.1, retries=5, timeout=60):
    """GET with a polite pause before each try. Returns the response, or None on 404.
    Retries timeouts, connection errors, 429 and 5xx with growing waits (5, 10, 20, 40 s).
    Raises requests.RequestException once the retries run out, or at once on other 4xx codes."""
    for attempt in range(retries):
        time.sleep(pause)
        try:
            r = session().get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if (400 <= code < 500 and code != 429) or attempt == retries - 1:
                raise                              # other client errors will not fix themselves
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(5 * 2 ** attempt)

