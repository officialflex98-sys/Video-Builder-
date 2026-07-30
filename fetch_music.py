"""
Fetches one royalty-free background music track from the Jamendo API and
saves it as music.mp3, which Build_video.py auto-detects and mixes in.

Run this BEFORE Build_video.py (order relative to generate_voiceover.py
doesn't matter - they write different files).

Requires: pip install requests
Env vars:  JAMENDO_CLIENT_ID (required - free, get one at
           https://devportal.jamendo.com/)

Jamendo's API and tracks are free to use for non-commercial projects. If this
video will be monetized (e.g. run ads on YouTube), check the licence field
on the chosen track and see https://licensing.jamendo.com for a commercial
licence - the free API tier doesn't clear that automatically.
"""

import os
import requests

JAMENDO_CLIENT_ID = os.environ.get("JAMENDO_CLIENT_ID", "")
JAMENDO_SEARCH_URL = "https://api.jamendo.com/v3.0/tracks/"

# Jamendo's "tags" filter only matches its own fixed taxonomy - a tag that
# looks reasonable (like "cinematic") can still return zero results if it
# isn't an exact match. Try several, in order, until one actually returns
# tracks. Override the first choice via the MUSIC_TAG env var.
MUSIC_TAGS = [
    os.environ.get("MUSIC_TAG", "cinematic"),
    "documentary",
    "inspiring",
    "ambient",
    "instrumental",
    "chill",
]
OUTPUT_PATH = "music.mp3"


def _search(tag: str) -> list:
    params = {
        "client_id": JAMENDO_CLIENT_ID,
        "format": "json",
        "limit": 5,
        "tags": tag,
        "audioformat": "mp32",
        "order": "popularity_total",
        "include": "musicinfo",
    }
    resp = requests.get(JAMENDO_SEARCH_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # Jamendo returns HTTP 200 even on API-level errors (e.g. bad client_id) -
    # the real status is nested in "headers".
    status = data.get("headers", {}).get("status")
    if status != "success":
        error_msg = data.get("headers", {}).get("error_message", "unknown error")
        raise SystemExit(f"Jamendo API error: {error_msg} - check JAMENDO_CLIENT_ID is valid.")

    return data.get("results", [])


def fetch_background_music(tags=MUSIC_TAGS, output_path: str = OUTPUT_PATH) -> str:
    if not JAMENDO_CLIENT_ID:
        raise SystemExit("Set the JAMENDO_CLIENT_ID environment variable first.")

    results = []
    for tag in tags:
        print(f"Searching Jamendo for tag '{tag}'...")
        results = _search(tag)
        if results:
            print(f"  Found {len(results)} track(s) for '{tag}'")
            break
        print(f"  No results for '{tag}', trying next tag...")

    if not results:
        raise SystemExit(
            f"No Jamendo tracks found for any of {tags}. "
            "Double check JAMENDO_CLIENT_ID is set and valid."
        )

    track = results[0]
    audio_url = track.get("audio")
    if not audio_url:
        raise SystemExit("Top result had no downloadable audio URL - try again.")

    print(f"Selected track: {track.get('name')} by {track.get('artist_name')} "
          f"(license: {track.get('license_ccurl', 'see Jamendo track page')})")

    with requests.get(audio_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)

    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    fetch_background_music()
