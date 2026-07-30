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

# Jamendo uses folksonomy tags rather than exact moods - "cinematic" and
# "documentary" tend to return calm, unobtrusive instrumental tracks that
# work well under a voiceover. Override via MUSIC_TAG env var if you want
# a different feel (e.g. "ambient", "inspiring", "emotional").
MUSIC_TAG = os.environ.get("MUSIC_TAG", "cinematic")
OUTPUT_PATH = "music.mp3"


def fetch_background_music(tag: str = MUSIC_TAG, output_path: str = OUTPUT_PATH) -> str:
    if not JAMENDO_CLIENT_ID:
        raise SystemExit("Set the JAMENDO_CLIENT_ID environment variable first.")

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
    results = resp.json().get("results", [])

    if not results:
        raise SystemExit(f"No Jamendo tracks found for tag '{tag}'. Try a different MUSIC_TAG.")

    track = results[0]
    audio_url = track.get("audio")
    if not audio_url:
        raise SystemExit("Top result had no downloadable audio URL - try again or change MUSIC_TAG.")

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
