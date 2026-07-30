"""
Generates ONE voiceover audio file PER SCENE using edge-tts (free, no daily
request cap), based on the "Voice-over Script" column in script.txt.

Run this BEFORE Build_video.py. It writes one MP3 per scene into
./voiceover_segments/, named scene_000.mp3, scene_001.mp3, etc., matching
each scene's position in script.txt. Build_video.py automatically loads
these files, uses each one's real duration to drive that scene's length,
and mixes the audio in per scene before concatenation.

Requires: pip install edge-tts>=7.0.0
No API key needed. edge-tts has no meaningful daily request quota, so it's
safe to call once per scene even on 30-40+ scene scripts - unlike Gemini's
TTS models, which cap out at roughly 10-15 requests/day on the free tier.

Rate limiting: Microsoft's speech endpoint will silently rate-limit or drop
connections (surfacing as NoAudioReceived, not a clean error) if requests
fire back-to-back with no gap. A short delay is added between every scene's
synthesis call, and retries use a longer backoff, so a 30-40 scene script
doesn't hammer the endpoint fast enough to trigger this.
"""

import os
import asyncio

import edge_tts

from Build_video import parse_scene_script, SCRIPT_PATH  # reuse the same parser

VOICE_NAME = "en-US-DavisNeural"   # deep, warm, cinematic/documentary tone
OUTPUT_DIR = "voiceover_segments"

MAX_RETRIES = 4
RETRY_BASE_DELAY = 10          # seconds, doubles each retry attempt
SCENE_DELAY_SECONDS = 3        # pause between each scene's synthesis call


async def _synthesize_scene(text: str, dest: str, voice: str = VOICE_NAME):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(dest)
            return
        except Exception as e:
            last_error = e
            wait = RETRY_BASE_DELAY * attempt
            print(f"    [retry] synthesis failed ({e.__class__.__name__}: {e}), "
                  f"retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed to synthesize after {MAX_RETRIES} attempts: {last_error}")


async def _generate_all(scenes, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, scene in enumerate(scenes):
        text = scene.voiceover.strip()
        if not text:
            print(f"  [skip] scene {i} has no voice-over text, no audio will be generated")
            paths.append(None)
            continue
        dest = os.path.join(output_dir, f"scene_{i:03d}.mp3")
        print(f"  [{i + 1}/{len(scenes)}] synthesizing ({len(text)} chars) -> {dest}")
        await _synthesize_scene(text, dest)
        paths.append(dest)
        if i < len(scenes) - 1:
            await asyncio.sleep(SCENE_DELAY_SECONDS)
    return paths


def generate_voiceover(script_path: str = SCRIPT_PATH, output_dir: str = OUTPUT_DIR):
    scenes = parse_scene_script(script_path)
    if not scenes:
        raise SystemExit(f"No scenes parsed from {script_path}")

    has_any_vo = any(s.voiceover.strip() for s in scenes)
    if not has_any_vo:
        raise SystemExit(
            "No voice-over text found in script.txt - make sure the "
            "Voice-over Script column is populated for at least one scene."
        )

    print(f"Synthesizing {len(scenes)} per-scene voiceover files with voice '{VOICE_NAME}'...")
    paths = asyncio.run(_generate_all(scenes, output_dir))
    made = sum(1 for p in paths if p)
    print(f"Done -> {made}/{len(scenes)} scene audio files written to {output_dir}/")
    return paths


if __name__ == "__main__":
    generate_voiceover()
