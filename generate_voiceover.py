"""
Generates the full documentary narration using Gemini's TTS API, reading
the ENTIRE script.txt in a small number of large chunks (not one call per
scene). This works within Gemini's per-call limits - roughly 8,000 input
characters and ~655 seconds (~11 minutes) of output audio per call.

IMPORTANT: the gemini-3.1-flash-tts free tier caps out at just 10 requests
PER DAY, PER PROJECT, PER MODEL - and that's shared across every run of
this script today, not reset per-run. Iterating on Build_video.py by
repeatedly re-running the full GitHub Actions pipeline will burn through
this fast even though a single run only needs a few chunks. If you hit a
"PerDay" 429, retrying won't help until the quota resets (daily, around
midnight Pacific time) - see https://ai.dev/rate-limits for current limits.

Chunks are concatenated afterward into a single continuous voiceover.wav,
so the narration is one uninterrupted vocal performance rather than dozens
of separately-synthesized clips stitched together with audible seams.

Run this BEFORE Build_video.py. It writes ./voiceover.wav. Build_video.py
picks that file up automatically and scales each scene's on-screen duration
proportionally to fit the real narration length - it does not try to match
each scene to an individually-generated audio slice.

If script.txt has no Voice-over Script text at all (e.g. the narration-free
dates/archival-footage-only timeline format, which has no VO column), this
is treated as intentional, not an error: the script exits cleanly with no
voiceover.wav written, and Build_video.py falls back to each scene's own
script.txt timestamp duration instead of a narration-scaled one.

Requires: pip install google-genai
Env vars:  GEMINI_API_KEY (required)
"""

import os
import time
import wave
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from Build_video import parse_scene_script, SCRIPT_PATH  # reuse the same parser

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TTS_MODEL = "gemini-2.5-flash-preview-tts"
VOICE_NAME = "Charon"          # deep, clear, "informative" documentary tone
OUTPUT_PATH = "voiceover.wav"
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2               # 16-bit PCM

# Gemini's documented limit is ~8,000 input characters per call, but in
# practice the full script (or even 6,000-char chunks) has proven too much
# for reliable single-call generation. 3,500 keeps each call comfortably
# smaller and, for a script this length, naturally lands on 3 chunks/calls.
MAX_CHUNK_CHARS = 3500

MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 20


# If the number of chunks a script needs reaches this, print a warning -
# free-tier gemini-3.1-flash-tts caps out at just 10 requests/day, shared
# across every run today (not per-run), so a long script (50+ scenes) can
# use up most or all of the day's quota in a single generation.
CHUNK_COUNT_WARNING_THRESHOLD = 6

import re as _re
_SENTENCE_SPLIT_RE = _re.compile(r"(?<=[.!?])\s+")


def _split_long_text(text: str, max_chars: int):
    """Splits a single scene's voiceover text that's already longer than
    max_chars on its own, breaking on sentence boundaries where possible
    so no one Gemini call ever exceeds max_chars - this matters once
    individual scenes get longer in future scripts."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = ""
        # A single sentence longer than max_chars on its own: hard-split it,
        # there's no cleaner boundary available.
        if len(sentence) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            for j in range(0, len(sentence), max_chars):
                pieces.append(sentence[j:j + max_chars].strip())
            continue
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return [p for p in pieces if p]


def _chunk_scenes(scenes, max_chars: int = MAX_CHUNK_CHARS):
    """Groups consecutive scenes' voice-over lines into chunks, each kept
    under max_chars, so every chunk fits in a single Gemini TTS call. A
    scene whose own voiceover text exceeds max_chars is split internally
    (on sentence boundaries) rather than producing an oversized chunk."""
    chunks = []
    current = []
    current_len = 0

    for scene in scenes:
        text = scene.voiceover.strip()
        if not text:
            continue

        for piece in _split_long_text(text, max_chars):
            if current and current_len + len(piece) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(piece)
            current_len += len(piece) + 1

    if current:
        chunks.append(" ".join(current))

    if len(chunks) >= CHUNK_COUNT_WARNING_THRESHOLD:
        print(f"  [warn] this script needs {len(chunks)} Gemini TTS calls - "
              f"check that's within today's remaining free-tier quota "
              f"before running, or you may hit a PerDay 429 partway through.")

    return chunks


def _synthesize(client: "genai.Client", text: str) -> bytes:
    """Calls Gemini TTS once for a chunk of the script, returns raw PCM
    bytes. Retries with backoff on transient errors: rate limits (429,
    excluding per-day caps, which retrying can't fix) and server-side
    errors (5xx, e.g. transient "500 INTERNAL"). ClientError and ServerError
    are siblings in this SDK - neither is a subclass of the other - so both
    must be caught via their shared base APIError, or 5xx errors slip past
    unretried."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                        )
                    ),
                ),
            )
            return response.candidates[0].content.parts[0].inline_data.data
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            is_server_error = isinstance(e, genai_errors.ServerError) or (
                code is not None and 500 <= code < 600
            )
            is_quota_error = code == 429 or "RESOURCE_EXHAUSTED" in str(e)
            is_daily_cap = "PerDay" in str(e)
            is_transient = is_server_error or (is_quota_error and not is_daily_cap)

            if not is_transient or attempt == MAX_RETRIES:
                if is_daily_cap:
                    print("  [quota] This is a PER-DAY quota - retrying won't help until it "
                          "resets.")
                raise
            wait = DEFAULT_RETRY_DELAY * attempt
            reason = "server error" if is_server_error else "rate limit"
            print(f"  [{reason}] hit a transient error, retrying in {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)

    raise RuntimeError("Unreachable: retries exhausted without raising")


def generate_voiceover(script_path: str = SCRIPT_PATH, output_path: str = OUTPUT_PATH) -> Optional[str]:
    if not GEMINI_API_KEY:
        raise SystemExit("Set the GEMINI_API_KEY environment variable first.")

    scenes = parse_scene_script(script_path)
    if not scenes:
        raise SystemExit(f"No scenes parsed from {script_path}")

    chunks = _chunk_scenes(scenes)
    if not chunks:
        # Not an error: a script with no Voice-over Script text at all is a
        # valid, intentional format (e.g. the narration-free dates/archival
        # -footage timeline prompt), not a documentary script someone forgot
        # to fill in. Exit cleanly (code 0) rather than failing the
        # workflow - Build_video.py already falls back to each scene's own
        # script.txt timestamp duration when no voiceover.wav exists.
        print(f"No voice-over text found in any of the {len(scenes)} scene(s) parsed from "
              f"{script_path}. Treating this as a narration-free script (e.g. a dates/"
              f"archival-footage-only timeline) and skipping TTS generation - "
              f"Build_video.py will use each scene's own script.txt timestamp duration "
              f"instead of a narration-scaled one.")
        return None

    print(f"Synthesizing {len(chunks)} chunk(s) covering {len(scenes)} scenes "
          f"with voice '{VOICE_NAME}'...")

    client = genai.Client(api_key=GEMINI_API_KEY)
    all_pcm = bytearray()

    for i, chunk_text in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] synthesizing ({len(chunk_text)} chars)...")
        pcm_data = _synthesize(client, chunk_text)
        all_pcm.extend(pcm_data)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(all_pcm))

    duration = len(all_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    print(f"Wrote {output_path} (~{duration:.1f}s total, {len(chunks)} chunk(s) concatenated)")
    return output_path


if __name__ == "__main__":
    generate_voiceover()
