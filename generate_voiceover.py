"""
Generates the full documentary narration using Gemini's TTS API, reading
the ENTIRE script.txt in a small number of large chunks (not one call per
scene). This works within Gemini's per-call limits - roughly 8,000 input
characters and ~655 seconds (~11 minutes) of output audio per call - while
keeping the total call count per day comfortably under the free tier's
10-15 requests/day cap.

Chunks are concatenated afterward into a single continuous voiceover.wav,
so the narration is one uninterrupted vocal performance rather than dozens
of separately-synthesized clips stitched together with audible seams.

Run this BEFORE Build_video.py. It writes ./voiceover.wav. Build_video.py
picks that file up automatically and scales each scene's on-screen duration
proportionally to fit the real narration length - it does not try to match
each scene to an individually-generated audio slice.

Requires: pip install google-genai
Env vars:  GEMINI_API_KEY (required)
"""

import os
import time
import wave

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from Build_video import parse_scene_script, SCRIPT_PATH  # reuse the same parser

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TTS_MODEL = "gemini-3.1-flash-tts-preview"
VOICE_NAME = "Charon"          # deep, clear, "informative" documentary tone
OUTPUT_PATH = "voiceover.wav"
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2               # 16-bit PCM

# Stay safely under Gemini's ~8,000 character input limit per call.
MAX_CHUNK_CHARS = 6000

MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 20


def _chunk_scenes(scenes, max_chars: int = MAX_CHUNK_CHARS):
    """Groups consecutive scenes' voice-over lines into chunks, each kept
    under max_chars, so every chunk fits in a single Gemini TTS call."""
    chunks = []
    current = []
    current_len = 0

    for scene in scenes:
        text = scene.voiceover.strip()
        if not text:
            continue
        if current and current_len + len(text) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(text)
        current_len += len(text) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _synthesize(client: "genai.Client", text: str) -> bytes:
    """Calls Gemini TTS once for a chunk of the script, returns raw PCM
    bytes. Retries with backoff on transient rate limits (429)."""
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
        except genai_errors.ClientError as e:
            is_quota_error = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            is_daily_cap = "PerDay" in str(e)
            if not is_quota_error or attempt == MAX_RETRIES:
                if is_daily_cap:
                    print("  [quota] This is a PER-DAY quota - retrying won't help until it "
                          "resets.")
                raise
            wait = DEFAULT_RETRY_DELAY * attempt
            print(f"  [rate limit] hit quota, retrying in {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)

    raise RuntimeError("Unreachable: retries exhausted without raising")


def generate_voiceover(script_path: str = SCRIPT_PATH, output_path: str = OUTPUT_PATH) -> str:
    if not GEMINI_API_KEY:
        raise SystemExit("Set the GEMINI_API_KEY environment variable first.")

    scenes = parse_scene_script(script_path)
    if not scenes:
        raise SystemExit(f"No scenes parsed from {script_path}")

    chunks = _chunk_scenes(scenes)
    if not chunks:
        raise SystemExit(
            "No voice-over text found in script.txt - make sure the "
            "Voice-over Script column is populated for at least one scene."
        )

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
