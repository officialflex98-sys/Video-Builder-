"""
Generates a single voiceover WAV file from the "Voice-over Script" column
in script.txt, using the Gemini API's text-to-speech model.

Run this BEFORE Build_video.py. It writes ./voiceover.wav, and Build_video.py
will automatically pick that file up as AUDIO_PATH if it exists (see the
AUDIO_PATH default there) - no manual edit needed once both steps run in CI.

Requires: pip install google-genai
Env vars:  GEMINI_API_KEY (required)

Notes / limits:
- Each Gemini TTS call caps input around ~8,000 bytes and output around ~655s
  of audio. This script generates one call PER SCENE (so long scripts are
  safe) and stitches the resulting WAV clips together in order.
- Output audio from the API is raw 16-bit PCM, mono, 24kHz - wrapped into a
  proper .wav file here via the standard `wave` module.
"""

import os
import sys
import time
import wave
from typing import List

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from Build_video import parse_scene_script, SCRIPT_PATH  # reuse the same parser

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TTS_MODEL = "gemini-3.1-flash-tts-preview"
VOICE_NAME = "Kore"          # pick any prebuilt voice: Kore, Puck, Charon, etc.
OUTPUT_PATH = "voiceover.wav"
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2              # 16-bit PCM

# Short silence inserted between scenes so lines don't run into each other.
GAP_SECONDS = 0.4

# Free tier for this model allows only 3 requests/minute. Space calls out
# well beyond that (20s would be the bare minimum) to avoid tripping the
# limit in the first place, and retry with backoff if it happens anyway.
MIN_SECONDS_BETWEEN_CALLS = 22
MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 20


def _synthesize(client: "genai.Client", text: str) -> bytes:
    """Calls Gemini TTS for one chunk of text, returns raw PCM bytes.
    Retries with backoff if the free-tier rate limit is hit (429)."""
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
            if not is_quota_error or attempt == MAX_RETRIES:
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
    lines = [s.voiceover.strip() for s in scenes if s.voiceover.strip()]

    if not lines:
        raise SystemExit(
            "No voice-over text found in script.txt - make sure the 4th "
            "column (Voice-over Script) is populated for at least one scene."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    silence_chunk = b"\x00" * int(SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * GAP_SECONDS)

    pcm_chunks: List[bytes] = []
    for i, line in enumerate(lines, 1):
        print(f"[{i}/{len(lines)}] Synthesizing: {line[:60]!r}...")
        pcm_chunks.append(_synthesize(client, line))
        if i != len(lines):
            pcm_chunks.append(silence_chunk)
            time.sleep(MIN_SECONDS_BETWEEN_CALLS)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(pcm_chunks))

    total_bytes = sum(len(c) for c in pcm_chunks)
    duration = total_bytes / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    print(f"Wrote {output_path} (~{duration:.1f}s)")
    return output_path


if __name__ == "__main__":
    generate_voiceover()
