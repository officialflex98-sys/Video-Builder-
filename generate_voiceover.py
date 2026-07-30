"""
Generates a single voiceover WAV file from the "Voice-over Script" column
in script.txt, using the Gemini API's text-to-speech model.

Run this BEFORE Build_video.py. It writes ./voiceover.wav, and Build_video.py
will automatically pick that file up as AUDIO_PATH if it exists (see the
AUDIO_PATH default there) - no manual edit needed once both steps run in CI.

Requires: pip install google-genai
Env vars:  GEMINI_API_KEY (required)

IMPORTANT - single call, not one per scene:
gemini-3.1-flash-tts-preview's free tier caps out at 10 requests PER DAY
(not per minute) - a script with more than ~10 scenes will always fail if
you call the API once per scene, no matter how much you pace or retry.
Instead, this script joins every scene's voiceover line into ONE prompt and
makes a SINGLE TTS call. Gemini's per-call limits (~8,000 bytes input,
~655s of output audio) comfortably cover a typical few-minute script.
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
VOICE_NAME = "Charon"        # "Informative" style - documentary narrator tone
OUTPUT_PATH = "voiceover.wav"
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2              # 16-bit PCM

MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 20


def _synthesize(client: "genai.Client", text: str) -> bytes:
    """Calls Gemini TTS once for the full script, returns raw PCM bytes.
    Retries with backoff if a transient rate limit is hit (429). Note: if
    the free tier's PER-DAY cap is what's hit, retrying won't help until
    the quota resets - the error message will say so explicitly."""
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
                          "resets. Consider upgrading the Gemini API tier if this recurs.")
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

    full_script = " ".join(lines)
    print(f"Synthesizing full voiceover in one call ({len(full_script)} chars, "
          f"{len(lines)} scenes)...")

    client = genai.Client(api_key=GEMINI_API_KEY)
    pcm_data = _synthesize(client, full_script)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    duration = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    print(f"Wrote {output_path} (~{duration:.1f}s)")
    return output_path


if __name__ == "__main__":
    generate_voiceover()
