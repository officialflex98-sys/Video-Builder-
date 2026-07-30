"""
Scene-script -> video assembler.

Pipeline:
  1. parse_scene_script()   reads script.txt into Scene objects
  2. fetch_clips_for_scene() pulls CLIPS_PER_SCENE unique clips per scene from Pexels
  3. build_scene_clip()     trims/loops those clips to fill the scene's duration
  4. assemble_video()       concatenates scenes, burns in captions, and (if AUDIO_PATH
                            is set) rescales the whole video to match the voiceover
                            track's length instead of the raw timestamp sum.

Requires: pip install moviepy requests python-dotenv
Env vars: PEXELS_API_KEY (required)
          SUPABASE_URL / SUPABASE_KEY (optional - logging only)
"""

import os
import re
import random
import requests
from dataclasses import dataclass, field
from typing import List, Optional

from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
    AudioFileClip,
    vfx,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SCRIPT_PATH = "script.txt"
OUTPUT_PATH = "final_video.mp4"
CLIPS_PER_SCENE = 2
TARGET_RESOLUTION = (1920, 1080)

# moviepy v2 requires a real font FILE (not a font family name like "Arial-Bold").
# This path is present by default on Ubuntu GitHub Actions runners. If you run
# locally on Mac/Windows, point this at a .ttf/.otf file that actually exists.
CAPTION_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Point this at your voiceover file to drive final length off the audio track
# instead of the raw scene timestamps. Leave as None to use timestamps as-is.
# If generate_voiceover.py already ran and produced voiceover.wav in this same
# directory, it's picked up automatically - no manual edit needed.
_DEFAULT_VOICEOVER = "voiceover.wav"
AUDIO_PATH: Optional[str] = (
    _DEFAULT_VOICEOVER if os.path.exists(_DEFAULT_VOICEOVER) else None
)

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class Scene:
    start: float                # seconds
    end: float                  # seconds
    description: str
    keywords: List[str]
    voiceover: str = ""         # optional 4th column, informational only
    clip_paths: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ----------------------------------------------------------------------------
# 1. Parsing
# ----------------------------------------------------------------------------
_TIMESTAMP_RE = re.compile(r"(\d+):(\d+)\s*[–-]\s*(\d+):(\d+)")


def _to_seconds(mm: str, ss: str) -> float:
    return int(mm) * 60 + int(ss)


def _split_keywords(cell: str) -> List[str]:
    """Splits a "kw one", "kw two" cell into a clean list, quotes stripped."""
    parts = [p.strip().strip('"').strip("'") for p in cell.split(",")]
    return [p for p in parts if p]


def _detect_delimiter(line: str) -> str:
    if "\t" in line:
        return "\t"
    if "|" in line:
        return "|"
    # fall back to splitting on 2+ spaces
    return None


def parse_scene_script(path: str = SCRIPT_PATH) -> List[Scene]:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f if l.strip()]

    if not raw_lines:
        return []

    delimiter = _detect_delimiter(raw_lines[0])
    scenes: List[Scene] = []

    for i, line in enumerate(raw_lines):
        if delimiter:
            cols = [c.strip() for c in line.split(delimiter)]
        else:
            cols = [c.strip() for c in re.split(r"\s{2,}", line)]

        if len(cols) < 3:
            continue

        # header row auto-skip: first row has no timestamp pattern
        if not _TIMESTAMP_RE.search(cols[0]):
            continue

        m = _TIMESTAMP_RE.search(cols[0])
        start = _to_seconds(m.group(1), m.group(2))
        end = _to_seconds(m.group(3), m.group(4))

        description = cols[1]
        keywords = _split_keywords(cols[2])
        voiceover = cols[3] if len(cols) > 3 else ""

        scenes.append(Scene(start=start, end=end, description=description,
                             keywords=keywords, voiceover=voiceover))

    return scenes


# ----------------------------------------------------------------------------
# 2. Fetching clips from Pexels
# ----------------------------------------------------------------------------
def _pexels_search(query: str, per_page: int = 5) -> list:
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page, "orientation": "landscape"}
    resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("videos", [])


def _best_video_file(video: dict) -> Optional[dict]:
    files = sorted(
        (f for f in video.get("video_files", []) if f.get("width")),
        key=lambda f: f["width"],
        reverse=True,
    )
    for f in files:
        if f["width"] <= TARGET_RESOLUTION[0]:
            return f
    return files[0] if files else None


def _download(url: str, dest: str) -> str:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
    return dest


def fetch_clips_for_scene(scene: Scene, used_ids: set, out_dir: str = "clips") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    search_terms = list(scene.keywords) + [scene.description]
    collected: List[str] = []

    for term in search_terms:
        if len(collected) >= CLIPS_PER_SCENE:
            break
        try:
            results = _pexels_search(term)
        except requests.RequestException as e:
            print(f"  [warn] Pexels search failed for '{term}': {e}")
            continue

        random.shuffle(results)
        for video in results:
            if len(collected) >= CLIPS_PER_SCENE:
                break
            vid = video.get("id")
            if vid in used_ids:
                continue
            vf = _best_video_file(video)
            if not vf:
                continue
            dest = os.path.join(out_dir, f"{vid}.mp4")
            if not os.path.exists(dest):
                try:
                    _download(vf["link"], dest)
                except requests.RequestException as e:
                    print(f"  [warn] download failed for video {vid}: {e}")
                    continue
            used_ids.add(vid)
            collected.append(dest)

    scene.clip_paths = collected
    return collected


# ----------------------------------------------------------------------------
# 3. Building each scene's clip
# ----------------------------------------------------------------------------
def build_scene_clip(clip_paths: List[str], duration: float):
    if not clip_paths:
        raise ValueError("No clips available to build scene")

    loaded = [VideoFileClip(p).without_audio() for p in clip_paths]

    # crop/resize each source clip to a consistent target resolution
    fitted = [c.resized(height=TARGET_RESOLUTION[1]) for c in loaded]

    segments = []
    remaining = duration
    i = 0
    while remaining > 0:
        clip = fitted[i % len(fitted)]
        take = min(remaining, clip.duration)
        segments.append(clip.subclipped(0, take))
        remaining -= take
        i += 1

    scene_clip = concatenate_videoclips(segments, method="compose")
    return scene_clip.with_duration(duration)


# ----------------------------------------------------------------------------
# 4. Assembling the final video
# ----------------------------------------------------------------------------
def _caption_for(scene: Scene, duration: float):
    return (
        TextClip(
            text=scene.description,
            font=CAPTION_FONT_PATH,
            font_size=42,
            color="white",
            method="caption",
            size=(int(TARGET_RESOLUTION[0] * 0.8), None),
            margin=(0, 60),
        )
        .with_position(("center", "bottom"))
        .with_duration(duration)
    )


def assemble_video(scenes: List[Scene], audio_path: Optional[str] = AUDIO_PATH,
                    output_path: str = OUTPUT_PATH):
    scene_clips = []
    for scene in scenes:
        base = build_scene_clip(scene.clip_paths, scene.duration)
        caption = _caption_for(scene, scene.duration)
        scene_clips.append(CompositeVideoClip([base, caption]))

    video = concatenate_videoclips(scene_clips, method="compose")

    if audio_path:
        audio = AudioFileClip(audio_path)
        # Scale the whole assembled video to match the voiceover's length,
        # rather than trusting the sum of scene timestamps.
        speed_factor = video.duration / audio.duration
        video = video.with_effects([vfx.MultiplySpeed(speed_factor)]).with_duration(audio.duration)
        video = video.with_audio(audio)
    # else: keep video length exactly as the timestamps dictate (no audio track)

    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")
    return output_path


# ----------------------------------------------------------------------------
# Optional Supabase logging
# ----------------------------------------------------------------------------
def log_to_supabase(scenes: List[Scene]):
    if not (SUPABASE_URL and SUPABASE_KEY):
        return
    try:
        payload = [
            {"start": s.start, "end": s.end, "description": s.description,
             "keywords": s.keywords, "clips": s.clip_paths}
            for s in scenes
        ]
        requests.post(
            f"{SUPABASE_URL}/rest/v1/scene_runs",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                      "Content-Type": "application/json"},
            json=payload, timeout=15,
        )
    except requests.RequestException as e:
        print(f"[warn] Supabase logging skipped: {e}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    if not PEXELS_API_KEY:
        raise SystemExit("Set the PEXELS_API_KEY environment variable first.")

    scenes = parse_scene_script(SCRIPT_PATH)
    print(f"Parsed {len(scenes)} scenes from {SCRIPT_PATH}")

    used_ids: set = set()
    for i, scene in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}] Fetching clips for: {scene.description!r}")
        fetch_clips_for_scene(scene, used_ids)
        if not scene.clip_paths:
            print(f"  [warn] no clips found for scene {i}, it will be skipped")

    scenes = [s for s in scenes if s.clip_paths]

    log_to_supabase(scenes)

    output = assemble_video(scenes, audio_path=AUDIO_PATH)
    print(f"Done -> {output}")


if __name__ == "__main__":
    main()
