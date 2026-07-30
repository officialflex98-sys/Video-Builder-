"""
Scene-script -> video assembler.

Pipeline:
  1. parse_scene_script()   reads script.txt into Scene objects. Handles:
                              (A) one-field-per-line format - Timestamp,
                                  Scene Description, Keywords, and VO Script
                                  each on their own line, repeating in
                                  groups of 4 (with optional stray header
                                  lines like "Timestamp" mixed in)
                              (B) tab or pipe-delimited rows, including full
                                  Markdown tables (leading/trailing "|" on
                                  each line, "|---|---|---|---|" separator)
  2. attach_voiceover()     loads the single voiceover.wav produced by
                            generate_voiceover.py (if present) and scales
                            every scene's duration proportionally so the
                            total matches the real narration length,
                            instead of using the raw script.txt timestamps
                            as-is
  3. fetch_clips_for_scene() pulls CLIPS_PER_SCENE unique clips per scene
                            from Pexels
  4. build_scene_clip()     trims/loops those clips to fill the scene's
                            (now audio-accurate) duration
  5. assemble_video()       concatenates all scenes in order, then lays the
                            single narration track and background music
                            over the full result

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
    CompositeAudioClip,
    afx,
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

# Single combined narration file produced by generate_voiceover.py. If
# present, its real duration drives scene timing proportionally; if
# missing, scenes fall back to their raw script.txt timestamp durations.
VOICEOVER_PATH = "voiceover.wav"

# Point this at a background music file (mp3/wav) to mix in under the
# voiceover. Loops automatically if shorter than the final video, and trims
# if longer. Leave as None for no music. If a file named music.mp3 sits next
# to this script, it's picked up automatically - no manual edit needed.
_DEFAULT_MUSIC = "music.mp3"
MUSIC_PATH: Optional[str] = _DEFAULT_MUSIC if os.path.exists(_DEFAULT_MUSIC) else None
MUSIC_VOLUME = 0.15    # 0.0-1.0, kept low so it sits under the voiceover
VOICEOVER_VOLUME = 1.0

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class Scene:
    start: float                # seconds (from script.txt - planning guidance only)
    end: float                  # seconds (from script.txt - planning guidance only)
    description: str
    keywords: List[str]
    voiceover: str = ""
    clip_paths: List[str] = field(default_factory=list)
    _duration_override: Optional[float] = None   # set once real VO audio is measured/scaled

    @property
    def duration(self) -> float:
        if self._duration_override is not None:
            return self._duration_override
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


def _detect_delimiter(line: str) -> Optional[str]:
    if "\t" in line:
        return "\t"
    if "|" in line:
        return "|"
    return None


def _is_markdown_separator(line: str) -> bool:
    """Detects a markdown table divider row like |---|---|---|---|"""
    stripped = line.strip().strip("|")
    return bool(stripped) and all(c in "-: \t" for c in stripped)


def _is_timestamp_only_line(line: str) -> bool:
    """True if the ENTIRE line (after stripping) is just a timestamp range,
    e.g. "0:00-0:30" with nothing else on it - the signature of the
    one-field-per-line script format (Timestamp, then Scene Description,
    then Keywords, then VO script, each on their own line, repeating)."""
    return bool(re.fullmatch(r"\d+:\d+\s*[–-]\s*\d+:\d+", line.strip()))


def parse_scene_script(path: str = SCRIPT_PATH) -> List[Scene]:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f if l.strip()]

    if not raw_lines:
        return []

    scenes: List[Scene] = []

    # --- Format A: one field per line (Timestamp / Description / Keywords /
    # VO script each on their own line, repeating in groups of 4) ---
    if any(_is_timestamp_only_line(l) for l in raw_lines):
        i = 0
        n = len(raw_lines)
        while i < n:
            if _is_timestamp_only_line(raw_lines[i]):
                m = _TIMESTAMP_RE.search(raw_lines[i])
                start = _to_seconds(m.group(1), m.group(2))
                end = _to_seconds(m.group(3), m.group(4))
                description = raw_lines[i + 1] if i + 1 < n else ""
                keywords_raw = raw_lines[i + 2] if i + 2 < n else ""
                voiceover = raw_lines[i + 3] if i + 3 < n else ""
                scenes.append(Scene(
                    start=start, end=end, description=description,
                    keywords=_split_keywords(keywords_raw), voiceover=voiceover,
                ))
                i += 4
            else:
                i += 1  # stray header line (e.g. "Timestamp"), skip it
        return scenes

    # --- Format B: delimited rows (tab, pipe/Markdown table, or 2+ spaces) ---
    delimiter = _detect_delimiter(raw_lines[0])

    for line in raw_lines:
        if _is_markdown_separator(line):
            continue

        if delimiter == "|":
            trimmed = line.strip()
            if trimmed.startswith("|"):
                trimmed = trimmed[1:]
            if trimmed.endswith("|"):
                trimmed = trimmed[:-1]
            cols = [c.strip() for c in trimmed.split("|")]
        elif delimiter:
            cols = [c.strip() for c in line.split(delimiter)]
        else:
            cols = [c.strip() for c in re.split(r"\s{2,}", line)]

        if len(cols) < 3:
            continue

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
# 2. Attaching the single combined voiceover (drives real duration)
# ----------------------------------------------------------------------------
def attach_voiceover(scenes: List[Scene], voiceover_path: str = VOICEOVER_PATH) -> Optional[str]:
    """Loads the single voiceover.wav produced by generate_voiceover.py and
    scales every scene's duration proportionally so the sum of scene
    durations matches the real narration length. Each scene's *share* of
    the total is preserved from script.txt - only the overall pacing is
    stretched/compressed to fit the actual spoken audio.

    Returns the voiceover path if it was applied, else None (scenes keep
    their raw script.txt timestamp durations)."""
    if not os.path.exists(voiceover_path):
        print(f"  [warn] no voiceover file found at {voiceover_path}, "
              f"using raw script timestamp durations")
        return None

    clip = AudioFileClip(voiceover_path)
    total_audio_duration = clip.duration
    clip.close()

    total_script_duration = sum(s.duration for s in scenes)
    if total_script_duration <= 0:
        print("  [warn] scenes have zero total scripted duration, "
              "skipping voiceover-based scaling")
        return None

    scale = total_audio_duration / total_script_duration
    for s in scenes:
        s._duration_override = s.duration * scale

    print(f"  Scaled {len(scenes)} scene(s) to match {total_audio_duration:.1f}s "
          f"of narration (scale factor {scale:.3f})")
    return voiceover_path


# ----------------------------------------------------------------------------
# 3. Fetching clips from Pexels
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
# 4. Building each scene's clip
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
# 5. Assembling the final video
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


def assemble_video(scenes: List[Scene], voiceover_path: Optional[str] = None,
                    music_path: Optional[str] = MUSIC_PATH,
                    output_path: str = OUTPUT_PATH):
    scene_clips = []
    for scene in scenes:
        base = build_scene_clip(scene.clip_paths, scene.duration)
        caption = _caption_for(scene, scene.duration)
        scene_clips.append(CompositeVideoClip([base, caption]))

    # Each scene is already the correct length (driven by the proportionally
    # scaled voiceover duration where available), so concatenation needs no
    # further global stretching.
    video = concatenate_videoclips(scene_clips, method="compose")

    audio_tracks = []

    if voiceover_path and os.path.exists(voiceover_path):
        narration = AudioFileClip(voiceover_path).with_effects(
            [afx.MultiplyVolume(VOICEOVER_VOLUME)]
        )
        # Trim in case rounding leaves it a hair longer than the video;
        # never stretch it - the whole point is it stays a real performance.
        if narration.duration > video.duration:
            narration = narration.subclipped(0, video.duration)
        audio_tracks.append(narration)
    else:
        print("  [warn] no voiceover attached to final video")

    if music_path:
        music = AudioFileClip(music_path).with_effects(
            [afx.AudioLoop(duration=video.duration), afx.MultiplyVolume(MUSIC_VOLUME)]
        )
        audio_tracks.append(music)

    if audio_tracks:
        final_audio = audio_tracks[0] if len(audio_tracks) == 1 else CompositeAudioClip(audio_tracks)
        video = video.with_audio(final_audio)

    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")
    return output_path


# ----------------------------------------------------------------------------
# Optional Supabase logging
# ----------------------------------------------------------------------------
def log_to_supabase(scenes: List[Scene], voiceover_path: Optional[str] = None):
    if not (SUPABASE_URL and SUPABASE_KEY):
        return
    try:
        payload = [
            {"start": s.start, "end": s.end, "duration": s.duration,
             "description": s.description, "keywords": s.keywords,
             "clips": s.clip_paths, "voiceover": voiceover_path}
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

    voiceover_path = attach_voiceover(scenes)

    used_ids: set = set()
    for i, scene in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}] Fetching clips for: {scene.description!r} "
              f"(duration {scene.duration:.1f}s)")
        fetch_clips_for_scene(scene, used_ids)
        if not scene.clip_paths:
            print(f"  [warn] no clips found for scene {i}, it will be skipped")

    scenes = [s for s in scenes if s.clip_paths]

    log_to_supabase(scenes, voiceover_path)

    output = assemble_video(scenes, voiceover_path=voiceover_path)
    print(f"Done -> {output}")


if __name__ == "__main__":
    main()
