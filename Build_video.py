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
  4. render_scene_to_file() builds ONE scene's clips + caption, writes it
                            to its own small silent video file on disk,
                            then immediately closes every clip and reader
                            subprocess it opened. Scenes are rendered one
                            at a time - never more than one scene's worth
                            of clips are open in memory simultaneously.
                            (Earlier versions built every scene's clip in
                            memory before writing anything, which held
                            dozens of ffmpeg reader subprocesses open at
                            once for longer scripts and could exhaust
                            memory on CI runners.)
  5. assemble_video()       joins all the per-scene files with ffmpeg's
                            concat demuxer (stream copy - no re-encode,
                            minimal memory), then lays the single
                            narration track and background music over
                            the joined result in one final encode pass

Requires: pip install moviepy requests python-dotenv
          ffmpeg on PATH (used directly for scene concatenation)
Env vars: PEXELS_API_KEY (required)
          SUPABASE_URL / SUPABASE_KEY (optional - logging only)
"""

import os
import re
import random
import shutil
import subprocess
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

# Each scene is rendered to its own silent video file here before final
# concatenation, so only one scene's clips are ever open in memory at a
# time. Removed automatically once the final video is assembled.
SCENE_RENDER_DIR = "scene_renders"

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
    idx: int = -1                # stable position in the original parsed list - used to
                                  # name this scene's render file consistently across retry
                                  # attempts, even if some scenes later get filtered out

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
# 4. Rendering each scene to its own file (and closing everything after)
# ----------------------------------------------------------------------------
def _existing_render_is_valid(path: str, expected_duration: float, tol: float = 1.5) -> bool:
    """True if path already holds a complete render of a scene at roughly
    the right duration. Used to resume after a workflow timeout/retry
    without redoing clip-fetching and rendering for scenes already
    finished in a prior attempt on the same runner."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        duration = float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return False  # unreadable/partial file from an interrupted attempt - re-render it
    return abs(duration - expected_duration) <= tol


def _make_caption(text: str, duration: float, target_resolution=TARGET_RESOLUTION):
    return (
        TextClip(
            text=text,
            font=CAPTION_FONT_PATH,
            font_size=42,
            color="white",
            method="caption",
            size=(int(target_resolution[0] * 0.8), None),
            margin=(0, 60),
        )
        .with_position(("center", "bottom"))
        .with_duration(duration)
    )


def render_scene_to_file(clip_paths: List[str], duration: float, caption_text: str,
                          out_path: str, fps: int = 30,
                          target_resolution=TARGET_RESOLUTION) -> str:
    """Builds ONE scene (its clips trimmed/looped to fill `duration`, plus
    its caption), writes it to its own silent video file at out_path, then
    closes every clip and underlying ffmpeg reader subprocess it opened -
    all in one call. Scenes are meant to be rendered one at a time via this
    function so peak memory never holds more than one scene's clips open,
    regardless of how many scenes the full script has."""
    if not clip_paths:
        raise ValueError("No clips available to build scene")

    opened = []
    try:
        loaded = [VideoFileClip(p).without_audio() for p in clip_paths]
        opened.extend(loaded)

        fitted = [c.resized(height=target_resolution[1]) for c in loaded]

        segments = []
        remaining = duration
        i = 0
        while remaining > 0:
            clip = fitted[i % len(fitted)]
            take = min(remaining, clip.duration)
            segments.append(clip.subclipped(0, take))
            remaining -= take
            i += 1

        scene_clip = concatenate_videoclips(segments, method="compose").with_duration(duration)
        opened.append(scene_clip)

        caption = _make_caption(caption_text, duration, target_resolution)
        opened.append(caption)

        composite = CompositeVideoClip([scene_clip, caption])
        opened.append(composite)

        composite.write_videofile(
            out_path, fps=fps, codec="libx264", audio=False, logger=None,
            preset="veryfast",  # this file is an intermediate - it gets re-encoded
                                 # again at final assembly, so speed matters more
                                 # than squeezing quality out of this pass
        )
    finally:
        for clip in opened:
            try:
                clip.close()
            except Exception:
                pass

    return out_path


# ----------------------------------------------------------------------------
# 5. Joining per-scene files (ffmpeg concat demuxer - stream copy)
# ----------------------------------------------------------------------------
def _ffmpeg_concat(scene_files: List[str], output_path: str) -> str:
    """Joins same-codec scene video files into one continuous silent video
    using ffmpeg's concat demuxer. This is a stream copy (no re-encode,
    minimal memory) since every scene file was rendered with identical
    codec settings - far cheaper than opening all scene files as moviepy
    clips simultaneously. Falls back to a re-encoding concat if the
    stream copy ever fails (e.g. a scene file with mismatched parameters)."""
    list_path = output_path + ".concat_list.txt"
    with open(list_path, "w") as f:
        for path in scene_files:
            escaped = os.path.abspath(path).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    base_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path]
    try:
        subprocess.run(base_cmd + ["-c", "copy", output_path],
                        check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr_tail = e.stderr[-300:] if e.stderr else "unknown error"
        print(f"  [warn] stream-copy concat failed, retrying with re-encode: {stderr_tail}")
        subprocess.run(base_cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", output_path],
                        check=True, capture_output=True, text=True)
    finally:
        os.remove(list_path)

    return output_path


# ----------------------------------------------------------------------------
# 6. Assembling the final video
# ----------------------------------------------------------------------------
def assemble_video(scenes: List[Scene], voiceover_path: Optional[str] = None,
                    music_path: Optional[str] = MUSIC_PATH,
                    output_path: str = OUTPUT_PATH,
                    scene_render_dir: str = SCENE_RENDER_DIR):
    # Deliberately NOT wiped here: on a workflow retry (same runner, same
    # workspace), any scene already rendered by a prior attempt is reused
    # instead of redone - see _existing_render_is_valid(). Stale files from
    # an unrelated earlier run aren't a real risk since CI runners are
    # ephemeral per workflow run; the directory only survives across the
    # retry-wrapper's own attempts within a single run.
    os.makedirs(scene_render_dir, exist_ok=True)

    scene_files = []
    for scene in scenes:
        scene_path = os.path.join(scene_render_dir, f"scene_{scene.idx:03d}.mp4")
        if _existing_render_is_valid(scene_path, scene.duration):
            print(f"  Scene {scene.idx + 1}/{len(scenes)} already rendered "
                  f"from a previous attempt, reusing it ({scene.duration:.1f}s)")
        else:
            print(f"  Rendering scene {scene.idx + 1}/{len(scenes)} ({scene.duration:.1f}s)...")
            render_scene_to_file(scene.clip_paths, scene.duration, scene.description, scene_path)
        scene_files.append(scene_path)

    silent_path = os.path.join(scene_render_dir, "_concatenated_silent.mp4")
    print(f"  Joining {len(scene_files)} scene(s) (stream copy)...")
    _ffmpeg_concat(scene_files, silent_path)

    video = VideoFileClip(silent_path)
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
    video.close()

    # Scene renders and the silent intermediate are no longer needed.
    shutil.rmtree(scene_render_dir, ignore_errors=True)

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
    for i, scene in enumerate(scenes):
        scene.idx = i

    voiceover_path = attach_voiceover(scenes)

    os.makedirs(SCENE_RENDER_DIR, exist_ok=True)

    used_ids: set = set()
    ready_scenes: List[Scene] = []
    for scene in scenes:
        render_path = os.path.join(SCENE_RENDER_DIR, f"scene_{scene.idx:03d}.mp4")
        if _existing_render_is_valid(render_path, scene.duration):
            # Already fully rendered by a previous (e.g. timed-out) attempt
            # on this same runner - skip re-fetching its clips entirely.
            print(f"[{scene.idx + 1}/{len(scenes)}] Already rendered from a "
                  f"previous attempt, skipping clip fetch: {scene.description!r}")
            ready_scenes.append(scene)
            continue

        print(f"[{scene.idx + 1}/{len(scenes)}] Fetching clips for: {scene.description!r} "
              f"(duration {scene.duration:.1f}s)")
        fetch_clips_for_scene(scene, used_ids)
        if not scene.clip_paths:
            print(f"  [warn] no clips found for scene {scene.idx + 1}, it will be skipped")
            continue
        ready_scenes.append(scene)

    log_to_supabase(ready_scenes, voiceover_path)

    output = assemble_video(ready_scenes, voiceover_path=voiceover_path)
    print(f"Done -> {output}")


if __name__ == "__main__":
    main()
