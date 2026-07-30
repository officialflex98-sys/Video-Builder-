"""
Runs AFTER Build_video.py. Takes final_video.mp4 and:
  1. Adds a fade-in at the start and a fade-out at the end (video + audio)
  2. Re-encodes with settings suited for YouTube upload (h264/yuv420p,
     faststart for fast web playback, high-quality CRF, AAC audio)

Requires: ffmpeg installed on PATH (already added as a workflow step),
          moviepy (already installed) - only used here to read duration.
"""

import subprocess
from moviepy import VideoFileClip

INPUT_PATH = "final_video.mp4"
OUTPUT_PATH = "final_video_youtube.mp4"
FADE_DURATION = 1.5   # seconds, for both fade-in and fade-out

# x264 quality/speed tradeoff: lower CRF = higher quality/larger file.
# 18 is visually near-lossless; 23 is the ffmpeg default. "slow" preset
# trades encode time for better compression at the same CRF.
CRF = 18
PRESET = "slow"
AUDIO_BITRATE = "192k"


def postprocess_video(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    duration = VideoFileClip(input_path).duration
    fade_out_start = max(duration - FADE_DURATION, 0)

    video_filter = (
        f"fade=t=in:st=0:d={FADE_DURATION},"
        f"fade=t=out:st={fade_out_start}:d={FADE_DURATION}"
    )
    audio_filter = (
        f"afade=t=in:st=0:d={FADE_DURATION},"
        f"afade=t=out:st={fade_out_start}:d={FADE_DURATION}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", video_filter,
        "-af", audio_filter,
        "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        output_path,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    postprocess_video()
