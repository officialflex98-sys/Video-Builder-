"""
Runs AFTER Build_video.py. Takes final_video.mp4 and:
  1. Adds a fade-in at the start and a fade-out at the end (video + audio)
  2. Re-encodes to YouTube's recommended 1080p upload spec:
     H.264 MP4, 30fps, 16:9, 10-12 Mbps video bitrate, AAC 320kbps audio

Requires: ffmpeg installed on PATH (already added as a workflow step),
          moviepy (already installed) - only used here to read duration.
"""

import subprocess
from moviepy import VideoFileClip

INPUT_PATH = "final_video.mp4"
OUTPUT_PATH = "final_video_youtube.mp4"
FADE_DURATION = 1.5   # seconds, for both fade-in and fade-out

# YouTube 1080p upload spec: H.264, 30fps, 16:9, 10-12 Mbps video, AAC 320kbps.
WIDTH, HEIGHT = 1920, 1080   # 16:9
FPS = 30
VIDEO_BITRATE = "11M"      # middle of the 10-12 Mbps target range
MAX_BITRATE = "12M"
BUFSIZE = "24M"            # ~2x maxrate, standard rate-control buffer sizing
PRESET = "slow"            # better compression at a given bitrate
AUDIO_BITRATE = "320k"


def postprocess_video(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    duration = VideoFileClip(input_path).duration
    fade_out_start = max(duration - FADE_DURATION, 0)

    video_filter = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
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
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", PRESET,
        "-b:v", VIDEO_BITRATE, "-maxrate", MAX_BITRATE, "-bufsize", BUFSIZE,
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
