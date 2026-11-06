#!/usr/bin/env python3
"""
Björn Rō — cuts a segment out of a source video with ffmpeg.

Tries stream-copy first (fast, no quality loss). Falls back to re-encoding
only if the copy fails, which can happen when the requested start time
doesn't land cleanly on a keyframe for this container/codec.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def cut_clip(source_path: Path, start: float, end: float, out_path: Path) -> None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    duration = end - start

    copy_cmd = [
        ffmpeg_exe, "-y", "-ss", str(start), "-i", str(source_path),
        "-t", str(duration), "-c", "copy", "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    result = subprocess.run(copy_cmd, capture_output=True, text=True)
    if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
        return

    reencode_cmd = [
        ffmpeg_exe, "-y", "-ss", str(start), "-i", str(source_path),
        "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero", str(out_path),
    ]
    result = subprocess.run(reencode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed for {start}-{end}:\n{result.stderr[-2000:]}")
