#!/usr/bin/env python3
"""
Björn Rō — cuts a clip out of a source video, frame- and sample-accurately.

Always re-encodes. Stream copy (the previous approach) can only start a
video on a keyframe, so ffmpeg silently pulled every clip's start back to
the previous keyframe -- measured 0.4-3.7s early on a real recording, which
is exactly the "hear the previous note fading first" problem. Re-encoding
at high quality is visually and audibly transparent (and the final Reel is
re-encoded by the renderer and Instagram anyway): x264 CRF 18 for 8-bit
sources, x265 Main10 CRF 20 for 10-bit sources so a 10-bit/HDR grade isn't
flattened to 8-bit; ffmpeg carries the source's colour tags through. Audio
is AAC 320k with a 10ms fade-in (prevents a click) and a per-clip fade-out
chosen by clean_points.py.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import imageio_ffmpeg

FADE_IN_SEC = 0.01
MAX_START_ERROR_SEC = 0.05
MAX_LATE_ERROR_SEC = 0.15
MAX_LENGTH_ERROR_SEC = 0.05
PROBE_SEC = 4.0


def video_encoder_args(source_path: Path) -> list[str]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(source_path)], capture_output=True, text=True).stderr
    video = next((line for line in probe.splitlines() if "Video:" in line), "")
    if not re.search(r"p1[02](le|be)", video):
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libx265" in encoders:
        return ["-c:v", "libx265", "-preset", "superfast", "-crf", "20", "-pix_fmt", "yuv420p10le",
                "-tag:v", "hvc1", "-x265-params", "log-level=error"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p10le"]


def cut_clip(source_path: Path, start: float, end: float, out_path: Path,
             encoder_args: list[str], fade_out: float) -> None:
    duration = end - start
    fade_out = min(fade_out, duration / 2)
    audio_filter = (f"afade=t=in:st=0:d={FADE_IN_SEC},"
                    f"afade=t=out:st={duration - fade_out:.3f}:d={fade_out:.3f}")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", f"{start:.3f}", "-i", str(source_path),
           "-t", f"{duration:.3f}", *encoder_args, "-af", audio_filter, "-c:a", "aac", "-b:a", "320k",
           "-movflags", "+faststart", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed for {start}-{end}:\n{result.stderr[-2000:]}")


def _decode_audio(path: Path, sr: int) -> np.ndarray:
    wav = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(path), "-vn",
                        "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav)], check=True)
        y, _ = sf.read(str(wav), dtype="float32")
        return y
    finally:
        wav.unlink(missing_ok=True)


def _envelope_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-6)


def _offset(source: np.ndarray, sr: int, probe: np.ndarray, expected: float, search: float = 1.0) -> float:
    """Where the probe's loudness envelope best matches the source, relative to
    where it should be. Matches envelopes (10ms steps), not raw waveforms: a
    held note repeats every cycle, so raw matching can lock onto the wrong one."""
    hop = sr // 100
    lo = max(0, int((expected - search) * sr))
    window = _envelope_db(source[lo:int((expected + search) * sr) + len(probe)], hop)
    p = _envelope_db(probe, hop)
    p = (p - p.mean()) / (p.std() + 1e-9)
    scores = [np.dot(p, (seg - seg.mean()) / (seg.std() + 1e-9))
              for seg in (window[k:k + len(p)] for k in range(len(window) - len(p) + 1))]
    return (lo + int(np.argmax(scores)) * hop) / sr - expected


def verify_cut(source_audio: np.ndarray, sr: int, clip_path: Path, start: float, end: float) -> None:
    """Confirms the finished file really starts where the analysis said, doesn't
    drift, and has the right length, by locating its opening and closing
    seconds in the source audio."""
    clip = _decode_audio(clip_path, sr)
    duration = end - start
    head, late = FADE_IN_SEC * 1.5, duration - PROBE_SEC - 0.3
    start_err = _offset(source_audio, sr, clip[int(head * sr):int((head + PROBE_SEC) * sr)], start + head)
    late_err = _offset(source_audio, sr, clip[int(late * sr):int((late + PROBE_SEC) * sr)], start + late)
    length_err = len(clip) / sr - duration
    if (abs(start_err) > MAX_START_ERROR_SEC or abs(late_err) > MAX_LATE_ERROR_SEC
            or abs(length_err) > MAX_LENGTH_ERROR_SEC):
        raise RuntimeError(f"cut {start}-{end} landed off target: start {start_err:+.3f}s, "
                           f"near end {late_err:+.3f}s, length {length_err:+.3f}s")
