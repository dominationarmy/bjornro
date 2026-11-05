#!/usr/bin/env python3
"""
Björn Rō — shakuhachi note-event analysis.

Detects musical note events in a shakuhachi recording and produces the
word-reveal timing schema consumed by the typography-render stage.

Usage:
    python3 tools/analyze_shakuhachi.py <input.mp4|input.wav> --clip-id BR-001 [--force]

Pipeline (see docs/architecture.md "Audio analysis methodology" for rationale):
  1. Extract/cache mono audio from the input media (ffmpeg via imageio-ffmpeg).
  2. Harmonic-percussive source separation (HPSS) to pull breath/attack noise
     away from the tonal signal before pitch tracking.
  3. pYIN pitch tracking on the harmonic component -> f0 contour + voiced probability.
  4. Two independent boundary detectors, merged:
       a) onset_detect on the harmonic onset-strength envelope (catches attacks)
       b) stable-pitch-region segmentation on the f0 contour (catches slurred
          meri/kari transitions that have no sharp onset)
  5. Segment classification: drop segments that are mostly unvoiced/noisy
     (breath, wind, birds) rather than pitched; label short segments as
     "ornamental", longer ones "sustained"; gaps of real silence are logged
     as pauses and excluded from word-reveal events.
  6. Confidence scoring per event, with a needs_review flag below threshold.
  7. Cache the result in data/analysis/<clip_id>.json so re-runs are free.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_CACHE_DIR = PROJECT_ROOT / "data" / "audio"
ANALYSIS_CACHE_DIR = PROJECT_ROOT / "data" / "analysis"

SR = 22050                     # analysis sample rate
MIN_NOTE_SEC = 0.09             # shorter than this -> discarded as noise/click
ORNAMENTAL_MAX_SEC = 0.40       # <= this duration -> labeled "ornamental"
MIN_PAUSE_SEC = 0.30            # silence gaps >= this are logged as real pauses
VOICED_PROB_THRESHOLD = 0.5     # pYIN frame counts as "pitched" above this
MIN_VOICED_FRACTION = 0.35      # segment needs >= this fraction of pitched frames to be a note
PITCH_STABLE_CENTS = 85         # frame-to-frame pitch jump below this stays in the same region
BOUNDARY_MERGE_SEC = 0.08       # onset- and pitch-boundary within this are treated as one
CONFIDENCE_REVIEW_THRESHOLD = 0.55

# A boundary that comes only from the pitch-contour detector (no onset-strength
# attack nearby) is often just vibrato/breath wobble within a single held note,
# not a real new note. We drop it unless the pitch actually moved by more than
# this many cents between the two sides of the boundary. Real meri/kari slurs
# move well past this; vibrato typically does not.
PITCH_WOBBLE_MERGE_CENTS = 70
LOCAL_PITCH_WINDOW_SEC = 0.15    # window used to estimate pitch just before/after a boundary

# Real outdoor recordings never reach near-digital-silence (wind, hiss, hum),
# so "silence" is judged relative to the clip's own noise floor, not a fixed
# absolute level.
SILENCE_FLOOR_PERCENTILE = 15
SILENCE_FLOOR_MULTIPLIER = 1.4

# shakuhachi fundamental range is roughly D4 (~294 Hz) down to meri'd low
# register (~130 Hz) and overtone/kari technique up past 1000 Hz.
FMIN, FMAX = 110.0, 1400.0


def extract_audio(input_path: Path, clip_id: str, force: bool) -> Path:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = AUDIO_CACHE_DIR / f"{clip_id}.wav"
    if wav_path.exists() and not force:
        return wav_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", str(SR),
        "-acodec", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr[-2000:]}")
    return wav_path


def hz_to_cents(f_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.maximum(f_hz, 1e-6) / ref_hz)
    return cents


def pitch_region_boundaries(times: np.ndarray, f0: np.ndarray, voiced_flag: np.ndarray) -> list[float]:
    """Frame index boundaries where the pitched signal starts, stops, or jumps
    by more than PITCH_STABLE_CENTS between consecutive voiced frames."""
    cents = hz_to_cents(np.nan_to_num(f0, nan=0.0))
    boundaries = []
    prev_voiced = False
    prev_cents = None
    for i in range(len(times)):
        v = bool(voiced_flag[i])
        if v != prev_voiced:
            boundaries.append(times[i])
        elif v and prev_cents is not None and abs(cents[i] - prev_cents) > PITCH_STABLE_CENTS:
            boundaries.append(times[i])
        if v:
            prev_cents = cents[i]
        prev_voiced = v
    return boundaries


def merge_boundaries(onset_times: np.ndarray, pitch_times: list[float], duration: float) -> list[float]:
    all_b = sorted(set([0.0, duration] + list(onset_times) + list(pitch_times)))
    merged = [all_b[0]]
    for t in all_b[1:]:
        if t - merged[-1] < BOUNDARY_MERGE_SEC:
            continue
        merged.append(t)
    if merged[-1] < duration - 1e-6:
        merged.append(duration)
    return merged


def local_median_pitch_cents(t: float, side: str, frame_times: np.ndarray,
                              f0: np.ndarray, voiced_mask: np.ndarray) -> float | None:
    if side == "before":
        mask = (frame_times >= t - LOCAL_PITCH_WINDOW_SEC) & (frame_times < t) & voiced_mask
    else:
        mask = (frame_times >= t) & (frame_times < t + LOCAL_PITCH_WINDOW_SEC) & voiced_mask
    vals = f0[mask]
    if len(vals) == 0:
        return None
    return float(np.median(hz_to_cents(vals)))


def prune_spurious_pitch_boundaries(boundaries: list[float], onset_times: np.ndarray,
                                     frame_times: np.ndarray, f0: np.ndarray,
                                     voiced_mask: np.ndarray) -> list[float]:
    """Drop interior boundaries that have no onset-strength attack nearby AND
    no real pitch movement across them -- these are vibrato/breath-wobble
    artifacts within a single held note, not genuine note changes."""
    if len(boundaries) <= 2:
        return boundaries
    kept = [boundaries[0]]
    for t in boundaries[1:-1]:
        onset_confirmed = bool(np.any(np.abs(onset_times - t) < BOUNDARY_MERGE_SEC))
        if onset_confirmed:
            kept.append(t)
            continue
        before = local_median_pitch_cents(t, "before", frame_times, f0, voiced_mask)
        after = local_median_pitch_cents(t, "after", frame_times, f0, voiced_mask)
        if before is None or after is None:
            kept.append(t)  # voiced/unvoiced transition -- keep, it's meaningful
            continue
        if abs(after - before) < PITCH_WOBBLE_MERGE_CENTS:
            continue  # same pitch region, no attack -> spurious, drop it
        kept.append(t)
    kept.append(boundaries[-1])
    return kept


def segment_stats(y_harm: np.ndarray, sr: int, start: float, end: float,
                   f0: np.ndarray, voiced_prob: np.ndarray, frame_times: np.ndarray) -> dict:
    i0, i1 = int(start * sr), int(end * sr)
    seg = y_harm[i0:i1]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0

    mask = (frame_times >= start) & (frame_times < end)
    seg_f0 = f0[mask]
    seg_vp = voiced_prob[mask]
    voiced = seg_vp >= VOICED_PROB_THRESHOLD
    voiced_fraction = float(np.mean(voiced)) if len(seg_vp) else 0.0

    if voiced.any():
        pitched = seg_f0[voiced]
        mean_pitch = float(np.nanmean(pitched))
        cents = hz_to_cents(pitched)
        pitch_stability_cents = float(np.nanstd(cents)) if len(cents) > 1 else 0.0
    else:
        mean_pitch = 0.0
        pitch_stability_cents = 0.0

    return {
        "rms": rms,
        "voiced_fraction": voiced_fraction,
        "mean_pitch_hz": round(mean_pitch, 1),
        "pitch_stability_cents": round(pitch_stability_cents, 1),
    }


def confidence_from_stats(stats: dict, duration: float, boundary_agreement: bool) -> float:
    voiced_score = min(stats["voiced_fraction"] / 0.8, 1.0)
    stability_score = max(0.0, 1.0 - stats["pitch_stability_cents"] / 150.0)
    duration_score = min(duration / 0.5, 1.0)
    score = 0.45 * voiced_score + 0.30 * stability_score + 0.15 * duration_score
    score += 0.10 if boundary_agreement else 0.0
    return round(float(np.clip(score, 0.0, 1.0)), 2)


def analyze(wav_path: Path, clip_id: str) -> dict:
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    duration = float(len(y) / sr)

    y_harm, y_perc = librosa.effects.hpss(y)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_harm, sr=sr, fmin=FMIN, fmax=FMAX,
        frame_length=2048, hop_length=256,
    )
    frame_times = librosa.times_like(f0, sr=sr, hop_length=256)
    voiced_flag = np.nan_to_num(voiced_prob, nan=0.0) >= VOICED_PROB_THRESHOLD

    onset_env = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=256)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=256, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)

    pitch_boundaries = pitch_region_boundaries(frame_times, f0, voiced_flag)
    boundaries = merge_boundaries(onset_times, pitch_boundaries, duration)
    boundaries = prune_spurious_pitch_boundaries(boundaries, onset_times, frame_times, f0, voiced_flag)

    # Noise floor relative to this clip's own quiet passages, not an absolute
    # digital-silence level -- outdoor recordings always carry wind/hiss.
    hop_length = 512
    rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    noise_floor = float(np.percentile(rms_frames, SILENCE_FLOOR_PERCENTILE))
    silence_rms_threshold = noise_floor * SILENCE_FLOOR_MULTIPLIER

    events, pauses = [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg_dur = end - start
        if seg_dur < 1e-3:
            continue
        stats = segment_stats(y_harm, sr, start, end, f0, voiced_prob, frame_times)

        is_silence = stats["rms"] < silence_rms_threshold and stats["voiced_fraction"] < 0.1
        if is_silence and seg_dur >= MIN_PAUSE_SEC:
            pauses.append({"start": round(start, 3), "end": round(end, 3),
                            "duration": round(seg_dur, 3)})
            continue

        if seg_dur < MIN_NOTE_SEC:
            continue
        if stats["voiced_fraction"] < MIN_VOICED_FRACTION:
            # mostly breath noise / wind / unpitched -> not a musical note
            continue

        near_onset = np.any(np.abs(onset_times - start) < BOUNDARY_MERGE_SEC)
        near_pitch = any(abs(pb - start) < BOUNDARY_MERGE_SEC for pb in pitch_boundaries)
        boundary_agreement = bool(near_onset and near_pitch)

        confidence = confidence_from_stats(stats, seg_dur, boundary_agreement)
        label = "ornamental" if seg_dur <= ORNAMENTAL_MAX_SEC else "sustained"

        events.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(seg_dur, 3),
            "confidence": confidence,
            "needs_review": confidence < CONFIDENCE_REVIEW_THRESHOLD,
            "label": label,
            "mean_pitch_hz": stats["mean_pitch_hz"],
            "voiced_fraction": round(stats["voiced_fraction"], 2),
        })

    low_confidence_count = sum(1 for e in events if e["needs_review"])

    return {
        "clip_id": clip_id,
        "source_audio": str(wav_path.relative_to(PROJECT_ROOT)),
        "duration": round(duration, 3),
        "events": events,
        "pauses": pauses,
        "recommended_word_count": len(events),
        "analysis": {
            "method": "hpss+pyin+onset_union_v1",
            "low_confidence_event_count": low_confidence_count,
            "params": {
                "min_note_sec": MIN_NOTE_SEC,
                "ornamental_max_sec": ORNAMENTAL_MAX_SEC,
                "min_pause_sec": MIN_PAUSE_SEC,
                "fmin": FMIN, "fmax": FMAX,
                "noise_floor_rms": round(noise_floor, 6),
                "silence_rms_threshold": round(silence_rms_threshold, 6),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to source video or audio file")
    parser.add_argument("--clip-id", required=True, help="Unique clip identifier, e.g. BR-001")
    parser.add_argument("--force", action="store_true", help="Re-extract audio and re-analyze even if cached")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    analysis_path = ANALYSIS_CACHE_DIR / f"{args.clip_id}.json"
    if analysis_path.exists() and not args.force:
        print(f"Cached analysis found at {analysis_path}, skipping re-analysis (use --force to redo).")
        print(analysis_path.read_text())
        return

    wav_path = extract_audio(args.input, args.clip_id, args.force)
    result = analyze(wav_path, args.clip_id)
    analysis_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {analysis_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
