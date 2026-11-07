#!/usr/bin/env python3
"""
Björn Rō — finds clean clip boundaries in a long shakuhachi recording.

Clean start: the attack of a phrase's first note (however short -- a 0.1s
tsu counts), preceded by a real breath pause in which nothing from the
previous note is still sounding. Clean end: after a phrase's last note has
faded into silence, before the inhale or the next phrase.

Measured in ~10ms frames from loudness and tonality (spectral flatness: a
flute tone is strongly tonal, breath and room noise are not), relative to
the recording's own room-silence floor instead of fixed levels. Calibrated
on a real outdoor recording (data/incoming/IMG_8350 2.mov): tones read
flatness ~0.001-0.01 and breath/room noise ~0.2-0.4; inhales sit ~10-15 dB
above the floor; attacks reach +35 dB within ~50ms.
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import binary_opening

N_FFT = 1024
HOP = 220
BAND_HZ = (150.0, 4000.0)
FLOOR_PERCENTILE = 5

# Every *_DB value below is relative to the floor.
TONAL_FLATNESS_MAX = 0.10
TONAL_MIN_DB = 8.0
QUIET_DB = 6.0
NEAR_QUIET_DB = 10.0
BREATH_DB = 12.0
AUDIBLE_DB = 20.0

MIN_TONAL_RUN_SEC = 0.03
BLIP_MAX_SEC = 0.15
BLIP_CONTEXT_SEC = 1.0
BLIP_CONTEXT_MIN_TONAL_SEC = 0.3
START_GAP_SEC = 0.30
PRE_ATTACK_QUIET_SEC = 0.15
MAX_ATTACK_BACKTRACK_SEC = 0.35
AUDIBLE_WITHIN_SEC = 0.30
PHRASE_WINDOW_SEC = 1.0
PHRASE_MIN_TONAL_FRAC = 0.5
START_PREROLL_SEC = 0.02

END_MIN_GAP_SEC = 0.40
END_TAIL_SEC = 0.50
END_BEFORE_NEXT_SEC = 0.10
END_BEFORE_BREATH_SEC = 0.03
DECAY_REF_WINDOW_SEC = 3.0
DECAY_REF_DROP_DB = 6.0
LONG_DECAY_SEC = 3.0
FADE_OUT_MIN_SEC = 0.03
FADE_OUT_MAX_SEC = 0.25


def frame_features(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame band energy (dB) and spectral flatness; frame i is centred at i*HOP/sr."""
    padded = np.pad(y.astype(np.float32), (N_FFT // 2, N_FFT // 2))
    frames = librosa.util.frame(padded, frame_length=N_FFT, hop_length=HOP)
    window = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    n = frames.shape[1]
    energy = np.empty(n, np.float32)
    flatness = np.empty(n, np.float32)
    for i in range(0, n, 4096):
        spectrum = np.fft.rfft(frames[:, i:i + 4096] * window[:, None], axis=0)
        power = np.abs(spectrum[band]) ** 2 + 1e-12
        mean = power.mean(axis=0)
        energy[i:i + 4096] = 10 * np.log10(mean)
        flatness[i:i + 4096] = np.exp(np.log(power).mean(axis=0)) / mean
    return energy, flatness


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _drop_isolated_blips(tonal: np.ndarray, short: int, context: int, min_tonal: int) -> np.ndarray:
    """A short tonal run counts as music only if real tone follows it (a tsu
    leading into the next note). Otherwise it's a blip -- e.g. the brief squeak
    an inhale often starts with, or a flicker in a noisy fading tail -- and
    must not be read as a new phrase."""
    prefix = np.concatenate(([0], np.cumsum(tonal)))
    kept = tonal.copy()
    for a, b in _runs(tonal):
        if b - a < short and prefix[min(len(tonal), a + context)] - prefix[a] < min_tonal:
            kept[a:b] = False
    return kept


def _attack_start(rel: np.ndarray, run_start: int, lower_bound: int, max_back: int) -> int:
    """Walk back from the first tonal frame over the attack's rising, still-noisy
    onset (a breathy attack or a short tsu often isn't tonal yet) until the
    level drops to silence."""
    i = run_start
    lo = max(lower_bound, run_start - max_back)
    while i > lo and rel[i - 1] > QUIET_DB:
        i -= 1
    return i


def find_clean_points(y: np.ndarray, sr: int) -> dict:
    energy, flatness = frame_features(y, sr)
    fs = HOP / sr
    n = len(energy)

    def frames(sec: float) -> int:
        return int(round(sec / fs))

    floor_db = float(np.percentile(energy, FLOOR_PERCENTILE))
    rel = energy - floor_db
    tonal = (flatness < TONAL_FLATNESS_MAX) & (rel > TONAL_MIN_DB)
    tonal = binary_opening(tonal, structure=np.ones(max(1, frames(MIN_TONAL_RUN_SEC)), dtype=bool))
    tonal = _drop_isolated_blips(tonal, frames(BLIP_MAX_SEC), frames(BLIP_CONTEXT_SEC),
                                 frames(BLIP_CONTEXT_MIN_TONAL_SEC))
    runs = _runs(tonal)

    attacks = []
    prev_end = 0
    for run_start, run_end in runs:
        attacks.append(_attack_start(rel, run_start, prev_end, frames(MAX_ATTACK_BACKTRACK_SEC)))
        prev_end = run_end

    starts, ends = [], []
    for k, (run_start, run_end) in enumerate(runs):
        attack = attacks[k]
        prev_end = runs[k - 1][1] if k > 0 else 0

        head = rel[attack:attack + frames(AUDIBLE_WITHIN_SEC)]
        phrase = tonal[attack:attack + frames(PHRASE_WINDOW_SEC)]
        if ((attack - prev_end) * fs >= START_GAP_SEC
                and np.all(rel[attack - frames(PRE_ATTACK_QUIET_SEC):attack] <= NEAR_QUIET_DB)
                and phrase.mean() >= PHRASE_MIN_TONAL_FRAC
                and head.max() >= AUDIBLE_DB):
            silent = 0
            while attack - silent - 1 >= prev_end and rel[attack - silent - 1] <= QUIET_DB:
                silent += 1
            rise = int(np.argmax(head >= AUDIBLE_DB))
            score = 0.5 * min(silent * fs / 0.3, 1.0) + 0.5 * (1.0 - rise * fs / AUDIBLE_WITHIN_SEC)
            starts.append({
                "time": round(max(0.0, attack * fs - START_PREROLL_SEC), 3),
                "score": round(score, 3),
                "gap_sec": round((attack - prev_end) * fs, 2),
            })

        has_next = k + 1 < len(runs)
        next_attack = attacks[k + 1] if has_next else n
        if (next_attack - run_end) * fs < END_MIN_GAP_SEC:
            continue
        quiet = np.flatnonzero(rel[run_end:next_attack] <= NEAR_QUIET_DB)
        if not quiet.size:
            continue
        q = run_end + int(quiet[0])
        limit = next_attack - frames(END_BEFORE_NEXT_SEC) if has_next else n - 1
        if q + frames(FADE_OUT_MIN_SEC) > limit:
            continue
        end = min(q + frames(END_TAIL_SEC), limit)
        loud = (rel[q:end] > BREATH_DB).astype(np.int8)
        breath = np.flatnonzero(np.convolve(loud, np.ones(3, np.int8), mode="valid") == 3)
        if breath.size:
            end = q + int(breath[0]) - frames(END_BEFORE_BREATH_SEC)
        end = max(end, q + frames(FADE_OUT_MIN_SEC))

        ref_lo = max(0, run_end - frames(DECAY_REF_WINDOW_SEC))
        ref = rel[ref_lo:run_end]
        decay_start = ref_lo + int(np.flatnonzero(ref >= ref.max() - DECAY_REF_DROP_DB)[-1])
        decay_sec = (q - decay_start) * fs
        tail_sec = (end - q) * fs
        ends.append({
            "time": round(end * fs, 3),
            "score": round(0.7 * min(decay_sec / LONG_DECAY_SEC, 1.0) + 0.3 * min(tail_sec / 0.3, 1.0), 3),
            "decay_sec": round(decay_sec, 2),
            "fade_out": round(min(max(tail_sec, FADE_OUT_MIN_SEC), FADE_OUT_MAX_SEC), 3),
        })

    return {"floor_db": floor_db, "frame_sec": fs, "tonal": tonal, "starts": starts, "ends": ends}
