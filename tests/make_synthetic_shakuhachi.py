#!/usr/bin/env python3
"""
Generates a synthetic test WAV that loosely mimics shakuhachi phrasing:
sustained tones at varying pitches/durations, a slurred pitch glide between
two notes, breathy unpitched noise bursts, and silent pauses.

This is a PIPELINE SMOKE TEST ONLY — it does not validate detection accuracy
against real shakuhachi playing. Real validation requires an actual recording.
"""
import numpy as np
import soundfile as sf
from pathlib import Path

SR = 22050

def tone(freq, dur, sr=SR, vibrato_hz=5.0, vibrato_cents=15, amp=0.5, attack=0.03, release=0.05):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    vibrato = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t) / 1200.0
    inst_freq = freq * (2.0 ** vibrato)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    sig = amp * np.sin(phase)
    env = np.ones_like(sig)
    a_n, r_n = int(attack * sr), int(release * sr)
    env[:a_n] = np.linspace(0, 1, a_n)
    env[-r_n:] = np.linspace(1, 0, r_n)
    breathiness = 0.03 * np.random.randn(len(sig))
    return sig * env + breathiness * env

def glide(f_start, f_end, dur, sr=SR, amp=0.5):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    inst_freq = np.linspace(f_start, f_end, len(t))
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    env = np.ones_like(t)
    a_n = int(0.02 * sr)
    env[:a_n] = np.linspace(0, 1, a_n)
    return amp * np.sin(phase) * env

def breath_noise(dur, sr=SR, amp=0.06):
    n = int(sr * dur)
    noise = np.random.randn(n)
    b = np.ones(5) / 5
    noise = np.convolve(noise, b, mode="same")
    env = np.hanning(n)
    return amp * noise * env

def silence(dur, sr=SR):
    return np.zeros(int(sr * dur))

def main():
    out_path = Path(__file__).resolve().parent / "fixtures" / "synth_shakuhachi_raw.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append(breath_noise(0.5))                  # breath attack noise (should be discarded)
    parts.append(tone(293.66, 2.6))                   # D4, sustained note 1
    parts.append(silence(0.15))
    parts.append(tone(392.00, 1.2))                   # G4, shorter note 2
    parts.append(glide(392.00, 440.00, 0.8))           # slurred meri/kari-style transition -> note 3
    parts.append(tone(440.00, 3.5))                    # A4, long sustained note 3 continuation
    parts.append(silence(0.6))                         # genuine pause between phrases
    parts.append(breath_noise(0.3))
    parts.append(tone(261.63, 0.3))                    # C4, ornamental grace note
    parts.append(tone(293.66, 2.0))                    # D4, note 4
    parts.append(silence(0.4))
    parts.append(tone(220.00, 4.0))                    # A3, final long note
    parts.append(silence(0.5))

    audio = np.concatenate(parts).astype(np.float32)
    audio = audio / max(1.0, np.max(np.abs(audio)))
    sf.write(str(out_path), audio, SR)
    print(f"Wrote synthetic test audio: {out_path} ({len(audio)/SR:.2f}s)")

if __name__ == "__main__":
    main()
