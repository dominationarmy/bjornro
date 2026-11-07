#!/usr/bin/env python3
"""
Checks tools/clean_points.py against a synthetic performance with known
answers. Run: ./.venv/bin/python3 tests/test_clean_points.py

  0.0- 1.0  silence
  1.0- 4.0  note, then a long 2s fade into nothing (4.0-6.0)
  6.0- 6.4  silence
  6.4- 7.2  inhale (breath noise), opening with a 40ms tonal squeak -- must not
            be mistaken for the next phrase
  7.2- 7.45 silence
  7.45-7.55 tsu (0.1s)  -> clean start must include it, not begin at the next note
  7.55-10.0 re, short decay to 10.3
 10.3-10.5  0.2s gap    -> too short to be a clean start or end
 10.5-13.0  note, decay to 13.5
 13.5-15.0  silence
 15.0-17.0  note, decay to 17.4, silence to 19.0
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from clean_points import find_clean_points  # noqa: E402
from select_clean_segments import select_clean_segments  # noqa: E402

SR = 22050
rng = np.random.default_rng(0)


def tone(f0, dur, amp):
    t = np.arange(int(dur * SR)) / SR
    wave = sum((0.6 ** h) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(6))
    return amp * wave / 2.0


def place(y, start, sig):
    i = int(start * SR)
    y[i:i + len(sig)] += sig


y = rng.normal(0, 0.0005, int(19.0 * SR)).astype(np.float64)

place(y, 1.0, tone(440, 3.0, 0.3))
fade = tone(440, 2.0, 0.3) * np.exp(-np.arange(int(2.0 * SR)) / SR * 4.0)
place(y, 4.0, fade)

breath = rng.normal(0, 1, int(0.8 * SR))
breath = np.convolve(breath, np.ones(8) / 8, mode="same") * 0.006
place(y, 6.4, breath)
place(y, 6.4, tone(900, 0.04, 0.03))

place(y, 7.45, tone(587, 0.10, 0.25))
place(y, 7.55, tone(523, 2.45, 0.3))
place(y, 10.0, tone(523, 0.3, 0.3) * np.linspace(1, 0, int(0.3 * SR)))
place(y, 10.5, tone(392, 2.5, 0.3))
place(y, 13.0, tone(392, 0.5, 0.3) * np.linspace(1, 0, int(0.5 * SR)))
place(y, 15.0, tone(440, 2.0, 0.3))
place(y, 17.0, tone(440, 0.4, 0.3) * np.linspace(1, 0, int(0.4 * SR)))

points = find_clean_points(y.astype(np.float32), SR)
starts = [s["time"] for s in points["starts"]]
ends = [e["time"] for e in points["ends"]]
print("starts:", starts)
print("ends:  ", ends)


def near(values, target, tol):
    return any(abs(v - target) <= tol for v in values)


assert near(starts, 7.45 - 0.02, 0.03), "clean start must land on the tsu's attack"
assert not near(starts, 7.55, 0.06), "must not start on the second note and cut off the tsu"
assert not any(6.3 <= s <= 7.3 for s in starts), "must not start on the inhale"
assert not near(starts, 10.5, 0.1), "0.2s gap is too short for a clean start"
assert near(starts, 15.0 - 0.02, 0.03), "clean start after the silent pause"
assert near(starts, 1.0 - 0.02, 0.03), "clean start after the opening silence"

assert any(5.0 <= e <= 6.4 for e in ends), "end after the long fade, before the inhale"
assert not any(6.4 < e < 7.45 for e in ends), "must not end after the inhale (inhale squeak is not a phrase)"
assert not any(10.0 <= e <= 10.5 for e in ends), "0.2s gap is too short for a clean end"
assert any(13.4 <= e <= 14.2 for e in ends), "end after the decay at 13.5"
long_fade_end = next(e for e in points["ends"] if 5.0 <= e["time"] <= 6.4)
short_end = next(e for e in points["ends"] if 13.4 <= e["time"] <= 14.2)
assert long_fade_end["score"] > short_end["score"], "a long fade into nothing should rank higher"

segments = select_clean_segments(points, target_count=5, min_duration=3.0, max_duration=12.0)
print("segments:", [(s["start"], s["end"]) for s in segments])
assert segments, "should select at least one clip"
for a, b in zip(segments, segments[1:]):
    assert a["end"] <= b["start"], "selected clips must not overlap"
for s in segments:
    assert s["start"] in starts and s["end"] in ends

print("OK")
