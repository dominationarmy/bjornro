#!/usr/bin/env python3
"""
Björn Rō — picks which clips to cut from a long recording, given its clean
start and end points (see clean_points.py). Pure timeline math: every clean
start is paired with every clean end min_duration..max_duration later, each
pair is scored (clean start, how long and complete the final fade is, how
much music it holds), and then the set of non-overlapping clips is chosen
that first maximises how many clips we get (up to target_count) and then
their total quality. That's weighted interval scheduling rather than greedy
picking, so one early high-scoring clip can't block a better full set.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np

COUNT_WEIGHT = 1000.0


def _candidates(points: dict, min_duration: float, max_duration: float) -> list[dict]:
    fs = points["frame_sec"]
    tonal_prefix = np.concatenate(([0], np.cumsum(points["tonal"])))
    ends = sorted(points["ends"], key=lambda e: e["time"])
    end_times = [e["time"] for e in ends]
    out = []
    for s in points["starts"]:
        lo = bisect_left(end_times, s["time"] + min_duration)
        hi = bisect_right(end_times, s["time"] + max_duration)
        for e in ends[lo:hi]:
            a, b = int(s["time"] / fs), int(e["time"] / fs)
            content = (tonal_prefix[b] - tonal_prefix[a]) / max(b - a, 1)
            score = 0.4 * s["score"] + 0.4 * e["score"] + 0.2 * min(content / 0.75, 1.0)
            out.append({
                "start": s["time"],
                "end": e["time"],
                "duration": round(e["time"] - s["time"], 3),
                "score": round(score, 4),
                "fade_out": e["fade_out"],
                "start_gap_sec": s["gap_sec"],
                "end_decay_sec": e["decay_sec"],
            })
    return out


def select_clean_segments(points: dict, target_count: int = 21, min_duration: float = 30.0,
                          max_duration: float = 60.0) -> list[dict]:
    cands = sorted(_candidates(points, min_duration, max_duration), key=lambda c: c["end"])
    if not cands or target_count <= 0:
        return []
    end_times = [c["end"] for c in cands]
    prev = [bisect_right(end_times, c["start"]) for c in cands]
    best = np.zeros((len(cands) + 1, target_count + 1))
    for i, c in enumerate(cands, start=1):
        best[i] = best[i - 1]
        np.maximum(best[i, 1:], best[prev[i - 1], :-1] + COUNT_WEIGHT + c["score"], out=best[i, 1:])

    chosen, i, k = [], len(cands), target_count
    while i > 0 and k > 0:
        if best[i, k] == best[i - 1, k]:
            i -= 1
        else:
            chosen.append(cands[i - 1])
            i, k = prev[i - 1], k - 1
    return sorted(chosen, key=lambda c: c["start"])
