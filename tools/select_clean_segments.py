#!/usr/bin/env python3
"""
Björn Rō — selects the best ~N "clean start, clean end" segments from a long
recording's note/pause timeline, for the 1-hour-to-N-shorts workflow.

A cut is "clean" when it falls at the midpoint of a detected pause (real
silence between musical phrases, from analyze_shakuhachi.py) -- never
mid-note. This module is pure timeline math: it takes the `events` and
`pauses` arrays from an analyze_shakuhachi.py result and returns candidate
segment boundaries. It does not touch audio or video files.

STATUS: untested against a real long recording's actual musical structure --
only unit-level sanity checked. The quality heuristic (average confidence +
sustained-note fraction) is a first-pass guess at "best moments," not
validated against what actually sounds good.
"""
from __future__ import annotations


def _cut_points(duration: float, pauses: list[dict]) -> list[float]:
    points = [0.0]
    for p in pauses:
        points.append((p["start"] + p["end"]) / 2.0)
    points.append(duration)
    return sorted(set(points))


def _segment_quality(events_in_range: list[dict]) -> float:
    """Average confidence, weighted toward segments with a healthy
    proportion of sustained (not just ornamental) notes."""
    if not events_in_range:
        return 0.0
    avg_conf = sum(e["confidence"] for e in events_in_range) / len(events_in_range)
    sustained_fraction = sum(1 for e in events_in_range if e["label"] == "sustained") / len(events_in_range)
    return round(avg_conf * 0.7 + sustained_fraction * 0.3, 4)


def select_clean_segments(events: list[dict], pauses: list[dict], duration: float,
                           target_count: int = 21, min_duration: float = 30.0,
                           max_duration: float = 60.0) -> list[dict]:
    """Returns up to target_count non-overlapping {start, end, duration,
    score} segments, each bounded by clean cut points, within
    [min_duration, max_duration], ranked by quality then ordered by time."""
    cuts = _cut_points(duration, pauses)

    candidates = []
    for i in range(len(cuts)):
        start = cuts[i]
        for j in range(i + 1, len(cuts)):
            end = cuts[j]
            span = end - start
            if span < min_duration:
                continue
            if span > max_duration:
                break  # cuts is sorted ascending, so further j only grows the span
            events_in_range = [e for e in events if e["start"] >= start and e["end"] <= end]
            candidates.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(span, 3),
                "score": _segment_quality(events_in_range),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    for c in candidates:
        overlaps = any(not (c["end"] <= s["start"] or c["start"] >= s["end"]) for s in selected)
        if overlaps:
            continue
        selected.append(c)
        if len(selected) >= target_count:
            break

    selected.sort(key=lambda c: c["start"])
    return selected
