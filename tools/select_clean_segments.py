#!/usr/bin/env python3
"""
Björn Rō — selects the best ~N "clean start, clean end" segments from a long
recording's note/pause timeline, for the 1-hour-to-N-shorts workflow.

A cut is "clean" when a segment starts exactly where a note's onset begins
and ends exactly where the previous note's audible content stopped (both
boundaries come straight from analyze_shakuhachi.py's pause detection) --
never mid-note, and never carrying an unrelated note's decay/fade across
the cut. This module is pure timeline math: it takes the `events` and
`pauses` arrays from an analyze_shakuhachi.py result and returns candidate
segment boundaries. It does not touch audio or video files.

STATUS: untested against a real long recording's actual musical structure --
only unit-level sanity checked. The quality heuristic (average confidence +
sustained-note fraction) is a first-pass guess at "best moments," not
validated against what actually sounds good.
"""
from __future__ import annotations


def _candidate_starts(pauses: list[dict]) -> list[float]:
    """Points usable as a segment's start: 0.0, or the end of a pause (i.e.
    exactly where the next note's onset begins -- starting here means zero
    bleed from the previous, unrelated note's decay/fade)."""
    points = [0.0] + [p["end"] for p in pauses]
    return sorted(set(points))


def _candidate_ends(duration: float, pauses: list[dict]) -> list[float]:
    """Points usable as a segment's end: the full duration, or the start of
    a pause (i.e. exactly where the previous note's audible content stopped
    -- ending here means no dead silence tacked on, and no bleed into the
    next, unrelated note)."""
    points = [duration] + [p["start"] for p in pauses]
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
    starts = _candidate_starts(pauses)
    ends = _candidate_ends(duration, pauses)

    candidates = []
    for start in starts:
        for end in ends:
            if end <= start:
                continue
            span = end - start
            if span < min_duration:
                continue
            if span > max_duration:
                break  # ends is sorted ascending, so further end only grows the span
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
