#!/usr/bin/env python3
"""
Björn Rō — maps a quote's words onto detected note events for word-reveal
timing. Expects `events` already filtered to the meaningful word-reveal
events from analyze_shakuhachi.py's output (pauses/noise already excluded).

- word count == event count: one word per event, in order.
- fewer words than events: evenly sample which events get a word (keeps
  pacing spread across the full phrase rather than front-loading it).
- more words than events: splits the longest-remaining event(s) in half to
  carry an extra word each, repeated until every word has a slot. This is a
  simple v1 heuristic, not a musical judgment -- a human should still review
  the result (spec note: "allow reasonable flexibility for ornamentation").
"""
from __future__ import annotations


def map_words_to_events(words: list[str], events: list[dict]) -> list[dict]:
    n_words, n_events = len(words), len(events)
    if n_events == 0:
        raise ValueError("no note events to map words to")

    if n_words <= n_events:
        idx, seen = [], set()
        for i in range(n_words):
            target = round(i * (n_events - 1) / max(n_words - 1, 1))
            while target in seen and target + 1 < n_events:
                target += 1
            seen.add(target)
            idx.append(target)
        slots = [{"start": events[i]["start"], "duration": events[i]["duration"]} for i in idx]
    else:
        slots = [{"start": e["start"], "duration": e["duration"]} for e in events]
        overflow = n_words - n_events
        for _ in range(overflow):
            slots.sort(key=lambda s: -s["duration"])
            biggest = slots.pop(0)
            half = biggest["duration"] / 2.0
            slots.append({"start": biggest["start"], "duration": half})
            slots.append({"start": biggest["start"] + half, "duration": half})
        slots.sort(key=lambda s: s["start"])

    return [{"word": w, "start": round(s["start"], 3), "duration": round(s["duration"], 3)}
            for w, s in zip(words, slots)]
