#!/usr/bin/env python3
"""
Björn Rō — builds a Creatomate RenderScript for one approved quote + clip,
and (once CREATOMATE_API_KEY is set) submits it for rendering.

STATUS: UNTESTED. No Creatomate account/API key exists for this project yet.
The per-word reveal design below is the documented best-guess implementation
(see docs/architecture.md, Risk 2) and must be visually validated against a
real render before being trusted:
  - one text element PER WORD, pre-positioned by text_layout.py so nothing
    moves as words appear
  - each word's own "text-reveal" animation, timed to its mapped note event
    (time = note start, duration = note duration * REVEAL_FRACTION so the
    reveal finishes slightly before the note ends, per the brand spec)
  - an author-attribution text element that fades in after the last word

Usage:
    python3 tools/render_creatomate.py \\
        --clip-id BR-001 --video-url https://.../BR-001.mp4 \\
        --quote "Nature does not hurry yet everything is accomplished" \\
        --author "Lao Tzu" [--font-path ...] [--submit]

Without --submit, writes the RenderScript JSON to
output/{clip_id}_creatomate_renderscript.json for inspection and does not
call the API. --submit requires CREATOMATE_API_KEY in the environment --
never hardcode the key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_layout import layout_quote, resolve_default_font  # noqa: E402
from quote_mapper import map_words_to_events  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_CACHE_DIR = PROJECT_ROOT / "data" / "analysis"
OUTPUT_DIR = PROJECT_ROOT / "output"

FRAME_WIDTH, FRAME_HEIGHT = 1080, 1920
TEXT_BLOCK_CENTER_Y_FRAC = 0.32   # "just above the head" -- manual preset per brand spec
FONT_SIZE = 46
TEXT_COLOR = "#F5EFE3"            # warm ivory / off-white
AUTHOR_COLOR = "#CFC6B4"          # slightly dimmer, for the gentle attribution
REVEAL_FRACTION = 0.85            # reveal finishes slightly before the note ends
REVEAL_MIN_SEC = 0.25             # floor, so very short ornamental notes still read as a soft fade
AUTHOR_DELAY_SEC = 0.6            # pause after the quote completes before attribution fades in
AUTHOR_FADE_SEC = 1.2


def build_renderscript(clip_id: str, video_url: str, quote: str, author: str,
                        font_path: str, duration: float) -> dict:
    analysis_path = ANALYSIS_CACHE_DIR / f"{clip_id}.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"No cached analysis for {clip_id} at {analysis_path}; run analyze_shakuhachi.py first.")
    analysis = json.loads(analysis_path.read_text())
    events = analysis["events"]

    words = quote.split()
    mapping = map_words_to_events(words, events)
    positioned, n_lines = layout_quote(words, font_path, FONT_SIZE, FRAME_WIDTH)

    elements = [{
        "id": "background-video",
        "type": "video",
        "track": 1,
        "time": 0,
        "source": video_url,
    }]

    block_center_y = FRAME_HEIGHT * TEXT_BLOCK_CENTER_Y_FRAC
    last_word_end = 0.0
    for pos, word_map in zip(positioned, mapping):
        reveal_duration = max(word_map["duration"] * REVEAL_FRACTION, REVEAL_MIN_SEC)
        elements.append({
            "id": f"word-{pos['word']}-{word_map['start']}",
            "type": "text",
            "track": 2,
            "time": round(word_map["start"], 3),
            "duration": round(duration - word_map["start"], 3),
            "x": round(FRAME_WIDTH / 2 + pos["x"], 1),
            "y": round(block_center_y + pos["y"], 1),
            "x_anchor": "50%",
            "y_anchor": "50%",
            "width": "auto",
            "font_family": "EB Garamond",
            "font_size": FONT_SIZE,
            "fill_color": TEXT_COLOR,
            "text": pos["word"],
            "animations": [{
                "time": 0,
                "duration": round(reveal_duration, 3),
                "easing": "quadratic-out",
                "type": "text-reveal",
                "split": "letter",
            }],
        })
        last_word_end = max(last_word_end, word_map["start"] + reveal_duration)

    if author:
        elements.append({
            "id": "author-attribution",
            "type": "text",
            "track": 3,
            "time": round(last_word_end + AUTHOR_DELAY_SEC, 3),
            "duration": round(max(duration - (last_word_end + AUTHOR_DELAY_SEC), 0.5), 3),
            "x": "50%",
            "y": round(block_center_y + (n_lines * FONT_SIZE * 1.4) / 2 + 40, 1),
            "x_anchor": "50%",
            "y_anchor": "50%",
            "font_family": "EB Garamond",
            "font_size": round(FONT_SIZE * 0.55),
            "fill_color": AUTHOR_COLOR,
            "text": f"— {author}",
            "animations": [{
                "time": 0,
                "duration": AUTHOR_FADE_SEC,
                "easing": "quadratic-out",
                "type": "fade",
            }],
        })

    return {
        "output_format": "mp4",
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "frame_rate": 30,
        "elements": elements,
    }


def submit_render(renderscript: dict) -> dict:
    api_key = os.environ.get("CREATOMATE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CREATOMATE_API_KEY is not set. Never hardcode it -- export it in "
            "the shell/n8n credential, then re-run with --submit."
        )
    resp = requests.post(
        "https://api.creatomate.com/v1/renders",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"source": renderscript},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--quote", required=True)
    parser.add_argument("--author", default="")
    parser.add_argument("--duration", type=float, required=True, help="Clip duration in seconds")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--submit", action="store_true", help="Actually call the Creatomate API (requires CREATOMATE_API_KEY)")
    args = parser.parse_args()

    font_path = args.font_path or resolve_default_font()
    renderscript = build_renderscript(
        args.clip_id, args.video_url, args.quote, args.author, font_path, args.duration,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.clip_id}_creatomate_renderscript.json"
    out_path.write_text(json.dumps(renderscript, indent=2))
    print(f"Wrote RenderScript to {out_path}", file=sys.stderr)

    if args.submit:
        result = submit_render(renderscript)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(renderscript, indent=2))


if __name__ == "__main__":
    main()
