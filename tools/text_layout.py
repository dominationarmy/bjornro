#!/usr/bin/env python3
"""
Björn Rō — quote line-layout and per-word position precomputation.

Computes line breaks and each word's fixed pixel position so the text block
never jumps, reshuffles, or resizes as words reveal one by one. Coordinates
are returned relative to the block's own center; the caller offsets them to
the brand's "just above the head" placement on the final 1080x1920 frame.

CAVEAT (untested): position accuracy depends on measuring with the exact
font the renderer will draw with. Pass --font-path at the real TTF for
whichever Google Font is configured in the Creatomate template (e.g. EB
Garamond, a free/open-license serif that matches the brand's "refined,
cinematic serif" requirement) -- this has not been cross-checked against an
actual Creatomate render yet, since no account/API key exists for this
project. Treat positions as a first draft pending that check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import ImageFont

DEFAULT_FONT_CANDIDATES = [
    str(Path(__file__).resolve().parent / "fonts" / "EBGaramond-Regular.ttf"),
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]


def resolve_default_font() -> str:
    for p in DEFAULT_FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "No fallback serif font found; pass --font-path pointing at the "
        "Google Font TTF configured in the Creatomate template (e.g. EB "
        "Garamond) so measurements match the real render."
    )


def layout_quote(words: list[str], font_path: str, font_size: int, canvas_width: int,
                  max_line_width_frac: float = 0.78, line_height_mult: float = 1.4) -> tuple[list[dict], int]:
    font = ImageFont.truetype(font_path, font_size)
    max_line_width = canvas_width * max_line_width_frac
    space_width = font.getlength(" ")

    lines: list[list[str]] = []
    current: list[str] = []
    current_width = 0.0
    for w in words:
        w_width = font.getlength(w)
        projected = current_width + (space_width if current else 0.0) + w_width
        if current and projected > max_line_width:
            lines.append(current)
            current, current_width = [w], w_width
        else:
            current.append(w)
            current_width = projected
    if current:
        lines.append(current)

    line_height = font_size * line_height_mult
    total_height = line_height * len(lines)
    top_y = -total_height / 2.0

    positioned = []
    for li, line_words in enumerate(lines):
        widths = [font.getlength(w) for w in line_words]
        line_width = sum(widths) + space_width * (len(line_words) - 1)
        x = -line_width / 2.0
        y = top_y + li * line_height + line_height / 2.0
        for w, ww in zip(line_words, widths):
            positioned.append({"word": w, "x": round(x + ww / 2.0, 1), "y": round(y, 1), "line": li})
            x += ww + space_width

    return positioned, len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quote", help="Full quote text")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--font-size", type=int, default=46)
    parser.add_argument("--canvas-width", type=int, default=1080)
    args = parser.parse_args()

    font_path = args.font_path or resolve_default_font()
    words = args.quote.split()
    positioned, n_lines = layout_quote(words, font_path, args.font_size, args.canvas_width)
    print(json.dumps({"lines": n_lines, "words": positioned}, indent=2))


if __name__ == "__main__":
    main()
