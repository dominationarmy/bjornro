# Björn Rō — Breath-Synced Quote Video Automation

Status: Phase 1 (proof of concept) in progress. See [docs/architecture.md](docs/architecture.md)
for the full architecture, risks, Airtable schema, and open questions blocking
real validation.

## What's here

- `tools/analyze_shakuhachi.py` — detects musical note events in a shakuhachi
  recording and outputs word-reveal timing JSON. See its docstring for the
  detection pipeline.
- `tools/quote_mapper.py` — maps a quote's words onto detected note events.
- `tools/text_layout.py` — precomputes line breaks and per-word pixel
  positions so text never jumps as words reveal.
- `tools/render_creatomate.py` — builds the Creatomate RenderScript JSON for
  an approved quote + clip; submits it once `CREATOMATE_API_KEY` is set.
- `tools/service.py` — FastAPI wrapper exposing the above over HTTP, since
  n8n Cloud has no filesystem access to call Python directly. Needs
  deploying somewhere with a public URL before any n8n workflow can use it.
- `tests/make_synthetic_shakuhachi.py` — generates a synthetic audio fixture
  used to smoke-test the analyzer before a real clip was available.
- `data/analysis/` — cached per-clip analysis JSON (gitignore-able, regenerable).
- `data/audio/` — cached extracted mono WAV per clip.
- `data/incoming/` — drop new source clips here.
- `data/verified_quotes.example.json` — quote-library format example. Not the
  real library — see architecture doc, Risk 3, before populating it further.

## Setup

```bash
# venv already has librosa/numpy/scipy/scikit-learn/soundfile installed
./.venv/bin/pip install imageio-ffmpeg   # bundles a static ffmpeg, no sudo/brew needed
./.venv/bin/pip install Pillow fastapi uvicorn python-multipart
```

## Running the HTTP service (local test only — not deployed)

```bash
./.venv/bin/uvicorn tools.service:app --reload --port 8000
```

## Running the analyzer

```bash
./.venv/bin/python3 tools/analyze_shakuhachi.py path/to/clip.mp4 --clip-id BR-001
```

Output is cached at `data/analysis/BR-001.json`; re-running without `--force`
returns the cached result instantly instead of re-analyzing.

## Next steps

See "Open questions blocking real validation" in
[docs/architecture.md](docs/architecture.md).
