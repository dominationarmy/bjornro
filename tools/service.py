#!/usr/bin/env python3
"""
Björn Rō — thin HTTP wrapper around the audio-analysis and render-prep tools.

Why this exists: the n8n instance for this project is n8n Cloud, which has no
filesystem or shell access (no Execute Command node -- confirmed by node
search, see docs/architecture.md Risk 6). n8n can only reach this logic over
HTTP, so this wraps analyze_shakuhachi.py and render_creatomate.py behind a
small FastAPI app that n8n calls with its HTTP Request node.

STATUS: import-checked only, not deployed or load-tested. It needs to run
somewhere n8n Cloud can reach it over the public internet -- see
docs/architecture.md, open question 5, before any workflow can call it.

Run locally for testing:
    ./.venv/bin/uvicorn tools.service:app --reload --port 8000
(run from the project root so the `tools.` package path resolves)

IMPORTANT — statelessness: Railway's container filesystem is ephemeral (no
attached volume), so anything written to data/analysis/ or data/audio/ does
NOT survive a redeploy or restart. The local file cache is kept as a
same-process convenience only. The caller (n8n) is responsible for
persisting the `/analyze` response (to Airtable) and passing analysis data
back explicitly to `/render-script` via the `events` field -- never assume
a previous `/analyze` call's cache is still present.

Endpoints:
  POST /analyze         {clip_id, video_url, force?} -> note-event analysis JSON
  POST /analyze-upload?clip_id=...&force=  (raw video bytes as the body) -> same JSON,
                         for private files n8n downloaded itself (e.g. from Google
                         Drive) rather than a publicly-fetchable URL
  POST /render-script   {clip_id, quote, author?, duration, video_url, events?} ->
                         Creatomate RenderScript JSON. Pass `events` (the `events`
                         array from a prior /analyze response) explicitly -- do not
                         rely on server-side caching across requests.
  POST /select-and-cut?recording_id=...&target_count=21&min_duration=30
                         &max_duration=60  (raw source video bytes as the
                         body) -> picks up to target_count non-overlapping
                         clips that start on a phrase's first attack after a
                         real breath pause and end after the last note has
                         faded out (clean_points.py), cuts each one
                         frame-accurately, checks it landed on target, and
                         returns them all as a single
                         zip file (one .mp4 per clip, plus manifest.json with
                         start/end/duration/score per clip). Google Drive
                         service accounts have no storage quota of their own
                         on a regular (non-Workspace) Drive, so this service
                         cannot upload there directly -- the caller (n8n) is
                         responsible for unzipping the response and uploading
                         each clip via its own, already-authenticated Google
                         Drive credential.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_shakuhachi  # noqa: E402
import render_creatomate  # noqa: E402
from text_layout import resolve_default_font  # noqa: E402
from clean_points import find_clean_points  # noqa: E402
from select_clean_segments import select_clean_segments  # noqa: E402
from cut_clips import cut_clip, video_encoder_args, verify_cut  # noqa: E402

app = FastAPI(title="Björn Rō audio/render service")


class AnalyzeRequest(BaseModel):
    clip_id: str
    video_url: str
    force: bool = False


class RenderScriptRequest(BaseModel):
    clip_id: str
    video_url: str
    quote: str
    author: str = ""
    duration: float
    events: Optional[list] = None


def _download(url: str) -> Path:
    tmp_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return tmp_path


def _run_analysis(video_path: Path, clip_id: str, force: bool) -> dict:
    try:
        wav_path = analyze_shakuhachi.extract_audio(video_path, clip_id, force)
        result = analyze_shakuhachi.analyze(wav_path, clip_id)
        analysis_path = analyze_shakuhachi.ANALYSIS_CACHE_DIR / f"{clip_id}.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        video_path.unlink(missing_ok=True)


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    analysis_path = analyze_shakuhachi.ANALYSIS_CACHE_DIR / f"{req.clip_id}.json"
    if analysis_path.exists() and not req.force:
        return json.loads(analysis_path.read_text())
    video_path = _download(req.video_url)
    return _run_analysis(video_path, req.clip_id, req.force)


@app.post("/analyze-upload")
async def analyze_upload(clip_id: str, request: Request, force: bool = False):
    analysis_path = analyze_shakuhachi.ANALYSIS_CACHE_DIR / f"{clip_id}.json"
    if analysis_path.exists() and not force:
        return json.loads(analysis_path.read_text())
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body -- expected raw video bytes")
    tmp_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
    tmp_path.write_bytes(body)
    return _run_analysis(tmp_path, clip_id, force)


@app.post("/select-and-cut")
async def select_and_cut(
    recording_id: str,
    request: Request,
    target_count: int = 21,
    min_duration: float = 30.0,
    max_duration: float = 60.0,
):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body -- expected raw video bytes")

    source_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
    source_path.write_bytes(body)

    wav_path = None
    cut_paths: list[Path] = []
    try:
        wav_path = analyze_shakuhachi.extract_audio(source_path, recording_id, force=True)
        audio, sr = sf.read(str(wav_path), dtype="float32")
        points = find_clean_points(audio, sr)
        segments = select_clean_segments(
            points, target_count=target_count, min_duration=min_duration, max_duration=max_duration,
        )
        encoder_args = video_encoder_args(source_path)

        clips, rejected = [], []
        for seg in segments:
            out_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
            cut_paths.append(out_path)
            cut_clip(source_path, seg["start"], seg["end"], out_path, encoder_args, seg["fade_out"])
            try:
                verify_cut(audio, sr, out_path, seg["start"], seg["end"])
            except RuntimeError as exc:
                rejected.append(str(exc))
                continue
            filename = f"{recording_id}-{len(clips) + 1:02d}.mp4"
            clips.append({"filename": filename, "path": out_path, **seg})
        if segments and not clips:
            raise RuntimeError(f"All {len(segments)} cuts failed verification, e.g. {rejected[0]}")

        summary = {
            "recording_id": recording_id,
            "source_duration": round(len(audio) / sr, 3),
            "clean_starts_found": len(points["starts"]),
            "clean_ends_found": len(points["ends"]),
            "segments_found": len(segments),
            "clips_delivered": len(clips),
            "rejected_cuts": rejected,
        }
        zip_path = Path(tempfile.mkstemp(suffix=".zip")[1])
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for c in clips:
                zf.write(c["path"], arcname=c["filename"])
            manifest = {**summary, "clips": [{k: v for k, v in c.items() if k != "path"} for c in clips]}
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        def _cleanup():
            for p in cut_paths:
                p.unlink(missing_ok=True)
            zip_path.unlink(missing_ok=True)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{recording_id}-clips.zip",
            headers={"X-Clip-Summary": json.dumps(summary)},
            background=BackgroundTask(_cleanup),
        )
    except HTTPException:
        for p in cut_paths:
            p.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for p in cut_paths:
            p.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        source_path.unlink(missing_ok=True)
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


@app.post("/render-script")
def render_script(req: RenderScriptRequest):
    try:
        font_path = resolve_default_font()
        return render_creatomate.build_renderscript(
            req.clip_id, req.video_url, req.quote, req.author, font_path, req.duration,
            events=req.events,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
