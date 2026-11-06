# Björn Rō — Architecture

Status: **Phase 1 proof-of-concept, mostly wired up.** The note-detection
engine has been run against a real shakuhachi recording (see Risk 1) and
tuned once against real findings, not just synthetic tones. The Python
service is deployed and confirmed reachable on Railway. The Airtable base
and all three tables exist with full field schema. The n8n render workflow
exists with the real service URL wired in. **Still unproven:** an actual
Creatomate render (no account/key yet) and a real `/analyze` call against a
publicly-reachable video (the only test so far used a local file path,
which only works when run locally, not from the deployed service).

## 1. Component boundaries

```
Google Drive (clip intake)
        |
        v
[n8n] Ingest workflow  ──creates──>  Airtable: Clips (status: Pending Analysis)
        |
        v
[Python] analyze_shakuhachi.py  ──writes──>  data/analysis/{clip_id}.json (cache)
        |                                     Airtable: Clips.analysis ref + word count
        v
[Python/OpenAI] quote_matcher.py  ──creates──>  Airtable: Quote Proposals (Pending Review)
        |
        v
   [HUMAN GATE] approve one quote in Airtable  ──flips──>  Clips.status = Ready to Render
        |
        v
[n8n] Render workflow  ──calls──>  text_layout.py (line breaks, word positions)
        |                          render_creatomate.py  OR  render_custom.py (fallback)
        v
   Rendered MP4  ──uploads──>  storage, registers──>  Airtable: Rendered Reels
        |
        v
   [HUMAN GATE] approve Reel + select 7/week  ──flips──>  Rendered Reels.selected_for_publishing
        |
        v
[n8n] Publishing workflow (Blotato)  — BUILT LATER, DISABLED BY DEFAULT
```

Each arrow is a replaceable module. The n8n workflows orchestrate; Python does
the audio/text-layout math; Airtable is the single source of truth for status
and human decisions; the renderer is the only part allowed to be swapped
(Creatomate vs. custom) without touching anything upstream.

### Why Python runs as an HTTP service, not Execute Command (corrected)

An earlier draft of this document assumed n8n could shell out to
`tools/analyze_shakuhachi.py` directly via an Execute Command node. That's
wrong: this project's n8n instance is **n8n Cloud**, confirmed by node search
turning up no Execute Command node at all (only SSH, which requires its own
separately-reachable server, and the sandboxed Code node, which has no
network access and can't import librosa/numpy anyway). n8n Cloud cannot
touch this Mac's filesystem.

So the Python side has to run as an HTTP service n8n can call with a normal
HTTP Request node. `tools/service.py` does exactly that — a small FastAPI
app wrapping the same functions the CLI scripts use:
- `POST /analyze` — `{clip_id, video_url}` → note-event analysis JSON
- `POST /render-script` — `{clip_id, video_url, quote, author, duration}` → Creatomate RenderScript JSON

**Deployed and confirmed working**: https://bjornro-production.up.railway.app
(Railway, repo https://github.com/dominationarmy/bjornro, `main` branch
auto-deploys). `/docs` returns 200, confirming the full dependency chain
(librosa/numpy/scipy/Pillow/etc.) imports cleanly in that container — not a
given, given how heavy this stack is. The first deploy attempt actually
failed: Railway's default Python (3.13) has no prebuilt `numpy==2.0.2` wheel
and failed trying to compile it from source; fixed with a `.python-version`
file pinning 3.11. A second real bug then surfaced on the first live
request: `text_layout.py`'s font fallback only checked macOS system font
paths (Georgia/Times), which don't exist on Linux — every `/render-script`
call 404'd. Fixed by bundling `tools/fonts/EBGaramond-Regular.ttf` (SIL Open
Font License, matches the brand's "refined, cinematic serif" requirement,
and is also available to Creatomate as a Google Font by name, so the same
typeface is used for both the Python layout measurements and the actual
Creatomate render). Re-tested after both fixes: the service now correctly
reaches the "no cached analysis for this clip_id" error for an unknown
clip — the right failure, proving the whole chain up to that point works.
**Not yet tested**: a real `/analyze` call (needs a publicly-reachable video
URL) or a real `/render-script` success (needs cached analysis data, which
only exists for clips that have gone through `/analyze` — the server has no
access to this Mac's local `data/` cache, by design, since `data/` is
gitignored and Railway only has what's in the repo).

The n8n workflow's "Build RenderScript" node now points at the real URL
above instead of a placeholder.

## 2. Technical risks (ranked by how much they could force an architecture change)

### Risk 1 — Note detection accuracy on real shakuhachi (highest risk)
`tools/analyze_shakuhachi.py` combines:
- HPSS (harmonic/percussive separation) to pull breath and attack noise away
  from the tonal signal before pitch-tracking it,
- pYIN pitch tracking with voiced-probability confidence,
- onset detection on the *harmonic* component (reduces false onsets from
  breath/wind), and
- a second, independent boundary detector that watches the pitch contour
  itself for stable-region changes — this is what catches slurred meri/kari
  transitions that produce no sharp onset at all.

This passed a synthetic test (sine tones + synthetic breath noise + silence)
cleanly, then was run against a real clip you provided
(`data/incoming/IMG_8350 2.mov`, 3:50, real shakuhachi audio). That run
immediately surfaced two real bugs synthetic audio couldn't have caught:

1. **Zero pauses detected.** The silence check used an absolute RMS
   threshold (`< 1e-4`), calibrated for digital silence. Real outdoor
   recordings never get that quiet (wind, hiss, ambient noise), so true
   musical pauses were being silently dropped instead of logged as pauses.
   **Fixed:** the threshold is now adaptive — the 15th percentile of the
   clip's own RMS, times a multiplier — so "silence" is judged relative to
   this recording's noise floor. Re-run: 0 → 26 pauses, with plausible
   1–3.5s gaps between phrases.
2. **Over-segmentation from vibrato/pitch wobble.** 150 of 266 events
   (56%) were sub-0.4s fragments, often at nearly identical pitch to their
   neighbor (e.g. two consecutive "notes" at 353Hz and 352Hz) — natural
   pitch wobble within one held breath was being read as a new note.
   **Fixed:** a boundary is now dropped unless it's either confirmed by an
   onset-strength attack, or the pitch actually moved by more than ~70
   cents across it. Re-run: 266 → 257 events — a smaller improvement than
   hoped.

**Still unresolved, and this is the honest state of it:** even after both
fixes, 138/257 events (54%) are still sub-0.4s "ornamental" fragments. Some
of this is very plausibly real — shakuhachi ornamentation (fast grace-note
articulations) is genuinely that short. Some of it may still be
over-segmentation I can't distinguish from real playing without a human ear
on the actual audio at specific timestamps. I did not keep tuning thresholds
blindly against a single clip with no ground truth to chase a rounder
number — that's a good way to overfit to one recording. **This needs you to
listen to a few flagged timestamps and confirm whether the density is real
musical ornamentation or still noise**, before the word-reveal count from
this engine is trusted for an actual render.

Also worth noting: the real test file was landscape 1920x1080, 3:50 long —
not a finished vertical 9:16 Reel, so while it was a legitimate way to
validate the *audio* pipeline, it isn't representative of the eventual
30–60s vertical inputs and can't be pushed through the layout/render stage
as-is.

Mitigation already built in regardless: every event carries a `confidence`
score and a `needs_review` flag, so low-confidence boundaries can be
manually corrected rather than silently trusted.

### Risk 2 — Creatomate: variable per-word reveal duration
Creatomate's built-in text presets (`text-reveal`, `text-typewriter`, etc.)
distribute a reveal evenly or by preset easing — they are not documented to
accept an *externally supplied, different duration per word*. Based on
Creatomate's RenderScript model (confirmed from docs: every element gets
independent `time`/`duration`, and keyframes/animations accept arbitrary
`time`/`duration`/`easing`), the implemented approach is: emit one small text
element per word, pre-positioned by `text_layout.py` (so nothing jumps or
reshuffles as words appear), each with its own `text-reveal` animation timed
to its mapped note's start/duration (`tools/render_creatomate.py`). This has
been smoke-tested as *code* (it builds valid-shaped JSON and round-trips over
HTTP through `tools/service.py`) but **never against the real Creatomate
API** — no account/key exists yet. You've said you'll add the key once you
have one; until then this is an unvalidated best guess at the right JSON
shape, not a working integration. A dedicated `@creatomate/n8n-nodes-creatomate`
community node exists in n8n's registry (resource `render`, operation
`create`/`get`) but is **not installed on this n8n Cloud instance** — it'll
need installing before any workflow using it can run. If the per-word
approach doesn't hold up once testable (API limits on element count,
rendering cost, visual artifacts at word boundaries), the documented
fallback is a custom Python renderer (Pillow + ffmpeg) with full manual
control.

### Risk 3 — Quote licensing (resolved for v1, confirmed by checking the actual policy)
Checked the Love Serve Remember Foundation's own copyright policy
(ramdass.org/copyright — site blocks direct fetch, confirmed via two
independent search summaries of its text): it explicitly states protected
content is for **personal use only**, and "no other use... without the
express written permission of LSRF and the copyright owner," with a named
contact to request permission for any other use of Ram Dass's audio, written,
or video content. Neem Karoli Baba himself left no written works — every
"quote" attributed to him comes from a disciple's compiled memoir, which
carries its own compilation copyright; there is no primary-source public
domain material to fall back on there.

Your instruction was "only use ram dass and maharaji quotes that are legal
and public to use... keep it ethical always." Given LSRF's policy as found,
that means: **no Ram Dass or Neem Karoli Baba quote ships in the library
without written permission from LSRF**, obtained by you (I can't request or
grant that on your behalf). `data/verified_quotes.example.json` ships with
only verifiably public-domain spiritual literature (public-domain
translations of the Tao Te Ching, Dhammapada — translator deceased 70+
years / pre-1923) and clearly-labeled original reflections. If you get
written permission from LSRF, tell me and I'll add their material with the
exact source/permission reference recorded per quote.

### Risk 4 — No system ffmpeg/Homebrew on this machine
Resolved for the Python side: `imageio-ffmpeg` (pip-installed into the
existing venv) bundles a static ffmpeg binary, no sudo/brew required. n8n's
own Execute Command node will need to call the same binary path — confirmed
working at
`.venv/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1`.

### Risk 5 — Duplicate processing
Needs a stable idempotency key. Recommend the Google Drive file ID (not
filename — filenames get renamed/re-uploaded) as the dedup key checked
against Airtable before creating a new Clip record.

### Risk 6 — n8n Cloud has no filesystem/shell access (resolved)
This n8n instance is n8n Cloud. Node search for "execute command" /
"executeCommand" found nothing — only SSH (needs its own reachable server)
and the sandboxed Code node (no network access, can't import librosa/numpy).
So n8n cannot shell out to the Python scripts directly, which is what an
earlier draft of this document assumed. Resolved by wrapping the Python
logic as an HTTP service (`tools/service.py`, FastAPI) deployed to Railway
at https://bjornro-production.up.railway.app, which n8n's HTTP Request node
calls directly. Live and responding correctly (see Section 1 above for the
two real deploy bugs found and fixed along the way: a Python-version
mismatch, and a macOS-only font fallback).

## 3. Airtable schema

Three tables, linked. Status fields are single-select so n8n can branch on
exact string match.

### Table: Clips
| Field | Type | Notes |
|---|---|---|
| Clip ID | Text (primary) | `BR-{week}-{seq}`, e.g. `BR-2026W45-03` |
| Drive File ID | Text | Idempotency key — checked before creating a new record |
| Original Video URL | URL | |
| Duration (s) | Number | |
| Processing Status | Single select | `Pending Analysis` → `Analyzed` → `Quotes Proposed` → `Ready to Render` → `Rendering` → `Rendered` → `Needs Review` → `Error` |
| Analysis JSON Path | Text | `data/analysis/{clip_id}.json` reference |
| Recommended Word Count | Number | from analysis |
| Analysis Confidence (avg) | Number | mean of event confidences |
| Low-Confidence Event Count | Number | drives manual-review prompt |
| Created At | Date | |
| Notes | Long text | |

### Table: Quote Proposals
| Field | Type | Notes |
|---|---|---|
| Proposal ID | Text (primary) | |
| Clip | Link → Clips | |
| Quote Text | Long text | |
| Author | Text | empty/"Björn Rō" if original reflection |
| Source | Text | book/publication + page, or "Original reflection" |
| Quote Type | Single select | `Verified` / `Original Reflection` |
| Word Count | Number | |
| Matching Score | Number | fit vs. recommended word count |
| Approval Status | Single select | `Pending Review` → `Approved` / `Rejected` / `Request Alternatives` |
| Reviewer Notes | Long text | |
| Previously Rejected? | Checkbox | prevents re-proposing the same text |

### Table: Rendered Reels
| Field | Type | Notes |
|---|---|---|
| Reel ID | Text (primary) | |
| Clip | Link → Clips | |
| Approved Quote | Link → Quote Proposals | |
| Preview URL | URL | |
| Rendering Status | Single select | `Queued` → `Rendering` → `Rendered` / `Render Failed` |
| Final Approval Status | Single select | `Pending Review` → `Approved` / `Rejected` / `Needs Different Quote` / `Needs Re-render` |
| Selected For Publishing | Checkbox | exactly 7/week, enforced in the review workflow, not the schema |
| Scheduled Date | Date | |
| Publishing Status | Single select | `Not Scheduled` → `Scheduled` → `Published` / `Publish Failed` — unused until Phase 5 |

### Status transition rules (enforced in n8n, not just Airtable)
- A Clip can only reach `Ready to Render` when it has exactly one Quote
  Proposal with `Approval Status = Approved`.
- `Request Alternatives` on a proposal triggers quote regeneration that
  excludes every proposal already marked `Previously Rejected? = true` for
  that clip.
- Approving a Reel's `Final Approval Status` never touches
  `Selected For Publishing` — those are independent gates, per the spec.
- Publishing workflow (Phase 5) only ever reads rows where
  `Selected For Publishing = true` AND `Publishing Status = Not Scheduled`,
  and is not scheduled/triggered automatically until explicitly authorized.

## 4. What exists right now

- `tools/analyze_shakuhachi.py` — note-detection engine. Validated against
  one real clip, not just synthetic tones (see Risk 1). Two real bugs found
  and fixed; ornamental-fragment density still needs your ear to confirm.
- `tools/quote_mapper.py` — maps a quote's words onto detected note events
  (handles word-count ≠ event-count via even sampling / event-splitting).
  Smoke-tested only.
- `tools/text_layout.py` — precomputes line breaks and fixed per-word pixel
  positions so text never jumps as words reveal. Smoke-tested only; position
  accuracy depends on matching Creatomate's actual font rendering, which is
  unverified (see Risk 2).
- `tools/render_creatomate.py` — builds the Creatomate RenderScript JSON
  (one text element per word, keyframed reveal timed to its note) and can
  submit it once `CREATOMATE_API_KEY` exists. Never hardcode that key.
  Smoke-tested as JSON structure only — never called the real API.
- `tools/service.py` — FastAPI wrapper exposing `/analyze` and
  `/render-script` over HTTP, because n8n Cloud can't shell out to Python
  directly (Risk 6). Import-checked and round-tripped once over local HTTP.
  **Not deployed anywhere n8n Cloud can reach it** — see open question 5.
- `tests/make_synthetic_shakuhachi.py` — synthetic smoke-test fixture
  generator, used before the real clip was available.
- `data/analysis/BR-TEST-001.json` — real analysis output from your test clip.
- `data/verified_quotes.example.json` — quote-library format, populated only
  with verifiably public-domain entries + original reflections (see Risk 3).
- ffmpeg available in-venv via `imageio-ffmpeg` (no system install needed).
- No n8n workflows exist yet (blocked on open question 5 — can't wire a
  workflow to a service with no reachable URL). No Airtable base exists yet
  (open question 6). No Creatomate account/credential exists yet (you'll add
  the key once you have one).

## 5. Open questions blocking further progress

1. ~~Real test clip~~ — provided (`IMG_8350 2.mov`). Used to validate and fix
   the detector (Risk 1). **Still needed from you:** confirm by ear whether
   the remaining dense ornamental-fragment count is real fast ornamentation
   or residual over-segmentation — I can point you at specific timestamps in
   `data/analysis/BR-TEST-001.json` if useful. Also, a genuine 30–60s
   vertical 9:16 clip would let the layout/render stage actually be tried.
2. ~~Creatomate decision~~ — you said build the workflow now, key comes
   later. Built: **"Björn Rō - Render Reel via Creatomate"**
   (https://zeras.app.n8n.cloud/workflow/iu7qnAulj8faWIYA) — Execute Workflow
   Trigger → HTTP Request (calls the Python service's `/render-script`) →
   HTTP Request (submits to Creatomate's REST API directly, since the
   dedicated `@creatomate/n8n-nodes-creatomate` community node isn't
   installed on this instance and its exact parameter shape can't be
   confirmed without installing it). Two things are flagged with sticky
   notes and left as placeholders on purpose: the service URL (blocked on
   open question 5), and the Creatomate HTTP node's credential, which does
   not exist yet — in n8n, create a new credential of type "Header Auth
   (Templated)", name it **Creatomate API Key** (exactly, so the node picks
   it up), set the header template to
   `{"headers":{"Authorization":"Bearer {{api_key}}"}}`, and fill in your
   real key once you have one. The workflow will not run until both are done.
3. ~~Quote sourcing~~ — resolved per Risk 3: public domain + original
   reflections for v1; Ram Dass/Neem Karoli Baba material only after you
   get written permission from LSRF.
4. ~~Airtable workspace~~ — you said a new base under the same account. Not
   yet created — see open question 6.
5. ~~Where should `tools/service.py` run?~~ — deployed to Railway:
   https://bjornro-production.up.railway.app (repo:
   https://github.com/dominationarmy/bjornro, auto-deploys from `main`).
   Confirmed working after fixing two real deploy bugs (Python version,
   font fallback — see Section 1). Wired into the n8n workflow's "Build
   RenderScript" node.
6. ~~Airtable base creation~~ — done. Base `appUc8UDDzYrXebsV`
   (https://airtable.com/appUc8UDDzYrXebsV), all three tables created with
   full field schema via Airtable's Meta API:
   - Clips: `tblE2gbbBY6XWjv9D`
   - Quote Proposals: `tblURfOUwaFmRS6rr` (linked to Clips)
   - Rendered Reels: `tblCqXgj8sTh5KIrT` (linked to both)

   The "ZerasMain" n8n credential turned out to lack `schema.bases:write`
   permission and wasn't scoped to this base, so a dedicated Airtable token
   (scopes: `data.records:read/write`, `schema.bases:read/write`, access
   limited to this one base) was created instead and used directly via the
   API for this one-time schema setup — it was not yet added as an n8n
   credential. **Still needed:** add that token as a new n8n credential
   (Settings → Credentials → New → Airtable Personal Access Token) before
   Phase 2's quote-proposal workflows can read/write records in this base.
