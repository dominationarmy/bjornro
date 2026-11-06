#!/usr/bin/env python3
"""
Björn Rō — uploads a file to Google Drive using a service account, so
Railway can write cut clips back to Drive without routing them through n8n.

Requires GOOGLE_SERVICE_ACCOUNT_JSON in the environment: the service
account's JSON key, base64-encoded (not raw JSON, not a file path). Never
hardcode it. Base64 is required because the key's PEM-formatted private_key
field contains literal `\n` escape sequences that dashboard env-var UIs
(Railway included) are prone to mangling when pasted as raw multi-line JSON
-- base64 has no newlines/backslashes/quotes for that UI to corrupt. To
produce the value: `base64 -i service-account.json | pbcopy` (macOS).

The service account must be shared (Editor) on the destination Drive
folder -- sharing a folder with it is a direct Drive ACL grant, independent
of OAuth scopes, so the restrictive `drive.file` scope below is sufficient:
it only needs to CREATE new files, never read/manage arbitrary existing ones.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _get_access_token() -> str:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not set in the environment.")
    info = json.loads(base64.b64decode(raw))
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    creds.refresh(GoogleAuthRequest())
    return creds.token


def upload_file(local_path: Path, filename: str, folder_id: str) -> dict:
    token = _get_access_token()
    metadata = {"name": filename, "parents": [folder_id]}
    with open(local_path, "rb") as f:
        resp = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files"
            "?uploadType=multipart&fields=id,name,webViewLink",
            headers={"Authorization": f"Bearer {token}"},
            files={
                "metadata": (None, json.dumps(metadata), "application/json"),
                "file": (filename, f, "video/mp4"),
            },
            timeout=300,
        )
    if not resp.ok:
        raise RuntimeError(f"Drive upload failed ({resp.status_code}): {resp.text}")
    return resp.json()
