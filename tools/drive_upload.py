#!/usr/bin/env python3
"""
Björn Rō — uploads a file to Google Drive using a service account, so
Railway can write cut clips back to Drive without routing them through n8n.

Requires GOOGLE_SERVICE_ACCOUNT_JSON in the environment: the full JSON key
content as a string (not a file path). Never hardcode it. The service
account must be shared (Editor) on the destination Drive folder -- sharing
a folder with it is a direct Drive ACL grant, independent of OAuth scopes,
so the restrictive `drive.file` scope below is sufficient: it only needs to
CREATE new files, never read/manage arbitrary existing ones.

STATUS: untested -- no service account has been created for this project
yet (see docs/architecture.md).
"""
from __future__ import annotations

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
    info = json.loads(raw)
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
    resp.raise_for_status()
    return resp.json()
