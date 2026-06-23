"""Filesystem helpers for Ipso staged uploads."""

from __future__ import annotations

import hashlib
import json
from datetime import timezone
from pathlib import Path

from django.conf import settings
from django.utils import timezone as django_timezone

from config.constants import (
    FIELD_MODE, FIELD_OPERATOR, FIELD_REFERENCE_VERSION, FIELD_SCHEMA_VERSION,
    FIELD_SESSION_ID, FIELD_WORK_PACKAGE_ID, IPSO_UPLOAD_FILE_CSV,
    IPSO_UPLOAD_FILE_JSON, IPSO_UPLOAD_FILE_SHA256, RECORDS, SESSION,
)


def payload_checksum(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def upload_inbox_path(session_id: str) -> Path:
    now = django_timezone.now().astimezone(timezone.utc)
    return Path(settings.IPSO_INBOX_DIR) / f'{now.year:04d}' / f'{now.month:02d}' / session_id


def upload_model_fields(payload: dict, checksum: str, inbox_path: Path) -> dict:
    session = payload[SESSION]
    return {
        FIELD_SESSION_ID: session[FIELD_SESSION_ID],
        FIELD_MODE: session[FIELD_MODE],
        FIELD_SCHEMA_VERSION: session[FIELD_SCHEMA_VERSION],
        FIELD_REFERENCE_VERSION: session.get(FIELD_REFERENCE_VERSION, ''),
        FIELD_WORK_PACKAGE_ID: session.get(FIELD_WORK_PACKAGE_ID, ''),
        FIELD_OPERATOR: session.get(FIELD_OPERATOR, ''),
        'record_count': len(payload[RECORDS]),
        'checksum': checksum,
        'inbox_path': str(inbox_path),
    }


def write_upload_files(
        session_dir: Path, payload: dict, checksum: str, csv_text: str | None,
) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        session_dir / IPSO_UPLOAD_FILE_JSON,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
    )
    _atomic_write_text(session_dir / IPSO_UPLOAD_FILE_SHA256, checksum + '\n')
    if csv_text:
        _atomic_write_text(session_dir / IPSO_UPLOAD_FILE_CSV, csv_text)
    return session_dir


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
