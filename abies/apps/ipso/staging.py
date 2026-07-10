"""Filesystem helpers for Ipso staged uploads."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import timezone
from pathlib import Path

from django.conf import settings
from django.utils import timezone as django_timezone

from config.constants import (
    FIELD_DATE, FIELD_MODE, FIELD_OPERATOR, FIELD_REFERENCE_VERSION,
    FIELD_SCHEMA_VERSION, FIELD_SESSION_ID, FIELD_WORK_PACKAGE_ID,
    IPSO_UPLOAD_FILE_CSV, IPSO_UPLOAD_FILE_JSON, IPSO_UPLOAD_FILE_READY,
    IPSO_UPLOAD_FILE_SHA256,
    RECORDS, SESSION,
)

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def payload_checksum(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def upload_inbox_path(session_id: str) -> Path:
    now = django_timezone.now().astimezone(timezone.utc)
    return Path(settings.IPSO_INBOX_DIR) / f'{now.year:04d}' / f'{now.month:02d}' / session_id


def upload_record_date(payload: dict) -> str:
    records = payload.get(RECORDS, []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return ''
    dates = [
        row.get(FIELD_DATE)
        for row in records
        if isinstance(row, dict)
        and isinstance(row.get(FIELD_DATE), str)
        and _DATE_RE.match(row.get(FIELD_DATE))
    ]
    return min(dates) if dates else ''


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
        'record_date': upload_record_date(payload),
        'checksum': checksum,
        'inbox_path': str(inbox_path),
    }


def write_upload_files(
        session_dir: Path, payload: dict, checksum: str, csv_text: str | None,
) -> Path:
    _write_payload_content(session_dir, payload, checksum)
    if csv_text:
        _atomic_write_text(session_dir / IPSO_UPLOAD_FILE_CSV, csv_text)
    _atomic_write_text(session_dir / IPSO_UPLOAD_FILE_READY, checksum + '\n')
    return session_dir


def write_payload_files(session_dir: Path, payload: dict, checksum: str) -> Path:
    _write_payload_content(session_dir, payload, checksum)
    _atomic_write_text(session_dir / IPSO_UPLOAD_FILE_READY, checksum + '\n')
    return session_dir


def _write_payload_content(session_dir: Path, payload: dict, checksum: str) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        session_dir / IPSO_UPLOAD_FILE_JSON,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
    )
    _atomic_write_text(session_dir / IPSO_UPLOAD_FILE_SHA256, checksum + '\n')


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
