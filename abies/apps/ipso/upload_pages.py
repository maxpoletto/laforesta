"""Receive ordered pages and publish one ordinary staged Ipso session."""
import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import timedelta
from pathlib import Path

from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from apps.base.file_io import fsync_directory
from apps.ipso import staging
from apps.ipso.models import IpsoUpload, IpsoUploadPage
from config.constants import (
    FIELD_CLIENT_RECORD_ID, FIELD_DATE, FIELD_SESSION_ID, IPSO_UPLOAD_FILE_CSV,
    IPSO_UPLOAD_FILE_JSON, IPSO_UPLOAD_FILE_PHOTOS_DIR, IPSO_UPLOAD_FILE_READY,
    IPSO_UPLOAD_FILE_SHA256, RECORDS, SESSION,
)


class PageConflict(Exception):
    pass


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def _page_body(records, csv_text, photos):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        archive.writestr('records', _json_bytes(records))
        archive.writestr('csv', (csv_text or '').encode())
        for name, content in photos.items():
            archive.writestr('photos/' + name, content)
    return out.getvalue()


def receive_page(payload, csv_text, photos, offset, total, transfer_id):
    session = payload[SESSION]
    session_id = session[FIELD_SESSION_ID]
    records = payload[RECORDS]
    end = offset + len(records)
    # Bind all pages to the frozen client transfer and normalized session header.
    transfer = staging.payload_checksum([transfer_id, total, session])
    checksum = staging.payload_checksum([payload, csv_text])
    body = _page_body(records, csv_text, photos)
    pages = IpsoUploadPage.objects.filter(session_id=session_id)
    with transaction.atomic():
        # INSERT first acquires SQLite's writer lock, including on replay.
        IpsoUploadPage.objects.bulk_create([IpsoUploadPage(
            session_id=session_id, offset=offset, end=end, transfer=transfer,
            checksum=checksum, body=body,
        )], ignore_conflicts=True)
        if offset == 0:
            # A phone may abandon its transfer permanently. Its frozen local
            # session can start over after these temporary bodies expire.
            pending = IpsoUploadPage.objects.filter(upload__isnull=True).exclude(session_id=session_id)
            expired = (pending.values('session_id').annotate(last=Max('received_at'))
                       .filter(last__lt=timezone.now() - timedelta(days=30)))
            pending.filter(session_id__in=expired.values('session_id')).delete()
        page = pages.only('transfer', 'checksum', 'upload_id').get(offset=offset)
        if page.checksum != checksum or page.transfer != transfer:
            raise PageConflict
        if page.upload_id:
            return {'next_offset': total, 'duplicate': True, 'stored_as': page.upload.inbox_path}
        previous = pages.filter(offset__lt=offset).order_by('-offset').values('end', 'transfer').first()
        if offset and (not previous or previous['end'] != offset or previous['transfer'] != transfer):
            raise PageConflict
        if IpsoUpload.objects.filter(session_id=session_id).exists():
            raise PageConflict
        next_offset = pages.order_by('-offset').values_list('end', flat=True).first()
        if next_offset < total:
            return {'next_offset': next_offset}
        path = staging.upload_inbox_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
            root = Path(scratch) / 'complete'
            root.mkdir()
            digest, record_date, row_count = _assemble(root, pages, session)
            fields = staging.upload_model_fields({SESSION: session, RECORDS: []}, digest, path)
            upload = IpsoUpload.objects.create(**(fields | {
                'record_count': row_count, 'record_date': record_date,
            }))
            # A crash before DB commit can leave an unpublished directory.
            # Publish the complete replacement together, including its photos.
            fsync_directory(root)
            if path.exists():
                path.replace(Path(scratch) / 'unpublished')
            root.replace(path)
            fsync_directory(path.parent)
        pages.update(body=None, upload=upload)
        return {'next_offset': total, 'stored_as': str(path), 'duplicate': False}


def _assemble(root, pages, session):
    """Stream the canonical checksum representation, one bounded page at a time."""
    digest = hashlib.sha256()
    seen = set()
    row_count = 0
    previous_segment = None
    record_date = ''
    with (root / IPSO_UPLOAD_FILE_JSON).open('wb') as data, (root / IPSO_UPLOAD_FILE_CSV).open('wb') as csv:
        def emit(b):
            data.write(b)
            digest.update(b)
        emit(b'{"records":[')
        for body in pages.order_by('offset').values_list('body', flat=True).iterator(chunk_size=1):
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                records = json.loads(archive.read('records'))
                row_count += staging.count_preview_rows(records, previous_segment)
                previous_segment = records[-1].get('segment')
                for row in records:
                    key = row[FIELD_CLIENT_RECORD_ID]
                    if key in seen:
                        raise PageConflict
                    if seen:
                        emit(b',')
                    seen.add(key)
                    emit(_json_bytes(row))
                    if not record_date or row[FIELD_DATE] < record_date:
                        record_date = row[FIELD_DATE]
                csv.write(archive.read('csv'))
                for member in archive.namelist():
                    if member.startswith('photos/'):
                        dest = root / IPSO_UPLOAD_FILE_PHOTOS_DIR / member.removeprefix('photos/')
                        if dest.exists():
                            raise PageConflict
                        dest.parent.mkdir(exist_ok=True)
                        staging._atomic_write_bytes(dest, archive.read(member))
        emit(b'],"session":' + _json_bytes(session) + b'}')
        for file in (data, csv):
            file.flush()
            os.fsync(file.fileno())
    checksum = digest.hexdigest()
    for name in (IPSO_UPLOAD_FILE_SHA256, IPSO_UPLOAD_FILE_READY):
        staging._atomic_write_text(root / name, checksum + '\n')
    return checksum, record_date, row_count
