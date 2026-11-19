"""Receive ordered pages and publish one ordinary staged Ipso session."""
import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import timedelta
from pathlib import Path

from django.db import IntegrityError, transaction
from django.db.models import Max
from django.utils import timezone

from apps.base.file_io import fsync_directory
from apps.ipso import staging
from apps.ipso.models import IpsoPendingIdentifier, IpsoUpload, IpsoUploadPage
from config.constants import (
    FIELD_CLIENT_PHOTO_ID, FIELD_CLIENT_RECORD_ID, FIELD_DATE, FIELD_MODE, FIELD_PHOTOS,
    FIELD_SESSION_ID, IPSO_UPLOAD_FILE_CSV, IPSO_UPLOAD_FILE_JSON,
    IPSO_UPLOAD_FILE_PHOTOS_DIR, IPSO_UPLOAD_FILE_READY,
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


def receive_page(payload, csv_text, photos, offset, total, transfer_id, validate_reference_ids):
    session = payload[SESSION]
    session_id = session[FIELD_SESSION_ID]
    records = payload[RECORDS]
    end = offset + len(records)
    transfer = staging.payload_checksum([transfer_id, total, session])
    checksum = staging.payload_checksum([payload, csv_text])
    body = _page_body(records, csv_text, photos)
    pages = IpsoUploadPage.objects.filter(session_id=session_id)
    with transaction.atomic():
        # INSERT first takes SQLite's writer lock even when two final pages race.
        try:
            with transaction.atomic():
                page = IpsoUploadPage.objects.create(
                    session_id=session_id, offset=offset, end=end, transfer=transfer,
                    checksum=checksum, body=body, category_ids=staging.observation_category_ids(records),
                    region_ids=staging.field_region_ids(session, records),
                )
        except IntegrityError:
            page = pages.defer('body').get(offset=offset)
            if page.checksum != checksum or page.transfer != transfer:
                raise PageConflict
            if page.upload_id:
                return _replay_completed(pages, page, body, session, total)
            return {'next_offset': pages.order_by('-offset').values_list('end', flat=True).first()}

        if IpsoUpload.objects.filter(session_id=session_id).exists():
            raise PageConflict
        if offset:
            previous = pages.filter(offset__lt=offset).order_by('-offset').values('end', 'transfer').first()
            if not previous or previous['end'] != offset or previous['transfer'] != transfer:
                raise PageConflict
        else:
            _expire_bodies(session_id)
        # Accepted receipts may outlive a referenced species or category.
        # Only new pages need to resolve those references against the database.
        validate_reference_ids(session[FIELD_MODE], records)
        identifiers = [('record', row[FIELD_CLIENT_RECORD_ID]) for row in records]
        identifiers.extend(('photo', name) for name in photos)
        try:
            IpsoPendingIdentifier.objects.bulk_create([
                IpsoPendingIdentifier(page=page, session_id=session_id, kind=kind, value=value)
                for kind, value in identifiers
            ])
        except IntegrityError as error:
            raise PageConflict from error
        if end < total:
            return {'next_offset': end}

        started = pages.values_list('received_at', flat=True).get(offset=0)
        path = staging.upload_inbox_path(session_id, started)
        _publish(pages, session, path)
        IpsoPendingIdentifier.objects.filter(session_id=session_id).delete()
        return {'next_offset': total, 'stored_as': str(path), 'duplicate': False}


def _expire_bodies(current_session):
    cutoff = timezone.now() - timedelta(days=30)
    other = IpsoUploadPage.objects.exclude(session_id=current_session)
    pending = other.filter(upload__isnull=True)
    expired = pending.values('session_id').annotate(last=Max('received_at')).filter(last__lt=cutoff)
    pending.filter(session_id__in=expired.values('session_id')).only('pk').delete()
    # A phone can abandon a repair too. Keep the original completed receipts.
    repairs = other.filter(upload__isnull=False, body__isnull=False)
    expired = repairs.values('session_id').annotate(last=Max('received_at')).filter(last__lt=cutoff)
    repairs.filter(session_id__in=expired.values('session_id')).update(body=None)


def _staged_files_present(upload):
    root = Path(upload.inbox_path)
    if not (root / IPSO_UPLOAD_FILE_CSV).is_file():
        return False
    payload, error = staging.read_staged_payload(upload, require_ready=True)
    if error:
        return False
    return all(
        (root / IPSO_UPLOAD_FILE_PHOTOS_DIR / photo[FIELD_CLIENT_PHOTO_ID]).is_file()
        for row in payload[RECORDS] for photo in row.get(FIELD_PHOTOS, [])
    )


def _replay_completed(pages, page, body, session, total):
    upload = page.upload
    if pages.filter(body__isnull=False).exists() or not _staged_files_present(upload):
        pages.filter(pk=page.pk).update(body=body, received_at=timezone.now())
        missing = pages.filter(body__isnull=True).order_by('offset').values_list('offset', flat=True).first()
        if missing is not None:
            return {'next_offset': missing, 'recovering': True}
        # Mode is editable in Abies after upload. Preserve the reviewed checksum,
        # import state and history, even when the whole inbox directory was lost.
        _publish(pages, {**session, FIELD_MODE: upload.mode}, Path(upload.inbox_path), upload)
    return {'next_offset': total, 'duplicate': True, 'stored_as': upload.inbox_path}


def _publish(pages, session, path, upload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        root = Path(scratch) / 'complete'
        root.mkdir()
        digest, record_date, row_count, category_ids, region_ids = _assemble(root, pages, session)
        if upload is None:
            fields = staging.upload_model_fields({SESSION: session, RECORDS: []}, digest, path)
            upload = IpsoUpload.objects.create(**(fields | {
                'record_count': row_count, 'record_date': record_date, 'category_ids': category_ids, 'region_ids': region_ids,
            }))
        elif digest != upload.checksum:
            raise PageConflict

        fsync_directory(root)
        previous = Path(scratch) / 'previous'
        if path.exists():
            path.replace(previous)
        try:
            root.replace(path)
        except OSError:
            if previous.exists():
                previous.replace(path)
            raise
        fsync_directory(path.parent)
    pages.update(body=None, category_ids=None, region_ids=None, upload=upload)


def _assemble(root, pages, session):
    """Stream the canonical checksum representation, one bounded page at a time."""
    digest = hashlib.sha256()
    seen = set()
    row_count = 0
    previous_segment = None
    category_ids = set()
    region_ids = set()
    record_date = ''
    with (root / IPSO_UPLOAD_FILE_JSON).open('wb') as data, (root / IPSO_UPLOAD_FILE_CSV).open('wb') as csv:
        def emit(b):
            data.write(b)
            digest.update(b)
        emit(b'{"records":[')
        for body in pages.order_by('offset').values_list('body', flat=True).iterator(chunk_size=1):
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                records = json.loads(archive.read('records'))
                category_ids.update(staging.observation_category_ids(records))
                region_ids.update(staging.field_region_ids(session, records))
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
    return checksum, record_date, row_count, sorted(category_ids), sorted(region_ids)
