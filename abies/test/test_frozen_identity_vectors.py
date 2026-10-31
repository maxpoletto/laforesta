"""These serializers deliberately represent different durable identities."""
from datetime import date
from decimal import Decimal

from django.test import RequestFactory

from apps.base.idempotency import _request_hash
from apps.ipso.staging import payload_checksum
from apps.piano_di_taglio.mark_import import (
    csv_mark_fingerprint, ipso_mark_fingerprint, legacy_csv_mark_fingerprint,
)


def identity_vectors():
    record = dict(client_record_id='7', date='2026-10-01', parcel_id=100,
        species_id=10, number=None, d_cm=42, h_m='22.10', lat=38.5, lon=16.3)
    body = dict(nonce='frozen-retry', row_id=17, note='Mário 🌲')
    request = RequestFactory().post('/api/frozen/?parcel=01', data='{}', content_type='application/json')
    return {
        'csv_v2': csv_mark_fingerprint(source_row=2, date=date(2026, 10, 1),
            parcel_id=100, species_id=10, number=None, d_cm=42, h_m=Decimal('22.10'),
            h_measured=False, lat=38.5, lon=16.3, acc_m=5, operator='Mário 🌲'),
        'csv_legacy': legacy_csv_mark_fingerprint(date=date(2026, 10, 1),
            species_name='Abete', d_cm=42, h_m=Decimal('22.10'), lat=38.5,
            lon=16.3, operator='Mário 🌲'),
        'ipso_record': ipso_mark_fingerprint('frozen-session', record),
        'staged_payload': payload_checksum(dict(schema_version=1,
            session=dict(session_id='frozen-session', operator='Mário 🌲'), records=[record])),
        'nonce_request': _request_hash(request, body),
    }


def test_frozen_identity_vectors():
    assert identity_vectors() == EXPECTED


# Golden values: change only with an explicit durable-identity migration.
EXPECTED = {'csv_v2': 'v2:dea38e83934ace7a34c0f321247e7959a676bd87e2cd711e5ec9665c675ee3bd',
 'csv_legacy': 'ba6b2282e4257d6d3d9c4bcc0015485d2981b86108e45531f5dd8c1bab57bb94',
 'ipso_record': 'f8b3aac36d01f64ff1cbf76aea96260cc82f36ae087fe90273b4fa50d1a994eb',
 'staged_payload': '75bc992d60ab291f1ff12cafb7d88f6713e5610fe67479f53cbf0a4d3d771594',
 'nonce_request': 'aa7d9b70a580296c72da4611457ccc8a707181592dfd3a8a0933380cf3b24c5e'}
