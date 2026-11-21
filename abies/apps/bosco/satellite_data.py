"""Shared identity contract for built and served satellite parcel products."""
import hashlib
import json


PARCEL_SCHEMA_VERSION = 3


def parcel_snapshot_key(rows):
    pieces = sorted(json.dumps([str(key), geometry], sort_keys=True, separators=(',', ':'))
                    for key, geometry in rows)
    return hashlib.sha256('\n'.join(pieces).encode()).hexdigest()
