"""Serve the Ipso offline PWA from the Abies origin."""

from __future__ import annotations

import json
import re
from datetime import timezone
from pathlib import Path

from django.conf import settings
from django.contrib.staticfiles import finders
from django.http import Http404, HttpRequest, HttpResponse
from django.utils import timezone as django_timezone
from django.views.decorators.http import require_GET

from apps.base.http import (
    CACHE_NO_CACHE, CACHE_NO_STORE, apply_cache_control,
    conditional_file_response,
)
from apps.base.models import HYPSO_FUNC_LN, HypsoParam, HypsoParamSet, Parcel, Species

SCHEMA_VERSION = 1

ASSET_CONTENT_TYPES = {
    '.css': 'text/css; charset=utf-8',
    '.gif': 'image/gif',
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.png': 'image/png',
    '.webmanifest': 'application/manifest+json',
}

ASSET_FILES = {
    'app.js',
    'csv.js',
    'download.js',
    'geo.js',
    'gps.js',
    'index.html',
    'ipso.js',
    'manifest.webmanifest',
    'numpad.js',
    'parcel-locator.js',
    'session.js',
    'store.js',
    'strings.js',
    'style.css',
    'sw.js',
    'upload-config.js',
    'upload.js',
    'version.js',
    'img/f.gif',
    'img/l.gif',
    'img/icon-192.png',
    'img/icon-512.png',
    'img/icon-512-maskable.png',
}

_NATKEY_RE = re.compile(r'(\d+)')


def _natural_key(value: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in _NATKEY_RE.split(value)]


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    return _asset_response(request, 'index.html')


@require_GET
def asset(request: HttpRequest, asset_path: str) -> HttpResponse:
    if asset_path in {'reference.json', 'terreni.geojson'}:
        raise Http404
    return _asset_response(request, asset_path)


def _asset_response(request: HttpRequest, asset_path: str) -> HttpResponse:
    if asset_path not in ASSET_FILES:
        raise Http404
    path = _static_path(asset_path)
    content_type = ASSET_CONTENT_TYPES.get(Path(asset_path).suffix)
    if content_type is None:
        raise Http404
    return conditional_file_response(
        request, path, content_type=content_type, cache_control=CACHE_NO_CACHE,
    )


def _static_path(asset_path: str) -> Path:
    found = finders.find(f'ipso/{asset_path}')
    if not found:
        raise Http404
    return Path(found)


@require_GET
def reference_json(request: HttpRequest) -> HttpResponse:
    """Current-client reference bundle, generated from Abies data.

    This intentionally preserves the legacy `reference.json` shape needed by
    the existing Ipso client. The stricter versioned manifest described in the
    integration design will be added as a separate API contract.
    """
    payload = {
        'schema_version': SCHEMA_VERSION,
        'generated_at': django_timezone.now().astimezone(timezone.utc)
                       .isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'species': _species_rows(),
        'parcels': _parcel_rows(),
        'ipsometrica': _ipsometrica(),
    }
    return _json_response(payload, content_type='application/json')


def _species_rows() -> list[dict]:
    return [
        {
            'common': sp.common_name,
            'latin': sp.latin_name,
            'sort_order': sp.sort_order,
            'density': float(sp.density),
        }
        for sp in Species.objects.filter(active=True).order_by('sort_order', 'common_name')
    ]


def _parcel_rows() -> list[dict]:
    rows = [
        {'compresa': p.region.name, 'particella': p.name}
        for p in (Parcel.objects
                  .select_related('region', 'eclass')
                  .filter(eclass__coppice=False))
    ]
    rows.sort(key=lambda p: (p['compresa'], _natural_key(p['particella'])))
    return rows


def _ipsometrica() -> dict:
    active = HypsoParamSet.objects.active().first()
    if active is None:
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    params = (HypsoParam.objects
              .filter(param_set=active, func=HYPSO_FUNC_LN)
              .select_related('region', 'species')
              .order_by('region__name', 'species__common_name'))
    for p in params:
        out.setdefault(p.region.name, {})[p.species.common_name] = {
            'a': float(p.a),
            'b': float(p.b),
        }
    return out


@require_GET
def terreni_geojson(request: HttpRequest) -> HttpResponse:
    path = Path(settings.GEO_DIR) / 'terreni.geojson'
    if path.is_file():
        return conditional_file_response(
            request, path, content_type='application/geo+json',
            cache_control=CACHE_NO_CACHE,
        )
    return _json_response(
        {'type': 'FeatureCollection', 'features': []},
        content_type='application/geo+json', cache_control=CACHE_NO_CACHE,
    )


def _json_response(
        payload: dict, *, content_type: str,
        cache_control: str = CACHE_NO_STORE,
) -> HttpResponse:
    response = HttpResponse(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
        content_type=content_type,
    )
    apply_cache_control(response, cache_control)
    return response
