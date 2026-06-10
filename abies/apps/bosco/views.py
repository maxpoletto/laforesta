"""Bosco API views."""

from django.contrib.auth.decorators import login_required

from apps.base.digests import serve_digest
from config.constants import (
    DIGEST_FUTURE_PRODUCTION, DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES, FIELD_SPECIES,
)


@login_required
def parcels_data(request):
    return serve_digest(request, 'parcels')


@login_required
def species_data(request):
    return serve_digest(request, FIELD_SPECIES)


@login_required
def preserved_trees_data(request):
    return serve_digest(request, DIGEST_PRESERVED_TREES)


@login_required
def future_production_data(request):
    return serve_digest(request, DIGEST_FUTURE_PRODUCTION)


@login_required
def parcel_dendrometry_data(request):
    return serve_digest(request, DIGEST_PARCEL_DENDROMETRY)


@login_required
def parcel_dendrometry_points_data(request):
    return serve_digest(request, DIGEST_PARCEL_DENDROMETRY_POINTS)
