"""Campionamenti API views.

M3a: stub endpoints serving the four eager digests + the per-survey
lazy digest.  Form endpoints come in M3d.
"""

from django.contrib.auth.decorators import login_required
from django.http import Http404

from apps.base.digests import serve_digest


@login_required
def grids_data(request):
    return serve_digest(request, 'grids')


@login_required
def surveys_data(request):
    return serve_digest(request, 'surveys')


@login_required
def sample_areas_data(request):
    return serve_digest(request, 'sample_areas')


@login_required
def samples_data(request):
    return serve_digest(request, 'samples')


@login_required
def sampled_trees_data(request, survey_id: int):
    """Per-survey sampled-tree digest.  Lazily generated on first hit."""
    if survey_id <= 0:
        raise Http404
    return serve_digest(request, f'sampled_trees_{survey_id}')
