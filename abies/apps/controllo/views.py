"""Controllo (audit) views: read-only audit log."""

from django.contrib.auth.decorators import login_required

from apps.base.digests import serve_digest


@login_required
def data_view(request):
    """Serve audit.json.gz (conditional GET + lazy regeneration)."""
    return serve_digest(request, 'audit')
