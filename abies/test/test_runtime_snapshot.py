"""The snapshot guard must outlive the nonce transaction and its commit hooks."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

import pytest
from django.db import transaction

from apps.base.idempotency import begin_nonce_write
from apps.base.middleware import NonceMiddleware, RuntimeSnapshotMiddleware
from apps.base.responses import success_response
from apps.base.runtime_snapshot import runtime_snapshot_lock


@pytest.mark.django_db(transaction=True)
def test_media_request_keeps_backup_waiting_through_commit_hooks(writer_user, rf, settings):
    request = rf.post('/api/bosco/observations/save/', json.dumps({'nonce': 'snapshot'}),
                      content_type='application/json')
    request.user = writer_user
    entered = Event()
    observed = []

    def backup():
        with runtime_snapshot_lock(settings.RUNTIME_SNAPSHOT_LOCK_PATH, exclusive=True):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        tasks = []

        def mutate(request):
            tasks.append(pool.submit(backup))
            assert not entered.wait(0.05)
            begin_nonce_write(request)
            transaction.on_commit(lambda: observed.append(entered.is_set()))
            return success_response(request, {'nonce': 'snapshot'})

        response = RuntimeSnapshotMiddleware(NonceMiddleware(mutate))(request)
        assert response.status_code == 200
        tasks[0].result(timeout=5)
    assert observed == [False]
    assert entered.is_set()
    assert settings.MIDDLEWARE.index('apps.base.middleware.RuntimeSnapshotMiddleware') < settings.MIDDLEWARE.index('apps.base.middleware.NonceMiddleware')
