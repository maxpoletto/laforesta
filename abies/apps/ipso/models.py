"""Models for Ipso staged uploads."""

from django.conf import settings
from django.db import models
from simple_history.models import HistoricalRecords

from config.constants import (
    IPSO_UPLOAD_STATE_CONFLICT, IPSO_UPLOAD_STATE_IMPORTED,
    IPSO_UPLOAD_STATE_RECEIVED, IPSO_UPLOAD_STATE_REJECTED,
)


class IpsoUploadState(models.TextChoices):
    RECEIVED = IPSO_UPLOAD_STATE_RECEIVED, IPSO_UPLOAD_STATE_RECEIVED
    IMPORTED = IPSO_UPLOAD_STATE_IMPORTED, IPSO_UPLOAD_STATE_IMPORTED
    REJECTED = IPSO_UPLOAD_STATE_REJECTED, IPSO_UPLOAD_STATE_REJECTED
    CONFLICT = IPSO_UPLOAD_STATE_CONFLICT, IPSO_UPLOAD_STATE_CONFLICT


class IpsoUpload(models.Model):
    """One completed Ipso session staged for later Abies import."""

    session_id = models.CharField(max_length=64, unique=True)
    mode = models.CharField(max_length=32)
    schema_version = models.IntegerField()
    reference_version = models.CharField(max_length=100, blank=True)
    work_package_id = models.CharField(max_length=100, blank=True)
    operator = models.CharField(max_length=100, blank=True)
    record_count = models.IntegerField(default=0)
    record_date = models.CharField(max_length=10, blank=True, default='')
    checksum = models.CharField(max_length=64)
    inbox_path = models.CharField(max_length=500)
    state = models.CharField(
        max_length=20,
        choices=IpsoUploadState.choices,
        default=IpsoUploadState.RECEIVED,
    )
    received_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    imported_at = models.DateTimeField(null=True, blank=True)
    imported_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='ipso_imports',
    )
    target_type = models.CharField(max_length=50, blank=True)
    target_id = models.IntegerField(null=True, blank=True)
    error_summary = models.TextField(blank=True)
    history = HistoricalRecords()

    class Meta:
        ordering = ['-received_at']

    def save(self, *args, **kwargs):
        update_fields = kwargs.get('update_fields')
        if update_fields is not None and 'updated_at' not in update_fields:
            kwargs['update_fields'] = [*update_fields, 'updated_at']
        super().save(*args, **kwargs)

    def __str__(self):
        return f'{self.mode}:{self.session_id}'
