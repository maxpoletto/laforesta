"""Bound photo multipart streams before Django buffers them in memory/on disk."""

from django.core.files.uploadhandler import FileUploadHandler, StopUpload

from apps.base.observation_images import PhotoError, UploadBudget


class ObservationUploadHandler(FileUploadHandler):
    def __init__(self, request=None):
        super().__init__(request)
        self.budget = UploadBudget() if getattr(request, 'path_info', '') in {
            '/api/bosco/observations/save/', '/api/ipso/uploads/',
        } else None

    def new_file(self, *args, **kwargs):
        super().new_file(*args, **kwargs)
        if self.budget is not None:
            self._check(self.budget.start_file)

    def receive_data_chunk(self, raw_data, start):
        if self.budget is not None:
            self._check(self.budget.add, len(raw_data))
        return raw_data

    def file_complete(self, file_size):
        return None

    def _check(self, operation, *args):
        try:
            operation(*args)
        except PhotoError as exc:
            self.request.photo_upload_error = exc
            raise StopUpload(connection_reset=False) from exc
