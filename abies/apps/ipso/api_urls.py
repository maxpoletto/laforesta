"""Ipso API URL patterns."""

from django.urls import path

from apps.ipso import views

urlpatterns = [
    path('inbox/', views.inbox_data, name='ipso-inbox-data'),
    path('uploads/', views.upload_session, name='ipso-upload-session'),
    path('uploads/<int:upload_id>/', views.upload_detail, name='ipso-upload-detail'),
    path('uploads/<int:upload_id>/reject/', views.reject_upload, name='ipso-upload-reject'),
    path('uploads/<int:upload_id>/import-martellate/',
         views.import_martellate_upload, name='ipso-upload-import-martellate'),
    path('uploads/<int:upload_id>/import-samples/',
         views.import_samples_upload, name='ipso-upload-import-samples'),
    path('uploads/<int:upload_id>/import-pai/',
         views.import_pai_upload, name='ipso-upload-import-pai'),
]
