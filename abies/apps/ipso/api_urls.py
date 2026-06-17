"""Ipso API URL patterns."""

from django.urls import path

from apps.ipso import views

urlpatterns = [
    path('inbox/', views.inbox_data, name='ipso-inbox-data'),
    path('uploads/', views.upload_session, name='ipso-upload-session'),
    path('uploads/<int:upload_id>/', views.upload_detail, name='ipso-upload-detail'),
    path('uploads/<int:upload_id>/reject/', views.reject_upload, name='ipso-upload-reject'),
]
