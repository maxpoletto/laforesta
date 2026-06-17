"""Ipso API URL patterns."""

from django.urls import path

from apps.ipso import views

urlpatterns = [
    path('uploads/', views.upload_session, name='ipso-upload-session'),
]
