"""Controllo URL patterns."""

from django.urls import path

from apps.controllo import views

urlpatterns = [
    path('data/', views.data_view, name='controllo-data'),
]
