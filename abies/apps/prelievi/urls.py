"""Prelievi URL patterns."""

from django.urls import path

from apps.prelievi import views

urlpatterns = [
    path('data/', views.data_view, name='prelievi-data'),
    path('form/', views.form_view, name='prelievi-form-add'),
    path('form/<int:op_id>/', views.form_view, name='prelievi-form-edit'),
    path('save/', views.save_view, name='prelievi-save'),
    path('delete/', views.delete_view, name='prelievi-delete'),
]
