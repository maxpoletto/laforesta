"""Mannesi URL patterns."""

from django.urls import path

from apps.mannesi import views

urlpatterns = [
    path('meta/', views.meta_view, name='mannesi-meta'),
    path('license-plates/save/', views.license_plate_save, name='mannesi-license-save'),

    path('hours/data/', views.hours_data, name='mannesi-hours-data'),
    path('hours/form/', views.hours_form, name='mannesi-hours-form-add'),
    path('hours/form/<int:obj_id>/', views.hours_form, name='mannesi-hours-form-edit'),
    path('hours/save/', views.hours_save, name='mannesi-hours-save'),
    path('hours/delete/', views.hours_delete, name='mannesi-hours-delete'),

    path('credits/data/', views.credits_data, name='mannesi-credits-data'),
    path('credits/form/', views.credits_form, name='mannesi-credits-form-add'),
    path('credits/form/<int:obj_id>/', views.credits_form, name='mannesi-credits-form-edit'),
    path('credits/save/', views.credits_save, name='mannesi-credits-save'),
    path('credits/delete/', views.credits_delete, name='mannesi-credits-delete'),
]
