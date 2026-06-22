"""Squadre URL patterns."""

from django.urls import path

from apps.mannesi import views

urlpatterns = [
    path('crews/data/', views.crews_data, name='squadre-crews-data'),
    path('crews/form/', views.crews_form, name='squadre-crews-form-add'),
    path('crews/form/<int:obj_id>/', views.crews_form, name='squadre-crews-form-edit'),
    path('crews/save/', views.crews_save, name='squadre-crews-save'),

    path('meta/', views.meta_view, name='squadre-meta'),
    path('hours/data/', views.hours_data, name='squadre-hours-data'),
    path('hours/form/', views.hours_form, name='squadre-hours-form-add'),
    path('hours/form/<int:obj_id>/', views.hours_form, name='squadre-hours-form-edit'),
    path('hours/save/', views.hours_save, name='squadre-hours-save'),
    path('hours/delete/', views.hours_delete, name='squadre-hours-delete'),

    path('credits/data/', views.credits_data, name='squadre-credits-data'),
    path('credits/form/', views.credits_form, name='squadre-credits-form-add'),
    path('credits/form/<int:obj_id>/', views.credits_form, name='squadre-credits-form-edit'),
    path('credits/save/', views.credits_save, name='squadre-credits-save'),
    path('credits/delete/', views.credits_delete, name='squadre-credits-delete'),
]
