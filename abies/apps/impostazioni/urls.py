"""Impostazioni URL patterns."""

from django.urls import path

from apps.impostazioni import views

urlpatterns = [
    # Password
    path('password/', views.password_view, name='impostazioni-password'),

    # Crews
    path('crews/data/', views.crews_data, name='impostazioni-crews-data'),
    path('crews/form/', views.crews_form, name='impostazioni-crews-form-add'),
    path('crews/form/<int:obj_id>/', views.crews_form, name='impostazioni-crews-form-edit'),
    path('crews/save/', views.crews_save, name='impostazioni-crews-save'),

    # Tractors
    path('tractors/data/', views.tractors_data, name='impostazioni-tractors-data'),
    path('tractors/form/', views.tractors_form, name='impostazioni-tractors-form-add'),
    path('tractors/form/<int:obj_id>/', views.tractors_form, name='impostazioni-tractors-form-edit'),
    path('tractors/save/', views.tractors_save, name='impostazioni-tractors-save'),

    # Species
    path('species/data/', views.species_data, name='impostazioni-species-data'),
    path('species/form/', views.species_form, name='impostazioni-species-form-add'),
    path('species/form/<int:obj_id>/', views.species_form, name='impostazioni-species-form-edit'),
    path('species/save/', views.species_save, name='impostazioni-species-save'),

    # Users (admin only)
    path('users/data/', views.users_data, name='impostazioni-users-data'),
    path('users/form/', views.users_form, name='impostazioni-users-form-add'),
    path('users/form/<int:obj_id>/', views.users_form, name='impostazioni-users-form-edit'),
    path('users/save/', views.users_save, name='impostazioni-users-save'),
]
