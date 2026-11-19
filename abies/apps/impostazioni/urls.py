"""Impostazioni URL patterns."""

from django.urls import path

from apps.impostazioni import views
from apps.impostazioni import cartography, equation_imports, regions
from config.constants import DATA_ID_ECLASSES, DATA_ID_ROAD_TYPES, IPSO_REF_OBSERVATION_CATEGORIES

urlpatterns = [
    path('cartography/', cartography.catalog),
    path('cartography/regions/', regions.save),
    path('cartography/regions/<int:region_id>/', regions.details),
    path('cartography/regions/<int:region_id>/save/', regions.save),
    path('cartography/regions/<int:region_id>/delete/', regions.save, {'delete': True}),
    path('cartography/<int:region_id>/', cartography.workspace),
    path('cartography/<int:region_id>/parcel-form/', cartography.parcel_form),
    path('cartography/<int:region_id>/<slug:action>/', cartography.change),
    path('equations/import/preview/', equation_imports.preview_import, name='impostazioni-equation-preview'),
    path('equations/import/confirm/', equation_imports.confirm_import, name='impostazioni-equation-confirm'),
    path('equations/volume/export/', equation_imports.export_volume, name='impostazioni-volume-export'),
    # Password
    path('password/', views.password_view, name='impostazioni-password'),

    # Landing page
    path('landing-page/data/', views.landing_page_data,
         name='impostazioni-landing-page-data'),
    path('landing-page/save/', views.landing_page_save,
         name='impostazioni-landing-page-save'),

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
    path('species/type/save/', views.species_type_save,
         name='impostazioni-species-type-save'),


    # Observation categories
    path('observation-categories/data/', views.observation_categories_data,
         name='impostazioni-observation-categories-data'),
    path('observation-categories/form/', views.observation_categories_form,
         name='impostazioni-observation-categories-form-add'),
    path('observation-categories/form/<int:obj_id>/',
         views.observation_categories_form,
         name='impostazioni-observation-categories-form-edit'),
    path('observation-categories/save/', views.observation_categories_save,
         name='impostazioni-observation-categories-save'),
    path('observation-categories/delete/', views.delete_reference, {'data_id': IPSO_REF_OBSERVATION_CATEGORIES},
         name='impostazioni-observation-categories-delete'),

    # Road types
    path('road-types/data/', views.road_types_data, name='impostazioni-road-types-data'),
    path('road-types/form/', views.road_types_form, name='impostazioni-road-types-form-add'),
    path('road-types/form/<int:obj_id>/', views.road_types_form, name='impostazioni-road-types-form-edit'),
    path('road-types/save/', views.road_types_save, name='impostazioni-road-types-save'),
    path('road-types/delete/', views.delete_reference, {'data_id': DATA_ID_ROAD_TYPES},
         name='impostazioni-road-types-delete'),

    path('eclasses/data/', views.eclasses_data, name='impostazioni-eclasses-data'),
    path('eclasses/form/', views.eclasses_form, name='impostazioni-eclasses-form-add'),
    path('eclasses/form/<int:obj_id>/', views.eclasses_form, name='impostazioni-eclasses-form-edit'),
    path('eclasses/save/', views.eclasses_save, name='impostazioni-eclasses-save'),
    path('eclasses/delete/', views.delete_reference, {'data_id': DATA_ID_ECLASSES},
         name='impostazioni-eclasses-delete'),

    # Bosco source settings (writer+)
    path('dendrometry/data/', views.dendrometry_data,
         name='impostazioni-dendrometry-data'),
    path('dendrometry/save/', views.dendrometry_save,
         name='impostazioni-dendrometry-save'),

    # Users (admin only)
    path('users/data/', views.users_data, name='impostazioni-users-data'),
    path('users/form/', views.users_form, name='impostazioni-users-form-add'),
    path('users/form/<int:obj_id>/', views.users_form, name='impostazioni-users-form-edit'),
    path('users/save/', views.users_save, name='impostazioni-users-save'),

    # Hypsometric parameters (writer+)
    path('hypso-params/data/', views.hypso_params_data,
         name='impostazioni-hypso-data'),
    path('hypso-params/active-set/', views.hypso_params_active_set,
         name='impostazioni-hypso-active-set'),
    path('hypso-params/surveys/', views.hypso_params_surveys,
         name='impostazioni-hypso-surveys'),
    path('hypso-params/compute/', views.hypso_params_compute,
         name='impostazioni-hypso-compute'),
    path('hypso-params/accept/', views.hypso_params_accept,
         name='impostazioni-hypso-accept'),
    path('hypso-params/import/', views.hypso_params_import,
         name='impostazioni-hypso-import'),
    path('hypso-params/export/', views.hypso_params_export,
         name='impostazioni-hypso-export'),
    path('hypso-params/clear/', views.hypso_params_clear,
         name='impostazioni-hypso-clear'),

    # Diameter classes (writer+)
    path('diameter-classes/data/', views.diameter_classes_data,
         name='impostazioni-diameter-classes-data'),
    path('diameter-classes/save/', views.diameter_classes_save,
         name='impostazioni-diameter-classes-save'),
]
