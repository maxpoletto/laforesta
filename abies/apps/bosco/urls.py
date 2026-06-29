"""Bosco URL patterns."""

from django.urls import path

from apps.bosco import views

urlpatterns = [
    path('parcels/data/', views.parcels_data, name='bosco-parcels-data'),
    path('parcels/metadata/form/<int:parcel_id>/',
         views.parcel_metadata_form_view,
         name='bosco-parcel-metadata-form'),
    path('parcels/metadata/save/', views.parcel_metadata_save_view,
         name='bosco-parcel-metadata-save'),
    path('preserved-trees/data/', views.preserved_trees_data,
         name='bosco-preserved-trees-data'),
    path('pai/form/', views.pai_form_view, name='bosco-pai-form-add'),
    path('pai/form/<int:tree_id>/', views.pai_form_view,
         name='bosco-pai-form-edit'),
    path('pai/save/', views.pai_save_view, name='bosco-pai-save'),
    path('pai/delete/', views.pai_delete_view, name='bosco-pai-delete'),
    path('future-production/data/', views.future_production_data,
         name='bosco-future-production-data'),
    path('parcel-dendrometry/data/', views.parcel_dendrometry_data,
         name='bosco-parcel-dendrometry-data'),
    path('parcel-dendrometry-points/data/', views.parcel_dendrometry_points_data,
         name='bosco-parcel-dendrometry-points-data'),
    path('satellite/<int:region_id>/manifest/', views.satellite_manifest,
         name='bosco-satellite-manifest'),
    path('satellite/<int:region_id>/timeseries/', views.satellite_timeseries,
         name='bosco-satellite-timeseries'),
    path('satellite/<int:region_id>/raw/parcel-mask.json',
         views.satellite_mask_raw, name='bosco-satellite-mask-raw'),
    path('satellite/<int:region_id>/raw/<str:layer>/<str:date>.json',
         views.satellite_raw, name='bosco-satellite-raw'),
]
