"""Bosco URL patterns."""

from django.urls import path

from apps.bosco import views

urlpatterns = [
    path('parcels/data/', views.parcels_data, name='bosco-parcels-data'),
    path('species/data/', views.species_data, name='bosco-species-data'),
    path('preserved-trees/data/', views.preserved_trees_data,
         name='bosco-preserved-trees-data'),
    path('future-production/data/', views.future_production_data,
         name='bosco-future-production-data'),
    path('parcel-dendrometry/data/', views.parcel_dendrometry_data,
         name='bosco-parcel-dendrometry-data'),
    path('parcel-dendrometry-points/data/', views.parcel_dendrometry_points_data,
         name='bosco-parcel-dendrometry-points-data'),
]
