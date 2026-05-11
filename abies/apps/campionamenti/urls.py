"""Campionamenti URL patterns."""

from django.urls import path

from apps.campionamenti import views

urlpatterns = [
    path('grids/data/', views.grids_data, name='campionamenti-grids-data'),
    path('surveys/data/', views.surveys_data, name='campionamenti-surveys-data'),
    path('sample-areas/data/', views.sample_areas_data,
         name='campionamenti-sample-areas-data'),
    path('samples/data/', views.samples_data, name='campionamenti-samples-data'),
    path('trees/<int:survey_id>/', views.sampled_trees_data,
         name='campionamenti-trees-data'),
]
