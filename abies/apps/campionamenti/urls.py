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
    # Tree-sample CRUD (M3d-write).
    path('tree/form/', views.tree_form_view, name='campionamenti-tree-form-add'),
    path('tree/form/<int:ts_id>/', views.tree_form_view,
         name='campionamenti-tree-form-edit'),
    path('tree/save/', views.tree_save_view, name='campionamenti-tree-save'),
    path('tree/delete/<int:ts_id>/', views.tree_delete_view,
         name='campionamenti-tree-delete'),
    # Sample area CRUD (Section 1).
    path('area/form/', views.area_form_view, name='campionamenti-area-form-add'),
    path('area/form/<int:area_id>/', views.area_form_view,
         name='campionamenti-area-form-edit'),
    path('area/save/', views.area_save_view, name='campionamenti-area-save'),
    path('area/delete/<int:area_id>/', views.area_delete_view,
         name='campionamenti-area-delete'),
    path('grid/form/', views.grid_form_view, name='campionamenti-grid-form'),
    path('grid/save/', views.grid_save_view, name='campionamenti-grid-save'),
    path('grid/save-auto/', views.grid_save_auto_view,
         name='campionamenti-grid-save-auto'),
    path('grid/import-csv/', views.grid_csv_import_view,
         name='campionamenti-grid-import-csv'),
    path('survey/import-csv/', views.tree_csv_import_view,
         name='campionamenti-tree-import-csv'),
    path('grid/edit/<int:grid_id>/', views.grid_edit_view,
         name='campionamenti-grid-edit'),
    path('grid/delete/<int:grid_id>/', views.grid_delete_view,
         name='campionamenti-grid-delete'),
    path('survey/form/', views.survey_form_view, name='campionamenti-survey-form'),
    path('survey/save/', views.survey_save_view, name='campionamenti-survey-save'),
    path('survey/edit/<int:survey_id>/', views.survey_edit_view,
         name='campionamenti-survey-edit'),
    path('survey/delete/<int:survey_id>/', views.survey_delete_view,
         name='campionamenti-survey-delete'),
]
