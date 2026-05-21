"""Piano di taglio URL patterns."""

from django.urls import path

from apps.piano_di_taglio import views

urlpatterns = [
    # Read endpoints (digests).
    path('plans/data/', views.plans_data_view,
         name='piano-di-taglio-plans-data'),
    path('items/data/', views.items_data_view,
         name='piano-di-taglio-items-data'),
    path('regressions/data/', views.regressions_data_view,
         name='piano-di-taglio-regressions-data'),
    path('mark-trees/<int:item_id>/', views.mark_trees_data_view,
         name='piano-di-taglio-mark-trees-data'),

    # Plan CRUD.
    path('plan/form/', views.plan_form_view,
         name='piano-di-taglio-plan-form-add'),
    path('plan/form/<int:plan_id>/', views.plan_form_view,
         name='piano-di-taglio-plan-form-edit'),
    path('plan/save/', views.plan_save_view,
         name='piano-di-taglio-plan-save'),
    path('plan/delete/<int:plan_id>/', views.plan_delete_view,
         name='piano-di-taglio-plan-delete'),

    # Plan CSV import / plan-level Esporta CSV.
    path('plan/import-csv/', views.plan_csv_import_view,
         name='piano-di-taglio-plan-import-csv'),
    path('plan/export/<int:plan_id>/', views.plan_export_view,
         name='piano-di-taglio-plan-export'),

    # Plan-item CRUD.
    path('item/data/<int:item_id>/', views.item_data_view,
         name='piano-di-taglio-item-data'),
    path('item/form/', views.item_form_view,
         name='piano-di-taglio-item-form-add'),
    path('item/form/<int:item_id>/', views.item_form_view,
         name='piano-di-taglio-item-form-edit'),
    path('item/save/', views.item_save_view,
         name='piano-di-taglio-item-save'),
    path('item/delete/<int:item_id>/', views.item_delete_view,
         name='piano-di-taglio-item-delete'),

    # Per-item Esporta CSV (martellate + prelievi zip).
    path('item/export/<int:item_id>/', views.item_export_view,
         name='piano-di-taglio-item-export'),

    # Apri / Chiudi cantiere.
    path('transition/save/', views.transition_save_view,
         name='piano-di-taglio-transition-save'),
]
