"""Ipso PWA URL patterns."""

from django.urls import path

from apps.ipso import views

urlpatterns = [
    path('', views.index, name='ipso-index'),
    path('index.html', views.index, name='ipso-index-html'),
    path('reference.json', views.reference_json, name='ipso-reference-json'),
    path('map.geojson', views.device_map, name='ipso-map-geojson'),
    path('terreni.geojson', views.device_map, name='ipso-terreni-geojson'),
    path('<path:asset_path>', views.asset, name='ipso-asset'),
]
