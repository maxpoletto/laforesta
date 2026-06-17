"""Ipso PWA URL patterns."""

from django.urls import path

from apps.ipso import views

urlpatterns = [
    path('', views.index, name='ipso-index'),
    path('index.html', views.index, name='ipso-index-html'),
    path('upload-config.js', views.upload_config_js, name='ipso-upload-config-js'),
    path('reference.json', views.reference_json, name='ipso-reference-json'),
    path('terreni.geojson', views.terreni_geojson, name='ipso-terreni-geojson'),
    path('<path:asset_path>', views.asset, name='ipso-asset'),
]
