"""Base URL patterns — auth + shell."""

from django.urls import path

from apps.base import views

urlpatterns = [
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
]
