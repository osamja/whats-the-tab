"""
URL configuration for musictranscription project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.contrib.auth import views as auth_views
from dj_rest_auth.registration.views import ConfirmEmailView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/accounts/', include('accounts.urls')),
    # Authentication API endpoints
    path('api/auth/', include('dj_rest_auth.urls')),  # Login, logout, password reset
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),  # Signup API
    path('transcribe/', include('transcribeapp.urls')),

    path(
        "api/auth/registration/account-confirm-email/<key>/",
        ConfirmEmailView.as_view(),
        name='account_confirm_email'
    ),
    
    # # Password reset URLs
    path('api/auth/password/reset/confirm/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(),
         name='password_reset_confirm'),
    path('api/auth/password/reset/complete/',
         auth_views.PasswordResetCompleteView.as_view(),
         name='password_reset_complete'),
]

