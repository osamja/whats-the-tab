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
from dj_rest_auth.views import PasswordResetConfirmView
from accounts.views import (
    CustomConfirmEmailView,
    CustomPasswordResetFromKeyView,
    CustomPasswordSetView,
    CustomPasswordChangeView,
    ResendEmailVerificationView
)

urlpatterns = [
    # Email confirmation URL
    path('api/auth/registration/account-confirm-email/<str:key>/', CustomConfirmEmailView.as_view(), name='account_confirm_email'),
    
    path('admin/', admin.site.urls),
    path('api/accounts/', include('accounts.urls')),
    # Authentication API endpoints
    path('api/auth/', include('dj_rest_auth.urls')),  # Login, logout, password reset
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),  # Signup API
    path('transcribe/', include('transcribeapp.urls')),
    
    # Password reset confirm URL - using our custom view
    path('api/auth/password/reset/confirm/<str:uidb64>/<str:token>/',
         CustomPasswordResetFromKeyView.as_view(),
         name='password_reset_confirm'),
         
    # Password management URLs
    path('api/auth/password/reset/key/<uidb36>/<key>/',
         CustomPasswordResetFromKeyView.as_view(),
         name='account_reset_password_from_key'),
    path('api/auth/password/set/',
         CustomPasswordSetView.as_view(),
         name='account_set_password'),
    path('api/auth/password/change/',
         CustomPasswordChangeView.as_view(),
         name='account_change_password'),
    path('api/auth/email/resend-verification/',
         ResendEmailVerificationView.as_view(),
         name='account_resend_verification'),
]

