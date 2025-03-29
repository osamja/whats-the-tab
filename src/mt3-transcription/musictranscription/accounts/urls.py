from django.urls import path, include
from .views import ResendEmailVerificationView

urlpatterns = [
    path('', include('allauth.urls')),
    path('resend-verification/', ResendEmailVerificationView.as_view(), name='resend_verification'),
]
