from allauth.account.views import (
    ConfirmEmailView,
    PasswordResetView,
    PasswordResetFromKeyView,
    PasswordChangeView,
    PasswordSetView
)
from allauth.account.models import EmailConfirmation, EmailAddress, get_emailconfirmation_model
from django.shortcuts import redirect
from django.conf import settings
from django.http import Http404
from django.views.generic import TemplateView

class CustomConfirmEmailView(ConfirmEmailView):
    template_name = "account/email_confirm.html"

    # Since ACCOUNT_CONFIRM_EMAIL_ON_GET is enabled, this get request will confirm our email
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

class CustomPasswordResetFromKeyView(PasswordResetFromKeyView):
    template_name = 'account/password_reset_from_key.html'
    success_url = '/api/auth/login/'

    def get_success_url(self):
        return getattr(settings, 'PASSWORD_RESET_SUCCESS_URL', self.success_url)

class CustomPasswordSetView(PasswordSetView):
    template_name = 'account/password_set.html'
    success_url = '/api/auth/login/'

class CustomPasswordChangeView(PasswordChangeView):
    template_name = 'account/password_change.html'
    success_url = '/api/auth/login/'

# accounts/views.py

from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from allauth.account.utils import send_email_confirmation

class ResendEmailVerificationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        send_email_confirmation(request, user)
        return Response({"detail": "Verification email sent."})
