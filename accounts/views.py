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
from django.urls import reverse
from urllib.parse import urlencode
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_decode
from django.contrib.auth import get_user_model

class CustomConfirmEmailView(ConfirmEmailView):
    template_name = "account/email_confirm.html"

    # Since ACCOUNT_CONFIRM_EMAIL_ON_GET is enabled, this get request will confirm our email
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

class CustomPasswordResetFromKeyView(PasswordResetFromKeyView):
    """
    Handles GET requests from password reset email links.
    Validates the token and redirects to the frontend with uid/token as query params.
    The actual password reset POST is handled by dj-rest-auth's PasswordResetConfirmView
    at /api/auth/password/reset/confirm/.
    """
    def get(self, request, *args, **kwargs):
        key = kwargs.get('key')
        uidb36 = kwargs.get('uidb36')
        uidb64 = kwargs.get('uidb64')
        token = kwargs.get('token')
        frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')

        if uidb64 and token:
            try:
                User = get_user_model()
                uid = urlsafe_base64_decode(uidb64).decode()
                user = User.objects.get(pk=uid)

                if default_token_generator.check_token(user, token):
                    params = {'key': token, 'uidb36': uidb64}
                    return redirect(f"{frontend_url}/reset-password?{urlencode(params)}")
                else:
                    return redirect(f"{frontend_url}/reset-password?error=invalid_token")
            except (TypeError, ValueError, OverflowError, User.DoesNotExist):
                return redirect(f"{frontend_url}/reset-password?error=invalid_user")

        if uidb36 and key:
            try:
                self.reset_user = self.get_user(uidb36, key)
            except Exception:
                return redirect(f"{frontend_url}/reset-password?error=invalid_key")

            params = {'key': key, 'uidb36': uidb36}
            return redirect(f"{frontend_url}/reset-password?{urlencode(params)}")

        return redirect(f"{frontend_url}/reset-password?error=missing_parameters")

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
from allauth.account.models import EmailAddress

class ResendEmailVerificationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        # Use the new API for sending email confirmation
        email_address = EmailAddress.objects.get_primary(user)
        if email_address:
            email_address.send_confirmation(request)
        return Response({"detail": "Verification email sent."})
