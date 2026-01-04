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
    Custom view that redirects to the frontend website for password reset confirmation
    instead of using Django templates.
    """
    def get(self, request, *args, **kwargs):
        # Get the key from the URL
        key = kwargs.get('key')
        uidb36 = kwargs.get('uidb36')
        uidb64 = kwargs.get('uidb64')
        token = kwargs.get('token')
        
        # If we have uidb64 and token, we need to validate them
        if uidb64 and token:
            try:
                # Decode the uidb64 to get the user
                User = get_user_model()
                uid = urlsafe_base64_decode(uidb64).decode()
                user = User.objects.get(pk=uid)
                
                # Check if the token is valid
                if default_token_generator.check_token(user, token):
                    # Token is valid, redirect to frontend with the key and uidb36
                    frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')
                    params = {
                        'key': token,
                        'uidb36': uidb64  # We'll use uidb64 as uidb36 for simplicity
                    }
                    return redirect(f"{frontend_url}/reset-password?{urlencode(params)}")
                else:
                    # Token is invalid
                    frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')
                    return redirect(f"{frontend_url}/reset-password?error=invalid_token")
            except (TypeError, ValueError, OverflowError, User.DoesNotExist):
                # Invalid uidb64
                frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')
                return redirect(f"{frontend_url}/reset-password?error=invalid_user")
        
        # If we have uidb36 and key, validate them
        if uidb36 and key:
            try:
                # This will raise an exception if the key is invalid
                self.reset_user = self.get_user(uidb36, key)
            except:
                # If the key is invalid, redirect to the frontend with an error parameter
                frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')
                return redirect(f"{frontend_url}/reset-password?error=invalid_key")
            
            # If the key is valid, redirect to the frontend with the key and uidb36
            frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')
            params = {
                'key': key,
                'uidb36': uidb36
            }
            return redirect(f"{frontend_url}/reset-password?{urlencode(params)}")
        
        # If we don't have either uidb64/token or uidb36/key, redirect to frontend with error
        frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')
        return redirect(f"{frontend_url}/reset-password?error=missing_parameters")
    
    def post(self, request, *args, **kwargs):
        # Handle the actual password reset
        response = super().post(request, *args, **kwargs)
        
        # If successful, redirect to the frontend with a success parameter
        if response.status_code == 302:  # Redirect status code
            frontend_url = getattr(settings, 'FRONTEND_URL', 'https://pyaar.ai')
            return redirect(f"{frontend_url}/login?reset=success")
        
        return response

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
