from allauth.account.views import ConfirmEmailView
from allauth.account.models import EmailConfirmation, EmailAddress, get_emailconfirmation_model
from django.shortcuts import redirect
from django.conf import settings
from django.http import Http404
from django.views.generic import TemplateView

class CustomConfirmEmailView(ConfirmEmailView):
    template_name = "account/email_confirm.html"

    def get_object(self, queryset=None):
        import pdb; pdb.set_trace()
        key = self.kwargs["key"]
        model = get_emailconfirmation_model()
        emailconfirmation = model.from_key(key)
        if not emailconfirmation:
            raise Http404()
        return emailconfirmation

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            self.object = self.get_object()
            self.logout_other_user(self.object)
            if settings.ACCOUNT_CONFIRM_EMAIL_ON_GET:
                self.post(self.request, *self.args, **self.kwargs)
        except Http404:
            self.object = None

        # Determine redirect URL based on verification status
        if self.object and self.object.email_address.verified:
            context['redirect_url'] = f"{settings.FRONTEND_URL}/email-confirmed?status=success"
        else:
            context['redirect_url'] = f"{settings.FRONTEND_URL}/email-confirmed?status=error"
            
        return context

    def get(self, *args, **kwargs):
        try:
            self.object = self.get_object()
            self.logout_other_user(self.object)
            if settings.ACCOUNT_CONFIRM_EMAIL_ON_GET:
                return self.post(*args, **kwargs)
        except Http404:
            self.object = None
            
        return super().get(*args, **kwargs) 

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
