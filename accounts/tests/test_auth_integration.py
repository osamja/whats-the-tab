from django.test import TestCase, Client
from django.urls import reverse
from django.core import mail
from django.contrib.auth import get_user_model
from allauth.account.models import EmailAddress, EmailConfirmation
from django.conf import settings
import re

User = get_user_model()

class AuthIntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.base_email = "support@pyaar.ai"
        self.test_password = "testpass123!"

    def _get_confirmation_link(self, email_body):
        """Extract confirmation link from email body"""
        match = re.search(r'href="([^"]+)"', email_body)
        return match.group(1) if match else None

    def _get_aliased_email(self, email):
        """Check if email should be aliased to support@pyaar.ai"""
        if "@pyaar.ai" in email and "support_auth_test" in email:
            return self.base_email
        return email

    def _intercept_email(self, to_email):
        """Get the last email sent to the specified address or its alias"""
        target_email = self._get_aliased_email(to_email)
        for email in reversed(mail.outbox):
            if target_email in email.to:
                return email
        return None

    def test_signup_and_email_confirmation(self):
        """Test the complete signup and email confirmation flow"""
        test_email = "support_auth_test_1@pyaar.ai"
        signup_data = {
            "email": test_email,
            "password1": self.test_password,
            "password2": self.test_password,
        }

        # Test signup
        response = self.client.post(reverse("account_signup"), signup_data)
        self.assertEqual(response.status_code, 302)  # Should redirect after signup

        # Verify email was sent
        self.assertEqual(len(mail.outbox), 1)
        email = self._intercept_email(test_email)
        self.assertIsNotNone(email)
        
        # Get confirmation link and verify email
        confirm_link = self._get_confirmation_link(email.body)
        self.assertIsNotNone(confirm_link)
        response = self.client.get(confirm_link)
        self.assertEqual(response.status_code, 200)

        # Verify email is confirmed
        user = User.objects.get(email=test_email)
        email_address = EmailAddress.objects.get(user=user)
        self.assertTrue(email_address.verified)

    def test_login_flow(self):
        """Test login functionality"""
        test_email = "support_auth_test_2@pyaar.ai"
        # Create verified user
        user = User.objects.create_user(email=test_email, password=self.test_password)
        EmailAddress.objects.create(user=user, email=test_email, verified=True, primary=True)

        # Test login
        login_data = {
            "login": test_email,
            "password": self.test_password,
        }
        response = self.client.post(reverse("account_login"), login_data)
        self.assertEqual(response.status_code, 302)  # Should redirect after login
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_password_reset_flow(self):
        """Test forgot password workflow"""
        test_email = "support_auth_test_3@pyaar.ai"
        # Create user
        user = User.objects.create_user(email=test_email, password=self.test_password)
        EmailAddress.objects.create(user=user, email=test_email, verified=True, primary=True)

        # Request password reset
        response = self.client.post(
            reverse("account_reset_password"), {"email": test_email}
        )
        self.assertEqual(response.status_code, 302)

        # Verify reset email was sent
        email = self._intercept_email(test_email)
        self.assertIsNotNone(email)
        
        # Extract reset link and use it
        reset_link = self._get_confirmation_link(email.body)
        self.assertIsNotNone(reset_link)
        
        # Visit reset link
        response = self.client.get(reset_link)
        self.assertEqual(response.status_code, 200)

        # Reset password
        new_password = "newpass123!"
        response = self.client.post(
            reset_link,
            {
                "password1": new_password,
                "password2": new_password,
            },
        )
        self.assertEqual(response.status_code, 302)

        # Verify can login with new password
        login_data = {
            "login": test_email,
            "password": new_password,
        }
        response = self.client.post(reverse("account_login"), login_data)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_resend_verification_email(self):
        """Test resending verification email"""
        test_email = "support_auth_test_4@pyaar.ai"
        # Create unverified user
        user = User.objects.create_user(email=test_email, password=self.test_password)
        EmailAddress.objects.create(user=user, email=test_email, verified=False, primary=True)

        # Login
        self.client.login(email=test_email, password=self.test_password)

        # Request new verification email
        response = self.client.post(reverse("resend_email_verification"))
        self.assertEqual(response.status_code, 200)

        # Verify email was sent
        email = self._intercept_email(test_email)
        self.assertIsNotNone(email) 