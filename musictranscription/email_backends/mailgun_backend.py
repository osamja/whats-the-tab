from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend
from django.core.mail.message import sanitize_address
import requests

class MailgunEmailBackend(BaseEmailBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = settings.MAILGUN_API_KEY
        self.domain = settings.MAILGUN_DOMAIN
        self.api_url = f"https://api.mailgun.net/v3/{self.domain}/messages"

    def send_messages(self, email_messages):
        """
        Send one or more EmailMessage objects and return the number of email
        messages sent.
        """
        if not email_messages:
            return 0

        num_sent = 0
        for email_message in email_messages:
            sent = self._send_message(email_message)
            if sent:
                num_sent += 1
        return num_sent

    def _send_message(self, email_message):
        """
        Send an EmailMessage using the Mailgun API.
        """
        if not email_message.recipients():
            return False

        from_email = sanitize_address(email_message.from_email, email_message.encoding)
        recipients = [sanitize_address(addr, email_message.encoding) for addr in email_message.recipients()]

        try:
            data = {
                "from": from_email,
                "to": recipients,
                "subject": email_message.subject,
                "text": email_message.body if email_message.content_subtype == "plain" else None,
                "html": email_message.body if email_message.content_subtype == "html" else None,
            }

            # Add reply-to if present
            if email_message.reply_to:
                data["h:Reply-To"] = ", ".join(sanitize_address(addr, email_message.encoding) for addr in email_message.reply_to)

            # Add CC if present
            if email_message.cc:
                data["cc"] = [sanitize_address(addr, email_message.encoding) for addr in email_message.cc]

            # Add BCC if present
            if email_message.bcc:
                data["bcc"] = [sanitize_address(addr, email_message.encoding) for addr in email_message.bcc]

            # Send the email using Mailgun API
            response = requests.post(
                self.api_url,
                auth=("api", self.api_key),
                data=data
            )
            
            if response.status_code == 200:
                return True
            else:
                if not self.fail_silently:
                    raise Exception(f"Mailgun API error: {response.text} {self.api_key} {self.domain}")
                return False

        except Exception as e:
            if not self.fail_silently:
                raise
            return False 