import os

from django.conf import settings
from django.db import migrations


def create_default_site(apps, schema_editor):
    Site = apps.get_model("sites", "Site")
    Site.objects.update_or_create(
        id=settings.SITE_ID,
        defaults={
            "domain": os.getenv("SITE_DOMAIN", "localhost:8008"),
            "name": os.getenv("SITE_NAME", "whats-the-tab"),
        },
    )


class Migration(migrations.Migration):

    dependencies = [
        ("sites", "0002_alter_domain_unique"),
    ]

    operations = [
        migrations.RunPython(create_default_site, migrations.RunPython.noop),
    ]
