from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("transcribeapp", "0010_audiomidi_current_chunk_audiomidi_total_chunks"),
    ]

    operations = [
        migrations.AddField(
            model_name="audiomidi",
            name="error_message",
            field=models.TextField(blank=True, default=""),
        ),
    ]
