from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from transcribeapp.models import AudioMIDI


class Command(BaseCommand):
    help = "Delete completed transcription jobs and their media older than N days"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=7,
            help="Delete jobs older than this many days (default: 7)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show how many jobs would be deleted without deleting them",
        )

    def handle(self, *args, **options):
        days = options["days"]
        if days < 0:
            raise ValueError("--days must be zero or greater")

        cutoff = timezone.now() - timedelta(days=days)
        jobs = AudioMIDI.objects.filter(created_at__lt=cutoff)
        count = jobs.count()

        if options["dry_run"]:
            self.stdout.write(
                f"Would delete {count} job(s) created before {cutoff.isoformat()}"
            )
            return

        jobs.delete()
        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {count} job(s) created before {cutoff.isoformat()}"
            )
        )
