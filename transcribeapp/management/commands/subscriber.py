import json

from django.conf import settings
from django.core.management.base import BaseCommand

from transcribeapp.queue import get_redis, get_task_state
from transcribeapp.models import AudioMIDI


class Command(BaseCommand):
    help = "Subscribe to Redis pub/sub for task lifecycle events and update the database"

    def handle(self, *args, **options):
        r = get_redis()

        self._subscribe_live(r)

    def _drain_results_backlog(self, r):
        self.stdout.write("Draining results backlog...")
        count = 0
        while True:
            task_id = r.lindex(settings.TASK_RESULTS_KEY, 0)
            if not task_id:
                break
            tid = task_id.decode() if isinstance(task_id, bytes) else task_id
            self._update_db_from_state(tid)
            r.lrem(settings.TASK_RESULTS_KEY, 1, task_id)
            count += 1
        self.stdout.write(f"Caught up on {count} results")

    def _subscribe_live(self, r):
        pubsub = r.pubsub()
        pubsub.subscribe(
            settings.TASK_CLAIMED_CHANNEL,
            settings.TASK_COMPLETED_CHANNEL,
            settings.TASK_FAILED_CHANNEL,
        )
        pubsub.psubscribe(f"{settings.TASK_PROGRESS_PREFIX}*")
        self._drain_results_backlog(r)

        self.stdout.write(
            self.style.SUCCESS(
                "Subscriber listening on: "
                f"{settings.TASK_CLAIMED_CHANNEL}, "
                f"{settings.TASK_COMPLETED_CHANNEL}, "
                f"{settings.TASK_FAILED_CHANNEL}, "
                f"{settings.TASK_PROGRESS_PREFIX}*"
            )
        )

        for message in pubsub.listen():
            # Pattern subscriptions arrive as "pmessage"; regular channel
            # subscriptions arrive as "message".
            if message["type"] not in ("message", "pmessage"):
                continue

            channel = (
                message["channel"].decode()
                if isinstance(message["channel"], bytes)
                else message["channel"]
            )
            data = (
                message["data"].decode()
                if isinstance(message["data"], bytes)
                else message["data"]
            )

            if channel.startswith(settings.TASK_PROGRESS_PREFIX):
                task_id = channel[len(settings.TASK_PROGRESS_PREFIX) :]
                progress = json.loads(data)
                self._update_progress(task_id, progress)
            elif channel in (
                settings.TASK_CLAIMED_CHANNEL,
                settings.TASK_COMPLETED_CHANNEL,
                settings.TASK_FAILED_CHANNEL,
            ):
                self._update_db_from_state(data)
                if channel in (settings.TASK_COMPLETED_CHANNEL, settings.TASK_FAILED_CHANNEL):
                    r.lrem(settings.TASK_RESULTS_KEY, 0, data)

    def _update_db_from_state(self, task_id):
        state = get_task_state(task_id)
        if not state:
            self.stderr.write(f"No state found for task {task_id}")
            return

        status = state.get("status")
        error = ""
        if status in ("claimed", "processing"):
            status = AudioMIDI.Status.PROCESSING
        elif status == "completed":
            status = state.get("result_status") or AudioMIDI.Status.COMPLETED
        elif status == "failed":
            error = state.get("error", "Unknown error")
        else:
            return

        # A completed callback takes precedence over a late queue event.
        AudioMIDI.objects.filter(pk=task_id).exclude(
            status=AudioMIDI.Status.COMPLETED
        ).update(status=status, error_message=error)
        self.stdout.write(f"DB updated: task={task_id} status={status}")

    def _update_progress(self, task_id, progress):
        AudioMIDI.objects.filter(pk=task_id).exclude(
            status=AudioMIDI.Status.COMPLETED
        ).update(current_chunk=progress.get("current", 0), total_chunks=progress.get("total", 0))
