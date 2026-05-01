import json
import time

from django.conf import settings
from django.core.management.base import BaseCommand

from transcribeapp.queue import get_redis, get_task_state
from transcribeapp.models import AudioMIDI


class Command(BaseCommand):
    help = "Subscribe to Redis pub/sub for task lifecycle events and update the database"

    def handle(self, *args, **options):
        r = get_redis()

        self._drain_results_backlog(r)

        self._subscribe_live(r)

    def _drain_results_backlog(self, r):
        self.stdout.write("Draining results backlog...")
        count = 0
        while True:
            task_id = r.rpop(settings.TASK_RESULTS_KEY)
            if not task_id:
                break
            tid = task_id.decode() if isinstance(task_id, bytes) else task_id
            self._update_db_from_state(tid)
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
            if message["type"] != "message":
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

    def _update_db_from_state(self, task_id):
        state = get_task_state(task_id)
        if not state:
            self.stderr.write(f"No state found for task {task_id}")
            return

        try:
            audio_midi = AudioMIDI.objects.get(id=task_id)
        except AudioMIDI.DoesNotExist:
            self.stderr.write(f"AudioMIDI {task_id} not found in DB")
            return

        new_status = state.get("status")
        result_status = state.get("result_status")
        if new_status == "claimed" or new_status == "processing":
            audio_midi.status = "processing"
        elif new_status == "completed":
            audio_midi.status = result_status or "completed"
        elif new_status == "failed":
            audio_midi.status = "failed"
            audio_midi.error_message = state.get("error", "Unknown error")
        else:
            self.stderr.write(f"Unknown status for task {task_id}: {new_status}")
            return

        audio_midi.save(update_fields=["status", "error_message"])
        self.stdout.write(f"DB updated: task={task_id} status={audio_midi.status}")

    def _update_progress(self, task_id, progress):
        try:
            audio_midi = AudioMIDI.objects.get(id=task_id)
            audio_midi.current_chunk = progress.get("current", 0)
            audio_midi.total_chunks = progress.get("total", 0)
            audio_midi.save(update_fields=["current_chunk", "total_chunks"])
        except AudioMIDI.DoesNotExist:
            pass
