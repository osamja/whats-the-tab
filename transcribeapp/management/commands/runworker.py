import json
import os
import signal
import requests
import tempfile
import threading
import time
import uuid

from django.conf import settings
from django.core.management.base import BaseCommand

from transcribeapp.queue import (
    claim_task,
    get_redis,
    heartbeat,
    mark_completed,
    mark_failed,
    publish_progress,
)
from transcribeapp.tasks import download_youtube_audio, transcribe_audio


class Command(BaseCommand):
    help = "Run a transcription worker that pulls tasks from Redis pub/sub"

    def handle(self, *args, **options):
        self.web_url = settings.WEB_URL.rstrip("/")
        self.worker_id = uuid.uuid4().hex[:8]
        self.current_task_id = None
        self.shutting_down = False

        signal.signal(signal.SIGTERM, self._on_shutdown)
        signal.signal(signal.SIGINT, self._on_shutdown)

        r = get_redis()
        self._start_heartbeat()

        pubsub = r.pubsub()
        pubsub.subscribe(settings.TASK_NEW_CHANNEL)

        self.stdout.write(
            self.style.SUCCESS(
                f"Worker {self.worker_id} listening on '{settings.TASK_NEW_CHANNEL}'..."
            )
        )

        for message in pubsub.listen():
            if self.shutting_down:
                self._flush_processing()
                break

            if message["type"] != "message":
                continue

            if message.get("channel") != settings.TASK_NEW_CHANNEL.encode():
                continue

            task_id = (
                message["data"].decode()
                if isinstance(message["data"], bytes)
                else message["data"]
            )

            state = claim_task()
            if not state:
                continue

            self.current_task_id = task_id

            self.stdout.write(
                f"[{self.worker_id}] Claimed task {task_id} ({state['payload']['type']})"
            )

            try:
                self._process_task(task_id, state)
            except Exception as e:
                self.stderr.write(f"[{self.worker_id}] Task {task_id} failed: {e}")
                mark_failed(task_id, str(e))

            self.current_task_id = None

        self.stdout.write(self.style.WARNING(f"Worker {self.worker_id} shut down"))

    def _on_shutdown(self, signum, frame):
        self.shutting_down = True
        self.stderr.write(
            f"[{self.worker_id}] Received signal {signum}, shutting down gracefully..."
        )

    def _flush_processing(self):
        if self.current_task_id:
            self.stderr.write(
                f"[{self.worker_id}] Flushing current task {self.current_task_id} to failed"
            )
            mark_failed(
                self.current_task_id,
                "worker shutdown mid-processing",
            )

    def _start_heartbeat(self):
        def beat():
            while True:
                time.sleep(10)
                heartbeat(self.worker_id, self.current_task_id)

        t = threading.Thread(target=beat, daemon=True)
        t.start()

    def _process_task(self, task_id, state):
        task_type = state["payload"]["type"]
        result_status = "completed"

        with tempfile.TemporaryDirectory() as tmpdir:
            if task_type == "youtube_download":
                output_path = download_youtube_audio(
                    state["payload"]["youtube_url"], tmpdir
                )
                result_status = "youtube_audio_downloaded"
                self._upload_result(task_id, result_status, output_path)

            elif task_type == "transcribe":
                audio_path = self._download_audio(
                    state["payload"]["audio_url"], tmpdir
                )

                def progress_callback(current, total):
                    publish_progress(task_id, current, total)

                midi_path = transcribe_audio(
                    audio_path, tmpdir, progress_callback=progress_callback
                )
                self._upload_result(task_id, result_status, midi_path)

            else:
                raise ValueError(f"Unknown task type: {task_type}")

        mark_completed(task_id, result_status)

    def _download_audio(self, audio_url, tmpdir):
        filename = os.path.basename(audio_url)
        local_path = os.path.join(tmpdir, filename)

        self.stdout.write(f"Downloading audio: {audio_url}")
        resp = requests.get(audio_url, stream=True, timeout=300)
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_path

    def _upload_result(self, task_id, status, file_path):
        url = f"{self.web_url}/transcribe/_result/"
        data = {"audio_midi_id": task_id, "task_type": status}

        self.stdout.write(f"[{self.worker_id}] Uploading result to {url} ({status})")

        with open(file_path, "rb") as f:
            resp = requests.post(url, data=data, files={"result_file": f}, timeout=60)
            resp.raise_for_status()

        self.stdout.write(
            f"[{self.worker_id}] Result uploaded: task={task_id} HTTP {resp.status_code}"
        )
