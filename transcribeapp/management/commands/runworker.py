import os
import signal
import requests
import tempfile
import threading
import time
import uuid
from urllib.parse import urlsplit

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from transcribeapp.queue import (
    claim_task,
    heartbeat,
    mark_completed,
    mark_failed,
    publish_progress,
    reap_stale_tasks,
)
from transcribeapp.tasks import download_youtube_audio, transcribe_audio
from transcribeapp.vast_client import check_vast_configuration, transcribe_via_vast


class Command(BaseCommand):
    help = "Run a transcription worker that pulls tasks from the Redis queue"

    def handle(self, *args, **options):
        if settings.VAST_ENDPOINT_NAME:
            try:
                check_vast_configuration()
            except RuntimeError as exc:
                raise CommandError(str(exc)) from None
        self.web_url = settings.WEB_URL.rstrip("/")
        self.worker_id = uuid.uuid4().hex[:8]
        self.current_task_id = None
        self.shutting_down = False

        signal.signal(signal.SIGTERM, self._on_shutdown)
        signal.signal(signal.SIGINT, self._on_shutdown)

        self._start_heartbeat()

        self.stdout.write(
            self.style.SUCCESS(
                f"Worker {self.worker_id} polling '{settings.TASK_QUEUE_KEY}'..."
            )
        )

        last_reap = 0.0
        while not self.shutting_down:
            now = time.time()
            if now - last_reap >= settings.TASK_REAP_INTERVAL:
                try:
                    reaped = reap_stale_tasks()
                    if reaped:
                        self.stdout.write(
                            f"[{self.worker_id}] Reaped {reaped} stale task(s)"
                        )
                except Exception as e:
                    self.stderr.write(
                        f"[{self.worker_id}] Failed to reap stale tasks: {e}"
                    )
                last_reap = now

            try:
                claimed = claim_task(self.worker_id)
            except Exception as e:
                self.stderr.write(f"[{self.worker_id}] Failed to claim task: {e}")
                time.sleep(1)
                continue

            if not claimed:
                continue

            task_id, state = claimed
            if not state:
                continue

            self.current_task_id = task_id
            task_type = state.get("payload", {}).get("type", "unknown")

            self.stdout.write(
                f"[{self.worker_id}] Claimed task {task_id} ({task_type})"
            )

            try:
                self._process_task(task_id, state)
            except Exception as e:
                self.stderr.write(f"[{self.worker_id}] Task {task_id} failed: {e}")
                mark_failed(task_id, str(e))

            self.current_task_id = None

        if self.current_task_id:
            self._flush_processing()

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
                try:
                    heartbeat(self.worker_id, self.current_task_id)
                except Exception as e:
                    self.stderr.write(
                        f"[{self.worker_id}] Heartbeat failed: {e}"
                    )

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
                use_vast = bool(settings.VAST_ENDPOINT_NAME)
                if use_vast:
                    publish_progress(task_id, 0, 1)
                    midi_path = transcribe_via_vast(task_id, tmpdir)
                else:
                    audio_path = self._download_audio(
                        state["payload"]["audio_url"], tmpdir
                    )

                    def progress_callback(current, total):
                        publish_progress(task_id, current, total)

                    midi_path = transcribe_audio(
                        audio_path, tmpdir, progress_callback=progress_callback
                    )
                self._upload_result(task_id, result_status, midi_path)
                if use_vast:
                    publish_progress(task_id, 1, 1)

            else:
                raise ValueError(f"Unknown task type: {task_type}")

        mark_completed(task_id, result_status)

    def _download_audio(self, audio_url, tmpdir):
        filename = os.path.basename(urlsplit(audio_url).path) or "audio"
        local_path = os.path.join(tmpdir, filename)

        self.stdout.write("Downloading audio")
        try:
            with requests.get(audio_url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(local_path, "wb") as audio:
                    for chunk in response.iter_content(chunk_size=8192):
                        audio.write(chunk)
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Audio download returned HTTP {exc.response.status_code}"
            ) from None
        except requests.RequestException:
            raise RuntimeError("Audio download failed") from None

        return local_path

    def _upload_result(self, task_id, status, file_path):
        url = f"{self.web_url}/transcribe/_result/"
        data = {"audio_midi_id": task_id, "task_type": status}
        headers = {"X-Worker-Token": settings.WORKER_SHARED_SECRET}

        self.stdout.write(f"[{self.worker_id}] Uploading result to {url} ({status})")

        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                data=data,
                files={"result_file": f},
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()

        self.stdout.write(
            f"[{self.worker_id}] Result uploaded: task={task_id} HTTP {resp.status_code}"
        )
