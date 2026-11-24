import asyncio
import base64
import os
import shutil
import sys
import tempfile
import time
import uuid
import wave
from datetime import timedelta
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import redis
import requests
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import RequestFactory, TestCase, override_settings
from django.urls import Resolver404, resolve, reverse
from django.utils import timezone

from transcribeapp.models import AudioMIDI
from transcribeapp.management.commands.runworker import Command as WorkerCommand
from transcribeapp.queue import (
    claim_task,
    enqueue_task,
    get_task_state,
    heartbeat,
    mark_completed,
    reap_stale_tasks,
    retry_task,
)
from transcribeapp.ratelimit import check_rate_limit
from transcribeapp.tasks import InvalidAudioError
from transcribeapp.vast_client import RetryableTranscriptionError, _request_transcription



@override_settings(VAST_ENDPOINT_NAME="test-endpoint", VAST_API_KEY="test-key")
class WorkerDispatchTests(TestCase):
    @patch("transcribeapp.management.commands.runworker.mark_completed")
    @patch("transcribeapp.management.commands.runworker.publish_progress")
    @patch("django.db.models.fields.files.FieldFile.url", new_callable=PropertyMock)
    def test_dispatcher_signs_fresh_url_and_uploads_midi(
        self, signed_url, progress, completed
    ):
        from mido import MidiFile, MidiTrack

        midi = MidiFile()
        midi.tracks.append(MidiTrack())
        buffer = BytesIO()
        midi.save(file=buffer)
        midi_bytes = buffer.getvalue()
        audio = AudioMIDI.objects.create(
            audio_filename="test.wav", audio_file="audios/test.wav"
        )
        task_id = str(audio.pk)
        signed_url.return_value = "https://media.example.test/test.wav?fresh=1"
        endpoint = SimpleNamespace(
            request=AsyncMock(return_value={
                "status": 200,
                "response": {"result": {"ok": base64.b64encode(midi_bytes).decode()}},
            })
        )
        serverless = MagicMock()
        serverless.__aenter__.return_value.get_endpoint = AsyncMock(return_value=endpoint)
        command = WorkerCommand()

        def upload(task_id, status, path):
            self.assertEqual(path.read_bytes(), midi_bytes)

        with (
            patch.dict(
                sys.modules, {"vastai": SimpleNamespace(Serverless=lambda **_: serverless)}
            ),
            patch.object(command, "_upload_result", side_effect=upload) as callback,
        ):
            command._process_task(task_id, {
                "payload": {"type": "transcribe", "audio_url": "expired"},
            })

        endpoint.request.assert_awaited_once_with(
            "/remote/transcribe",
            {"kwargs": {"audio_url": signed_url.return_value}},
            retry=False,
        )
        self.assertEqual(callback.call_args.args[:2], (task_id, "completed"))
        self.assertEqual([call.args for call in progress.call_args_list], [
            (task_id, 0, 1), (task_id, 1, 1),
        ])
        completed.assert_called_once_with(task_id, "completed")

    @patch("transcribeapp.management.commands.runworker.mark_completed")
    @patch("transcribeapp.management.commands.runworker.publish_progress")
    @patch("transcribeapp.vast_client._request_transcription", return_value="bm90IG1pZGk=")
    @patch("django.db.models.fields.files.FieldFile.url", new_callable=PropertyMock)
    def test_invalid_midi_never_reaches_callback(
        self, signed_url, _request, progress, completed
    ):
        signed_url.return_value = "https://media.example.test/test.wav?fresh=1"
        audio = AudioMIDI.objects.create(
            audio_filename="test.wav", audio_file="audios/test.wav"
        )
        command = WorkerCommand()
        with patch.object(command, "_upload_result") as callback:
            with self.assertRaisesMessage(RuntimeError, "Vast returned invalid MIDI"):
                command._process_task(str(audio.pk), {"payload": {"type": "transcribe"}})
        callback.assert_not_called()
        completed.assert_not_called()
        progress.assert_called_once_with(str(audio.pk), 0, 1)

    @override_settings(VAST_ENDPOINT_NAME="")
    @patch("transcribeapp.management.commands.runworker.mark_completed")
    @patch("transcribeapp.management.commands.runworker.transcribe_audio")
    def test_local_worker_uses_queued_audio_url(self, transcribe, completed):
        command = WorkerCommand()
        with (
            patch.object(command, "_download_audio", return_value="download.wav") as download,
            patch.object(command, "_upload_result") as callback,
        ):
            command._process_task("42", {
                "payload": {"type": "transcribe", "audio_url": "http://web/audio.wav"},
            })
        self.assertEqual(download.call_args.args[0], "http://web/audio.wav")
        self.assertEqual(transcribe.call_args.args[0], "download.wav")
        callback.assert_called_once_with("42", "completed", transcribe.return_value)
        completed.assert_called_once_with("42", "completed")

    @patch("transcribeapp.management.commands.runworker.requests.get")
    def test_download_failure_keeps_signed_url_out_of_worker_errors(self, get):
        audio_url = "https://media.example.test/test.wav?signature=private"
        response = requests.Response()
        response.status_code = 403
        response.url = audio_url
        get.return_value.__enter__.return_value = response

        command = WorkerCommand(stdout=StringIO())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesMessage(RuntimeError, "Audio download returned HTTP 403"):
                command._download_audio(audio_url, directory)
        self.assertNotIn("signature", command.stdout.getvalue())

    def test_vast_input_error_is_permanent_and_unavailability_is_retryable(self):
        endpoint = SimpleNamespace(request=AsyncMock())
        serverless = MagicMock()
        serverless.__aenter__.return_value.get_endpoint = AsyncMock(return_value=endpoint)
        with patch.dict(sys.modules, {"vastai": SimpleNamespace(Serverless=lambda **_: serverless)}):
            endpoint.request.return_value = {
                "status": 200,
                "response": {
                    "result": {
                        "ok": {
                            "type": "dict",
                            "contents": [["invalid_audio", "Audio is 601.0s (max 600s)."]],
                        }
                    }
                },
            }
            with self.assertRaisesMessage(InvalidAudioError, "Audio is 601.0s (max 600s)."):
                asyncio.run(_request_transcription("https://example.test/audio"))
            endpoint.request.return_value = {
                "status": 200,
                "response": {"result": {"ok": {"invalid_audio": "Audio exceeds the limit."}}},
            }
            with self.assertRaisesMessage(InvalidAudioError, "Audio exceeds the limit."):
                asyncio.run(_request_transcription("https://example.test/audio"))
            endpoint.request.return_value = {"status": 503, "response": None}
            with self.assertRaises(RetryableTranscriptionError):
                asyncio.run(_request_transcription("https://example.test/audio"))

    @patch("transcribeapp.management.commands.runworker.requests.post", side_effect=requests.Timeout)
    def test_callback_timeout_checks_if_midi_was_already_saved(self, post):
        job = AudioMIDI.objects.create(
            audio_filename="test.wav", midi_file="midi/result.midi",
            status=AudioMIDI.Status.COMPLETED,
        )
        command = WorkerCommand(stdout=StringIO())
        command.web_url = "http://web:8008"
        command.worker_id = "dispatcher"
        with tempfile.NamedTemporaryFile() as midi:
            command._upload_result(str(job.pk), "completed", midi.name)

        post.assert_called_once()

    @patch("transcribeapp.management.commands.runworker.mark_failed")
    @patch("transcribeapp.management.commands.runworker.claim_task")
    @patch("transcribeapp.management.commands.runworker.heartbeat")
    @patch("transcribeapp.management.commands.runworker.reap_stale_tasks", return_value=0)
    @patch(
        "transcribeapp.management.commands.runworker.download_youtube_audio",
        side_effect=RuntimeError("ERROR: [youtube] Sign in to confirm you're not a bot"),
    )
    def test_youtube_download_error_is_sanitized(
        self, _download, _reap, _heartbeat, claim, mark_failed
    ):
        command = WorkerCommand(stdout=StringIO(), stderr=StringIO())

        def claim_once(_worker_id):
            command.shutting_down = True
            return (
                "77",
                {"payload": {"type": "youtube_download", "youtube_url": "https://youtu.be/abc"}},
            )

        claim.side_effect = claim_once
        with patch.object(command, "_start_heartbeat"), patch("signal.signal"):
            with self.settings(VAST_ENDPOINT_NAME=""):
                command.handle()

        mark_failed.assert_called_once_with(
            "77", "YouTube audio download could not be completed."
        )


class MediaRootTests(TestCase):
    def test_media_root_is_scoped_to_media_directory(self):
        self.assertEqual(settings.MEDIA_ROOT, Path(settings.BASE_DIR) / "media")
        self.assertNotEqual(settings.MEDIA_ROOT, Path(settings.BASE_DIR))

    def test_media_route_is_not_registered_when_not_serving_media(self):
        if settings.DEBUG or settings.SERVE_MEDIA:
            self.skipTest("Media route is intentionally enabled in this environment")
        with self.assertRaises(Resolver404):
            resolve("/media/audios/example.wav")


class UploadApiTests(TestCase):
    def setUp(self):
        self.rate_limit_patcher = patch(
            "transcribeapp.views.check_rate_limit",
            return_value=(True, 1),
        )
        self.rate_limit_patcher.start()
        self.addCleanup(self.rate_limit_patcher.stop)

    @override_settings(MAX_UPLOAD_BYTES=10)
    def test_upload_rejects_files_over_size_limit(self):
        response = self.client.post(
            reverse("upload_audio"),
            {
                "audio": SimpleUploadedFile(
                    "too-large.wav",
                    b"x" * 11,
                    content_type="audio/wav",
                )
            },
        )

        self.assertEqual(response.status_code, 413)

    def test_upload_rejects_unsupported_content_type(self):
        response = self.client.post(
            reverse("upload_audio"),
            {
                "audio": SimpleUploadedFile(
                    "not-audio.txt",
                    b"hello",
                    content_type="text/plain",
                )
            },
        )

        self.assertEqual(response.status_code, 415)

    def test_upload_rejects_invalid_mp3_before_saving(self):
        response = self.client.post(
            reverse("upload_audio"),
            {"audio": SimpleUploadedFile("bad.mp3", b"not audio", "audio/mpeg")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("readable audio", response.json()["message"])
        self.assertFalse(AudioMIDI.objects.exists())

    def test_upload_keeps_all_bytes_after_validation(self):
        audio = BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * 800)
        data = audio.getvalue()
        with tempfile.TemporaryDirectory() as media_root, self.settings(MEDIA_ROOT=media_root):
            response = self.client.post(
                reverse("upload_audio"),
                {"audio": SimpleUploadedFile("valid.wav", data, "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            job = AudioMIDI.objects.get(pk=response.json()["audio_midi_id"])
            with job.audio_file.open("rb") as stored:
                self.assertEqual(stored.read(), data)

    @override_settings(MAX_AUDIO_DURATION_SECONDS=1)
    def test_upload_rejects_audio_over_duration_limit_before_saving(self):
        audio = BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * 16000)
        response = self.client.post(
            reverse("upload_audio"),
            {"audio": SimpleUploadedFile("too-long.wav", audio.getvalue(), "audio/wav")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("maximum supported length", response.json()["message"])
        self.assertFalse(AudioMIDI.objects.exists())

    def test_upload_from_youtube_is_disabled_by_default(self):
        response = self.client.post(
            reverse("upload_from_youtube"),
            {"youtube_url": "https://www.youtube.com/watch?v=example"},
        )

        self.assertEqual(response.status_code, 410)
        self.assertIn("disabled", response.json()["message"])
        self.assertFalse(AudioMIDI.objects.exists())

    @override_settings(ENABLE_YOUTUBE_UPLOAD=True)
    @patch("transcribeapp.views.enqueue_task")
    def test_upload_from_youtube_enqueues_when_enabled(self, enqueue):
        response = self.client.post(
            reverse("upload_from_youtube"),
            {"youtube_url": "https://www.youtube.com/watch?v=example"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(AudioMIDI.objects.count(), 1)
        enqueue.assert_called_once()


class TranscribeApiTests(TestCase):
    def setUp(self):
        self.rate_limit_patcher = patch(
            "transcribeapp.views.check_rate_limit",
            return_value=(True, 1),
        )
        self.rate_limit_patcher.start()
        self.addCleanup(self.rate_limit_patcher.stop)

    def test_transcribe_rejects_job_without_audio(self):
        audio_midi = AudioMIDI.objects.create(
            audio_filename="pending-youtube.mp4",
            youtube_url="https://www.youtube.com/watch?v=example",
            status=AudioMIDI.Status.QUEUED,
        )

        response = self.client.post(
            reverse("transcribe"),
            {"audio_midi_id": audio_midi.id},
        )

        self.assertEqual(response.status_code, 409)

    def test_status_field_accepts_youtube_downloaded_status(self):
        field = AudioMIDI._meta.get_field("status")
        self.assertGreaterEqual(field.max_length, len("youtube_audio_downloaded"))


class MidiPreviewTests(TestCase):
    @patch("django.db.models.fields.files.FieldFile.open")
    @patch("django.db.models.fields.files.FieldFile.url", new_callable=PropertyMock)
    def test_preview_serves_private_object_on_the_application_origin(self, signed_url, open_file):
        signed_url.return_value = "https://media.example.test/preview.midi?signed=1"
        midi_bytes = b"MThd"
        open_file.return_value = BytesIO(midi_bytes)
        job = AudioMIDI.objects.create(
            audio_filename="test.wav",
            audio_file="audios/test.wav",
            midi_file="midi/test.midi",
            status=AudioMIDI.Status.COMPLETED,
        )

        response = self.client.get(reverse("get_midi", args=[job.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "audio/midi")
        self.assertEqual(b"".join(response.streaming_content), midi_bytes)
        signed_url.assert_not_called()


class PruneJobsTests(TestCase):
    def test_prune_jobs_dry_run_and_delete(self):
        job = AudioMIDI.objects.create(audio_filename="old.wav")
        AudioMIDI.objects.filter(pk=job.pk).update(
            created_at=timezone.now() - timedelta(days=10)
        )
        output = StringIO()

        call_command("prune_jobs", "--days", "7", "--dry-run", stdout=output)
        self.assertTrue(AudioMIDI.objects.filter(pk=job.pk).exists())
        self.assertIn("Would delete 1 job", output.getvalue())

        call_command("prune_jobs", "--days", "7", stdout=output)
        self.assertFalse(AudioMIDI.objects.filter(pk=job.pk).exists())


class ReadinessTests(TestCase):
    @patch("transcribeapp.views.get_redis")
    def test_readiness_reports_ok_when_dependencies_are_healthy(self, get_redis):
        redis_client = get_redis.return_value
        redis_client.ping.return_value = True
        redis_client.scan_iter.return_value = [b"worker:abc123"]

        response = self.client.get(reverse("readiness"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["checks"]["workers_online"], 1)

    @patch("transcribeapp.views.get_redis")
    def test_readiness_reports_degraded_when_redis_is_down(self, get_redis):
        get_redis.side_effect = redis.RedisError("connection refused")

        response = self.client.get(reverse("readiness"))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertIn("error", response.json()["checks"]["redis"])


@override_settings(TRANSCRIBE_API_KEY="test-api-key")
class ApiKeyTests(TestCase):
    def setUp(self):
        self.job = AudioMIDI.objects.create(audio_filename="test.wav")

    def test_client_endpoint_rejects_missing_api_key(self):
        response = self.client.get(
            reverse("audio_status", args=[self.job.id])
        )

        self.assertEqual(response.status_code, 401)

    def test_client_endpoint_accepts_x_api_key(self):
        response = self.client.get(
            reverse("audio_status", args=[self.job.id]),
            HTTP_X_API_KEY="test-api-key",
        )

        self.assertEqual(response.status_code, 200)

    def test_client_endpoint_accepts_bearer_token(self):
        response = self.client.get(
            reverse("audio_status", args=[self.job.id]),
            HTTP_AUTHORIZATION="Bearer test-api-key",
        )

        self.assertEqual(response.status_code, 200)


@override_settings(WORKER_SHARED_SECRET="test-worker-secret")
class ResultCallbackTests(TestCase):
    def setUp(self):
        self.media_root = tempfile.mkdtemp()
        self.settings_ctx = self.settings(MEDIA_ROOT=self.media_root)
        self.settings_ctx.enable()
        self.addCleanup(self.settings_ctx.disable)
        self.addCleanup(shutil.rmtree, self.media_root, True)

    def _post_result(self, audio_midi, task_type, token=None):
        headers = {}
        if token is not None:
            headers["HTTP_X_WORKER_TOKEN"] = token
        return self.client.post(
            reverse("result"),
            {
                "audio_midi_id": audio_midi.id,
                "task_type": task_type,
                "result_file": SimpleUploadedFile(
                    "result.bin",
                    b"test-result",
                    content_type="application/octet-stream",
                ),
            },
            **headers,
        )

    def test_result_rejects_missing_worker_token(self):
        audio_midi = AudioMIDI.objects.create(audio_filename="test.wav")

        response = self._post_result(audio_midi, "completed")

        self.assertEqual(response.status_code, 403)

    def test_result_rejects_invalid_task_type(self):
        audio_midi = AudioMIDI.objects.create(audio_filename="test.wav")

        response = self._post_result(
            audio_midi,
            "arbitrary-status",
            token="test-worker-secret",
        )

        self.assertEqual(response.status_code, 400)

    def test_result_accepts_completed_midi(self):
        audio_midi = AudioMIDI.objects.create(audio_filename="test.wav")

        response = self._post_result(
            audio_midi,
            AudioMIDI.Status.COMPLETED,
            token="test-worker-secret",
        )

        self.assertEqual(response.status_code, 200)
        audio_midi.refresh_from_db()
        self.assertEqual(audio_midi.status, AudioMIDI.Status.COMPLETED)
        self.assertTrue(audio_midi.midi_file)

    def test_result_marks_all_chunks_complete(self):
        audio_midi = AudioMIDI.objects.create(
            audio_filename="test.wav",
            total_chunks=3,
            current_chunk=1,
        )

        response = self._post_result(
            audio_midi,
            AudioMIDI.Status.COMPLETED,
            token="test-worker-secret",
        )

        self.assertEqual(response.status_code, 200)
        audio_midi.refresh_from_db()
        self.assertEqual(audio_midi.current_chunk, 3)

    def test_result_accepts_youtube_downloaded_audio(self):
        audio_midi = AudioMIDI.objects.create(
            audio_filename="youtube.mp4",
            youtube_url="https://www.youtube.com/watch?v=example",
            status=AudioMIDI.Status.QUEUED,
        )

        response = self._post_result(
            audio_midi,
            AudioMIDI.Status.YOUTUBE_AUDIO_DOWNLOADED,
            token="test-worker-secret",
        )

        self.assertEqual(response.status_code, 200)
        audio_midi.refresh_from_db()
        self.assertEqual(
            audio_midi.status,
            AudioMIDI.Status.YOUTUBE_AUDIO_DOWNLOADED,
        )
        self.assertTrue(audio_midi.audio_file)


class RateLimitTests(TestCase):
    redis_url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/14")

    def setUp(self):
        try:
            self.redis = redis.from_url(self.redis_url)
            self.redis.ping()
        except redis.RedisError:
            self.skipTest(f"Redis is not available at {self.redis_url}")

        self.redis.flushdb()
        self.settings_ctx = self.settings(REDIS_URL=self.redis_url)
        self.settings_ctx.enable()
        self.addCleanup(self.settings_ctx.disable)

    def test_rate_limit_allows_then_blocks(self):
        request = RequestFactory().get("/", REMOTE_ADDR="203.0.113.10")

        allowed, count = check_rate_limit(request, "test", limit=1)
        self.assertTrue(allowed)
        self.assertEqual(count, 1)

        allowed, count = check_rate_limit(request, "test", limit=1)
        self.assertFalse(allowed)
        self.assertEqual(count, 2)

    def test_rate_limit_is_per_ip(self):
        first = RequestFactory().get("/", REMOTE_ADDR="203.0.113.10")
        second = RequestFactory().get("/", REMOTE_ADDR="203.0.113.11")

        self.assertTrue(check_rate_limit(first, "test", limit=1)[0])
        self.assertTrue(check_rate_limit(second, "test", limit=1)[0])


class QueueTests(TestCase):
    redis_url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")

    def setUp(self):
        try:
            self.redis = redis.from_url(self.redis_url)
            self.redis.ping()
        except redis.RedisError:
            self.skipTest(f"Redis is not available at {self.redis_url}")

        self.redis.flushdb()
        self.settings_ctx = self.settings(REDIS_URL=self.redis_url)
        self.settings_ctx.enable()
        self.addCleanup(self.settings_ctx.disable)

    def _task_id(self):
        return f"test-{uuid.uuid4().hex}"

    def test_claim_returns_the_claimed_task_id(self):
        task_id = self._task_id()
        enqueue_task({"audio_midi_id": task_id, "type": "transcribe"})

        claimed_id, state = claim_task("test-worker", timeout=1)

        self.assertEqual(claimed_id, task_id)
        self.assertEqual(state["status"], "processing")
        self.assertEqual(state["worker_id"], "test-worker")

    def test_completed_task_gets_a_ttl(self):
        task_id = self._task_id()
        enqueue_task({"audio_midi_id": task_id, "type": "transcribe"})
        claim_task("test-worker", timeout=1)

        mark_completed(task_id)

        self.assertGreater(
            self.redis.ttl(f"{settings.TASK_HASH_PREFIX}{task_id}"),
            0,
        )

    def test_reaper_schedules_a_lost_worker_job_for_another_dispatcher(self):
        task_id = self._task_id()
        enqueue_task({"audio_midi_id": task_id, "type": "transcribe"})
        claim_task("dead-worker", timeout=1)
        now = time.time()
        self.redis.zadd(settings.TASK_PROCESSING_TIME_KEY, {task_id: now - 1000})

        with self.settings(TASK_PROCESSING_TIMEOUT=1):
            self.assertEqual(reap_stale_tasks(), 1)
            self.assertEqual(reap_stale_tasks(), 0)

        state = get_task_state(task_id)
        self.assertEqual(state["status"], "retrying")
        self.assertEqual(state["retries"], 1)
        self.assertGreater(state["retry_at"], now)
        self.assertIsNone(claim_task("replacement", timeout=0.1))

        self.redis.zadd(settings.TASK_RETRY_KEY, {task_id: now - 1})
        claimed_id, state = claim_task("replacement", timeout=1)
        self.assertEqual(claimed_id, task_id)
        self.assertEqual(state["worker_id"], "replacement")
        self.assertEqual(state["retries"], 1)
        self.assertFalse(retry_task(task_id, "dead-worker", "late failure"))

    @override_settings(TASK_MAX_RETRIES=2)
    def test_transient_errors_exhaust_three_attempts(self):
        task_id = self._task_id()
        enqueue_task({"audio_midi_id": task_id, "type": "transcribe"})
        for retries in range(3):
            claimed_id, state = claim_task("dispatcher", timeout=1)
            self.assertEqual(claimed_id, task_id)
            self.assertEqual(state["retries"], retries)
            self.assertTrue(retry_task(task_id, "dispatcher", "Vast temporarily unavailable"))
            if retries < 2:
                self.assertEqual(get_task_state(task_id)["status"], "retrying")
                self.redis.zadd(settings.TASK_RETRY_KEY, {task_id: time.time() - 1})

        state = get_task_state(task_id)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"], "Vast temporarily unavailable")
        self.assertIsNone(self.redis.zscore(settings.TASK_RETRY_KEY, task_id))

    def test_status_exposes_scheduled_retry_for_the_same_job(self):
        job = AudioMIDI.objects.create(
            audio_filename="audio.wav", status=AudioMIDI.Status.PROCESSING
        )
        job_id = str(job.pk)
        enqueue_task({"audio_midi_id": job_id, "type": "transcribe"})
        claim_task("dispatcher", timeout=1)
        retry_task(job_id, "dispatcher", "Temporary connection failure")

        response = self.client.get(reverse("audio_status", args=[job.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["audio_midi_id"], job.pk)
        self.assertEqual(response.json()["status"], "retrying")
        self.assertEqual(response.json()["message"], "Retrying automatically")

    def test_reaper_leaves_live_worker_task_alone(self):
        task_id = self._task_id()
        enqueue_task({"audio_midi_id": task_id, "type": "transcribe"})
        claim_task("live-worker", timeout=1)
        heartbeat("live-worker", task_id)
        self.redis.zadd(
            settings.TASK_PROCESSING_TIME_KEY,
            {task_id: time.time() - 1000},
        )

        with self.settings(TASK_PROCESSING_TIMEOUT=1):
            reaped = reap_stale_tasks()

        self.assertEqual(reaped, 0)
        state = get_task_state(task_id)
        self.assertEqual(state["status"], "processing")
