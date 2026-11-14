import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import redis
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory, TestCase, override_settings
from django.urls import Resolver404, resolve, reverse

from transcribeapp.models import AudioMIDI
from transcribeapp.queue import (
    claim_task,
    enqueue_task,
    get_task_state,
    heartbeat,
    mark_completed,
    reap_stale_tasks,
)
from transcribeapp.ratelimit import check_rate_limit


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

    def test_reaper_requeues_task_without_heartbeat(self):
        task_id = self._task_id()
        enqueue_task({"audio_midi_id": task_id, "type": "transcribe"})
        claim_task("dead-worker", timeout=1)
        self.redis.zadd(
            settings.TASK_PROCESSING_TIME_KEY,
            {task_id: time.time() - 1000},
        )

        with self.settings(TASK_PROCESSING_TIMEOUT=1, TASK_MAX_RETRIES=3):
            reaped = reap_stale_tasks()

        self.assertEqual(reaped, 1)
        state = get_task_state(task_id)
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["retries"], 1)
        queued = [
            value.decode() if isinstance(value, bytes) else value
            for value in self.redis.lrange(settings.TASK_QUEUE_KEY, 0, -1)
        ]
        self.assertIn(task_id, queued)

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
