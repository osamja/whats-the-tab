import json
import time
import uuid
import redis
from django.conf import settings


def get_redis():
    return redis.from_url(settings.REDIS_URL)


def enqueue_task(task_data: dict) -> str:
    task_id = task_data.get("audio_midi_id", uuid.uuid4().hex)
    r = get_redis()

    payload = json.dumps(task_data)
    now = time.time()
    task_key = f"{settings.TASK_HASH_PREFIX}{task_id}"

    pipeline = r.pipeline()
    pipeline.hset(
        task_key,
        mapping={
            "status": "pending",
            "payload": payload,
            "created_at": now,
            "retries": 0,
        },
    )
    pipeline.rpush(settings.TASK_QUEUE_KEY, task_id)
    pipeline.publish(settings.TASK_NEW_CHANNEL, task_id)
    pipeline.execute()

    return task_id


def claim_task(worker_id=None, timeout=None):
    """Claim the next task, blocking up to ``timeout`` seconds.

    Returns a ``(task_id, state)`` tuple, or ``None`` when the queue is empty.
    The claimed task ID is returned explicitly so callers never have to infer
    it from a pub/sub notification or a different queue entry.
    """
    r = get_redis()

    if timeout is None:
        timeout = settings.TASK_POLL_TIMEOUT

    task_id = r.brpoplpush(
        settings.TASK_QUEUE_KEY,
        settings.TASK_PROCESSING_KEY,
        timeout=timeout,
    )

    if not task_id:
        return None

    tid = task_id.decode() if isinstance(task_id, bytes) else task_id
    task_key = f"{settings.TASK_HASH_PREFIX}{tid}"

    # A task hash may have expired or been deleted while it sat in the queue.
    # Clean up the processing list rather than inventing a state with no payload.
    if not r.exists(task_key):
        r.lrem(settings.TASK_PROCESSING_KEY, 0, tid)
        r.zrem(settings.TASK_PROCESSING_TIME_KEY, tid)
        return None

    now = time.time()
    owner = worker_id or uuid.uuid4().hex[:8]

    pipeline = r.pipeline()
    pipeline.zadd(settings.TASK_PROCESSING_TIME_KEY, {tid: now})
    pipeline.hset(
        task_key,
        mapping={
            "status": "processing",
            "started_at": now,
            "worker_id": owner,
        },
    )
    pipeline.publish(settings.TASK_CLAIMED_CHANNEL, tid)
    pipeline.execute()

    return tid, get_task_state(tid)


def reap_stale_tasks():
    """Requeue or fail tasks whose worker has disappeared.

    A task is considered stale when its processing deadline has passed *and*
    its worker heartbeat is gone. Live workers keep their heartbeat key alive
    while a long transcription is running, so long jobs are not reaped.
    """
    r = get_redis()
    cutoff = time.time() - settings.TASK_PROCESSING_TIMEOUT
    stale_ids = r.zrangebyscore(
        settings.TASK_PROCESSING_TIME_KEY,
        "-inf",
        cutoff,
        start=0,
        num=100,
    )

    reaped = 0
    for raw_id in stale_ids:
        tid = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
        task_key = f"{settings.TASK_HASH_PREFIX}{tid}"

        # Only one reaper may own this task. If another one won the race, or
        # the worker completed and removed the zset entry, skip it.
        if r.zrem(settings.TASK_PROCESSING_TIME_KEY, tid) != 1:
            continue

        r.lrem(settings.TASK_PROCESSING_KEY, 0, tid)
        state = get_task_state(tid)

        if not state or state.get("status") != "processing":
            continue

        worker_id = state.get("worker_id")
        if worker_id and r.exists(f"worker:{worker_id}"):
            # The worker is alive and still heartbeating. Extend its lease.
            r.zadd(settings.TASK_PROCESSING_TIME_KEY, {tid: time.time()})
            continue

        try:
            retries = int(state.get("retries", 0) or 0)
        except (TypeError, ValueError):
            retries = 0

        if retries < settings.TASK_MAX_RETRIES:
            r.hset(
                task_key,
                mapping={
                    "status": "pending",
                    "retries": retries + 1,
                },
            )
            r.hdel(task_key, "started_at", "worker_id", "error")
            r.rpush(settings.TASK_QUEUE_KEY, tid)
            r.publish(settings.TASK_NEW_CHANNEL, tid)
        else:
            mark_failed(
                tid,
                f"task timed out after {retries} retries",
            )

        reaped += 1

    return reaped


def get_task_state(task_id):
    r = get_redis()
    state = r.hgetall(f"{settings.TASK_HASH_PREFIX}{task_id}")
    if not state:
        return None
    decoded = {}
    for k, v in state.items():
        key = k.decode() if isinstance(k, bytes) else k
        val = v.decode() if isinstance(v, bytes) else v
        if key == "payload":
            val = json.loads(val)
        elif key in ("created_at", "started_at", "completed_at", "retries"):
            try:
                val = float(val)
            except (ValueError, TypeError):
                pass
        decoded[key] = val
    return decoded


def mark_completed(task_id, result_status="completed"):
    r = get_redis()
    task_key = f"{settings.TASK_HASH_PREFIX}{task_id}"

    r.hset(
        task_key,
        mapping={
            "status": "completed",
            "result_status": result_status,
            "completed_at": time.time(),
        },
    )

    r.lrem(settings.TASK_PROCESSING_KEY, 0, task_id)
    r.zrem(settings.TASK_PROCESSING_TIME_KEY, task_id)
    r.rpush(settings.TASK_RESULTS_KEY, task_id)
    r.publish(settings.TASK_COMPLETED_CHANNEL, task_id)
    r.expire(task_key, settings.TASK_RESULT_TTL)


def mark_failed(task_id, error):
    r = get_redis()
    task_key = f"{settings.TASK_HASH_PREFIX}{task_id}"

    r.hset(
        task_key,
        mapping={
            "status": "failed",
            "completed_at": time.time(),
            "error": str(error),
        },
    )

    r.lrem(settings.TASK_PROCESSING_KEY, 0, task_id)
    r.zrem(settings.TASK_PROCESSING_TIME_KEY, task_id)
    r.rpush(settings.TASK_FAILED_KEY, task_id)
    r.ltrim(settings.TASK_FAILED_KEY, -settings.TASK_FAILED_MAX, -1)
    r.publish(settings.TASK_FAILED_CHANNEL, task_id)
    r.expire(task_key, settings.TASK_FAILED_TTL)


def publish_progress(task_id, current, total):
    r = get_redis()
    r.publish(
        f"{settings.TASK_PROGRESS_PREFIX}{task_id}",
        json.dumps({"current": current, "total": total}),
    )


def heartbeat(worker_id, task_id=None):
    r = get_redis()
    r.hset(
        f"worker:{worker_id}",
        mapping={
            "last_heartbeat": time.time(),
            "current_task": task_id or "",
        },
    )
    r.expire(f"worker:{worker_id}", 30)


def get_queue_stats():
    r = get_redis()
    pipeline = r.pipeline()
    pipeline.llen(settings.TASK_QUEUE_KEY)
    pipeline.llen(settings.TASK_PROCESSING_KEY)
    pipeline.llen(settings.TASK_FAILED_KEY)
    pipeline.zcard(settings.TASK_PROCESSING_TIME_KEY)
    pipeline.pubsub_numsub(settings.TASK_NEW_CHANNEL)
    pipeline.info("clients")
    pipeline.info("server")
    pipeline.info("memory")
    results = pipeline.execute()
    return {
        "queue_pending": results[0],
        "queue_processing": results[1],
        "queue_failed": results[2],
        "processing_timed": results[3],
        "subscribers_task_new": results[4][0][1] if results[4] else 0,
        "redis_connected_clients": results[5].get("connected_clients", 0),
        "redis_uptime_seconds": results[6].get("uptime_in_seconds", 0),
        "redis_used_memory_human": results[7].get("used_memory_human", "0"),
    }
