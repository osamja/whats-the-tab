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

    r.hset(
        f"{settings.TASK_HASH_PREFIX}{task_id}",
        mapping={
            "status": "pending",
            "payload": payload,
            "created_at": time.time(),
            "retries": 0,
        },
    )

    r.rpush(settings.TASK_QUEUE_KEY, task_id)
    r.publish(settings.TASK_NEW_CHANNEL, task_id)

    return task_id


def claim_task():
    r = get_redis()
    task_id = r.rpoplpush(settings.TASK_QUEUE_KEY, settings.TASK_PROCESSING_KEY)

    if not task_id:
        return None

    r.zadd(settings.TASK_PROCESSING_TIME_KEY, {task_id: time.time()})

    r.hset(
        f"{settings.TASK_HASH_PREFIX}{task_id}",
        mapping={
            "status": "processing",
            "started_at": time.time(),
            "worker_id": uuid.uuid4().hex[:8],
        },
    )

    r.publish(settings.TASK_CLAIMED_CHANNEL, task_id)

    return get_task_state(task_id)


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

    r.hset(
        f"{settings.TASK_HASH_PREFIX}{task_id}",
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


def mark_failed(task_id, error):
    r = get_redis()

    r.hset(
        f"{settings.TASK_HASH_PREFIX}{task_id}",
        mapping={
            "status": "failed",
            "completed_at": time.time(),
            "error": str(error),
        },
    )

    r.lrem(settings.TASK_PROCESSING_KEY, 0, task_id)
    r.zrem(settings.TASK_PROCESSING_TIME_KEY, task_id)
    r.rpush(settings.TASK_FAILED_KEY, task_id)
    r.publish(settings.TASK_FAILED_CHANNEL, task_id)
    r.expire(f"{settings.TASK_HASH_PREFIX}{task_id}", 86400)


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
