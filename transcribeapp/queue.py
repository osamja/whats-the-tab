import json
import random
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


_CLAIM_TASK = """
local id = redis.call('RPOP', KEYS[1])
if not id then return false end
local key = ARGV[1] .. id
if redis.call('HGET', key, 'status') ~= 'pending' then return false end
redis.call('LPUSH', KEYS[2], id)
redis.call('ZADD', KEYS[3], ARGV[2], id)
redis.call('HSET', key, 'status', 'processing', 'started_at', ARGV[2],
           'worker_id', ARGV[3])
redis.call('PUBLISH', KEYS[4], id)
return id
"""


def claim_task(worker_id=None, timeout=None):
    """Claim a pending task and record its lease in one Redis operation."""
    r = get_redis()
    timeout = settings.TASK_POLL_TIMEOUT if timeout is None else timeout
    deadline = time.monotonic() + timeout
    owner = worker_id or uuid.uuid4().hex[:8]
    while True:
        promote_due_retries(r)
        task_id = r.eval(
            _CLAIM_TASK, 4, settings.TASK_QUEUE_KEY, settings.TASK_PROCESSING_KEY,
            settings.TASK_PROCESSING_TIME_KEY, settings.TASK_CLAIMED_CHANNEL,
            settings.TASK_HASH_PREFIX, time.time(), owner,
        )
        if task_id:
            tid = task_id.decode()
            return tid, get_task_state(tid)
        remaining = deadline - time.monotonic()
        if timeout and remaining <= 0:
            return None
        time.sleep(min(0.5, remaining) if timeout else 0.5)


def retry_task(task_id, worker_id, error, *, stale_before=None):
    """Release the current claim and schedule an attempt or record failure."""
    r = get_redis()
    key = f"{settings.TASK_HASH_PREFIX}{task_id}"
    owner = worker_id.encode() if isinstance(worker_id, str) else worker_id
    with r.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)
                status, current_owner, count = pipe.hmget(key, "status", "worker_id", "retries")
                if status != b"processing" or current_owner != owner:
                    return False
                now = time.time()
                if stale_before is not None:
                    pipe.watch(settings.TASK_PROCESSING_TIME_KEY)
                    lease = pipe.zscore(settings.TASK_PROCESSING_TIME_KEY, task_id)
                    if lease is None or lease > stale_before:
                        return False
                    if worker_id:
                        heartbeat_key = f"worker:{worker_id}"
                        pipe.watch(heartbeat_key)
                        if pipe.exists(heartbeat_key):
                            pipe.multi()
                            pipe.zadd(settings.TASK_PROCESSING_TIME_KEY, {task_id: now})
                            pipe.execute()
                            return False

                retries = int(count or 0)
                pipe.multi()
                _release_claim(pipe, task_id)
                if retries < settings.TASK_MAX_RETRIES:
                    delay = (30 if retries == 0 else 120) * random.uniform(0.85, 1.15)
                    retry_at = now + delay
                    pipe.hset(key, mapping={
                        "status": "retrying", "retries": retries + 1, "retry_at": retry_at,
                    })
                    pipe.hdel(key, "started_at", "worker_id", "error")
                    pipe.zadd(settings.TASK_RETRY_KEY, {task_id: retry_at})
                else:
                    _record_failure(pipe, key, task_id, error)
                pipe.execute()
                return True
            except redis.WatchError:
                continue


def promote_due_retries(r):
    now = time.time()
    for task_id in r.zrangebyscore(settings.TASK_RETRY_KEY, "-inf", now, start=0, num=100):
        key = f"{settings.TASK_HASH_PREFIX}{task_id.decode()}"
        with r.pipeline() as pipe:
            try:
                pipe.watch(key, settings.TASK_RETRY_KEY)
                waiting = pipe.hget(key, "status") == b"retrying"
                retry_at = pipe.zscore(settings.TASK_RETRY_KEY, task_id)
                if retry_at is None or retry_at > now:
                    continue
                pipe.multi()
                pipe.zrem(settings.TASK_RETRY_KEY, task_id)
                if not waiting:
                    pipe.execute()
                    continue
                pipe.hset(key, "status", "pending")
                pipe.hdel(key, "retry_at")
                pipe.rpush(settings.TASK_QUEUE_KEY, task_id)
                pipe.publish(settings.TASK_NEW_CHANNEL, task_id)
                pipe.execute()
            except redis.WatchError:
                continue


def reap_stale_tasks():
    """Recover claims past their lease when the dispatcher heartbeat is gone."""
    r = get_redis()
    cutoff = time.time() - settings.TASK_PROCESSING_TIMEOUT
    stale_ids = r.zrangebyscore(
        settings.TASK_PROCESSING_TIME_KEY, "-inf", cutoff, start=0, num=100
    )
    reaped = 0
    for raw_id in stale_ids:
        task_id = raw_id.decode()
        key = f"{settings.TASK_HASH_PREFIX}{task_id}"
        owner = r.hget(key, "worker_id")
        reaped += retry_task(
            task_id, owner.decode() if owner else None,
            "Transcription worker stopped responding.", stale_before=cutoff,
        )
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
        elif key in ("created_at", "started_at", "completed_at", "retries", "retry_at"):
            try:
                val = float(val)
            except (ValueError, TypeError):
                pass
        decoded[key] = val
    return decoded


def _release_claim(pipe, task_id):
    pipe.lrem(settings.TASK_PROCESSING_KEY, 0, task_id)
    pipe.zrem(settings.TASK_PROCESSING_TIME_KEY, task_id)
    pipe.zrem(settings.TASK_RETRY_KEY, task_id)


def _record_failure(pipe, key, task_id, error):
    pipe.hset(key, mapping={
        "status": "failed", "completed_at": time.time(), "error": str(error),
    })
    pipe.rpush(settings.TASK_RESULTS_KEY, task_id)
    pipe.rpush(settings.TASK_FAILED_KEY, task_id)
    pipe.ltrim(settings.TASK_FAILED_KEY, -settings.TASK_FAILED_MAX, -1)
    pipe.publish(settings.TASK_FAILED_CHANNEL, task_id)
    pipe.expire(key, settings.TASK_FAILED_TTL)


def mark_completed(task_id, result_status="completed"):
    r = get_redis()
    key = f"{settings.TASK_HASH_PREFIX}{task_id}"
    with r.pipeline() as pipe:
        _release_claim(pipe, task_id)
        pipe.hset(key, mapping={
            "status": "completed", "result_status": result_status,
            "completed_at": time.time(),
        })
        pipe.rpush(settings.TASK_RESULTS_KEY, task_id)
        pipe.publish(settings.TASK_COMPLETED_CHANNEL, task_id)
        pipe.expire(key, settings.TASK_RESULT_TTL)
        pipe.execute()


def mark_failed(task_id, error):
    r = get_redis()
    with r.pipeline() as pipe:
        _release_claim(pipe, task_id)
        _record_failure(pipe, f"{settings.TASK_HASH_PREFIX}{task_id}", task_id, error)
        pipe.execute()


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
