# System Design

## Architecture

```
                         POST /transcribe/upload/
                         POST /transcribe/generate/
                         GET  /transcribe/status/{id}
                         GET  /transcribe/midi/{id}
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         LINODE / ANY CLOUD HOST                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────┐   ┌──────────────────────────┐ │
│  │              WEB CONTAINER                    │   │     POSTGRESQL           │ │
│  │                                              │   │     (local port)         │ │
│  │  ┌──────────────────────────────────────┐   │   │                          │ │
│  │  │  gunicorn :8008                      │   │   │  AudioMIDI table         │ │
│  │  │  ┌─────────────────────────────────┐ │   │   │  ├── id                  │ │
│  │  │  │ POST /transcribe/upload/        │ │   │   │  ├── audio_file          │ │
│  │  │  │ POST /transcribe/generate/      │ │   │◄──┤  ├── midi_file           │ │
│  │  │  │ GET  /transcribe/status/{id}    │ │   │   │  ├── status              │ │
│  │  │  │ GET  /transcribe/midi/{id}      │ │   │   │  ├── current_chunk       │ │
│  │  │  │ POST /transcribe/_result/       │ │   │   │  ├── total_chunks        │ │
│  │  │  └──────┬──────────────┬───────────┘ │   │   │  └── error_message       │ │
│  │  │         │              │             │   │   │                          │ │
│  │  │         │ enqueue      │ file ingest │   │   └──────────────────────────┘ │
│  │  │         ▼              │ (MIDI bin)  │   │                                │
│  │  └─────────┼──────────────┼─────────────┘   │                                │
│  │            │              │                  │                                │
│  │  ┌─────────┴──────────────┴──────────────┐   │                                │
│  │  │  subscriber (background process)      │   │                                │
│  │  │                                       │   │                                │
│  │  │  startup: drain task:results list     │   │                                │
│  │  │  live:    SUBSCRIBE task:claimed      │   │                                │
│  │  │           SUBSCRIBE task:completed    │   │                                │
│  │  │           SUBSCRIBE task:failed       │   │                                │
│  │  │           PSUBSCRIBE task:progress:*  │   │                                │
│  │  │                                       │   │                                │
│  │  │  for each event → HGETALL task:{id}   │   │                                │
│  │  │                 → UPDATE AudioMIDI    │───┤ (sole DB writer)              │
│  │  └──────────────┬────────────────────────┘   │                                │
│  └─────────────────┼────────────────────────────┘                                │
└────────────────────┼────────────────────────────────────────────────────────────┘
                     │
                     │  REDIS_URL=rediss://default:token@upstash.io:6379
                     │
          ┌──────────┴────────────────────────────────────┐
          │                                               │
          ▼                                               ▼
┌───────────────────────────┐            ┌────────────────────────────────────────┐
│    MANAGED REDIS           │            │            VAST.AI INSTANCES           │
│    (Upstash)               │            │                                        │
│                            │            │  ┌──────────────────────────────────┐  │
│  DATA AT REST              │            │  │  WORKER CONTAINER (GPU)           │  │
│  ─────────────             │            │  │                                  │  │
│  task:queue     LIST       │            │  │  runworker (main loop)           │  │
│  task:processing LIST      │            │  │    │                             │  │
│  task:failed    LIST       │            │  │    ├─ SUBSCRIBE task:new         │  │
│  task:results   LIST       │            │  │    ├─ RPOPLPUSH task:queue       │  │
│  task:proc:time ZSET       │            │  │    ├─ HSET task:{id} status      │  │
│  task:{uuid}    HASH       │            │  │    ├─ PUBLISH task:claimed       │  │
│                            │            │  │    ├─ GET /media/audio (HTTP)    │  │
│  DATA IN MOTION            │            │  │    ├─ transcribe_audio() GPU     │  │
│  ──────────────            │            │  │    ├─ PUBLISH task:progress:*    │  │
│  PUB/SUB: task:new         │            │  │    ├─ POST /_result/ (HTTP)      │  │
│  PUB/SUB: task:claimed     │            │  │    ├─ HSET task:{id} completed   │  │
│  PUB/SUB: task:progress:*  │            │  │    ├─ LREM task:processing       │  │
│  PUB/SUB: task:completed   │            │  │    ├─ RPUSH task:results         │  │
│  PUB/SUB: task:failed      │            │  │    └─ PUBLISH task:completed     │  │
│                            │            │  └──────────────────────────────────┘  │
└───────────────────────────┘            │                                        │
                                         │  ┌──────────────────────────────────┐  │
                                         │  │  WORKER CONTAINER (GPU) 2        │  │
                                         │  │  (same loop, competes on RPOPLPUSH) │
                                         │  └──────────────────────────────────┘  │
                                         └────────────────────────────────────────┘

HTTP binary file exchange:
  Worker GET  → web/media/audios/{file}
  Worker POST → web/transcribe/_result/
```

## Task Lifecycle (Redis)

Each task has a hash at `task:{task_id}` tracking its full lifecycle:

```
  HGETALL task:{id}
  ┌────────────────────────────┐
  │ status:        "pending"   │  RPUSH task:queue, PUBLISH task:new
  │ payload:       <json>      │
  │ created_at:    1714...     │
  │ retries:       0           │
  └────────────────────────────┘

  ┌────────────────────────────┐
  │ status:        "processing"│  RPOPLPUSH → task:processing, ZADD time
  │ started_at:    1714...     │  PUBLISH task:claimed
  │ worker_id:     "w3x.."     │
  └────────────────────────────┘

  ┌────────────────────────────┐
  │ status:        "completed" │  PUBLISH task:completed
  │ completed_at:  1714...     │  RPUSH task:results
  │ result_archived: ...       │  LREM task:processing, ZREM time
  └────────────────────────────┘
           OR
  ┌────────────────────────────┐
  │ status:        "failed"    │  PUBLISH task:failed
  │ completed_at:  1714...     │  RPUSH task:failed (dead letter queue)
  │ error:         "OOM..."    │  LREM task:processing, ZREM time
  └────────────────────────────┘
```

## Data Structures

### Data at Rest (Redis)

| Key | Type | Purpose |
|-----|------|---------|
| `task:queue` | LIST | Pending task IDs |
| `task:processing` | LIST | Claimed task IDs |
| `task:processing:time` | ZSET | id → timestamp (timeout detection) |
| `task:failed` | LIST | Dead letter queue |
| `task:results` | LIST | Terminal task IDs — subscriber catch-up |
| `task:{id}` | HASH | Full payload, status, timestamps, error, retries, worker_id |

### Data in Motion (pub/sub)

| Channel | Fires when | Consumer |
|---------|------------|----------|
| `task:new` | Task enqueued | All workers |
| `task:claimed` | Worker acquires | Web subscriber |
| `task:progress:{id}` | Inference progress | Web subscriber (PSUBSCRIBE `task:progress:*`) |
| `task:completed` | Result saved + state written | Web subscriber |
| `task:failed` | Exception caught | Web subscriber |

## Flow

### Enqueue (Web)
1. Client POSTs to `/transcribe/upload/` or `/transcribe/generate/`
2. Web creates `AudioMIDI` record, saves file to Django storage
3. `HSET task:{id}` with payload + status "pending"
4. `RPUSH task:queue {id}`
5. `PUBLISH task:new {id}`

### Claim (Worker)
1. Worker receives `task:new` via pub/sub
2. `RPOPLPUSH task:queue task:processing {id}` (atomic claim)
3. `ZADD task:processing:time {timestamp} {id}`
4. `HSET task:{id} status processing worker_id ...`
5. `PUBLISH task:claimed {id}`

### Process (Worker)
1. `GET http://web/media/audios/{file}` (download audio)
2. Run `transcribe_audio()` (PyTorch GPU inference)
3. Periodically `PUBLISH task:progress:{id} {current_chunk, total_chunks}`

### Complete (Worker)
1. `POST http://web/transcribe/_result/` (upload MIDI file)
2. `HSET task:{id} status completed`
3. `LREM task:processing {id}`, `ZREM task:processing:time {id}`
4. `RPUSH task:results {id}`
5. `PUBLISH task:completed {id}`

### Fail (Worker)
1. `HSET task:{id} status failed error "{msg}"`
2. `LREM task:processing {id}`, `ZREM task:processing:time {id}`
3. `RPUSH task:failed {id}`
4. `PUBLISH task:failed {id}`

### Subscriber (Web background process)
1. **Startup**: drain `task:results` list (BRPOP) → for each `HGETALL` → UPDATE DB
2. **Live**: SUBSCRIBE `task:claimed`, `task:completed`, `task:failed`
3. PSUBSCRIBE `task:progress:*`
4. On each message → `HGETALL task:{id}` → UPDATE AudioMIDI record in PostgreSQL

## Containers

| Service | Dockerfile | Base Image | Size | Purpose |
|---------|-----------|------------|------|---------|
| web | Dockerfile.web | python:3.11-slim | ~600MB | gunicorn + subscriber + endpoint for file exchange |
| worker | Dockerfile | nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 | ~16GB | GPU transcription, no DB access |
| postgres | (official) | postgres:15-alpine | ~274MB | Database, web-exclusive |
| redis | (official) | redis:7-alpine | ~41MB | Local dev only; production uses managed Redis |

## Web Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/transcribe/upload/` | POST | Upload audio file |
| `/transcribe/upload_from_yt/` | POST | Submit YouTube URL |
| `/transcribe/generate/` | POST | Enqueue transcription |
| `/transcribe/status/{id}` | GET | Poll task status |
| `/transcribe/midi/{id}` | GET | Serve MIDI file |
| `/media/{path}` | GET | Serve uploaded audio (worker download) |
| `/transcribe/_result/` | POST | Worker uploads MIDI + final status |

## Management Commands

| Command | Runs on | Purpose |
|---------|---------|---------|
| `manage.py runworker` | Worker container | SUBSCRIBE → RPOPLPUSH → process → PUBLISH |
| `manage.py subscriber` | Web container | Drain results list + SUBSCRIBE live events → update DB |

## Production Hardening

### Task TTL
Failed task hashes auto-expire after 24 hours: `EXPIRE task:{id} 86400`. The dead letter queue entries are transient — inspect, learn, let Redis clean up.

### Worker Heartbeat
A daemon thread writes to `worker:{worker_id}` every 10 seconds with `{last_heartbeat, current_task}`. Hashes expire after 30s of silence so dead workers disappear. Query `HGETALL worker:*` to see live workers.

### Graceful Shutdown
SIGTERM/SIGINT caught by signal handler. Workers flush the currently-processing task to `failed` before exiting. No orphaned tasks left in `task:processing`.

### Idempotent Result Endpoint
`POST /_result/` checks if the task already has a completed file. Duplicate uploads from re-processing return 200 without re-saving.

### Metrics Endpoint
`GET /transcribe/metrics/` returns:
```json
{
  "queue_pending": 3,
  "queue_processing": 1,
  "queue_failed": 2,
  "processing_timed": 1,
  "subscribers_task_new": 2,
  "redis_connected_clients": 5,
  "redis_uptime_seconds": 3600,
  "redis_used_memory_human": "1.2M"
}
```
