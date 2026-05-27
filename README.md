# WP AB Worker

**Version:** 0.2.0  
**System Type:** Backend Worker Service  

A narrow backend processing service for [WP Audio Buddy](https://github.com/adrielpartners/wp-audio-buddy). Handles heavy audio-processing work that should not run inside WordPress — audio downloading, FFmpeg chunking, provider-backed transcription, and signed callbacks.

---

## Quick Start

```bash
# Copy environment template
cp .env.example .env
# Edit .env with your settings
# Minimum: one provider API key + WPAB_ALLOWED_SITES

# Start all services
docker compose up --build

# Or run locally (requires Redis)
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Architecture

```
HTTP Request
→ API Route (thin)
→ HMAC Verification (multi-site, timestamp, signature)
→ RQ Queue → Background Worker
  → Audio Download Service
  → Chunking Service (FFmpeg)
  → Transcription Provider
  → Result Assembly Service
  → Callback Client (signed POST to WordPress)
→ Cleanup Service
```

### Project Structure

```
wpab-worker/
├── app/
│   ├── main.py                    # FastAPI app bootstrap
│   ├── api/
│   │   └── routes.py              # All HTTP endpoints
│   ├── core/
│   │   ├── config.py              # Centralized environment config
│   │   ├── logging.py             # Structured logging setup
│   │   └── security.py            # HMAC signing/verification
│   ├── models/
│   │   └── payloads.py            # Pydantic request/response models
│   ├── services/
│   │   ├── audio_download_service.py
│   │   ├── chunking_service.py
│   │   ├── transcription_service.py
│   │   ├── result_assembly_service.py
│   │   ├── callback_client.py
│   │   ├── cleanup_service.py
│   │   └── job_service.py         # Pipeline orchestrator
│   └── workers/
│       ├── transcription_worker.py  # RQ worker entrypoint
│       └── cleanup_daemon.py        # Stale file cleanup loop
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API Endpoints

### `GET /health`
Health check. Returns `{"ok": true}`.

### `POST /v1/jobs/transcribe`
Submit a transcription job. Returns immediately after enqueueing.

**Headers:**
- `X-WPAB-Signature` — HMAC-SHA256 of the timestamp/site/body signing payload
- `X-WPAB-Timestamp` — Unix timestamp
- `X-WPAB-Site-ID` — Site identifier from `WPAB_ALLOWED_SITES`

**Body:**
```json
{
  "job_id": 123,
  "attachment_id": 42,
  "audio_url": "https://example.com/wp-json/wpab/v1/audio-download?token=short-lived-signed-token",
  "callback_url": "https://example.com/wp-json/wpab/v1/worker-callback",
  "job_uuid": "uuid-from-wordpress",
  "operation": "transcribe",
  "model": "gpt-4o-mini-transcribe",
  "chunk_seconds": 660,
  "timestamp": 1712345678,
  "site_id": "site-1"
}
```

**Response:**
```json
{"ok": true, "data": {"job_id": "rq-job-id", "status": "accepted"}}
```

### `GET /v1/jobs/{job_id}`
Get the current status of a job.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `DEEPGRAM_API_KEY` | — | Deepgram API key |
| `WPAB_ALLOWED_SITES` | — | Comma-separated `site_id=secret` pairs (required in production) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `WPAB_LOG_LEVEL` | `INFO` | Logging level |
| `WPAB_WORKER_ENV` | `development` | Environment name |
| `WPAB_DEFAULT_MODEL` | `gpt-4o-mini-transcribe` | Fallback transcription model when provider metadata has no default |
| `WPAB_MAX_DOWNLOAD_MB` | `200` | Max audio file size to download (MB) |
| `WPAB_DEFAULT_CHUNK_SECONDS` | `720` | Fallback seconds per audio chunk; provider defaults may override |
| `WPAB_REQUEST_TIMESTAMP_TOLERANCE_SECONDS` | `300` | Max clock drift for HMAC timestamps (seconds) |
| `WPAB_WORK_ROOT` | `/work` | Directory for temporary audio files |
| `WPAB_RETENTION_HOURS` | `168` | Hours to keep stale job files before cleanup |
| `WPAB_KEEP_JOB_FILES` | `0` | Set to `1` to preserve temp files for debugging |
| `WPAB_MAX_RETRY_ATTEMPTS` | `4` | Max retries for transient provider/callback failures |

---

## HMAC Signing

Requests and callbacks use HMAC-SHA256 with site-specific shared secrets.

**Request signing (WordPress → Worker):**
1. JSON-encode the body with compact separators
2. Build the signing payload:
   `TIMESTAMP + "\n" + SITE_ID + "\n" + RAW_JSON_BODY`
3. Compute `hmac_sha256(signing_payload, site_secret)`
4. Send `X-WPAB-Site-ID`, `X-WPAB-Timestamp`, and `X-WPAB-Signature`

**Worker verification:**
1. Look up secret for `X-WPAB-Site-ID`
2. Check `X-WPAB-Timestamp` is within tolerance window
3. Rebuild the timestamp/site/body signing payload
4. Compute expected HMAC and compare with `X-WPAB-Signature`

---

## Running the Stack

```bash
# Start everything
docker compose up --build

# Run API only (for development)
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Run worker only
rq worker wpab-transcription --url redis://localhost:6379

# Run cleanup daemon
python3 -m app.workers.cleanup_daemon
```

---

## Callbacks

After processing, the worker sends a signed POST to the callback URL provided by WordPress.

**Success payload:**
```json
{
  "job_id": "123",
  "job_uuid": "uuid-from-wordpress",
  "attachment_id": 42,
  "status": "done",
  "timestamp": 1712345678,
  "site_id": "site-1",
  "transcript": "Full transcribed text...",
  "model": "gpt-4o-mini-transcribe",
  "seconds": 1245
}
```

**Failure payload:**
```json
{
  "job_id": "123",
  "job_uuid": "uuid-from-wordpress",
  "attachment_id": 42,
  "status": "error",
  "timestamp": 1712345678,
  "site_id": "site-1",
  "error": "The audio could not be transcribed."
}
```

---

## Known Limitations

- **No durable storage** — completed job results are not preserved after successful callback
- **Single worker concurrency** — RQ processes one job at a time by default; adjust with `rq worker --num-workers`
- **Download URL signing is owned by WordPress** — the worker receives a short-lived signed audio URL, fetches it exactly as provided without cookies or WordPress auth headers, and validates size/type while downloading
- **FFmpeg required** — audio chunking depends on FFmpeg (included in the Docker image)

---

## Roadmap

- [x] Phase 0-1: Baseline and structure normalization
- [x] Phase 2-3: Config/logging and HMAC security
- [x] Phase 4-5: API contracts and Redis queue
- [x] Phase 6-10: Service modules (download, chunk, transcribe, assemble, callback)
- [x] Phase 11-12: Cleanup and Docker deployment
- [ ] Phase 13: Testing and hardening
- [ ] Phase 14: Final documentation
- [x] Signed temporary download URLs
- [ ] Per-site model overrides
- [ ] Prometheus metrics endpoint
