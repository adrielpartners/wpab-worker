# PROJECT_RULES.md

Version: 1.0  
Project: WP AB Worker  
Repository: `wpab-worker`  
System Type: Backend Worker Service  
Last Updated: 2026-05-25

---

# Purpose

This file defines repository-specific rules for AI agents and developers working on WP AB Worker.

This is not the architecture document. This file tells agents how to work inside this repo without turning the worker into a second product core, adding unnecessary durability, or creating unsafe processing behavior.

Before making substantial changes, read:

1. `CODING_CONSTITUTION.md`
2. `AGENTS.md`
3. `ARCHITECTURE.md`
4. `DECISIONS.md`
5. `PROJECT_RULES.md`

---

# 1. Repository Role

This repo contains the backend worker service for WP Audio Buddy.

The worker owns:

- signed job intake
- request validation
- active queueing
- audio download
- audio chunking
- OpenAI transcription in worker mode
- transcript assembly
- signed callbacks to WordPress
- temporary file cleanup
- operational logs
- health endpoint

The worker does not own:

- WordPress admin UI
- final transcript storage
- summary/excerpt display
- WordPress attachment relationships
- durable product job history
- plugin settings
- user accounts
- billing
- a public dashboard in v1

---

# 2. Absolute Rules

AI agents must follow these rules:

1. Do not turn the worker into the product core.
2. Do not add a durable database in v1 unless explicitly approved.
3. Do not store final transcripts permanently in the worker.
4. Do not accept unsigned requests.
5. Do not send unsigned callbacks.
6. Do not process long-running jobs inside the API request cycle.
7. Do not allow unbounded file size, memory use, disk use, or concurrency.
8. Do not log secrets, signed URLs, API keys, or raw sensitive content.
9. Do not add a frontend or dashboard unless explicitly approved.
10. Do not add unrelated endpoints.

---

# 3. Architectural Flow

Use this flow:

```text
HTTP API Request
→ Request Validation / HMAC Verification
→ Job Service
→ Queue
→ Worker Process
→ Audio Download Service
→ Chunking Service
→ Transcription Service
→ Result Assembly Service
→ Callback Client
→ Cleanup Service
```

## API Routes

Routes should be thin.

Routes may:

- parse request body
- validate payload model
- verify signature
- enqueue job
- return response

Routes must not:

- download audio
- chunk audio
- call OpenAI
- perform long-running transcription
- contain processing workflow logic

## Services

Services own behavior.

Service responsibilities:

- job coordination
- audio download
- chunking
- transcription
- result assembly
- callback delivery
- cleanup

## Worker Processes

Worker processes run queued jobs.

They may:

- fetch queued jobs
- call processing services
- update active job state
- call callback service
- run cleanup

They must not:

- expose HTTP routes
- own durable product data
- assume callback success without checking

---

# 4. File and Folder Rules

Use this preferred structure as the target:

```text
wpab-worker/
  app/
    main.py
    api/
      routes.py
      health.py
      jobs.py
    core/
      config.py
      logging.py
      security.py
    services/
      job_service.py
      audio_download_service.py
      chunking_service.py
      transcription_service.py
      result_assembly_service.py
      callback_client.py
      cleanup_service.py
    workers/
      transcription_worker.py
    models/
      job.py
      payloads.py
      results.py
    tests/
  scripts/
  docker/
  Dockerfile
  docker-compose.yml
  README.md
  ARCHITECTURE.md
  PROJECT_RULES.md
  DECISIONS.md
```

## File Placement

Place files according to responsibility:

- FastAPI app bootstrapping: `app/main.py`
- HTTP routes: `app/api/`
- config/security/logging: `app/core/`
- processing behavior: `app/services/`
- queue workers: `app/workers/`
- request/result models: `app/models/`
- tests: `app/tests/` or `tests/`
- operational scripts: `scripts/`
- Docker support: root `Dockerfile`, `docker-compose.yml`, or `docker/`

Do not create vague folders such as:

```text
helpers
misc
stuff
temp
old
new
```

Runtime temp files must go in the configured temp directory, not in source-controlled folders.

---

# 5. API Rules

## Expected Endpoints

Keep the API tiny.

Expected v1 endpoints:

```text
GET /health
POST /v1/jobs/transcribe
GET /v1/jobs/{job_id}
```

Do not add new endpoints without a clear reason.

## Response Shape

Use consistent JSON envelopes.

Success:

```json
{
  "ok": true,
  "data": {}
}
```

Failure:

```json
{
  "ok": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "User-safe message."
  }
}
```

Do not return stack traces, raw provider errors, secrets, or internal file paths.

## Long-Running Work

`POST /v1/jobs/transcribe` should enqueue work and return quickly.

It must not perform full audio processing before responding.

---

# 6. Security Rules

## HMAC Required

All job submission requests must be HMAC signed.

Expected headers:

```text
X-WPAB-Site-ID
X-WPAB-Timestamp
X-WPAB-Signature
```

## Signature Validation

Validate:

- site ID exists
- timestamp is within tolerance
- signature matches the exact timestamp/site/body signing payload:
  `TIMESTAMP + "\n" + SITE_ID + "\n" + RAW_JSON_BODY`

Reject invalid requests before:

- downloading audio
- enqueueing jobs
- calling OpenAI
- touching temp storage

## Callback Signing

Callbacks to WordPress must also be signed.

Never send unsigned success or failure callbacks.

## Secrets

Never log:

- OpenAI API key
- site shared secrets
- full signed audio URLs
- callback signatures
- raw Authorization headers

---

# 7. Configuration Rules

Use environment variables for deployment configuration.

Recommended variables:

```text
WPAB_WORKER_ENV
WPAB_WORKER_PUBLIC_BASE_URL
WPAB_ALLOWED_SITES
WPAB_SITE_SECRETS
WPAB_OPENAI_API_KEY
WPAB_REDIS_URL
WPAB_TEMP_DIR
WPAB_MAX_AUDIO_BYTES
WPAB_MAX_JOB_SECONDS
WPAB_CALLBACK_TIMEOUT_SECONDS
WPAB_REQUEST_TIMESTAMP_TOLERANCE_SECONDS
WPAB_LOG_LEVEL
```

## Config Handling

All config should be loaded in one configuration module.

Recommended file:

```text
app/core/config.py
```

Do not scatter environment variable reads throughout the codebase.

## Defaults

Use safe defaults.

Examples:

- low concurrency by default
- bounded file size
- short timestamp tolerance
- strict log level in production
- temp directory outside source tree

---

# 8. Queue and Job Rules

## Queue

Use Redis-backed queueing in v1.

Preferred:

```text
Redis + RQ
```

or an equivalent lightweight Python queue if explicitly approved.

## Active Job State

The worker may store active job status in Redis.

The worker must not treat Redis as the durable product database.

## Job Lifecycle

Expected states:

```text
accepted
queued
running
callback_pending
completed
failed
expired
```

These are worker-side active states only. WordPress owns durable job state.

## Retry Rules

Recommended defaults:

- audio download network failure: retry up to 2 times
- transient OpenAI error: retry up to 2 times
- callback network failure: retry up to 3 times
- invalid signature: no retry
- unsupported file type: no retry
- file too large: no retry
- invalid callback URL: no retry

Do not create unbounded retry loops.

---

# 9. Audio Processing Rules

## Download

The worker should download audio from a short-lived signed URL supplied by WordPress.

Download rules:

- use explicit timeout
- enforce max file size
- validate response status
- validate content length when available
- validate content type when practical
- write to temp storage only
- do not expose downloaded file publicly
- fetch the URL exactly as provided, without cookies or WordPress auth headers
- treat 403 and 404 responses as normal job failures with safe diagnostics

## Chunking

Chunking service should:

- preserve chunk order
- enforce max chunk size/duration
- create predictable temp filenames
- avoid loading entire large files into memory when practical
- return structured metadata for each chunk

## Transcription

Transcription service should:

- call OpenAI through one integration wrapper
- process chunks in order unless concurrency is explicitly controlled
- normalize provider errors
- retry transient failures only
- avoid leaking raw provider errors to callback payloads

## Result Assembly

Result assembly should:

- preserve chunk order
- stitch text cleanly
- include segment metadata if available
- produce a stable callback payload

---

# 10. Callback Rules

## WordPress Callback Endpoint

Expected WordPress route:

```text
POST /wp-json/wpab/v1/worker-callback
```

The worker receives the callback URL dynamically from the job submission payload and does not hardcode the route.

## Success Payload

Callback success payload should include:

```json
{
  "job_id": "wpab_123",
  "job_uuid": "uuid-from-wordpress",
  "attachment_id": 456,
  "status": "done",
  "timestamp": 1712345678,
  "site_id": "site-1",
  "transcript": "...",
  "model": "gpt-4o-mini-transcribe",
  "seconds": 1245
}
```

## Failure Payload

Callback failure payload should include:

```json
{
  "job_id": "wpab_123",
  "job_uuid": "uuid-from-wordpress",
  "attachment_id": 456,
  "status": "error",
  "timestamp": 1712345678,
  "site_id": "site-1",
  "error": "User-safe message."
}
```

## Callback Behavior

The callback client must:

- sign every callback
- use explicit timeout
- retry transient failures
- log final failure
- never assume success without checking response

---

# 11. Temporary File Rules

Temporary files may include:

- downloaded source audio
- audio chunks
- intermediate transcript files if needed

Rules:

- store only in configured temp directory
- never commit temp files
- clean after success
- clean after failure when safe
- run stale cleanup
- avoid unbounded disk growth

Do not preserve temp files indefinitely for debugging unless explicitly configured in development mode.

---

# 12. Error Handling Rules

Use consistent error codes.

Recommended codes:

```text
INVALID_SIGNATURE
TIMESTAMP_EXPIRED
UNKNOWN_SITE
INVALID_PAYLOAD
AUDIO_DOWNLOAD_FAILED
AUDIO_TOO_LARGE
UNSUPPORTED_AUDIO_FORMAT
CHUNKING_FAILED
TRANSCRIPTION_FAILED
CALLBACK_FAILED
JOB_NOT_FOUND
INTERNAL_ERROR
```

Rules:

- log technical details internally
- return user-safe error messages
- do not expose stack traces
- do not swallow errors silently
- do not mark a job completed unless callback succeeds or completion is clearly recorded according to the job policy

---

# 13. Logging Rules

Use structured logs.

Log:

- job accepted
- job rejected
- job started
- audio download started/completed
- chunking started/completed
- chunk transcription started/completed
- result assembly completed
- callback sent
- callback failed
- cleanup completed
- job failed

Do not log:

- OpenAI API keys
- shared secrets
- full signed URLs
- raw audio content
- sensitive user data
- full transcript text by default

A transcript may be logged only in explicit local development debugging, and that behavior must not be enabled by default.

---

# 14. Testing and Verification Rules

Use pytest when tests exist or are added.

Priority tests:

- HMAC verification
- timestamp tolerance
- unknown site rejection
- request payload validation
- callback signing
- callback payload format
- audio download failure handling
- chunk ordering
- transcript assembly
- cleanup behavior
- retry limits

If automated tests are not available, perform a manual smoke test and report exactly what was tested.

Do not claim tests passed if they were not run.

---

# 15. Docker and Deployment Rules

The worker should run cleanly in Docker.

Production shape:

```text
reverse proxy
→ FastAPI API container
→ Redis
→ worker process/container
```

Rules:

- API and worker should be separate processes or containers in production.
- Redis must not be publicly exposed.
- API must be served over HTTPS in production.
- secrets must come from environment variables or secret management.
- temp storage must be bounded and cleanup-aware.
- container logs should be sufficient for v1 diagnosis.

Do not require manual shell steps for normal startup if Docker Compose can handle them.

---

# 16. Dependency Rules

Do not add dependencies casually.

Before adding a dependency, confirm:

- the standard library cannot reasonably handle it
- FastAPI/Pydantic cannot already solve it
- the dependency is maintained
- the dependency does not introduce heavy operational burden
- the dependency is appropriate for a backend worker

Avoid dependencies for trivial wrappers, formatting, or one-off utilities.

If a dependency is added, explain why in the work summary.

---

# 17. Git and Agent Workflow

Before work:

```bash
git status
```

If there are existing changes, do not overwrite them.

Preferred workflow:

```bash
git pull
git checkout -b feature/short-description
# make changes
git status
# run checks
git add .
git commit -m "Short clear message"
git push
```

Do not use destructive commands such as:

```bash
git reset --hard
git clean -fd
git checkout -- .
```

unless explicitly instructed.

---

# 18. Documentation Update Rules

Update documentation when changing:

- API endpoints
- callback payloads
- HMAC signing format
- allowed site configuration
- OpenAI model usage
- queue system
- retry rules
- temp file cleanup
- deployment requirements
- environment variables
- worker/plugin boundary

Update:

- `ARCHITECTURE.md` for current system facts
- `DECISIONS.md` for major choices
- `PROJECT_RULES.md` for repo-specific rules

---

# 19. Definition of Done

A task is done when:

- the change matches the request
- code is in the correct layer
- request validation is present
- HMAC behavior is preserved where relevant
- external calls use timeouts
- resource limits are respected
- temp files are cleaned up
- retry behavior is bounded
- errors are logged safely
- secrets are not exposed
- relevant checks were run or honestly reported
- docs were updated if architecture or rules changed

---

# 20. Agent Work Summary Format

At the end of a coding task, summarize:

```text
Summary:
- Changed ...
- Added ...
- Updated ...

Verification:
- Ran ...
- Not run: ... because ...

Docs:
- Updated ...
- Not updated because ...

Notes:
- Assumptions ...
- Risks ...
- Follow-up ...
```

Mention important files changed.

Mention whether API, security, queueing, processing, callback, or deployment behavior was affected.

---

# Final Rule

Keep WP AB Worker narrow.

It receives signed jobs, processes audio safely, sends signed callbacks, and cleans up after itself.

It is not a dashboard, product database, SaaS platform, or replacement for WP Audio Buddy.
