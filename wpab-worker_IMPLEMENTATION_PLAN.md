# IMPLEMENTATION_PLAN.md

Version: 1.0  
Project: WP AB Worker  
Repository: `wpab-worker`  
System Type: Backend Worker Service  
Last Updated: 2026-05-25

---

# Purpose

This file turns the WP AB Worker architecture into an ordered build plan.

Use this as the working roadmap for AI agents and developers. It is not permanent doctrine. Update it as work is completed, priorities change, or implementation details become clearer.

Before working on any phase, read:

1. `AGENTS.md`
2. `CODING_CONSTITUTION.md`
3. `ARCHITECTURE.md`
4. `DECISIONS.md`
5. `PROJECT_RULES.md`
6. `IMPLEMENTATION_PLAN.md`

Do not read or apply `MODE_WORDPRESS_NATIVE.md` in this repo. This repo is not a WordPress plugin.

---

# Agent Instructions

When using this plan:

1. Work on one phase or one step at a time.
2. Inspect the repo before editing.
3. Preserve the worker boundary.
4. Do not skip ahead without being asked.
5. Do not add durable storage unless explicitly approved.
6. Mark checklist items complete only after implementation and verification.
7. If the code conflicts with the docs, stop and report the conflict.
8. If a step requires a new decision, ask or document the assumption.
9. End each task with changed files, verification notes, risks, and next step.

Suggested prompt:

```text
You are working in the wpab-worker repo.

Read:
- AGENTS.md
- CODING_CONSTITUTION.md
- ARCHITECTURE.md
- DECISIONS.md
- PROJECT_RULES.md
- IMPLEMENTATION_PLAN.md

Task:
Work on Phase __, Step __ from IMPLEMENTATION_PLAN.md.

Rules:
- Do not skip ahead.
- Do not change unrelated files.
- Preserve the worker boundary.
- Do not add durable worker storage.
- Update the checklist only for completed work.
- Run the smallest relevant verification available.
- Summarize changed files, verification, assumptions, risks, and next step.
```

---

# Build Strategy

Build in this order:

1. Establish repo baseline.
2. Normalize FastAPI/worker structure.
3. Add configuration and logging.
4. Add HMAC security.
5. Add API contracts.
6. Add Redis queue and worker process.
7. Add audio download.
8. Add chunking.
9. Add OpenAI transcription.
10. Add result assembly.
11. Add signed callbacks.
12. Add cleanup and resource limits.
13. Add Docker/deployment readiness.
14. Test, harden, and document.

---

# Phase 0: Repository Baseline

Goal: Understand the current repo and avoid accidental overwrite.

## Checklist

- [ ] Run `git status`.
- [ ] Review current file structure.
- [ ] Identify current framework/app entrypoint.
- [ ] Identify dependency management approach.
- [ ] Identify Docker files, if any.
- [ ] Identify config handling, if any.
- [ ] Identify worker/queue handling, if any.
- [ ] Identify OpenAI usage, if any.
- [ ] Identify audio processing/chunking code, if any.
- [ ] Identify callback behavior, if any.
- [ ] Summarize the baseline before editing.

## Verification

- [ ] Repo state is known.
- [ ] No user changes were overwritten.

---

# Phase 1: Normalize Worker Structure

Goal: Create or normalize the backend worker structure without adding behavior prematurely.

## Target Structure

```text
wpab-worker/
  app/
    main.py
    api/
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
  Dockerfile
  docker-compose.yml
```

## Checklist

- [ ] Ensure `app/main.py` exists as FastAPI app entrypoint.
- [ ] Create `app/api/`.
- [ ] Create `app/core/`.
- [ ] Create `app/services/`.
- [ ] Create `app/workers/`.
- [ ] Create `app/models/`.
- [ ] Create `tests/` or `app/tests/`.
- [ ] Normalize dependency file: `pyproject.toml` or `requirements.txt`.
- [ ] Ensure the app can start with a basic health endpoint.
- [ ] Keep behavior minimal.

## Verification

- [ ] App starts locally.
- [ ] `GET /health` works.
- [ ] Python syntax/import check passes.

---

# Phase 2: Configuration and Logging

Goal: Centralize environment configuration and structured logging.

## Recommended Environment Variables

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

## Checklist

- [ ] Add config module in `app/core/config.py`.
- [ ] Load all environment variables through config module.
- [ ] Add safe local defaults.
- [ ] Validate required production secrets.
- [ ] Add structured logging setup.
- [ ] Ensure logs do not expose secrets.
- [ ] Add `.env.example`.
- [ ] Document config in README or docs.

## Verification

- [ ] App starts with local defaults.
- [ ] Missing required production config fails clearly.
- [ ] `.env.example` contains placeholders only.
- [ ] Logs show startup without secrets.

---

# Phase 3: HMAC Security

Goal: Implement request verification and callback signing.

## Expected Headers

```text
X-WPAB-Site-ID
X-WPAB-Timestamp
X-WPAB-Signature
```

## Checklist

- [ ] Add site secret lookup.
- [ ] Add timestamp tolerance verification.
- [ ] Add request signing payload format.
- [ ] Add signature generation helper.
- [ ] Add signature verification helper.
- [ ] Sign `timestamp + "\n" + site_id + "\n" + raw_json_body`.
- [ ] Preserve the exact raw JSON body used for signing.
- [ ] Reject unknown site IDs.
- [ ] Reject stale timestamps.
- [ ] Reject invalid signatures.
- [ ] Add tests for valid and invalid signatures.

## Verification

- [ ] Valid signed request passes.
- [ ] Invalid signature fails.
- [ ] Unknown site fails.
- [ ] Expired timestamp fails.
- [ ] Signature tests pass.

---

# Phase 4: API Contracts

Goal: Establish worker API endpoints and payload models.

## Expected Endpoints

```text
GET /health
POST /v1/jobs/transcribe
GET /v1/jobs/{job_id}
```

## Checklist

- [ ] Define Pydantic model for job submission.
- [ ] Define model for accepted response.
- [ ] Define model for job status response.
- [ ] Define model for callback success payload.
- [ ] Define model for callback failure payload.
- [ ] Implement `POST /v1/jobs/transcribe` as enqueue-only behavior.
- [ ] Implement `GET /v1/jobs/{job_id}` for short-term status if available.
- [ ] Use consistent response envelopes.
- [ ] Use consistent error envelopes.

## Response Envelopes

Success:

```json
{ "ok": true, "data": {} }
```

Failure:

```json
{ "ok": false, "error": { "code": "ERROR_CODE", "message": "User-safe message." } }
```

## Verification

- [ ] Health endpoint works.
- [ ] Invalid payload returns structured error.
- [ ] Valid signed payload returns accepted response.
- [ ] API does not run long processing inline.

---

# Phase 5: Redis Queue and Worker Process

Goal: Move transcription work into a background worker.

## Checklist

- [ ] Add Redis connection config.
- [ ] Add queue abstraction/service.
- [ ] Add job enqueue behavior.
- [ ] Add worker process entrypoint.
- [ ] Add active job status tracking.
- [ ] Add basic job lifecycle states.
- [ ] Add bounded retry behavior.
- [ ] Add Docker Compose Redis service if not present.
- [ ] Document how to run API and worker separately.

## Expected Worker States

```text
accepted
queued
running
callback_pending
completed
failed
expired
```

## Verification

- [ ] Job is enqueued.
- [ ] Worker can consume a test job.
- [ ] Job status updates in Redis.
- [ ] API and worker can run separately.
- [ ] Redis outage produces clear failure.

---

# Phase 6: Audio Download Service

Goal: Download source audio from short-lived signed WordPress URLs safely.

## Checklist

- [ ] Add `audio_download_service.py`.
- [ ] Use explicit request timeout.
- [ ] Validate URL shape.
- [ ] Validate response status.
- [ ] Enforce max content length when available.
- [ ] Enforce max downloaded bytes.
- [ ] Validate content type or extension when practical.
- [ ] Stream download to temp file.
- [ ] Fetch signed URLs exactly as provided without cookies or WordPress auth headers.
- [ ] Treat 403/404 signed URL responses as normal job failures with safe diagnostics.
- [ ] Store only in configured temp directory.
- [ ] Return structured download result.
- [ ] Normalize errors.

## Verification

- [ ] Valid test URL downloads to temp storage.
- [ ] Oversized file is rejected.
- [ ] Timeout is handled.
- [ ] Invalid URL is rejected.
- [ ] Partial failed download is cleaned up.

---

# Phase 7: Audio Chunking Service

Goal: Split large audio files into processable chunks.

## Checklist

- [ ] Decide chunking tool/library.
- [ ] Add `chunking_service.py`.
- [ ] Preserve chunk order.
- [ ] Enforce max chunk size/duration.
- [ ] Generate predictable temp chunk filenames.
- [ ] Return structured chunk metadata.
- [ ] Handle unsupported audio formats.
- [ ] Clean chunks on failure.
- [ ] Document system dependency if external tool such as ffmpeg is required.

## Verification

- [ ] Test audio file chunks successfully.
- [ ] Chunk order is preserved.
- [ ] Unsupported format fails clearly.
- [ ] Chunk files are cleaned when appropriate.
- [ ] Docker image includes required system tool if used.

---

# Phase 8: OpenAI Transcription Service

Goal: Transcribe audio chunks using OpenAI through a controlled wrapper.

## Checklist

- [ ] Add OpenAI client/wrapper.
- [ ] Load API key from config.
- [ ] Add transcription service.
- [ ] Send individual chunks to OpenAI.
- [ ] Normalize provider errors.
- [ ] Retry transient errors only.
- [ ] Enforce timeout.
- [ ] Avoid logging API key or raw provider dumps.
- [ ] Return structured chunk transcript result.
- [ ] Record enough metadata for result assembly.

## Verification

- [ ] Test chunk transcribes successfully.
- [ ] Invalid API key fails safely.
- [ ] Transient failure retry is bounded.
- [ ] Provider error is normalized.
- [ ] No secrets appear in logs.

---

# Phase 9: Result Assembly

Goal: Stitch chunk transcripts into a stable final result.

## Checklist

- [ ] Add `result_assembly_service.py`.
- [ ] Sort chunks by original order.
- [ ] Stitch transcript text cleanly.
- [ ] Preserve segment metadata if available.
- [ ] Add result metadata:
  - model
  - chunk count
  - processing duration if available
  - source file metadata if useful
- [ ] Produce callback-ready result object.
- [ ] Handle missing chunk result.
- [ ] Handle partial failure policy.

## Verification

- [ ] Ordered chunks assemble correctly.
- [ ] Missing chunk fails clearly.
- [ ] Result payload matches callback model.
- [ ] Metadata does not expose internals.

---

# Phase 10: Callback Client

Goal: Send signed success or failure callbacks to WordPress.

## Expected WordPress Endpoint

```text
POST /wp-json/wpab/v1/worker-callback
```

The worker receives the callback URL dynamically from the job payload and does not hardcode this route.

## Checklist

- [ ] Add `callback_client.py`.
- [ ] Generate signed callback headers.
- [ ] Send success payload.
- [ ] Send failure payload.
- [ ] Use explicit timeout.
- [ ] Retry transient callback failures.
- [ ] Log callback attempts safely.
- [ ] Log final callback failure safely.
- [ ] Do not assume callback success without response check.
- [ ] Do not log full transcript text by default.

## Verification

- [ ] Signed callback is generated.
- [ ] Callback succeeds against test endpoint.
- [ ] Invalid callback URL fails safely.
- [ ] Callback retry limit works.
- [ ] Final failure is logged.

---

# Phase 11: Cleanup and Resource Limits

Goal: Prevent disk, memory, and queue problems.

## Checklist

- [ ] Add cleanup service.
- [ ] Clean source audio after success.
- [ ] Clean chunks after success.
- [ ] Clean temp files after failure when safe.
- [ ] Add stale file cleanup.
- [ ] Add max job duration.
- [ ] Add max audio bytes.
- [ ] Add queue/concurrency limits.
- [ ] Document cleanup behavior.

## Verification

- [ ] Success cleanup removes temp files.
- [ ] Failure cleanup removes temp files.
- [ ] Stale cleanup works.
- [ ] Oversized jobs fail safely.
- [ ] Concurrency is bounded.

---

# Phase 12: Docker and Deployment Readiness

Goal: Make the worker easy to run locally and deploy to a VPS.

## Target Shape

```text
reverse proxy
→ FastAPI API container
→ Redis
→ worker process/container
→ temp storage
→ OpenAI API
→ signed callback to WordPress
```

## Checklist

- [ ] Create or normalize `Dockerfile`.
- [ ] Create or normalize `docker-compose.yml`.
- [ ] Include API service.
- [ ] Include worker service.
- [ ] Include Redis service.
- [ ] Include temp volume or configured temp directory.
- [ ] Install required audio tools such as ffmpeg if needed.
- [ ] Add healthcheck if practical.
- [ ] Document Traefik/Nginx expectations.
- [ ] Document production environment variables.
- [ ] Ensure Redis is not exposed publicly.

## Verification

- [ ] `docker compose up` starts services.
- [ ] API health endpoint responds.
- [ ] Worker connects to Redis.
- [ ] Test job can be enqueued.
- [ ] Required audio tools are available inside container.
- [ ] Logs are readable.

---

# Phase 13: Testing and Hardening

Goal: Increase confidence and safety.

## Priority Tests

- [ ] HMAC valid signature.
- [ ] HMAC invalid signature.
- [ ] Unknown site rejection.
- [ ] Expired timestamp rejection.
- [ ] Payload validation.
- [ ] Audio download success/failure.
- [ ] Oversized audio rejection.
- [ ] Chunk order preservation.
- [ ] Transcription error normalization.
- [ ] Result assembly.
- [ ] Callback signing.
- [ ] Callback retry limit.
- [ ] Cleanup behavior.

## Manual Smoke Tests

- [ ] Start API.
- [ ] Start Redis.
- [ ] Start worker.
- [ ] Submit signed job.
- [ ] Download test audio.
- [ ] Chunk test audio.
- [ ] Transcribe test audio.
- [ ] Assemble transcript.
- [ ] Send callback to test WordPress endpoint.
- [ ] Confirm cleanup.

---

# Phase 14: Documentation and Handoff

Goal: Make the repo understandable to future agents and developers.

## Checklist

- [ ] Update `README.md`.
- [ ] Update `ARCHITECTURE.md` with actual final flows.
- [ ] Update `DECISIONS.md` with any new decisions.
- [ ] Update `PROJECT_RULES.md` if repo-specific rules changed.
- [ ] Add local development instructions.
- [ ] Add Docker instructions.
- [ ] Add environment variable reference.
- [ ] Add API contract docs.
- [ ] Add HMAC signing docs.
- [ ] Add callback payload docs.
- [ ] Add troubleshooting section.
- [ ] Add known limitations.
- [ ] Add next roadmap section.

---

# Progress Tracker

- [x] Phase 0: Repository Baseline
- [x] Phase 1: Normalize Worker Structure
- [x] Phase 2: Configuration and Logging
- [x] Phase 3: HMAC Security
- [x] Phase 4: API Contracts
- [x] Phase 5: Redis Queue and Worker Process
- [x] Phase 6: Audio Download Service
- [x] Phase 7: Audio Chunking Service
- [x] Phase 8: OpenAI Transcription Service
- [x] Phase 9: Result Assembly
- [x] Phase 10: Callback Client
- [x] Phase 11: Cleanup and Resource Limits
- [x] Phase 12: Docker and Deployment Readiness
- [ ] Phase 13: Testing and Hardening
- [ ] Phase 14: Documentation and Handoff

## Next Recommended Step

Phase 13: Testing and Hardening — start the worker stack with Docker Compose, submit a test job, verify the full pipeline, and harden edge cases.

Do not implement transcription until configuration, security, API contracts, and queue behavior are clear.

---

# Maintenance Rule

Update this implementation plan when a phase is completed, skipped, expanded, or reordered.

Do not let the checklist drift away from reality.
