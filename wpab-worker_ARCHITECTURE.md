# ARCHITECTURE.md

Version: 1.0  
Project: WP AB Worker  
Repository: `wpab-worker`  
System Type: Backend Worker Service

---

# Purpose

WP AB Worker is a narrow backend processing service for WP Audio Buddy.

Its primary job is to handle heavy audio-processing work that should not run inside WordPress, especially large audio chunking and transcription.

The worker is not the product core. WP Audio Buddy remains the primary product and durable source of truth.

---

# 1. Project Identity

## Project Name

WP AB Worker

## One-Sentence Summary

WP AB Worker is a backend worker service for WP Audio Buddy that processes large audio files outside WordPress and returns transcription results back to the originating WordPress site.

## Primary Audience

The worker is not directly used by end users.

Primary technical users:

- WP Audio Buddy plugin
- site administrators indirectly through plugin settings
- internal Adriel Partners infrastructure
- developers maintaining the WP Audio Buddy system

## Core Problem

Large audio files can exceed normal WordPress hosting limits for request time, memory, CPU, file handling, and API processing.

WordPress should orchestrate audio jobs but should not perform heavy media processing when that risks site stability.

## Core Value

The worker allows WP Audio Buddy to process long or large audio files safely without exhausting WordPress hosting resources.

---

# 2. System Type

## Classification

Backend Worker Service

## Why This Classification Is Correct

WP AB Worker is not a standalone user-facing application and not a WordPress plugin.

It is a trusted backend service that receives signed processing jobs from WP Audio Buddy, processes audio, and sends signed callbacks to WordPress.

The worker owns temporary processing. It does not own final transcripts, user-facing settings, attachment relationships, or durable plugin data.

---

# 3. Product Scope

## Version One Goals

- Accept signed transcription job requests from WP Audio Buddy.
- Validate site identity and request signatures.
- Download audio from WordPress using short-lived signed URLs.
- Chunk large audio files into processable segments.
- Send chunks to OpenAI transcription.
- Stitch transcript output into a coherent result.
- Send a signed success or failure callback to WordPress.
- Clean up temporary files.
- Expose a health endpoint.
- Log failures clearly enough for diagnosis.

## Explicit Non-Goals

- No public user interface.
- No SaaS dashboard.
- No billing.
- No user account system.
- No permanent transcript storage.
- No WordPress attachment ownership.
- No plugin settings ownership.
- No durable product database in v1.
- No multi-customer self-service onboarding in v1.
- No analytics dashboard.
- No editing interface.

## Success Criteria

v1 is successful when:

- WordPress can securely submit a large audio transcription job.
- The worker can download, chunk, transcribe, stitch, and callback successfully.
- Temporary files are cleaned up.
- Failures are visible in logs and communicated back to WordPress when possible.
- WordPress remains the durable source of truth.

---

# 4. Core Technology Stack

## Server

Recommended:

- Python
- FastAPI
- Uvicorn or equivalent ASGI runtime

## Queue / Background Processing

Recommended:

- Redis
- RQ or equivalent lightweight Python job queue

## AI Services

- OpenAI transcription API

## Infrastructure

- Docker
- Docker Compose
- Traefik or Nginx reverse proxy
- environment-based configuration

## Storage

v1 should avoid durable worker database storage.

Use:

- Redis for active queue and short-term job status
- local container or mounted temp storage for transient audio chunks
- logs for operational diagnosis

## Deviations From Standard Constitution

This service intentionally deviates from the default Nuxt/Vue application stack.

Reason:

It is a backend worker, not a user-facing web application.

The worker should be simple, small, and optimized for audio-processing tasks.

---

# 5. Hosting and Portability

## Hosting Model

The worker runs as a Dockerized backend service.

Expected environments:

- VPS with Docker
- local Docker Compose development
- reverse proxy behind Traefik or Nginx

## Portability Requirement

The worker should remain portable across standard Docker-capable VPS environments.

It must not depend on vendor-specific infrastructure.

## Infrastructure Constraints

Assume the worker may run on modest VPS infrastructure.

The service must:

- limit concurrency
- avoid unbounded memory usage
- clean up temp files
- avoid unbounded queue growth
- expose health status
- fail safely

---

# 6. Domain Model

## Worker Job

- Represents a processing request received from WP Audio Buddy.
- Durable: no, not in v1
- Stored in: Redis while active
- Owned by: worker during active processing
- WordPress owns the durable job record.

## Source Audio

- Represents the audio file downloaded from WordPress.
- Durable: no
- Stored in: temporary worker storage
- Owned by: worker only during processing
- Must be deleted after processing or failure cleanup.

## Audio Chunk

- Represents a segment of the original audio file.
- Durable: no
- Stored in: temporary worker storage
- Owned by: worker only during processing
- Must be deleted after processing or failure cleanup.

## Transcript Result

- Represents the final stitched transcript result.
- Durable: no in worker
- Stored in: memory/temp until callback
- Owned durably by: WP Audio Buddy after callback

## Site Connection

- Represents an allowed WordPress site that may submit jobs.
- Durable: yes, as configuration
- Stored in: worker environment configuration or config file
- Owned by: worker configuration
- Includes site ID and shared secret.

## Callback Result

- Represents the signed success or failure payload sent to WordPress.
- Durable: no in worker
- Stored in: logs only
- Owned durably by: WordPress after accepted callback

---

# 7. System Layers

## Actual Flow

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

## API Layer

The API layer may:

- expose health endpoint
- receive job requests
- validate request shape
- verify HMAC signatures
- enqueue jobs
- expose temporary job status if needed

The API layer must not:

- perform long-running transcription inline
- contain chunking logic
- call OpenAI directly
- store durable product data

## Job Service

The Job Service may:

- create internal job records in Redis
- enqueue processing jobs
- update short-term job status
- coordinate worker execution

The Job Service must not:

- own WordPress job history
- become a durable product database
- replace WordPress job records

## Audio Download Service

The Audio Download Service may:

- download source audio from a signed URL
- validate file size and content type
- enforce timeouts
- write to temporary storage
- fetch signed WordPress download URLs exactly as provided, without cookies or WordPress auth headers

The Audio Download Service must not:

- trust arbitrary URLs without validation
- expose downloaded files publicly
- keep files after cleanup
- log full signed URLs

## Chunking Service

The Chunking Service may:

- split audio into acceptable segments
- preserve ordering metadata
- produce chunk file paths
- enforce max duration or file size rules

The Chunking Service must not:

- make product decisions about transcripts
- write final results to WordPress directly

## Transcription Service

The Transcription Service may:

- call OpenAI transcription models
- transcribe chunks
- return structured chunk results
- handle retryable OpenAI failures

The Transcription Service must not:

- own OpenAI key rotation UI
- expose raw API errors to WordPress users
- silently fail

## Result Assembly Service

The Result Assembly Service may:

- stitch chunks in order
- normalize transcript text
- preserve segment metadata if available
- produce callback-ready result payload

## Callback Client

The Callback Client may:

- send signed success callbacks
- send signed failure callbacks
- retry transient callback failures
- log callback outcomes

The Callback Client must not:

- send unsigned callbacks
- expose secrets
- assume callback success without checking response

## Cleanup Service

The Cleanup Service may:

- remove temporary source files
- remove chunk files
- clean stale jobs
- limit disk growth

Cleanup must happen after success and after failure when possible.

---

# 8. Folder Structure

The final folder structure should be documented here once the repo is normalized.

Recommended shape:

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

## Folder Responsibilities

- `app/main.py` - FastAPI app creation and bootstrapping
- `app/api` - HTTP route definitions
- `app/core` - config, logging, security, shared infrastructure
- `app/services` - business and processing services
- `app/workers` - queue worker entrypoints
- `app/models` - typed request/result models
- `app/tests` - automated tests
- `scripts` - operational scripts
- `docker` - Docker-specific supporting files if needed

Avoid vague folders such as `helpers`, `misc`, `stuff`, or `temp`.

Temporary audio files should live in a configured runtime temp directory, not in source-controlled project folders.

---

# 9. Request and Data Flows

## Job Submission Flow

```text
WP Audio Buddy creates local job
→ WordPress sends signed POST /v1/jobs/transcribe
→ Worker verifies site ID, timestamp, and HMAC signature
→ Worker validates payload
→ Worker enqueues processing job
→ Worker returns accepted status to WordPress
```

## Processing Flow

```text
Queued job starts
→ Worker downloads audio from signed URL
→ Worker validates file constraints
→ Worker chunks audio
→ Worker transcribes each chunk
→ Worker stitches transcript
→ Worker sends signed success callback to WordPress
→ Worker cleans up temporary files
```

## Failure Flow

```text
Failure occurs
→ Worker records technical log context
→ Worker attempts signed failure callback to WordPress
→ Worker cleans up temporary files
→ WordPress marks durable job failed
```

## Stale Job Cleanup Flow

```text
Scheduled cleanup runs
→ Find old temp files and stale active jobs
→ Remove expired files
→ Log cleanup summary
```

---

# 10. Authentication and Authorization

## Does This System Have Accounts?

No.

The worker does not have a user account system in v1.

## Authentication Method

HMAC-signed service-to-service requests.

Expected headers:

```text
X-WPAB-Site-ID
X-WPAB-Timestamp
X-WPAB-Signature
```

The signature is computed over the exact timestamp/site/body payload:

```text
TIMESTAMP + "\n" + SITE_ID + "\n" + RAW_JSON_BODY
```

## Site Authorization

The worker must only accept requests from configured site IDs.

Each site ID maps to a shared secret.

In v1, this may be stored in environment configuration.

Future versions may use a config file or database if multi-site management becomes complex.

## Authorization Boundary

Authorization happens before job creation.

Unsigned or invalid requests must be rejected before any audio URL is fetched or any job is enqueued.

---

# 11. Validation Strategy

## Boundary Validation

Validate:

- site ID
- timestamp freshness
- HMAC signature
- callback URL
- audio download URL
- job ID
- attachment ID
- requested operation
- file size limits
- allowed audio formats
- model names
- processing options

## Validation Tooling

Recommended:

- Pydantic models for request and response payloads
- explicit allowlists for modes and model names
- strict URL validation
- max file size settings
- timeout settings

## Validation Rule

The API validates shape, signature, and authorization.

Services validate processing rules.

OpenAI and download service wrappers validate external responses.

---

# 12. Error Handling Pattern

## API Error Envelope

Use a consistent error shape.

Example:

```json
{
  "ok": false,
  "error": {
    "code": "INVALID_SIGNATURE",
    "message": "Invalid request signature."
  }
}
```

## Success Envelope

Example:

```json
{
  "ok": true,
  "data": {
    "job_id": "wpab_123",
    "status": "accepted"
  }
}
```

## Error Codes

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

## Client Safety Rule

Do not return:

- stack traces
- OpenAI API details
- secrets
- internal file paths
- full signed URLs
- raw provider error dumps

---

# 13. Background Jobs and Async Processing

## Do We Use Background Jobs?

Yes.

## What Runs Async?

- audio download
- audio chunking
- transcription
- transcript assembly
- callback retry
- cleanup

## Job Tooling

Recommended:

- Redis
- RQ or equivalent lightweight Python worker queue

## Retry Strategy

Suggested defaults:

- audio download timeout: retry up to 2 times
- transient OpenAI error: retry up to 2 times
- callback network failure: retry up to 3 times
- invalid signature: no retry
- unsupported file type: no retry
- file too large beyond configured hard limit: no retry
- invalid callback URL: no retry

## Job Ownership

Worker owns active processing state only.

WordPress owns durable processing job history.

---

# 14. External Services and Integrations

## WP Audio Buddy

- Purpose: source of job requests and destination for final callbacks
- Called from: Callback Client
- Failure behavior: retry transient callback failures; log final failure
- Critical: high

## OpenAI

- Purpose: audio transcription
- Called from: Transcription Service
- Failure behavior: retry transient failures, fail job on unrecoverable errors
- Critical: high

## Redis

- Purpose: queue and short-term active job state
- Called from: Job Service and worker process
- Failure behavior: worker cannot accept/process jobs if unavailable
- Critical: high

## Reverse Proxy

- Purpose: expose worker API over HTTPS
- Expected: Traefik or Nginx
- Failure behavior: WordPress cannot submit jobs if unavailable
- Critical: high

---

# 15. Design System and Visual Identity

This project has no user-facing design system in v1.

If a minimal admin/status UI is ever added, it must be documented before implementation.

Do not add a frontend framework or UI layer unless there is a clear need.

---

# 16. Testing Strategy

## Test Runner

Recommended:

- pytest

## Priority Test Targets

- HMAC signing and verification
- timestamp expiration
- site authorization
- payload validation
- audio download failure handling
- chunk ordering
- transcript assembly
- callback signing
- callback retry behavior
- cleanup behavior

## Things We Intentionally Will Not Over-Test

- FastAPI framework internals
- simple config accessors
- trivial models without behavior
- cosmetic log formatting

---

# 17. Browser and Device Support

Not applicable.

This is a backend service with no browser UI in v1.

---

# 18. Performance Strategy

## Performance Priorities

- bounded memory usage
- bounded disk usage
- predictable job concurrency
- safe temp file cleanup
- timeouts for external requests
- reasonable chunk sizes
- no long-running work in API request cycle

## Known Bottlenecks or Risks

- large audio files
- slow WordPress file downloads
- OpenAI API latency
- temporary disk growth
- Redis outage
- reverse proxy timeout
- low-memory VPS

## Caching Strategy

No caching in v1.

---

# 19. Observability and Monitoring

## Logging

Use structured logs.

Log:

- job accepted
- job started
- audio download started/completed
- chunking started/completed
- transcription chunk started/completed
- result assembly completed
- callback sent
- callback failed
- cleanup completed
- job failed

Never log:

- OpenAI API keys
- shared secrets
- full signed URLs
- raw audio content
- sensitive user data

## Monitoring

v1 expected monitoring:

- `/health` endpoint
- container logs
- reverse proxy logs
- Redis availability
- disk usage

## Alerting

Not required in v1.

Future options:

- uptime monitor for `/health`
- disk usage alerts
- repeated job failure alerts
- callback failure alerts

---

# 20. Deployment Architecture

## Local Development

```text
Docker Compose
→ FastAPI API container
→ Redis container
→ Worker process/container
→ local temp volume
→ OpenAI API
```

## Production

```text
WordPress Site
→ HTTPS reverse proxy
→ wpab-worker API container
→ Redis
→ worker process
→ temp storage
→ OpenAI API
→ signed callback to WordPress
```

## Production Notes

- Run API and worker as separate processes or containers.
- Use environment variables for configuration.
- Persist only what must be persisted.
- Temp storage should be cleaned aggressively.
- Redis should be protected from public access.
- Worker API must be HTTPS-only in production.

---

# 21. Environment Configuration

Recommended environment variables:

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

## Secret Values

Secrets include:

- `WPAB_OPENAI_API_KEY`
- site shared secrets
- any callback signing secrets

Secrets must never be committed.

---

# 22. Data Durability and Backup Strategy

## Durable Data

The worker should not own durable product data in v1.

Durable transcript and job data lives in WordPress.

## Backup Strategy

No worker database backup required in v1.

If future durable worker storage is added, this architecture document must be updated before implementation.

## Deletion Policy

Temporary files must be deleted after processing and during cleanup.

Failed jobs should not leave large files on disk indefinitely.

---

# 23. Worker-Specific Architecture

## Worker Role

The worker is a processing helper.

It may:

- download audio
- chunk audio
- transcribe audio
- assemble transcript results
- send callbacks
- clean temporary files

It must not:

- own final transcripts
- own WordPress attachment relationships
- own user-facing plugin settings
- implement WordPress admin features
- become a SaaS dashboard
- become a durable product database without an explicit architectural decision

## WordPress Boundary

WordPress sends jobs and receives results.

WordPress remains responsible for:

- admin experience
- job history
- final transcript storage
- user-safe failure display
- retries initiated by admin
- attachment relationships

---

# 24. Architectural Decisions

Maintain a companion file:

```text
DECISIONS.md
```

Initial decisions to record:

1. WP AB Worker is a backend worker service, not a product app.
2. WordPress owns durable transcript and job data.
3. Worker owns temporary heavy processing only.
4. Worker owns OpenAI key in the recommended deployment model.
5. HMAC signatures are required for all WordPress-worker communication.
6. Worker uses Redis for active processing state in v1.
7. Worker does not use a durable database in v1.

---

# 25. Implementation Readiness Checklist

Before major coding continues, confirm:

- FastAPI app structure
- queue tooling
- Redis configuration
- exact HMAC signing payload
- allowed site configuration format
- audio download URL format
- max file size default
- chunking implementation
- OpenAI transcription model
- callback payload format
- callback retry rules
- cleanup schedule
- deployment shape

---

# 26. Maintenance Rule

Update this document when:

- queue tooling changes
- callback format changes
- HMAC signing changes
- allowed site configuration changes
- OpenAI model strategy changes
- chunking logic changes
- worker starts storing durable data
- deployment architecture changes
- new external integrations are added

An outdated architecture document is worse than none.

---

# Final Principle

WP AB Worker should stay narrow.

It exists to make WP Audio Buddy safer and more capable, not to become a second application core.

When in doubt, keep durable product data in WordPress, keep the worker stateless where practical, and keep every heavy processing step bounded, logged, signed, and cleanup-aware.
