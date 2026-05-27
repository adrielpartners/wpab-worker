# DECISIONS.md

Version: 1.0  
Project: WP AB Worker  
Repository: `wpab-worker`  
Last Updated: 2026-05-25

---

# Purpose

This file records major architectural and product decisions for WP AB Worker.

Use this file to prevent future developers or AI agents from turning the worker into a second product core, durable database, or unnecessary SaaS platform.

Each decision should include:

- decision
- rationale
- tradeoffs
- date adopted
- reversibility

---

# Decision 001: WP AB Worker is a backend processing service

## Decision

WP AB Worker is a backend worker service for WP Audio Buddy. It is not a user-facing application and not a product dashboard.

## Rationale

The worker exists to process large/heavy audio jobs that should not run inside WordPress. Its role is narrow: receive signed jobs, process audio, and return results.

WP Audio Buddy remains the product.

## Tradeoffs

- No built-in user interface in v1.
- Operational visibility depends on logs, health endpoints, and WordPress-side job status.
- Any future dashboard must be justified separately.

## Date Adopted

2026-05-25

## Reversibility

Moderate. A dashboard could be added later, but that would be a new product decision.

---

# Decision 002: WordPress remains the durable source of truth

## Decision

The worker must not own final transcripts, summaries, excerpts, generated outputs, or durable product job history.

WordPress owns durable data.

## Rationale

The worker is infrastructure. WordPress owns the plugin experience, attachment relationships, admin workflow, and final generated content.

Keeping durable data in WordPress avoids split-brain data ownership and simplifies backup/recovery.

## Tradeoffs

- The worker must send complete callback results to WordPress.
- Callback payloads and result sizes must be handled carefully.
- The worker cannot be used independently as a transcript archive in v1.

## Date Adopted

2026-05-25

## Reversibility

Difficult after launch. A future cloud data layer could be added, but it should not be introduced casually.

---

# Decision 003: Worker owns temporary processing state only

## Decision

The worker may store active job state, downloaded files, audio chunks, and temporary transcript results only while processing.

It should clean these up after success, failure, or expiration.

## Rationale

Temporary state is necessary for processing. Durable product state belongs in WordPress.

This keeps the worker simple and reduces backup, privacy, and storage complexity.

## Tradeoffs

- If WordPress callback fails permanently, the worker does not preserve the final result indefinitely.
- Logs must be good enough to diagnose failures.
- Retry behavior must happen before cleanup or be coordinated carefully.

## Date Adopted

2026-05-25

## Reversibility

Moderate. Durable worker storage could be added later if necessary.

---

# Decision 004: No durable worker database in v1

## Decision

Do not add PostgreSQL, MySQL, SQLite, or another durable worker database in v1.

Use Redis for queue/active state and local temporary storage for processing files.

## Rationale

A durable database would add operational complexity before it is needed. The worker does not own final product data.

v1 should remain small, deployable, and easy to reason about.

## Tradeoffs

- Worker-side historical reporting is limited.
- Completed job history lives in WordPress, not the worker.
- Some debugging depends on logs.
- Failed callback recovery windows are shorter unless explicitly handled.

## Date Adopted

2026-05-25

## Reversibility

Easy to moderate. A database can be added later if durable multi-site worker history becomes necessary.

---

# Decision 005: Use Redis for queue and active job state

## Decision

Use Redis for background queueing and active job state in v1.

Recommended worker queue tooling:

```text
Redis + RQ
```

or an equivalent lightweight Python queue if later justified.

## Rationale

Audio processing should not run inside the API request cycle. Redis-backed queues are simple, proven, and suitable for this worker’s needs.

## Tradeoffs

- Requires a Redis service.
- Redis availability becomes critical for processing.
- Requires operational safeguards around queue growth and worker concurrency.

## Date Adopted

2026-05-25

## Reversibility

Moderate. Another queue system could replace it later, but all job handling would need to be updated.

---

# Decision 006: Use FastAPI as the worker API framework

## Decision

Use FastAPI as the HTTP API layer for worker endpoints.

Expected endpoints include:

```text
GET /health
POST /v1/jobs/transcribe
GET /v1/jobs/{job_id}
```

## Rationale

FastAPI is a strong fit for a small Python service with typed request validation, clean JSON APIs, and async-friendly request handling.

The worker is backend-only, so a lightweight API framework is appropriate.

## Tradeoffs

- Requires Python runtime and dependency management.
- Developers/agents must not treat the worker as a full web app.
- FastAPI should remain thin and not absorb business logic into route handlers.

## Date Adopted

2026-05-25

## Reversibility

Moderate. A different Python API framework could be used, but FastAPI is the preferred default.

---

# Decision 007: Worker owns the OpenAI key in worker mode

## Decision

When jobs are processed by the worker, the worker uses its own OpenAI API key from environment configuration.

WordPress should not send an OpenAI API key to the worker in job payloads.

## Rationale

This reduces the chance of leaking API keys through request payloads, logs, queues, or callback failures.

It also makes worker-mode configuration cleaner for sites that should not individually manage OpenAI credentials.

## Tradeoffs

- The worker becomes a trusted service.
- OpenAI usage is centralized under the worker deployment.
- If used across multiple sites, usage tracking and quotas may later be needed.

## Date Adopted

2026-05-25

## Reversibility

Easy. The worker could later support per-site keys if needed, but that should be a deliberate design change.

---

# Decision 008: Worker downloads audio from short-lived signed WordPress URLs

## Decision

The worker should receive a short-lived signed download URL for the source audio and download the audio directly.

The initial job request should not upload large audio files directly.

## Rationale

Large file uploads through the job request are fragile and can trigger request timeouts, memory limits, and reverse proxy limits.

Signed URLs keep access controlled while allowing the worker to pull the file efficiently.

## Tradeoffs

- WordPress must generate signed temporary URLs.
- The worker must handle download failure, expiration, and timeout.
- The worker must fetch the signed URL exactly as provided and must not add WordPress cookies or authentication headers.
- Some hosting environments may require special handling for protected files.

## Date Adopted

2026-05-25

## Reversibility

Easy. Direct upload could be supported later as an alternate mode.

---

# Decision 009: All WordPress-worker communication requires HMAC signing

## Decision

The worker must verify signed requests from WordPress and must sign callbacks to WordPress.

Expected headers:

```text
X-WPAB-Site-ID
X-WPAB-Timestamp
X-WPAB-Signature
```

## Rationale

The worker accepts processing jobs and sends content back into WordPress. Unsigned requests or callbacks would allow job abuse, forged results, or unauthorized updates.

## Tradeoffs

- Requires shared secret configuration.
- Requires timestamp tolerance.
- Requires consistent signing implementation in both repos.
- Requires careful testing.

## Date Adopted

2026-05-25

## Reversibility

Should not be reversed. Only the exact signing format may evolve.

---

# Decision 010: Use configured allowed site IDs and secrets

## Decision

The worker should only accept jobs from configured site IDs. Each site ID maps to a shared secret.

In v1, this configuration may live in environment variables or a simple config file.

## Rationale

The worker should not be open to arbitrary WordPress sites. Explicit site authorization keeps the deployment simple and secure.

## Tradeoffs

- Adding a new site requires configuration changes.
- A future dashboard or database may be needed for easier multi-site management.
- Environment variable formatting must be documented clearly.

## Date Adopted

2026-05-25

## Reversibility

Easy. Configuration can be moved to a database later if needed.

---

# Decision 011: Worker sends signed callbacks to WordPress

## Decision

The worker reports success or failure by sending a signed callback to a WordPress REST endpoint.

Expected WordPress endpoint:

```text
POST /wp-json/wpab/v1/worker-callback
```

The worker receives the callback URL dynamically in the signed job payload and does not hardcode this route.

## Rationale

Callbacks allow WordPress to own durable job status and transcript storage without polling the worker constantly.

Signed callbacks protect against forged job updates.

## Tradeoffs

- WordPress must expose a secure callback endpoint.
- Callback failures need retry rules.
- Callback payload format must be stable and documented.
- Large results may require payload size planning.

## Date Adopted

2026-05-25

## Reversibility

Moderate. Polling could be added later, but callbacks are the v1 default.

---

# Decision 012: Callback failures are retried but not stored forever

## Decision

Transient callback failures should be retried, but the worker should not keep completed results forever.

Recommended default:

- retry callback network failures up to 3 times
- log final failure
- cleanup according to retention policy

## Rationale

The worker is not a durable result archive. Keeping results forever would create privacy, storage, and backup obligations.

## Tradeoffs

- A permanently failed callback may require rerunning the job from WordPress.
- Logs become important for diagnosis.
- Cleanup timing must allow reasonable retry attempts.

## Date Adopted

2026-05-25

## Reversibility

Moderate. A future durable outbox could be added if needed.

---

# Decision 013: Temporary files must be cleaned aggressively

## Decision

Downloaded audio files, chunks, and temporary transcript assembly files must be removed after processing or failure handling.

A cleanup process must also remove stale files.

## Rationale

Audio files can be large. Without cleanup, disk usage could grow quickly and destabilize the VPS.

Temporary audio may also contain sensitive content, so it should not linger.

## Tradeoffs

- Debugging failed jobs may be harder after cleanup.
- Logs must capture enough detail before files are deleted.
- Cleanup must avoid deleting active files.

## Date Adopted

2026-05-25

## Reversibility

Should not be reversed. Retention windows can be adjusted.

---

# Decision 014: Keep the worker API tiny

## Decision

The worker API should remain minimal.

Expected v1 endpoints:

```text
GET /health
POST /v1/jobs/transcribe
GET /v1/jobs/{job_id}
```

No admin dashboard, no public UI, no account system, and no unrelated endpoints in v1.

## Rationale

A small API keeps the worker safer, easier to test, and easier for agents to maintain.

The worker’s job is processing, not product management.

## Tradeoffs

- Less operational convenience in v1.
- Some debugging may require logs or CLI access.
- Future admin tools will need a new decision.

## Date Adopted

2026-05-25

## Reversibility

Easy. Endpoints can be added when justified.

---

# Decision 015: Run API and worker as separate processes or containers

## Decision

The API server and the background worker process should run separately.

Recommended production shape:

```text
reverse proxy
→ FastAPI API container
→ Redis
→ worker process/container
```

## Rationale

API requests should return quickly. Long-running transcription should happen in a background worker.

Separate processes make scaling, restarting, and monitoring cleaner.

## Tradeoffs

- Deployment requires more than one running process/container.
- Docker Compose configuration must be clear.
- Logs may come from multiple containers.

## Date Adopted

2026-05-25

## Reversibility

Easy for development, but production should keep the separation.

---

# Decision 016: Limit concurrency to protect small VPS environments

## Decision

The worker must use explicit concurrency limits.

Do not allow unbounded simultaneous audio jobs.

## Rationale

The expected deployment may run on modest VPS infrastructure. Audio processing can consume CPU, memory, disk, and network resources.

Concurrency must be controlled to prevent crashes or degraded service.

## Tradeoffs

- Large job queues may take longer to process.
- Multiple sites may compete for limited worker capacity.
- Future quota/prioritization logic may be needed if usage grows.

## Date Adopted

2026-05-25

## Reversibility

Easy. Limits can be tuned over time.

---

# Decision 017: Use structured logs, not a database dashboard, for v1 observability

## Decision

Use structured logs, `/health`, and WordPress-side job visibility for v1 observability.

Do not build a worker dashboard in v1.

## Rationale

The worker should stay small. Most durable job visibility belongs in WordPress.

Structured logs are enough for early development and operations.

## Tradeoffs

- Debugging may require server access.
- No built-in historical dashboard.
- Repeated failure analysis may be less convenient until tooling improves.

## Date Adopted

2026-05-25

## Reversibility

Easy. A dashboard or log viewer can be added later if justified.

---

# Decision 018: Maintain separate architecture docs for worker and plugin

## Decision

The worker and plugin each maintain their own `ARCHITECTURE.md`, `PROJECT_RULES.md`, and `DECISIONS.md`.

## Rationale

The two repos have different responsibilities and stacks.

Separate docs prevent agents from accidentally applying WordPress-native plugin rules to the worker or backend-worker rules to the WordPress plugin.

## Tradeoffs

- Shared integration decisions appear in both repos.
- Docs must be kept aligned.
- Integration changes require updates in both places.

## Date Adopted

2026-05-25

## Reversibility

Easy, but not recommended.

|---

# Decision 019: Modularize worker into layered service architecture

## Decision

Restructured the worker from flat files (`server.py`, `jobs.py`, `hmac_util.py`, `cleanup.py`) into a layered Python package with 15 modules across 7 directories:

- `app/core/` — config, logging, HMAC security
- `app/models/` — Pydantic request/response payloads
- `app/api/` — FastAPI routes (health, job submission, job status)
- `app/services/` — audio download, chunking, transcription, result assembly, callback, cleanup, job orchestration
- `app/workers/` — RQ worker entrypoint, cleanup daemon
- `app/main.py` — FastAPI app bootstrap

## Rationale

The flat file structure made reasoning about the code difficult and would have confused future AI agents. The new structure:

- Separates concerns into clear layers (API → services → workers)
- Centralizes config in `config.py` instead of scattered `os.getenv()` calls
- Uses Pydantic models for typed request/response validation
- Splits the 399-line `jobs.py` into 7 focused service modules
- Adds proper multi-site HMAC verification with timestamp and site ID checks

## Tradeoffs

- More files to navigate, but each file has a single clear responsibility
- Requires Python path setup for imports (handled by `app/` package structure)
- Old flat files (`server.py`, `jobs.py`, `hmac_util.py`, `cleanup.py`) removed

## Date Adopted

2026-05-26

## Reversibility

Moderate. The old flat files can be restored from git history, but all module references would need updating.

---

# Maintenance Rule

Update this file when:

- queue tooling changes
- worker starts owning durable data
- OpenAI key ownership changes
- callback strategy changes
- HMAC signing changes
- allowed site configuration changes
- deployment shape changes
- temp file retention changes
- API endpoints are added or removed

Do not let major decisions live only in chat history.
