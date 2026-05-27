# CODING_CONSTITUTION.md

Version: 1.0  
Purpose: Define the universal engineering principles, architectural defaults, portability standards, design system expectations, and AI collaboration rules used across all software projects.

This document is the global source of truth for how software should be built unless a project-specific `ARCHITECTURE.md` explicitly and intentionally overrides it.

It applies to:

- standalone web applications
- backend services
- hybrid WordPress integrations
- WordPress-native plugins, together with the applicable WordPress mode document
- internal tools
- AI-assisted coding workflows

Project-specific facts belong in `ARCHITECTURE.md`.  
Repository-specific enforcement rules belong in `PROJECT_RULES.md`.  
Major decisions and their reasons belong in `DECISIONS.md`.  
Agent operating instructions belong in `AGENTS.md`.

---

# 1. Core Philosophy

All systems must prioritize:

- clarity over cleverness
- explicit behavior over hidden abstraction
- consistency over personal preference
- minimal dependencies
- long-term maintainability
- architectural integrity
- operational simplicity
- AI-friendly structure
- human-readable code

A system should become more understandable over time, not less.

Readable, boring, predictable code is usually better than clever code.

---

# 2. Enduring Design Principles

Every system should embody three enduring qualities.

## Resilience

The system remains stable under imperfect conditions.

It handles failures intentionally, degrades safely when possible, and avoids silent breakage.

## Longevity

The system can evolve safely over many years.

It uses clear data models, documented decisions, migrations, backups, and simple architecture that future developers and AI agents can understand.

## Elegance

The system remains coherent and understandable as it grows.

Elegance does not mean complexity or trendiness. It means the system feels intentionally designed, visually clear, and structurally disciplined.

When tradeoffs arise, prefer solutions that preserve clarity, stability, portability, and maintainability.

---

# 3. Default Technology Stack

Unless explicitly overridden in `ARCHITECTURE.md`, the default application stack is:

## Frontend

- Nuxt
- Vue 3
- TypeScript

## Server

- Nitro

## Infrastructure

- Docker
- Nginx or an explicitly documented reverse proxy

## Database

- PostgreSQL

## Icons

- SVG-based icons
- Iconify when an icon system is needed

## Package Management

- npm unless the project intentionally chooses another package manager

Any deviation from this standard stack must be documented in `ARCHITECTURE.md`.

A deviation should include:

- what changed
- why it changed
- what benefit it provides
- what tradeoff it introduces
- whether the choice is reversible

---

# 4. Portability and Hosting Doctrine

Default to self-hostable, portable architecture.

Operational freedom is a product feature.

Rules:

- Prefer infrastructure that can run on a standard VPS without managed platform lock-in.
- Prefer Dockerized services and environment-based configuration.
- Avoid hard dependence on vendor-specific runtime features unless explicitly approved.
- Critical systems must be migratable to another host with reasonable effort.
- Do not assume Vercel, Supabase, Firebase, Netlify, or other managed platforms by default.
- When a managed service is used, document why it was chosen, what it replaces, and what migration risk it creates.
- Secrets must never be committed to the repository.
- Configuration must be explicit and environment-driven.

Managed platforms may be used, but they must be chosen intentionally rather than by default.

---

# 5. System Architecture Standard

Default architectural flow:

```text
UI / Page
→ Server Route
→ Service Layer
→ Database
```

Rules:

- UI handles rendering, interaction, local state, and client orchestration only.
- Server routes remain thin request handlers.
- Server routes parse input, perform authorization checks, call services, and return responses.
- Core business logic lives in service modules.
- Database access occurs inside services or repository modules explicitly owned by services.
- UI components must not contain business logic.
- UI components must not directly access the database.
- Route handlers must not become business logic containers.
- Data access should remain readable and predictable.

A project may adapt this flow, but the actual flow must be documented in `ARCHITECTURE.md`.

---

# 6. Service Layer Standard

The service layer is the home of business behavior.

A service should generally:

- accept validated input
- execute business rules
- coordinate data access
- call external systems when needed
- create jobs when work should be asynchronous
- return clear domain results
- throw or return consistent domain errors

Services should not:

- render UI
- know about component state
- emit raw HTML
- contain styling concerns
- become generic dumping grounds
- become vague `utils` collections

Preferred service shape:

- one service file per domain concern when practical
- descriptive names such as `invoice-service.ts`, `auth-service.ts`, or `theme-service.ts`
- exported functions that are readable and narrowly scoped
- shared domain types defined at clear boundaries

Do not invent extra abstraction layers unless there is a real recurring need.

---

# 7. Data and Database Principles

Use PostgreSQL by default for standalone applications and backend systems.

Rules:

- Prefer explicit SQL or a lightweight query layer.
- Avoid heavy ORMs that obscure database behavior unless explicitly justified.
- Schema and migrations must live in the repository.
- Never rely on manual production database edits.
- Data models should remain inspectable and predictable.
- Destructive schema changes must be deliberate and justified.
- Durable data ownership must be clear.
- Data deletion must be deliberate, auditable when appropriate, and recoverable when practical.

Data must remain:

- readable
- versioned
- evolvable
- recoverable

Prefer incremental schema evolution over destructive rebuilds.

---

# 8. Migration Policy

All schema changes must be tracked through migration files.

Rules:

- Do not change production schemas manually.
- Do not depend on undocumented database state.
- Migrations must be committed to the repository.
- Migrations must be understandable.
- Destructive migrations must explain the risk and data impact.
- Rollback or recovery expectations should be documented when the risk is meaningful.

If a project uses WordPress custom tables, migrations or activation/update routines must be documented in `ARCHITECTURE.md`.

---

# 9. Validation and Security

All external input is untrusted.

Rules:

- Validate all request data at the server boundary.
- Validate query parameters, request bodies, form submissions, webhook payloads, file uploads, and admin actions.
- Reject invalid input early.
- Prefer explicit validation schemas.
- Never rely on client-side validation alone.
- Always use parameterized queries.
- Never construct SQL through unsafe string concatenation.
- Sanitize and escape output appropriately for its context.
- Authorization must always be enforced at the server layer.
- Never expose secrets, tokens, stack traces, SQL details, or internal provider errors to clients.

Security is not a final cleanup step. It is part of the architecture.

---

# 10. Authentication and Authorization

Authentication should be modular and role-aware.

Rules:

- Add authentication only when the application requires accounts or protected access.
- Do not introduce auth complexity prematurely.
- Use server-side enforcement for all authorization rules.
- Do not trust client-side role checks.
- Keep authentication decisions separate from visual display logic.
- Define roles and permissions clearly in `ARCHITECTURE.md`.
- Define where authorization checks happen.

Common authorization boundaries include:

- server routes before calling services
- services for domain-level enforcement
- WordPress capability checks for WordPress-native admin actions

If a project has no authentication in version one, say so explicitly in `ARCHITECTURE.md`.

---

# 11. API Design and Response Shape

Use explicit REST-style endpoints by default unless the project intentionally chooses another API model.

API principles:

- clear endpoint names
- predictable request and response structures
- thin handlers
- validated input
- consistent error shape
- user-safe error messages
- machine-readable error codes

Default API error envelope:

```json
{
  "ok": false,
  "error": {
    "code": "SOME_ERROR_CODE",
    "message": "Human-readable message."
  }
}
```

Default success envelope when useful:

```json
{
  "ok": true,
  "data": {}
}
```

Projects may adapt the exact response shape, but the shape must remain consistent within the project and documented in `ARCHITECTURE.md`.

---

# 12. Error Handling Standard

Errors must be handled intentionally and consistently.

Rules:

- Never swallow errors silently.
- Distinguish expected operational errors from programmer mistakes.
- Return user-safe messages to clients.
- Never expose stack traces or internal details to clients.
- Log actionable server-side context for diagnosis.
- Use consistent domain error codes.
- Make failures visible and traceable.

Silent failure is unacceptable.

When uncertainty exists, fail safely rather than pretending success.

---

# 13. Background Processing Standard

Request-response cycles should remain fast and predictable.

Tasks that are slow, heavy, or failure-prone must run outside the request lifecycle.

Common examples:

- sending email
- processing media
- importing data
- exporting data
- webhook processing
- AI processing
- scheduled jobs
- large batch updates
- retry operations

Rules:

- Heavy work must not block user-facing requests.
- Jobs should be idempotent when practical.
- Retries must be explicitly defined.
- Failures must be logged clearly.
- Jobs must never silently fail.
- Job status should be inspectable when job behavior matters to the product.
- Queue, worker, or scheduler strategy must be defined in `ARCHITECTURE.md`.

AI agents must not choose a queue strategy arbitrarily.

---

# 14. Observability and Logging

Applications must be diagnosable in production.

Rules:

- Prefer structured logs for server events.
- Include timestamps and relevant request, user, job, or integration context.
- Avoid noisy debug logs that obscure real failures.
- Never log secrets, tokens, passwords, or sensitive personal data.
- Important failures must be visible, traceable, and diagnosable.
- Logging should support debugging without exposing private information.

For projects with background jobs, job status and job failures must be visible through logs, an admin screen, or another documented mechanism.

---

# 15. Resilience and Stability

Applications should remain stable under imperfect conditions.

Rules:

- Timeouts must exist for external requests.
- External API calls should include retry logic where appropriate.
- Degraded functionality is preferable to total collapse when possible.
- Critical operations should fail safely.
- Integration failures should be isolated when feasible.
- Recovery paths should be intentional, not accidental.
- Repeated failures should not create infinite loops or runaway costs.

Systems should be built with the assumption that networks fail, APIs change, users submit bad data, and infrastructure sometimes behaves badly.

---

# 16. Data Durability and Backups

Applications must assume they may exist for many years.

Rules:

- Production databases must be backed up automatically.
- Backups must be periodically verified.
- Durable data ownership must be clear.
- Data deletion must be intentional.
- Soft deletion should be considered for business-critical records.
- Critical destructive actions should be logged.
- Backup and deletion policies must be documented in `ARCHITECTURE.md`.

A system that cannot recover its important data is not production-ready.

---

# 17. State Management

Default rule:

- Prefer local component state.

Introduce shared state only when clearly necessary.

Avoid large global stores unless application complexity truly requires them.

Shared state is appropriate when:

- multiple distant components need the same state
- state must persist across routes
- the application has complex client-side workflows
- local state creates duplication or inconsistency

Do not add a state management library just because one exists.

---

# 18. Design System Standard

Beauty is a first-class engineering concern.

A beautiful system has:

- clear visual hierarchy
- consistent spacing
- intentional typography
- coherent color usage
- elegant interaction states
- calm, readable interfaces
- understandable structure

Applications must use a token-driven design system.

Rules:

- Use CSS variables for design tokens.
- Primitives implement token usage.
- Feature components compose primitives.
- Theme values must be driven by tokens.
- Components must not hardcode theme logic.
- Do not rely on utility-first styling as the default approach unless explicitly approved.
- Avoid arbitrary one-off styling that fragments the visual system.

Required token categories:

- color
- typography scale
- spacing scale
- radius
- border
- shadow
- z-index
- motion timing

Minimum expectations:

- typography scale must be defined before major feature components are built
- spacing scale must be defined before layout-heavy UI is built
- color tokens must include semantic intent, not only raw palette values
- hover, focus, active, disabled, loading, and error states must be considered explicitly
- responsive behavior must be designed, not improvised

Suggested token locations belong in `ARCHITECTURE.md`.

---

# 19. Visual Identity Checkpoint

Before substantial UI implementation begins, the project must define:

- visual tone
- typography direction
- spacing density
- primary color behavior
- border and radius philosophy
- card, surface, and shadow approach
- interaction feel
- light and dark mode expectations if applicable
- accessibility baseline

AI must not jump directly from idea to arbitrary component styling.

A visual system must exist before large-scale component generation begins.

If the project is intentionally rough or prototype-only, that should be documented.

---

# 20. Component Architecture

Default UI architecture:

```text
Design Tokens
→ UI Primitives
→ Feature Components
→ Pages
```

## UI Primitives

Examples:

- `AppButton`
- `AppCard`
- `AppInput`
- `AppDialog`
- `AppTabs`
- `AppBadge`

Primitives own shared visual styling behavior.

## Feature Components

Feature components compose primitives and implement domain-specific UI behavior.

They must not redefine the visual language independently.

## Pages

Pages orchestrate layout, data loading, route-level behavior, and high-level composition.

Pages should remain as thin as practical.

Rules:

- UI components must not directly access the database.
- Feature components must not contain core business logic.
- Styling should flow through tokens and primitives.
- Avoid parallel component systems.

---

# 21. TypeScript Standard

TypeScript should improve clarity, not add ceremony.

Defaults:

- `strict: true`
- avoid `any` unless there is a documented and justified boundary case
- prefer explicit types at important boundaries
- share domain types where doing so improves clarity
- avoid type-level cleverness that harms readability
- do not use types to hide poor domain modeling

Type definitions should help a future developer understand the system faster.

---

# 22. Testing Standard

Testing should be pragmatic, targeted, and behavior-focused.

Default testing priorities:

- service-layer logic
- critical user flows
- reusable utilities
- interactive components with meaningful behavior
- auth and permission logic
- queue and job behavior when relevant
- data transformation logic

Defaults:

- define the test runner in `ARCHITECTURE.md`
- define test file placement in `PROJECT_RULES.md`
- do not over-test trivial implementation details
- test behavior, not incidental structure
- add regression tests when fixing meaningful bugs

If a project intentionally has no tests in a given area, that should be a deliberate documented choice rather than an omission.

---

# 23. Browser and Device Support

Every project must define its support baseline in `ARCHITECTURE.md`.

At minimum, document:

- primary device contexts
- mobile responsiveness expectations
- supported browser baseline
- keyboard accessibility expectations
- touch interaction expectations
- whether screen reader support is required for core workflows

AI must not assume desktop-only behavior unless the project explicitly says so.

---

# 24. Dependency Discipline

Before adding a dependency, confirm:

- the platform does not already solve the problem
- the problem is recurring rather than one-off
- the dependency is actively maintained
- the dependency does not introduce disproportionate architectural weight
- the dependency does not compromise portability without good reason
- the dependency does not create unnecessary build or hosting complexity

Avoid dependencies for trivial problems.

Every dependency should have a clear justification.

If a dependency meaningfully affects architecture, document it in `DECISIONS.md`.

---

# 25. Repository Discipline

Every repository must remain coherent and predictable.

Rules:

- Prefer clear descriptive names for files and modules.
- Avoid vague folders such as `misc`, `helpers`, `temp`, `stuff`, or `utils` unless their purpose is tightly defined.
- Keep one primary responsibility per file.
- Place code in the correct architectural layer.
- Avoid duplication unless abstraction is clearly better.
- Do not create speculative abstractions.
- Avoid broad rewrites unless they are explicitly approved.
- Update documentation when architecture meaningfully changes.
- The repository should become more understandable over time, not less.

A healthy repository rewards inspection.

---

# 26. Change Discipline

Every change must be:

- scoped
- intentional
- reviewable
- reversible when possible

Avoid:

- mixing refactors with feature work
- broad renames without reason
- directory reshuffles without payoff
- changing architecture while implementing a small feature
- introducing parallel patterns to existing ones
- cleaning unrelated files just because they were nearby

Small, clear changes are easier for humans and AI agents to review.

---

# 27. Documentation System

Each project should use a small set of durable documentation files.

## `CODING_CONSTITUTION.md`

Global engineering principles and defaults.

## `AGENTS.md`

Operational instructions for AI coding agents.

This should be short, direct, and action-oriented.

## `ARCHITECTURE.md`

Project-specific factual blueprint.

It describes what the system is, how it is built, where logic lives, where data lives, how it deploys, and what constraints matter.

## `PROJECT_RULES.md`

Repository-specific enforcement rules.

It defines naming, placement, commands, Definition of Done, and local consistency rules.

## `DECISIONS.md`

Architecture decision history.

Each major decision should include:

- decision summary
- why it was made
- alternatives considered
- date adopted
- whether it is reversible

## Mode Documents

Mode documents apply only when relevant.

Examples:

- `MODE_WORDPRESS_NATIVE.md`
- `MODE_WORDPRESS_HYBRID.md`
- `MODE_NUXT_APP.md`

Do not create mode files prematurely. Add them when a repeated project type needs durable rules.

---

# 28. AI Collaboration Rules

AI-generated code must remain understandable by human developers.

AI must:

- inspect existing patterns before inventing new ones
- read applicable project documentation before coding
- keep logic in the correct architectural layer
- avoid unnecessary dependencies
- avoid speculative abstractions
- use explicit naming
- preserve local repository consistency
- note assumptions when uncertainty remains
- make narrow, reviewable diffs
- summarize changed files after meaningful work
- update documentation when architecture changes
- run relevant checks when available
- avoid touching unrelated code

When unsure, AI must prefer:

- the simpler implementation
- the local repository pattern
- fewer moving parts
- narrower diffs
- clearer naming

AI must not:

- introduce new architecture without justification
- create parallel patterns to existing ones
- make broad unrelated edits
- hide uncertainty
- hardcode secrets
- remove safeguards to make tests pass
- invent project facts that belong in `ARCHITECTURE.md`

---

# 29. AI Project Initialization Protocol

This protocol defines how AI must begin all new software projects.

## Trigger

When the user says things like:

- start a new project
- build an app
- create a plugin
- help me plan a new system
- scaffold a new product

AI must enter Initialization Mode.

## Initialization Goals

Before implementation begins, AI must determine:

1. What is being built
2. Who it serves
3. System type
4. Whether the standard stack is being used
5. Hosting and portability expectations
6. Key product constraints
7. Design identity direction
8. Required initial project artifacts

## System Type Detection

AI must classify the project as one of:

- Standalone Application
- WordPress Plugin (Native)
- WordPress Plugin (Hybrid)
- Other, explicitly defined

This decision must happen early because it changes architecture.

## Questioning Rules

AI must:

- ask no more than 3 questions at a time
- wait for answers before continuing
- adapt later questions based on earlier answers
- prefer structured choices over vague open-ended prompts
- anchor to the default stack and architecture unless overridden
- explicitly confirm all deviations from defaults

Good question examples:

- Are we using the standard stack, or changing anything?
- Is this a standalone app, a WordPress-native plugin, or a hybrid WordPress plugin?
- Are we targeting the usual VPS-friendly deployment model, or something else?

Avoid vague prompts like:

- What stack do you want?
- How should this be designed?
- What do you want me to do?

## Recommended Question Sequence

### Round 1: Core Product Definition

Ask:

1. What are we building, and who is it for?
2. Is this a standalone app, a WordPress-native plugin, a hybrid WordPress plugin, or something else?
3. Are we using the usual stack and deployment model, or changing anything important?

### Round 2: Product Constraints

Adapt based on answers, then ask up to 3:

- Does this need user accounts or roles?
- What are the most important features for version one?
- Are there any important hosting, integration, performance, or portability constraints?

### Round 3: Design Identity

Ask up to 3:

- What should the UI feel like visually?
- Do we already have brand colors, typography direction, or style references?
- Is this interface meant to feel minimal, bold, elegant, technical, playful, premium, utilitarian, or something else?

### Round 4: Delivery Readiness

Ask up to 3:

- What files or artifacts do we want before coding starts?
- Should we generate a project brief, architecture draft, and starter folder plan first?
- Do any deviations from the standard rules need to be documented up front?

## Mandatory Pre-Build Outputs

Before substantial implementation begins, AI must produce these artifacts in some form:

- `PROJECT_BRIEF.md`
- initial `ARCHITECTURE.md` draft or populated architecture sections
- starter folder structure or implementation plan
- list of explicit deviations from the default constitution, if any

## Project Brief Confirmation

Before writing substantial code, AI must restate the project in a structured brief that includes:

- purpose
- audience
- system type
- standard stack confirmed or modified
- hosting and portability assumptions
- key features for version one
- design identity direction
- required integrations
- open questions or risks

The user must confirm or correct this brief before major implementation begins.

## First Build Boundary

AI must not jump from project kickoff directly into broad implementation.

Before major coding begins, the minimum acceptable foundation is:

- project brief confirmed
- system type confirmed
- architecture direction stated
- design direction stated
- folder and layering plan stated

## Failure Condition

If requirements remain unclear, AI must continue clarifying.

AI must not fill major gaps with confident guessing.

---

# 30. Deployment Standard

Applications deploy with:

- Docker containers when practical
- environment-based configuration
- secrets outside the repository
- reverse proxy behavior defined explicitly
- infrastructure choices documented clearly

Deployment architecture must be described in `ARCHITECTURE.md`.

At minimum, deployment documentation should explain:

- local environment
- staging environment if used
- production environment
- database location
- reverse proxy
- worker processes
- environment variables
- backup expectations

---

# 31. Operational Simplicity

Operational complexity is hidden technical debt.

Prefer:

- predictable infrastructure
- fewer moving parts
- explicit configuration
- inspectable services
- maintainable deployment
- simple recovery paths

Avoid:

- unnecessary microservices
- accidental platform lock-in
- orchestration complexity without clear payoff
- background systems no one can inspect
- infrastructure that only one person understands

The best system is not the most impressive system. It is the one that can be understood, operated, and improved reliably.

---

# 32. WordPress Context

WordPress-specific architecture requirements belong in project architecture documentation and, where needed, WordPress-specific mode documents.

The global constitution defines universal principles.

Mode documents define project-type-specific implementation rules.

## WordPress Plugin (Native)

Use a WordPress-native mode document when WordPress is the application host and the primary application logic lives inside WordPress.

Native plugins should respect WordPress conventions while still using clean application architecture.

Default native plugin flow:

```text
Hook
→ Controller
→ Service
→ Data Layer
```

## WordPress Plugin (Hybrid)

Use a hybrid mode document or architecture section when WordPress acts as a UI and integration layer for an external backend.

Hybrid plugin flow commonly looks like:

```text
WordPress Plugin
→ External API
→ Backend Services
→ Durable Database
```

The project’s `ARCHITECTURE.md` must state which model applies.

---

# 33. Human Review Standard

Agent-led coding still requires human judgment.

Before accepting meaningful AI-generated changes, the human reviewer should check:

- Did it solve the requested problem?
- Did it change only relevant files?
- Did it follow the documented architecture?
- Did it introduce new dependencies?
- Did it put logic in the correct layer?
- Did it preserve security and validation rules?
- Did it update docs when needed?
- Did the UI remain visually consistent?
- Did the app still run or tests still pass?

AI can do the work, but humans remain responsible for direction and acceptance.

---

# 34. Final Principle

Every system should reflect:

- clarity
- discipline
- portability
- intentional design
- long-term thinking

If a choice makes the system harder to understand, harder to move, harder to debug, or harder to maintain, it is probably the wrong default.
