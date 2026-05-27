# AGENTS.md

Version: 1.1  
Purpose: Define the operating instructions AI coding agents must follow when working in this repository.

This file is the first-stop instruction file for AI agents. It does not replace the project documentation. It tells agents how to use the documentation, how to make changes safely, and how to avoid creating entropy.

---

# 1. Required Reading Order

Before making substantial changes, read these files in order:

1. `CODING_CONSTITUTION.md`
2. `ARCHITECTURE.md`
3. `PROJECT_RULES.md`
4. `DECISIONS.md`
5. Any applicable mode file, such as:
   - `MODE_WORDPRESS_NATIVE.md`
   - `MODE_WORDPRESS_HYBRID.md`
   - `MODE_NUXT_APP.md`

If one of these files does not exist, say so in your work summary and continue with the documents that are available.

Do not treat this file as the whole instruction set. This file is the operating guide. The other documents contain the durable engineering standards and project-specific facts.

---

# 2. Documentation Roles

Use the documentation this way:

## `CODING_CONSTITUTION.md`

Global engineering defaults.

Use it for:

- core philosophy
- default architecture
- default stack
- security standards
- portability principles
- design system standards
- AI collaboration principles

Do not override it casually.

## `ARCHITECTURE.md`

Project-specific factual blueprint.

Use it for:

- what the project is
- who it serves
- actual stack
- actual system type
- domain model
- data ownership
- request flows
- authentication and authorization
- background jobs
- deployment
- integrations
- environment variables

If the code and `ARCHITECTURE.md` disagree, do not silently choose one. Note the mismatch and make the smallest safe change.

## `PROJECT_RULES.md`

Repository-specific enforcement rules.

Use it for:

- naming conventions
- folder placement
- local patterns
- testing commands
- dependency rules
- file creation rules
- Definition of Done

Follow local project rules even when another style would also be valid.

## `DECISIONS.md`

Decision history.

Use it to avoid relitigating previous architecture choices.

Before reversing a major decision, check whether it has already been documented. If a new major decision is made, add an entry to `DECISIONS.md`.

## Mode Files

Mode files apply only when the project type requires them.

For example:

- Native WordPress plugins should follow `MODE_WORDPRESS_NATIVE.md`
- Hybrid WordPress plugins should follow `MODE_WORDPRESS_HYBRID.md`
- Standalone Nuxt apps may follow `MODE_NUXT_APP.md`

Do not apply WordPress-native rules to a standalone app. Do not apply standalone app assumptions to a WordPress-native plugin.

---

# 3. Core Operating Principle

Follow `CODING_CONSTITUTION.md` as the global source of truth.

In practice, every change should be small, clear, consistent with local patterns, and easy to review.

If a choice would make the system harder to understand, harder to maintain, harder to move, or harder for another agent to continue, it is probably the wrong default.

---

# 4. Before Editing Code

Before writing code:

1. Inspect the relevant folder structure.
2. Inspect nearby files that already solve similar problems.
3. Identify the correct architectural layer.
4. Check whether the change affects documentation.
5. Check whether tests, build scripts, or linting exist.
6. State any important assumption if requirements are unclear.

Do not jump directly into implementation without first understanding the local pattern.

---

# 5. Change Discipline

Every change must be:

- scoped
- intentional
- reviewable
- reversible when practical

Avoid:

- broad rewrites
- unrelated formatting changes
- mixing refactors with feature work
- moving files without clear benefit
- renaming public interfaces casually
- changing architecture to solve a local problem

If a task requires a large change, break it into smaller steps.

---

# 6. Architecture Boundaries

Respect the project architecture.

Default application flow:

```text
UI / Page
→ Server Route / Controller
→ Service Layer
→ Data Layer
→ Database
```

Default WordPress-native plugin flow:

```text
Hook
→ Controller
→ Service
→ Data Layer
```

Rules:

- UI code must not contain database logic.
- UI code must not contain core business rules.
- Route handlers and controllers should stay thin.
- Services own business behavior.
- Data access belongs in the appropriate data layer.
- Hooks connect WordPress to application logic. Hooks must not become application logic.
- Background jobs must not be improvised outside the documented job system.

If the project uses a different flow, follow `ARCHITECTURE.md`.

---

# 7. File Creation Rules

Create new files only when justified.

Before creating a file, ask:

- Does this belong in an existing file?
- Is this a real responsibility or a speculative abstraction?
- Does the project already have a pattern for this?
- Will this make the system easier to understand?

Rules:

- one primary responsibility per file
- clear descriptive names
- no vague folders such as `misc`, `helpers`, `stuff`, or `temp`
- no parallel patterns when an existing pattern works
- no speculative abstractions for imagined future needs

---

# 8. Dependency Rules

Do not add dependencies casually.

Before adding a dependency, confirm:

- the platform does not already solve the problem
- the need is recurring, not one-off
- the package is actively maintained
- the dependency does not add disproportionate weight
- the dependency does not reduce portability without good reason

If a dependency is added, explain why in the work summary. If the project has a dependency policy in `PROJECT_RULES.md`, follow it.

---

# 9. Security Rules

Treat all external input as untrusted.

Required standards:

- validate input at the server or trusted boundary
- enforce authorization on the server side
- use parameterized queries
- sanitize and escape output for its context
- never expose stack traces or internal details to users
- never commit secrets, tokens, credentials, or private keys
- never log secrets or sensitive personal data

For WordPress projects, also use:

- capability checks
- nonce verification
- WordPress sanitization functions
- WordPress escaping functions
- prepared SQL statements

---

# 10. Design System Rules

Do not invent visual styling independently if a design system exists.

Before building UI:

1. Check token files.
2. Check primitive components.
3. Check feature component patterns.
4. Check existing spacing, typography, surface, radius, and interaction rules.

Rules:

- primitives own reusable visual styling
- feature components compose primitives
- pages orchestrate layout and data
- themeable values should use tokens
- avoid hardcoded colors, spacing, and radius when tokens exist
- do not introduce a second visual language

If no visual system exists yet, create or request a visual identity checkpoint before heavy UI implementation.

---

# 11. Testing and Verification

After changes, run the smallest relevant verification available.

Examples:

- unit tests
- type check
- lint
- build
- targeted script
- manual smoke test

Do not claim something is fully verified unless it was actually verified.

If checks cannot be run, say why.

When tests are missing but the changed logic is important, add focused tests if the project has a testing pattern.

---

# 12. Documentation Updates

Update documentation when a change affects:

- architecture
- data ownership
- folder structure
- environment variables
- integrations
- background jobs
- authentication
- deployment
- durable design system rules
- major technical decisions

Use:

- `ARCHITECTURE.md` for current project facts
- `DECISIONS.md` for major decisions and tradeoffs
- `PROJECT_RULES.md` for repository-specific enforcement rules
- mode files for project-type-specific rules

Do not let code and documentation drift apart.

---

# 13. Git and Repository Safety

Before starting:

```bash
git status
```

If there are existing user changes, do not overwrite them.

Rules:

- never discard user changes without explicit instruction
- avoid destructive Git commands unless explicitly requested
- do not rewrite history unless explicitly requested
- keep commits focused when asked to commit
- do not commit secrets or generated junk files
- respect `.gitignore`

If a task is risky, recommend a branch.

Suggested branch naming:

```text
feature/short-description
fix/short-description
chore/short-description
```

---

# 14. Long-Running Agent Workflow

For longer tasks:

1. Restate the task briefly.
2. Identify the files or areas likely involved.
3. Make a small plan.
4. Inspect before editing.
5. Implement in narrow steps.
6. Verify each meaningful step when possible.
7. Summarize changes and remaining risks.

Do not disappear into a large rewrite.

Prefer incremental progress that can be reviewed.

---

# 15. When Requirements Are Unclear

When uncertainty affects architecture, data, security, or user-facing behavior:

- pause and ask a focused question, or
- make the smallest reasonable assumption and state it clearly

Do not fill major gaps with confident guessing.

When choosing without asking, prefer:

- simpler implementation
- fewer moving parts
- local repository pattern
- reversible change
- no new dependency
- no new architecture

---

# 16. Prohibited Behavior

AI agents must not:

- introduce new architecture without justification
- create duplicate systems for the same responsibility
- add dependencies for trivial problems
- hide business logic in UI components
- place SQL or persistence logic in the wrong layer
- swallow errors silently
- expose internal errors to users
- hardcode secrets
- commit credentials
- rewrite large areas unrelated to the task
- change formatting across unrelated files
- invent documentation facts not supported by the code or user direction
- claim tests passed if they were not run

---

# 17. Work Summary Format

At the end of a coding task, provide a concise summary:

```text
Summary:
- Changed ...
- Added ...
- Updated ...

Verification:
- Ran ...
- Not run: ... because ...

Notes:
- Assumptions ...
- Follow-up recommended ...
```

Mention important files changed.

Mention any documentation updated.

Mention any risks or incomplete work.

---

# 18. Definition of Done

Work is done only when:

- the change satisfies the request
- code is in the correct architectural layer
- naming is clear and consistent
- no unnecessary dependencies were added
- relevant validation and error handling are present
- relevant checks were run or honestly reported
- documentation was updated if architecture changed
- another developer or agent can understand the change quickly

“Works locally” is not enough if the implementation creates confusion or drift.

---

# Final Instruction

Act like a careful senior developer building for long-term AI-assisted development.

Optimize for correctness, maintainability, portability, reviewability, and the user’s ability to keep building with AI agents over time.
