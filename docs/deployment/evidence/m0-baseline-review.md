# M0 Baseline Review

- **Review date:** 2026-08-18
- **Issue:** [#2 — M0: Recover and validate the deployment baseline](https://github.com/dontknowhow2namemzself/SageSpace/issues/2)
- **Branch:** `codex/deployment-foundation`
- **Baseline commit:** `2f28ad5` (`main`, equal to `origin/main` at review time)

## Purpose

Record the pre-existing uncommitted deployment work before changing it. This
review separates sound intent from verified behavior and prevents the work from
being accepted, discarded, or committed as an opaque batch.

## Working-tree snapshot

- 13 modified tracked files
- 72 tracked insertions and 28 tracked deletions
- 6 untracked files
- no staged changes
- `git diff --check` passed
- switching from `main` to `codex/deployment-foundation` preserved the exact
  modified/untracked file list

### Modified tracked files

```text
backend/.env.example
backend/api/books.py
backend/api/chat.py
backend/api/export.py
backend/api/ingest.py
backend/core/cover.py
backend/core/database.py
backend/core/raptor.py
backend/core/tools.py
backend/main.py
frontend/app/chat/[bookId]/page.tsx
frontend/lib/api.ts
frontend/next.config.mjs
```

### Untracked files

```text
AGENTS.md
backend/core/paths.py
backend/core/ratelimit.py
docs/agents/domain.md
docs/agents/issue-tracker.md
docs/deployment/plan.md
```

## Keep/change/reject review

| Area | Files | Decision | Reason and required verification/change |
| --- | --- | --- | --- |
| Repository conventions | `AGENTS.md`, `docs/agents/*` | Keep | Establishes GitHub Issues and repository documentation conventions. Commit separately from application behavior. |
| Canonical deployment plan | `docs/deployment/plan.md` | Keep | Approved scope and milestone source. Keep synchronized with GitHub Issues. |
| Mutable data root | `backend/core/paths.py` | Change | A single `SAGESPACE_DATA_DIR` is correct, but directory creation must not depend on an unrelated route import. The “all runtime writes” claim must account for or retire `chat_debug.log`. |
| SQLite and checkpoint paths | `backend/core/database.py`, existing `backend/core/graph/build.py` | Keep with tests | Main SQLite moves under `DATA_DIR`; the checkpoint DB already derives from `DB_PATH` at call time. Add default/custom path tests. |
| Chroma path and deletion | `backend/core/raptor.py`, `backend/api/books.py` | Keep with tests | Both operations now share one Chroma path. Verify deletion uses the configured data root. |
| Uploads and covers | `backend/api/ingest.py`, `backend/core/cover.py` | Keep with changes | Both belong under the data root. Centralize directory lifecycle and test the configured root. Keep `PUBLIC_APP_URL`, with a configuration test. |
| Exports | `backend/core/tools.py`, `backend/api/export.py` | Change | Export generation correctly moves under `DATA_DIR`, but `api.export.EXPORT_DIR` becomes unused and existing tests patch the wrong seam. Remove duplication and test the complete exported-file response. |
| CORS | `backend/main.py`, `backend/.env.example` | Keep with tests | Explicit environment allowlist with safe local defaults is appropriate. Verify whitespace, empty entries, and custom origins. Same-origin production should not depend on permissive CORS. |
| Rate limiting | `backend/core/ratelimit.py`, `backend/api/chat.py`, `backend/api/ingest.py` | Change | Shared limiter and required `Request` parameters are correct. Limit values are still hard-coded; make them configurable. Verify disabled/enabled behavior and define the trusted proxy boundary before relying on client IP. |
| Same-origin frontend API | `frontend/lib/api.ts`, `frontend/app/chat/[bookId]/page.tsx` | Keep with build verification | `??` intentionally preserves an empty string for same-origin production while retaining the localhost fallback when unset. Verify with the production build and smoke flow. |
| Standalone Next.js output | `frontend/next.config.mjs` | Keep with build verification | Required for a smaller runtime image. Verify the standalone artifact after dependency restoration. Framework upgrade remains M1 work. |

No reviewed change is rejected outright. “Keep” does not mean “already
verified”; every item still needs the listed tests/build evidence before it can
be committed.

## Newly discovered gaps

These gaps pre-date or sit outside the reviewed deployment diff, but affect its
claims:

1. `backend/core/graph/nodes.py` writes answer and plan excerpts to
   `backend/chat_debug.log`. It is ignored by Git, but it is mutable data and
   may contain user content. Production logging must retire or explicitly gate
   this file before the data-handling claim is true.
2. `/health` currently returns a static `{"status": "ok"}` and does not perform
   the approved lightweight SQLite readiness check. This belongs to M1.
3. The rate limiter is only an effective per-client control after Nginx and
   Uvicorn agree on a restricted trusted-proxy configuration.
4. The approved local runtimes are now installed and verified: uv-managed
   Python 3.12.13 in `.venv`, plus Node 24.19.0 and npm 11.17.0 via nvm.
   Project-level version declarations are included so a fresh shell or machine
   does not silently select the previous Python 3.9 or Node 22 defaults.
5. Docker and a persistent GitHub CLI installation are absent. Docker is not
   required until M2; GitHub writes can continue to use short-lived,
   repository-scoped authorization windows.
6. The restored Next.js 14 dependency tree has 8 high-severity audit findings
   in the complete development tree and 3 in the production tree (`next`, its
   bundled `postcss`, and transitive `nanoid`). The registry's complete fix is
   the planned major upgrade to Next.js and `eslint-config-next` 16.3.1. M0
   retains the old version only long enough to establish a behavioral baseline;
   this audit result is a production-release blocker until the M1 upgrade and a
   clean follow-up audit. No automatic `npm audit fix` was applied.

## Resolved during M0

- Startup now creates and validates the complete persistent data-directory
  layout before initializing SQLite; importing the ingestion route no longer
  creates directories as a side effect.
- CORS parsing, strict environment booleans, and configurable chat/ingestion
  limits have executable offline tests. Invalid values fail at startup rather
  than silently weakening the intended control.
- Export tests now patch the real data-root seam and verify the complete file
  response. Cover tests verify the configured public application URL.
- ChromaDB and ONNX Runtime telemetry default to disabled before their modules
  load. The pytest process applies the same offline default early enough to
  avoid repository-local telemetry artifacts.
- The pytest asyncio fixture scope is explicit, removing the upstream
  deprecation warning and preserving current function-scoped behavior.

## Current verification status

| Check | Result |
| --- | --- |
| Git baseline and upstream equality | Pass |
| Dirty-tree preservation across branch creation | Pass |
| `git diff --check` | Pass |
| Secret-pattern scan | Pass (documented placeholders only) |
| Static review of every changed/untracked file | Pass |
| Python 3.12.13 runtime and `.venv` | Pass |
| Node 24.19.0 and npm 11.17.0 runtime | Pass |
| Python dependency installation and `uv pip check` | Pass (134 packages; compatible) |
| Backend tests | Pass (551 passed, 5 skipped, 3 expected convergence warnings) |
| Frontend `npm ci` | Pass (484 packages installed) |
| Full npm security audit | Blocked for production (8 high; planned M1 upgrade) |
| Production-only npm security audit | Blocked for production (3 high; planned M1 upgrade) |
| Frontend lint | Pass (no warnings or errors) |
| Frontend production build | Pass (5 routes generated) |
| Next.js standalone artifact | Pass (`.next/standalone/server.js`) |
| Deployment configuration tests | Pass (included in full backend suite) |

The next step is to split the reviewed and verified work into focused commits,
then open the M0 pull request. Production deployment remains prohibited until
the M1 framework upgrade clears the recorded audit blocker.

## Final commit split

1. `chore: configure repository agent conventions`
2. `docs: add production deployment plan and M0 evidence`
3. `build: declare supported local runtimes`
4. `feat: externalize backend deployment configuration`
5. `build: prepare same-origin standalone frontend`

This consolidates the initially proposed backend feature/test commits because
paths, startup validation, CORS, rate limiting, privacy defaults, and their
tests share configuration seams. Keeping them together avoids artificial
partial staging and incomplete intermediate states while preserving small,
reviewable documentation, runtime, backend, and frontend boundaries.
