# Documentation Index

This directory is the repository-local knowledge base for agent work. Keep
`AGENTS.md` short and use these files as the durable system of record.

## Core Documents

- `ARCHITECTURE.md`: service shape, module responsibilities, and request flow.
- `RELIABILITY.md`: runtime invariants, verification expectations, and failure
  handling.
- `SECURITY.md`: authentication boundaries, secret handling, and safe logging.
- `PLANS.md`: when and how to capture execution plans for multi-step work.

## Maintenance Rules

- Update docs in the same change as behavior when a route, configuration
  option, auth flow, or operational assumption changes.
- Prefer short, current documents over long manuals. Delete stale guidance
  rather than preserving contradictory history.
- Promote repeated review feedback into tests, lints, scripts, or explicit
  docs so future agents can discover it from the repo.
