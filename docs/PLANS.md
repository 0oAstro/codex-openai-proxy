# Plans

Plans are first-class artifacts for work that is too large or risky to hold in
chat context alone.

## When To Write A Plan

Use a checked-in plan for work that changes multiple modules, touches auth or
streaming behavior, adds a new endpoint family, changes Docker or CI behavior,
or requires staged validation.

Small localized edits can use an ephemeral checklist in the agent response
instead.

## Plan Shape

Use this structure for larger work:

```markdown
# Title

## Goal

One paragraph describing the user-visible outcome.

## Acceptance Criteria

- Concrete behavior that must be true when the work is complete.

## Steps

- [ ] Investigation
- [ ] Implementation
- [ ] Tests
- [ ] Documentation
- [ ] Verification

## Decisions

- Date-stamped decisions that future agents need to understand.

## Verification Log

- Commands run, results, and any remaining gaps.
```

## Storage

If plan volume grows, create:

- `docs/exec-plans/active/` for in-progress plans.
- `docs/exec-plans/completed/` for completed plans.
- `docs/exec-plans/tech-debt-tracker.md` for known cleanup work.

Until then, keep this file as the planning protocol and avoid creating empty
directories.
