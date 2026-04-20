# Documentation Guide

This file explains the documentation structure of the repository and the intended role of each document.

## Primary Entry Point

### `README.md`

Use `README.md` as the top-level entry point for the repository.

It should answer:

- what the project is
- what currently works
- what does not yet work
- where the key outputs are
- how the pipeline fits together
- where to go next in the docs

## Status and Planning Docs

### `REPORT_BRIEFING.md`

Use this file as the fastest handoff for teammates writing the report.

It should answer:

- what the current project phase is
- which results are safe to cite
- which claims should be avoided
- which detector baseline is current
- which docs and output files are most relevant for writing

### `PROJECT_STATUS.md`

Use this file for the current project state.

It should answer:

- what has already been completed
- what artifacts currently exist
- what remains incomplete
- what the immediate next steps are
- what a fresh clone should expect locally

### `PROJECT_EXECUTION_PLAN.md`

Use this file for the broader roadmap and execution structure.

It should answer:

- what the project milestones are
- what the staged plan looks like
- how work is organized over time

## Requirement and Scope Docs

### `REQUIREMENTS_GAP_ANALYSIS.md`

Use this file to compare the current implementation against the course requirements.

It should answer:

- which requirements are fully met
- which are partially met
- which are still missing

### `VIDEO_GENERATION_PLAN.md`

Use this file for the next-stage technical plan around video generation.

It should answer:

- what the current video MVP can do
- what needs to be extended next
- what the staged technical path is from short clips to assignment-scale videos

## Platform / Operations Docs

### `WINDOWS_RUN_TROUBLESHOOTING.md`

Use this file as the Windows rerun and debugging manual.

It should answer:

- what setup problems were encountered
- what exact fixes worked
- what commands are known-good on Windows

## Asset and Data Reference Docs

### `assets/models/MODEL_MANIFEST.md`

Use this file to document the expected 3D asset layout and naming convention.

### `data_sources/SYNTHETIC_DATA_SOURCES.md`

Use this file to record the upstream data and asset sources used by the project.

## Submission Docs

### `submissions/README.md`

Use this file for submission-package-specific notes, not day-to-day development workflow.

## Collaboration Rules for Docs

When updating the repository:

1. If runtime behavior changes, update `README.md` and `PROJECT_STATUS.md`.
2. If project scope changes, update `REQUIREMENTS_GAP_ANALYSIS.md`.
3. If the next technical phase changes, update `VIDEO_GENERATION_PLAN.md`.
4. If a new Windows issue is discovered, append it to `WINDOWS_RUN_TROUBLESHOOTING.md`.
5. If asset expectations change, update `MODEL_MANIFEST.md`.

## Recommended Reading Order

For a new teammate:

1. `README.md`
2. `PROJECT_STATUS.md`
3. `REQUIREMENTS_GAP_ANALYSIS.md`
4. `VIDEO_GENERATION_PLAN.md`
5. `WINDOWS_RUN_TROUBLESHOOTING.md` if running on Windows

For a report-writing teammate:

1. `REPORT_BRIEFING.md`
2. `README.md`
3. `PROJECT_STATUS.md`
4. `REQUIREMENTS_GAP_ANALYSIS.md`

For close-out / final packaging:

1. `README.md`
2. `PROJECT_STATUS.md`
3. `REPORT_BRIEFING.md`
4. `submissions/README.md`

## Recommended Maintenance Rule

Keep `README.md` concise and navigational.

Move detailed explanations into the specialized documents above instead of letting the main README become the only place where project knowledge lives.

When the current recommended baseline changes, update these together:

1. `README.md`
2. `PROJECT_STATUS.md`
3. `REPORT_BRIEFING.md`
