---
name: workspace-curator
description: Audit and organize research or analysis workspaces whose active code, generated outputs, prototypes, and obsolete experiments have become difficult to distinguish. Use when the user wants a project map, directory cleanup, archival plan, canonical-path decision, or ongoing workspace hygiene. Do not use for ordinary small refactors inside an already coherent project.
---

# Workspace Curator

Make the workspace navigable before making it smaller. The desired outcome is that a returning researcher can identify the current narrative, canonical code, selected evidence, reproducible outputs, and unresolved work without searching historical experiments.

## Core model

Classify material by role rather than age alone:

- `ACTIVE` — directly supports current work.
- `SOURCE` — canonical code, configuration, or input data.
- `GENERATED` — reproducible output or cache; never a navigation source.
- `CANDIDATE` — valid work without a settled role.
- `SUPERSEDED` — rejected variant or stale state replaced by something identifiable.
- `ARCHIVED` — intentionally excluded but retained for history.
- `UNKNOWN` — provenance or purpose is insufficient for safe action.

Do not infer that newer means authoritative. Determine authority from current narrative documents, explicit user decisions, dependency references, manifests, and reproducibility paths.

## Audit

Begin read-only unless the user already authorized documentation changes.

1. Read repository guidance and inspect Git status before proposing mutations.
2. Identify competing sources of truth, including external documents the workspace cannot inspect.
3. Inventory top-level directories, large output trees, variant families, caches, nested repositories, and uncommitted reorganizations.
4. Trace active narrative claims or deliverables to exact figures, scripts, configurations, and required inputs.
5. Report contradictions between candidate authority files instead of silently choosing one.

Create or update these lightweight entry points when authorized:

- `PROJECT_MAP.md` — where to start, authority order, active scope, canonical code, and current deliverable shape.
- `CLEANUP_MANIFEST.md` — path-level classifications, proposed dispositions, blockers, and cleanup batches.

Adapt filenames when the project already has equivalent canonical documents. Do not create parallel maps that compete with them.

## Cleanup batches

Clean in reviewable batches ordered by confidence and reversibility:

1. Disposable machine products such as bytecode and empty staging directories.
2. Rejected presentation variants whose selected replacement is verified.
3. Superseded narrative alternatives after the active narrative is synchronized.
4. Generated runs after their producing command, inputs, and selected outputs are documented.
5. Historical experiments only after their scientific disposition and provenance are explicit.

For each batch, state:

- exact paths;
- classification and reason;
- active replacement or retained canonical copy;
- whether regeneration is verified;
- proposed operation;
- recovery method;
- expected space or complexity reduction.

Prefer a dated quarantine move over deletion when scientific provenance is uncertain. Keep quarantine outside active navigation paths and include a manifest. Never treat Git history as sufficient recovery for ignored, untracked, large, or generated files.

## Mutation guardrails

- Do not move, rename, or delete material during an audit-only request.
- Obtain explicit approval for each cleanup batch before destructive or broad filesystem changes.
- Preserve unrelated dirty-worktree changes; do not normalize them as part of cleanup.
- Stop when a move would cross repository boundaries, break an unresolved reference, or overwrite an existing path.
- Never delete the only known copy of data, model weights, results, or provenance.
- Never delete generated work merely because it is large; first establish whether regeneration is practical and documented.
- After moves, repair references and verify every active link, import, command, and selected output affected by the batch.

## Ongoing maintenance

When new experiments are added:

- place exploratory work under a clearly provisional area;
- give each experiment a purpose, status, producing command, and disposition;
- keep only accepted deliverables in the active figure or results index;
- move rejected alternatives out of active navigation once a replacement is selected;
- update the project map when authority, active scope, or canonical paths change;
- update the cleanup manifest when candidates become active, superseded, archived, or safely removable.

Do not turn this skill into a rigid universal directory template. Preserve structures that already communicate authority and provenance clearly.

## Report

End with:

- authoritative entry point;
- contradictions or unknowns;
- files created or updated;
- actions taken versus merely proposed;
- next cleanup batch requiring approval;
- recovery information for any moved or deleted material.
