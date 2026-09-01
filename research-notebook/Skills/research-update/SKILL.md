---
name: research-update
description: Update the shared research notebook after a research or coding session. Use when the user asks to log, record, summarize, hand off, or update research progress.
disable-model-invocation: true
---

# Configuration

## Locate the research notebook

Check these locations in order:

1. `/Users/romanagle/Research`
2. `~/research/notebook`
3. The current working directory, if it contains:
   - `AGENTS.md`
   - `Daily/`
   - `Projects/`

Use the first valid location.

If none exists, ask for the notebook path and stop.

The notebook root must contain:

- `AGENTS.md`
- `Daily/`
- `Projects/`
- `Skills/`

### Repository guardrail

Before any Git operation, run `git remote get-url origin`. Continue only when the remote identifies `romanagle/research-notebook.git`. If it identifies a code repository or any other remote, stop and locate the standalone Research vault; never commit or push research outputs there unless the user explicitly names that repository.

---
## New Project Initialization

If the identified research project does not yet have a project note:

1. Confirm that the work represents a distinct research project.
2. Choose a concise, descriptive project name.
3. Create:

   `Projects/<Project Name>.md`

4. If a project template exists, use it.
5. Otherwise initialize the note with:

```markdown
# Project Name

## Objective

## Current status

## Active blockers

## Important decisions

## Durable research memory

## Open research questions

## Next milestone

## Future directions

## Literature Tracker

| Title | Author | Link | Date researched | Summary |
| --- | --- | --- | --- | --- |
```

6. Populate only information supported by:
   - the current session,
   - the current code repository,
   - user-provided context,
   - existing notebook notes.

7. Do not invent historical progress.
8. Link the daily session to the project using:

   `[[Project Name]]`

9. Create the project artifact directory when needed:

   `Artifacts/<Project Name>/`

10. Include the new project note and directories in the same Git commit as the research update.


# Research Notebook Update

The Obsidian vault is the canonical research notebook.

## Purpose

Maintain:

- a concise daily research record,
- current project state,
- durable research memory,
- important research artifacts.

---

## Writing Style

Use concise, direct language.

- Prefer bullets over paragraphs.
- Keep each bullet to one sentence.
- Aim for fewer than 20 words per bullet.
- Remove filler, repetition, and obvious context.
- Do not restate the same fact under multiple headings.
- Preserve technical precision.
- Use complete sentences only when needed.
- Summarize repeated observations into one stronger bullet.

Bad:

- A long explanation of every command, attempt, and intermediate thought.

Good:

- Fixed probing-data normalization.
- Remaining issue: negative reactivities distort likelihood estimates.

---

## Workflow

When invoked:

1. Locate the notebook.
2. Run the **Before editing** Git workflow.
3. Read `AGENTS.md`.
4. Identify the project or projects worked on.
5. Determine today's date.
6. If `Daily/YYYY-MM-DD.md` exists:
   - Read it.
1. Otherwise:
   - Create it from `Templates/Daily.md`.
   - Then read the newly created note.
1. Read the relevant project notes.
2. Review:
   - conversation context,
   - code changes,
   - commands,
   - results,
   - generated files,
   - unresolved issues.
8. Identify likely artifacts.
9. Update today's daily summary.
10. Insert the new session at the top of the session history.
11. Update project notes only where durable state changed.
12. Archive approved artifacts.
13. Review all changes.
14. Run the **After editing** Git workflow.
15. Provide the completion summary.

---

## Git Synchronization

The Research vault may be edited concurrently by Obsidian, Claude, Codex, and remote sessions. Synchronize conservatively — absorb harmless repository drift without stopping, but never guess through an actual conflict.

### Machine-local files

Treat the following as machine-local UI state, never research content:

- `.obsidian/workspace.json`
- `.obsidian/workspace-mobile.json`

Confirm `.gitignore` includes both paths before continuing. Never stage, commit, restore from the remote, or report changes to them as meaningful research changes.

### Before editing

1. Change to the notebook root.
2. Confirm the repository: `git rev-parse --show-toplevel`.
3. If meaningful uncommitted changes already exist (anything other than the machine-local files above), inspect them with `git status --short` and `git diff` before continuing — see Handling Uncommitted Changes.
4. If the remote has moved ahead, pull with rebase and autostash:

   `git pull --rebase --autostash`

   This absorbs non-conflicting remote commits and any harmless local drift automatically. Stop only if Git reports an actual conflict in user-authored content — not merely because the remote branch had moved ahead, or because non-conflicting local changes existed before the pull.

5. If Git reports an actual merge or rebase conflict:
   - stop,
   - do not guess at a resolution,
   - report the conflicting files, which files were modified locally vs. remotely, and a brief summary of both sides.

### Before committing

1. Review:

   `git status --short`
   `git diff --stat`

2. Stage only files intentionally changed by this update. Do not use `git add .` or `git add -A` unless every changed file has been reviewed.
3. Never stage `.obsidian/workspace.json` or `.obsidian/workspace-mobile.json`.
4. Commit using:

   `Research: <Project Name> — <short session summary>`

5. Do not create an empty commit.

### Before pushing

Another session may have pushed since this skill began.

1. Run `git pull --rebase --autostash` again.
2. Push: `git push`.
3. If the push is rejected because the remote moved ahead, repeat the pull-and-rebase once and retry the push.
4. Never use `git push --force` or `git push --force-with-lease`.
5. If commit or push still fails:
   - preserve local changes,
   - report the exact error,
   - do not claim synchronization succeeded.

### Handling Uncommitted Changes

If meaningful, user-authored changes already exist when this skill begins (excluding the machine-local files above):

1. Inspect them.
2. Preserve them — never discard user-authored notes, tasks, project files, or skill edits without explicit user instruction.
3. If they form a coherent change, commit them separately with an accurate message before continuing.
4. Otherwise, set them aside with a named stash:

   `git stash push -u -m "Temporary pre-skill changes"`

   and restore with `git stash pop` once this skill's own sync, edits, and push are complete.
5. Stop only if restoring the stash creates an actual conflict.

---

## Read Before Writing

Before editing:

1. Read the full daily note.
2. Read the relevant project notes.
3. Identify information already recorded.
4. Avoid duplicate bullets.
5. Merge overlapping points.
6. Preserve prior session history.

### Manual Deletions

Check `git diff` against the last commit for today's daily note.

If a task checkbox line under `Next steps` (or a carried-over `Tomorrow's Tasks` snapshot) was removed by hand, not through this skill:

1. Find the matching task in that project's `Tasks/<Project Name>.md`.
2. Delete it from wherever it appears (`Now`, `Next`, or `Waiting`).
3. Do not add it to `Completed recently` — a manual deletion is not a completion.
4. Mention it in the completion summary.

If the wording changed rather than being deleted outright, or intent is unclear, ask before deleting it from the canonical file.

---

## Daily Note Structure

If today's daily note does not exist, create it from `Templates/Daily.md` before making any updates. 

Each daily note has two parts:

1. **Daily Summary**
2. **Session History**

The daily summary is updated in place.

The session history preserves individual sessions.

Use this structure:

```markdown

## Daily Summary

### Projects
- [[Project Name]]

### Progress
- Concise summary of cumulative progress today.

### Problems solved
- Problems solved across all sessions today.

### Open problems
- Current unresolved problems only.

### Questions
- Only questions the user explicitly raised this session.

### Decisions
- Important decisions made today.

### Key results
- Important results from all sessions today.

### Next steps

**[[Project Name]]**
- [ ] Highest-priority next action.
- [ ] Additional action.

### Handoff
- Brief context needed to resume work.

---

## Session History

### HH:MM — Agent Name

**Project:** [[Project Name]]

- Objective:
- Work completed:
- Problems solved:
- New results:
- Remaining issue:
- Immediate next step:

---
```

---

## Daily Summary Rules

Maintain only one copy of each daily summary section.

Do not create repeated headings such as:

- multiple `Open problems` sections,
- multiple `Questions` sections,
- multiple `Next steps` sections.

For each update:

1. Merge new information into the existing daily summary.
2. Remove resolved items from `Open problems`.
3. Move solved items into `Problems solved` when useful.
4. Remove answered questions.
5. Replace outdated next steps.
6. Keep only current handoff context.
7. Deduplicate similar bullets.
8. Preserve distinct findings from separate sessions.
9. Group `Next steps` under a `[[Project Name]]` sub-heading for each project touched today, in the same order as `### Projects`. Omit the sub-heading only when exactly one project was touched.
10. Summarize related findings into one bullet when possible.

The daily summary should represent the current state of the entire day, not one session.

---

## Session History Rules

Each invocation creates one new session entry.

- Insert the newest session directly below `## Session History`.
- Keep sessions in reverse chronological order.
- Never delete an older session.
- Do not repeat the full daily summary inside each session.
- Record only what is specific to that session.
- Omit empty fields.
- Keep the session entry brief.

The newest session must appear first.

---

## Project Notes

Project notes represent the current state of a project.

Update them only when information will remain useful beyond today.

Maintain:

- Objective
- Current status
- Active blockers
- Important decisions and rationale
- Durable research memory
- Open research questions
- Next milestone
- Future directions
- Literature Tracker

Durable research memory includes:

- important insights,
- failed approaches worth remembering,
- experimental assumptions,
- design decisions,
- references,
- lessons learned.

Before adding information, ask:

> Will this still help someone reading the project note three months from now?

If no, keep it only in the daily note.

### Open Research Questions and Future Directions

Add to `Open research questions` or `Future directions` only when the user explicitly states a question or direction during the session.

Do not synthesize new questions or directions from session content, code changes, or results, even when they seem like a natural next step. Existing entries may be reworded for clarity or removed once resolved, but never added by inference.

### Literature Tracker

Every project note ends with a `## Literature Tracker` table:

```markdown
| Title | Author | Link | Date researched | Summary |
| --- | --- | --- | --- | --- |
```

Add a row whenever a specific paper, preprint, or external reference is discussed during the session — the user asks about it, cites it, or it's used to justify a decision.

- Title: full paper title.
- Author: first author `et al.` if more than one.
- Link: canonical URL (publisher page, DOI, or arXiv), only if known with confidence; leave blank rather than guess.
- Date researched: the date it came up in conversation (`YYYY-MM-DD`), not the paper's publication date.
- Summary: one sentence on how it came up in this conversation, not a paper abstract.

Do not synthesize entries for papers only implied by code, filenames, or prior notes — only add a row when the paper was explicitly discussed. Never delete existing rows; append newest at the bottom.

### Project Note Editing Rules

- Update existing sections instead of adding duplicate sections.
- Replace stale status information.
- Remove blockers that are resolved.
- Merge overlapping questions.
- Preserve important failed approaches.
- Keep bullets concise.
- Do not turn the project note into a chronological diary.

---

## Tables

If a session produced a table, or its result condenses into one — counts compared across
conditions, per-class or per-method metrics, before/after values, a ranked list — write the
**actual Markdown table**, not a sentence describing it.

Ask of every numeric result:

> Could this be read as rows and columns?

If yes, it is a table. Prose like "counts are 432/791 by RNA but 577/576 by file" is a table
that was flattened — expand it.

Where it goes:

1. **Daily note** — inside `### Key results`, under a one-line lead-in bullet giving the
   conclusion. The table is the evidence for that bullet.
2. **Project note** — in the section the result belongs to (`Current status`,
   `Important decisions`, or `Durable research memory`), whenever the result is durable.
3. **Artifact note** — additionally write
   `Artifacts/<Project Name>/<slug>.md` when the table needs method, units, limits, or more
   than roughly eight rows. Keep the abridged table in the daily and project notes and link
   the artifact with `[[Artifacts/<Project Name>/<slug>|caption]]`. The link never replaces
   the inline table.

Rules:

- Include the real numbers. Never write "see the script output" or "results in the repo".
- Head every column, including units.
- Use `—` for a value that was not computed; do not leave a cell blank and do not invent it.
- Keep a table under about eight rows in the daily and project notes; put the full version in
  the artifact note and say what was trimmed.
- A table that also has a figure gets both — the figure embed does not replace the numbers.
- Never delete or condense a table already written into a note.

---

## Artifact Handling

Artifacts are outputs that help reproduce or understand a session.

Examples:

- figures,
- tables,
- reports,
- notebooks,
- important logs,
- slides,
- user-selected files.

When updating:

1. Find new or modified likely artifacts.
2. Present a short candidate list.
3. Ask whether to:
   - archive all,
   - archive selected files,
   - skip archival.
4. Copy approved files into:

   `Artifacts/<Project Name>/`

5. Never move or modify originals.
6. Embed images using:

   `![[Artifacts/<Project Name>/<filename>]]`

7. Link non-image artifacts from the daily note.
8. Add a one-line caption or explanation.
9. Avoid archiving temporary or redundant outputs.

---

## Editing Rules

- Preserve all session history.
- Keep newest sessions at the top.
- Maintain one daily summary.
- Update summaries instead of duplicating sections.
- Project notes reflect current state.
- Do not invent results.
- Do not claim commands ran when they did not.
- Do not invent entries for `Questions`, `Open research questions`, or `Future directions` — record only what the user explicitly stated.
- Update each relevant project separately.
- Use Obsidian wiki links when possible.
- Keep sentences and bullets short.
- Prefer synthesis over repetition.
- Write tabular results as real Markdown tables, never as prose — see Tables.

---

## Completion Summary

Before finishing, report:

### Research notebook updated

- Daily summary: Updated / Not updated
- Session entry: Added / Not added
- Project notes: Updated / Not updated
- Artifacts archived: <number>
- Figures embedded: <number>
- Tables written: <number>
- Questions recorded: <number>
- Outstanding blockers: <number>
- Tasks removed via manual deletion: <number, with names, or "None">
- Git commit: <hash or "No commit created">
- Remote synchronization: Pushed / Not pushed

Mention anything skipped or failed in one concise sentence.
