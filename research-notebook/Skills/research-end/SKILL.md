---
name: research-end
description: Close the current research day, summarize progress, reconcile unfinished work, and prepare the next day's research note.
disable-model-invocation: true
---

# Configuration

## Locate the Research Notebook

Check these locations in order:

1. `/Users/romanagle/Research`
2. `~/research/notebook`
3. The current working directory, if it contains:
   - `AGENTS.md`
   - `Daily/`
   - `Projects/`
   - `Tasks/`

Use the first valid location.

If no valid notebook exists, ask for the notebook path and stop.

### Repository guardrail

Before any Git operation, run `git remote get-url origin`. Continue only when the remote identifies `romanagle/research-notebook.git`. If it identifies a code repository or any other remote, stop and locate the standalone Research vault; never commit or push research outputs there unless the user explicitly names that repository.

---

# Research End

## Purpose

Close the current research day and prepare a clear starting point for the next day.

This skill should:

- summarize today's research,
- reconcile completed and unfinished tasks,
- preserve a concise handoff,
- create tomorrow's daily note,
- add tomorrow's planned tasks,
- add known meetings and reminders,
- commit and push the notebook changes.

Use this skill only when the user is finished working for the day or explicitly wants to prepare the next day.

---

## Core Responsibilities

`research-end` answers:

> What did I accomplish, what remains, and what should I resume tomorrow?

It is different from `research-update`:

- `research-update` records an individual session.
- `research-end` consolidates the full day and prepares tomorrow.

---

## Git Synchronization

The Research vault may be edited concurrently by Obsidian, Claude, Codex, and remote sessions. Synchronize conservatively — absorb harmless repository drift without stopping, but never guess through an actual conflict.

### Machine-local files

Treat the following as machine-local UI state, never research content:

- `.obsidian/workspace.json`
- `.obsidian/workspace-mobile.json`

Confirm `.gitignore` includes both paths before continuing. Never stage, commit, restore from the remote, or report changes to them as meaningful research changes.

### Before Editing

1. Change to the notebook root.
2. Confirm the Git repository: `git rev-parse --show-toplevel`.
3. If meaningful uncommitted changes already exist (anything other than the machine-local files above), inspect them with `git status --short` and `git diff` before continuing — see Handling Uncommitted Changes.
4. If the remote has moved ahead, pull with rebase and autostash:

   `git pull --rebase --autostash`

   This absorbs non-conflicting remote commits and any harmless local drift automatically. Stop only if Git reports an actual conflict in user-authored content — not merely because the remote branch had moved ahead, or because non-conflicting local changes existed before the pull.

5. If Git reports an actual merge or rebase conflict:
   - stop,
   - do not guess at a resolution,
   - report the conflicting files, which files were modified locally vs. remotely, and a brief summary of both sides.

### Before Committing

0. Complete the Confirm Before Closing step before proceeding.
1. Review:

   `git status --short`
   `git diff --stat`

2. Stage only files intentionally changed by this skill. Do not use `git add .` or `git add -A` unless every changed file has been reviewed.
3. Never stage `.obsidian/workspace.json` or `.obsidian/workspace-mobile.json`.
4. Commit using:

   `Research end: YYYY-MM-DD — <short summary>`

5. Do not create an empty commit.

### Before Pushing

Another session may have pushed since this skill began.

1. Run `git pull --rebase --autostash` again.
2. Push: `git push`.
3. If the push is rejected because the remote moved ahead, repeat the pull-and-rebase once and retry the push.
4. Never use `git push --force` or `git push --force-with-lease`.
5. If commit or push still fails:
   - preserve all local changes,
   - report the exact failure,
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

## Determine the Dates

1. Determine the current local date.
2. Determine the next calendar date.
3. Use:

   `Daily/YYYY-MM-DD.md`

4. Do not assume the next workday is Monday or skip weekends unless the user explicitly requests workday-only behavior.

---

## Read Before Writing

Read:

1. Today's full daily note.
2. The relevant project notes.
3. The relevant task files.
4. Any notes or meeting information supplied during the current session.
5. Tomorrow's daily note, if it already exists.

Identify:

- work completed today,
- problems solved,
- key results,
- unresolved problems,
- unanswered questions,
- unfinished tasks,
- newly created tasks,
- tomorrow's meetings,
- reminders or notes for tomorrow.

Do not invent information.

---

## Due-Today Reminder

Before reconciling tasks, check for anything due today.

1. Scan today's daily note's to-do items and every affected project's `Tasks/<Project Name>.md` `Now` section for a due date matching today.
2. Recognize due dates written inline — "due today", "due Monday", "due tomorrow (Tue, Aug 11) 3:00 PM ET", or an explicit date matching today's date.
3. For each item due today, ask the user directly whether it was done. Do not assume completion from silence, tone, or the day's session notes alone.
4. If the user confirms completion, carry that into `Reconcile the Canonical Task List` as a completed task.
5. If the user says it is not done, ask whether to keep it in `Now`, move it to `Waiting`, or carry it into tomorrow — do not silently drop it.
6. If nothing is due today, skip this step without comment.

---

## Finalize Today's Daily Note

Before preparing tomorrow:

1. Read the complete daily note.
2. Read every session recorded today.
3. Read the relevant task files.
4. Produce one final daily summary.

The final summary should:

- remove duplicate information,
- merge repeated observations,
- remove solved blockers,
- preserve unresolved problems,
- summarize the day's progress,
- preserve only the latest handoff,
- remain concise.

Do not modify the individual session history.

The daily summary is the authoritative summary of the entire day's research.

### Tables

Consolidation never flattens a table into prose.

- Preserve every Markdown table already in the daily note; keep it under `### Key results`.
- Merge duplicate tables of the same result into one; do not summarize them away.
- If a session recorded a tabular result as prose ("counts are 432/791 by RNA but 577/576 by
  file"), expand it into a real table during consolidation.
- Carry durable tables into the project note per the Tables section of `research-update`.


## Reconcile the Canonical Task List

Task files are the canonical source of actionable work.

For each affected project:

1. Read `Tasks/<Project Name>.md`.
2. Identify tasks completed today.
3. Move confirmed completed tasks to `Completed recently`.
4. Preserve unfinished tasks.
5. Move blocked tasks to `Waiting`.
6. Add newly identified tasks supported by today's work.
7. Remove duplicates.
8. Reorder `Now` by priority.
9. Keep at most five tasks in `Now`.
10. Preserve user-written tasks unless they are obsolete.

If task changes are ambiguous, present the proposed edits before applying them.

The task list should reflect the work that remains after today's research.

### Manual Deletions

Before reconciling, check `git diff` against the last commit for today's daily note.

If a task checkbox line was removed by hand from a `Tomorrow's Tasks` or `Next steps` list (not through this skill):

1. Find the matching task in that project's `Tasks/<Project Name>.md`.
2. Delete it from wherever it appears (`Now`, `Next`, or `Waiting`).
3. Do not add it to `Completed recently` — a manual deletion is not a completion.
4. Report it in the completion summary as a removed task.

If the wording changed rather than being deleted outright, or intent is unclear, ask before deleting it from the canonical file.

---


## Prepare Tomorrow's Daily Note

If tomorrow's note does not exist:

1. Locate the canonical daily template.
2. Create the new daily note from that template.
3. Preserve all formatting, frontmatter, placeholders, links, and headings.

If tomorrow's note already exists:

1. Read it.
2. Preserve all existing content.
3. Merge planning information.
4. Avoid duplicate headings.

Never maintain a second daily-note template inside this skill.

## Tomorrow's Tasks

Tomorrow's task section is a **snapshot** of the work you intend to begin with.

The canonical work list always remains:

`Tasks/<Project Name>.md`

When populating tomorrow's note:

1. For each affected project, copy only the highest-priority tasks from its `Now`.
2. Group the copied tasks under a `[[Project Name]]` sub-heading per project. Omit the sub-heading only when exactly one project was affected today.
3. Include at most five tasks per project.
4. Preserve their order from the task file.
5. Do not copy `Waiting`, `Ideas`, or the full backlog.
6. Add a link to each project's canonical task file.

If tomorrow's note already contains planned tasks:

- merge intelligently, keeping the per-project grouping,
- avoid duplicates,
- preserve user-written tasks.

---

## Deferring a Project

The user may explicitly exclude a project from tomorrow's snapshot — e.g. "I don't want to work on digtRNA tomorrow," "skip X tomorrow," "leave Y out tomorrow."

When this happens:

1. Omit that project's sub-heading and tasks from tomorrow's `Tomorrow's Tasks` section.
2. Do not modify that project's canonical task file — its `Now` list is untouched.
3. Do not write the deferral as a flag, tag, or note in any file a future `research-end` run would read as input. The exclusion applies to tomorrow's note only.
4. Add one line under tomorrow's `Notes and Reminders` stating the project was deferred by request, so it reads as intentional rather than forgotten — e.g. "digtRNA deferred by request; resumes in the normal rotation after tomorrow."
5. Take no other action. The next `research-end` run reads the canonical task file fresh, with no memory of this exclusion — the deferred project reappears in the day-after's `Tomorrow's Tasks` unless the user defers it again at that time.

This is a per-run instruction, not a persistent setting. A project deferred for more than one day requires the user to say so on each run, or to ask for it to be moved to `Waiting` in the canonical task file as a durable change.

---

## Meetings

If tomorrow contains known meetings, populate the meeting section.

Meetings may come from:

- the user's instructions,
- today's research notes,
- trusted calendar integrations,
- existing notebook notes.

For each meeting include only information that is known:

- time,
- title,
- related project,
- preparation,
- questions to ask.

Do not invent meetings.

---

## Notes and Reminders

Carry forward only reminders that remain relevant tomorrow.

Examples:

- files to inspect,
- commands to rerun,
- expected cluster jobs,
- people to contact,
- assumptions to verify.

Do not carry forward:

- completed reminders,
- obsolete notes,
- duplicated information already present in the project note.

## Pending Work

Carry forward work that may continue after today's session.

Examples include:

- running Slurm jobs,
- background scripts,
- overnight analyses,
- files expected to be generated,
- experiments awaiting results,
- emails or responses expected from collaborators.

For each pending item, record only information that is known.

Include when available:

- description,
- relevant project,
- job ID,
- machine,
- expected output,
- expected completion or next check,
- follow-up action.

Example:

### Pending Work

- Slurm job 18492031 running on Cannon.
  - Project: [[Chemical Probing G6]]
  - Expected output: Benchmark likelihoods.
  - Tomorrow: Check completion and inspect logs.

Do not invent job IDs, file names, or expected outputs.

---

## Starting Context

Populate a short "Starting Context" section that allows work to resume immediately.

Include only:

- where work stopped,
- the first task to resume,
- important machine-specific context,
- pending jobs,
- important constraints.

Keep this section to a few bullets.

## Artifact Check

Before closing the day:

1. Determine whether today's artifacts have already been archived.
2. If not, perform the normal artifact archival workflow.
3. Do not archive temporary files.
4. Preserve original artifact locations.

## Project Notes

Update project notes only if today's work changed durable project knowledge.

Examples include:

- new design decisions,
- important experimental conclusions,
- abandoned approaches worth remembering,
- updated milestones.

Do not update project notes for routine daily progress.

Do not add entries to `Open research questions` or `Future directions` unless the user explicitly raised them during the session. Never synthesize new questions or directions from today's work.

---

## Confirm Before Closing

Before running the Git "Before Committing" steps, show the user the reconciled to-do list exactly as it will be written — for each project touched today, its `Now`, `Next`, and `Waiting` sections from `Tasks/<Project Name>.md` after reconciliation.

Ask:

> Does this look right? Anything wrong, or anything you want added before I close the day?

Wait for the user's response before continuing.

If the user requests changes:

1. Apply them to the task file(s).
2. Re-show the updated section if the change was non-trivial.

Do not commit or push until the user has confirmed or explicitly says to proceed.

---

## Writing Style

- Prefer bullets.
- Keep bullets under twenty words whenever possible.
- One idea per bullet.
- Remove repetition.
- Prefer verbs for tasks.
- Prefer nouns for summaries.
- Preserve technical terminology.
- Omit empty sections when appropriate.
- Tables are exempt from bullet-length rules — keep them whole.

---

## Editing Rules

- Preserve all session history.
- Preserve template formatting.
- Preserve user-written notes unless replacing obsolete planning information.
- Never invent meetings, commands, results, or tasks.
- Never duplicate the canonical task list.
- The daily note contains only a planning snapshot.
- The task file remains the source of truth.
- Preserve wiki links whenever possible.

Before finishing, ask:

"If I opened VS Code tomorrow and ran `research-start`, would I immediately know what to do?"

If not:

- improve the handoff,
- improve tomorrow's note,
- improve the canonical task list,

before completing the workflow.



---

## Completion Summary

Report:

### Research day closed

- Daily summary: Updated / Not updated
- Tomorrow's note: Created / Updated
- Canonical task list: Updated / Not updated
- Tasks carried forward: <number>
- Completed tasks: <number>
- Projects deferred: <number, with names, or "None">
- Meetings added: <number>
- Notes carried forward: <number>
- Git commit: <hash or "No commit created">
- Remote synchronization: Pushed / Not pushed>

If any planned action was skipped or failed, explain why in one concise sentence.
