---
name: research-start
description: Restore research context and manage the active project work list. Use when the user wants to resume research, review priorities, add or remove tasks, mark work complete, or decide what to work on next.
disable-model-invocation: true
---

# Configuration

## Locate the research notebook

Check these locations in order:

1. `/Users/romanagle/Research`
2. `~/research/notebook`
3. The current working directory, if it contains:
   - `AGENTS.md`
   - `Projects/`
   - `Tasks/`

Use the first valid location.

If no valid notebook exists, ask for the notebook path and stop.

### Repository guardrail

Before any Git operation, run `git remote get-url origin`. Continue only when the remote identifies `romanagle/research-notebook.git`. If it identifies a code repository or any other remote, stop and locate the standalone Research vault; never commit or push research outputs there unless the user explicitly names that repository.

---

# Research Start

## Purpose

Restore working context quickly and maintain a trustworthy research work list.

This skill may be used:

- at the beginning of the day,
- after a break,
- after switching projects,
- after changing machines,
- whenever the user needs to resume work.

It is not limited to one use per day.

---

## Core Principles

- The task file is the canonical source of actionable work.
- The project note is the canonical source of durable project state.
- The daily note provides recent session context.
- Keep the briefing short.
- Do not invent priorities.
- Do not modify project or daily notes. Creating today's note when it is missing (see Today's Daily Note) is creating a file, not modifying an existing one, so it is not an exception to this rule.
- Modify task files only when the user requests or approves a task change.

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

6. After the pull succeeds, check whether the diff removed a task checkbox line from a daily note's `Tomorrow's Tasks` or `Next steps` list. If so, delete that same task from the corresponding project's `Tasks/<Project Name>.md` (from `Now`, `Next`, or `Waiting`, not `Completed recently`) as part of resuming work, and mention it in the completion summary. If the change is ambiguous, ask before deleting.

### Before committing

1. Inspect the repository:

   `git status --short`
   `git diff`

2. Stage only files intentionally changed by this skill. Do not use `git add .` or `git add -A` unless every changed file has been reviewed.
3. Never stage `.obsidian/workspace.json` or `.obsidian/workspace-mobile.json`.
4. Commit using:

   `Tasks: <Project Name> — <short change summary>`

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

## Today's Daily Note

Before identifying the active project, check whether `Daily/YYYY-MM-DD.md` exists for the current local date.

If it exists, continue to Read Context normally.

If it does not exist, `research-end` was not run at the end of the previous session. Recover by creating today's note the same way `research-end` prepares the next day's note (see its "Prepare Tomorrow's Daily Note", "Tomorrow's Tasks", "Meetings", "Notes and Reminders", "Pending Work", and "Starting Context" sections):

1. Treat the most recent existing daily note as the basis, the same way `research-end` treats today's note when preparing tomorrow's.
2. Read that note and the `Now` list of each project's `Tasks/<Project Name>.md` it mentions.
3. Create `Daily/YYYY-MM-DD.md` from the canonical daily template, following `research-end`'s rules for Tomorrow's Tasks, Meetings, Notes and Reminders, Pending Work, and Starting Context — substituting "today" for "tomorrow" throughout.
4. Carry forward only meetings, reminders, and pending work that are still relevant.
5. Do not modify the most recent existing daily note — only create the new note for today.
6. Do not touch any project's canonical task file during this recovery. Reconciling completed and unfinished tasks is `research-end`'s job, not `research-start`'s.
7. Commit and push the new note following Git Synchronization's Before Committing / Before Pushing steps, using the message:

   `Daily: YYYY-MM-DD — created by research-start (research-end not run)`

8. Note at the start of the briefing that today's note was created because `research-end` appears to have been skipped the previous day.

Never maintain a second daily-note template inside this skill — reuse `research-end`'s template and section rules rather than redefining them here.

---

## Identify the Active Project

Determine the project from:

1. The user's explicit request.
2. The current code repository.
3. The latest daily note.
4. The most recently updated task or project note.

If multiple projects are plausible, ask which project to resume.

If the user asks for an overview rather than a single project — e.g. "what should I work on", "show my tasks", "show all my tasks" — brief across **all active projects** instead of picking one. A project counts as active if `Tasks/<Project Name>.md` has any item under `Now`, `Next`, or `Waiting`.

Always reuse the exact canonical project name from `Projects/`.

---

## Read Context

For a single-project briefing, read:

1. `Tasks/<Project Name>.md`
2. `Projects/<Project Name>.md`
3. Today's daily note (created via Today's Daily Note if it did not already exist).
4. If today's note was created by research-end, or by this skill's Today's Daily Note recovery, also read its planning sections.
5. Otherwise read the most recent daily note mentioning this project.

For an all-projects overview, read `Tasks/<Project Name>.md` for every active project (skip `Projects/*.md` and daily notes unless the user asks for more detail on one project).

Focus on:

- current objective,
- latest stopping point,
- active blockers,
- unresolved questions,
- immediate next steps,
- waiting dependencies,
- explicit decisions.

Do not replay the full session history.

Also read the most recent Pending State section from the latest daily note.

If present, identify:

- running jobs,
- overnight computations,
- expected outputs,
- collaborators or emails awaiting response,
- external dependencies that may have changed,
- anything the user should check before beginning work.

---

## Create a Missing Task File

If `Tasks/<Project Name>.md` does not exist:

1. Create it from `Templates/Project Tasks.md`.
2. Replace `Project Name` with the canonical project name.
3. Populate only tasks clearly supported by existing notes or user instructions.
4. Do not invent work items.
5. If no tasks can be inferred confidently, create the headings and leave them empty.

---

## Briefing Format

Produce a concise briefing in this format:

### Research context

**Project:** [[Project Name]]

**Current objective**
- One concise statement.

**Pending state**

- Running Slurm jobs.
- Expected outputs.
- Collaborators awaiting response.
- Other work that may have changed while away.

**Continue here**
- The most immediate next action.

**Now**
- [ ] Up to three current tasks.

**Current blockers**
- Only active blockers.

**Open questions**
- Only questions relevant to the current work.

**Waiting**
- Only blocked or externally dependent items.

**Do not forget**
- One or two important decisions, failed approaches, or constraints — only items already explicitly recorded in project or daily notes, never synthesized during this briefing.

End with one question:

> What would you like to accomplish in this session?

Omit empty sections.

Do not include time estimates unless the user asks.

---

## All-Projects Overview Format

Use this format instead of the single-project briefing when the user asks for an overview across projects.

Stratify `Now` and `Waiting` by project, ordered by most recently updated task file first:

### Research overview

**Now**

**[[Project Name]]**
- [ ] Up to three current tasks.

**[[Project Name 2]]**
- [ ] Up to three current tasks.

**Waiting**

**[[Project Name]]**
- Only blocked or externally dependent items.

End with one question:

> Which project would you like to work on?

Omit any project with no items in the relevant section. Do not include `Current blockers`, `Open questions`, or `Do not forget` in the overview — those stay in the single-project briefing.

---

## Task File Structure

Each project task file should use:

```markdown
# [[Project Name]]

## Now

- [ ] Immediate work.

## Next

- [ ] Work that follows current priorities.

## Waiting

- [ ] Blocked or externally dependent work.

## Ideas

- Possible directions that are not yet commitments.

## Completed recently

- Recently completed work.
```

Maintain exactly one copy of each heading.

---

## Task Management Commands

Interpret natural-language requests such as:

- “Add this to my work list.”
- “Add a task to benchmark RNaseP.”
- “Move this to Waiting.”
- “Move this to Next.”
- “Mark parser cleanup complete.”
- “Remove that task.”
- “What should I work on next?”
- “Show my tasks for DIGtRNA.”
- “Show all my tasks.” / “What's on my plate across everything?” — use the All-Projects Overview Format.

### Add

Add the task to the requested section.

If no section is specified:

- use `Now` for immediate work,
- use `Next` for later actionable work,
- use `Waiting` for blocked work,
- use `Ideas` for speculative directions.

Avoid duplicates.

### Move

Remove the existing task from its current section and add it to the requested section.

Preserve the wording unless clarification improves precision.

### Complete

When a task is completed:

1. Remove it from `Now`, `Next`, or `Waiting`.
2. Add a concise version under `Completed recently`.
3. Keep no more than ten items under `Completed recently`.
4. Remove the oldest entries when the section grows beyond ten.

### Remove

Delete the requested task.

Do not preserve removed tasks elsewhere unless the user asks.

### Edit

Update the existing task instead of creating a duplicate.

### Prioritize

Reorder tasks within `Now`.

Put the highest-priority task first.

Keep no more than five items in `Now`.

If `Now` exceeds five items, suggest moving lower-priority items to `Next`.

---

## Writing Style

- Prefer short bullets.
- Keep each task action-oriented.
- Start tasks with a verb.
- Keep tasks under 15 words when possible.
- Avoid vague tasks such as “work on analysis.”
- Prefer specific wording such as “Compare paired and unpaired likelihood distributions.”
- Do not duplicate the same task across sections.

---

## Editing Rules

- Never modify daily notes.
- Never modify project notes.
- Never mark a task complete without evidence or user confirmation.
- Never remove a task unless the user requests it or approves it.
- Do not convert every research question into a task.
- Do not add entries to `Ideas` unless the user explicitly states the idea. Never synthesize speculative directions from context.
- Do not synthesize new items for the briefing's `Do not forget` section. Surface only decisions, failed approaches, or constraints already explicitly recorded in existing notes.
- Preserve distinctions between tasks, waiting items, and ideas.
- Use the canonical project name and capitalization.
- Make the smallest necessary edit.

---

## Completion Summary

After changing tasks, report briefly:

### Work list updated

- Project: [[Project Name]]
- Added: <number>
- Completed: <number>
- Moved: <number>
- Removed: <number>
- Git commit: <hash or "No commit created">
- Remote synchronization: Pushed / Not pushed

For a read-only briefing, do not commit or push.
