# Research Vault

This repository is my shared research notebook.

## Structure

- Daily/ contains chronological research session logs.
- Projects/ contains the current state of each research project.

## Principles

- This vault's only authorized Git remote is `romanagle/research-notebook.git`.
- Never commit or push research notes, narrative maps, tasks, skills, or artifacts to a code repository unless the user explicitly requests that repository.
- Before every research-workflow commit or push, verify `git remote get-url origin` identifies `romanagle/research-notebook.git`; stop if it does not.
- Daily notes are append-only.
- Project notes should always represent the latest state.
- Never delete historical information.
- When updating a project, preserve important decisions and rationale.
- Whenever an analysis produces a table, or its result condenses into one, write the actual
  Markdown table into the daily note and the project note. Never replace it with prose.

## When asked to update the research notebook

1. Determine which project(s) were worked on.
2. Append a session to today's daily note.
3. Update the relevant project note if the project's state changed.
4. If uncertain, ask before making major changes.
5. Be as concise as possible. No long paragraphs. Few sentences.
