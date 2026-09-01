# Research Notebook Setup

Use these instructions to configure this research notebook on a new computer, cluster, or agent environment.

## Repository

GitHub repository:

`git@github.com:romanagle/research-notebook.git`

The repository is private. Do not make it public.

## Preferred locations

Use the first location appropriate for the current system:

- macOS: `~/Research`
- HPC or Linux: `~/research/notebook`

Do not create a second notebook copy if one already exists.

## Before setup

1. Determine the operating system.
2. Check whether the preferred notebook location already exists.
3. Search for an existing clone before creating a new one.
4. If an existing clone is found, confirm its Git remote.
5. Preserve all existing files and uncommitted changes.

## Clone or update the notebook

If the notebook does not exist:

```bash
mkdir -p ~/research
git clone git@github.com:romanagle/research-notebook.git ~/research/notebook
```

On macOS, use:

```bash
git clone git@github.com:romanagle/research-notebook.git ~/Research
```

If the notebook already exists:

```bash
cd <notebook-root>
git status --short
git remote -v
git pull --rebase
```

If uncommitted changes or conflicts exist, stop and report them. Never discard them automatically.

## Verify the notebook

The notebook root should contain:

- `AGENTS.md`
- `Daily/`
- `Projects/`
- `Artifacts/`
- `Templates/`
- `Skills/research-update/SKILL.md`

If an expected empty directory is absent, create it with a `.gitkeep` file.

## Configure skills (Claude Code and Codex)

Both agents read skills through per-skill symlinks into this repo. Run:

```bash
bash <notebook-root>/Skills/link-skills.sh
```

This links every `Skills/<name>/` directory into each skill root that applies:

| Root | Read by |
| --- | --- |
| `~/.claude/skills/` | Claude Code |
| `~/.agents/skills/` | Codex |
| `~/.codex/skills/` | Codex (alongside its own installed skills) |

`~/.claude/skills` and `~/.agents/skills` are created if absent. `~/.codex/skills` is
only populated when it already exists, so the script does not invent a directory on
machines without Codex. A real (non-symlink) directory already occupying a skill name is
reported as a conflict and left untouched.

To report what is missing without changing anything:

```bash
bash <notebook-root>/Skills/link-skills.sh --check
```

It exits non-zero if any link is missing or conflicting.

### After a pull that adds a new skill

Because the links are per-skill rather than a symlink of `Skills/` itself, `git pull`
updates existing skills automatically but a **new** skill directory stays invisible until
`link-skills.sh` is run again. Symptom: the agent does not recognize the skill and may try
to run its name as a shell command. Re-run the script, then restart the agent — both
Claude Code and Codex build their skill list at session start.

Verify Codex loaded a skill without starting a session:

```bash
codex debug prompt-input | grep <skill-name>
```

`codex doctor` does not report skill roots.

## GitHub authentication

Test authentication:

```bash
ssh -T git@github.com
```

If authentication fails:

1. Check for an existing GitHub SSH key.
2. Do not overwrite existing keys.
3. If needed, create a new machine-specific Ed25519 key.
4. Display only the `.pub` key for the user to add to GitHub.
5. Never display or copy the private key.
6. Configure `~/.ssh/config` to use the correct key.
7. Retest authentication before cloning or pushing.

## Final verification

Confirm:

```bash
cd <notebook-root>
git status
git remote -v
```

Confirm every skill link resolves:

```bash
bash <notebook-root>/Skills/link-skills.sh --check
```

Report:

- Notebook location
- Git remote
- Current branch
- Claude skill status
- Codex skill status
- Any unresolved authentication or synchronization problems
