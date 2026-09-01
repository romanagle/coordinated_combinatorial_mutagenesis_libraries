#!/usr/bin/env bash
#
# Link every notebook skill into the agent skill roots.
#
# The links are per-skill, not a link of Skills/ itself, so `git pull` keeps
# existing skills up to date automatically but a NEW skill directory is invisible
# until this script runs. Run it after any pull that adds a skill.
#
# Usage:
#   bash Skills/link-skills.sh          # link into every root that applies
#   bash Skills/link-skills.sh --check  # report only, change nothing
#
# Restart Claude Code / Codex afterwards; both build their skill list at session start.

set -euo pipefail

SKILLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Skill roots. Claude Code reads ~/.claude/skills. Codex reads both ~/.agents/skills
# and ~/.codex/skills; we populate both so the skills survive either convention.
ROOTS=(
    "$HOME/.claude/skills"
    "$HOME/.agents/skills"
    "$HOME/.codex/skills"
)

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

linked=0
skipped=0
missing=0

for skill in "$SKILLS_DIR"/*/; do
    skill="${skill%/}"
    name="$(basename "$skill")"

    if [ ! -f "$skill/SKILL.md" ]; then
        echo "skip   $name (no SKILL.md)"
        continue
    fi

    for root in "${ROOTS[@]}"; do
        # Only populate a root that exists, or that we own entirely (~/.claude, ~/.agents).
        # Never create ~/.codex/skills from scratch -- if Codex is not installed we would
        # be inventing a directory it does not read.
        if [ ! -d "$root" ]; then
            case "$root" in
                "$HOME/.codex/skills")
                    continue
                    ;;
            esac
            if [ "$CHECK_ONLY" = 1 ]; then
                echo "would create $root"
                continue
            fi
            mkdir -p "$root"
        fi

        target="$root/$name"

        # A real directory here is someone else's skill of the same name -- do not
        # clobber it, and do not let `ln` drop the link *inside* it.
        if [ -d "$target" ] && [ ! -L "$target" ]; then
            echo "CONFLICT  $target is a real directory, not a link -- left alone"
            skipped=$((skipped + 1))
            continue
        fi

        if [ -L "$target" ] && [ "$(readlink "$target")" = "$skill" ]; then
            continue
        fi

        if [ "$CHECK_ONLY" = 1 ]; then
            echo "would link $target -> $skill"
            missing=$((missing + 1))
            continue
        fi

        ln -sfn "$skill" "$target"
        echo "link   $target -> $skill"
        linked=$((linked + 1))
    done
done

if [ "$CHECK_ONLY" = 1 ]; then
    echo
    echo "$missing link(s) missing, $skipped conflict(s)"
    [ "$missing" -eq 0 ] && [ "$skipped" -eq 0 ]
    exit $?
fi

echo
echo "$linked link(s) created or updated, $skipped conflict(s)"
echo "Restart Claude Code and Codex to pick up new skills."
