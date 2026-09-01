---
name: explaining-patterns
description: Investigate why a plot, chart, or summary statistic looks the way it does by building a differential of falsifiable hypotheses and checking each against the real data. Use when the user asks why a graph looks a certain way, wants hypotheses tested for a pattern, or flags an unexpected shape, outlier, trend, or cluster in a plot or stat.
---

# Explaining Patterns

Borrow the clinician's differential diagnosis: hold several candidate explanations open at once, rank them by how cheap they are to rule out, and let a check against real data close each one. Commit to whichever explanation survives — never to the first plausible story.

## Out of scope

- A suspected code or pipeline bug producing wrong output → `/diagnosing-bugs`
- A UI or state design question → `/prototype`

## Step 1 — Pin the observation

State exactly what's seen: the shape, magnitude, location, and which subset of the data shows it. "The tail is heavy" is not pinned; "seqs over 400nt show 3x the DMS reactivity of shorter ones, driven by the SHAPE subset only" is.

Done when the observation is specific enough that someone who never saw the plot could reproduce your description from the data alone.

## Step 2 — Open the differential

Brainstorm across categories before narrowing to anything:

- **Domain** — a real biological/physical effect
- **Statistical artifact** — regression to the mean, Simpson's paradox, small-n variance
- **Sampling / coverage** — which seqs or conditions are over/under-represented
- **Data quality** — bad reads, mis-parsed fields, unit mismatches
- **Methodological** — how the stat or plot itself was constructed (binning, normalization, aggregation choice)

Done when the differential spans at least three of these five categories — a differential drawn from a single category is a straight line, not a differential.

## Step 3 — Sharpen each entry into a falsifiable prediction

For every hypothesis still on the differential, state one sentence: "if true, the data would show ___; if false, ___." Drop any entry that can't produce this sentence — it's a vibe, not a hypothesis.

Done when every surviving entry has its if-true/if-false sentence written down.

## Step 4 — Order by cheapest check first

Rank the differential so a quick computational check (subsetting, recomputing a stat, a comparison plot) runs before anything that requires new data, a collaborator's input, or rerunning a pipeline.

## Step 5 — Run the check against the real data

For the top entry, execute the check and paste its literal output. A verdict is only earned by an executed check — reasoning about what the data would probably show does not close an entry.

Repeat down the ordered differential.

Done when every entry remaining on the differential has an executed check with recorded output.

## Step 6 — Record verdict + evidence

For each hypothesis: **confirmed**, **ruled out**, or **inconclusive**, plus the specific evidence (the number, subset, or comparison) that drove the call.

## Step 7 — Reopen the differential if everything's ruled out

If every hypothesis is ruled out, return to Step 2 informed by what the checks revealed — don't settle for the least-bad survivor. A ruled-out hypothesis's evidence usually points at the next category to search.

## Report

Close with a table, most-checked-first, ending in the surviving explanation(s):

| Hypothesis | Check | Verdict | Evidence |
|---|---|---|---|
| ... | ... | confirmed / ruled out / inconclusive | ... |
