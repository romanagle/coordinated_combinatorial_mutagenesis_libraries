---
name: narrative-map
description: Build or revise a manuscript narrative from existing analyses, mapping figures to Methods, hierarchical Results parts and subpoints, Supplement, Discussion, and Conclusion while tracking evidence holes and project tasks. Use when starting or restructuring a manuscript, editing result subpoints, arranging figures, creating explicit figure placeholders, or resuming narrative work.
---

A manuscript is a fixed shape: a **question**, the **methods** that answer it, **results** broken into subsections, each arguing one subpoint off a specific graph, a **discussion** and **conclusion** that accumulate only after themes emerge from the results, and a **supplement** for supporting analyses that have a settled role outside the main argument. An analysis either has a main-text or supplement section to live in, or it belongs under **Revisit** until its role is resolved. A subsection with no analysis behind it yet is a **hole**. Naming these turns a pile of graphs into a structure you can keep coming back to.

**Interpretation is never yours to supply.** For every graph, ask what it shows and record the user's own words. You propose structure; the user owns every claim about what the data means.

**Always link the graph being discussed.** Whenever asking the user what a graph shows or requesting any other figure-level decision, include a clickable Markdown link to the exact figure. Ask about one graph at a time; if a decision necessarily covers a matched set, link every figure in that set.

**Defer presentation advice until the inventory is interpreted.** Only after the user has explained the intended message of every selected analysis, suggest ways to improve individual graphs or group graphs into figures. Ground every suggestion in the user's stated messages; never use presentation advice to introduce a new scientific interpretation.

**Prototype every redesign suggestion.** Do not offer a graph redesign or grouping only in prose. First build a clearly labeled throwaway visual prototype from the real data when available, without changing canonical analysis code or outputs. Link the rendered prototype when presenting the suggestion, state any data substitutions or omissions, and wait for the user's reaction before treating the design as accepted.

**Every entry is a bullet or a one-sentence claim — never a paragraph.** The whole point is a document you can scan in ten seconds. If a section is growing into prose, that's the signal to split it into another subpoint.

**Distinguish observations, hypotheses, and placeholders.** State completed results as observations only after the user interprets the real figure. Write predicted patterns as expectations. Label simulated or empty visuals as placeholders, never as results; record a hole requiring their replacement.

**Distinguish manuscript holes from project tasks.** A hole is missing evidence required by the manuscript argument and belongs in `NARRATIVE.md`. A to-do is actionable project work and belongs in the project's canonical task file. If the user says “add a to-do,” do not add it to Holes. Keep linked hole and task wording consistent when one analysis is both.

**Use exact metric language.** Preserve evaluation-library names and define subtraction direction for deltas. Do not call cosine similarity a magnitude metric; it measures alignment. Define robustness concretely, such as mean ± SD across initializations, before building a summary table.

## Out of scope

Running the analysis that fills a hole is the user's work, not this skill's. Identify the hole, discuss what would fill it, then wait for the user to bring back a finished graph.

## Editing protocol

- If `NARRATIVE.md` is open in the user's editor, ask them to save before the first write and avoid typing during each patch.
- Verify the edited passage from disk after every write when editor synchronization has already caused an overwrite.
- Move prerequisite validation evidence into Methods when the user says it establishes that the experimental setup is usable rather than making a manuscript finding.
- Preserve a subsection when moving its figure unless the user explicitly asks to remove or merge the subsection.

## The file

State lives in `NARRATIVE.md` at the project root, created once the question is pinned:

```markdown
## Question

<domain/background, in a few bullets — what field, what's already known>
<the specific question this manuscript answers, one sentence>

## Methods

- <analysis/method> — <one line: what it measures>

## Results

### Part I. <major argumentative stage>

#### 1. <subsection — one subpoint of the argument>

<one sentence: the claim this subsection makes>

- [<graph>](path) — <the user's own words on what it shows>

## Discussion

## Conclusion

## Holes

- <subsection>: <what's missing, specific enough to know when it's filled>

## Supplement

### <supplementary subsection — one supporting subpoint>

<one sentence: the supporting claim this subsection makes>

- [<graph>](path) — <the user's own words on what it shows>

## Revisit

- [<graph>](path) — <why it has no section; disposal: parked / cut / promoted to a new subsection>
```

## Build the structure (no `NARRATIVE.md` yet)

1. **Pin the question.** Run a `/grilling` session on the general domain and the specific question this manuscript answers. Done when the user confirms both in a few sentences.

2. **Inventory the analyses.** Ask which analyses are done and which graphs belong in this manuscript — not everything ever run, just what's on the table now. For each, ask what it shows and record the answer verbatim. Done when every selected graph carries the user's own interpretation, not yours.

3. **Prototype figure presentation.** Once every selected analysis has the user's interpretation, build throwaway prototypes of clearer graph designs and figure groupings that serve those messages. Present each suggestion through its linked prototype, not prose alone. Do not create or offer redesigns earlier. Use deterministic plotting for scientific layouts, empty boxes, and simulated expectations; never use generative imagery to fabricate scientific data.

4. **Propose a structure.** From the question and the inventory, draft a Methods list, a Results outline, and a Supplement outline — subsections, each one subpoint, each grounded in specific graphs already inventoried. When the argument has distinct stages, group Results into named Parts and restart subpoint numbering inside each Part. Add empty Discussion and Conclusion headings; populate them only when the user identifies themes supported by the Results structure. The supplement supports the manuscript but does not carry a necessary step in the main argument. This is your proposal to react to, not a final answer.

5. **Tweak until it holds.** Grill on the proposal one question at a time — reorder, split, merge, rename — per `/grilling`'s rule: don't lock it in until the user confirms shared understanding.

6. **Recognize the holes.** Any subsection with no supporting graph is a hole — list it. Put any inventoried graph without a settled main-text or supplement subsection under Revisit and ask whether to fold it into the main text, place it in the supplement, promote it to a new subsection, park it, or cut it. Done when every subsection has a graph or a hole entry, and every inventoried graph is placed or listed under Revisit.

7. Write `NARRATIVE.md`.

## Figure and table decisions

- Treat a prototype as accepted only after the user approves it; link only the accepted version from `NARRATIVE.md`.
- Keep rejected variants as throwaway work unless the user asks to retain them; never present them as selected evidence.
- For a placeholder, show only the requested layout and labels. Use `TBD`, `N/A`, or empty boxes rather than invented values.
- For an illustrative expected plot, use reproducible simulated points, label the entire figure as simulated, and display no value as experimental.
- For grouped tables, put experimental conditions in rows and landscapes or systems in grouped columns when that makes comparisons scan cleanly.
- Gray structurally impossible cells as `N/A`; use `TBD` for values that exist conceptually but have not been computed.
- Combine robustness with the reported metric when possible, such as `mean ρ ± SD`, instead of adding a vague robustness column.
- If a requested interpretation is not supported by current values, prototype the real values and report the contradiction before updating the narrative claim.

## Consult the map (`NARRATIVE.md` exists)

The file is the guiding light from here on — check new work against it before anything else. Whenever the user brings a new analysis, whether it targets a listed hole or came from somewhere else entirely:

1. Ask what it shows; record their words.
2. Check it against the structure: does it close a hole, support an existing main-text or supplement subsection, or lack a settled role? Put an unresolved graph under Revisit and ask for its disposition.
3. Update `NARRATIVE.md` — move it out of Holes, add it under its main-text or supplement subsection, or log it under Revisit.
4. If the new analysis reveals a subsection the structure was missing, add it and note whether it opens fresh holes of its own.
5. When the user identifies a broader implication, limitation, or takeaway grounded in the developing Results structure, add it as a bullet under Discussion or Conclusion in the user's own words.

## Session handoff

When the user stops narrative work:

1. Report the exact Part and subpoint where work stopped.
2. Reconcile `NARRATIVE.md` Holes with the canonical project task file without duplicating one category into the other.
3. Use the research-update workflow when the user asks to log or preserve the session.
4. Before any Git operation, verify `git remote get-url origin` identifies `romanagle/research-notebook.git`; stop if it identifies a code repository.
5. Commit and push only when explicitly requested, staging only narrative, accepted prototypes, and notebook handoff files in the standalone Research vault.
6. Preserve unrelated working-tree changes.

## Report

Close with the Results and Supplement outlines, each subsection marked supported or holed, so the user sees the manuscript's shape and what's still missing at a glance.
