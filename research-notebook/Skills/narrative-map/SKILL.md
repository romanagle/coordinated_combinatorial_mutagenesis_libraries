---
name: narrative-map
description: Build a manuscript's narrative structure from your existing analyses, mapping each graph to the section it supports and surfacing the holes still open. Run by hand when starting or restructuring a manuscript.
disable-model-invocation: true
---

A manuscript is a fixed shape: a **question**, the **methods** that answer it, **results** broken into subsections, each arguing one subpoint off a specific graph, a **discussion** and **conclusion** that accumulate only after themes emerge from the results, and a **supplement** for supporting analyses that have a settled role outside the main argument. An analysis either has a main-text or supplement section to live in, or it belongs under **Revisit** until its role is resolved. A subsection with no analysis behind it yet is a **hole**. Naming these turns a pile of graphs into a structure you can keep coming back to.

**Interpretation is never yours to supply.** For every graph, ask what it shows and record the user's own words. You propose structure; the user owns every claim about what the data means.

**Always link the graph being discussed.** Whenever asking the user what a graph shows or requesting any other figure-level decision, include a clickable Markdown link to the exact figure. Ask about one graph at a time; if a decision necessarily covers a matched set, link every figure in that set.

**Defer presentation advice until the inventory is interpreted.** Only after the user has explained the intended message of every selected analysis, suggest ways to improve individual graphs or group graphs into figures. Ground every suggestion in the user's stated messages; never use presentation advice to introduce a new scientific interpretation.

**Prototype every redesign suggestion.** Do not offer a graph redesign or grouping only in prose. First build a clearly labeled throwaway visual prototype from the real data when available, without changing canonical analysis code or outputs. Link the rendered prototype when presenting the suggestion, state any data substitutions or omissions, and wait for the user's reaction before treating the design as accepted.

**Every entry is a bullet or a one-sentence claim — never a paragraph.** The whole point is a document you can scan in ten seconds. If a section is growing into prose, that's the signal to split it into another subpoint.

## Out of scope

Running the analysis that fills a hole is the user's work, not this skill's. Identify the hole, discuss what would fill it, then wait for the user to bring back a finished graph.

## The file

State lives in `NARRATIVE.md` at the project root, created once the question is pinned:

```markdown
## Question

<domain/background, in a few bullets — what field, what's already known>
<the specific question this manuscript answers, one sentence>

## Methods

- <analysis/method> — <one line: what it measures>

## Results

### <subsection — one subpoint of the argument>

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

3. **Prototype figure presentation.** Once every selected analysis has the user's interpretation, build throwaway prototypes of clearer graph designs and figure groupings that serve those messages. Present each suggestion through its linked prototype, not prose alone. Do not create or offer redesigns earlier.

4. **Propose a structure.** From the question and the inventory, draft a Methods list, a Results outline, and a Supplement outline — subsections, each one subpoint, each grounded in specific graphs already inventoried. Add empty Discussion and Conclusion headings; populate them only when the user identifies themes supported by the Results structure. The supplement supports the manuscript but does not carry a necessary step in the main argument. This is your proposal to react to, not a final answer.

5. **Tweak until it holds.** Grill on the proposal one question at a time — reorder, split, merge, rename — per `/grilling`'s rule: don't lock it in until the user confirms shared understanding.

6. **Recognize the holes.** Any subsection with no supporting graph is a hole — list it. Put any inventoried graph without a settled main-text or supplement subsection under Revisit and ask whether to fold it into the main text, place it in the supplement, promote it to a new subsection, park it, or cut it. Done when every subsection has a graph or a hole entry, and every inventoried graph is placed or listed under Revisit.

7. Write `NARRATIVE.md`.

## Consult the map (`NARRATIVE.md` exists)

The file is the guiding light from here on — check new work against it before anything else. Whenever the user brings a new analysis, whether it targets a listed hole or came from somewhere else entirely:

1. Ask what it shows; record their words.
2. Check it against the structure: does it close a hole, support an existing main-text or supplement subsection, or lack a settled role? Put an unresolved graph under Revisit and ask for its disposition.
3. Update `NARRATIVE.md` — move it out of Holes, add it under its main-text or supplement subsection, or log it under Revisit.
4. If the new analysis reveals a subsection the structure was missing, add it and note whether it opens fresh holes of its own.
5. When the user identifies a broader implication, limitation, or takeaway grounded in the developing Results structure, add it as a bullet under Discussion or Conclusion in the user's own words.

## Report

Close with the Results and Supplement outlines, each subsection marked supported or holed, so the user sees the manuscript's shape and what's still missing at a glance.
