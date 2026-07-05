# Importing OKF knowledge into prompt chains

`OKFLoader` lets a PromptChain **import knowledge** from an OKF bundle — the way `PrePrompt` imports named
prompt templates. Because OKF is a *known format*, PromptChain understands the bundle's structure and can
inject it at the right granularity.

## What is OKF?

**OKF (Open Knowledge Format)** is a vendor-neutral way to represent knowledge as a **directory of markdown
concept files**:

- **One concept per file.** The file path (minus `.md`) is its **Concept ID** (e.g. `tests/mcnemar`).
- **YAML frontmatter** with a required `type:` (e.g. `StatisticalTest`), plus optional `title`,
  `description`, `tags`.
- **Reserved files:** `index.md` (a curated directory listing / progressive-disclosure map — *no*
  frontmatter) and `log.md` (history — *no* frontmatter).
- **Links** between concepts are bundle-relative markdown links (`[McNemar](/tests/mcnemar.md)`); a link
  asserts a relationship.

A shipped example lives at `promptchain/validity/okf/` (the experiment-validity knowledge). Get its path
with `promptchain.validity.okf_path()`.

## Quick start

```python
from promptchain import OKFLoader, okf_step, okf_agentic_context, okf_reader_tool
from promptchain.validity import okf_path

ld = OKFLoader(okf_path())          # point at any OKF folder (the "folder target")
ld.concepts()                        # ['checks/above-noise', 'tests/mcnemar', 'validation-workflow', ...]
ld.get("tests/mcnemar").type         # 'StatisticalTest'
ld.load("tests/mcnemar")             # the concept's body (frontmatter stripped)
```

## The two injection modes

Choose granularity by the kind of step:

### 1. Sequential step — import a SPECIFIC concept (targeted)

Inject one (or a few) concept file(s) directly into an instruction:

```python
from promptchain import PromptChain, okf_step

chain = PromptChain(instructions=[
    okf_step(ld, ["tests/mcnemar"], "decide whether this delta is statistically significant."),
])
```
`okf_step(loader, concept_ids, task)` returns a ready instruction string: the concept bodies followed by
your task. Use this when the step needs *specific* knowledge.

### 2. Agentic step — give the FOLDER + let the agent navigate (progressive disclosure)

For an `AgenticStepProcessor`, give it the **outline** (what concepts exist) as always-on context, plus an
`okf_read(id)` tool so it pulls concepts **on demand** — OKF's progressive-disclosure purpose:

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain import okf_agentic_context, okf_reader_tool

asp = AgenticStepProcessor(
    objective="Validate this experiment before we report the result.",
    instructions=[okf_agentic_context(ld, mode="outline")],   # the MAP: id/type/title/description
)
asp.register_tool(okf_reader_tool(ld))                        # okf_read(concept_id) -> body, on demand
```
Use `mode="full"` to inline all (or selected) concept bodies instead of the navigable outline — good for a
small bundle you want fully in-context.

## OKFLoader API

| Method | Returns |
|---|---|
| `OKFLoader(*bundle_dirs)` | loader over one or more bundle folders |
| `.concepts()` | sorted list of Concept IDs |
| `.get(cid)` | `OKFConcept` (`.type` `.title` `.description` `.tags` `.body` `.links`) |
| `.load(cid)` | the concept body text (mirrors `PrePrompt.load`) |
| `.tree()` | nested dict of the bundle's concept tree |
| `.outline()` | navigation map: ``- `id` [Type] Title — description`` per concept |
| `.index_md(subdir=".")` | the curated `index.md`, if authored |
| `.render(ids, header=True)` | concept bodies joined (optionally with `## Title` headers) |
| `.context(ids, title=...)` | a titled reference block |
| helper `okf_step(ld, ids, task)` | sequential-step instruction string |
| helper `okf_agentic_context(ld, ids, mode)` | agentic context (`outline` map or `full`) |
| helper `okf_reader_tool(ld)` | `okf_read(id)` callable to register as an agent tool |

## OKF vs PrePrompt — which to use?

| | PrePrompt | OKFLoader |
|---|---|---|
| Unit | a named prompt **template** | a **concept** (typed knowledge) |
| Structure | flat prompt IDs | a **tree** with types, links, an index |
| Best for | reusable instruction text / strategies | structured knowledge an agent navigates |
| Injection | resolve an instruction ID to text | targeted concept (step) or outline+read-tool (agentic) |

Reach for OKF when the knowledge is *structured* (multiple related concepts with types and cross-links)
and you want an agent to navigate it; reach for PrePrompt for a single named prompt.

## Authoring a bundle

Put concept files under a folder, one concept per file, each with `type:` frontmatter; add an `index.md`
(no frontmatter) listing them; optionally a `log.md`. See `promptchain/validity/okf/` for a conformant
example, and the OKF spec for the full rules.
