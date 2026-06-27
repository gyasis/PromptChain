# Contract: assembled-prompt layout, budget order, and rendered shapes

The deterministic structure of a prompt F3 assembles. This is the F3-internal layout contract
(how the pieces are ordered + trimmed); it does NOT change F1's transcript schema or F2's profile
schema.

## Section order (top → bottom)

1. **Static base** (parity floor — VERBATIM, never dropped): `TUI_FOUNDATION_PROMPT` with
   `{objective}` substituted, exactly as `DynamicTUIPromptGenerator` renders it.
2. **Family-adapted framing** (format-only): delimiters / role-framing for the model family applied
   over the base parts; `default` family ≈ the agnostic core (no change).
3. **Tier optional modules** (present per `PromptTier`, dropped under budget per the order below):
   - `extended_guidance` (EXTENDED only)
   - `examples` (EXTENDED; never in the always-on base)
   - (CORE = base + essentials, no examples; TINY = reduced protocol, may omit guidance entirely)
4. **Tool inventory** (parity floor — never dropped): either native `AVAILABLE TOOLS` / `MCP TOOLS`
   blocks (as `DynamicTUIPromptGenerator` renders) OR the toolshim `<tools>` block (shim modes).
5. **Prior context** (when provided): the `PRIOR CONTEXT:` block, verbatim (existing behavior).

## Budget enforcement (D5)

- Effective budget = `min(jacket.budget_tokens (if present), target_max=1000)`; hard cap = 1500.
- Measure the full assembly (tiktoken; `len//4` fallback).
- While over the effective budget, drop **optional modules** in ascending `drop_priority`:
  1. `examples`
  2. `extended_guidance`
  3. `toolshim` extra text (only if a viable non-shim path exists — otherwise keep it)
- **Never drop**: the static base, the substituted objective, the tool inventory, prior context.
- If the parity floor alone exceeds the hard cap → return it and surface the over-cap condition
  (the prompt is correct, just flagged), never truncate the base.

## `<tools>` JSON-in-text protocol (shim modes only)

Rendered when `tool_mode ∈ {shim_prompt, shim_interpreter}` (D4). Shape:

```text
<tools>
You cannot call tools natively. To use a tool, emit EXACTLY one JSON object:
{"tool": "<name>", "arguments": { ... }}
Available tools:
- <name>: <description>  (parameters: <param: type, ...>)
- ...
</tools>
```

- `shim_interpreter` adds an interpreter-style note (the model writes a call the harness executes).
- Native mode renders NO `<tools>` block — tools are passed through unchanged.

## `<turn-context>` block (US3, FR-011)

Injected each turn for weak tiers:

```text
<turn-context turn="<N>">
GOAL: <objective re-injected verbatim>
<optional: current step / last result summary>
</turn-context>
```

## Progress doc (US3, Document-&-Clear, FR-012)

Written at the compression threshold (`PROGRESS.md` / `todo.md` under the working dir):

```markdown
# Progress — <goal>
## Plan
- ...
## Decisions
- ...
## Done / State
- ...
## Resume
Continue toward GOAL from the state above.
```

After writing, the working context is cleared and resumed from this doc. If the working dir is not
writable → fall back to `HistorySummarizer` (lossy compaction).

## Invariants (testable)

- The static base substring is present and byte-identical in every assembled prompt (SC-003).
- Two materially-different profiles → different assembled strings (SC-002).
- Measured tokens ≤ hard cap whenever the parity floor fits; ≤ effective budget after trimming when
  optional modules can be dropped (SC-001).
- Native vs shim tool rendering is mutually exclusive and matches `tool_mode` (SC-004).
- Identical inputs → identical output (SC-008).
