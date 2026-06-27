# Contract: Dynamic Prompt Layer public API surface

The public surface exported from `promptchain.prompts` for F3. Pure-mapping helpers are synchronous
and deterministic; the generator is a drop-in `BasePromptBuilder`.

## Generator (the integration seam)

```python
# promptchain.prompts.model_dynamic
class DynamicModelPromptGenerator:
    """BasePromptBuilder drop-in that assembles a per-model prompt from the F2 profile/jacket,
    layered additively over the intact static base (DynamicTUIPromptGenerator / TUI_FOUNDATION_PROMPT)."""

    def __init__(self, *, model: str | None = None, store=None, base_generator=None,
                 store_path=None) -> None:
        """model: the model id bound to this builder (the protocol path uses it). store: an F2
        ModelProfiler / profile store (defaults to reading ~/.promptchain/model_profiles.json).
        base_generator: the static-base builder to compose over (defaults to
        DynamicTUIPromptGenerator())."""

    # --- BasePromptBuilder protocol (drop-in at agentic_step_processor.py:1006) ---
    def generate(self, objective: str, tools: list[dict], context: str | None = None,
                 *, model: str | None = None) -> str:
        """Assemble the system prompt for `model` (explicit kwarg → construction-time → None=default
        CORE). Reads the profile/jacket, selects tier (CORE/EXTENDED/TINY), applies the family
        adapter, renders the toolshim per tool_mode, includes per-tier optional modules, then
        measures + trims to budget (dropping optional modules, never the static base). The static
        base appears VERBATIM. Deterministic for fixed (objective, tools, model, profile)."""

    def get_token_estimate(self, objective: str, tools: list[dict]) -> int:
        """Non-negative token estimate of generate(objective, tools) (tiktoken or len//4)."""
```

**Guarantees**
- **Drop-in**: conforms to `promptchain.prompts.base.BasePromptBuilder` (same `generate`/
  `get_token_estimate`); usable wherever `DynamicTUIPromptGenerator` is.
- **Static base intact (SC-003)**: `TUI_FOUNDATION_PROMPT` content appears verbatim in every output
  and is never dropped under budget pressure.
- **Per-model difference (SC-002)**: two models with materially different profiles → measurably
  different prompts (tier and/or family framing).
- **Budget (SC-001)**: output within the configured budget (target 300–1,000, hard cap ~1,500),
  enforced by dropping optional modules.
- **Fallbacks (SC-005)**: null jacket → `recommended_tier` + `budget_tokens`; no profile → default
  CORE; never raises (except an empty objective, which is rejected per the base contract).
- **Determinism (SC-008)**: identical inputs → identical output.

## Pure mappings (synchronous, deterministic)

```python
# promptchain.prompts.tiers
class PromptTier(str, Enum): CORE; EXTENDED; TINY
def select_tier(profile) -> PromptTier            # capability/recommended_tier → tier (D2)
def modules_for_tier(tier) -> list[OptionalModule] # per-tier optional module set

# promptchain.prompts.family
def family_of(model_id: str) -> str               # → anthropic|openai|google|qwen|llama|default
def adapt_format(parts: list[str], family: str) -> list[str]  # FORMAT-only (D3)

# promptchain.prompts.toolshim
def resolve_tool_mode(jacket) -> str              # jacket.tool_mode or "native" (D4)
def render_tools_block(tools: list[dict]) -> str  # <tools> JSON-in-text (shim modes)
def serialize_history_plaintext(history: list[dict]) -> str

# promptchain.prompts.budget
def measure(text: str) -> int                     # tiktoken or len//4
def fit_to_budget(base: str, optional: list[OptionalModule], *, target_max: int,
                  hard_cap: int) -> tuple[str, list[str]]  # returns (assembled, dropped_keys); base never dropped
```

## Longevity (US3)

```python
# promptchain.prompts.longevity
def build_turn_context(goal: str, *, turn: int, extra: str | None = None) -> str
    """The <turn-context> block re-injecting the goal for this turn (FR-011)."""

class DocumentAndClear:
    def __init__(self, *, compress_at: float = 0.60, min_turns: int = 10, jacket=None): ...
    def should_compress(self, context_usage_fraction: float) -> bool   # usage ≥ compress_at
    def is_stalled(self, progress_signals: list) -> bool               # no measurable progress
    def should_escalate(self, *, stalled: bool) -> bool                # stalled AND jacket.escalate
    def document_and_clear(self, working_dir: str, state: dict) -> list[dict]:
        """Write the progress doc (PROGRESS.md/todo.md) under working_dir, clear working context,
        return the doc-seeded resumed history. Falls back to lossy HistorySummarizer when the doc
        path is unavailable (FR-012/014)."""
```

**Guarantees**: pure decision functions are deterministic + offline-testable (fake usage signal);
`document_and_clear` is the only I/O (temp dir in tests); escalation respects `jacket.escalate`
(no escalate when the jacket forbids it) and fires only on stall, not on every compression.
