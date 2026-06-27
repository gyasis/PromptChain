# Refinement — "never invent a tool" reconciled with PromptChain's tool/sub-chain creation

## The catch (user, 2026-06-27)
The draft `<tools>` said *"Use ONLY the tools listed; never invent a tool."* That's a blunt anti-hallucination
rule borrowed from **fixed-toolset** harnesses (Cursor/Claude/Devin) — and it conflicts with PromptChain's
superpower: **PromptChain can build tools.** A PromptChain IS a tool; the TUI can spawn sub-PromptChains
(static or dynamic) — *including authoring their tools* — and use them as tools (a "pool"). (cf. Ornith building
full harnesses; opencode's `agent-generate` factory; Goose toolshim.)

## The distinction the rule conflated (the key insight)
- **Hallucinating a CALL** — emitting `search_web(...)` when no such tool is registered, then **fabricating its
  output** → garbage. **MUST be forbidden** — it's the exact ungrounding bug we hit in the default agent.
- **CREATING a capability** — invoking a **real meta-tool** (`spawn_promptchain` / `build_chain` /
  `delegate_task`) that produces a new, *executable, registered* tool/sub-chain → **PromptChain's flagship
  feature.** This is NOT "inventing a tool"; it's *building* one with a real tool, then calling it.

The leaked prompts forbid invention because their toolset is **fixed**. Ours isn't. So we **re-scope, not
delete** the rule.

## The resolution — ground AND empower, tiered
- **Grounding (all tiers, always):** never call a tool that isn't available; never fabricate a tool's output;
  if a capability can't be obtained or built, say so. (Stops hallucination.)
- **Empowerment (planner / strong / big-brother tier):** *"If you lack a capability, BUILD it — spawn a
  sub-PromptChain with its own prompt + tools via the builder, then use it as a tool."* Weak EXECUTOR models do
  NOT author tools (they'd do it badly) — they receive pre-built tools from the big-brother. → fits the
  heterogeneous-model design (planners author sub-chains/tools; executors use them).

## Guardrails
- The spawn/build is itself a **real registered meta-tool** → calling it is grounded (no contradiction).
- The created tool must be **real** (executable + registered), never a fiction; outputs are never faked.
- **Bounded recursion** — a max spawn depth (cf. Goose/Claude "subagents cannot spawn subagents", or depth ≤ N)
  to prevent runaway trees + cost blowups.
- Tool-creation is **role-gated** (Constraint B); the Model Profiler sets whether a model is trusted to author.

## Reuse (mostly already there)
`DynamicChainBuilder` (manages multiple chains), `delegate_task`, `AgentChain`, `OrchestratorSupervisor`,
blackboard, `agentic_chat`. "Spawn a sub-PromptChain as a tool" = wrap a (static or dynamic) sub-chain as a
callable tool in the pool. The Phase-5 subagency becomes **create-on-demand** sub-chains/tools, not just
routing to fixed agents — the agent **extending its own harness**, the inversion thesis applied recursively.

## Tool creation runs as a Karpathy / autoresearch loop (resolves the weak-model concern)
The earlier worry "a weak model would author tools badly" is **resolved by wrapping tool creation in a
Karpathy loop** — iterate-with-validation until the tool PASSES a test:
`write tool code → run its test in a sandbox → pass? register + exit; fail? feed the error back,
keep-if-closer / git-revert, retry` — bounded to ~100–200 iterations / a time+token budget.
**The loop's pass-gate IS the guardrail:** only a *working, tested* tool exits, so even a weak model produces a
real tool (brute-force iteration > per-shot quality — the Ralph/Karpathy thesis). This also *satisfies*
grounding: the tool is never "invented and trusted," it's **built and proven**.

**This is the user's existing `autoresearch` system — reuse, don't rebuild.** autoresearch already runs a
doer/critic loop in a **real Docker sandbox** against a **falsifiable success bar** (PASS = it actually ran,
not "looks right") and returns RUNNER-VERIFIED code. Point it at the brief *"build tool X that passes test T"*
→ out comes a proven tool → register it in the pool → the parent chain calls it. autoresearch (`ar`) **IS** the
Karpathy tool-forge.

**Tiering update:** tool authoring is gated by the **loop's test-pass, NOT by model role.** Any model can forge
a tool via the loop; strong models converge in fewer iterations, weak models brute-force more (higher cost,
same guarantee). Big-brother is *optional*, not required — the validation loop replaces "trust the planner" as
the guarantee. (This loosens the earlier role-gating: the gate is the test, not the model.)

**Guardrails:** bounded iterations + time + token budget (Karpathy's stopping criteria); a defined success test
(the model proposes a test → the loop validates, or a spec supplies it); sandboxed execution (DockerExecutor);
recursion bound on tools-building-tools.

## Net
Re-word the foundation `<tools>` rule: **ground (no fake calls/outputs) for everyone; add "build it via the
chain-builder" for the planner/strong tier.** The vision — the TUI spawns a *pool* of sub-PromptChains as
tools, each with its own authored tools — is sound, on-brand, and largely buildable on existing primitives.
A flagship Phase-5 feature.
