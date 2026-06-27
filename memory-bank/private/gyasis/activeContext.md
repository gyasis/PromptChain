# Active Context

**Last Updated**: 2026-06-27 16:35:37

## Current Focus
Merge F3 (Dynamic Prompt Layer — per-model prompt assembly reading the F2 jacket) into epic/adaptive-prompting

F3 = the dynamic half of the two-layer prompt: DynamicModelPromptGenerator reads the
F2 profile/jacket and assembles per-model — CORE/EXTENDED/TINY tiering, family adapter,
toolshim (+ additive Jacket.tool_mode), token-budget trim, weak-model Document-&-Clear
longevity. Static base verbatim for CORE/EXTENDED; TINY swaps to a simpler base (Goose
tiny_model_system) — the sanctioned SC-003 exception that lets F3 actually help a weak model.
Test-first, 24/24 tasks. F1+F2+F3 together 165 pass, no regression.
SC-007 offline live smoke (llama3.2:1b): f3 40% vs static_base 30% (+10%).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## Recent Changes
```
 .claude/activity_stream.md | 1 +
 1 file changed, 1 insertion(+)
```

## Modified Files
.claude/activity_stream.md
memory-bank/private/gyasis/activeContext.md

## Next Actions
- Continue implementation
- Run tests
- Create checkpoint
