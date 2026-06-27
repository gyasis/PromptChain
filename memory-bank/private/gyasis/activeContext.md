# Active Context

**Last Updated**: 2026-06-27 13:11:26

## Current Focus
F2 (Model Profiler) COMPLETE + merged into epic/adaptive-prompting (merge 3b125df). Next: F3.

F2 delivers `promptchain/profiler/`: a cheap adaptive IRT-3PL + CAT probe → a
persisted per-model CapabilityProfile + jacket that F3 will read. Each probe trial
is emitted as a FROZEN-F1-schema transcript (sio-mineable, model_used set).
- US1 (MVP): probe harness (per-dimension CAT), irt/cat/scoring math, ProfileStore →
  profile {per-skill θ̂, capability C, recommended tier, budget}.
- US2: composite Ω = 0.7·C·K − 0.3·F → jacket {tier, budget, mode, spawn_temp,
  compress_at, max_turns, role, escalate}.
- US3: EWMA refine from telemetry, two-sided model×jacket Δθ fit, DSPy
  ModelPromptGenerator (degrades gracefully).
Persisted store contract: specs/013-model-profiler/contracts/profile-schema.md
(~/.promptchain/model_profiles.json — F3 reads this). 66 profiler tests green; F1+F2
together 88 green (no frozen-F1 regression); offline live smoke vs LAN ollama
qwen3-coder:30b passed. Built test-first via the spec-kit→dev-kid loop (Opus
subagents per US wave; manual orchestration — sentinel off). epic→main deferred
until F3 green.

F1 (prior): opt-in TranscriptEmitter, locked schema at
specs/012-sio-output-integration/contracts/transcript-schema.md (model_call.model
= F2 model_used linchpin). 22 tests green.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## Recent Changes
```
 .devkid/config.json                         | 11 ++++++++---
 .specify/integrations/claude.manifest.json  |  2 +-
 .specify/integrations/speckit.manifest.json | 10 ++--------
 tasks.md                                    |  2 +-
 4 files changed, 12 insertions(+), 13 deletions(-)
```

## Modified Files
.devkid/config.json
.specify/integrations/claude.manifest.json
.specify/integrations/speckit.manifest.json
memory-bank/private/gyasis/activeContext.md
tasks.md

## Next Actions
- Continue implementation
- Run tests
- Create checkpoint
