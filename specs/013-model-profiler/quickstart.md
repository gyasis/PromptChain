# Quickstart — Model Profiler

## Profile a model (US1, MVP)

```python
from promptchain.profiler import ModelProfiler

profiler = ModelProfiler()                      # store → ~/.promptchain/model_profiles.json
profile = profiler.run_probe("ollama/qwen3-coder:30b")   # runs the cheap CAT probe (offline)

print(profile.capability)         # C ∈ [0,1]
print(profile.recommended_tier)   # e.g. "standard"
print(profile.budget_tokens)      # e.g. 4000
print(profile.skills["reasoning_depth"].theta, profile.skills["reasoning_depth"].se)
```

Each probe trial is written as an F1 transcript under `~/.promptchain/transcripts/<project>/`,
so `sio mine --agent promptchain` ingests them with `model_used` set.

## Inspect the jacket (US2)

```python
j = profile.jacket
print(j.mode)        # "single-shot" | "single-shot+retry" | "heavy-loop" | "escalate"
print(j.spawn_temp)  # 1 − C
print(j.compress_at) # e.g. 0.75
print(j.escalate)    # True when P_route≥α OR Ω<0.25 OR SE>0.4
```

## Refine from telemetry (US3)

```python
# after real sessions accumulate model-attributed telemetry
profiler.refine("ollama/qwen3-coder:30b", session_metrics={"reasoning_depth_score": 0.81})
# EWMA-updates the stored estimate; no-op if session_metrics is empty
```

## Two-sided model×jacket fit (US3)

```python
best = profiler.jacket_fit("ollama/qwen3-coder:30b", jackets=[jacket_a, jacket_b])
# Δθ lift per jacket vs baseline → returns the max-lift jacket
```

## Offline live smoke (no secrets — real LAN model)

```bash
OLLAMA_API_BASE=http://192.168.0.159:11434 \
PYTHONPATH=/home/gyasis/Documents/PromptChain.wt-epic-adaptive-prompting \
python - <<'PY'
from promptchain.profiler import ModelProfiler
p = ModelProfiler().run_probe("ollama/qwen3-coder:30b")
print("capability", round(p.capability, 3), "tier", p.recommended_tier, "budget", p.budget_tokens)
PY
```

## Run the math tests (test-first gate)

```bash
cd /home/gyasis/Documents/PromptChain.wt-epic-adaptive-prompting
python -m pytest tests/test_profiler_irt.py tests/test_profiler_cat.py \
                 tests/test_profiler_composite.py -q
```

The math tests recover a known θ from synthetic items with known (a,b,c) within tolerance, and
assert the 3PL probability / Fisher information / Ω formulas match closed-form expected values
(SC-004 / FR-016/17).
