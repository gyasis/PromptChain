# Model Profiler — the math (IRT / BT / RouteLLM) + SIO integration

The rigorous backbone for Component E. Replaces the hand-waved capability score with established
psychometrics, and wires the whole thing into SIO (which is already PromptChain's telemetry + scoring +
DSPy-optimization engine).

## 1. The formula — IRT + CAT is the answer for "capability from ~10 probes"

The **"10-cast isolated trial" IS Computerized Adaptive Testing (CAT)** over an Item-Response-Theory bank.

- **3PL item model:** `P_i(θ) = c_i + (1−c_i)·σ(a_i(θ−b_i))`, `σ(x)=1/(1+e^{−x})`.
  θ = model ability (what we estimate); b_i = item difficulty; a_i = discrimination; c_i = guessing.
- **CAT loop:** pick the next probe maximizing **Fisher information** `I_i(θ̂)=a_i²(P_i−c_i)²(1−P_i)/[P_i(1−c_i)²]`;
  estimate `θ̂` via **EAP** (or **WLE** for all-right/all-wrong edge cases); **stop when**
  `SE(θ̂)=1/√(Σ I_i(θ̂)) ≤ τ`. τ=0.3 → ~10–30 high-discrimination items (90%+ fewer than a full benchmark).
- **Prereq:** pre-calibrate the probe bank's (a_i,b_i,c_i) across many models ONCE (filter items: variance≥1%,
  acc≤95%, point-biserial≥0.1).
- Capability: **`C = σ(θ̂) ∈ [0,1]`**.
- arXiv: tinyBenchmarks 2402.14992 · ATLAS (adaptive testing) 2511.04689 · IRT scaling 2606.07616 · IRT-for-LLM 2505.15055.

**Cross-checks / adjacent math:**
- **Bradley-Terry / Elo** (pairwise probe wins): `P(i≻j)=π_i/(π_i+π_j)`, `Elo≈173.7·x_m+const` (Chatbot Arena 2403.04132).
- **RouteLLM** (escalation threshold): matrix-factorization router predicts `P(win_strong|q)`; route to big-brother
  when `≥ α`; α calibrated to a cost target via the APGR cost-quality frontier (2406.18665).
- **Calibration** (stop/escalation gating): ECE `Σ(|B_k|/N)|acc−conf|` · self-consistency · **semantic entropy**
  `−Σ p(C_j|x)log p(C_j|x)` (2303.08896). High entropy → escalate.

## 2. Composite Profiler Score Ω → the "jacket"

```
C = σ(θ̂)                                   # capability (IRT)
K = 1 − ECE        (or 0.4·(1−ECE)+0.35·SC+0.25·e^{−SemEntropy})   # calibration confidence
F = (latency/latency_max)·(cost/cost_ref)   # cost+latency penalty
Ω = 0.7·C·K − 0.3·F
```

| Ω / θ̂ / SE | Tier | Budget | Mode (single-shot vs loop) | Sub-agent spawn | Compress @ |
|---|---|---|---|---|---|
| Ω>0.55, θ̂>1.5, SE<0.2 | lean | 1–2k | **single-shot** | low (1−C) | 85% |
| 0.40–0.55 | standard | 4k | single-shot + retry | med | 75% |
| 0.25–0.40 | rich + few-shot | 8k | **heavy loop (3×)** | high | 60% |
| <0.25, θ̂<−1 | max-rich | 16k | **needs bigger model → escalate** | route | 50% |

Escalate if `P_route ≥ α  OR  Ω < 0.25  OR  SE > 0.4`. Sub-agent "temperature" = `1 − C`. This answers the
user's exact questions: single-shot vs heavy-loop vs build-out-with-a-bigger-model-first, the spawn
propensity, and the compression probability — all from one composite.

## 3. SIO integration — the data + scoring + jacket-compile engine

SIO already IS the telemetry/scoring/DSPy backbone. **`session_metrics.model_used` already exists** —
a per-model ledger (latency, tokens, cost, error_count, correction_count, positive_signal_count) is being
collected right now.

| Profiler need | Reuse (exists) | Build (gap) |
|---|---|---|
| per-model session metrics | `SELECT * FROM session_metrics WHERE model_used='X'` (live in `~/.sio/sio.db`) | tag the 178/274 NULL `model_used` rows; `sio profile --model X` scorecard |
| per-model error/flow profile | `error_records`/`flow_events` JOIN `session_metrics` on `session_id` | a `--model` filter on `velocity`/`flows`/`suggest` (adds the JOIN) |
| capability score | aggregate signals + `patterns.rank_score`/`grade` | feed IRT `θ̂` from the probe harness |
| A/B a jacket version | `sio experiment start/close` cohort + `config_hash` snapshot (no drift) | a `model` column (or naming convention `project=model:X`) |
| **compile the jacket** | `sio optimize --optimizer gepa` with `SIO_TASK_LM=ollama/<target>` → optimized module in `~/.sio/optimized/` | a **`model_prompt_generator`** DSPy module (per-model error profile → jacket), distinct from `suggestion_generator` |
| **the 10-cast probe runner** | — (SIO observes ORGANIC sessions only) | **build the probe harness** — runs the CAT/IRT probes as sessions so `sio mine` ingests them + tags `model_used` |

DeepLake corpus corroborated: RouteLLM/Bradley-Terry routing, black-box per-instance behavior prediction
(arXiv 2501.01558 — probe-prompts → capability without weights), OPIK evolutionary prompt optimization.

## 4. The "jacket"

The **jacket** = the per-model compiled config `{tier, budget, family-adapter, modules, compress@, max_turns,
spawn_temp, role}` derived from Ω/θ̂ — optionally with a **GEPA-optimized per-model system prompt** compiled
by SIO (`SIO_TASK_LM`=the target model). `sio experiment` A/Bs jacket versions; `config_hash` versions them.
A model gets fitted with its own tailored jacket — the literal embodiment of "the harness fits the model."

## 5. TWO-SIDED — the same trial also measures the JACKET's fit (not just the model)

The 10-cast trial as described fixes the PROMPT and varies the MODEL → it scores **model ability θ**.
But IRT is **symmetric** (subjects ↔ items): **fix the MODEL and vary the PROMPT/jacket → it scores the
PROMPT's fit.** Run the full **model × jacket grid** → the **interaction** tells you *which jacket fits
which model* — the measured form of "the harness fits the model."

Two-facet model:
```
logit P(success_i) = a_i · ( θ_model + γ_jacket + δ_{model×jacket} − b_item )
```
- `θ_model` — model ability (PROFILE mode: jacket fixed, model is the unknown).
- `γ_jacket` — jacket main effect (overall prompt quality across models).
- **`δ_{model×jacket}` — the interaction = the FIT** (this jacket on THIS model). The load-bearing term:
  a jacket that lifts a weak model may do nothing for a strong one.

**Prompt-fit score** of jacket J on model M = the effective-ability **lift** vs a baseline jacket on the
same bank: `Δθ_{M,J} = (θ_M + γ_J + δ_{M,J}) − (θ_M + γ_base)`. Positive Δθ → J fits M; ship the jacket with
**max Δθ per model**. A jacket is never hand-tuned — it's the *measured best fit*.

**What reruns the trial "over everything" = SIO experiment + GEPA:**
- `sio experiment` A/Bs two jackets on the same model + task bank → the error/flow/score delta = Δθ (fit).
- `sio optimize --optimizer gepa` (`SIO_TASK_LM`=the model) literally **searches jacket space to maximize
  the metric** on the bank, model fixed → automated prompt-fit optimization. GEPA finds the best-fit jacket;
  the experiment cohort A/B validates it; `config_hash` versions it.

**One bank, two modes, same scoring (θ̂/Ω):**
- *Profile a model* — fix tasks + jacket → MODEL is the unknown → `θ_model`.
- *Evaluate a jacket* — fix tasks + model → JACKET is the unknown → `Δθ` (fit).
The probe harness runs both; the (model × jacket) grid IS the experiment sweep.

## Net / build list (Phase 5+)
- **2 new builds:** the **probe harness** (10-cast CAT/IRT trial → θ̂) + a **`model_prompt_generator`** DSPy
  module (compiles the jacket). Plus quality-of-life: `sio profile --model`, `--model` filters, a `model`
  column on experiments.
- **Reuse everything else** from SIO: telemetry (`session_metrics`), scoring, `sio experiment` A/B, `sio
  optimize` GEPA, config-hash versioning, MLflow.
- Pipeline: **probe (CAT) → θ̂ → Ω → jacket → run → SIO telemetry → refine (EWMA / GEPA re-compile).**
