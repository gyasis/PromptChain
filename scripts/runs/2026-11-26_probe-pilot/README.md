# 2026-11-26 probe pilot

Intent: first real test of the P7c latency and native tool-calling probes plus the Layer-B
availability gate, on the local Atelier fleet. Pilot size only (R-SPEND4): 5 latency calls
(small x2, medium, large, and a long-answer decode call) and 5 BFCL items (3 simple, 2 irrelevance) per model.

Cohort (exact tags from `~/.config/ollama-cloud/local-models.md`): ornith-1.5-9b:latest,
gemma4:12b, gpt-oss:20b, ornith-1.5-35b:latest, qwen3.8:27b-mtp-q4_K_M,
hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL.

Run settings (environment variables):
- `PROBE_COHORT`: comma-separated tags. `llamacpp:<alias>` runs that alias on the governor's
  llamacpp lane; the memory gate and the catalog check do not apply there.
- `PROBE_GATE=advisory`: record the local memory gate but let the governor alone decide admission.
- `PROBE_N_SIMPLE`, `PROBE_N_IRRELEVANCE`: BFCL items per model (default 3 and 2).
- `PROBE_BUDGET_MIN`: wall-clock budget in minutes (default 45).
- `PROBE_OUT`: results JSON path.

From a worktree, put the worktree first on the path (`PYTHONPATH=$PWD`), because the editable
install points at the main checkout.

Safety rules:
- No paid or cloud calls. `netguard.py` (in this folder, stdlib-only, imported before any promptchain import) allows ONLY 192.168.0.159:8799. Loopback is blocked too (127.0.0.1:37400 is the LiteLLM gateway to paid providers), as is raw :11434. It sets `LITELLM_LOCAL_MODEL_COST_MAP=True`, patches `connect`, `connect_ex` and `getaddrinfo`, and strips secret-looking env vars. Blocked attempts are listed in the results as `blocked_connections`.
- Every call goes through the Atelier governor `:8799` (`/llm/ollama/api/chat`), never raw `:11434`.
- No num_ctx / num_predict. No `/make-room`, no `/force-stop`: a model that does not fit is DROPPED.
- One model at a time; thinking turned off with the endpoint's parameter and the emitted reasoning measured.

Output: raw `model_measurement/2` lines in `~/.promptchain/model_measurements.jsonl` (not committed) and `$PROBE_OUT`
(default `/tmp/probe-pilot-results.json`) with before/after governor and meter snapshots.
