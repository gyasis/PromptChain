# 2026-11-26 probe pilot

Intent: first real test of the P7c latency and native tool-calling probes plus the Layer-B
availability gate, on the local Atelier fleet. Pilot size only (R-SPEND4): 3 latency calls
(small/medium/large) and 5 BFCL items (3 simple, 2 irrelevance) per model.

Cohort (exact tags from `~/.config/ollama-cloud/local-models.md`): ornith-1.5-9b:latest,
gemma4:12b, gpt-oss:20b, ornith-1.5-35b:latest, qwen3.8:27b-mtp-q4_K_M,
hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL.

Safety rules:
- No paid or cloud calls; `promptchain.profiler.netguard` blocks every host except 192.168.0.159 and loopback.
- Every call goes through the Atelier governor `:8799` (`/llm/ollama/api/chat`), never raw `:11434`.
- No num_ctx / num_predict. No `/make-room`, no `/force-stop`: a model that does not fit is DROPPED.
- One model at a time; thinking turned off with the endpoint's parameter and the emitted reasoning measured.
- Wall-clock budget 45 minutes.

Output: raw lines in `~/.promptchain/model_measurements.jsonl` (not committed) and `$PROBE_OUT`
(default `/tmp/probe-pilot-results.json`) with before/after governor and meter snapshots.
