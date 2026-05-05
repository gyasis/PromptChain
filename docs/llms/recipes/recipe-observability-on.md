---
id: recipe-observability-on
title: Turn on MLflow observability
when: You want token / latency / error metrics for every LLM call, task, routing decision, and session.
api_version: 0.6.1+
---

# Observability On

PromptChain ships an MLflow-backed observability layer. **It's a no-op when MLflow isn't installed.** That means decorating your code is safe even in environments without MLflow — zero overhead.

## Quickstart — initialise + use built-in tracking

```python
import asyncio
from promptchain import PromptChain
from promptchain.observability import init_mlflow, shutdown_mlflow

async def main():
    init_mlflow()   # picks up .promptchain.yml in cwd or env vars (MLFLOW_TRACKING_URI etc.)

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Say hi"],
        verbose=True,
    )
    # Every LLM call inside .process_prompt_async() is auto-tracked
    result = await chain.process_prompt_async("start")
    print(result)

    shutdown_mlflow()

asyncio.run(main())
```

## Decorating your own code

```python
from promptchain.observability import track_llm_call, track_task, track_routing, track_session

@track_session()
async def my_workflow():

    @track_task(operation_type="ingest")
    def ingest(): ...

    @track_routing(extract_decision=lambda r: r["agent"])
    def route(input): ...

    @track_llm_call(model_param="model")
    async def call(prompt: str, model: str): ...
```

## Env knobs

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | Where runs are written (default: `./mlruns/`) |
| `PROMPTCHAIN_MLFLOW_BACKGROUND=1` | Non-blocking background queue for log writes |

## Per-project YAML

`./.promptchain.yml`:
```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment: "promptchain-dev"
  background: true
```

## Common failures + fix

- **No data appears in MLflow UI** — `init_mlflow()` was never called, or `MLFLOW_TRACKING_URI` is wrong. Check stdout for "MLflow init" log lines.
- **High overhead** — set `PROMPTCHAIN_MLFLOW_BACKGROUND=1` so writes happen on a background thread.
- **Decorator silently does nothing** — MLflow not installed. `pip install mlflow` to activate.
- **`from promptchain import track_llm_call`** → `ImportError`. Use `from promptchain.observability import track_llm_call`.
