# MLflow Observability - Quick Fix Guide

## Problem

User reports "0/0 activities" in MLflow tracking despite having both observability systems present.

## Root Cause

MLflow tracking is **disabled by design** because:
1. Environment variable `PROMPTCHAIN_MLFLOW_ENABLED` not set (defaults to `false`)
2. MLflow library not installed
3. Ghost pattern correctly disables decorators when tracking is off

This is **working as designed** - MLflow is opt-in, not opt-out.

## Evidence

```bash
$ python test_observability_conflict.py

DIAGNOSTIC SUMMARY:
   1. MLflow tracking is DISABLED (PROMPTCHAIN_MLFLOW_ENABLED not set to true)
   2. MLflow is NOT installed (pip install mlflow)
   3. Callback system is ACTIVE (captured 6 events)
   4. Decorator system is INACTIVE (ghost pattern returned identity decorator)
```

**Key Finding:** The old callback system (v0.4.1) is fully operational and capturing events. The new MLflow system (spec 005) is correctly disabled via the ghost pattern.

## Quick Fix (5 minutes)

### Step 1: Install MLflow

```bash
pip install mlflow
```

### Step 2: Enable Tracking

```bash
export PROMPTCHAIN_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Or create `.promptchain.yml` in your home directory:

```yaml
mlflow:
  enabled: true
  tracking_uri: http://localhost:5000
  experiment_name: promptchain-cli
```

### Step 3: Start MLflow Server

```bash
# In a separate terminal
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

This will start the MLflow UI at http://localhost:5000

### Step 4: Verify It Works

```bash
# Run the diagnostic
python test_observability_conflict.py

# Expected output:
# MLflow tracking is ENABLED
# MLflow runs: 1+ (runs created)
# Decorator system is ACTIVE (ghost pattern disabled)
```

### Step 5: Test in CLI

```bash
promptchain --session test

# Send a message like "What is 2+2?"
# Then check MLflow UI at http://localhost:5000
# You should see:
# - Experiment: "promptchain-cli"
# - Run with model name, tokens, execution time
```

## What Changed?

**Before (Ghost Pattern Active):**
```python
@track_llm_call(...)  # Returns identity decorator
async def run_model_async():
    pass

# Equivalent to:
async def run_model_async():  # No wrapping
    pass
```

**After (MLflow Enabled):**
```python
@track_llm_call(...)  # Returns actual tracking decorator
async def run_model_async():
    pass

# Equivalent to:
async def async_wrapper(*args, **kwargs):  # Wraps with tracking
    with run_context("llm_call"):
        queue_log_param("model", ...)
        result = await run_model_async_original(*args, **kwargs)
        queue_log_metric("execution_time", ...)
        return result
```

## Current Architecture

**Two Systems Present:**

1. **Callback System (v0.4.1)** - ACTIVE
   - `CallbackManager` initialized at line 234 in `promptchaining.py`
   - 50+ `emit()` calls throughout chain execution
   - Captures 6 event types: CHAIN_START, STEP_START, MODEL_CALL_START, MODEL_CALL_END, STEP_END, CHAIN_END
   - Zero MLflow dependency

2. **MLflow Decorator System (spec 005)** - INACTIVE (until enabled)
   - `@track_llm_call` decorator on `run_model_async()` (line 1834)
   - Ghost pattern disables when `PROMPTCHAIN_MLFLOW_ENABLED != true`
   - Zero overhead when disabled (identity function)
   - Requires MLflow installation + environment variable

**When MLflow is Enabled:** Both systems will track simultaneously. This creates redundancy but no conflicts (they use different mechanisms).

## Expected Behavior After Fix

**CLI Session:**
```bash
$ export PROMPTCHAIN_MLFLOW_ENABLED=true
$ promptchain --session test

Welcome to PromptChain CLI!

> What is 2+2?
[Agent responds: "4"]

# Behind the scenes:
# 1. Callback system emits 6 events (as before)
# 2. MLflow decorator logs to background queue
# 3. Background queue flushes to MLflow server
# 4. MLflow UI updates with new run
```

**MLflow UI (http://localhost:5000):**
```
Experiment: promptchain-cli
└── Run (2026-01-10 08:21:45)
    ├── Metrics:
    │   ├── execution_time_seconds: 1.234
    │   ├── prompt_tokens: 123
    │   ├── completion_tokens: 45
    │   └── total_tokens: 168
    ├── Parameters:
    │   ├── model: openai/gpt-4o-mini
    │   ├── temperature: 0.7
    │   └── max_tokens: 2048
    └── Tags:
        └── status: success
```

## Configuration Options

### Environment Variables

```bash
# Required to enable
export PROMPTCHAIN_MLFLOW_ENABLED=true

# Optional (defaults shown)
export MLFLOW_TRACKING_URI=http://localhost:5000
export PROMPTCHAIN_MLFLOW_EXPERIMENT=promptchain-cli
export PROMPTCHAIN_MLFLOW_BACKGROUND=true  # Use background logging queue
```

### YAML Configuration

Create `.promptchain.yml` in project directory OR `~/.promptchain.yml`:

```yaml
mlflow:
  enabled: true
  tracking_uri: http://localhost:5000
  experiment_name: my-project
  background_logging: true
```

**Precedence:** Environment variables override YAML configuration.

## Verifying Installation

### 1. Check MLflow Installation

```bash
python -c "import mlflow; print(mlflow.__version__)"
# Expected: 2.x.x
# If error: pip install mlflow
```

### 2. Check Environment

```bash
env | grep -i mlflow
env | grep -i promptchain

# Expected:
# PROMPTCHAIN_MLFLOW_ENABLED=true
# MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Check Ghost Pattern Status

```python
from promptchain.observability.ghost import is_tracking_enabled, is_mlflow_available

print(f"Tracking enabled: {is_tracking_enabled()}")
print(f"MLflow available: {is_mlflow_available()}")

# Expected when enabled:
# Tracking enabled: True
# MLflow available: True
```

### 4. Check Decorator Wrapping

```python
from promptchain.utils.promptchaining import PromptChain

chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Test"])
func = chain.run_model_async

print(f"Has closure: {func.__closure__ is not None}")

# Expected when enabled:
# Has closure: True  (decorator is wrapping)
```

## Troubleshooting

### "0/0 activities" persists after enabling

**Check:**
1. MLflow server is running (`mlflow ui` in separate terminal)
2. Environment variable is set in same shell as CLI
3. MLflow library is installed (`pip list | grep mlflow`)
4. Ghost pattern reports enabled (`is_tracking_enabled()` returns `True`)

**Fix:**
```bash
# Restart CLI with explicit environment
PROMPTCHAIN_MLFLOW_ENABLED=true promptchain --session test
```

### "MLflow not available" warning

**Cause:** MLflow library not installed

**Fix:**
```bash
pip install mlflow
```

### "Connection refused" error

**Cause:** MLflow server not running

**Fix:**
```bash
# Start server in separate terminal
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### Callback system still shows events but MLflow doesn't

**Cause:** Ghost pattern still active (decorator not wrapping)

**Check:**
```python
from promptchain.observability.ghost import _ENABLED
print(f"Ghost pattern state: {_ENABLED}")

# If False: Environment variable not set when module imported
# Solution: Restart Python process with env var set
```

## Performance Impact

**With MLflow Disabled (default):**
- Zero overhead (ghost pattern returns identity function)
- No MLflow dependency
- Callback system only (minimal memory)

**With MLflow Enabled:**
- Background queue processing (non-blocking)
- HTTP calls to MLflow server (async)
- Estimated overhead: <5% on LLM call duration
- Memory: ~1-2MB for queue + context tracking

## Long-Term Recommendations

See `OBSERVABILITY_CONFLICT_REPORT.md` for detailed analysis of both systems and recommended unification strategies.

**Summary:**
1. **Short-term:** Use callback adapter to bridge both systems
2. **Long-term:** Make callbacks primary, MLflow optional backend
3. **Cleanup:** Remove duplicate tracking in v0.5.0+

## References

- Spec: `/specs/005-mlflow-observability/spec.md`
- Architecture: `OBSERVABILITY_ARCHITECTURE.md`
- Conflict Analysis: `OBSERVABILITY_CONFLICT_REPORT.md`
- Diagnostic Script: `test_observability_conflict.py`
- System Diagram: `observability_systems_diagram.txt`

## Quick Verification Script

```bash
#!/bin/bash
# verify_mlflow.sh

echo "=== MLflow Installation Check ==="
python -c "import mlflow; print('✓ MLflow installed:', mlflow.__version__)" || echo "✗ MLflow NOT installed"

echo ""
echo "=== Environment Check ==="
echo "PROMPTCHAIN_MLFLOW_ENABLED: ${PROMPTCHAIN_MLFLOW_ENABLED:-NOT SET}"
echo "MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI:-NOT SET}"

echo ""
echo "=== Ghost Pattern Check ==="
python -c "
from promptchain.observability.ghost import is_tracking_enabled, is_mlflow_available
print(f'Tracking enabled: {is_tracking_enabled()}')
print(f'MLflow available: {is_mlflow_available()}')
"

echo ""
echo "=== MLflow Server Check ==="
curl -s http://localhost:5000/health 2>&1 | grep -q "OK" && echo "✓ MLflow server running" || echo "✗ MLflow server NOT running"

echo ""
echo "=== Recommendation ==="
if [ "$PROMPTCHAIN_MLFLOW_ENABLED" != "true" ]; then
    echo "Set: export PROMPTCHAIN_MLFLOW_ENABLED=true"
fi

python -c "import mlflow" 2>/dev/null || echo "Install: pip install mlflow"

curl -s http://localhost:5000/health 2>&1 | grep -q "OK" || echo "Start server: mlflow ui --backend-store-uri sqlite:///mlruns.db"
```

Save as `verify_mlflow.sh`, make executable (`chmod +x verify_mlflow.sh`), and run:

```bash
./verify_mlflow.sh
```
