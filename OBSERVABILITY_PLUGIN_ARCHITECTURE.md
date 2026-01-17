# PromptChain Observability Plugin Architecture

## Executive Summary

**Primary System**: Internal CallbackManager (REQUIRED - no external dependencies)
**Plugin System**: MLflow as optional development tool (OPTIONAL - install separately)

This document defines the correct architecture based on user requirements:
1. Internal observability MUST work without any external packages
2. MLflow is a development/debugging plugin, NOT a core dependency
3. Both systems stay in sync when MLflow is enabled

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PromptChain Core                          │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         CallbackManager (PRIMARY SYSTEM)               │ │
│  │  - Always active, no external dependencies             │ │
│  │  - Emits events for all LLM calls, steps, chains       │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                       │
│         ┌─────────────┼─────────────┬────────────────────┐  │
│         │             │             │                    │  │
│         ▼             ▼             ▼                    ▼  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐│
│  │ Observe  │  │ Activity │  │ History  │  │ MLflowObserver││
│  │  Panel   │  │  Logger  │  │ Manager  │  │  (OPTIONAL)  ││
│  │ (TUI)    │  │ (Logs)   │  │ (Memory) │  │  [PLUGIN]    ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘│
│                                                     │         │
└─────────────────────────────────────────────────────┼────────┘
                                                      │
                                          ┌───────────▼──────────┐
                                          │   MLflow Server       │
                                          │   (External - Dev)    │
                                          │   pip install mlflow  │
                                          └──────────────────────┘
```

---

## System Comparison

| Feature | Internal Callbacks | MLflow Plugin |
|---------|-------------------|---------------|
| **Required** | ✓ YES | ✗ NO (optional) |
| **Dependencies** | None (built-in) | `pip install mlflow` |
| **Use Case** | Users, production, TUI | Development, debugging |
| **Data Storage** | In-memory + SQLite | External server |
| **Activation** | Always on | Env var: `PROMPTCHAIN_MLFLOW_ENABLED=true` |
| **TUI Integration** | Direct connection | Via observer pattern |
| **Overhead** | Minimal (~0.1ms) | Higher (50-200ms HTTP) |

---

## Current Problems (Why 0/0 Shows)

### Problem 1: ObservePanel Not Connected to CallbackManager

**Current State**:
```python
# promptchain/cli/tui/observe_panel.py
class ObservePanel(Container):
    def __init__(self):
        self._entries: List[dict] = []  # Empty local storage
        # NO connection to CallbackManager!
```

**Impact**: ObservePanel only receives data from AgenticStepProcessor reasoning updates, NOT from LLM calls.

**Fix Needed**: Register callback in `app.py:on_mount()` to bridge CallbackManager → ObservePanel.

---

### Problem 2: MLflow Implemented as Primary System (BACKWARDS)

**Current Implementation** (WRONG):
```python
# Decorators wrap methods (invasive)
@track_llm_call  # Primary tracking mechanism
async def run_model_async(...):
    # MLflow logging happens in decorator
```

**Problems**:
- MLflow becomes REQUIRED dependency
- Users need external MLflow server for basic TUI
- Decorators are invasive (hard to remove)
- Callback system becomes redundant

**Correct Implementation** (what user wants):
```python
# CallbackManager emits events (primary)
async def run_model_async(...):
    self.callback_manager.emit(event)  # Primary tracking
    # MLflowObserver listens IF enabled (optional plugin)
```

**Benefits**:
- No external dependencies for basic functionality
- MLflow is truly optional
- Easy to enable/disable MLflow without code changes
- Callback system remains the source of truth

---

## Quick Fix for 0/0 Issue

### Step 1: Connect ObservePanel to CallbackManager

Add to `promptchain/cli/tui/app.py` in `on_mount()` method (after line 623):

```python
async def on_mount(self):
    # ... existing session load code ...

    # T118++: Connect ObservePanel to CallbackManager
    if self.verbose_mode and self.observe_panel:
        await self._setup_callback_bridge()

async def _setup_callback_bridge(self):
    """Connect CallbackManager events to ObservePanel (T118++)"""
    from promptchain.utils.execution_events import ExecutionEventType

    # Get current agent's chain
    active_agent = self.session.agents.get(self.session.active_agent)
    if not active_agent or not hasattr(active_agent, 'chain'):
        return

    chain = active_agent.chain  # PromptChain instance

    # Define callback for LLM events
    def llm_callback(event):
        """Bridge callback events to ObservePanel"""
        if event.event_type == ExecutionEventType.MODEL_CALL_START:
            # LLM call started
            model = event.data.get("model_name", "unknown")
            prompt_preview = str(event.data.get("messages", ""))[:100]
            self.observe_panel.log_entry(
                "llm-request",
                f"[{model}] {prompt_preview}..."
            )

        elif event.event_type == ExecutionEventType.MODEL_CALL_END:
            # LLM call completed
            model = event.data.get("model_name", "unknown")
            tokens = event.data.get("usage", {})
            response_preview = str(event.data.get("response", ""))[:100]

            # Log to observe panel with token info
            token_info = f"({tokens.get('prompt_tokens', 0)}p + {tokens.get('completion_tokens', 0)}c = {tokens.get('total_tokens', 0)}t)"
            self.observe_panel.log_entry(
                "llm-response",
                f"[{model}] {response_preview}... {token_info}"
            )

        elif event.event_type == ExecutionEventType.TOOL_CALL_START:
            # Tool call started
            tool_name = event.data.get("tool_name", "unknown")
            self.observe_panel.log_entry("tool-call", f"Calling: {tool_name}")

        elif event.event_type == ExecutionEventType.TOOL_CALL_END:
            # Tool call completed
            tool_name = event.data.get("tool_name", "unknown")
            result_preview = str(event.data.get("result", ""))[:100]
            self.observe_panel.log_entry("tool-result", f"{tool_name}: {result_preview}...")

    # Register callback with chain
    chain.register_callback(llm_callback)
    logger.info("Callback bridge established: CallbackManager → ObservePanel")
```

**Result**: ObservePanel will show LLM calls in real-time! No MLflow required.

---

## Proper MLflow Integration (Plugin Pattern)

### Step 2: Create MLflowObserver (Optional Plugin)

New file: `promptchain/observability/mlflow_observer.py`

```python
"""MLflow Observer - Optional plugin for CallbackManager"""
import logging
from typing import Optional
from ..utils.execution_events import ExecutionEvent, ExecutionEventType

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.debug("MLflow not installed - observer disabled")


class MLflowObserver:
    """
    Optional observer that logs CallbackManager events to MLflow.

    This is a PLUGIN - it requires:
    1. MLflow installed: pip install mlflow
    2. Environment variable: PROMPTCHAIN_MLFLOW_ENABLED=true
    3. MLflow server running: mlflow ui --port 5000

    Usage:
        from promptchain.observability import MLflowObserver

        observer = MLflowObserver()
        if observer.is_available():
            chain.register_callback(observer.handle_event)
    """

    def __init__(
        self,
        experiment_name: str = "promptchain-cli",
        tracking_uri: str = "http://localhost:5000"
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = False
        self.current_run = None

        # Check if enabled
        import os
        enabled_str = os.getenv("PROMPTCHAIN_MLFLOW_ENABLED", "false").lower()
        self.enabled = enabled_str in ("true", "1", "yes", "on")

        if self.enabled and _MLFLOW_AVAILABLE:
            self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Initialize MLflow connection"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow observer initialized: {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.enabled = False

    def is_available(self) -> bool:
        """Check if MLflow observer is available"""
        return self.enabled and _MLFLOW_AVAILABLE

    def handle_event(self, event: ExecutionEvent):
        """Handle callback event and log to MLflow"""
        if not self.is_available():
            return

        try:
            if event.event_type == ExecutionEventType.CHAIN_START:
                # Start new MLflow run
                self.current_run = mlflow.start_run()
                mlflow.set_tag("chain_id", event.data.get("chain_id", "unknown"))

            elif event.event_type == ExecutionEventType.MODEL_CALL_END:
                # Log LLM metrics
                usage = event.data.get("usage", {})
                mlflow.log_metric("prompt_tokens", usage.get("prompt_tokens", 0))
                mlflow.log_metric("completion_tokens", usage.get("completion_tokens", 0))
                mlflow.log_metric("total_tokens", usage.get("total_tokens", 0))
                mlflow.log_param("model", event.data.get("model_name", "unknown"))

            elif event.event_type == ExecutionEventType.CHAIN_END:
                # End MLflow run
                if self.current_run:
                    mlflow.end_run()
                    self.current_run = None

        except Exception as e:
            logger.error(f"MLflow logging error: {e}")

    def shutdown(self):
        """Cleanup MLflow resources"""
        if self.current_run:
            mlflow.end_run()
```

### Step 3: Optional MLflow Setup in CLI

Update `promptchain/cli/main.py` to make MLflow optional:

```python
from promptchain.observability import MLflowObserver

@track_session()
def _launch_tui(...):
    # ... existing TUI launch code ...

    # Optional: Register MLflow observer if enabled
    try:
        mlflow_observer = MLflowObserver()
        if mlflow_observer.is_available():
            # TODO: Register with active agent's chain
            logger.info("MLflow plugin activated")
    except Exception as e:
        logger.debug(f"MLflow plugin not available: {e}")
```

---

## Installation & Usage

### For Users (Internal Observability Only)

**No MLflow required!** Just use PromptChain normally:

```bash
# Install PromptChain
pip install promptchain

# Run CLI
promptchain --verbose

# ObservePanel shows LLM calls via CallbackManager
```

### For Developers (With MLflow Plugin)

**Optional MLflow for detailed tracking:**

```bash
# 1. Install MLflow plugin
pip install mlflow

# 2. Enable MLflow
export PROMPTCHAIN_MLFLOW_ENABLED=true

# 3. Start MLflow server
mlflow ui --port 5000 &

# 4. Run PromptChain
promptchain --verbose

# Now you get:
# - TUI shows real-time calls (CallbackManager)
# - MLflow UI shows detailed metrics (http://localhost:5000)
```

---

## Benefits of This Architecture

1. **Zero Required Dependencies**: Internal observability works out-of-the-box
2. **Optional MLflow**: Install only when needed for development
3. **Systems in Sync**: CallbackManager → MLflowObserver keeps data consistent
4. **Easy Plugin Management**: Enable/disable MLflow with env var
5. **Performance**: No overhead when MLflow disabled
6. **Clean Separation**: Core vs plugin code clearly separated

---

## Migration Path from Current Implementation

### Phase 1: Fix 0/0 Issue (Immediate)
- [ ] Add callback bridge in `app.py:on_mount()` (Step 1 above)
- [ ] Test ObservePanel shows LLM calls

### Phase 2: Create MLflow Plugin (This Sprint)
- [ ] Create `mlflow_observer.py` (Step 2 above)
- [ ] Make MLflow optional in `setup.py`
- [ ] Document plugin installation

### Phase 3: Deprecate Decorator Approach (Future)
- [ ] Mark `@track_llm_call` decorators as deprecated
- [ ] Remove decorators from core code
- [ ] Keep only MLflowObserver for optional tracking

---

## Setup.py Changes

Make MLflow optional:

```python
setup(
    name="promptchain",
    install_requires=[
        "litellm>=1.0.0",
        "python-dotenv>=1.0.0",
        # NO mlflow in required dependencies!
    ],
    extras_require={
        "dev": [
            "mlflow>=2.9.0",  # Optional dev dependency
            "pytest>=7.0.0",
            "black>=23.0.0",
        ]
    }
)
```

Install for development:
```bash
pip install "promptchain[dev]"  # With MLflow
pip install promptchain         # Without MLflow (users)
```

---

## Summary

**CORRECT Architecture**:
- **Primary**: CallbackManager (always active, no dependencies)
- **Plugin**: MLflowObserver (optional, requires `pip install mlflow`)
- **Connection**: Callbacks → MLflowObserver when enabled

**WRONG Architecture** (current spec 005 implementation):
- **Primary**: MLflow decorators (requires external server)
- **Secondary**: CallbackManager (becomes redundant)
- **Problem**: Users need MLflow for basic TUI functionality

**Next Steps**:
1. Implement callback bridge (fixes 0/0 immediately)
2. Create MLflowObserver plugin (proper integration)
3. Make MLflow optional dependency
4. Remove invasive decorators (future)
