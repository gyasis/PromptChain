# Installation Guide

This guide covers different installation modes for PromptChain based on your use case.

## Installation Modes

### 1. Core Installation (End Users)

For most users who want to use PromptChain for building LLM applications:

```bash
pip install promptchain
```

This installs only the core dependencies required for:
- LLM access via LiteLLM
- Prompt chaining and agent frameworks
- Token counting and execution history
- Basic utilities

**Note**: This does NOT include MLflow or development tools.

### 2. CLI Installation (Interactive Terminal)

If you want to use the interactive CLI with terminal UI:

```bash
pip install "promptchain[cli]"
```

This adds:
- Textual (terminal UI framework)
- Click (CLI framework)
- Prompt_toolkit (input handling)

### 3. Development Installation (Contributors)

For developers working on PromptChain or who want observability features:

```bash
pip install "promptchain[dev]"
```

This adds:
- **MLflow** (observability and tracking)
- Testing frameworks (pytest, pytest-asyncio, pytest-mock)
- Code quality tools (black, isort, flake8, mypy)
- Documentation tools (mkdocs)

Alternatively, install from requirements-dev.txt:

```bash
pip install -r requirements-dev.txt
```

### 4. Full Installation (Everything)

To install all optional features:

```bash
pip install "promptchain[all]"
```

This includes CLI, development tools, and MLflow observability.

### 5. From Source (Development)

For active development:

```bash
# Clone the repository
git clone https://github.com/gyasis/promptchain.git
cd promptchain

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements files
pip install -r requirements.txt       # Core only
pip install -r requirements-dev.txt   # Core + dev tools
```

## MLflow Observability (Optional)

PromptChain includes optional MLflow integration for tracking LLM calls, performance metrics, and debugging.

**MLflow is NOT required** for core functionality. The observability system gracefully degrades when MLflow is not installed.

### Enabling MLflow Tracking

If you have MLflow installed (via `[dev]` extras or manually), enable tracking via environment variables:

```bash
# In your .env file or shell
export PROMPTCHAIN_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://localhost:5000
export PROMPTCHAIN_MLFLOW_EXPERIMENT=my-llm-project
```

Or programmatically:

```python
import os
from promptchain.observability.config import is_enabled

# Enable MLflow tracking
os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

# Check if tracking is enabled
if is_enabled():
    print("MLflow tracking is active")

# Your PromptChain code here - now tracked automatically
```

### Without MLflow

If MLflow is not installed, decorators become "ghost decorators" with <0.1% overhead:

```python
from promptchain.observability.config import is_enabled

# Check tracking status (will be False if MLflow not installed)
enabled = is_enabled()
print(f"Tracking enabled: {enabled}")  # False

# Your PromptChain code works normally without tracking
```

## Environment Configuration

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Optional MLflow tracking server
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Verification

### Test Core Installation

```bash
python -c "from promptchain import PromptChain; print('Core OK')"
```

### Test MLflow Integration (if installed)

```bash
python -c "import mlflow; from promptchain.observability import enable_tracking; print('MLflow OK')"
```

### Test CLI (if installed)

```bash
promptchain --help
```

## Dependency Details

### Core Dependencies (Always Installed)

| Package | Purpose |
|---------|---------|
| litellm | Unified LLM API access (OpenAI, Anthropic, etc.) |
| python-dotenv | Environment variable management |
| pydantic | Data validation |
| tiktoken | Token counting for history management |
| rich | Console output formatting |
| httpx | HTTP requests |
| pyyaml | YAML configuration |
| nest-asyncio | Async support for nested event loops |

### Optional Dependencies

| Package | Extra | Purpose |
|---------|-------|---------|
| mlflow | `[dev]` | Observability and tracking |
| textual | `[cli]` | Terminal UI framework |
| click | `[cli]` | CLI framework |
| pytest | `[dev]` | Testing framework |
| black | `[dev]` | Code formatting |
| mypy | `[dev]` | Type checking |

## Troubleshooting

### MLflow Not Found (Expected)

If you see:
```
ImportError: No module named 'mlflow'
```

This is normal if you installed core only. MLflow is optional. Either:
1. Ignore it (tracking will be disabled automatically)
2. Install dev extras: `pip install "promptchain[dev]"`
3. Install MLflow separately: `pip install mlflow>=2.9.0`

### Import Errors

If you see import errors for core packages:
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Version Conflicts

If you have version conflicts:
```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install promptchain[dev]
```

## Upgrading

To upgrade PromptChain:

```bash
pip install --upgrade promptchain         # Core only
pip install --upgrade "promptchain[dev]"  # With dev tools
pip install --upgrade "promptchain[all]"  # Everything
```

## Uninstalling

```bash
pip uninstall promptchain
```

This removes PromptChain but keeps dependencies. To remove everything:

```bash
pip freeze | grep -E "promptchain|litellm|mlflow" | xargs pip uninstall -y
```
