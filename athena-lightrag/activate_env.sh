#!/bin/bash
# Environment activation script for Athena LightRAG MCP Server development

set -e

echo "🚀 Activating Athena LightRAG MCP Server development environment..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root directory."
    exit 1
fi

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: UV is not installed. Please install UV first: https://docs.astral.sh/uv/"
    exit 1
fi

# Sync dependencies if needed
echo "🔄 Syncing dependencies with UV..."
uv sync

# Activate the virtual environment
echo "✅ Environment setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the environment validation:"
echo "  uv run python validate_environment.py"
echo ""
echo "To run tests:"
echo "  uv run pytest"
echo ""
echo "To start developing:"
echo "  export PYTHONPATH=$PYTHONPATH:$(pwd)/src"
echo ""

# Clean environment function to avoid conda interference
clean_environment() {
    # Clear conda-related environment variables that can interfere
    unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
    unset CONDA_PYTHON_EXE
    
    # Ensure PYTHONPATH doesn't interfere with UV environment
    export PYTHONPATH=""
    
    # Update PATH to prioritize the UV virtual environment
    export PATH="$(pwd)/.venv/bin:$PATH"
}

# Source the environment if running via source
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "🎯 Sourcing virtual environment..."
    clean_environment
    source .venv/bin/activate
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    echo "✅ Environment activated! You're ready to develop."
    echo "📋 Environment info:"
    echo "   Python: $(which python)"
    echo "   Version: $(python --version)"
    echo "   Virtual env: $VIRTUAL_ENV"
fi