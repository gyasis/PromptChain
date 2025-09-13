#!/bin/bash
"""
Interactive LightRAG Multi-Hop Reasoning Test Runner
===================================================
Easy script to run the enhanced LightRAG test with proper environment setup.
"""

echo "🏥 Starting Interactive LightRAG Multi-Hop Reasoning Test"
echo "========================================================"

# Check if we're in the right directory
if [ ! -f "lightrag_multihop_isolation_test.py" ]; then
    echo "❌ Error: Please run this script from the testing/ directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: lightrag_multihop_isolation_test.py"
    exit 1
fi

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: UV is not installed or not in PATH"
    echo "   Please install UV: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if we're in a UV project
if [ ! -f "../pyproject.toml" ]; then
    echo "❌ Error: Not in a UV project directory"
    echo "   Please run from the athena-lightrag project root"
    exit 1
fi

echo "✅ Environment checks passed"
echo "🚀 Starting interactive test session..."
echo ""

# Run the test with UV
cd ..
uv run python testing/lightrag_multihop_isolation_test.py
