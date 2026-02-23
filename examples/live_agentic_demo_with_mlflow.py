"""
Live AgenticStepProcessor Demo with MLflow Tracking
Run this in Jupyter or VS Code with # %% cell markers

This shows REAL Phase 2-4 features with MLflow observability:
- THINK phases (explicit reasoning)
- ACT phases (tool execution)
- OBSERVE phases (result analysis)
- Blackboard state evolution
- CoVe verification decisions
- All logged to MLflow
"""

# %% [markdown]
# # Live AgenticStepProcessor with MLflow Observability
#
# This notebook demonstrates **REAL** Phase 2-4 features:
# - Phase 2: Blackboard Architecture
# - Phase 3: Chain of Verification + Checkpointing
# - Phase 4: TAO Loop (THINK → ACT → OBSERVE)
#
# All internal steps are logged to MLflow for visualization.

# %% Setup and Imports
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from dotenv import load_dotenv
load_dotenv()

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain import PromptChain
import asyncio

# MLflow setup
try:
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("live_agentic_demo")
    print("✅ MLflow tracking enabled")
    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  MLflow not available - install with: pip install mlflow")
    MLFLOW_AVAILABLE = False

print("✅ All imports successful!")

# %% Define Real Tools
def search_files(query: str) -> str:
    """Search for files in codebase."""
    print(f"\n🔍 TOOL EXECUTED: search_files('{query}')")
    results = {
        "authentication": "Found 3 files:\n- promptchain/auth/login.py\n- promptchain/auth/jwt_handler.py\n- promptchain/middleware/auth.py",
        "database": "Found 2 files:\n- promptchain/models/user.py\n- promptchain/db/connection.py",
        "api": "Found 2 files:\n- promptchain/api/routes.py\n- promptchain/api/handlers.py"
    }
    return results.get(query.lower(), f"No files found matching '{query}'")

def read_code(filename: str) -> str:
    """Read code file content."""
    print(f"\n📖 TOOL EXECUTED: read_code('{filename}')")
    mock_content = f"""# {filename}
import bcrypt
from datetime import datetime

def authenticate_user(username: str, password: str) -> dict:
    '''Authenticate user with credentials.'''
    # Validate input
    if not username or not password:
        return {{"success": False, "error": "Missing credentials"}}

    # Check password
    user = db.get_user(username)
    if not user:
        return {{"success": False, "error": "User not found"}}

    if not bcrypt.checkpw(password.encode(), user['password_hash']):
        return {{"success": False, "error": "Invalid password"}}

    return {{"success": True, "user_id": user['id']}}
"""
    return mock_content

def analyze_security(code: str) -> str:
    """Analyze code for security issues."""
    print(f"\n🔐 TOOL EXECUTED: analyze_security(<code snippet>)")
    return """Security Analysis Results:

✅ Strengths:
- Uses bcrypt for password hashing
- Proper input validation
- Error messages don't reveal user existence

⚠️  Potential Issues:
- No rate limiting for failed login attempts
- Missing MFA support
- Password complexity not enforced

💡 Recommendations:
1. Add rate limiting (e.g., 5 attempts per minute)
2. Implement MFA for sensitive accounts
3. Enforce password complexity rules
"""

# %% Create Tool Definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files in the codebase by topic (e.g., 'authentication', 'database', 'api')",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": "Read the content of a code file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to read"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_security",
            "description": "Analyze code for security vulnerabilities and best practices",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to analyze"}
                },
                "required": ["code"]
            }
        }
    }
]

print("✅ Tools defined")

# %% Create AgenticStepProcessor with ALL Phase 2-4 Features

print("\n" + "=" * 80)
print("🚀 Creating AgenticStepProcessor with ALL features enabled")
print("=" * 80)

agentic_step = AgenticStepProcessor(
    objective="Analyze the authentication system for security vulnerabilities and recommend improvements",
    max_internal_steps=8,
    model_name="gemini/gemini-2.0-flash-exp",

    # Phase 1: Two-Tier Routing (cost optimization)
    enable_two_tier_routing=True,
    fallback_model="gemini/gemini-1.5-flash-8b",

    # Phase 2: Blackboard Architecture (token optimization)
    enable_blackboard=True,

    # Phase 3: Safety & Reliability
    enable_cove=True,
    cove_confidence_threshold=0.7,
    enable_checkpointing=True,

    # Phase 4: TAO Loop + Transparent Reasoning
    enable_tao_loop=True,
    enable_dry_run=True,
)

print("\n✅ AgenticStepProcessor created with:")
print("   ✅ Two-Tier Routing (Phase 1)")
print("   ✅ Blackboard Architecture (Phase 2)")
print("   ✅ Chain of Verification (Phase 3)")
print("   ✅ Checkpointing (Phase 3)")
print("   ✅ TAO Loop (Phase 4)")
print("   ✅ Dry Run Prediction (Phase 4)")

# %% Create PromptChain and Register Tools

chain = PromptChain(
    models=["gemini/gemini-2.0-flash-exp"],
    instructions=[agentic_step]
)

# Register tool functions
chain.register_tool_function(search_files)
chain.register_tool_function(read_code)
chain.register_tool_function(analyze_security)

# Add tool definitions
chain.add_tools(TOOLS)

print("✅ PromptChain configured with tools")

# %% Run Analysis with MLflow Tracking

print("\n" + "🎯" * 40)
print("STARTING AGENTIC ANALYSIS")
print("🎯" * 40)
print("\nWatch for THINK → ACT → OBSERVE phases...\n")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="live_agentic_demo"):
        # Log configuration
        mlflow.log_params({
            "objective": agentic_step.objective,
            "max_steps": agentic_step.max_internal_steps,
            "model": agentic_step.model_name,
            "fallback_model": agentic_step.fallback_model,
            "enable_blackboard": agentic_step.enable_blackboard,
            "enable_cove": agentic_step.enable_cove,
            "enable_tao_loop": agentic_step.enable_tao_loop,
            "enable_dry_run": agentic_step.enable_dry_run,
        })

        # Run the analysis
        result = await chain.process_prompt_async("Begin the security analysis")

        # Log metrics
        mlflow.log_metrics({
            "total_prompt_tokens": agentic_step.total_prompt_tokens,
            "total_completion_tokens": agentic_step.total_completion_tokens,
            "total_tokens": agentic_step.total_prompt_tokens + agentic_step.total_completion_tokens,
        })

        # Log Blackboard state as artifact
        if agentic_step.blackboard:
            import json
            blackboard_state = agentic_step.blackboard.get_state()
            mlflow.log_dict(blackboard_state, "blackboard_final_state.json")

        print(f"\n✅ MLflow run logged - view at http://localhost:5000")
else:
    # Run without MLflow
    result = await chain.process_prompt_async("Begin the security analysis")

# %% Display Results

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)

print("\n📊 Final Analysis:")
print(result)

# %% Show Blackboard State (Structured Memory)

if agentic_step.blackboard:
    print("\n" + "=" * 80)
    print("📋 BLACKBOARD STATE - Structured Memory (Phase 2)")
    print("=" * 80)

    state = agentic_step.blackboard.get_state()

    print(f"\n🎯 Objective:")
    print(f"   {state['objective']}")

    print(f"\n📝 Facts Discovered ({len(state['facts_discovered'])}):")
    for i, (key, value) in enumerate(list(state['facts_discovered'].items())[:5], 1):
        print(f"   {i}. {key}:")
        print(f"      {value[:150]}...")

    print(f"\n👁️  Observations ({len(state['observations'])}):")
    for i, obs in enumerate(state['observations'][:5], 1):
        print(f"   {i}. {obs[:150]}...")

    print(f"\n📋 Execution Plan ({len(state['current_plan'])} steps):")
    for i, step in enumerate(state['current_plan'][:5], 1):
        print(f"   {i}. {step}")

    print(f"\n🔧 Tool Results (last {min(3, len(state['last_tool_results']))}):")
    for i, tool_result in enumerate(state['last_tool_results'][:3], 1):
        print(f"   {i}. {tool_result['tool_name']}:")
        print(f"      {tool_result['result'][:100]}...")

# %% Display Metrics

print("\n" + "=" * 80)
print("📊 EXECUTION METRICS")
print("=" * 80)

print(f"\n🎯 Token Usage:")
print(f"   Prompt tokens: {agentic_step.total_prompt_tokens:,}")
print(f"   Completion tokens: {agentic_step.total_completion_tokens:,}")
print(f"   Total tokens: {agentic_step.total_prompt_tokens + agentic_step.total_completion_tokens:,}")

if agentic_step.enable_two_tier_routing:
    print(f"\n⚡ Two-Tier Routing (Phase 1):")
    print(f"   Fast model calls: {agentic_step._fast_model_count}")
    print(f"   Slow model calls: {agentic_step._slow_model_count}")
    total_calls = agentic_step._fast_model_count + agentic_step._slow_model_count
    if total_calls > 0:
        fast_pct = (agentic_step._fast_model_count / total_calls) * 100
        print(f"   Cost savings: ~{fast_pct:.1f}% of calls routed to cheap model")

# %% Summary

print("\n" + "=" * 80)
print("🎉 DEMONSTRATION COMPLETE")
print("=" * 80)

print("\n✅ You just saw:")
print("   • THINK phases - Explicit reasoning before actions")
print("   • ACT phases - Tool executions")
print("   • OBSERVE phases - Result analysis")
print("   • Blackboard - Structured state management (Phase 2)")
print("   • CoVe - Pre-execution verification (Phase 3)")
print("   • TAO Loop - Transparent reasoning (Phase 4)")

if MLFLOW_AVAILABLE:
    print("\n🔍 View detailed metrics in MLflow UI:")
    print("   1. Run: mlflow ui")
    print("   2. Open: http://localhost:5000")
    print("   3. Navigate to 'live_agentic_demo' experiment")
    print("   4. View artifacts: blackboard_final_state.json")

print("\n🚀 All Phase 2-4 features working together!")
