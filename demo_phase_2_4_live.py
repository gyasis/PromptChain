"""
Live demonstration of AgenticStepProcessor with ALL Phase 2-4 features enabled.
Shows THINK → ACT → OBSERVE in real-time.
"""
import sys
import os
from dotenv import load_dotenv
load_dotenv()

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain import PromptChain
import asyncio

# Simple mock tools for demonstration
def search_files(query: str) -> str:
    """Mock file search tool."""
    print(f"\n🔍 EXECUTING TOOL: search_files('{query}')")
    results = {
        "authentication": "Found 3 files: login.py, jwt_handler.py, auth_middleware.py",
        "database": "Found 2 files: models.py, connection.py",
        "api": "Found 2 files: routes.py, handlers.py"
    }
    result = results.get(query.lower(), f"No files found for '{query}'")
    print(f"   Result: {result}")
    return result

def read_code(filename: str) -> str:
    """Mock file reading tool."""
    print(f"\n📖 EXECUTING TOOL: read_code('{filename}')")
    mock_code = f"# Content of {filename}\ndef example_function():\n    pass"
    print(f"   Result: Returned {len(mock_code)} characters")
    return mock_code

def analyze_security(code: str) -> str:
    """Mock security analysis tool."""
    print(f"\n🔐 EXECUTING TOOL: analyze_security(<code>)")
    result = "Security Analysis:\n- ✅ No SQL injection vulnerabilities\n- ⚠️  Missing input validation\n- ✅ Proper authentication checks"
    print(f"   Result: Found 3 findings")
    return result

async def run_demo():
    """Run live demonstration of Phase 2-4 features."""

    print("=" * 80)
    print("🚀 LIVE DEMONSTRATION: AgenticStepProcessor with Phase 2-4 Features")
    print("=" * 80)
    print("\nFeatures Enabled:")
    print("  ✅ Phase 1: Two-Tier Routing (cost optimization)")
    print("  ✅ Phase 2: Blackboard Architecture (token optimization)")
    print("  ✅ Phase 3: Chain of Verification (safety)")
    print("  ✅ Phase 3: Checkpointing (stuck state detection)")
    print("  ✅ Phase 4: TAO Loop (Think-Act-Observe)")
    print("  ✅ Phase 4: Dry Run (prediction before execution)")
    print("\n" + "=" * 80)
    print("\n⏳ Starting agentic workflow...\n")

    # Create AgenticStepProcessor with ALL features enabled
    agentic_step = AgenticStepProcessor(
        objective="Analyze the authentication system in the codebase for security issues",
        max_internal_steps=5,
        model_name="gemini/gemini-2.0-flash-exp",  # Fast model for demo

        # Phase 1: Two-Tier Routing
        enable_two_tier_routing=True,
        fallback_model="gemini/gemini-1.5-flash-8b",

        # Phase 2: Blackboard Architecture
        enable_blackboard=True,

        # Phase 3: Safety & Reliability
        enable_cove=True,
        cove_confidence_threshold=0.7,
        enable_checkpointing=True,

        # Phase 4: TAO Loop + Transparent Reasoning
        enable_tao_loop=True,
        enable_dry_run=True,

        verbose=True  # Show all reasoning
    )

    # Create PromptChain with tools
    chain = PromptChain(
        models=["gemini/gemini-2.0-flash-exp"],
        instructions=[agentic_step],
        verbose=True
    )

    # Register tools
    chain.register_tool_function(search_files)
    chain.register_tool_function(read_code)
    chain.register_tool_function(analyze_security)

    chain.add_tools([
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for files in the codebase by topic (e.g., 'authentication', 'database')",
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
                        "filename": {"type": "string", "description": "Name of file to read"}
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_security",
                "description": "Analyze code for security vulnerabilities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to analyze"}
                    },
                    "required": ["code"]
                }
            }
        }
    ])

    print("\n" + "🎯" * 40)
    print("Starting analysis with THINK → ACT → OBSERVE pattern...")
    print("🎯" * 40 + "\n")

    # Run the chain
    result = await chain.process_prompt_async(
        "Start the analysis"
    )

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n📊 Final Result:")
    print(result)

    # Show Blackboard state if available
    if agentic_step.blackboard:
        print("\n" + "=" * 80)
        print("📋 BLACKBOARD STATE (Structured Memory)")
        print("=" * 80)
        state = agentic_step.blackboard.get_state()
        print(f"\n🎯 Objective: {state['objective']}")
        print(f"\n📝 Facts Discovered: {len(state['facts_discovered'])}")
        for key, value in list(state['facts_discovered'].items())[:3]:
            print(f"   • {key}: {value[:100]}...")
        print(f"\n👁️  Observations: {len(state['observations'])}")
        for obs in state['observations'][:3]:
            print(f"   • {obs[:100]}...")
        print(f"\n📋 Current Plan: {len(state['current_plan'])} steps")
        for step in state['current_plan'][:3]:
            print(f"   • {step[:100]}...")

    print("\n" + "=" * 80)
    print("🎉 Demonstration Complete!")
    print("=" * 80)
    print("\nYou just saw:")
    print("  ✅ Explicit THINK phases (reasoning before action)")
    print("  ✅ ACT phases (tool execution)")
    print("  ✅ OBSERVE phases (result analysis)")
    print("  ✅ Blackboard maintaining structured state")
    print("  ✅ CoVe verifying tool calls before execution")
    print("  ✅ Dry run predicting outcomes")
    print("\nAll Phase 2-4 features working together! 🚀")

if __name__ == "__main__":
    asyncio.run(run_demo())
