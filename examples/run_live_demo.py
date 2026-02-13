"""
Quick runner for live demo - shows Phase 2-4 features in action
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from dotenv import load_dotenv
load_dotenv()

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain import PromptChain

# Simple tools
def search_files(query: str) -> str:
    """Search for files in codebase."""
    print(f"\n🔍 EXECUTING: search_files('{query}')")
    results = {
        "authentication": "Found:\n- login.py (JWT authentication)\n- auth_middleware.py (token validation)\n- session_manager.py (session handling)",
        "database": "Found:\n- user_model.py (User ORM)\n- connection.py (DB connection pool)",
    }
    result = results.get(query.lower(), f"No files found for '{query}'")
    print(f"   → {result[:80]}...")
    return result

def read_code(filename: str) -> str:
    """Read code file."""
    print(f"\n📖 EXECUTING: read_code('{filename}')")
    content = f"""# {filename}
def authenticate(username, password):
    user = db.get_user(username)
    if not user or not bcrypt.check(password, user.hash):
        return None
    return create_jwt_token(user.id)
"""
    print(f"   → Returned {len(content)} characters")
    return content

def analyze_security(code: str) -> str:
    """Analyze for security issues."""
    print(f"\n🔐 EXECUTING: analyze_security(<code>)")
    analysis = """Issues found:
✅ Uses bcrypt for hashing
⚠️ Missing rate limiting
⚠️ No MFA support
💡 Recommend: Add rate limiting, enable MFA"""
    print(f"   → {analysis[:60]}...")
    return analysis

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files by topic",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": "Read code file",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_security",
            "description": "Analyze code security",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        }
    }
]

async def main():
    print("=" * 80)
    print("🚀 LIVE DEMO: Phase 2-4 Features")
    print("=" * 80)
    print("\nEnabled Features:")
    print("  ✅ Phase 2: Blackboard Architecture (token optimization)")
    print("  ✅ Phase 3: Chain of Verification (safety)")
    print("  ✅ Phase 3: Checkpointing (stuck detection)")
    print("  ✅ Phase 4: TAO Loop (THINK → ACT → OBSERVE)")
    print("  ✅ Phase 4: Dry Run (prediction)")
    print("\n" + "=" * 80)

    # Create processor with ALL features
    processor = AgenticStepProcessor(
        objective="Analyze authentication security and recommend improvements",
        max_internal_steps=6,
        model_name="gemini/gemini-2.0-flash-exp",  # Fast Gemini model with valid key

        # All phases enabled
        enable_blackboard=True,
        enable_cove=True,
        enable_checkpointing=True,
        enable_tao_loop=True,
        enable_dry_run=True,
        cove_confidence_threshold=0.7,
    )

    # Create chain
    chain = PromptChain(
        models=["gemini/gemini-2.0-flash-exp"],  # Fast Gemini model with valid key
        instructions=[processor]
    )

    # Register tools
    chain.register_tool_function(search_files)
    chain.register_tool_function(read_code)
    chain.register_tool_function(analyze_security)
    chain.add_tools(TOOLS)

    print("\n🎯 Starting analysis...")
    print("Watch for THINK → ACT → OBSERVE phases:\n")

    # Run
    result = await chain.process_prompt_async("Start the analysis")

    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"\n{result}\n")

    # Show Blackboard
    if processor.blackboard:
        print("=" * 80)
        print("📋 BLACKBOARD STATE (Phase 2 - Structured Memory)")
        print("=" * 80)
        state = processor.blackboard.get_state()
        print(f"\n📝 Facts: {len(state['facts_discovered'])}")
        for k, v in list(state['facts_discovered'].items())[:3]:
            print(f"   • {k}: {v[:80]}...")
        print(f"\n👁️ Observations: {len(state['observations'])}")
        for obs in state['observations'][:3]:
            print(f"   • {obs[:80]}...")
        print(f"\n📋 Plan: {len(state['current_plan'])} steps")
        for step in state['current_plan'][:3]:
            print(f"   • {step}")

    print(f"\n📊 Tokens: {processor.total_prompt_tokens + processor.total_completion_tokens:,}")
    print("\n🎉 All Phase 2-4 features demonstrated!\n")

if __name__ == "__main__":
    asyncio.run(main())
