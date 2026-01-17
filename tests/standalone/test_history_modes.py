#!/usr/bin/env python3
"""
Test script demonstrating the three history modes in AgenticStepProcessor:
1. minimal (default) - Only last assistant + tool results
2. progressive - Accumulates assistant messages + tool results
3. kitchen_sink - Keeps everything

This script shows how different modes affect multi-hop reasoning.
"""

import asyncio
import os
from dotenv import load_dotenv

from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

load_dotenv()

# Mock database for testing
MOCK_DATABASE = {
    "users": ["Alice", "Bob", "Charlie", "Diana"],
    "locations": {
        "Alice": "New York",
        "Bob": "London",
        "Charlie": "Tokyo",
        "Diana": "Paris"
    },
    "ages": {
        "Alice": 28,
        "Bob": 35,
        "Charlie": 42,
        "Diana": 31
    },
    "hobbies": {
        "Alice": "painting",
        "Bob": "chess",
        "Charlie": "photography",
        "Diana": "cooking"
    }
}


def get_users() -> str:
    """Get list of all users."""
    return f"Found {len(MOCK_DATABASE['users'])} users: {', '.join(MOCK_DATABASE['users'])}"


def get_location(user: str) -> str:
    """Get user's location."""
    location = MOCK_DATABASE["locations"].get(user)
    if location:
        return f"{user} lives in {location}"
    return f"No location found for {user}"


def get_age(user: str) -> str:
    """Get user's age."""
    age = MOCK_DATABASE["ages"].get(user)
    if age:
        return f"{user} is {age} years old"
    return f"No age found for {user}"


def get_hobby(user: str) -> str:
    """Get user's hobby."""
    hobby = MOCK_DATABASE["hobbies"].get(user)
    if hobby:
        return f"{user} enjoys {hobby}"
    return f"No hobby found for {user}"


def create_chain_with_mode(history_mode: str, max_context_tokens: int = None):
    """Create a PromptChain with specified history mode."""

    agentic_step = AgenticStepProcessor(
        objective="""
        Answer the user's question by gathering information from multiple tool calls.

        MULTI-HOP REASONING STRATEGY:
        1. First, identify what information you need
        2. Make multiple tool calls to gather all required data
        3. Connect information from different calls
        4. Synthesize a comprehensive answer

        NOTE: Each tool call should build on previous knowledge.
        """,
        max_internal_steps=8,
        model_name="openai/gpt-4o-mini",
        history_mode=history_mode,
        max_context_tokens=max_context_tokens
    )

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Understand the question: {input}",
            agentic_step,
            "Provide final answer based on gathered information: {input}"
        ],
        verbose=True
    )

    # Register tools
    chain.register_tool_function(get_users)
    chain.register_tool_function(get_location)
    chain.register_tool_function(get_age)
    chain.register_tool_function(get_hobby)

    # Add tool schemas
    chain.add_tools([
        {
            "type": "function",
            "function": {
                "name": "get_users",
                "description": "Get list of all users in the system",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_location",
                "description": "Get the location where a user lives",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string", "description": "Username"}
                    },
                    "required": ["user"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_age",
                "description": "Get the age of a user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string", "description": "Username"}
                    },
                    "required": ["user"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_hobby",
                "description": "Get the hobby of a user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string", "description": "Username"}
                    },
                    "required": ["user"]
                }
            }
        }
    ])

    return chain


async def test_multi_hop_question(mode: str, question: str):
    """Test a multi-hop question with the specified history mode."""

    print("\n" + "=" * 80)
    print(f"🔍 TESTING HISTORY MODE: {mode.upper()}")
    print("=" * 80)
    print(f"Question: {question}")
    print("-" * 80)

    chain = create_chain_with_mode(
        history_mode=mode,
        max_context_tokens=2000 if mode == "minimal" else 4000
    )

    try:
        result = await chain.process_prompt_async(question)
        print("\n" + "=" * 80)
        print("✅ FINAL ANSWER:")
        print("=" * 80)
        print(result)
        print("\n")
        return result
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        return None


async def main():
    """Run comprehensive tests of all three history modes."""

    print("\n" + "=" * 80)
    print("🧪 AGENTIC STEP PROCESSOR - HISTORY MODE TESTING")
    print("=" * 80)
    print("\nThis test demonstrates how different history modes affect multi-hop reasoning.")
    print("\nTest Question: 'Who lives in Tokyo and what are their age and hobby?'")
    print("\nExpected behavior:")
    print("  1. Call get_users() to see all users")
    print("  2. Call get_location() for each user to find who lives in Tokyo")
    print("  3. Call get_age() and get_hobby() for that user")
    print("\n" + "=" * 80)

    test_question = "Who lives in Tokyo and what are their age and hobby?"

    # Test 1: MINIMAL mode (baseline - original behavior)
    result_minimal = await test_multi_hop_question("minimal", test_question)

    # Test 2: PROGRESSIVE mode
    result_progressive = await test_multi_hop_question("progressive", test_question)

    # Test 3: KITCHEN_SINK mode
    result_kitchen = await test_multi_hop_question("kitchen_sink", test_question)

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY OF RESULTS")
    print("=" * 80)

    print("\n🔹 MINIMAL MODE (original behavior):")
    print("   - Only keeps last assistant message + tool results")
    print("   - May lose context from earlier tool calls")
    print(f"   - Result: {result_minimal[:100] if result_minimal else 'Failed'}...")

    print("\n🔹 PROGRESSIVE MODE (recommended for multi-hop):")
    print("   - Accumulates assistant messages + tool results")
    print("   - Preserves reasoning chain across iterations")
    print(f"   - Result: {result_progressive[:100] if result_progressive else 'Failed'}...")

    print("\n🔹 KITCHEN_SINK MODE (maximum context):")
    print("   - Keeps everything including all intermediate steps")
    print("   - Best for complex reasoning but uses most tokens")
    print(f"   - Result: {result_kitchen[:100] if result_kitchen else 'Failed'}...")

    print("\n" + "=" * 80)
    print("✅ Testing Complete!")
    print("=" * 80)
    print("\n💡 KEY TAKEAWAYS:")
    print("   - Use 'minimal' for simple single-tool tasks (default, backward compatible)")
    print("   - Use 'progressive' for multi-hop reasoning (recommended)")
    print("   - Use 'kitchen_sink' for maximum context retention")
    print("   - All modes are backward compatible - defaults to 'minimal'")
    print("\n")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or environment variables")
        exit(1)

    asyncio.run(main())
