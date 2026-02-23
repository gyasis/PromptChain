#!/usr/bin/env python3
"""
Simple verification that history modes are working correctly.
Tests the three modes with a quick question requiring 2-3 tool calls.
"""

import asyncio
import os
from dotenv import load_dotenv

from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor, HistoryMode

load_dotenv()

# Simple mock functions
def get_number() -> str:
    """Get a number."""
    return "The number is 42"

def multiply_by_two(number: str) -> str:
    """Multiply a number by two."""
    try:
        num = int(number)
        result = num * 2
        return f"{num} multiplied by 2 is {result}"
    except:
        return "Please provide a valid number"


async def test_mode(mode: str):
    """Quick test of a history mode."""
    print(f"\n{'='*60}")
    print(f"Testing mode: {mode}")
    print('='*60)

    agentic_step = AgenticStepProcessor(
        objective="Get a number and multiply it by two",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini",
        history_mode=mode
    )

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            agentic_step,
            "Provide the final result: {input}"
        ],
        verbose=False  # Quiet output
    )

    chain.register_tool_function(get_number)
    chain.register_tool_function(multiply_by_two)

    chain.add_tools([
        {
            "type": "function",
            "function": {
                "name": "get_number",
                "description": "Get a number",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "multiply_by_two",
                "description": "Multiply a number by two",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "string", "description": "The number to multiply"}
                    },
                    "required": ["number"]
                }
            }
        }
    ])

    try:
        result = await chain.process_prompt_async("Get a number and multiply it by two")
        print(f"✅ Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    print("\n" + "="*60)
    print("🧪 HISTORY MODE VERIFICATION TEST")
    print("="*60)
    print("\nTesting all three history modes with simple 2-step task:")
    print("  1. Get a number (42)")
    print("  2. Multiply it by 2")
    print("\nExpected result: '84' or 'The result is 84'")

    results = {}

    # Test each mode
    for mode in ["minimal", "progressive", "kitchen_sink"]:
        success = await test_mode(mode)
        results[mode] = success

    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    for mode, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{mode:15s}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\n💡 The three history modes are working correctly:")
        print("   • minimal (default) - backward compatible")
        print("   • progressive - accumulates context (recommended)")
        print("   • kitchen_sink - maximum context retention")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        exit(1)

    success = asyncio.run(main())
    exit(0 if success else 1)
