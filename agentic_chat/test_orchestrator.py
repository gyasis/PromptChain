#!/usr/bin/env python3
"""
Quick test script to validate AgenticStepProcessor orchestrator routing logic
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_team_chat import create_agentic_orchestrator

# Test queries with expected routing
test_cases = [
    {
        "query": "What is Rust's Candle library?",
        "expected": "research",
        "reason": "Unknown/recent tech requires web research"
    },
    {
        "query": "Explain how neural networks work",
        "expected": "documentation",
        "reason": "Well-known concept in training data"
    },
    {
        "query": "Create a backup script for /home/user/data",
        "expected": "coding",
        "reason": "Needs write_script tool"
    },
    {
        "query": "Run ls -la in the current directory",
        "expected": "terminal",
        "reason": "Needs execute_terminal_command"
    }
]

agent_descriptions = {
    "research": "Gemini-powered web research specialist with MCP tools",
    "analysis": "Data analysis expert (no tools)",
    "coding": "Script writing specialist with write_script tool",
    "terminal": "System operations with execute_terminal_command tool",
    "documentation": "Technical writing specialist (no tools)",
    "synthesis": "Strategic planning specialist (no tools)"
}

async def test_routing():
    """Test the orchestrator routing decisions"""
    print("=" * 80)
    print("🧪 TESTING AGENTIC ORCHESTRATOR ROUTING")
    print("=" * 80)

    # Create the orchestrator
    orchestrator_wrapper = create_agentic_orchestrator(agent_descriptions)

    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']} ({test['reason']})")

        # Call the orchestrator
        result_json = await orchestrator_wrapper(
            user_input=test['query'],
            conversation_history=[],
            agent_descriptions=agent_descriptions
        )

        print(f"Result: {result_json}")

        # Parse JSON to check chosen_agent
        import json
        try:
            result_dict = json.loads(result_json)
            chosen = result_dict.get('chosen_agent')
            reasoning = result_dict.get('reasoning', 'No reasoning provided')

            match = "✅ CORRECT" if chosen == test['expected'] else "❌ WRONG"
            print(f"{match} - Chose '{chosen}'")
            print(f"Reasoning: {reasoning}")

            results.append({
                "test": test['query'],
                "expected": test['expected'],
                "got": chosen,
                "correct": chosen == test['expected']
            })
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Failed to parse JSON - {e}")
            results.append({
                "test": test['query'],
                "expected": test['expected'],
                "got": "PARSE_ERROR",
                "correct": False
            })

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Target: 95%+ (AgenticStepProcessor with multi-hop reasoning)")

    if accuracy >= 95:
        print("\n🎉 SUCCESS! Orchestrator exceeds 95% routing accuracy target")
    elif accuracy >= 75:
        print("\n⚠️  Good but needs improvement. Check reasoning logic.")
    else:
        print("\n❌ FAILED. Orchestrator needs significant refinement.")

    return results

if __name__ == "__main__":
    asyncio.run(test_routing())
