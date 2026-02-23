#!/usr/bin/env python3
"""
Comprehensive tests for time context hooks.

Tests the logic that:
1. Removes outdated year references (2024, 2025)
2. Replaces with current year
3. Injects time context when missing
"""

import json
import subprocess
import sys
import os
from datetime import datetime, timedelta


def get_script_dir():
    """Get the directory where this test script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def test_gemini_hook(tool_input):
    """
    Test the Gemini hook using the standalone Python script.
    """
    script_path = os.path.join(get_script_dir(), "gemini_time_cleaner.py")

    # Simulate Claude Code's input format
    claude_input = {
        "tool_input": tool_input
    }

    try:
        result = subprocess.run(
            ["python3", script_path],
            input=json.dumps(claude_input),
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            print(f"❌ Hook script failed: {result.stderr}")
            return None

        output = json.loads(result.stdout)
        return output.get('updatedInput', {})

    except Exception as e:
        print(f"❌ Error running hook: {e}")
        return None


def run_tests():
    """Run all test cases."""
    print("=" * 80)
    print("TESTING TIME CONTEXT HOOKS - Using Standalone Python Scripts")
    print("=" * 80)

    current_year = datetime.now().year
    current_date = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = datetime.now() - timedelta(days=180)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Basic year replacement
    print("\n" + "-" * 80)
    print("Test 1: Basic Year Replacement (2024 -> current year)")
    print("-" * 80)

    input_data = {"query": "2024 LangChain tutorial"}
    result = test_gemini_hook(input_data)

    if result and "2024" not in result.get("query", "") and str(current_year) in result.get("query", ""):
        print(f"✅ PASS: Year replaced")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Year not replaced")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 2: Pattern matching (2025 best practices -> current year best practices)
    print("\n" + "-" * 80)
    print("Test 2: Pattern Matching (2025 best practices)")
    print("-" * 80)

    input_data = {"query": "2025 best practices for Python"}
    result = test_gemini_hook(input_data)

    if result and "2025" not in result.get("query", "") and str(current_year) in result.get("query", ""):
        print(f"✅ PASS: Pattern replaced")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Pattern not replaced correctly")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 3: Time context injection
    print("\n" + "-" * 80)
    print("Test 3: Time Context Injection")
    print("-" * 80)

    input_data = {"query": "Python async patterns"}
    result = test_gemini_hook(input_data)

    if result and "Current date:" in result.get("query", ""):
        print(f"✅ PASS: Time context added")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Time context not added")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 4: No duplicate time context
    print("\n" + "-" * 80)
    print("Test 4: No Duplicate Time Context")
    print("-" * 80)

    input_data = {"query": "AI trends (Current date: 2026-01-15)"}
    result = test_gemini_hook(input_data)

    # Count occurrences of "Current date:"
    occurrences = result.get("query", "").count("Current date:") if result else 0

    if occurrences == 1:
        print(f"✅ PASS: No duplication")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Time context duplicated ({occurrences} occurrences)")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 5: Gemini tool with 'prompt' parameter
    print("\n" + "-" * 80)
    print("Test 5: Gemini Tool (ask_gemini with 'prompt' parameter)")
    print("-" * 80)

    input_data = {"prompt": "What are 2024 AI trends?"}
    result = test_gemini_hook(input_data)

    if result and "2024" not in result.get("prompt", "") and str(current_year) in result.get("prompt", ""):
        print(f"✅ PASS: Gemini prompt parameter handled")
        print(f"   Input:  {input_data['prompt']}")
        print(f"   Output: {result['prompt']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Gemini prompt not processed")
        print(f"   Input:  {input_data['prompt']}")
        print(f"   Output: {result.get('prompt') if result else 'None'}")
        tests_failed += 1

    # Test 6: Gemini tool with 'topic' parameter
    print("\n" + "-" * 80)
    print("Test 6: Gemini Tool (research with 'topic' parameter)")
    print("-" * 80)

    input_data = {"topic": "2025 ML breakthroughs"}
    result = test_gemini_hook(input_data)

    if result and "2025" not in result.get("topic", "") and str(current_year) in result.get("topic", ""):
        print(f"✅ PASS: Gemini topic parameter handled")
        print(f"   Input:  {input_data['topic']}")
        print(f"   Output: {result['topic']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Gemini topic not processed")
        print(f"   Input:  {input_data['topic']}")
        print(f"   Output: {result.get('topic') if result else 'None'}")
        tests_failed += 1

    # Test 7: Empty query handling
    print("\n" + "-" * 80)
    print("Test 7: Empty Query Handling")
    print("-" * 80)

    input_data = {"query": ""}
    result = test_gemini_hook(input_data)

    if result and result.get("query") == "":
        print(f"✅ PASS: Empty query handled gracefully")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Empty query not handled correctly")
        print(f"   Output: {result}")
        tests_failed += 1

    # Test 8: Multiple year replacements
    print("\n" + "-" * 80)
    print("Test 8: Multiple Year Replacements")
    print("-" * 80)

    input_data = {"query": "Compare 2024 vs 2025 Python frameworks"}
    result = test_gemini_hook(input_data)

    if result and "2024" not in result.get("query", "") and "2025" not in result.get("query", ""):
        print(f"✅ PASS: Multiple years replaced")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Multiple years not replaced")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 9: Year in middle of sentence
    print("\n" + "-" * 80)
    print("Test 9: Year in Middle of Sentence")
    print("-" * 80)

    input_data = {"query": "Looking for 2024 tutorial on Docker"}
    result = test_gemini_hook(input_data)

    if result and "2024 tutorial" not in result.get("query", "") and f"{current_year} tutorial" in result.get("query", ""):
        print(f"✅ PASS: Year in pattern replaced")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Year in pattern not replaced")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 10: Case insensitive pattern matching
    print("\n" + "-" * 80)
    print("Test 10: Case Insensitive Pattern Matching")
    print("-" * 80)

    input_data = {"query": "2024 Tutorial for beginners"}
    result = test_gemini_hook(input_data)

    if result and "2024 Tutorial" not in result.get("query", "") and str(current_year) in result.get("query", ""):
        print(f"✅ PASS: Case insensitive match works")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Case insensitive match failed")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 11: Query with "latest" should not add time context
    print("\n" + "-" * 80)
    print("Test 11: Query with 'latest' keyword (no duplicate context)")
    print("-" * 80)

    input_data = {"query": "latest Python frameworks"}
    result = test_gemini_hook(input_data)

    occurrences = result.get("query", "").count("Current date:") if result else 0

    if result and occurrences == 0:
        print(f"✅ PASS: No time context added when 'latest' present")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Time context added unnecessarily")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Test 12: Query with specific date should not add time context
    print("\n" + "-" * 80)
    print("Test 12: Query with specific date (no duplicate context)")
    print("-" * 80)

    input_data = {"query": "Python updates since 2023-01-01"}
    result = test_gemini_hook(input_data)

    occurrences = result.get("query", "").count("Current date:") if result else 0

    if result and occurrences == 0:
        print(f"✅ PASS: No time context added when date present")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result['query']}")
        tests_passed += 1
    else:
        print(f"❌ FAIL: Time context added unnecessarily")
        print(f"   Input:  {input_data['query']}")
        print(f"   Output: {result.get('query') if result else 'None'}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print(f"📊 Success Rate: {tests_passed / (tests_passed + tests_failed) * 100:.1f}%")
    print("=" * 80)

    if tests_failed == 0:
        print("\n🎉 All tests passed! Hook is working correctly.")
    else:
        print(f"\n⚠️ {tests_failed} test(s) failed. Review the failures above.")

    print("\n📝 Next Steps:")
    print("   1. Hook is installed at: ~/.claude/hooks/comprehensive_time_context_hook.json")
    print("   2. Open Claude Code and type: /hooks")
    print("   3. Find and APPROVE the comprehensive_time_context_hook")
    print("   4. Test in Claude Code with queries like:")
    print("      - 'Search for 2024 LangChain tutorial'")
    print("      - 'Use gemini_research to find 2025 Python updates'")

    return tests_passed, tests_failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
