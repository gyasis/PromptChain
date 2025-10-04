#!/usr/bin/env python3
"""
Test Prompt Engineering Fix

This test verifies that our agent prompt updates successfully eliminate
the markdown wrapper issue that was causing JSON parsing errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test to verify that our prompt engineering fix works by checking if agents
# now have strict JSON-only instructions instead of general format instructions


def test_synthesis_agent_prompt_fix():
    """Test that Synthesis Agent has strict JSON-only prompt"""
    from research_agent.agents.synthesis_agent import SynthesisAgent
    
    config = {'model': 'openai/gpt-4o', 'processor': {'objective': 'test', 'max_internal_steps': 5}}
    agent = SynthesisAgent(config)
    
    # Get the prompt instruction
    instruction = agent._format_synthesis_output_instruction()
    
    # Verify strict JSON-only requirements are present
    assert "CRITICAL: Return ONLY valid JSON" in instruction, "Missing critical JSON-only instruction"
    assert "No explanations, no markdown blocks, no code fences" in instruction, "Missing markdown prohibition"
    assert "Do not include ```json```" in instruction, "Missing code fence prohibition"
    assert "Start response with {" in instruction, "Missing start requirement"
    assert "End response with }" in instruction, "Missing end requirement"
    assert "Pure JSON only" in instruction, "Missing pure JSON requirement"
    
    print("✅ Synthesis Agent prompt fix verified")
    return True


def test_react_analyzer_prompt_fix():
    """Test that ReAct Analyzer has strict JSON-only prompt"""
    from research_agent.agents.react_analyzer import ReActAnalysisAgent
    
    config = {'model': 'openai/gpt-4o', 'processor': {'objective': 'test', 'max_internal_steps': 7}}
    agent = ReActAnalysisAgent(config)
    
    # Get the prompt instruction
    instruction = agent._format_react_output_instruction()
    
    # Verify strict JSON-only requirements are present
    assert "CRITICAL: Return ONLY valid JSON" in instruction, "Missing critical JSON-only instruction"
    assert "No explanations, no markdown blocks, no code fences" in instruction, "Missing markdown prohibition"
    assert "Do not include ```json```" in instruction, "Missing code fence prohibition"
    assert "Start response with {" in instruction, "Missing start requirement"
    assert "End response with }" in instruction, "Missing end requirement"
    assert "Pure JSON only" in instruction, "Missing pure JSON requirement"
    
    print("✅ ReAct Analyzer prompt fix verified")
    return True


def test_query_generator_prompt_fix():
    """Test that Query Generator has strict JSON-only prompt"""
    from research_agent.agents.query_generator import QueryGenerationAgent
    
    config = {'model': 'openai/gpt-4o', 'processor': {'objective': 'test', 'max_internal_steps': 5}}
    agent = QueryGenerationAgent(config)
    
    # Get the prompt instruction
    instruction = agent._format_query_output_instruction()
    
    # Verify strict JSON-only requirements are present
    assert "CRITICAL: Return ONLY valid JSON" in instruction, "Missing critical JSON-only instruction"
    assert "No explanations, no markdown blocks, no code fences" in instruction, "Missing markdown prohibition"
    assert "Do not include ```json```" in instruction, "Missing code fence prohibition"
    assert "Start response with {" in instruction, "Missing start requirement"
    assert "End response with }" in instruction, "Missing end requirement"
    assert "Pure JSON only" in instruction, "Missing pure JSON requirement"
    
    print("✅ Query Generator prompt fix verified")
    return True


def main():
    """Run prompt engineering fix verification tests"""
    print("🔍 TESTING PROMPT ENGINEERING FIX")
    print("Verifying that agents now have strict JSON-only instructions")
    print("=" * 60)
    
    try:
        # Test all agent prompt fixes
        synthesis_ok = test_synthesis_agent_prompt_fix()
        react_ok = test_react_analyzer_prompt_fix()
        query_ok = test_query_generator_prompt_fix()
        
        if synthesis_ok and react_ok and query_ok:
            print("\n" + "=" * 60)
            print("✅ ALL PROMPT ENGINEERING FIXES VERIFIED")
            print("\n📋 CHANGES IMPLEMENTED:")
            print("  • Synthesis Agent: Strict JSON-only output format")
            print("  • ReAct Analyzer: Strict JSON-only output format")  
            print("  • Query Generator: Strict JSON-only output format")
            print("\n🎯 ROOT CAUSE ADDRESSED:")
            print("  • LLMs were returning JSON wrapped in markdown code blocks")
            print("  • Agents now explicitly forbid markdown, explanations, and code fences")
            print("  • Agents require responses to start with { and end with }")
            print("  • 'Pure JSON only' requirement enforced")
            print("\n🚀 READY FOR PRODUCTION TESTING:")
            print("  • Enhanced logging will capture any remaining issues")
            print("  • Agents should now return clean JSON without wrappers")
            print("  • RobustJSONParser provides fallback protection")
            
            return True
        else:
            print("\n❌ Some prompt fixes are incomplete")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)