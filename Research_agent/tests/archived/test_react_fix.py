#!/usr/bin/env python3

import json
from src.research_agent.agents.react_analyzer import ReActAnalysisAgent

def test_json_extraction():
    agent = ReActAnalysisAgent({'model': 'openai/gpt-4o-mini'})

    # Test response with markdown code blocks
    markdown_response = """```json
{
    "analysis_summary": {
        "current_state": "test",
        "completion_score": 0.85
    },
    "gaps_identified": [],
    "coverage_analysis": {},
    "new_queries": [],
    "iteration_recommendation": {
        "should_continue": true,
        "confidence": 0.8
    }
}
```"""

    print('Testing markdown JSON extraction:')
    try:
        validated = agent._validate_analysis_response(markdown_response)
        result = json.loads(validated)
        print('SUCCESS: Extracted and validated JSON')
        print('Keys:', list(result.keys()))
        print('Completion score:', result['analysis_summary']['completion_score'])
    except Exception as e:
        print('FAILED:', str(e))

    # Test response with mixed content
    mixed_response = """Here is the analysis:

{
    "analysis_summary": {
        "current_state": "mixed content test",
        "completion_score": 0.75
    },
    "gaps_identified": [],
    "coverage_analysis": {},
    "new_queries": [],
    "iteration_recommendation": {
        "should_continue": false,
        "confidence": 0.9
    }
}

The analysis is complete."""

    print('\nTesting mixed content extraction:')
    try:
        validated = agent._validate_analysis_response(mixed_response)
        result = json.loads(validated)
        print('SUCCESS: Extracted JSON from mixed content')
        print('Should continue:', result['iteration_recommendation']['should_continue'])
    except Exception as e:
        print('FAILED:', str(e))

if __name__ == "__main__":
    test_json_extraction()