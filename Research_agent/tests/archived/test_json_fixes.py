#!/usr/bin/env python3

import json
from src.research_agent.agents.react_analyzer import ReActAnalysisAgent
from src.research_agent.agents.search_strategist import SearchStrategistAgent
from src.research_agent.agents.query_generator import QueryGenerationAgent

def test_react_analyzer_json_extraction():
    """Test ReAct analyzer JSON extraction"""
    agent = ReActAnalysisAgent({'model': 'openai/gpt-4o-mini'})

    # Test response with markdown code blocks
    markdown_response = """Here's my analysis:

```json
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
```

The analysis shows good progress."""

    print('Testing ReAct Analyzer JSON extraction:')
    try:
        validated = agent._validate_analysis_response(markdown_response)
        result = json.loads(validated)
        print('✓ SUCCESS: ReAct Analyzer extracted and validated JSON')
        print(f'  Completion score: {result["analysis_summary"]["completion_score"]}')
    except Exception as e:
        print(f'✗ FAILED: {str(e)}')

def test_search_strategist_json_extraction():
    """Test Search Strategist JSON extraction"""
    agent = SearchStrategistAgent({'model': 'openai/gpt-4o-mini'})

    # Test response with markdown code blocks
    markdown_response = """Based on the analysis, here's the optimal search strategy:

```json
{
    "search_strategy": {
        "primary_keywords": ["machine learning", "deep learning"],
        "secondary_keywords": ["neural networks", "AI"],
        "boolean_queries": [],
        "filters": {
            "publication_years": [2020, 2024],
            "paper_types": ["research"],
            "languages": ["en"]
        }
    },
    "database_allocation": {
        "sci_hub": {
            "priority": 1.0,
            "max_papers": 40,
            "search_terms": ["machine learning"],
            "rationale": "Primary source"
        }
    },
    "search_optimization": {
        "iteration_focus": "Core concepts",
        "gap_targeting": [],
        "expansion_areas": ["applications"],
        "exclusion_criteria": []
    }
}
```

This strategy optimizes for comprehensive coverage."""

    print('\nTesting Search Strategist JSON extraction:')
    try:
        validated = agent._validate_strategy_response(markdown_response)
        result = json.loads(validated)
        print('✓ SUCCESS: Search Strategist extracted and validated JSON')
        print(f'  Primary keywords: {result["search_strategy"]["primary_keywords"]}')
        print(f'  Sci-hub max papers: {result["database_allocation"]["sci_hub"]["max_papers"]}')
    except Exception as e:
        print(f'✗ FAILED: {str(e)}')

def test_query_generator_json_extraction():
    """Test Query Generator JSON extraction"""
    agent = QueryGenerationAgent({'model': 'openai/gpt-4o-mini'})

    # Test response with markdown code blocks
    markdown_response = """I'll generate comprehensive research queries:

```json
{
    "primary_queries": [
        {
            "text": "What are the main machine learning techniques?",
            "priority": 1.0,
            "category": "techniques"
        }
    ],
    "secondary_queries": [
        {
            "text": "What datasets are used for training?",
            "priority": 0.7,
            "category": "datasets"
        }
    ],
    "exploratory_queries": [
        {
            "text": "What are future trends in AI?",
            "priority": 0.5,
            "category": "trends"
        }
    ]
}
```

These queries provide comprehensive coverage."""

    print('\nTesting Query Generator JSON extraction:')
    try:
        validated = agent._validate_query_response(markdown_response)
        result = json.loads(validated)
        print('✓ SUCCESS: Query Generator extracted and validated JSON')
        print(f'  Primary queries count: {len(result["primary_queries"])}')
        print(f'  Total queries: {len(result["primary_queries"]) + len(result["secondary_queries"]) + len(result["exploratory_queries"])}')
    except Exception as e:
        print(f'✗ FAILED: {str(e)}')

def test_mixed_content_extraction():
    """Test mixed content with JSON buried in text"""
    agent = ReActAnalysisAgent({'model': 'openai/gpt-4o-mini'})

    mixed_response = """I need to analyze this research progress.

Looking at the current state, I can see several important factors.

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

The analysis indicates good progress has been made."""

    print('\nTesting mixed content extraction:')
    try:
        validated = agent._validate_analysis_response(mixed_response)
        result = json.loads(validated)
        print('✓ SUCCESS: Extracted JSON from mixed content')
        print(f'  Should continue: {result["iteration_recommendation"]["should_continue"]}')
        print(f'  Confidence: {result["iteration_recommendation"]["confidence"]}')
    except Exception as e:
        print(f'✗ FAILED: {str(e)}')

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING JSON EXTRACTION FIXES FOR ALL AGENTS")
    print("=" * 60)
    
    test_react_analyzer_json_extraction()
    test_search_strategist_json_extraction()
    test_query_generator_json_extraction()
    test_mixed_content_extraction()
    
    print("\n" + "=" * 60)
    print("JSON EXTRACTION TESTING COMPLETE")
    print("=" * 60)