#!/usr/bin/env python3

import json
from datetime import datetime
from src.research_agent.core.session import IterationSummary

def test_iteration_summary_serialization():
    """Test IterationSummary JSON serialization and deserialization"""
    
    print("Testing IterationSummary JSON serialization:")
    
    # Create an IterationSummary with test data
    original_summary = IterationSummary(
        iteration=1,
        queries_processed=5,
        papers_found=12,
        gaps_identified=["gap1", "gap2", "methodology gap"],
        new_queries_generated=3,
        completion_score=0.75,
        timestamp=datetime.now()
    )
    
    print(f"Original summary:")
    print(f"  Iteration: {original_summary.iteration}")
    print(f"  Queries processed: {original_summary.queries_processed}")
    print(f"  Papers found: {original_summary.papers_found}")
    print(f"  Gaps: {original_summary.gaps_identified}")
    print(f"  New queries: {original_summary.new_queries_generated}")
    print(f"  Completion score: {original_summary.completion_score}")
    print(f"  Timestamp: {original_summary.timestamp}")
    
    # Test serialization to dict
    try:
        summary_dict = original_summary.to_dict()
        print(f"\n✓ SUCCESS: Serialized to dict")
        print(f"  Timestamp as string: {summary_dict['timestamp']}")
        print(f"  Dict keys: {list(summary_dict.keys())}")
    except Exception as e:
        print(f"\n✗ FAILED: Dict serialization failed - {str(e)}")
        return False
    
    # Test JSON serialization
    try:
        json_string = json.dumps(summary_dict)
        print(f"\n✓ SUCCESS: Serialized to JSON string")
        print(f"  JSON length: {len(json_string)} characters")
    except Exception as e:
        print(f"\n✗ FAILED: JSON serialization failed - {str(e)}")
        return False
    
    # Test deserialization from dict
    try:
        restored_summary = IterationSummary.from_dict(summary_dict)
        print(f"\n✓ SUCCESS: Deserialized from dict")
        print(f"  Restored iteration: {restored_summary.iteration}")
        print(f"  Restored timestamp: {restored_summary.timestamp}")
        print(f"  Timestamp type: {type(restored_summary.timestamp)}")
    except Exception as e:
        print(f"\n✗ FAILED: Dict deserialization failed - {str(e)}")
        return False
    
    # Test round-trip JSON serialization/deserialization
    try:
        # Full round trip: object -> dict -> JSON -> dict -> object
        summary_dict = original_summary.to_dict()
        json_string = json.dumps(summary_dict)
        parsed_dict = json.loads(json_string)
        final_summary = IterationSummary.from_dict(parsed_dict)
        
        print(f"\n✓ SUCCESS: Complete round-trip serialization")
        print(f"  Original completion score: {original_summary.completion_score}")
        print(f"  Final completion score: {final_summary.completion_score}")
        print(f"  Scores match: {original_summary.completion_score == final_summary.completion_score}")
        print(f"  Gaps match: {original_summary.gaps_identified == final_summary.gaps_identified}")
        print(f"  Timestamps match: {original_summary.timestamp == final_summary.timestamp}")
    except Exception as e:
        print(f"\n✗ FAILED: Round-trip serialization failed - {str(e)}")
        return False
    
    return True

def test_iteration_summary_list_serialization():
    """Test serialization of lists containing IterationSummary objects"""
    
    print("\n" + "="*50)
    print("Testing IterationSummary list serialization:")
    
    # Create multiple summaries
    summaries = [
        IterationSummary(
            iteration=1,
            queries_processed=5,
            papers_found=10,
            gaps_identified=["gap1", "gap2"],
            new_queries_generated=3,
            completion_score=0.6
        ),
        IterationSummary(
            iteration=2,
            queries_processed=8,
            papers_found=18,
            gaps_identified=["gap3"],
            new_queries_generated=2,
            completion_score=0.8
        ),
        IterationSummary(
            iteration=3,
            queries_processed=10,
            papers_found=25,
            gaps_identified=[],
            new_queries_generated=0,
            completion_score=0.95
        )
    ]
    
    print(f"Created {len(summaries)} IterationSummary objects")
    
    # Test list serialization
    try:
        summaries_dicts = [s.to_dict() for s in summaries]
        json_string = json.dumps(summaries_dicts)
        print(f"✓ SUCCESS: Serialized list to JSON")
        print(f"  JSON length: {len(json_string)} characters")
    except Exception as e:
        print(f"✗ FAILED: List serialization failed - {str(e)}")
        return False
    
    # Test list deserialization
    try:
        parsed_dicts = json.loads(json_string)
        restored_summaries = [IterationSummary.from_dict(d) for d in parsed_dicts]
        
        print(f"✓ SUCCESS: Deserialized list from JSON")
        print(f"  Restored {len(restored_summaries)} IterationSummary objects")
        
        # Verify data integrity
        for i, (orig, restored) in enumerate(zip(summaries, restored_summaries)):
            print(f"  Summary {i+1}: iteration {orig.iteration} -> {restored.iteration} ({'✓' if orig.iteration == restored.iteration else '✗'})")
            print(f"             completion {orig.completion_score} -> {restored.completion_score} ({'✓' if orig.completion_score == restored.completion_score else '✗'})")
            
    except Exception as e:
        print(f"✗ FAILED: List deserialization failed - {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ITERATIONSUMMARY JSON SERIALIZATION")
    print("=" * 60)
    
    success1 = test_iteration_summary_serialization()
    success2 = test_iteration_summary_list_serialization()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("ALL TESTS PASSED ✓")
        print("IterationSummary JSON serialization is now working correctly!")
    else:
        print("SOME TESTS FAILED ✗")
        print("Further investigation needed.")
    print("=" * 60)