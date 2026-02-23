"""Example: LightRAG Multi-Hop Retrieval Pattern

This example demonstrates how to use the multi-hop retrieval pattern to answer
complex questions by decomposing them into sub-questions and using multi-hop
reasoning via LightRAG's agentic search.

Prerequisites:
    pip install git+https://github.com/gyasis/hybridrag.git
    pip install promptchain

Requirements:
    - OpenAI API key set in environment (OPENAI_API_KEY)
    - LightRAG knowledge base prepared with relevant documents
"""

import asyncio
import os
from pathlib import Path

from promptchain.integrations.lightrag.core import LightRAGIntegration, LightRAGConfig
from promptchain.integrations.lightrag.multi_hop import (
    LightRAGMultiHop,
    MultiHopConfig,
)


async def basic_multi_hop_example():
    """Basic multi-hop retrieval without question decomposition."""
    print("=" * 80)
    print("Example 1: Basic Multi-Hop Retrieval (No Decomposition)")
    print("=" * 80)

    # Initialize LightRAG
    config = LightRAGConfig(
        working_dir="./lightrag_data",
        model_name="openai/gpt-4o-mini",
        embedding_model="text-embedding-3-small",
    )
    integration = LightRAGIntegration(config=config)

    # Create multi-hop pattern (decomposition disabled)
    pattern_config = MultiHopConfig(
        max_hops=5,
        decompose_first=False,  # Direct multi-hop search
        emit_events=True,
        timeout_seconds=60.0,
    )

    pattern = LightRAGMultiHop(
        search_interface=integration.search,
        config=pattern_config,
    )

    # Add event handler to track progress
    def event_handler(event_type: str, data: dict):
        print(f"[EVENT] {event_type}: {data.get('hops_executed', 'N/A')} hops")

    pattern.add_event_handler(event_handler)

    # Execute multi-hop retrieval
    question = "What are the key differences between transformers and RNNs in deep learning?"

    result = await pattern.execute(question=question)

    print(f"\nQuestion: {result.original_question}")
    print(f"Hops Executed: {result.hops_executed}")
    print(f"\nAnswer:\n{result.unified_answer}")
    print(f"\nExecution Time: {result.execution_time_ms:.2f}ms")

    if result.unanswered_aspects:
        print(f"\nUnanswered Aspects: {result.unanswered_aspects}")


async def decomposed_multi_hop_example():
    """Multi-hop retrieval with automatic question decomposition."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Hop with Question Decomposition")
    print("=" * 80)

    # Initialize LightRAG
    config = LightRAGConfig(
        working_dir="./lightrag_data",
        model_name="openai/gpt-4o-mini",
    )
    integration = LightRAGIntegration(config=config)

    # Create multi-hop pattern with decomposition
    pattern_config = MultiHopConfig(
        max_hops=7,
        max_sub_questions=5,
        decompose_first=True,  # Decompose question first
        synthesizer_model="openai/gpt-4o-mini",
        emit_events=True,
    )

    pattern = LightRAGMultiHop(
        search_interface=integration.search,
        config=pattern_config,
    )

    # Execute with complex question
    question = (
        "How has the attention mechanism evolved from early seq2seq models "
        "to modern transformers, and what are the computational trade-offs?"
    )

    result = await pattern.execute(question=question)

    print(f"\nOriginal Question: {result.original_question}")
    print(f"\nSub-Questions Generated: {len(result.sub_questions)}")

    for i, sq in enumerate(result.sub_questions, 1):
        print(f"\n{i}. {sq.question_text}")
        print(f"   Rationale: {sq.rationale}")
        print(f"   Dependencies: {sq.dependencies}")

    print(f"\n{'=' * 80}")
    print("Unified Answer:")
    print("=" * 80)
    print(result.unified_answer)

    print(f"\nTotal Hops: {result.hops_executed}")
    print(f"Execution Time: {result.execution_time_ms:.2f}ms")


async def multi_hop_with_blackboard():
    """Multi-hop retrieval with Blackboard state sharing."""
    print("\n" + "=" * 80)
    print("Example 3: Multi-Hop with Blackboard Integration")
    print("=" * 80)

    # Simple Blackboard mock for demonstration
    class SimpleBlackboard:
        def __init__(self):
            self._data = {}

        def write(self, key: str, value, source: str = "unknown"):
            self._data[key] = value
            print(f"[BLACKBOARD] Written: {key} by {source}")

        def read(self, key: str):
            return self._data.get(key)

    # Initialize components
    config = LightRAGConfig(working_dir="./lightrag_data")
    integration = LightRAGIntegration(config=config)
    blackboard = SimpleBlackboard()

    # Create pattern with Blackboard enabled
    pattern_config = MultiHopConfig(
        max_hops=5,
        decompose_first=True,
        use_blackboard=True,  # Enable Blackboard sharing
        emit_events=False,
    )

    pattern = LightRAGMultiHop(
        search_interface=integration.search,
        config=pattern_config,
    )
    pattern.connect_blackboard(blackboard)

    # Execute
    question = "What are the advantages and limitations of transfer learning in NLP?"
    result = await pattern.execute(question=question)

    # Results are automatically shared to Blackboard
    shared_key = f"multi_hop_result_{pattern.config.pattern_id}"
    shared_result = blackboard.read(shared_key)

    print(f"\nQuestion: {question}")
    print(f"Answer: {result.unified_answer[:200]}...")
    print(f"\nResult shared to Blackboard: {shared_result is not None}")


async def custom_event_tracking():
    """Track multi-hop execution with custom event handlers."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Event Tracking")
    print("=" * 80)

    config = LightRAGConfig(working_dir="./lightrag_data")
    integration = LightRAGIntegration(config=config)

    pattern_config = MultiHopConfig(
        max_hops=5,
        decompose_first=True,
        emit_events=True,
    )

    pattern = LightRAGMultiHop(
        search_interface=integration.search,
        config=pattern_config,
    )

    # Track execution stages
    events_log = []

    def track_events(event_type: str, data: dict):
        events_log.append({
            "type": event_type,
            "timestamp": data.get("timestamp"),
            "data": {k: v for k, v in data.items() if k != "timestamp"},
        })

        if event_type == "pattern.multi_hop.started":
            print(f"\n[START] Question: {data.get('question')[:60]}...")
        elif event_type == "pattern.multi_hop.decomposed":
            print(f"[DECOMPOSE] Generated {data.get('num_sub_questions')} sub-questions")
        elif event_type == "pattern.multi_hop.hop_completed":
            print(f"[HOP] Executed {data.get('hops_executed')} hops")
        elif event_type == "pattern.multi_hop.synthesizing":
            print(f"[SYNTHESIZE] Combining {data.get('num_sub_answers')} answers")
        elif event_type == "pattern.multi_hop.completed":
            print(f"[COMPLETE] Finished in {data.get('execution_time_ms'):.2f}ms")

    pattern.add_event_handler(track_events)

    # Execute
    question = "Compare the architectural differences between BERT and GPT models."
    result = await pattern.execute(question=question)

    print(f"\nTotal Events Captured: {len(events_log)}")
    print(f"Success: {result.success}")


async def main():
    """Run all examples."""
    # Check if LightRAG data exists
    if not Path("./lightrag_data").exists():
        print("WARNING: ./lightrag_data directory not found!")
        print("Please create a LightRAG knowledge base first.")
        print("Example:")
        print("  from promptchain.integrations.lightrag.core import LightRAGIntegration")
        print("  integration = LightRAGIntegration()")
        print("  await integration.core.insert_text('Your text here...')")
        print("\nSkipping examples for now.\n")
        return

    # Run examples
    await basic_multi_hop_example()
    await decomposed_multi_hop_example()
    await multi_hop_with_blackboard()
    await custom_event_tracking()


if __name__ == "__main__":
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    asyncio.run(main())
