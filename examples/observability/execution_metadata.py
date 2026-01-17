#!/usr/bin/env python3
"""
Execution Metadata Example

This example demonstrates how to use return_metadata=True to get rich
execution details from AgentChain and AgenticStepProcessor.
"""

import asyncio
import json
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


def demo_agent_metadata():
    """Demonstrate AgentChain execution metadata."""
    print("=" * 70)
    print("AgentChain Execution Metadata Demo")
    print("=" * 70)

    # Create simple agents
    analyzer_agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Analyze this request: {input}"],
        verbose=False
    )

    writer_agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Write a detailed response: {input}"],
        verbose=False
    )

    # Create agent chain with router
    agent_chain = AgentChain(
        agents={
            "analyzer": analyzer_agent,
            "writer": writer_agent
        },
        agent_descriptions={
            "analyzer": "Analyzes and breaks down complex questions",
            "writer": "Writes detailed, well-structured responses"
        },
        execution_mode="router",
        router={
            "models": ["openai/gpt-4o-mini"],
            "instructions": [None, "{input}"]
        },
        verbose=False
    )

    # Get metadata
    print("\n1. Basic Metadata")
    print("-" * 70)

    result = agent_chain.process_input(
        "Explain quantum entanglement",
        return_metadata=True
    )

    print(f"Response (truncated): {result.response[:150]}...")
    print(f"\nExecution Metadata:")
    print(f"  Agent used: {result.agent_name}")
    print(f"  Execution time: {result.execution_time_ms:.1f}ms")
    print(f"  Router steps: {result.router_steps}")
    print(f"  Fallback used: {result.fallback_used}")

    # Router decision
    print("\n2. Router Decision Details")
    print("-" * 70)

    if result.router_decision:
        print(f"  Chosen agent: {result.router_decision.get('chosen_agent')}")
        print(f"  Refined query: {result.router_decision.get('refined_query', 'N/A')[:100]}")
        if 'reasoning' in result.router_decision:
            print(f"  Reasoning: {result.router_decision['reasoning'][:100]}")

    # Token usage
    print("\n3. Token Usage")
    print("-" * 70)

    if result.total_tokens:
        print(f"  Total tokens: {result.total_tokens}")
        print(f"  Prompt tokens: {result.prompt_tokens}")
        print(f"  Completion tokens: {result.completion_tokens}")
    else:
        print("  Token information not available")

    # Errors and warnings
    print("\n4. Errors and Warnings")
    print("-" * 70)

    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"    - {error}")
    else:
        print("  No errors")

    if result.warnings:
        print(f"  Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"    - {warning}")
    else:
        print("  No warnings")

    # Full metadata export
    print("\n5. Full Metadata Export")
    print("-" * 70)

    full_metadata = result.to_dict()
    print(f"  Full metadata keys: {list(full_metadata.keys())}")

    # Summary export
    print("\n6. Summary Metadata")
    print("-" * 70)

    summary = result.to_summary_dict()
    print(f"  Summary: {json.dumps(summary, indent=2)}")


async def demo_agentic_step_metadata():
    """Demonstrate AgenticStepProcessor metadata."""
    print("\n\n" + "=" * 70)
    print("AgenticStepProcessor Metadata Demo")
    print("=" * 70)

    # Create a simple tool
    def search_tool(query: str) -> str:
        """Simulated search tool."""
        return f"Search results for '{query}': Found 10 relevant articles about the topic."

    # Create agentic step processor
    agentic_step = AgenticStepProcessor(
        objective="Research the topic and provide key findings",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini",
        history_mode="minimal"
    )

    # Create chain with agentic step
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            agentic_step,  # Agentic reasoning step
            "Final summary: {input}"
        ],
        verbose=False
    )

    # Register the tool
    chain.register_tool_function(search_tool)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "search_tool",
            "description": "Search for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }])

    # Get agentic step metadata
    print("\n1. Basic Agentic Metadata")
    print("-" * 70)

    # Note: We need to access the agentic step directly for metadata
    result = await agentic_step.run_async(
        "Research renewable energy trends",
        return_metadata=True
    )

    print(f"Final answer (truncated): {result.final_answer[:150]}...")
    print(f"\nAgentic Execution Metadata:")
    print(f"  Total steps: {result.total_steps}")
    print(f"  Max steps reached: {result.max_steps_reached}")
    print(f"  Objective achieved: {result.objective_achieved}")
    print(f"  Total tools called: {result.total_tools_called}")
    print(f"  Total tokens used: {result.total_tokens_used}")
    print(f"  Total execution time: {result.total_execution_time_ms:.1f}ms")

    # Configuration info
    print("\n2. Configuration Used")
    print("-" * 70)

    print(f"  History mode: {result.history_mode}")
    print(f"  Max internal steps: {result.max_internal_steps}")
    print(f"  Model name: {result.model_name}")

    # Step-by-step details
    print("\n3. Step-by-Step Breakdown")
    print("-" * 70)

    for step in result.steps:
        print(f"\n  Step {step.step_number}:")
        print(f"    Tool calls: {len(step.tool_calls)}")
        print(f"    Tokens used: {step.tokens_used}")
        print(f"    Execution time: {step.execution_time_ms:.1f}ms")
        print(f"    Clarification attempts: {step.clarification_attempts}")

        if step.error:
            print(f"    Error: {step.error}")

        # Tool call details
        for tool_call in step.tool_calls:
            print(f"    - Tool: {tool_call.get('name', 'unknown')}")
            print(f"      Args: {tool_call.get('args', {})}")
            result_preview = str(tool_call.get('result', ''))[:50]
            print(f"      Result: {result_preview}...")
            print(f"      Time: {tool_call.get('time_ms', 0):.1f}ms")

    # Analysis and insights
    print("\n4. Performance Analysis")
    print("-" * 70)

    if result.total_steps > 0:
        avg_time_per_step = result.total_execution_time_ms / result.total_steps
        avg_tokens_per_step = result.total_tokens_used / result.total_steps
        tools_per_step = result.total_tools_called / result.total_steps

        print(f"  Average time per step: {avg_time_per_step:.1f}ms")
        print(f"  Average tokens per step: {avg_tokens_per_step:.1f}")
        print(f"  Average tools per step: {tools_per_step:.1f}")

    # Check for issues
    if result.max_steps_reached and not result.objective_achieved:
        print(f"\n  ⚠️  Warning: Max steps reached without achieving objective")
        print(f"     Consider increasing max_internal_steps")

    if result.errors:
        print(f"\n  ❌ Errors encountered ({len(result.errors)}):")
        for error in result.errors:
            print(f"     - {error}")

    # Export metadata
    print("\n5. Metadata Export")
    print("-" * 70)

    full_data = result.to_dict()
    summary_data = result.to_summary_dict()

    print(f"  Full metadata fields: {len(full_data)} fields")
    print(f"  Summary metadata: {json.dumps(summary_data, indent=2)}")


def demo_comparison():
    """Compare execution with and without metadata."""
    print("\n\n" + "=" * 70)
    print("Metadata vs No Metadata Comparison")
    print("=" * 70)

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Answer briefly: {input}"],
        verbose=False
    )

    analyzer_agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Analyze: {input}"],
        verbose=False
    )

    agent_chain = AgentChain(
        agents={"analyzer": analyzer_agent},
        execution_mode="single_agent",
        verbose=False
    )

    # Without metadata (default)
    print("\n1. Without Metadata (return_metadata=False)")
    print("-" * 70)

    result_no_meta = agent_chain.process_input("What is AI?")
    print(f"Type: {type(result_no_meta)}")
    print(f"Value: {result_no_meta[:100]}...")

    # With metadata
    print("\n2. With Metadata (return_metadata=True)")
    print("-" * 70)

    result_with_meta = agent_chain.process_input(
        "What is AI?",
        return_metadata=True
    )
    print(f"Type: {type(result_with_meta)}")
    print(f"Response: {result_with_meta.response[:100]}...")
    print(f"Has metadata: Yes")
    print(f"  - agent_name: {result_with_meta.agent_name}")
    print(f"  - execution_time_ms: {result_with_meta.execution_time_ms}")
    print(f"  - and {len(result_with_meta.to_dict())} more fields...")


async def main():
    """Run all metadata examples."""
    # Agent metadata
    demo_agent_metadata()

    # Agentic step metadata
    await demo_agentic_step_metadata()

    # Comparison
    demo_comparison()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
