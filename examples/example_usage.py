#!/usr/bin/env python3
"""
Example Usage of Agentic Chat Team
===================================

This script demonstrates various ways to use the 5-agent collaborative system.
"""

import asyncio
from pathlib import Path

# Ensure the PromptChain package is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_team_chat import (
    create_research_agent,
    create_analysis_agent,
    create_terminal_agent,
    create_documentation_agent,
    create_synthesis_agent,
    setup_gemini_mcp_config
)


async def example_1_individual_agents():
    """Example 1: Using individual agents directly"""
    print("=" * 80)
    print("EXAMPLE 1: Using Individual Agents Directly")
    print("=" * 80 + "\n")

    # Setup
    mcp_config = setup_gemini_mcp_config()

    # Create a research agent
    research_agent = create_research_agent(mcp_config)

    # Use the research agent directly
    print("🔍 Testing Research Agent with Gemini MCP access...")
    query = "What are the latest trends in AI agents in 2025?"
    result = await research_agent.process_prompt_async(query)
    print(f"\nResult: {result}\n")


async def example_2_terminal_agent():
    """Example 2: Using the Terminal Agent for system operations"""
    print("=" * 80)
    print("EXAMPLE 2: Using Terminal Agent")
    print("=" * 80 + "\n")

    # Create terminal agent
    terminal_agent = create_terminal_agent()

    # Execute terminal commands
    print("💻 Testing Terminal Agent for system operations...")
    queries = [
        "List all Python files in the current directory",
        "Check the current working directory",
        "Show disk usage summary"
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        result = await terminal_agent.process_prompt_async(query)
        print(f"Result: {result}\n")
        print("-" * 80)


async def example_3_analysis_agent():
    """Example 3: Using the Analysis Agent"""
    print("=" * 80)
    print("EXAMPLE 3: Using Analysis Agent")
    print("=" * 80 + "\n")

    # Create analysis agent
    analysis_agent = create_analysis_agent()

    # Analyze data
    print("📊 Testing Analysis Agent for pattern recognition...")
    query = """
    Analyze the following software development team metrics and provide insights:
    - Sprint velocity: Week 1: 25, Week 2: 30, Week 3: 28, Week 4: 35
    - Bug count: Week 1: 12, Week 2: 10, Week 3: 8, Week 4: 6
    - Code review time (hours): Week 1: 8, Week 2: 7, Week 3: 6, Week 4: 5

    What patterns do you see and what recommendations do you have?
    """

    result = await analysis_agent.process_prompt_async(query)
    print(f"\nResult: {result}\n")


async def example_4_documentation_agent():
    """Example 4: Using the Documentation Agent"""
    print("=" * 80)
    print("EXAMPLE 4: Using Documentation Agent")
    print("=" * 80 + "\n")

    # Create documentation agent
    doc_agent = create_documentation_agent()

    # Create documentation
    print("📝 Testing Documentation Agent for technical writing...")
    query = """
    Create a simple tutorial explaining how to use Python's asyncio library
    for beginners. Include a basic example.
    """

    result = await doc_agent.process_prompt_async(query)
    print(f"\nResult: {result}\n")


async def example_5_synthesis_agent():
    """Example 5: Using the Synthesis Agent"""
    print("=" * 80)
    print("EXAMPLE 5: Using Synthesis Agent")
    print("=" * 80 + "\n")

    # Create synthesis agent
    synthesis_agent = create_synthesis_agent()

    # Synthesize insights
    print("🎯 Testing Synthesis Agent for strategic recommendations...")
    query = """
    Based on these findings from different team members:

    Research: "AI agents are trending towards more autonomous decision-making"
    Analysis: "Our team velocity is improving by 10% weekly"
    Technical: "Current system can handle 100 req/sec"

    Synthesize these insights and provide strategic recommendations
    for our product roadmap.
    """

    result = await synthesis_agent.process_prompt_async(query)
    print(f"\nResult: {result}\n")


async def example_6_combined_workflow():
    """Example 6: Combining multiple agents in a workflow"""
    print("=" * 80)
    print("EXAMPLE 6: Combined Multi-Agent Workflow")
    print("=" * 80 + "\n")

    # Setup
    mcp_config = setup_gemini_mcp_config()

    # Create agents
    research_agent = create_research_agent(mcp_config)
    analysis_agent = create_analysis_agent()
    synthesis_agent = create_synthesis_agent()

    # Step 1: Research
    print("🔍 Step 1: Research Phase...")
    research_query = "What are the key technologies in modern AI systems?"
    research_result = await research_agent.process_prompt_async(research_query)
    print(f"Research findings: {research_result[:200]}...\n")

    # Step 2: Analysis
    print("📊 Step 2: Analysis Phase...")
    analysis_query = f"Analyze these research findings and identify key patterns:\n\n{research_result}"
    analysis_result = await analysis_agent.process_prompt_async(analysis_query)
    print(f"Analysis insights: {analysis_result[:200]}...\n")

    # Step 3: Synthesis
    print("🎯 Step 3: Synthesis Phase...")
    synthesis_query = f"""
    Synthesize these research findings and analysis insights into actionable recommendations:

    Research: {research_result[:300]}
    Analysis: {analysis_result[:300]}
    """
    synthesis_result = await synthesis_agent.process_prompt_async(synthesis_query)
    print(f"Final recommendations: {synthesis_result}\n")


async def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("AGENTIC CHAT TEAM - USAGE EXAMPLES")
    print("=" * 80 + "\n")

    # Run examples (comment out any you don't want to run)
    # await example_1_individual_agents()
    await example_2_terminal_agent()  # Safe to run - just lists files
    # await example_3_analysis_agent()
    # await example_4_documentation_agent()
    # await example_5_synthesis_agent()
    # await example_6_combined_workflow()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nTo run the full interactive chat system, execute:")
    print("  python agentic_team_chat.py")
    print("\n")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
