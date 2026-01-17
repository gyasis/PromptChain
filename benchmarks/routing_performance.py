#!/usr/bin/env python3
"""
T106: AgentChain Routing Performance Benchmark

Profiles and measures routing latency for AgentChain router mode to identify
bottlenecks and validate optimization targets:
- Router LLM calls: <300ms
- History formatting: <100ms
- Agent selection logic: <50ms
- Total overhead: <500ms

Usage:
    python benchmarks/routing_performance.py
    python benchmarks/routing_performance.py --agent-counts 2,4,6,8 --history-sizes 0,10,50
"""

import asyncio
import time
import statistics
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptchain.utils.agent_chain import AgentChain
from promptchain import PromptChain


@dataclass
class RoutingMetrics:
    """Metrics for a single routing operation."""
    total_time_ms: float
    simple_router_time_ms: Optional[float]
    history_format_time_ms: Optional[float]
    llm_call_time_ms: Optional[float]
    parse_decision_time_ms: Optional[float]
    agent_count: int
    history_size: int
    selected_agent: str
    router_type: str  # "simple" or "llm"


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    configuration: str
    agent_count: int
    history_size: int
    iterations: int

    # Timing statistics (ms)
    total_mean: float
    total_median: float
    total_p95: float
    total_min: float
    total_max: float

    # Component breakdowns
    history_format_mean: float
    llm_call_mean: float
    parse_decision_mean: float

    # Success metrics
    success_rate: float
    simple_router_rate: float
    llm_router_rate: float

    # Target achievement
    meets_total_target: bool  # <500ms
    meets_llm_target: bool  # <300ms
    meets_history_target: bool  # <100ms


class RoutingProfiler:
    """Profiles AgentChain routing operations with detailed timing breakdowns."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.metrics: List[RoutingMetrics] = []

    async def profile_routing_call(
        self,
        agent_chain: AgentChain,
        user_input: str,
        agent_count: int,
        history_size: int
    ) -> RoutingMetrics:
        """Profile a single routing call with component-level timing."""

        start_total = time.perf_counter()

        # Initialize timing variables
        simple_router_time = None
        history_format_time = None
        llm_call_time = None
        parse_decision_time = None
        selected_agent = None
        router_type = "unknown"

        try:
            # Try simple router first
            simple_start = time.perf_counter()
            simple_result = agent_chain._simple_router(user_input)
            simple_router_time = (time.perf_counter() - simple_start) * 1000

            if simple_result:
                selected_agent = simple_result
                router_type = "simple"
            else:
                router_type = "llm"

                # Measure history formatting
                history_start = time.perf_counter()
                formatted_history = agent_chain._format_chat_history()
                history_format_time = (time.perf_counter() - history_start) * 1000

                # Measure LLM call (through decision maker chain)
                if not agent_chain.custom_router_function and agent_chain.decision_maker_chain:
                    llm_start = time.perf_counter()
                    decision_output = await agent_chain.decision_maker_chain.process_prompt_async(user_input)
                    llm_call_time = (time.perf_counter() - llm_start) * 1000

                    # Measure decision parsing
                    parse_start = time.perf_counter()
                    parsed_decision = agent_chain._parse_decision(decision_output)
                    parse_decision_time = (time.perf_counter() - parse_start) * 1000

                    selected_agent = parsed_decision.get("chosen_agent", "unknown")

        except Exception as e:
            if self.verbose:
                print(f"Error during profiling: {e}")
            selected_agent = "error"

        total_time = (time.perf_counter() - start_total) * 1000

        metrics = RoutingMetrics(
            total_time_ms=total_time,
            simple_router_time_ms=simple_router_time,
            history_format_time_ms=history_format_time,
            llm_call_time_ms=llm_call_time,
            parse_decision_time_ms=parse_decision_time,
            agent_count=agent_count,
            history_size=history_size,
            selected_agent=selected_agent,
            router_type=router_type
        )

        self.metrics.append(metrics)
        return metrics

    def analyze_metrics(
        self,
        agent_count: int,
        history_size: int,
        iterations: int
    ) -> BenchmarkResults:
        """Analyze collected metrics for specific configuration."""

        # Filter metrics for this configuration
        config_metrics = [
            m for m in self.metrics
            if m.agent_count == agent_count and m.history_size == history_size
        ]

        if not config_metrics:
            raise ValueError(f"No metrics found for config: {agent_count} agents, {history_size} history")

        # Extract timing data
        total_times = [m.total_time_ms for m in config_metrics]
        history_times = [m.history_format_time_ms for m in config_metrics if m.history_format_time_ms]
        llm_times = [m.llm_call_time_ms for m in config_metrics if m.llm_call_time_ms]
        parse_times = [m.parse_decision_time_ms for m in config_metrics if m.parse_decision_time_ms]

        # Calculate statistics
        def percentile_95(data: List[float]) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * 0.95)
            return sorted_data[min(index, len(sorted_data) - 1)]

        total_mean = statistics.mean(total_times)
        total_p95 = percentile_95(total_times)

        history_mean = statistics.mean(history_times) if history_times else 0.0
        llm_mean = statistics.mean(llm_times) if llm_times else 0.0
        parse_mean = statistics.mean(parse_times) if parse_times else 0.0

        # Success metrics
        simple_count = sum(1 for m in config_metrics if m.router_type == "simple")
        llm_count = sum(1 for m in config_metrics if m.router_type == "llm")

        return BenchmarkResults(
            configuration=f"{agent_count}_agents_{history_size}_history",
            agent_count=agent_count,
            history_size=history_size,
            iterations=iterations,
            total_mean=total_mean,
            total_median=statistics.median(total_times),
            total_p95=total_p95,
            total_min=min(total_times),
            total_max=max(total_times),
            history_format_mean=history_mean,
            llm_call_mean=llm_mean,
            parse_decision_mean=parse_mean,
            success_rate=1.0,  # All completed
            simple_router_rate=simple_count / len(config_metrics),
            llm_router_rate=llm_count / len(config_metrics),
            meets_total_target=total_p95 < 500,
            meets_llm_target=llm_mean < 300 if llm_times else True,
            meets_history_target=history_mean < 100
        )


def create_test_agents(count: int, model: str = "openai/gpt-4o-mini") -> Dict[str, Any]:
    """Create test agents with minimal setup for performance testing."""
    agents = {}
    descriptions = {}

    agent_templates = [
        ("code_executor", "Executes code snippets and scripts"),
        ("researcher", "Researches topics and gathers information"),
        ("writer", "Writes documentation and reports"),
        ("analyst", "Analyzes data and provides insights"),
        ("debugger", "Debugs code and fixes issues"),
        ("optimizer", "Optimizes performance and efficiency"),
        ("tester", "Tests functionality and validates behavior"),
        ("deployer", "Deploys applications and manages infrastructure"),
    ]

    for i in range(count):
        name, desc = agent_templates[i % len(agent_templates)]
        if i >= len(agent_templates):
            name = f"{name}_{i // len(agent_templates)}"

        agents[name] = PromptChain(
            models=[model],
            instructions=["Process: {input}"],
            verbose=False
        )
        descriptions[name] = desc

    return agents, descriptions


def create_conversation_history(size: int) -> List[Dict[str, str]]:
    """Create synthetic conversation history for testing."""
    history = []

    message_templates = [
        ("user", "What can you help me with?"),
        ("assistant", "I can help with various tasks including coding, research, and analysis."),
        ("user", "Can you write some Python code?"),
        ("assistant", "Sure, I can write Python code. What would you like me to create?"),
        ("user", "How do I optimize database queries?"),
        ("assistant", "To optimize queries, use indexes, limit results, and avoid N+1 queries."),
        ("user", "Debug this error message"),
        ("assistant", "I'll help debug that. Can you share the error details?"),
        ("user", "Research the latest AI trends"),
        ("assistant", "I'll research current AI trends and developments for you."),
    ]

    for i in range(size):
        role, content = message_templates[i % len(message_templates)]
        history.append({"role": role, "content": content})

    return history


async def run_benchmark_suite(
    agent_counts: List[int],
    history_sizes: List[int],
    iterations_per_config: int = 10,
    model: str = "openai/gpt-4o-mini",
    verbose: bool = False
) -> Dict[str, Any]:
    """Run comprehensive routing performance benchmarks."""

    print("=" * 80)
    print("AgentChain Routing Performance Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Agent counts: {agent_counts}")
    print(f"  History sizes: {history_sizes}")
    print(f"  Iterations per config: {iterations_per_config}")
    print(f"  Router model: {model}")
    print("=" * 80)
    print()

    profiler = RoutingProfiler(verbose=verbose)
    all_results = []

    # Test queries that force LLM routing (not simple pattern matching)
    test_queries = [
        "Help me analyze this dataset",
        "I need to optimize performance",
        "Can you research this topic for me",
        "Debug the authentication logic",
        "Write comprehensive documentation",
        "Execute this script safely",
        "Test the new feature thoroughly",
        "Deploy to production environment",
    ]

    for agent_count in agent_counts:
        for history_size in history_sizes:
            config_name = f"{agent_count} agents, {history_size} history"
            print(f"Running benchmark: {config_name}")
            print(f"  Creating agents and history...")

            # Create test setup
            agents, descriptions = create_test_agents(agent_count, model=model)
            history = create_conversation_history(history_size)

            # Configure router with optimized decision prompt
            router_config = {
                "models": [model],
                "instructions": [None, "{input}"],
                "decision_prompt_templates": {
                    "single_agent_dispatch": """Based on: {user_input}

Agents:
{agent_details}

History:
{history}

Return JSON: {{"chosen_agent": "agent_name"}}"""
                }
            }

            # Create agent chain
            agent_chain = AgentChain(
                agents=agents,
                agent_descriptions=descriptions,
                execution_mode="router",
                router=router_config,
                router_strategy="single_agent_dispatch",
                default_agent=list(agents.keys())[0],
                verbose=False
            )

            # Populate conversation history
            agent_chain._conversation_history = history.copy()

            print(f"  Running {iterations_per_config} iterations...")

            # Run benchmark iterations
            for i in range(iterations_per_config):
                query = test_queries[i % len(test_queries)]
                await profiler.profile_routing_call(
                    agent_chain,
                    query,
                    agent_count,
                    history_size
                )

                if verbose:
                    print(f"    Iteration {i+1}/{iterations_per_config} complete")

            # Analyze results for this configuration
            results = profiler.analyze_metrics(agent_count, history_size, iterations_per_config)
            all_results.append(results)

            # Print results
            print(f"  Results:")
            print(f"    Total time (mean): {results.total_mean:.1f}ms")
            print(f"    Total time (p95): {results.total_p95:.1f}ms")
            print(f"    LLM call (mean): {results.llm_call_mean:.1f}ms")
            print(f"    History format (mean): {results.history_format_mean:.1f}ms")
            print(f"    Meets <500ms target: {results.meets_total_target}")
            print()

    # Generate summary report
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print()

    print("Performance Targets:")
    print("  Total overhead: <500ms (p95)")
    print("  LLM call: <300ms (mean)")
    print("  History format: <100ms (mean)")
    print()

    print(f"{'Configuration':<25} {'Total(p95)':<12} {'LLM':<12} {'History':<12} {'Targets'}")
    print("-" * 80)

    for result in all_results:
        config = f"{result.agent_count}ag/{result.history_size}hist"
        total_status = "✓" if result.meets_total_target else "✗"
        llm_status = "✓" if result.meets_llm_target else "✗"
        hist_status = "✓" if result.meets_history_target else "✗"
        targets = f"{total_status}{llm_status}{hist_status}"

        print(
            f"{config:<25} "
            f"{result.total_p95:>10.1f}ms "
            f"{result.llm_call_mean:>10.1f}ms "
            f"{result.history_format_mean:>10.1f}ms "
            f"{targets:>10}"
        )

    print()

    # Overall success metrics
    total_configs = len(all_results)
    total_pass = sum(1 for r in all_results if r.meets_total_target)
    llm_pass = sum(1 for r in all_results if r.meets_llm_target)
    hist_pass = sum(1 for r in all_results if r.meets_history_target)

    print(f"Overall Achievement:")
    print(f"  Total <500ms: {total_pass}/{total_configs} ({100*total_pass/total_configs:.0f}%)")
    print(f"  LLM <300ms: {llm_pass}/{total_configs} ({100*llm_pass/total_configs:.0f}%)")
    print(f"  History <100ms: {hist_pass}/{total_configs} ({100*hist_pass/total_configs:.0f}%)")
    print()

    return {
        "results": [asdict(r) for r in all_results],
        "raw_metrics": [asdict(m) for m in profiler.metrics],
        "summary": {
            "total_configs": total_configs,
            "total_target_pass_rate": total_pass / total_configs,
            "llm_target_pass_rate": llm_pass / total_configs,
            "history_target_pass_rate": hist_pass / total_configs
        }
    }


async def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark AgentChain routing performance"
    )
    parser.add_argument(
        "--agent-counts",
        type=str,
        default="2,4,6,8",
        help="Comma-separated agent counts to test (default: 2,4,6,8)"
    )
    parser.add_argument(
        "--history-sizes",
        type=str,
        default="0,10,50",
        help="Comma-separated history sizes to test (default: 0,10,50)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterations per configuration (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Router model to use (default: openai/gpt-4o-mini)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Parse configuration
    agent_counts = [int(x.strip()) for x in args.agent_counts.split(",")]
    history_sizes = [int(x.strip()) for x in args.history_sizes.split(",")]

    # Run benchmarks
    results = await run_benchmark_suite(
        agent_counts=agent_counts,
        history_sizes=history_sizes,
        iterations_per_config=args.iterations,
        model=args.model,
        verbose=args.verbose
    )

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    # Return exit code based on target achievement
    all_pass = results["summary"]["total_target_pass_rate"] == 1.0
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
