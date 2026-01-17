#!/usr/bin/env python3
"""
Monitoring Dashboard Example

This example demonstrates a real-world monitoring system using callbacks
and metadata for comprehensive observability.
"""

import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any

from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.agent_chain import AgentChain


@dataclass
class DashboardMetrics:
    """Centralized metrics for monitoring dashboard."""

    # Execution counts
    total_chains: int = 0
    successful_chains: int = 0
    failed_chains: int = 0

    # Timing metrics
    total_execution_time_ms: float = 0.0
    execution_times: List[float] = field(default_factory=list)

    # Model metrics
    model_calls: int = 0
    total_tokens: int = 0
    tokens_by_model: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Tool metrics
    tool_calls: int = 0
    tool_execution_times: List[float] = field(default_factory=list)
    tools_by_name: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    error_types: Counter = field(default_factory=Counter)

    # Performance tracking
    slow_executions: List[Dict[str, Any]] = field(default_factory=list)

    def add_chain_start(self):
        self.total_chains += 1

    def add_chain_end(self, execution_time_ms: float):
        self.successful_chains += 1
        self.total_execution_time_ms += execution_time_ms
        self.execution_times.append(execution_time_ms)

        # Track slow executions (>2s)
        if execution_time_ms > 2000:
            self.slow_executions.append({
                'time_ms': execution_time_ms,
                'timestamp': datetime.now()
            })

    def add_chain_error(self):
        self.failed_chains += 1

    def add_model_call(self, model_name: str, tokens: int):
        self.model_calls += 1
        self.total_tokens += tokens
        self.tokens_by_model[model_name] += tokens

    def add_tool_call(self, tool_name: str, execution_time_ms: float):
        self.tool_calls += 1
        self.tool_execution_times.append(execution_time_ms)
        self.tools_by_name[tool_name] += 1

    def add_error(self, error_type: str, error_msg: str, context: Dict[str, Any]):
        self.errors.append({
            'type': error_type,
            'message': error_msg,
            'context': context,
            'timestamp': datetime.now()
        })
        self.error_types[error_type] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        avg_execution_time = (
            sum(self.execution_times) / len(self.execution_times)
            if self.execution_times else 0
        )

        avg_tool_time = (
            sum(self.tool_execution_times) / len(self.tool_execution_times)
            if self.tool_execution_times else 0
        )

        success_rate = (
            (self.successful_chains / self.total_chains * 100)
            if self.total_chains > 0 else 0
        )

        return {
            'execution_summary': {
                'total_chains': self.total_chains,
                'successful': self.successful_chains,
                'failed': self.failed_chains,
                'success_rate': f"{success_rate:.1f}%"
            },
            'performance': {
                'avg_execution_time_ms': f"{avg_execution_time:.1f}",
                'total_execution_time_ms': f"{self.total_execution_time_ms:.1f}",
                'slow_executions': len(self.slow_executions),
                'avg_tool_time_ms': f"{avg_tool_time:.1f}"
            },
            'model_usage': {
                'total_calls': self.model_calls,
                'total_tokens': self.total_tokens,
                'avg_tokens_per_call': (
                    f"{self.total_tokens / self.model_calls:.1f}"
                    if self.model_calls > 0 else "0"
                ),
                'tokens_by_model': dict(self.tokens_by_model)
            },
            'tool_usage': {
                'total_calls': self.tool_calls,
                'unique_tools': len(self.tools_by_name),
                'tools_by_name': dict(self.tools_by_name)
            },
            'errors': {
                'total': len(self.errors),
                'by_type': dict(self.error_types),
                'recent': self.errors[-5:] if self.errors else []
            }
        }


class MonitoringDashboard:
    """Complete monitoring dashboard with callbacks."""

    def __init__(self, alert_threshold_ms=3000, error_alert_threshold=5):
        self.metrics = DashboardMetrics()
        self.alert_threshold_ms = alert_threshold_ms
        self.error_alert_threshold = error_alert_threshold
        self.start_times = {}  # Track start times by step

    def __call__(self, event: ExecutionEvent):
        """Main callback handler."""

        # Chain lifecycle
        if event.event_type == ExecutionEventType.CHAIN_START:
            self.metrics.add_chain_start()
            self.start_times['chain'] = event.timestamp

        elif event.event_type == ExecutionEventType.CHAIN_END:
            exec_time = event.metadata.get('execution_time_ms', 0)
            self.metrics.add_chain_end(exec_time)

            # Alert on slow execution
            if exec_time > self.alert_threshold_ms:
                self._alert_slow_execution(exec_time)

        elif event.event_type == ExecutionEventType.CHAIN_ERROR:
            self.metrics.add_chain_error()

        # Model calls
        elif event.event_type == ExecutionEventType.MODEL_CALL_END:
            model_name = event.metadata.get('model', event.model_name)
            tokens = event.metadata.get('tokens_used', 0)
            self.metrics.add_model_call(model_name, tokens)

        # Tool calls
        elif event.event_type == ExecutionEventType.TOOL_CALL_END:
            tool_name = event.metadata.get('tool_name', 'unknown')
            exec_time = event.metadata.get('execution_time_ms', 0)
            self.metrics.add_tool_call(tool_name, exec_time)

        # Errors
        elif event.event_type.name.endswith('_ERROR'):
            error_msg = event.metadata.get('error', 'Unknown error')
            context = {
                'step': event.step_number,
                'model': event.model_name,
                'instruction': event.step_instruction
            }
            self.metrics.add_error(event.event_type.name, error_msg, context)

            # Alert on error threshold
            if len(self.metrics.errors) >= self.error_alert_threshold:
                self._alert_error_threshold()

    def _alert_slow_execution(self, execution_time_ms: float):
        """Alert on slow execution."""
        print(f"\n🐌 ALERT: Slow execution detected!")
        print(f"   Execution time: {execution_time_ms:.1f}ms")
        print(f"   Threshold: {self.alert_threshold_ms}ms")

    def _alert_error_threshold(self):
        """Alert when error threshold is reached."""
        print(f"\n🚨 ALERT: Error threshold reached!")
        print(f"   Total errors: {len(self.metrics.errors)}")
        print(f"   Recent errors: {self.metrics.error_types.most_common(3)}")

    def display_dashboard(self):
        """Display formatted dashboard."""
        summary = self.metrics.get_summary()

        print("\n" + "=" * 80)
        print(" " * 30 + "MONITORING DASHBOARD")
        print("=" * 80)

        # Execution Summary
        print("\n📊 EXECUTION SUMMARY")
        print("-" * 80)
        exec_summary = summary['execution_summary']
        print(f"  Total Chains:    {exec_summary['total_chains']}")
        print(f"  Successful:      {exec_summary['successful']} ")
        print(f"  Failed:          {exec_summary['failed']}")
        print(f"  Success Rate:    {exec_summary['success_rate']}")

        # Performance
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 80)
        perf = summary['performance']
        print(f"  Avg Execution:   {perf['avg_execution_time_ms']}ms")
        print(f"  Total Time:      {perf['total_execution_time_ms']}ms")
        print(f"  Slow Executions: {perf['slow_executions']}")
        print(f"  Avg Tool Time:   {perf['avg_tool_time_ms']}ms")

        # Model Usage
        print("\n🤖 MODEL USAGE")
        print("-" * 80)
        model = summary['model_usage']
        print(f"  Total Calls:     {model['total_calls']}")
        print(f"  Total Tokens:    {model['total_tokens']}")
        print(f"  Avg Tokens:      {model['avg_tokens_per_call']}")
        if model['tokens_by_model']:
            print(f"  By Model:")
            for model_name, tokens in model['tokens_by_model'].items():
                print(f"    {model_name}: {tokens} tokens")

        # Tool Usage
        print("\n🔧 TOOL USAGE")
        print("-" * 80)
        tools = summary['tool_usage']
        print(f"  Total Calls:     {tools['total_calls']}")
        print(f"  Unique Tools:    {tools['unique_tools']}")
        if tools['tools_by_name']:
            print(f"  By Tool:")
            for tool_name, count in tools['tools_by_name'].items():
                print(f"    {tool_name}: {count} calls")

        # Errors
        print("\n❌ ERROR TRACKING")
        print("-" * 80)
        errors = summary['errors']
        print(f"  Total Errors:    {errors['total']}")
        if errors['by_type']:
            print(f"  By Type:")
            for error_type, count in errors['by_type'].items():
                print(f"    {error_type}: {count}")
        if errors['recent']:
            print(f"  Recent Errors:")
            for error in errors['recent'][-3:]:
                print(f"    [{error['type']}] {error['message'][:50]}")

        print("\n" + "=" * 80)

    def export_metrics(self, filename: str = None):
        """Export metrics to JSON file."""
        summary = self.metrics.get_summary()

        if filename:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n📁 Metrics exported to: {filename}")
        else:
            print(f"\n📄 Metrics JSON:")
            print(json.dumps(summary, indent=2, default=str))


def demo_monitoring_dashboard():
    """Demonstrate monitoring dashboard."""
    print("=" * 80)
    print(" " * 25 + "MONITORING DASHBOARD DEMO")
    print("=" * 80)

    # Create dashboard
    dashboard = MonitoringDashboard(
        alert_threshold_ms=2000,
        error_alert_threshold=3
    )

    # Create agents
    analyzer_agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Analyze: {input}"],
        verbose=False
    )

    writer_agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Write detailed response: {input}"],
        verbose=False
    )

    agent_chain = AgentChain(
        agents={
            "analyzer": analyzer_agent,
            "writer": writer_agent
        },
        agent_descriptions={
            "analyzer": "Analyzes questions",
            "writer": "Writes responses"
        },
        execution_mode="router",
        router={
            "models": ["openai/gpt-4o-mini"],
            "instructions": [None, "{input}"]
        },
        verbose=False
    )

    # Register dashboard callback
    analyzer_agent.register_callback(dashboard)
    writer_agent.register_callback(dashboard)

    # Simulate multiple executions
    print("\n🔄 Running multiple executions...")
    print("-" * 80)

    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does AI work?",
        "What is reinforcement learning?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nExecution {i}: {query}")
        result = agent_chain.process_input(query, return_metadata=True)
        print(f"  Agent: {result.agent_name}")
        print(f"  Time: {result.execution_time_ms:.1f}ms")
        print(f"  Tokens: {result.total_tokens or 'N/A'}")
        time.sleep(0.5)  # Small delay between executions

    # Display dashboard
    dashboard.display_dashboard()

    # Export metrics
    dashboard.export_metrics("monitoring_metrics.json")

    print("\n" + "=" * 80)
    print("Monitoring demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    demo_monitoring_dashboard()
