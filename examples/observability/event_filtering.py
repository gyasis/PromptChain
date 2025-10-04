#!/usr/bin/env python3
"""
Event Filtering Example

This example demonstrates advanced event filtering techniques to subscribe
to specific event types and reduce callback overhead.
"""

from collections import defaultdict, Counter
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType


class EventCounter:
    """Count events by type."""

    def __init__(self, name="Counter"):
        self.name = name
        self.counts = Counter()

    def __call__(self, event: ExecutionEvent):
        self.counts[event.event_type.name] += 1

    def report(self):
        print(f"\n{self.name} - Event Counts:")
        for event_type, count in self.counts.most_common():
            print(f"  {event_type}: {count}")
        print(f"  Total: {sum(self.counts.values())}")


class LifecycleMonitor:
    """Monitor chain lifecycle events only."""

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.CHAIN_START:
            print(f"🚀 Chain started at {event.timestamp.strftime('%H:%M:%S')}")
            print(f"   Instructions: {event.metadata.get('total_instructions')}")
            print(f"   Models: {event.metadata.get('models')}")

        elif event.event_type == ExecutionEventType.CHAIN_END:
            print(f"✓ Chain completed in {event.metadata.get('execution_time_ms', 0):.1f}ms")
            print(f"   Total steps: {event.metadata.get('total_steps')}")

        elif event.event_type == ExecutionEventType.CHAIN_ERROR:
            print(f"❌ Chain failed: {event.metadata.get('error')}")


class ModelCallAnalyzer:
    """Analyze model call patterns."""

    def __init__(self):
        self.model_calls = []

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.MODEL_CALL_START:
            self.model_calls.append({
                'model': event.metadata.get('model'),
                'prompt_length': event.metadata.get('prompt_length'),
                'step': event.step_number,
                'timestamp': event.timestamp
            })

        elif event.event_type == ExecutionEventType.MODEL_CALL_END:
            if self.model_calls:
                call = self.model_calls[-1]
                call['tokens_used'] = event.metadata.get('tokens_used', 0)
                call['time_ms'] = event.metadata.get('execution_time_ms', 0)

    def report(self):
        if not self.model_calls:
            print("\nNo model calls recorded")
            return

        print(f"\nModel Call Analysis:")
        print(f"  Total calls: {len(self.model_calls)}")

        total_tokens = sum(c.get('tokens_used', 0) for c in self.model_calls)
        total_time = sum(c.get('time_ms', 0) for c in self.model_calls)

        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {total_time:.1f}ms")
        print(f"  Avg tokens/call: {total_tokens / len(self.model_calls):.1f}")
        print(f"  Avg time/call: {total_time / len(self.model_calls):.1f}ms")


class ToolCallTracker:
    """Track tool call execution."""

    def __init__(self):
        self.tool_calls = defaultdict(list)

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.TOOL_CALL_START:
            tool_name = event.metadata.get('tool_name', 'unknown')
            print(f"🔧 Tool call started: {tool_name}")

        elif event.event_type == ExecutionEventType.TOOL_CALL_END:
            tool_name = event.metadata.get('tool_name', 'unknown')
            time_ms = event.metadata.get('execution_time_ms', 0)
            success = event.metadata.get('success', True)

            self.tool_calls[tool_name].append({
                'time_ms': time_ms,
                'success': success
            })

            status = "✓" if success else "✗"
            print(f"{status} Tool call completed: {tool_name} ({time_ms:.1f}ms)")

    def report(self):
        if not self.tool_calls:
            print("\nNo tool calls recorded")
            return

        print(f"\nTool Call Summary:")
        for tool_name, calls in self.tool_calls.items():
            total_calls = len(calls)
            successes = sum(1 for c in calls if c['success'])
            avg_time = sum(c['time_ms'] for c in calls) / total_calls

            print(f"  {tool_name}:")
            print(f"    Calls: {total_calls}")
            print(f"    Success rate: {successes/total_calls*100:.1f}%")
            print(f"    Avg time: {avg_time:.1f}ms")


class ErrorCollector:
    """Collect all error events."""

    def __init__(self):
        self.errors = []

    def __call__(self, event: ExecutionEvent):
        error_info = {
            'type': event.event_type.name,
            'error': event.metadata.get('error'),
            'step': event.step_number,
            'model': event.model_name,
            'timestamp': event.timestamp
        }
        self.errors.append(error_info)
        print(f"⚠️  Error captured: {event.event_type.name} - {error_info['error']}")

    def has_errors(self):
        return len(self.errors) > 0

    def report(self):
        if not self.errors:
            print("\n✓ No errors encountered")
            return

        print(f"\n❌ Error Summary ({len(self.errors)} errors):")
        for error in self.errors:
            print(f"  [{error['type']}] {error['error']}")
            print(f"    Step: {error['step']}, Model: {error['model']}")


def demo_filter_patterns():
    """Demonstrate different filtering patterns."""

    print("=" * 70)
    print("Event Filtering Patterns Demo")
    print("=" * 70)

    # Pattern 1: Lifecycle events only
    print("\n1. Lifecycle Events Only")
    print("-" * 70)

    chain1 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Analyze: {input}",
            "Summarize: {input}"
        ],
        verbose=False
    )

    lifecycle_monitor = LifecycleMonitor()
    chain1.register_callback(
        lifecycle_monitor,
        event_filter=[
            ExecutionEventType.CHAIN_START,
            ExecutionEventType.CHAIN_END,
            ExecutionEventType.CHAIN_ERROR
        ]
    )

    result1 = chain1.process_prompt("What is machine learning?")

    # Pattern 2: Model call analysis
    print("\n\n2. Model Call Analysis")
    print("-" * 70)

    chain2 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Step 1: {input}",
            "Step 2: {input}",
            "Step 3: {input}"
        ],
        verbose=False
    )

    model_analyzer = ModelCallAnalyzer()
    chain2.register_callback(
        model_analyzer,
        event_filter=[
            ExecutionEventType.MODEL_CALL_START,
            ExecutionEventType.MODEL_CALL_END
        ]
    )

    result2 = chain2.process_prompt("Explain AI briefly")
    model_analyzer.report()

    # Pattern 3: Error events only
    print("\n\n3. Error Monitoring")
    print("-" * 70)

    chain3 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Answer: {input}"],
        verbose=False
    )

    error_collector = ErrorCollector()
    chain3.register_callback(
        error_collector,
        event_filter=[
            ExecutionEventType.CHAIN_ERROR,
            ExecutionEventType.STEP_ERROR,
            ExecutionEventType.MODEL_CALL_ERROR,
            ExecutionEventType.TOOL_CALL_ERROR,
            ExecutionEventType.AGENTIC_STEP_ERROR
        ]
    )

    result3 = chain3.process_prompt("What is Python?")
    error_collector.report()

    # Pattern 4: Count all events vs filtered
    print("\n\n4. Filtered vs Unfiltered Comparison")
    print("-" * 70)

    chain4 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Quick answer: {input}", "Refine: {input}"],
        verbose=False
    )

    # Counter 1: All events
    all_events_counter = EventCounter("All Events")
    chain4.register_callback(all_events_counter)

    # Counter 2: Filtered events (START and END only)
    filtered_counter = EventCounter("Filtered Events")
    chain4.register_callback(
        filtered_counter,
        event_filter=[
            ExecutionEventType.CHAIN_START,
            ExecutionEventType.CHAIN_END,
            ExecutionEventType.STEP_START,
            ExecutionEventType.STEP_END
        ]
    )

    result4 = chain4.process_prompt("Define recursion")

    all_events_counter.report()
    filtered_counter.report()

    reduction = (1 - filtered_counter.counts.total() / all_events_counter.counts.total()) * 100
    print(f"\nFiltering reduced events by {reduction:.1f}%")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_filter_patterns()
