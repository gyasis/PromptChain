#!/usr/bin/env python3
"""
Migration Example: v0.4.0 → v0.4.1h

This example demonstrates migrating from old patterns to new observability APIs.
Shows before/after code for common scenarios.
"""

from promptchain import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType


def demo_history_manager_migration():
    """Migrate ExecutionHistoryManager usage."""
    print("=" * 80)
    print("ExecutionHistoryManager Migration")
    print("=" * 80)

    history = ExecutionHistoryManager(max_tokens=4000, max_entries=100)
    history.add_entry("user_input", "Hello world", source="user")
    history.add_entry("agent_output", "Hi there!", source="agent")

    # OLD WAY (v0.4.0 and earlier) - DEPRECATED
    print("\n❌ OLD WAY (Deprecated):")
    print("-" * 80)
    print("# ❌ Using private attributes (will break in v0.5.0)")
    print("token_count = history._current_tokens")
    print("entries = history._history")
    print("entry_count = len(history._history)")
    print()

    # Simulate old way (commented out to avoid deprecation warnings)
    # token_count_old = history._current_tokens
    # entries_old = history._history
    # entry_count_old = len(history._history)

    # NEW WAY (v0.4.1a+) - RECOMMENDED
    print("\n✅ NEW WAY (Recommended):")
    print("-" * 80)
    print("# ✅ Using public API (stable, won't break)")

    # Public API
    token_count = history.current_token_count
    entries = history.history
    entry_count = history.history_size

    print(f"token_count = history.current_token_count  # {token_count}")
    print(f"entries = history.history  # {entry_count} entries")
    print(f"entry_count = history.history_size  # {entry_count}")

    # Get statistics
    stats = history.get_statistics()
    print(f"\nstats = history.get_statistics()")
    print(f"# Stats: {list(stats.keys())}")
    print(f"# Token usage: {stats['current_token_count']}/{stats['max_tokens']}")

    print("\n💡 Benefits of new API:")
    print("   - Stable: Won't change without deprecation")
    print("   - Safe: Returns copies, not references")
    print("   - Rich: get_statistics() provides comprehensive info")


def demo_agent_metadata_migration():
    """Migrate to using execution metadata."""
    print("\n\n" + "=" * 80)
    print("AgentChain Metadata Migration")
    print("=" * 80)

    # Create simple agent
    agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Answer briefly: {input}"],
        verbose=False
    )

    agent_chain = AgentChain(
        agents={"responder": agent},
        execution_mode="single_agent",
        verbose=False
    )

    # OLD WAY (v0.4.0 and earlier)
    print("\n❌ OLD WAY:")
    print("-" * 80)
    print("# Only got string response, no metadata")
    print("result = agent_chain.process_input('What is AI?')")
    print("# result is just a string")
    print("# No execution time, no agent info, no tool info")
    print()

    result_old = agent_chain.process_input("What is AI?")
    print(f"type(result): {type(result_old).__name__}")
    print(f"result: {result_old[:80]}...")

    # NEW WAY (v0.4.1b+)
    print("\n\n✅ NEW WAY:")
    print("-" * 80)
    print("# Get rich metadata with return_metadata=True")
    print("result = agent_chain.process_input('What is AI?', return_metadata=True)")
    print()

    result_new = agent_chain.process_input(
        "What is AI?",
        return_metadata=True
    )

    print(f"type(result): {type(result_new).__name__}")
    print(f"result.response: {result_new.response[:80]}...")
    print(f"result.agent_name: {result_new.agent_name}")
    print(f"result.execution_time_ms: {result_new.execution_time_ms:.1f}")
    print(f"result.total_tokens: {result_new.total_tokens}")

    print("\n💡 Benefits of new API:")
    print("   - Rich metadata: execution time, tokens, errors, warnings")
    print("   - Router info: decision details, routing steps")
    print("   - Tool tracking: all tools called with timing")
    print("   - Optional: Use return_metadata=False for old behavior")


def demo_monitoring_migration():
    """Migrate from manual monitoring to event callbacks."""
    print("\n\n" + "=" * 80)
    print("Monitoring Migration: Manual → Event Callbacks")
    print("=" * 80)

    # OLD WAY (v0.4.0 and earlier)
    print("\n❌ OLD WAY:")
    print("-" * 80)
    print("# Manual timing and error tracking")
    print("""
import time

start = time.time()
try:
    result = chain.process_prompt("Input")
    end = time.time()
    print(f"Success: {(end - start) * 1000}ms")
except Exception as e:
    print(f"Error: {e}")
    # Limited error context
""")

    # NEW WAY (v0.4.1d+)
    print("\n\n✅ NEW WAY:")
    print("-" * 80)
    print("# Automatic monitoring with callbacks")
    print("""
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

def monitor_callback(event: ExecutionEvent):
    if event.event_type == ExecutionEventType.CHAIN_END:
        print(f"Success: {event.metadata['execution_time_ms']}ms")
    elif event.event_type == ExecutionEventType.CHAIN_ERROR:
        error = event.metadata['error']
        print(f"Error: {error}")
        print(f"Step: {event.step_number}, Model: {event.model_name}")

chain.register_callback(
    monitor_callback,
    event_filter=[ExecutionEventType.CHAIN_END, ExecutionEventType.CHAIN_ERROR]
)

result = chain.process_prompt("Input")
""")

    # Demo with actual callback
    print("\nRunning with callback:")

    def monitor_callback(event: ExecutionEvent):
        if event.event_type == ExecutionEventType.CHAIN_END:
            print(f"  ✓ Success: {event.metadata.get('execution_time_ms', 0):.1f}ms")
        elif event.event_type == ExecutionEventType.CHAIN_ERROR:
            print(f"  ✗ Error: {event.metadata.get('error')}")

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Answer: {input}"],
        verbose=False
    )

    chain.register_callback(
        monitor_callback,
        event_filter=[ExecutionEventType.CHAIN_END, ExecutionEventType.CHAIN_ERROR]
    )

    result = chain.process_prompt("What is 2+2?")

    print("\n💡 Benefits of callbacks:")
    print("   - Automatic: No manual timing needed")
    print("   - Rich context: Full error details, step info, metadata")
    print("   - Flexible: Filter for specific events")
    print("   - Async support: Both sync and async callbacks")


def demo_complete_migration():
    """Show a complete before/after migration."""
    print("\n\n" + "=" * 80)
    print("Complete Migration Example")
    print("=" * 80)

    # OLD WAY
    print("\n❌ OLD WAY (v0.4.0):")
    print("-" * 80)
    print("""
from promptchain import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
import time

# Manual monitoring
history = ExecutionHistoryManager(max_tokens=4000)
chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Answer: {input}"])

# Manual timing
start = time.time()

try:
    result = chain.process_prompt("Your input")
    end = time.time()

    # Manual tracking
    print(f"Result: {result}")
    print(f"Time: {(end - start) * 1000}ms")

    # Access private attributes (DEPRECATED)
    token_count = history._current_tokens
    print(f"Tokens: {token_count}")

except Exception as e:
    print(f"Error: {e}")
    # Limited error info
""")

    # NEW WAY
    print("\n\n✅ NEW WAY (v0.4.1h):")
    print("-" * 80)
    print("""
from promptchain import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

# Setup with monitoring
history = ExecutionHistoryManager(max_tokens=4000)

def performance_callback(event: ExecutionEvent):
    if event.event_type == ExecutionEventType.CHAIN_END:
        print(f"Time: {event.metadata['execution_time_ms']}ms")
        print(f"Steps: {event.metadata['total_steps']}")

def error_callback(event: ExecutionEvent):
    print(f"Error: {event.metadata['error']}")
    print(f"Context: step={event.step_number}, model={event.model_name}")

chain = PromptChain(
    models=["openai/gpt-4o-mini"],
    instructions=["Answer: {input}"],
    verbose=False
)

# Register callbacks
chain.register_callback(
    performance_callback,
    event_filter=ExecutionEventType.CHAIN_END
)
chain.register_callback(
    error_callback,
    event_filter=[
        ExecutionEventType.CHAIN_ERROR,
        ExecutionEventType.STEP_ERROR
    ]
)

# Execution (monitoring automatic)
result = chain.process_prompt("Your input")
print(f"Result: {result}")

# Use public API
token_count = history.current_token_count
stats = history.get_statistics()
print(f"Tokens: {token_count}")
print(f"Stats: {stats}")
""")

    print("\n\n💡 Migration Benefits Summary:")
    print("-" * 80)
    print("✅ Public APIs: Stable, won't break")
    print("✅ Rich metadata: Comprehensive execution info")
    print("✅ Event system: Automatic monitoring")
    print("✅ Backward compatible: Old code still works")
    print("✅ Performance: <1% overhead")
    print("✅ Async support: Callbacks work with async code")


def demo_migration_checklist():
    """Provide migration checklist."""
    print("\n\n" + "=" * 80)
    print("Migration Checklist")
    print("=" * 80)

    checklist = """
    📋 Step-by-Step Migration:

    1. Update Dependencies
       □ pip install promptchain>=0.4.1h

    2. ExecutionHistoryManager
       □ Replace history._current_tokens → history.current_token_count
       □ Replace history._history → history.history
       □ Replace len(history._history) → history.history_size
       □ Use history.get_statistics() for detailed stats

    3. AgentChain Metadata
       □ Add return_metadata=True where needed
       □ Update code to handle AgentExecutionResult
       □ Access .response, .agent_name, .execution_time_ms, etc.

    4. AgenticStepProcessor Metadata
       □ Add return_metadata=True where needed
       □ Update code to handle AgenticStepResult
       □ Access step-by-step details via .steps

    5. Add Event Monitoring (Optional)
       □ Define callback functions
       □ Register with chain.register_callback()
       □ Use event_filter to reduce overhead
       □ Unregister when done

    6. Testing
       □ Run existing tests (should pass - backward compatible)
       □ Check for deprecation warnings: pytest -W default
       □ Fix deprecation warnings
       □ Test new metadata features
       □ Test event callbacks

    7. Validation
       □ No deprecation warnings: pytest -W error::DeprecationWarning
       □ All tests passing
       □ Monitoring working correctly

    ⏰ Timeline:
       - v0.4.1h: Private attributes deprecated (warnings shown)
       - v0.5.0: Private attributes removed (Q2 2025)
       - Recommendation: Migrate within 3 months
    """

    print(checklist)

    print("\n📚 Resources:")
    print("-" * 80)
    print("  - Public APIs: docs/observability/public-apis.md")
    print("  - Event System: docs/observability/event-system.md")
    print("  - Migration Guide: docs/observability/migration-guide.md")
    print("  - Examples: examples/observability/")


def main():
    """Run all migration examples."""

    # History Manager migration
    demo_history_manager_migration()

    # Agent metadata migration
    demo_agent_metadata_migration()

    # Monitoring migration
    demo_monitoring_migration()

    # Complete migration
    demo_complete_migration()

    # Migration checklist
    demo_migration_checklist()

    print("\n\n" + "=" * 80)
    print("Migration examples complete!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. Review the migration checklist")
    print("   2. Update your code gradually")
    print("   3. Test thoroughly")
    print("   4. Enjoy enhanced observability!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
