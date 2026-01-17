"""Demonstration of T055 exhaustion logging with visual output."""
import tempfile
import json
from pathlib import Path
from promptchain.cli.session_manager import SessionManager
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

def demo_session_manager_logging():
    """Demonstrate SessionManager exhaustion logging with JSONL output."""
    print("=" * 80)
    print("DEMO: SessionManager Exhaustion Logging")
    print("=" * 80)

    # Setup temp directory
    temp_dir = tempfile.mkdtemp()
    sessions_dir = Path(temp_dir) / "sessions"
    sessions_dir.mkdir()

    try:
        # Create session manager and session
        sm = SessionManager(sessions_dir=sessions_dir)
        session = sm.create_session("demo-session")

        print(f"\n✓ Created session: {session.id}")

        # Log multiple exhaustion events
        exhaustion_scenarios = [
            {
                "agent_name": "researcher",
                "objective": "Perform comprehensive literature review on quantum computing",
                "max_steps": 8,
                "steps_completed": 8,
                "partial_result": "Reviewed 45 papers, identified 12 key themes, started taxonomy development..."
            },
            {
                "agent_name": "data_analyst",
                "objective": "Analyze customer churn patterns across all segments",
                "max_steps": 5,
                "steps_completed": 5,
                "partial_result": "Analyzed 3 of 7 segments, found significant patterns in segment A and B..."
            },
            {
                "agent_name": "code_reviewer",
                "objective": "Review security vulnerabilities in authentication module",
                "max_steps": 3,
                "steps_completed": 3,
                "partial_result": None  # No partial result
            }
        ]

        print("\n📝 Logging exhaustion events...")
        for scenario in exhaustion_scenarios:
            sm.log_agentic_exhaustion(
                session_id=session.id,
                **scenario
            )
            print(f"  ✓ Logged: {scenario['agent_name']} - {scenario['objective'][:50]}...")

        # Display JSONL log contents
        log_file = sessions_dir / session.id / "history.jsonl"
        print(f"\n📄 JSONL Log File: {log_file}")
        print("-" * 80)

        with open(log_file, "r") as f:
            for i, line in enumerate(f, 1):
                entry = json.loads(line.strip())
                print(f"\nEntry #{i}:")
                print(json.dumps(entry, indent=2))

        # Retrieve and display exhaustion history
        print("\n" + "=" * 80)
        print("EXHAUSTION HISTORY (via get_exhaustion_history)")
        print("=" * 80)

        exhaustions = sm.get_exhaustion_history(session.id)
        print(f"\n📊 Total exhaustion events: {len(exhaustions)}")

        for i, event in enumerate(exhaustions, 1):
            print(f"\n{i}. Agent: {event['agent_name']}")
            print(f"   Objective: {event['objective'][:60]}...")
            print(f"   Steps: {event['steps_completed']}/{event['max_steps']}")
            print(f"   Partial Result: {event['partial_result_length']} chars")
            print(f"   Timestamp: {event['timestamp']}")

        # Test limit parameter
        print("\n" + "-" * 80)
        print("LIMITED QUERY (limit=2)")
        print("-" * 80)

        limited = sm.get_exhaustion_history(session.id, limit=2)
        print(f"\nRetrieved {len(limited)} most recent events:")
        for event in limited:
            print(f"  - {event['agent_name']}: {event['objective'][:50]}...")

    finally:
        import shutil
        shutil.rmtree(temp_dir)
        print("\n🧹 Cleaned up temp directory")

def demo_execution_history_manager():
    """Demonstrate ExecutionHistoryManager exhaustion entries."""
    print("\n" + "=" * 80)
    print("DEMO: ExecutionHistoryManager Exhaustion Entries")
    print("=" * 80)

    manager = ExecutionHistoryManager(max_tokens=10000)

    # Simulate workflow with exhaustion
    print("\n📝 Simulating agentic workflow...")

    manager.add_entry("user_input", "Research best practices for API design", source="user")
    manager.add_entry("agentic_step", "Starting research on RESTful APIs...", source="agent")
    manager.add_entry("tool_call", "search_web('REST API best practices')", source="agent")
    manager.add_entry("tool_result", "Found 25 relevant articles", source="tool")
    manager.add_entry("agentic_step", "Analyzing GraphQL vs REST...", source="agent")

    # Add exhaustion entry
    print("\n⚠️  Adding exhaustion entry...")
    manager.add_exhaustion_entry(
        objective="Research and compare API design patterns (REST, GraphQL, gRPC)",
        max_steps=5,
        steps_completed=5,
        partial_result="Completed REST analysis, started GraphQL comparison, identified 8 key differences..."
    )

    # Display history
    print("\n" + "=" * 80)
    print("EXECUTION HISTORY")
    print("=" * 80)

    history = manager.get_history()
    print(f"\n📊 Total entries: {len(history)}")

    for i, entry in enumerate(history, 1):
        print(f"\n{i}. Type: {entry['type']}")
        print(f"   Source: {entry['source']}")
        print(f"   Content: {str(entry['content'])[:80]}...")
        if entry.get('metadata'):
            print(f"   Metadata keys: {list(entry['metadata'].keys())}")

    # Show exhaustion entry details
    exhaustion_entry = [e for e in history if e['type'] == 'agentic_exhaustion'][0]
    print("\n" + "=" * 80)
    print("EXHAUSTION ENTRY DETAILS")
    print("=" * 80)
    print(json.dumps(exhaustion_entry, indent=2, default=str))

    # Display suggestions
    print("\n" + "=" * 80)
    print("ACTIONABLE SUGGESTIONS")
    print("=" * 80)
    suggestions = exhaustion_entry['metadata']['suggestions']
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

if __name__ == "__main__":
    demo_session_manager_logging()
    demo_execution_history_manager()
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
