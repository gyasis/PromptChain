"""Test T055 exhaustion logging functionality."""
import tempfile
import shutil
import json
from pathlib import Path
from promptchain.cli.session_manager import SessionManager
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

def test_session_manager_exhaustion_logging():
    """Test SessionManager logs exhaustion correctly"""
    # Setup temp directory
    temp_dir = tempfile.mkdtemp()
    sessions_dir = Path(temp_dir) / "sessions"
    sessions_dir.mkdir()

    try:
        # Create session manager
        sm = SessionManager(sessions_dir=sessions_dir)
        session = sm.create_session("test-exhaustion")

        # Log exhaustion
        sm.log_agentic_exhaustion(
            session_id=session.id,
            agent_name="researcher",
            objective="Perform comprehensive analysis of complex topic",
            max_steps=5,
            steps_completed=5,
            partial_result="Partial analysis completed..."
        )

        # Verify logged
        exhaustions = sm.get_exhaustion_history(session.id)
        assert len(exhaustions) == 1, f"Expected 1 exhaustion, got {len(exhaustions)}"
        assert exhaustions[0]["agent_name"] == "researcher"
        assert exhaustions[0]["max_steps"] == 5
        assert exhaustions[0]["completion_detected"] == False
        assert exhaustions[0]["event_type"] == "agentic_exhaustion"

        # Verify JSONL format
        log_file = sessions_dir / session.id / "history.jsonl"
        assert log_file.exists(), "JSONL log file should exist"

        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) >= 1, "Should have at least one JSONL entry"

            # Parse last line (should be our exhaustion entry)
            last_entry = json.loads(lines[-1].strip())
            assert last_entry["event_type"] == "agentic_exhaustion"
            assert "timestamp" in last_entry
            assert last_entry["objective"] == "Perform comprehensive analysis of complex topic"

        print("✓ SessionManager exhaustion logging works")

    finally:
        shutil.rmtree(temp_dir)

def test_execution_history_exhaustion():
    """Test ExecutionHistoryManager exhaustion entries"""
    manager = ExecutionHistoryManager(max_tokens=10000)

    # Add exhaustion entry
    manager.add_exhaustion_entry(
        objective="Research Python best practices",
        max_steps=3,
        steps_completed=3,
        partial_result="Found 5 best practices..."
    )

    # Verify entry
    history = manager.get_history()
    assert len(history) == 1, f"Expected 1 entry, got {len(history)}"

    entry = history[0]
    assert entry["type"] == "agentic_exhaustion"
    assert "Max steps (3) reached" in entry["content"]
    assert "suggestions" in entry["metadata"]
    assert len(entry["metadata"]["suggestions"]) == 4
    assert entry["metadata"]["completion_status"] == "exhausted"
    assert entry["metadata"]["max_steps"] == 3
    assert entry["metadata"]["steps_completed"] == 3
    assert "partial_result_preview" in entry["metadata"]

    # Verify suggestions are actionable
    suggestions = entry["metadata"]["suggestions"]
    assert any("max_internal_steps" in s for s in suggestions)
    assert any("Simplify" in s for s in suggestions)
    assert any("sub-objectives" in s for s in suggestions)
    assert any("achievable" in s for s in suggestions)

    print("✓ ExecutionHistoryManager exhaustion entries work")

def test_multiple_exhaustions():
    """Test logging multiple exhaustion events"""
    temp_dir = tempfile.mkdtemp()
    sessions_dir = Path(temp_dir) / "sessions"
    sessions_dir.mkdir()

    try:
        sm = SessionManager(sessions_dir=sessions_dir)
        session = sm.create_session("multi-exhaustion")

        # Log multiple exhaustions
        for i in range(3):
            sm.log_agentic_exhaustion(
                session_id=session.id,
                agent_name=f"agent_{i}",
                objective=f"Objective {i}",
                max_steps=5 + i,
                steps_completed=5 + i
            )

        # Verify all logged
        exhaustions = sm.get_exhaustion_history(session.id)
        assert len(exhaustions) == 3, f"Expected 3 exhaustions, got {len(exhaustions)}"

        # Verify newest first (agent_2 should be first)
        assert exhaustions[0]["agent_name"] == "agent_2"
        assert exhaustions[1]["agent_name"] == "agent_1"
        assert exhaustions[2]["agent_name"] == "agent_0"

        # Test limit parameter
        limited = sm.get_exhaustion_history(session.id, limit=2)
        assert len(limited) == 2
        assert limited[0]["agent_name"] == "agent_2"
        assert limited[1]["agent_name"] == "agent_1"

        print("✓ Multiple exhaustion events work correctly")

    finally:
        shutil.rmtree(temp_dir)

def test_exhaustion_with_no_partial_result():
    """Test exhaustion logging without partial result"""
    manager = ExecutionHistoryManager(max_tokens=10000)

    # Add exhaustion entry without partial result
    manager.add_exhaustion_entry(
        objective="Quick task that failed",
        max_steps=2,
        steps_completed=2,
        partial_result=None  # No partial result
    )

    # Verify entry
    history = manager.get_history()
    entry = history[0]

    assert entry["type"] == "agentic_exhaustion"
    assert "partial_result_preview" not in entry["metadata"]
    assert "\n\nPartial result" not in entry["content"]

    print("✓ Exhaustion without partial result works")

if __name__ == "__main__":
    test_session_manager_exhaustion_logging()
    test_execution_history_exhaustion()
    test_multiple_exhaustions()
    test_exhaustion_with_no_partial_result()
    print("\n✅ All T055 verification tests passed!")
