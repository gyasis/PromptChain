"""
Integration tests for Verification System (CoVe + Checkpointing).

Tests the complete safety and reliability system including:
- Chain of Verification (CoVe) preventing risky tool calls
- Checkpoint Manager detecting stuck states and rolling back
- Combined system reducing errors by 40%+ vs baseline

Test Coverage:
- CoVe rejection of low-confidence tool calls
- Stuck state detection and automatic rollback
- Error reduction benchmarking
- Real-world failure scenarios
"""

import pytest
import json
from typing import List, Dict, Any
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class ErrorTrackingMockLLM:
    """Mock LLM that can simulate various failure scenarios."""

    def __init__(self, responses: List[Dict[str, Any]], error_scenarios: Dict[str, bool] = None):
        """
        Initialize with predefined responses and optional error scenarios.

        Args:
            responses: List of response dicts
            error_scenarios: Dict mapping scenario names to whether they should fail
        """
        self.responses = responses
        self.error_scenarios = error_scenarios or {}
        self.call_count = 0
        self.tool_calls_attempted = []
        self.tool_calls_executed = []
        self.errors_encountered = []
        self.rollbacks_triggered = 0

    async def __call__(self, messages: List[Dict], model: str = None, tools: List = None, **kwargs):
        """Mock LLM call that tracks attempted tool calls."""
        if self.call_count >= len(self.responses):
            return {"content": "Task completed", "tool_calls": []}

        response = self.responses[self.call_count]
        self.call_count += 1

        # Track tool calls that were attempted
        if "tool_calls" in response and response["tool_calls"]:
            for tc in response["tool_calls"]:
                self.tool_calls_attempted.append(tc["function"]["name"])

        return response


class TestCoVeErrorPrevention:
    """Test that CoVe prevents risky tool executions."""

    @pytest.mark.asyncio
    async def test_cove_rejects_low_confidence_calls(self):
        """Test that CoVe rejects tool calls below confidence threshold."""

        # Scenario: Agent wants to delete a file, but verification says confidence is low
        responses = [
            {
                "content": "I'll delete the old backup file",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "delete_file", "arguments": json.dumps({"path": "/data/backup.db"})}
                }]
            },
            {"content": "Task completed", "tool_calls": []}
        ]

        # Mock verification response (low confidence)
        verification_responses = [
            json.dumps({
                "should_execute": True,
                "confidence": 0.3,  # Below default threshold of 0.7
                "assumptions": ["File exists", "Not currently in use"],
                "risks": ["File might be needed", "No backup of backup"],
                "reasoning": "Low confidence - unclear if file is safe to delete"
            })
        ]

        mock_llm = ErrorTrackingMockLLM(responses)
        mock_verification_llm = ErrorTrackingMockLLM([{"content": verification_responses[0]}])

        processor = AgenticStepProcessor(
            objective="Clean up old files",
            max_internal_steps=3,
            enable_cove=True,  # Enable CoVe
            cove_confidence_threshold=0.7,  # Default threshold
            enable_blackboard=True
        )

        # Override verification LLM
        from promptchain.utils.verification import CoVeVerifier
        processor.cove_verifier = CoVeVerifier(
            llm_runner=mock_verification_llm,
            model_name="mock"
        )

        tool_execution_count = 0

        async def tool_executor_that_counts(tool_call):
            nonlocal tool_execution_count
            tool_execution_count += 1
            mock_llm.tool_calls_executed.append(tool_call["function"]["name"])
            return "File deleted"

        await processor.run_async(
            initial_input="Clean up old backup files",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=tool_executor_that_counts
        )

        # Assertions
        assert len(mock_llm.tool_calls_attempted) == 1, "Agent attempted 1 tool call"
        assert len(mock_llm.tool_calls_executed) == 0, "CoVe should block low-confidence call"
        assert tool_execution_count == 0, "No tools should be executed"
        assert "delete_file" in mock_llm.tool_calls_attempted
        assert "delete_file" not in mock_llm.tool_calls_executed

    @pytest.mark.asyncio
    async def test_cove_allows_high_confidence_calls(self):
        """Test that CoVe allows tool calls with high confidence."""

        responses = [
            {
                "content": "I'll read the configuration file",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "/config/app.json"})}
                }]
            },
            {"content": "Configuration loaded", "tool_calls": []}
        ]

        # Mock verification response (high confidence)
        verification_responses = [
            json.dumps({
                "should_execute": True,
                "confidence": 0.95,  # Well above threshold
                "assumptions": ["File exists and is readable"],
                "risks": ["Minimal - read-only operation"],
                "reasoning": "High confidence - safe read operation"
            })
        ]

        mock_llm = ErrorTrackingMockLLM(responses)
        mock_verification_llm = ErrorTrackingMockLLM([{"content": verification_responses[0]}])

        processor = AgenticStepProcessor(
            objective="Load configuration",
            max_internal_steps=3,
            enable_cove=True,
            cove_confidence_threshold=0.7,
            enable_blackboard=True
        )

        from promptchain.utils.verification import CoVeVerifier
        processor.cove_verifier = CoVeVerifier(
            llm_runner=mock_verification_llm,
            model_name="mock"
        )

        tool_execution_count = 0

        async def tool_executor_that_counts(tool_call):
            nonlocal tool_execution_count
            tool_execution_count += 1
            mock_llm.tool_calls_executed.append(tool_call["function"]["name"])
            return "Config content: {}"

        await processor.run_async(
            initial_input="Load the application configuration",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=tool_executor_that_counts
        )

        # Assertions
        assert len(mock_llm.tool_calls_attempted) == 1, "Agent attempted 1 tool call"
        assert len(mock_llm.tool_calls_executed) == 1, "CoVe should allow high-confidence call"
        assert tool_execution_count == 1, "Tool should be executed"
        assert "read_file" in mock_llm.tool_calls_executed


class TestCheckpointStuckStateDetection:
    """Test that CheckpointManager detects and recovers from stuck states."""

    @pytest.mark.asyncio
    async def test_stuck_state_detection_and_rollback(self):
        """Test that agent detects stuck state (same tool 3+ times) and rolls back."""

        # Scenario: Agent keeps trying the same failed operation
        responses = [
            {
                "content": "I'll search the database",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_db", "arguments": "{}"}
                }]
            },
            {
                "content": "Let me try searching again",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search_db", "arguments": "{}"}
                }]
            },
            {
                "content": "One more search attempt",
                "tool_calls": [{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "search_db", "arguments": "{}"}
                }]
            },
            {"content": "Giving up", "tool_calls": []}
        ]

        mock_llm = ErrorTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Find user data",
            max_internal_steps=5,
            enable_checkpointing=True,  # Enable checkpoint manager (stuck_threshold defaults to 3)
            enable_blackboard=True
        )

        tool_call_history = []

        async def tool_executor_that_fails(tool_call):
            tool_call_history.append(tool_call["function"]["name"])
            return "Error: Database connection failed"

        await processor.run_async(
            initial_input="Search for user data",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=tool_executor_that_fails
        )

        # Check if stuck state was detected
        assert len(tool_call_history) >= 3, "At least 3 search attempts"
        assert processor.checkpoint_manager is not None
        assert processor.checkpoint_manager.is_stuck(), "Should detect stuck state"

        # Verify blackboard recorded the error
        if processor.blackboard:
            errors = processor.blackboard._state.get("errors", [])
            # Check if any error mentions stuck state or rollback
            stuck_detected = any("stuck" in str(e).lower() or "rolled back" in str(e).lower() for e in errors)
            assert stuck_detected, "Blackboard should record stuck state detection"


class TestCombinedVerificationSystem:
    """Test CoVe + Checkpointing working together."""

    @pytest.mark.asyncio
    async def test_combined_system_prevents_stuck_loops(self):
        """Test that CoVe + Checkpointing together prevent stuck loops."""

        # Scenario: Agent tries risky operation repeatedly
        responses = [
            {
                "content": "I'll try to connect to the server",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "connect_server", "arguments": json.dumps({"host": "unreachable.server"})}
                }]
            }
            for i in range(5)
        ]
        responses.append({"content": "Connection attempts failed", "tool_calls": []})

        # Mock verification responses (varying confidence)
        verification_responses = [
            json.dumps({
                "should_execute": True,
                "confidence": 0.8,  # First attempt: high confidence
                "assumptions": ["Server is reachable"],
                "risks": ["Connection timeout"],
                "reasoning": "Initial connection attempt seems reasonable"
            }),
            json.dumps({
                "should_execute": True,
                "confidence": 0.6,  # Second attempt: lower confidence
                "assumptions": ["Server might come back"],
                "risks": ["Wasting resources"],
                "reasoning": "Previous attempt failed, less confident"
            }),
            json.dumps({
                "should_execute": True,
                "confidence": 0.3,  # Third attempt: very low confidence
                "assumptions": ["Maybe network recovered"],
                "risks": ["Stuck in retry loop"],
                "reasoning": "Multiple failures suggest futile attempts"
            })
        ]

        mock_llm = ErrorTrackingMockLLM(responses)
        mock_verification_llm = ErrorTrackingMockLLM(
            [{"content": vr} for vr in verification_responses]
        )

        processor = AgenticStepProcessor(
            objective="Connect to remote server",
            max_internal_steps=6,
            enable_cove=True,
            cove_confidence_threshold=0.7,
            enable_checkpointing=True,
            enable_blackboard=True
        )

        from promptchain.utils.verification import CoVeVerifier
        processor.cove_verifier = CoVeVerifier(
            llm_runner=mock_verification_llm,
            model_name="mock"
        )

        tool_execution_count = 0
        failed_connections = []

        async def tool_executor_that_fails(tool_call):
            nonlocal tool_execution_count
            tool_execution_count += 1
            failed_connections.append(tool_call["function"]["name"])
            return "Error: Connection timeout"

        await processor.run_async(
            initial_input="Connect to the server at unreachable.server",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=tool_executor_that_fails
        )

        # Assertions
        # CoVe should block attempts after confidence drops below threshold
        assert tool_execution_count <= 2, "CoVe should prevent excessive retry attempts"
        assert len(failed_connections) <= 2, "System should limit failed connection attempts"


class TestErrorReductionBenchmark:
    """Benchmark error reduction with CoVe + Checkpointing enabled."""

    @pytest.mark.asyncio
    async def test_error_reduction_benchmark(self):
        """
        Benchmark showing 40%+ error reduction with verification enabled.

        Simulates a workflow with potential errors:
        - Bad parameter values
        - Stuck retry loops
        - Unsafe operations

        Compares error count with and without verification system.
        """

        # Define error-prone scenario responses
        error_prone_responses = [
            # Bad file path (should be rejected by CoVe)
            {
                "content": "Deleting system files",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "delete_file", "arguments": json.dumps({"path": "/system/critical.dat"})}
                }]
            },
            # Retry same operation (should trigger stuck detection)
            {
                "content": "Retrying connection",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "connect", "arguments": "{}"}
                }]
            },
            {
                "content": "Retrying connection again",
                "tool_calls": [{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "connect", "arguments": "{}"}
                }]
            },
            {
                "content": "Retrying connection one more time",
                "tool_calls": [{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "connect", "arguments": "{}"}
                }]
            },
            # Invalid operation (should be rejected)
            {
                "content": "Executing unsafe command",
                "tool_calls": [{
                    "id": "call_5",
                    "type": "function",
                    "function": {"name": "exec_shell", "arguments": json.dumps({"cmd": "rm -rf /"})}
                }]
            },
            {"content": "Done", "tool_calls": []}
        ]

        # Verification responses (rejecting risky operations)
        verification_responses = [
            json.dumps({
                "should_execute": False,
                "confidence": 0.1,
                "assumptions": [],
                "risks": ["Critical system file deletion"],
                "reasoning": "DANGEROUS: Deleting system files"
            }),
            json.dumps({
                "should_execute": True,
                "confidence": 0.8,
                "assumptions": ["Connection might work"],
                "risks": ["Timeout"],
                "reasoning": "First retry is reasonable"
            }),
            json.dumps({
                "should_execute": True,
                "confidence": 0.5,
                "assumptions": ["Maybe network recovered"],
                "risks": ["Wasting resources"],
                "reasoning": "Second retry less promising"
            }),
            json.dumps({
                "should_execute": False,
                "confidence": 0.2,
                "assumptions": [],
                "risks": ["Stuck in retry loop"],
                "reasoning": "Too many retries, giving up"
            }),
            json.dumps({
                "should_execute": False,
                "confidence": 0.0,
                "assumptions": [],
                "risks": ["CATASTROPHIC: System destruction"],
                "reasoning": "BLOCKED: Extremely dangerous command"
            })
        ]

        # Test WITHOUT verification (baseline)
        mock_llm_baseline = ErrorTrackingMockLLM(error_prone_responses.copy())

        processor_baseline = AgenticStepProcessor(
            objective="System maintenance",
            max_internal_steps=7,
            enable_cove=False,  # No CoVe
            enable_checkpointing=False,  # No checkpointing
            enable_blackboard=True
        )

        baseline_errors = []
        baseline_dangerous_ops = []

        async def tool_executor_baseline(tool_call):
            tool_name = tool_call["function"]["name"]

            # Track dangerous operations
            if tool_name in ["delete_file", "exec_shell"]:
                baseline_dangerous_ops.append(tool_name)
                baseline_errors.append(f"DANGER: {tool_name} executed")
                return "ERROR: Operation failed catastrophically"

            if tool_name == "connect":
                baseline_errors.append("Connection failed")
                return "ERROR: Connection timeout"

            return "Success"

        await processor_baseline.run_async(
            initial_input="Perform system maintenance",
            available_tools=[],
            llm_runner=mock_llm_baseline,
            tool_executor=tool_executor_baseline
        )

        baseline_error_count = len(baseline_errors)
        baseline_dangerous_count = len(baseline_dangerous_ops)

        # Test WITH verification (protected)
        mock_llm_protected = ErrorTrackingMockLLM(error_prone_responses.copy())
        mock_verification_llm = ErrorTrackingMockLLM(
            [{"content": vr} for vr in verification_responses]
        )

        processor_protected = AgenticStepProcessor(
            objective="System maintenance",
            max_internal_steps=7,
            enable_cove=True,  # Enable CoVe
            cove_confidence_threshold=0.7,
            enable_checkpointing=True,  # Enable checkpointing
            enable_blackboard=True
        )

        from promptchain.utils.verification import CoVeVerifier
        processor_protected.cove_verifier = CoVeVerifier(
            llm_runner=mock_verification_llm,
            model_name="mock"
        )

        protected_errors = []
        protected_dangerous_ops = []

        async def tool_executor_protected(tool_call):
            tool_name = tool_call["function"]["name"]

            # Track dangerous operations (shouldn't happen with CoVe)
            if tool_name in ["delete_file", "exec_shell"]:
                protected_dangerous_ops.append(tool_name)
                protected_errors.append(f"DANGER: {tool_name} executed")
                return "ERROR: Operation failed catastrophically"

            if tool_name == "connect":
                protected_errors.append("Connection failed")
                return "ERROR: Connection timeout"

            return "Success"

        await processor_protected.run_async(
            initial_input="Perform system maintenance",
            available_tools=[],
            llm_runner=mock_llm_protected,
            tool_executor=tool_executor_protected
        )

        protected_error_count = len(protected_errors)
        protected_dangerous_count = len(protected_dangerous_ops)

        # Calculate reduction
        if baseline_error_count > 0:
            error_reduction_pct = ((baseline_error_count - protected_error_count) / baseline_error_count) * 100
        else:
            error_reduction_pct = 0

        print(f"\n=== Error Reduction Benchmark ===")
        print(f"Baseline errors:     {baseline_error_count}")
        print(f"Protected errors:    {protected_error_count}")
        print(f"Error reduction:     {error_reduction_pct:.1f}%")
        print(f"Baseline dangerous:  {baseline_dangerous_count}")
        print(f"Protected dangerous: {protected_dangerous_count}")
        print(f"Dangerous ops prevented: {baseline_dangerous_count - protected_dangerous_count}")

        # Assertions
        assert protected_error_count < baseline_error_count, "Protected system should have fewer errors"
        assert protected_dangerous_count < baseline_dangerous_count, "Protected system should prevent dangerous ops"
        assert error_reduction_pct >= 40, f"Expected ≥40% error reduction, got {error_reduction_pct:.1f}%"
        assert protected_dangerous_count == 0, "No dangerous operations should execute with CoVe enabled"


class TestRealWorldScenarios:
    """Test verification system in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_database_migration_safety(self):
        """Test that CoVe prevents unsafe database migrations."""

        responses = [
            {
                "content": "I'll drop the production table to migrate schema",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "drop_table", "arguments": json.dumps({"table": "users_production"})}
                }]
            },
            {"content": "Migration complete", "tool_calls": []}
        ]

        verification_responses = [
            json.dumps({
                "should_execute": False,
                "confidence": 0.05,
                "assumptions": ["Backup exists"],
                "risks": ["DATA LOSS", "Service outage", "No rollback"],
                "reasoning": "CRITICAL: Dropping production table without backup is catastrophic"
            })
        ]

        mock_llm = ErrorTrackingMockLLM(responses)
        mock_verification_llm = ErrorTrackingMockLLM([{"content": verification_responses[0]}])

        processor = AgenticStepProcessor(
            objective="Migrate database schema",
            max_internal_steps=3,
            enable_cove=True,
            cove_confidence_threshold=0.7,
            enable_blackboard=True
        )

        from promptchain.utils.verification import CoVeVerifier
        processor.cove_verifier = CoVeVerifier(
            llm_runner=mock_verification_llm,
            model_name="mock"
        )

        dangerous_ops_executed = []

        async def tool_executor(tool_call):
            tool_name = tool_call["function"]["name"]
            if tool_name == "drop_table":
                dangerous_ops_executed.append(tool_name)
            return "Success"

        await processor.run_async(
            initial_input="Migrate the database schema",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=tool_executor
        )

        # CoVe should block the dangerous operation
        assert len(dangerous_ops_executed) == 0, "CoVe should prevent production table drop"

    @pytest.mark.asyncio
    async def test_api_rate_limit_stuck_prevention(self):
        """Test that checkpointing prevents API rate limit stuck loops."""

        # Agent keeps hitting rate limit
        responses = [
            {
                "content": f"Calling API attempt {i}",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "call_api", "arguments": "{}"}
                }]
            }
            for i in range(6)
        ]
        responses.append({"content": "Too many failures", "tool_calls": []})

        mock_llm = ErrorTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Fetch API data",
            max_internal_steps=8,
            enable_checkpointing=True,  # CheckpointManager uses default stuck_threshold=3
            enable_blackboard=True
        )

        api_calls = []

        async def tool_executor(tool_call):
            api_calls.append("call_api")
            return "Error: Rate limit exceeded (429)"

        await processor.run_async(
            initial_input="Fetch data from API",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=tool_executor
        )

        # Should detect stuck state after 3 failed attempts
        assert processor.checkpoint_manager.is_stuck(), "Should detect stuck state"
        # Checkpoint system will rollback and allow retries, but should eventually stop
        # With rollback, multiple cycles can occur, so allow up to 7 calls
        assert len(api_calls) >= 3, "Should make at least 3 calls to detect stuck state"
        assert len(api_calls) <= 7, "Should limit API calls with rollback cycles"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
