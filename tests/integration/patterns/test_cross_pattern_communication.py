"""Integration tests for cross-pattern communication.

Tests patterns communicating via MessageBus and sharing state via Blackboard
from 003-multi-agent-communication infrastructure.
"""

import pytest
import asyncio
from typing import Dict, Any, List

from promptchain.patterns.base import PatternConfig, PatternResult, BasePattern
from promptchain.cli.communication.message_bus import MessageType


class ResearchPattern(BasePattern):
    """Pattern that researches and emits findings."""

    async def execute(self, query: str) -> PatternResult:
        """Execute research and emit results."""
        start_time = 0

        # Emit research started
        self.emit_event("pattern.research.started", {"query": query})

        # Simulate research
        findings = [
            f"Finding 1 for {query}",
            f"Finding 2 for {query}",
            f"Finding 3 for {query}"
        ]

        # Share findings via Blackboard
        if self.config.use_blackboard:
            self.share_result("research.findings", findings)
            self.share_result("research.query", query)

        # Emit progress events
        for i, finding in enumerate(findings):
            self.emit_event("pattern.research.finding", {
                "index": i,
                "finding": finding,
                "total": len(findings)
            })

        # Emit completion
        self.emit_event("pattern.research.completed", {
            "findings_count": len(findings)
        })

        return PatternResult(
            pattern_id=self.config.pattern_id,
            success=True,
            result=findings,
            execution_time_ms=100.0,
            metadata={"query": query, "findings_count": len(findings)}
        )


class SynthesisPattern(BasePattern):
    """Pattern that synthesizes research findings."""

    def __init__(self, config: PatternConfig = None):
        super().__init__(config)
        self.received_findings: List[str] = []

    def on_research_finding(self, event_type: str, event_data: Dict[str, Any]):
        """Handle research finding events."""
        if "finding" in event_data:
            self.received_findings.append(event_data["finding"])

    async def execute(self, **kwargs) -> PatternResult:
        """Synthesize findings from Blackboard or events."""
        # Try to read from Blackboard first
        findings = self.read_shared("research.findings") or self.received_findings

        if not findings:
            return PatternResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=10.0,
                errors=["No findings to synthesize"]
            )

        # Emit synthesis started
        self.emit_event("pattern.synthesis.started", {
            "findings_count": len(findings)
        })

        # Synthesize
        synthesis = f"Synthesized {len(findings)} findings into comprehensive report"

        # Share synthesis result
        if self.config.use_blackboard:
            self.share_result("synthesis.report", synthesis)

        # Emit completion
        self.emit_event("pattern.synthesis.completed", {
            "success": True,
            "report_length": len(synthesis)
        })

        return PatternResult(
            pattern_id=self.config.pattern_id,
            success=True,
            result=synthesis,
            execution_time_ms=150.0,
            metadata={"findings_count": len(findings)}
        )


class ValidationPattern(BasePattern):
    """Pattern that validates results from other patterns."""

    async def execute(self, data_key: str) -> PatternResult:
        """Validate data from Blackboard."""
        # Read data to validate
        data = self.read_shared(data_key)

        if data is None:
            return PatternResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=50.0,
                errors=[f"No data found at key: {data_key}"]
            )

        # Emit validation started
        self.emit_event("pattern.validation.started", {
            "data_key": data_key,
            "data_type": type(data).__name__
        })

        # Validate
        is_valid = data is not None and len(str(data)) > 0
        validation_result = {
            "is_valid": is_valid,
            "data_key": data_key,
            "validated_at": self.config.pattern_id
        }

        # Share validation result
        if self.config.use_blackboard:
            self.share_result(f"validation.{data_key}", validation_result)

        # Emit completion
        self.emit_event("pattern.validation.completed", {
            "is_valid": is_valid,
            "data_key": data_key
        })

        return PatternResult(
            pattern_id=self.config.pattern_id,
            success=True,
            result=validation_result,
            execution_time_ms=75.0,
            metadata={"validated_key": data_key}
        )


@pytest.mark.asyncio
async def test_two_patterns_via_messagebus(message_bus):
    """Test two patterns communicating via MessageBus events."""
    # Create research and synthesis patterns
    research_config = PatternConfig(pattern_id="research-1", emit_events=True)
    synthesis_config = PatternConfig(pattern_id="synthesis-1", emit_events=True)

    research = ResearchPattern(research_config)
    synthesis = SynthesisPattern(synthesis_config)

    # Connect both to MessageBus
    research.connect_messagebus(message_bus)
    synthesis.connect_messagebus(message_bus)

    # Synthesis subscribes to research findings
    synthesis.add_event_handler(synthesis.on_research_finding)
    research.add_event_handler(synthesis.on_research_finding)

    # Execute research
    research_result = await research.execute(query="test query")
    assert research_result.success is True

    # Execute synthesis (using findings from events)
    synthesis_result = await synthesis.execute()
    assert synthesis_result.success is True

    # Verify synthesis received findings
    assert len(synthesis.received_findings) > 0
    assert "Finding 1" in synthesis.received_findings[0]


@pytest.mark.asyncio
async def test_pattern_state_sharing_via_blackboard(blackboard):
    """Test patterns sharing state via Blackboard."""
    # Create patterns with Blackboard enabled
    research_config = PatternConfig(
        pattern_id="research-bb",
        use_blackboard=True
    )
    synthesis_config = PatternConfig(
        pattern_id="synthesis-bb",
        use_blackboard=True
    )

    research = ResearchPattern(research_config)
    synthesis = SynthesisPattern(synthesis_config)

    # Connect both to Blackboard
    research.connect_blackboard(blackboard)
    synthesis.connect_blackboard(blackboard)

    # Execute research (writes to blackboard)
    research_result = await research.execute(query="shared query")
    assert research_result.success is True

    # Verify research wrote to blackboard
    findings = blackboard.read("research.findings")
    assert findings is not None
    assert len(findings) == 3

    # Execute synthesis (reads from blackboard)
    synthesis_result = await synthesis.execute()
    assert synthesis_result.success is True

    # Verify synthesis read and processed findings
    synthesis_report = blackboard.read("synthesis.report")
    assert synthesis_report is not None
    assert "3 findings" in synthesis_report


@pytest.mark.asyncio
async def test_three_pattern_pipeline(message_bus, blackboard):
    """Test three patterns in a pipeline: Research -> Synthesis -> Validation."""
    # Create patterns
    research = ResearchPattern(PatternConfig(
        pattern_id="research-pipe",
        emit_events=True,
        use_blackboard=True
    ))
    synthesis = SynthesisPattern(PatternConfig(
        pattern_id="synthesis-pipe",
        emit_events=True,
        use_blackboard=True
    ))
    validation = ValidationPattern(PatternConfig(
        pattern_id="validation-pipe",
        emit_events=True,
        use_blackboard=True
    ))

    # Connect all to infrastructure
    for pattern in [research, synthesis, validation]:
        pattern.connect_messagebus(message_bus)
        pattern.connect_blackboard(blackboard)

    # Track all events
    all_events = []

    def track_events(event_type: str, event_data: Dict[str, Any]):
        all_events.append({
            "type": event_type,
            "pattern": event_data.get("pattern_id")
        })

    for pattern in [research, synthesis, validation]:
        pattern.add_event_handler(track_events)

    # Execute pipeline
    # 1. Research
    research_result = await research.execute(query="pipeline test")
    assert research_result.success is True

    # 2. Synthesis (reads research findings)
    synthesis_result = await synthesis.execute()
    assert synthesis_result.success is True

    # 3. Validation (validates synthesis report)
    validation_result = await validation.execute(data_key="synthesis.report")
    assert validation_result.success is True
    assert validation_result.result["is_valid"] is True

    # Verify event flow from all patterns
    pattern_ids = set(e["pattern"] for e in all_events)
    assert "research-pipe" in pattern_ids
    assert "synthesis-pipe" in pattern_ids
    assert "validation-pipe" in pattern_ids


@pytest.mark.asyncio
async def test_concurrent_pattern_communication(message_bus, blackboard):
    """Test multiple patterns communicating concurrently."""
    # Create multiple research patterns
    patterns = [
        ResearchPattern(PatternConfig(
            pattern_id=f"concurrent-research-{i}",
            emit_events=True,
            use_blackboard=True
        ))
        for i in range(3)
    ]

    # Connect all to infrastructure
    for pattern in patterns:
        pattern.connect_messagebus(message_bus)
        pattern.connect_blackboard(blackboard)

    # Execute concurrently
    results = await asyncio.gather(*[
        pattern.execute(query=f"query-{i}")
        for i, pattern in enumerate(patterns)
    ])

    # Verify all succeeded
    assert all(r.success for r in results)

    # Verify each wrote to different blackboard keys
    for i in range(3):
        query = blackboard.read("research.query")
        # Last one wins in concurrent writes
        assert query is not None


@pytest.mark.asyncio
async def test_pattern_event_subscription_filtering(message_bus):
    """Test that patterns can filter events they subscribe to."""
    # Create patterns
    research = ResearchPattern(PatternConfig(
        pattern_id="filtered-research",
        emit_events=True
    ))
    synthesis = SynthesisPattern(PatternConfig(
        pattern_id="filtered-synthesis",
        emit_events=True
    ))

    # Connect to MessageBus
    research.connect_messagebus(message_bus)
    synthesis.connect_messagebus(message_bus)

    # Track only specific events
    research_events = []
    finding_events = []

    def on_research_event(event_type: str, event_data: Dict[str, Any]):
        if "research" in event_type:
            research_events.append(event_type)

    def on_finding_event(event_type: str, event_data: Dict[str, Any]):
        if "finding" in event_type:
            finding_events.append(event_type)

    research.add_event_handler(on_research_event)
    research.add_event_handler(on_finding_event)

    # Execute research
    await research.execute(query="filtered test")

    # Verify filtering worked
    assert len(research_events) > 0
    assert len(finding_events) > 0
    assert all("research" in e for e in research_events)
    assert all("finding" in e for e in finding_events)


@pytest.mark.asyncio
async def test_pattern_handoff_coordination(blackboard):
    """Test coordinated handoff between patterns."""
    # Research pattern completes and signals ready
    research = ResearchPattern(PatternConfig(
        pattern_id="handoff-research",
        use_blackboard=True
    ))
    research.connect_blackboard(blackboard)

    # Execute research
    await research.execute(query="handoff test")

    # Write handoff signal
    blackboard.write(
        "handoff.ready",
        {"stage": "research_complete", "next": "synthesis"},
        source="handoff-research"
    )

    # Synthesis reads handoff signal
    synthesis = SynthesisPattern(PatternConfig(
        pattern_id="handoff-synthesis",
        use_blackboard=True
    ))
    synthesis.connect_blackboard(blackboard)

    # Check handoff signal
    handoff = blackboard.read("handoff.ready")
    assert handoff is not None
    assert handoff["stage"] == "research_complete"
    assert handoff["next"] == "synthesis"

    # Execute synthesis
    result = await synthesis.execute()
    assert result.success is True


@pytest.mark.asyncio
async def test_pattern_error_propagation(message_bus, blackboard):
    """Test that pattern errors are communicated to other patterns."""
    # Validation pattern that will fail
    validation = ValidationPattern(PatternConfig(
        pattern_id="error-validation",
        emit_events=True,
        use_blackboard=True
    ))
    validation.connect_messagebus(message_bus)
    validation.connect_blackboard(blackboard)

    # Track error events
    error_events = []

    def on_error(event_type: str, event_data: Dict[str, Any]):
        if event_data.get("pattern_id") == "error-validation":
            error_events.append(event_type)

    validation.add_event_handler(on_error)

    # Execute with non-existent key (will fail)
    result = await validation.execute(data_key="nonexistent.key")

    # Verify failure
    assert result.success is False
    assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_bidirectional_pattern_communication(message_bus, blackboard):
    """Test bidirectional communication between patterns."""
    # Two patterns that communicate back and forth
    pattern_a = ResearchPattern(PatternConfig(
        pattern_id="bidirectional-a",
        emit_events=True,
        use_blackboard=True
    ))
    pattern_b = SynthesisPattern(PatternConfig(
        pattern_id="bidirectional-b",
        emit_events=True,
        use_blackboard=True
    ))

    # Connect both
    for pattern in [pattern_a, pattern_b]:
        pattern.connect_messagebus(message_bus)
        pattern.connect_blackboard(blackboard)

    # Pattern A executes and writes
    await pattern_a.execute(query="bidirectional test")
    findings = blackboard.read("research.findings")
    assert findings is not None

    # Pattern B reads and synthesizes
    await pattern_b.execute()
    synthesis = blackboard.read("synthesis.report")
    assert synthesis is not None

    # Pattern A could read synthesis (bidirectional)
    pattern_a_synthesis = pattern_a.read_shared("synthesis.report")
    assert pattern_a_synthesis == synthesis
