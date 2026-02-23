"""
Unit tests for CoVeVerifier class (Chain of Verification).

Tests cover:
- Verification with different confidence levels
- Tool approval/rejection logic
- Suggested modifications
- JSON parsing (success and fallback cases)
- Edge cases and error handling
"""

import pytest
import json
from promptchain.utils.verification import CoVeVerifier, VerificationResult


class MockLLMRunner:
    """Mock LLM runner for testing verification."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.call_count = 0
        self.last_messages = None

    async def __call__(self, messages, model=None):
        """Mock LLM call."""
        self.call_count += 1
        self.last_messages = messages
        return {"content": self.response_content}


class TestCoVeVerifierInitialization:
    """Test CoVeVerifier initialization."""

    def test_init_with_model(self):
        """Test initialization with model name."""
        mock_runner = MockLLMRunner("")
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        assert verifier.llm_runner == mock_runner
        assert verifier.model_name == "openai/gpt-4"


class TestVerificationDecisions:
    """Test verification decision logic."""

    @pytest.mark.asyncio
    async def test_approve_high_confidence_tool(self):
        """Test approval of tool with high confidence."""
        verification_response = json.dumps({
            "should_execute": True,
            "confidence": 0.9,
            "assumptions": ["Database connection is available"],
            "risks": ["Query might be slow"],
            "reasoning": "Query is safe and well-formed"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="query_database",
            tool_args={"query": "SELECT * FROM users LIMIT 10"},
            context="User requested to list users",
            available_tools=[]
        )

        assert result.should_execute is True
        assert result.confidence == 0.9
        assert len(result.assumptions) == 1
        assert len(result.risks) == 1
        assert "safe" in result.verification_reasoning.lower()

    @pytest.mark.asyncio
    async def test_reject_low_confidence_tool(self):
        """Test rejection of tool with low confidence."""
        verification_response = json.dumps({
            "should_execute": False,
            "confidence": 0.3,
            "assumptions": ["File exists", "User has permission"],
            "risks": ["Might delete wrong file", "No backup available"],
            "reasoning": "File path looks suspicious, better to verify first"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="delete_file",
            tool_args={"path": "/important/data.db"},
            context="User asked to clean up files",
            available_tools=[]
        )

        assert result.should_execute is False
        assert result.confidence == 0.3
        assert len(result.assumptions) == 2
        assert len(result.risks) == 2
        assert "suspicious" in result.verification_reasoning.lower()

    @pytest.mark.asyncio
    async def test_suggested_modifications(self):
        """Test tool verification with suggested modifications."""
        verification_response = json.dumps({
            "should_execute": True,
            "confidence": 0.85,
            "assumptions": ["API is available"],
            "risks": ["Timeout on slow network"],
            "reasoning": "API call is safe but should add timeout",
            "suggested_modifications": {"timeout": 30}
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="api_call",
            tool_args={"endpoint": "/users"},
            context="Fetching user data",
            available_tools=[]
        )

        assert result.should_execute is True
        assert result.confidence == 0.85
        assert result.suggested_modifications is not None
        assert result.suggested_modifications["timeout"] == 30


class TestConfidenceBounds:
    """Test confidence value clamping."""

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_max(self):
        """Test confidence is clamped to 1.0."""
        verification_response = json.dumps({
            "should_execute": True,
            "confidence": 1.5,  # Invalid: > 1.0
            "assumptions": [],
            "risks": [],
            "reasoning": "Perfectly safe"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="safe_tool",
            tool_args={},
            context="",
            available_tools=[]
        )

        assert result.confidence == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_min(self):
        """Test confidence is clamped to 0.0."""
        verification_response = json.dumps({
            "should_execute": False,
            "confidence": -0.2,  # Invalid: < 0.0
            "assumptions": [],
            "risks": [],
            "reasoning": "Very risky"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="risky_tool",
            tool_args={},
            context="",
            available_tools=[]
        )

        assert result.confidence == 0.0  # Clamped to min


class TestJSONParsing:
    """Test JSON response parsing."""

    @pytest.mark.asyncio
    async def test_parse_clean_json(self):
        """Test parsing clean JSON response."""
        verification_response = json.dumps({
            "should_execute": True,
            "confidence": 0.8,
            "assumptions": ["Assumption 1"],
            "risks": ["Risk 1"],
            "reasoning": "Looks good"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="test_tool",
            tool_args={},
            context="",
            available_tools=[]
        )

        assert result.should_execute is True
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code fences."""
        verification_response = f"""```json
{json.dumps({
    "should_execute": True,
    "confidence": 0.75,
    "assumptions": [],
    "risks": [],
    "reasoning": "Safe to proceed"
})}
```"""

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="test_tool",
            tool_args={},
            context="",
            available_tools=[]
        )

        assert result.should_execute is True
        assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        """Test fallback behavior when JSON parsing fails."""
        verification_response = "This is not valid JSON at all"

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="test_tool",
            tool_args={},
            context="",
            available_tools=[]
        )

        # Should fallback to allowing execution with low confidence
        assert result.should_execute is True
        assert result.confidence == 0.4  # Default fallback confidence
        assert "Could not parse" in result.risks[0]

    @pytest.mark.asyncio
    async def test_fallback_detects_negative_phrases(self):
        """Test fallback uses heuristics to detect negative indicators."""
        verification_response = "Do not execute this tool. It is dangerous and risky."

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="dangerous_tool",
            tool_args={},
            context="",
            available_tools=[]
        )

        # Should detect negative phrases and set should_execute=False
        assert result.should_execute is False
        assert result.confidence == 0.4


class TestToolSchemaFinding:
    """Test tool schema discovery."""

    @pytest.mark.asyncio
    async def test_find_tool_schema(self):
        """Test finding tool schema from available tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Search database records",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }
            }
        ]

        verification_response = json.dumps({
            "should_execute": True,
            "confidence": 0.8,
            "assumptions": [],
            "risks": [],
            "reasoning": "Valid tool"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        await verifier.verify_tool_call(
            tool_name="search_database",
            tool_args={"query": "SELECT *"},
            context="",
            available_tools=tools
        )

        # Verify the prompt includes the correct tool schema
        prompt = mock_runner.last_messages[0]["content"]
        assert "search_database" in prompt
        assert "Search database records" in prompt

    @pytest.mark.asyncio
    async def test_missing_tool_schema(self):
        """Test behavior when tool schema not found."""
        tools = []  # Empty tool list

        verification_response = json.dumps({
            "should_execute": True,
            "confidence": 0.5,
            "assumptions": [],
            "risks": [],
            "reasoning": "Unknown tool"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="unknown_tool",
            tool_args={},
            context="",
            available_tools=tools
        )

        # Should still work, just without schema info
        assert result.should_execute is True


class TestContextTruncation:
    """Test context truncation for long inputs."""

    @pytest.mark.asyncio
    async def test_long_context_truncated(self):
        """Test that very long context is truncated."""
        long_context = "A" * 5000  # 5000 characters

        verification_response = json.dumps({
            "should_execute": True,
            "confidence": 0.7,
            "assumptions": [],
            "risks": [],
            "reasoning": "Context truncated"
        })

        mock_runner = MockLLMRunner(verification_response)
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        await verifier.verify_tool_call(
            tool_name="test_tool",
            tool_args={},
            context=long_context,
            available_tools=[]
        )

        # Verify context was truncated in prompt
        prompt = mock_runner.last_messages[0]["content"]
        assert "(truncated)" in prompt


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_exception_during_verification(self):
        """Test graceful handling of exceptions."""

        async def failing_runner(messages, model=None):
            raise RuntimeError("LLM call failed")

        verifier = CoVeVerifier(failing_runner, "openai/gpt-4")

        result = await verifier.verify_tool_call(
            tool_name="test_tool",
            tool_args={},
            context="",
            available_tools=[]
        )

        # Should return fallback result allowing execution
        assert result.should_execute is True
        assert result.confidence == 0.5
        assert "Verification error" in result.risks[0]


class TestVerificationResultDataclass:
    """Test VerificationResult dataclass."""

    def test_verification_result_creation(self):
        """Test creating VerificationResult."""
        result = VerificationResult(
            should_execute=True,
            confidence=0.85,
            assumptions=["Assumption 1", "Assumption 2"],
            risks=["Risk 1"],
            verification_reasoning="Looks safe",
            suggested_modifications={"timeout": 30}
        )

        assert result.should_execute is True
        assert result.confidence == 0.85
        assert len(result.assumptions) == 2
        assert len(result.risks) == 1
        assert result.suggested_modifications["timeout"] == 30

    def test_verification_result_without_modifications(self):
        """Test VerificationResult without suggested modifications."""
        result = VerificationResult(
            should_execute=True,
            confidence=0.9,
            assumptions=[],
            risks=[],
            verification_reasoning="Perfect"
        )

        assert result.suggested_modifications is None


class TestGetStats:
    """Test verifier statistics."""

    def test_get_stats(self):
        """Test get_stats method."""
        mock_runner = MockLLMRunner("")
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4o-mini")

        stats = verifier.get_stats()

        assert stats["model"] == "openai/gpt-4o-mini"
        assert stats["status"] == "active"


class TestReprMethod:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        mock_runner = MockLLMRunner("")
        verifier = CoVeVerifier(mock_runner, "openai/gpt-4")

        repr_str = repr(verifier)

        assert "CoVeVerifier" in repr_str
        assert "openai/gpt-4" in repr_str


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
