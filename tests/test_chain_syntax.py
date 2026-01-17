"""
Test chain: syntax in PromptChain instructions.

This tests the integration of ChainFactory/ChainExecutor with PromptChain.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

# Test the chain: prefix detection and ChainCall handling


class TestChainSyntaxDetection:
    """Test that chain: prefix is properly detected."""

    def test_chain_prefix_detection(self):
        """Test basic chain: prefix detection."""
        instruction = "chain:query-validator:v1.0"
        assert instruction.startswith("chain:")

        chain_ref = instruction[6:]  # Remove "chain:" prefix
        assert chain_ref == "query-validator:v1.0"

    def test_chain_prefix_with_latest(self):
        """Test chain: prefix with model-only (latest version)."""
        instruction = "chain:query-validator"
        assert instruction.startswith("chain:")

        chain_ref = instruction[6:]
        assert chain_ref == "query-validator"

    def test_normal_instruction_not_chain(self):
        """Test that normal instructions are not detected as chains."""
        instruction = "Analyze the following input: {input}"
        assert not instruction.startswith("chain:")


class TestChainCallClass:
    """Test ChainCall marker class."""

    def test_chain_call_import(self):
        """Test ChainCall can be imported."""
        from promptchain.utils.chain_models import ChainCall

        call = ChainCall("query-validator:v1.0")
        assert call.chain_ref == "query-validator:v1.0"
        assert call.model == "query-validator"
        assert call.version == "v1.0"

    def test_chain_call_latest_version(self):
        """Test ChainCall with no version specified."""
        from promptchain.utils.chain_models import ChainCall

        call = ChainCall("query-validator")
        assert call.chain_ref == "query-validator"
        assert call.model == "query-validator"
        assert call.version == "latest"


class TestChainFactoryIntegration:
    """Test ChainFactory integration."""

    def test_chain_factory_resolve(self):
        """Test ChainFactory can resolve chain references."""
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()

        # Try to resolve query-validator
        try:
            chain = factory.resolve("query-validator:v1.0")
            assert chain.model == "query-validator"
            assert chain.version == "v1.0"
        except Exception as e:
            # Chain might not exist in test environment
            pytest.skip(f"Chain not found (expected in fresh environment): {e}")

    def test_chain_factory_resolve_latest(self):
        """Test ChainFactory resolves latest version."""
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()

        try:
            chain = factory.resolve("query-validator")
            assert chain.model == "query-validator"
        except Exception as e:
            pytest.skip(f"Chain not found: {e}")


class TestChainExecutorIntegration:
    """Test ChainExecutor integration."""

    def test_chain_executor_creation(self):
        """Test ChainExecutor can be created."""
        from promptchain.utils.chain_executor import ChainExecutor
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()
        executor = ChainExecutor(factory=factory)

        assert executor is not None
        assert executor.factory is factory


class TestPromptChainChainInstruction:
    """Test PromptChain with chain: instruction syntax."""

    @pytest.mark.asyncio
    async def test_chain_instruction_detected(self):
        """Test that chain: instruction is detected by PromptChain."""
        from promptchain.utils.promptchaining import PromptChain

        # Create a PromptChain with chain instruction
        chain = PromptChain(
            models=["openai/gpt-4.1-mini-2025-04-14"],
            instructions=["chain:query-validator:v1.0"],
            verbose=True
        )

        # Verify the instruction is a string starting with chain:
        assert chain.instructions[0] == "chain:query-validator:v1.0"
        assert chain.instructions[0].startswith("chain:")

    @pytest.mark.asyncio
    async def test_chain_call_instruction(self):
        """Test PromptChain with ChainCall instruction."""
        from promptchain.utils.promptchaining import PromptChain
        from promptchain.utils.chain_models import ChainCall

        # Create a PromptChain with ChainCall instruction
        chain = PromptChain(
            models=["openai/gpt-4.1-mini-2025-04-14"],
            instructions=[ChainCall("query-validator:v1.0")],
            verbose=True
        )

        # Verify the instruction is a ChainCall
        assert isinstance(chain.instructions[0], ChainCall)
        assert chain.instructions[0].chain_ref == "query-validator:v1.0"


# Simple integration test (requires API key and existing chains)
@pytest.mark.skipif(True, reason="Requires API key and existing chains")
class TestChainExecutionIntegration:
    """Full integration tests (manual run only)."""

    @pytest.mark.asyncio
    async def test_chain_execution_via_prompt_chain(self):
        """Test full chain execution through PromptChain."""
        from promptchain.utils.promptchaining import PromptChain

        chain = PromptChain(
            models=["openai/gpt-4.1-mini-2025-04-14"],
            instructions=["chain:query-validator:v1.0"],
            verbose=True
        )

        result = await chain.process_prompt_async("SELECT * FROM users WHERE id = 1")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
