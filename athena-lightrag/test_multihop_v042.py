#!/usr/bin/env python3
"""
Test multi-hop reasoning with PromptChain v0.4.2 compatibility check
"""
import asyncio
import logging
import sys
import os

sys.path.insert(0, '/home/gyasis/Documents/code/PromptChain')
sys.path.insert(0, '/home/gyasis/Documents/code/PromptChain/athena-lightrag')

# Enable verbose logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_multi_hop():
    """Test multi-hop reasoning and identify v0.4.2 compatibility issues."""

    logger.info("=" * 80)
    logger.info("PromptChain v0.4.2 Compatibility Test for Multi-hop Reasoning")
    logger.info("=" * 80)

    try:
        # Import with diagnostics
        logger.info("1. Importing AgenticStepProcessor...")
        from promptchain.utils.agentic_step_processor import AgenticStepProcessor
        logger.info("✅ AgenticStepProcessor imported successfully")

        # Check for new observability features
        logger.info("\n2. Checking for v0.4.2 observability features...")

        # Check ExecutionEvent system
        try:
            from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
            logger.info("✅ ExecutionEvent system available")
            if hasattr(ExecutionEventType, 'AGENTIC_INTERNAL_STEP'):
                logger.info("✅ AGENTIC_INTERNAL_STEP event type exists")
            else:
                logger.warning("⚠️  AGENTIC_INTERNAL_STEP event type NOT found - may be missing observability")
        except ImportError as e:
            logger.error(f"❌ ExecutionEvent system not found: {e}")

        # Check callback support
        logger.info("\n3. Checking AgenticStepProcessor initialization...")
        test_processor = AgenticStepProcessor(
            objective="Test objective",
            max_internal_steps=3,
            history_mode="progressive"
        )
        logger.info("✅ AgenticStepProcessor created without errors")

        # Check for callback_manager attribute
        if hasattr(test_processor, 'callback_manager'):
            logger.info("✅ callback_manager attribute exists")
        else:
            logger.warning("⚠️  No callback_manager attribute - callbacks may not work")

        # Import our multi-hop implementation
        logger.info("\n4. Testing agentic_lightrag imports...")
        from agentic_lightrag import AgenticLightRAG, create_agentic_lightrag
        logger.info("✅ AgenticLightRAG imported successfully")

        # Create instance
        logger.info("\n5. Creating AgenticLightRAG instance...")
        agentic_rag = create_agentic_lightrag(
            working_dir="/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db",
            verbose=True
        )
        logger.info("✅ AgenticLightRAG instance created")

        # Test tool registration
        logger.info("\n6. Testing tool registration...")
        from promptchain import PromptChain
        chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Test: {input}"],
            verbose=True
        )

        # Register tools
        agentic_rag.tools_provider.register_tool_functions(chain)
        logger.info(f"✅ Registered {len(chain.local_tool_functions)} tool functions")

        # Check for async compatibility
        logger.info("\n7. Checking async tool compatibility...")
        import inspect
        for name, func in chain.local_tool_functions.items():
            is_async = inspect.iscoroutinefunction(func)
            logger.info(f"  - {name}: {'async' if is_async else 'sync'}")

        # Test simple multi-hop execution
        logger.info("\n8. Testing multi-hop execution...")
        query = "List all schemas in the athena database"

        # Note: execute_multi_hop_reasoning doesn't accept max_internal_steps directly
        # It's set during AgenticLightRAG initialization
        result = await agentic_rag.execute_multi_hop_reasoning(
            query=query,
            timeout_seconds=30
        )

        if result["success"]:
            logger.info("✅ Multi-hop reasoning executed successfully")
            logger.info(f"   - Steps executed: {len(result['reasoning_steps'])}")
            logger.info(f"   - Contexts accumulated: {len(result['accumulated_contexts'])}")
            logger.info(f"   - Result preview: {result['result'][:200]}...")
        else:
            logger.error(f"❌ Multi-hop reasoning failed: {result.get('error', 'Unknown')}")

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("Test Complete - Check logs above for compatibility issues")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_multi_hop())