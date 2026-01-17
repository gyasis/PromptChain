#!/usr/bin/env python3
"""
Standalone verification script for communication handlers module.

Verifies all requirements for US4 - Agent-to-Agent Messaging:
1. MessageType enum with correct values
2. @cli_communication_handler decorator
3. HandlerRegistry singleton with filtering and routing
"""

import sys
import asyncio
sys.path.insert(0, '/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication')

from handlers import (
    MessageType,
    CommunicationHandler,
    HandlerRegistry,
    cli_communication_handler,
    get_handler_registry
)


def test_message_type_enum():
    """Test MessageType enum has all required values."""
    print("Testing MessageType enum...")
    assert MessageType.REQUEST == "request"
    assert MessageType.RESPONSE == "response"
    assert MessageType.BROADCAST == "broadcast"
    assert MessageType.DELEGATION == "delegation"
    assert MessageType.STATUS == "status"
    print("✓ MessageType enum has all required values")


def test_communication_handler_filtering():
    """Test CommunicationHandler filtering logic."""
    print("\nTesting CommunicationHandler filtering...")

    # Test with all filters
    handler = CommunicationHandler(
        func=lambda: None,
        name="test",
        message_types={MessageType.REQUEST},
        senders={"agent1"},
        receivers={"agent2"}
    )

    assert handler.matches(MessageType.REQUEST, "agent1", "agent2")
    assert not handler.matches(MessageType.RESPONSE, "agent1", "agent2")
    assert not handler.matches(MessageType.REQUEST, "other", "agent2")
    print("✓ CommunicationHandler filtering works correctly")

    # Test with empty filters (match all)
    handler_all = CommunicationHandler(
        func=lambda: None,
        name="all",
        message_types=set(),
        senders=set(),
        receivers=set()
    )

    assert handler_all.matches(MessageType.REQUEST, "anyone", "anyone")
    assert handler_all.matches(MessageType.BROADCAST, "any1", "any2")
    print("✓ Empty filters match all messages")


def test_handler_registry_singleton():
    """Test HandlerRegistry singleton pattern."""
    print("\nTesting HandlerRegistry singleton...")
    HandlerRegistry.reset()

    registry1 = HandlerRegistry()
    registry2 = HandlerRegistry()
    registry3 = get_handler_registry()

    assert registry1 is registry2
    assert registry1 is registry3
    print("✓ HandlerRegistry is a proper singleton")


def test_handler_registration():
    """Test handler registration and priority sorting."""
    print("\nTesting handler registration...")
    HandlerRegistry.reset()
    registry = HandlerRegistry()

    # Register handlers with different priorities
    h1 = CommunicationHandler(func=lambda: 1, name="low", priority=1)
    h2 = CommunicationHandler(func=lambda: 2, name="high", priority=10)
    h3 = CommunicationHandler(func=lambda: 3, name="med", priority=5)

    registry.register(h1)
    registry.register(h2)
    registry.register(h3)

    handlers = registry.handlers
    assert len(handlers) == 3
    assert handlers[0].name == "high"
    assert handlers[1].name == "med"
    assert handlers[2].name == "low"
    print("✓ Handlers registered and sorted by priority")


def test_handler_unregistration():
    """Test handler unregistration."""
    print("\nTesting handler unregistration...")
    HandlerRegistry.reset()
    registry = HandlerRegistry()

    handler = CommunicationHandler(func=lambda: None, name="test")
    registry.register(handler)
    assert len(registry.handlers) == 1

    result = registry.unregister("test")
    assert result is True
    assert len(registry.handlers) == 0

    result = registry.unregister("nonexistent")
    assert result is False
    print("✓ Handler unregistration works")


def test_get_matching_handlers():
    """Test getting matching handlers."""
    print("\nTesting handler matching...")
    HandlerRegistry.reset()
    registry = HandlerRegistry()

    h1 = CommunicationHandler(
        func=lambda: 1,
        name="request_handler",
        message_types={MessageType.REQUEST},
        senders={"supervisor"}
    )
    h2 = CommunicationHandler(
        func=lambda: 2,
        name="broadcast_handler",
        message_types={MessageType.BROADCAST}
    )
    h3 = CommunicationHandler(
        func=lambda: 3,
        name="all_handler",
        message_types=set()
    )

    registry.register(h1)
    registry.register(h2)
    registry.register(h3)

    # Should match h1 and h3
    matches = registry.get_matching_handlers(
        MessageType.REQUEST, "supervisor", "worker"
    )
    assert len(matches) == 2
    names = {h.name for h in matches}
    assert "request_handler" in names
    assert "all_handler" in names
    print("✓ Handler matching filters work correctly")


async def test_dispatch_sync_handler():
    """Test dispatching to synchronous handler."""
    print("\nTesting sync handler dispatch...")
    HandlerRegistry.reset()
    registry = HandlerRegistry()

    def sync_handler(payload, sender, receiver):
        return f"Processed: {payload['data']}"

    handler = CommunicationHandler(
        func=sync_handler,
        name="sync_test",
        message_types={MessageType.REQUEST}
    )
    registry.register(handler)

    results = await registry.dispatch(
        MessageType.REQUEST,
        "agent1",
        "agent2",
        {"data": "test"}
    )

    assert len(results) == 1
    assert results[0] == "Processed: test"
    print("✓ Sync handler dispatch works")


async def test_dispatch_async_handler():
    """Test dispatching to async handler."""
    print("\nTesting async handler dispatch...")
    HandlerRegistry.reset()
    registry = HandlerRegistry()

    async def async_handler(payload, sender, receiver):
        await asyncio.sleep(0.001)
        return f"Async: {payload['value']}"

    handler = CommunicationHandler(
        func=async_handler,
        name="async_test",
        message_types={MessageType.STATUS}
    )
    registry.register(handler)

    results = await registry.dispatch(
        MessageType.STATUS,
        "agent1",
        "agent2",
        {"value": 42}
    )

    assert len(results) == 1
    assert results[0] == "Async: 42"
    print("✓ Async handler dispatch works")


async def test_dispatch_exception_handling():
    """Test that dispatch continues on handler exception."""
    print("\nTesting exception handling...")
    HandlerRegistry.reset()
    registry = HandlerRegistry()

    def failing_handler(payload, sender, receiver):
        raise ValueError("Handler failed")

    def working_handler(payload, sender, receiver):
        return "success"

    registry.register(CommunicationHandler(
        func=failing_handler,
        name="failing",
        message_types={MessageType.REQUEST},
        priority=10
    ))
    registry.register(CommunicationHandler(
        func=working_handler,
        name="working",
        message_types={MessageType.REQUEST},
        priority=5
    ))

    results = await registry.dispatch(
        MessageType.REQUEST,
        "sender",
        "receiver",
        {}
    )

    assert len(results) == 2
    assert "error" in results[0]
    assert results[0]["handler"] == "failing"
    assert results[1] == "success"
    print("✓ Exception handling works (system continues)")


def test_decorator_basic():
    """Test basic decorator usage."""
    print("\nTesting @cli_communication_handler decorator...")
    HandlerRegistry.reset()

    @cli_communication_handler(type=MessageType.REQUEST)
    def my_handler(payload, sender, receiver):
        return "handled"

    registry = get_handler_registry()
    handlers = registry.handlers
    assert len(handlers) == 1
    assert handlers[0].name == "my_handler"
    assert MessageType.REQUEST in handlers[0].message_types
    print("✓ Basic decorator registration works")


def test_decorator_all_options():
    """Test decorator with all options."""
    print("\nTesting decorator with all options...")
    HandlerRegistry.reset()

    @cli_communication_handler(
        types=[MessageType.REQUEST, MessageType.RESPONSE],
        senders=["agent1", "agent2"],
        receivers=["target"],
        priority=100,
        name="custom_name"
    )
    def complex_handler(payload, sender, receiver):
        return "complex"

    registry = get_handler_registry()
    handler = registry.handlers[0]

    assert handler.name == "custom_name"
    assert handler.priority == 100
    assert MessageType.REQUEST in handler.message_types
    assert MessageType.RESPONSE in handler.message_types
    assert "agent1" in handler.senders
    assert "agent2" in handler.senders
    assert "target" in handler.receivers
    print("✓ Decorator with all options works")


def test_decorator_preserves_function():
    """Test decorator returns callable function."""
    print("\nTesting decorator preserves function...")
    HandlerRegistry.reset()

    @cli_communication_handler(type=MessageType.REQUEST)
    def test_func(payload, sender, receiver):
        return "original"

    result = test_func({"test": 1}, "s", "r")
    assert result == "original"
    print("✓ Decorator preserves original function")


async def test_integration_supervisor_worker():
    """Test realistic supervisor-worker pattern."""
    print("\nTesting supervisor-worker integration...")
    HandlerRegistry.reset()

    work_log = []

    @cli_communication_handler(
        type=MessageType.DELEGATION,
        sender="supervisor",
        receiver="worker"
    )
    def handle_work(payload, sender, receiver):
        work_log.append(payload["task"])
        return {"status": "completed", "task": payload["task"]}

    registry = get_handler_registry()
    results = await registry.dispatch(
        MessageType.DELEGATION,
        "supervisor",
        "worker",
        {"task": "process_data"}
    )

    assert len(work_log) == 1
    assert work_log[0] == "process_data"
    assert results[0]["status"] == "completed"
    print("✓ Supervisor-worker pattern works")


async def test_integration_broadcast():
    """Test broadcast to multiple handlers."""
    print("\nTesting broadcast integration...")
    HandlerRegistry.reset()

    responses = []

    @cli_communication_handler(type=MessageType.BROADCAST)
    def agent1_handler(payload, sender, receiver):
        responses.append("agent1")
        return "agent1_ack"

    @cli_communication_handler(type=MessageType.BROADCAST)
    def agent2_handler(payload, sender, receiver):
        responses.append("agent2")
        return "agent2_ack"

    registry = get_handler_registry()
    results = await registry.dispatch(
        MessageType.BROADCAST,
        "coordinator",
        "*",
        {"message": "update"}
    )

    assert len(responses) == 2
    assert "agent1" in responses
    assert "agent2" in responses
    assert len(results) == 2
    print("✓ Broadcast pattern works")


async def test_integration_priority_ordering():
    """Test priority execution order."""
    print("\nTesting priority ordering...")
    HandlerRegistry.reset()

    execution_order = []

    @cli_communication_handler(type=MessageType.REQUEST, priority=1)
    def low_priority(payload, sender, receiver):
        execution_order.append("low")
        return "low"

    @cli_communication_handler(type=MessageType.REQUEST, priority=100)
    def high_priority(payload, sender, receiver):
        execution_order.append("high")
        return "high"

    @cli_communication_handler(type=MessageType.REQUEST, priority=50)
    def med_priority(payload, sender, receiver):
        execution_order.append("med")
        return "med"

    registry = get_handler_registry()
    await registry.dispatch(
        MessageType.REQUEST,
        "sender",
        "receiver",
        {}
    )

    assert execution_order == ["high", "med", "low"]
    print("✓ Priority ordering works correctly")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Communication Handlers Module Verification")
    print("=" * 70)

    # Synchronous tests
    test_message_type_enum()
    test_communication_handler_filtering()
    test_handler_registry_singleton()
    test_handler_registration()
    test_handler_unregistration()
    test_get_matching_handlers()
    test_decorator_basic()
    test_decorator_all_options()
    test_decorator_preserves_function()

    # Async tests
    print("\nRunning async tests...")
    asyncio.run(test_dispatch_sync_handler())
    asyncio.run(test_dispatch_async_handler())
    asyncio.run(test_dispatch_exception_handling())
    asyncio.run(test_integration_supervisor_worker())
    asyncio.run(test_integration_broadcast())
    asyncio.run(test_integration_priority_ordering())

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nVerified functionality:")
    print("1. ✓ MessageType enum (request, response, broadcast, delegation, status)")
    print("2. ✓ @cli_communication_handler decorator with filtering")
    print("3. ✓ HandlerRegistry singleton")
    print("4. ✓ Handler routing by sender, receiver, message_type")
    print("5. ✓ Priority-based handler execution")
    print("6. ✓ Async/sync handler support")
    print("7. ✓ Exception handling (system continues)")
    print("8. ✓ Integration patterns (supervisor-worker, broadcast, priority)")
    print("\nModule is complete and ready for use!")


if __name__ == "__main__":
    main()
