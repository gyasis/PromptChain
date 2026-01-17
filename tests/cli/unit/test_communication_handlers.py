"""
Unit tests for communication handlers module.

Tests the @cli_communication_handler decorator and HandlerRegistry
for agent-to-agent messaging (US4 - T052-T053 equivalent).
"""

import pytest
from typing import Dict, Any, Optional
from promptchain.cli.communication.handlers import (
    MessageType,
    CommunicationHandler,
    HandlerRegistry,
    cli_communication_handler,
    get_handler_registry,
)


class TestMessageType:
    """Test MessageType enum."""

    def test_message_type_values(self):
        """Test all message types are defined."""
        assert MessageType.REQUEST == "request"
        assert MessageType.RESPONSE == "response"
        assert MessageType.BROADCAST == "broadcast"
        assert MessageType.DELEGATION == "delegation"
        assert MessageType.STATUS == "status"

    def test_message_type_from_string(self):
        """Test creating MessageType from string."""
        assert MessageType("request") == MessageType.REQUEST
        assert MessageType("broadcast") == MessageType.BROADCAST


class TestCommunicationHandler:
    """Test CommunicationHandler dataclass."""

    def test_handler_creation(self):
        """Test creating a handler."""
        def dummy_func():
            pass

        handler = CommunicationHandler(
            func=dummy_func,
            name="test_handler",
            message_types={MessageType.REQUEST},
            senders={"agent1"},
            receivers={"agent2"},
            priority=10
        )

        assert handler.func == dummy_func
        assert handler.name == "test_handler"
        assert MessageType.REQUEST in handler.message_types
        assert "agent1" in handler.senders
        assert "agent2" in handler.receivers
        assert handler.priority == 10

    def test_handler_matches_all_filters(self):
        """Test handler matching with all filters specified."""
        handler = CommunicationHandler(
            func=lambda: None,
            name="test",
            message_types={MessageType.REQUEST},
            senders={"supervisor"},
            receivers={"worker"}
        )

        assert handler.matches(MessageType.REQUEST, "supervisor", "worker")
        assert not handler.matches(MessageType.RESPONSE, "supervisor", "worker")
        assert not handler.matches(MessageType.REQUEST, "other", "worker")
        assert not handler.matches(MessageType.REQUEST, "supervisor", "other")

    def test_handler_matches_empty_filters(self):
        """Test handler with empty filters matches everything."""
        handler = CommunicationHandler(
            func=lambda: None,
            name="test",
            message_types=set(),
            senders=set(),
            receivers=set()
        )

        assert handler.matches(MessageType.REQUEST, "any_sender", "any_receiver")
        assert handler.matches(MessageType.BROADCAST, "agent1", "agent2")

    def test_handler_matches_partial_filters(self):
        """Test handler with only some filters specified."""
        handler = CommunicationHandler(
            func=lambda: None,
            name="test",
            message_types={MessageType.REQUEST, MessageType.RESPONSE},
            senders=set(),  # Match all senders
            receivers={"target_agent"}
        )

        assert handler.matches(MessageType.REQUEST, "any_sender", "target_agent")
        assert handler.matches(MessageType.RESPONSE, "another_sender", "target_agent")
        assert not handler.matches(MessageType.BROADCAST, "sender", "target_agent")
        assert not handler.matches(MessageType.REQUEST, "sender", "wrong_agent")


class TestHandlerRegistry:
    """Test HandlerRegistry singleton and functionality."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        HandlerRegistry.reset()
        yield
        HandlerRegistry.reset()

    def test_singleton_pattern(self):
        """Test HandlerRegistry is a singleton."""
        registry1 = HandlerRegistry()
        registry2 = HandlerRegistry()
        assert registry1 is registry2

    def test_register_handler(self):
        """Test registering a handler."""
        registry = HandlerRegistry()

        def test_func():
            return "result"

        handler = CommunicationHandler(
            func=test_func,
            name="test_handler",
            message_types={MessageType.REQUEST}
        )

        registry.register(handler)
        assert len(registry.handlers) == 1
        assert registry.handlers[0].name == "test_handler"

    def test_register_multiple_handlers_sorted_by_priority(self):
        """Test handlers are sorted by priority (highest first)."""
        registry = HandlerRegistry()

        h1 = CommunicationHandler(func=lambda: 1, name="low", priority=1)
        h2 = CommunicationHandler(func=lambda: 2, name="high", priority=10)
        h3 = CommunicationHandler(func=lambda: 3, name="medium", priority=5)

        registry.register(h1)
        registry.register(h2)
        registry.register(h3)

        handlers = registry.handlers
        assert handlers[0].name == "high"
        assert handlers[1].name == "medium"
        assert handlers[2].name == "low"

    def test_unregister_handler(self):
        """Test unregistering a handler."""
        registry = HandlerRegistry()

        handler = CommunicationHandler(func=lambda: None, name="test")
        registry.register(handler)
        assert len(registry.handlers) == 1

        result = registry.unregister("test")
        assert result is True
        assert len(registry.handlers) == 0

    def test_unregister_nonexistent_handler(self):
        """Test unregistering a handler that doesn't exist."""
        registry = HandlerRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_matching_handlers(self):
        """Test getting handlers that match criteria."""
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
            message_types=set()  # Matches all
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

    @pytest.mark.asyncio
    async def test_dispatch_sync_handler(self):
        """Test dispatching to synchronous handler."""
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

    @pytest.mark.asyncio
    async def test_dispatch_async_handler(self):
        """Test dispatching to async handler."""
        registry = HandlerRegistry()

        async def async_handler(payload, sender, receiver):
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

    @pytest.mark.asyncio
    async def test_dispatch_multiple_handlers(self):
        """Test dispatching to multiple matching handlers."""
        registry = HandlerRegistry()

        def handler1(payload, sender, receiver):
            return "h1"

        def handler2(payload, sender, receiver):
            return "h2"

        registry.register(CommunicationHandler(
            func=handler1,
            name="h1",
            message_types={MessageType.REQUEST}
        ))
        registry.register(CommunicationHandler(
            func=handler2,
            name="h2",
            message_types={MessageType.REQUEST}
        ))

        results = await registry.dispatch(
            MessageType.REQUEST,
            "sender",
            "receiver",
            {}
        )

        assert len(results) == 2
        assert "h1" in results
        assert "h2" in results

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception(self):
        """Test that dispatch continues even if handler raises exception."""
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
        assert results[0]["error"] == "Handler failed"
        assert results[0]["handler"] == "failing"
        assert results[1] == "success"


class TestCliCommunicationHandlerDecorator:
    """Test @cli_communication_handler decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        HandlerRegistry.reset()
        yield
        HandlerRegistry.reset()

    def test_decorator_basic_registration(self):
        """Test basic decorator usage registers handler."""
        @cli_communication_handler(type=MessageType.REQUEST)
        def my_handler(payload, sender, receiver):
            return "handled"

        registry = get_handler_registry()
        handlers = registry.handlers
        assert len(handlers) == 1
        assert handlers[0].name == "my_handler"
        assert MessageType.REQUEST in handlers[0].message_types

    def test_decorator_with_custom_name(self):
        """Test decorator with custom handler name."""
        @cli_communication_handler(
            type=MessageType.STATUS,
            name="custom_status_handler"
        )
        def status_func(payload, sender, receiver):
            return "status"

        registry = get_handler_registry()
        assert registry.handlers[0].name == "custom_status_handler"

    def test_decorator_with_multiple_types(self):
        """Test decorator with multiple message types."""
        @cli_communication_handler(
            types=[MessageType.REQUEST, MessageType.RESPONSE]
        )
        def multi_type_handler(payload, sender, receiver):
            return "multi"

        registry = get_handler_registry()
        handler = registry.handlers[0]
        assert MessageType.REQUEST in handler.message_types
        assert MessageType.RESPONSE in handler.message_types

    def test_decorator_with_sender_filter(self):
        """Test decorator with sender filter."""
        @cli_communication_handler(
            type=MessageType.REQUEST,
            sender="supervisor"
        )
        def supervisor_handler(payload, sender, receiver):
            return "from_supervisor"

        registry = get_handler_registry()
        handler = registry.handlers[0]
        assert "supervisor" in handler.senders

    def test_decorator_with_multiple_senders(self):
        """Test decorator with multiple sender filters."""
        @cli_communication_handler(
            type=MessageType.DELEGATION,
            senders=["manager", "coordinator"]
        )
        def delegation_handler(payload, sender, receiver):
            return "delegated"

        registry = get_handler_registry()
        handler = registry.handlers[0]
        assert "manager" in handler.senders
        assert "coordinator" in handler.senders

    def test_decorator_with_receiver_filter(self):
        """Test decorator with receiver filter."""
        @cli_communication_handler(
            type=MessageType.STATUS,
            receiver="monitor"
        )
        def monitor_handler(payload, sender, receiver):
            return "monitoring"

        registry = get_handler_registry()
        handler = registry.handlers[0]
        assert "monitor" in handler.receivers

    def test_decorator_with_priority(self):
        """Test decorator with priority setting."""
        @cli_communication_handler(
            type=MessageType.REQUEST,
            priority=100
        )
        def high_priority_handler(payload, sender, receiver):
            return "high_priority"

        registry = get_handler_registry()
        assert registry.handlers[0].priority == 100

    def test_decorator_preserves_function(self):
        """Test decorator returns original function."""
        def original_func(payload, sender, receiver):
            return "original"

        decorated = cli_communication_handler(type=MessageType.REQUEST)(original_func)

        # Function should still be callable directly
        assert decorated({"test": 1}, "s", "r") == "original"

    def test_decorator_async_function(self):
        """Test decorator works with async functions."""
        @cli_communication_handler(type=MessageType.REQUEST)
        async def async_handler(payload, sender, receiver):
            return "async_result"

        registry = get_handler_registry()
        handler = registry.handlers[0]
        # Verify it's recognized as async
        import asyncio
        assert asyncio.iscoroutinefunction(handler.func)

    def test_multiple_decorators_register_multiple_handlers(self):
        """Test multiple decorated functions register independently."""
        @cli_communication_handler(type=MessageType.REQUEST)
        def handler1(payload, sender, receiver):
            return "h1"

        @cli_communication_handler(type=MessageType.RESPONSE)
        def handler2(payload, sender, receiver):
            return "h2"

        registry = get_handler_registry()
        assert len(registry.handlers) == 2
        names = {h.name for h in registry.handlers}
        assert "handler1" in names
        assert "handler2" in names

    @pytest.mark.asyncio
    async def test_decorated_handler_invoked_on_dispatch(self):
        """Test decorated handler is invoked during dispatch."""
        results = []

        @cli_communication_handler(
            type=MessageType.REQUEST,
            sender="agent1"
        )
        def track_handler(payload, sender, receiver):
            results.append({"sender": sender, "data": payload})
            return "tracked"

        registry = get_handler_registry()
        dispatch_results = await registry.dispatch(
            MessageType.REQUEST,
            "agent1",
            "agent2",
            {"value": 123}
        )

        assert len(results) == 1
        assert results[0]["sender"] == "agent1"
        assert results[0]["data"]["value"] == 123
        assert dispatch_results[0] == "tracked"

    def test_decorator_all_filters_combined(self):
        """Test decorator with all filter types combined."""
        @cli_communication_handler(
            types=[MessageType.REQUEST, MessageType.DELEGATION],
            senders=["supervisor", "manager"],
            receivers=["worker1", "worker2"],
            priority=50,
            name="complex_handler"
        )
        def complex_handler(payload, sender, receiver):
            return "complex"

        registry = get_handler_registry()
        handler = registry.handlers[0]

        assert handler.name == "complex_handler"
        assert handler.priority == 50
        assert MessageType.REQUEST in handler.message_types
        assert MessageType.DELEGATION in handler.message_types
        assert "supervisor" in handler.senders
        assert "manager" in handler.senders
        assert "worker1" in handler.receivers
        assert "worker2" in handler.receivers


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        HandlerRegistry.reset()
        yield
        HandlerRegistry.reset()

    @pytest.mark.asyncio
    async def test_supervisor_worker_pattern(self):
        """Test supervisor delegating work to workers."""
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

    @pytest.mark.asyncio
    async def test_broadcast_to_all_agents(self):
        """Test broadcast message handled by multiple agents."""
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
            {"message": "global_update"}
        )

        assert len(responses) == 2
        assert "agent1" in responses
        assert "agent2" in responses
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test handlers execute in priority order."""
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
        def medium_priority(payload, sender, receiver):
            execution_order.append("medium")
            return "medium"

        registry = get_handler_registry()
        await registry.dispatch(
            MessageType.REQUEST,
            "sender",
            "receiver",
            {}
        )

        assert execution_order == ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_request_response_pattern(self):
        """Test request-response messaging pattern."""
        @cli_communication_handler(
            type=MessageType.REQUEST,
            receiver="data_agent"
        )
        def handle_data_request(payload, sender, receiver):
            query = payload.get("query")
            # Simulate data retrieval
            return {"data": f"Result for {query}"}

        registry = get_handler_registry()

        # Send request
        results = await registry.dispatch(
            MessageType.REQUEST,
            "client_agent",
            "data_agent",
            {"query": "SELECT * FROM users"}
        )

        assert len(results) == 1
        assert "Result for SELECT * FROM users" in results[0]["data"]


def test_get_handler_registry_returns_singleton():
    """Test get_handler_registry() returns the singleton instance."""
    HandlerRegistry.reset()

    registry1 = get_handler_registry()
    registry2 = get_handler_registry()
    registry3 = HandlerRegistry()

    assert registry1 is registry2
    assert registry1 is registry3
