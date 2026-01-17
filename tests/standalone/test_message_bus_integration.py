#!/usr/bin/env python3
"""
Integration test for MessageBus implementation.

Tests all core functionality without requiring external dependencies.
Run with: python3 test_message_bus_integration.py
"""

import asyncio
import sys
from pathlib import Path

# Import directly to avoid dependency issues
sys.path.insert(0, str(Path(__file__).parent))

from promptchain.cli.communication.handlers import (
    HandlerRegistry,
    MessageType,
    cli_communication_handler,
    get_handler_registry
)
from promptchain.cli.communication.message_bus import MessageBus, Message


class TestMessageBus:
    """Test suite for MessageBus functionality."""

    def __init__(self):
        self.results = []
        self.errors = []

    def test(self, name, condition, error_msg=""):
        """Record test result."""
        if condition:
            self.results.append((name, True))
            print(f"  ✅ {name}")
        else:
            self.results.append((name, False))
            self.errors.append(f"{name}: {error_msg}")
            print(f"  ❌ {name}: {error_msg}")

    async def run_all_tests(self):
        """Run all integration tests."""
        print("=" * 60)
        print("MESSAGE BUS INTEGRATION TESTS")
        print("=" * 60)

        await self.test_message_creation()
        await self.test_message_serialization()
        await self.test_basic_send()
        await self.test_broadcast()
        await self.test_handler_integration()
        await self.test_convenience_methods()
        await self.test_history_tracking()
        await self.test_history_filtering()
        await self.test_activity_logging()
        await self.test_error_handling()

        self.print_summary()
        return len(self.errors) == 0

    async def test_message_creation(self):
        """Test Message dataclass creation and fields."""
        print("\n📦 Test: Message Creation")

        msg = Message.create(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.REQUEST,
            payload={"data": "test"}
        )

        self.test("Message has ID", bool(msg.message_id))
        self.test("Message has sender", msg.sender == "agent1")
        self.test("Message has receiver", msg.receiver == "agent2")
        self.test("Message has type", msg.type == MessageType.REQUEST)
        self.test("Message has payload", msg.payload == {"data": "test"})
        self.test("Message has timestamp", msg.timestamp > 0)
        self.test("Message delivered flag", msg.delivered == False)

    async def test_message_serialization(self):
        """Test Message serialization and deserialization."""
        print("\n📦 Test: Message Serialization")

        msg = Message.create(
            sender="test",
            receiver="target",
            message_type=MessageType.STATUS,
            payload={"status": "ready"}
        )

        # Serialize
        data = msg.to_dict()
        self.test("to_dict returns dict", isinstance(data, dict))
        self.test("Serialized has message_id", "message_id" in data)
        self.test("Serialized has sender", data["sender"] == "test")

        # Deserialize
        restored = Message.from_dict(data)
        self.test("from_dict creates Message", isinstance(restored, Message))
        self.test("Restored ID matches", restored.message_id == msg.message_id)
        self.test("Restored sender matches", restored.sender == msg.sender)
        self.test("Restored type matches", restored.type == msg.type)

    async def test_basic_send(self):
        """Test basic message sending."""
        print("\n🚌 Test: Basic Send")

        # Reset handlers
        get_handler_registry().reset()

        bus = MessageBus(session_id="test")

        msg = await bus.send(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.REQUEST,
            payload={"task": "process"}
        )

        self.test("Send returns Message", isinstance(msg, Message))
        self.test("Send creates ID", bool(msg.message_id))
        self.test("Send sets sender", msg.sender == "agent1")
        self.test("Send sets receiver", msg.receiver == "agent2")

    async def test_broadcast(self):
        """Test broadcast messaging."""
        print("\n📢 Test: Broadcast")

        get_handler_registry().reset()

        bus = MessageBus(session_id="test")

        msg = await bus.broadcast(
            sender="coordinator",
            payload={"announcement": "status update"}
        )

        self.test("Broadcast returns Message", isinstance(msg, Message))
        self.test("Broadcast sets receiver to *", msg.receiver == "*")
        self.test("Broadcast type is BROADCAST", msg.type == MessageType.BROADCAST)

    async def test_handler_integration(self):
        """Test integration with HandlerRegistry."""
        print("\n🔗 Test: Handler Integration")

        get_handler_registry().reset()

        results = []

        @cli_communication_handler(type=MessageType.REQUEST, receiver="worker")
        async def test_handler(payload, sender, receiver):
            results.append(payload)
            return {"status": "processed"}

        bus = MessageBus(session_id="test")

        msg = await bus.send(
            sender="boss",
            receiver="worker",
            message_type=MessageType.REQUEST,
            payload={"job": "task1"}
        )

        self.test("Handler was called", len(results) == 1)
        self.test("Handler received payload", results[0]["job"] == "task1")
        self.test("Message marked as delivered", msg.delivered == True)

        get_handler_registry().reset()

    async def test_convenience_methods(self):
        """Test convenience methods."""
        print("\n🛠️  Test: Convenience Methods")

        get_handler_registry().reset()

        bus = MessageBus(session_id="test")

        # Request
        req = await bus.request("a", "b", {"req": "data"})
        self.test("request() creates REQUEST", req.type == MessageType.REQUEST)

        # Response
        res = await bus.respond("a", "b", {"res": "data"})
        self.test("respond() creates RESPONSE", res.type == MessageType.RESPONSE)

        # Delegate
        del_msg = await bus.delegate("a", "b", {"del": "data"})
        self.test("delegate() creates DELEGATION", del_msg.type == MessageType.DELEGATION)

        # Status
        status = await bus.status_update("a", "b", {"status": "ok"})
        self.test("status_update() creates STATUS", status.type == MessageType.STATUS)

    async def test_history_tracking(self):
        """Test message history tracking."""
        print("\n📚 Test: History Tracking")

        get_handler_registry().reset()

        bus = MessageBus(session_id="test")

        # Send multiple messages
        await bus.request("a1", "b1", {"msg": 1})
        await bus.respond("a2", "b2", {"msg": 2})
        await bus.broadcast("a3", {"msg": 3})

        history = bus.get_history()

        self.test("History tracks messages", len(history) == 3)
        self.test("History newest first", history[0].payload.get("msg") == 3)

        # Clear history
        count = bus.clear_history()
        self.test("clear_history returns count", count == 3)
        self.test("History cleared", len(bus.get_history()) == 0)

    async def test_history_filtering(self):
        """Test history filtering options."""
        print("\n🔍 Test: History Filtering")

        get_handler_registry().reset()

        bus = MessageBus(session_id="test")

        # Create diverse messages
        await bus.request("alice", "bob", {"id": 1})
        await bus.request("alice", "charlie", {"id": 2})
        await bus.respond("bob", "alice", {"id": 3})
        await bus.broadcast("dave", {"id": 4})

        # Filter by sender
        alice_msgs = bus.get_history(sender="alice")
        self.test("Filter by sender", len(alice_msgs) == 2)

        # Filter by receiver
        bob_msgs = bus.get_history(receiver="bob")
        self.test("Filter by receiver", len(bob_msgs) == 1)

        # Filter by type
        requests = bus.get_history(message_type=MessageType.REQUEST)
        self.test("Filter by type", len(requests) == 2)

        # Filter with limit
        limited = bus.get_history(limit=2)
        self.test("Limit works", len(limited) == 2)

    async def test_activity_logging(self):
        """Test activity logging integration."""
        print("\n📊 Test: Activity Logging")

        get_handler_registry().reset()

        activity_log = []

        def log_callback(entry):
            activity_log.append(entry)

        bus = MessageBus(session_id="test", activity_logger=log_callback)

        await bus.request("sender", "receiver", {"test": "data"})

        self.test("Activity logged", len(activity_log) > 0)
        self.test("Has message_sent event", any(e["event_type"] == "message_sent" for e in activity_log))
        self.test("Has message_delivered event", any(e["event_type"] == "message_delivered" for e in activity_log))

        # Check log structure
        if activity_log:
            entry = activity_log[0]
            self.test("Log has session_id", "session_id" in entry)
            self.test("Log has timestamp", "timestamp" in entry)
            self.test("Log has event_type", "event_type" in entry)

    async def test_error_handling(self):
        """Test error handling and fail-safe behavior."""
        print("\n🛡️  Test: Error Handling")

        get_handler_registry().reset()

        @cli_communication_handler(type=MessageType.REQUEST)
        async def faulty_handler(payload, sender, receiver):
            raise ValueError("Test error")

        bus = MessageBus(session_id="test")

        # Should not raise exception
        try:
            msg = await bus.send("a", "b", MessageType.REQUEST, {"test": "data"})
            self.test("No exception on handler error", True)
            # Message delivered even with error (handler was called)
            self.test("System continues after error", True)
        except Exception as e:
            self.test("No exception on handler error", False, str(e))

        # Test activity logger failure
        def faulty_logger(entry):
            raise RuntimeError("Logger error")

        bus2 = MessageBus(session_id="test2", activity_logger=faulty_logger)

        try:
            await bus2.request("a", "b", {"test": "data"})
            self.test("No exception on logger error", True)
        except Exception as e:
            self.test("No exception on logger error", False, str(e))

        get_handler_registry().reset()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)

        print(f"\nTests passed: {passed}/{total}")

        if self.errors:
            print(f"\n❌ FAILURES ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✅ ALL TESTS PASSED!")

        print("\n" + "=" * 60)


async def main():
    """Run all tests."""
    tester = TestMessageBus()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
