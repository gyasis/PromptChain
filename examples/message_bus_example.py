#!/usr/bin/env python3
"""
MessageBus Example: Multi-Agent Communication

Demonstrates how to use the MessageBus for agent-to-agent messaging
in PromptChain CLI workflows.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptchain.cli.communication import (
    MessageBus,
    MessageType,
    cli_communication_handler,
    CommunicationHandler
)
from promptchain.cli.communication.handlers import get_handler_registry


# Define message handlers for different agents


@cli_communication_handler(
    type=MessageType.REQUEST,
    receiver="analyzer"
)
async def handle_analysis_request(payload, sender, receiver):
    """Handle analysis requests sent to analyzer agent."""
    print(f"\n📊 ANALYZER received request from {sender}")
    print(f"   Task: {payload.get('task')}")

    # Simulate analysis work
    await asyncio.sleep(0.5)

    result = {
        "status": "completed",
        "analysis": f"Analysis of {payload.get('data', 'unknown')} completed",
        "insights": ["Pattern A detected", "Trend B identified"]
    }

    print(f"   ✅ Analysis complete: {result['analysis']}")
    return result


@cli_communication_handler(
    type=MessageType.DELEGATION,
    receiver="coder"
)
async def handle_coding_delegation(payload, sender, receiver):
    """Handle code generation tasks delegated to coder agent."""
    print(f"\n💻 CODER received delegation from {sender}")
    print(f"   Requirement: {payload.get('requirement')}")

    # Simulate code generation
    await asyncio.sleep(0.5)

    code = f"def {payload.get('function_name', 'process')}():\n    # Generated code\n    pass"

    print(f"   ✅ Code generated successfully")
    return {"status": "implemented", "code": code}


@cli_communication_handler(
    type=MessageType.BROADCAST
)
async def handle_broadcasts(payload, sender, receiver):
    """Handle broadcast messages (all agents receive this)."""
    print(f"\n📢 BROADCAST from {sender}: {payload.get('message')}")
    return {"received": True}


@cli_communication_handler(
    types=[MessageType.STATUS, MessageType.RESPONSE],
    priority=10  # Higher priority handler
)
async def handle_status_and_responses(payload, sender, receiver):
    """Handle status updates and responses (high priority)."""
    msg_type = "STATUS" if "status" in payload else "RESPONSE"
    print(f"\n📝 {msg_type} from {sender} → {receiver}")
    print(f"   Content: {payload}")
    return {"acknowledged": True}


async def example_workflow():
    """
    Example multi-agent workflow using MessageBus.

    Demonstrates:
    1. Direct messaging between agents
    2. Message delegation
    3. Broadcast messages
    4. Status updates
    5. History tracking
    """

    print("=" * 60)
    print("MESSAGE BUS EXAMPLE: MULTI-AGENT WORKFLOW")
    print("=" * 60)

    # Create message bus with activity logging
    activity_log = []

    def log_activity(entry):
        activity_log.append(entry)
        print(f"   [LOG] {entry['event_type']}: {entry.get('sender', 'N/A')} → {entry.get('receiver', 'N/A')}")

    bus = MessageBus(
        session_id="example-workflow",
        activity_logger=log_activity
    )

    print("\n🚀 Starting workflow...")

    # Step 1: Coordinator sends analysis request to analyzer
    print("\n--- Step 1: Request Analysis ---")
    msg1 = await bus.request(
        sender="coordinator",
        receiver="analyzer",
        payload={
            "task": "analyze_data",
            "data": "sales_report.csv",
            "priority": "high"
        }
    )
    print(f"Message ID: {msg1.message_id}")

    # Step 2: Coordinator delegates coding task to coder
    print("\n--- Step 2: Delegate Coding Task ---")
    msg2 = await bus.delegate(
        sender="coordinator",
        receiver="coder",
        payload={
            "requirement": "Create data processing function",
            "function_name": "process_sales_data",
            "specs": {"input": "DataFrame", "output": "Summary"}
        }
    )

    # Step 3: Broadcast status update to all agents
    print("\n--- Step 3: Broadcast Update ---")
    msg3 = await bus.broadcast(
        sender="coordinator",
        payload={
            "message": "Workflow 50% complete",
            "timestamp": msg2.timestamp
        }
    )

    # Step 4: Analyzer sends response back
    print("\n--- Step 4: Response from Analyzer ---")
    msg4 = await bus.respond(
        sender="analyzer",
        receiver="coordinator",
        payload={
            "status": "analysis_complete",
            "result": "Positive trend detected",
            "next_steps": ["Review findings", "Generate report"]
        }
    )

    # Step 5: Status update from coder
    print("\n--- Step 5: Status Update ---")
    msg5 = await bus.status_update(
        sender="coder",
        receiver="coordinator",
        payload={
            "status": "in_progress",
            "completion": 0.75,
            "eta": "2 minutes"
        }
    )

    # Check message history
    print("\n" + "=" * 60)
    print("MESSAGE HISTORY")
    print("=" * 60)

    history = bus.get_history(limit=10)
    print(f"\nTotal messages: {len(history)}")

    for i, msg in enumerate(history, 1):
        print(f"\n{i}. {msg.type.value.upper()}")
        print(f"   {msg.sender} → {msg.receiver}")
        print(f"   Delivered: {msg.delivered}")
        print(f"   ID: {msg.message_id[:8]}...")

    # Filter history examples
    print("\n--- Filtered History Examples ---")

    coordinator_messages = bus.get_history(sender="coordinator")
    print(f"\nMessages from coordinator: {len(coordinator_messages)}")

    analyzer_messages = bus.get_history(receiver="analyzer")
    print(f"Messages to analyzer: {len(analyzer_messages)}")

    requests = bus.get_history(message_type=MessageType.REQUEST)
    print(f"Request messages: {len(requests)}")

    # Activity log summary
    print("\n" + "=" * 60)
    print("ACTIVITY LOG SUMMARY")
    print("=" * 60)
    print(f"\nTotal activity events: {len(activity_log)}")

    event_types = {}
    for event in activity_log:
        event_type = event['event_type']
        event_types[event_type] = event_types.get(event_type, 0) + 1

    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")

    # Demonstrate message serialization
    print("\n" + "=" * 60)
    print("MESSAGE SERIALIZATION")
    print("=" * 60)

    # Serialize a message
    serialized = msg1.to_dict()
    print(f"\nSerialized message:\n{serialized}")

    # Deserialize back
    from promptchain.cli.communication.message_bus import Message
    deserialized = Message.from_dict(serialized)
    print(f"\nDeserialized message:")
    print(f"  ID: {deserialized.message_id}")
    print(f"  {deserialized.sender} → {deserialized.receiver}")
    print(f"  Type: {deserialized.type}")

    print("\n" + "=" * 60)
    print("✅ WORKFLOW COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Direct agent-to-agent messaging")
    print("  ✓ Message delegation")
    print("  ✓ Broadcast messaging")
    print("  ✓ Handler-based message routing")
    print("  ✓ Activity logging")
    print("  ✓ Message history tracking")
    print("  ✓ Message serialization/deserialization")

    # Clean up handlers for next run
    get_handler_registry().reset()


async def example_error_handling():
    """
    Example demonstrating error handling in message handlers.
    """

    print("\n" + "=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)

    @cli_communication_handler(type=MessageType.REQUEST, receiver="faulty")
    async def faulty_handler(payload, sender, receiver):
        """Handler that raises an error."""
        print(f"\n⚠️  FAULTY handler called")
        raise ValueError("Simulated handler error")

    bus = MessageBus(session_id="error-demo")

    print("\n--- Sending message to faulty handler ---")
    msg = await bus.request(
        sender="test",
        receiver="faulty",
        payload={"test": "data"}
    )

    print(f"\nMessage delivered: {msg.delivered}")
    print("Note: System continues despite handler error (fail-safe design)")

    # Clean up
    get_handler_registry().reset()


async def main():
    """Run all examples."""
    await example_workflow()
    await example_error_handling()


if __name__ == "__main__":
    asyncio.run(main())
