#!/usr/bin/env python3
"""
Basic Callbacks Example

This example demonstrates basic callback usage with PromptChain's event system.
Shows how to register callbacks, receive events, and access event metadata.
"""

import asyncio
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType


def simple_logger(event: ExecutionEvent):
    """Simple callback that logs all events."""
    print(f"[{event.timestamp.strftime('%H:%M:%S')}] {event.event_type.name}")
    if event.step_number is not None:
        print(f"  Step: {event.step_number}")
    if event.model_name:
        print(f"  Model: {event.model_name}")


def performance_monitor(event: ExecutionEvent):
    """Monitor execution performance."""
    if "execution_time_ms" in event.metadata:
        time_ms = event.metadata["execution_time_ms"]
        print(f"⏱️  {event.event_type.name}: {time_ms:.1f}ms")


def error_logger(event: ExecutionEvent):
    """Log only errors."""
    error_msg = event.metadata.get("error", "Unknown error")
    error_type = event.metadata.get("error_type", "Error")
    print(f"❌ {error_type}: {error_msg}")
    print(f"   Step: {event.step_number}, Model: {event.model_name}")


async def async_callback(event: ExecutionEvent):
    """Async callback example."""
    # Simulate async operation (e.g., database write)
    await asyncio.sleep(0.01)
    print(f"✓ Async logged: {event.event_type.name}")


def main():
    """Run basic callbacks example."""
    print("=" * 60)
    print("Basic Callbacks Example")
    print("=" * 60)

    # Example 1: Register single callback
    print("\n1. Simple Logger (all events)")
    print("-" * 60)

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Analyze this request: {input}",
            "Provide a brief summary: {input}"
        ],
        verbose=False
    )

    chain.register_callback(simple_logger)

    result = chain.process_prompt("What is quantum computing?")
    print(f"\nResult: {result[:100]}...")

    # Example 2: Multiple callbacks with filtering
    print("\n\n2. Filtered Callbacks")
    print("-" * 60)

    chain2 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Answer concisely: {input}"
        ],
        verbose=False
    )

    # Performance monitor: Only START and END events
    chain2.register_callback(
        performance_monitor,
        event_filter=[
            ExecutionEventType.CHAIN_START,
            ExecutionEventType.CHAIN_END,
            ExecutionEventType.STEP_START,
            ExecutionEventType.STEP_END,
            ExecutionEventType.MODEL_CALL_END
        ]
    )

    # Error logger: Only ERROR events
    chain2.register_callback(
        error_logger,
        event_filter=[
            ExecutionEventType.CHAIN_ERROR,
            ExecutionEventType.STEP_ERROR,
            ExecutionEventType.MODEL_CALL_ERROR
        ]
    )

    result2 = chain2.process_prompt("Explain photosynthesis briefly")
    print(f"\nResult: {result2[:100]}...")

    # Example 3: Async callback
    print("\n\n3. Async Callback")
    print("-" * 60)

    chain3 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Quick answer: {input}"],
        verbose=False
    )

    # Register async callback
    chain3.register_callback(async_callback)

    result3 = chain3.process_prompt("What is 2+2?")
    print(f"\nResult: {result3}")

    # Example 4: Unregister callbacks
    print("\n\n4. Callback Management")
    print("-" * 60)

    def temp_callback(event: ExecutionEvent):
        print(f"  Temp: {event.event_type.name}")

    chain4 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Answer: {input}"],
        verbose=False
    )

    print("With callback:")
    chain4.register_callback(temp_callback)
    result4a = chain4.process_prompt("Hello")

    print("\nCallback unregistered:")
    chain4.unregister_callback(temp_callback)
    result4b = chain4.process_prompt("World")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
