"""
Performance tests for ToolExecutor

Validates the <50ms execution overhead requirement from T116.
"""

import pytest
import time
import statistics

from promptchain.cli.tools.registry import ToolRegistry, ToolCategory
from promptchain.cli.tools.executor import ToolExecutor


@pytest.fixture
def performance_registry():
    """Create registry with performance test tools."""
    reg = ToolRegistry()

    # Instant tool (baseline)
    @reg.register(
        category=ToolCategory.UTILITY,
        description="Instant return",
        parameters={},
    )
    def instant() -> str:
        return "done"

    # Tool with single parameter
    @reg.register(
        category=ToolCategory.UTILITY,
        description="Echo with parameter",
        parameters={
            "value": {"type": "string", "required": True, "description": "Value to echo"}
        },
    )
    def echo(value: str) -> str:
        return value

    # Tool with multiple parameters
    @reg.register(
        category=ToolCategory.UTILITY,
        description="Complex calculation",
        parameters={
            "a": {"type": "integer", "required": True, "description": "First number"},
            "b": {"type": "integer", "required": True, "description": "Second number"},
            "c": {"type": "integer", "required": True, "description": "Third number"},
            "d": {"type": "integer", "required": True, "description": "Fourth number"},
        },
    )
    def complex_calc(a: int, b: int, c: int, d: int) -> int:
        return a + b + c + d

    return reg


def test_validation_overhead_target(performance_registry):
    """
    Test that validation overhead is <50ms (T116 acceptance criteria).

    This is the CRITICAL performance requirement.
    """
    executor = ToolExecutor(performance_registry)

    # Run multiple iterations to get reliable measurement
    iterations = 100
    overhead_times = []

    for _ in range(iterations):
        result = executor.execute("instant")
        assert result.success is True

        # Extract validation time from metrics
        validation_time = result.metadata["metrics"]["validation_time_ms"]
        overhead_times.append(validation_time)

    # Calculate statistics
    avg_overhead = statistics.mean(overhead_times)
    max_overhead = max(overhead_times)
    p95_overhead = sorted(overhead_times)[int(0.95 * len(overhead_times))]

    print(f"\nValidation Overhead Statistics ({iterations} runs):")
    print(f"  Average: {avg_overhead:.3f}ms")
    print(f"  Maximum: {max_overhead:.3f}ms")
    print(f"  P95: {p95_overhead:.3f}ms")
    print(f"  Target: <50ms")

    # CRITICAL: P95 must be under 50ms
    assert p95_overhead < 50, (
        f"Validation overhead P95 ({p95_overhead:.2f}ms) exceeds 50ms target. "
        f"Average: {avg_overhead:.2f}ms, Max: {max_overhead:.2f}ms"
    )

    # Ideally, average should also be well under target
    assert avg_overhead < 30, (
        f"Average validation overhead ({avg_overhead:.2f}ms) is too high. "
        f"Should be <30ms for good performance."
    )


def test_execution_overhead_with_parameters(performance_registry):
    """Test overhead with different parameter counts."""
    executor = ToolExecutor(performance_registry)

    # Single parameter
    result_1 = executor.execute("echo", value="test")
    validation_1 = result_1.metadata["metrics"]["validation_time_ms"]

    # Multiple parameters
    result_4 = executor.execute("complex_calc", a=1, b=2, c=3, d=4)
    validation_4 = result_4.metadata["metrics"]["validation_time_ms"]

    print(f"\nValidation time by parameter count:")
    print(f"  1 parameter:  {validation_1:.3f}ms")
    print(f"  4 parameters: {validation_4:.3f}ms")

    # Validation should be fast regardless of parameter count
    assert validation_1 < 50
    assert validation_4 < 50

    # More parameters shouldn't dramatically increase overhead
    assert validation_4 < validation_1 * 3, (
        f"Validation overhead increases too much with parameters: "
        f"{validation_1:.2f}ms (1 param) vs {validation_4:.2f}ms (4 params)"
    )


def test_type_coercion_overhead(performance_registry):
    """Test overhead with type coercion enabled vs disabled."""
    executor_coerce = ToolExecutor(performance_registry, enable_type_coercion=True)
    executor_no_coerce = ToolExecutor(performance_registry, enable_type_coercion=False)

    # With coercion (string to int)
    result_coerce = executor_coerce.execute("complex_calc", a="1", b="2", c="3", d="4")
    validation_coerce = result_coerce.metadata["metrics"]["validation_time_ms"]

    # Without coercion (already int)
    result_no_coerce = executor_no_coerce.execute("complex_calc", a=1, b=2, c=3, d=4)
    validation_no_coerce = result_no_coerce.metadata["metrics"]["validation_time_ms"]

    print(f"\nType coercion overhead:")
    print(f"  With coercion:    {validation_coerce:.3f}ms")
    print(f"  Without coercion: {validation_no_coerce:.3f}ms")
    print(f"  Difference:       {validation_coerce - validation_no_coerce:.3f}ms")

    # Both should be under target
    assert validation_coerce < 50
    assert validation_no_coerce < 50

    # Coercion shouldn't add much overhead
    assert validation_coerce < validation_no_coerce + 20, (
        "Type coercion adds too much overhead"
    )


def test_total_execution_time(performance_registry):
    """Test total execution time (validation + execution)."""
    executor = ToolExecutor(performance_registry)

    iterations = 50
    total_times = []

    for _ in range(iterations):
        result = executor.execute("instant")
        total_time = result.metadata["metrics"]["total_time_ms"]
        total_times.append(total_time)

    avg_total = statistics.mean(total_times)
    p95_total = sorted(total_times)[int(0.95 * len(total_times))]

    print(f"\nTotal Execution Time ({iterations} runs):")
    print(f"  Average: {avg_total:.3f}ms")
    print(f"  P95: {p95_total:.3f}ms")

    # Total time should be very fast for instant tool
    assert p95_total < 100, f"Total execution time too high: {p95_total:.2f}ms"


def test_performance_stats_accuracy(performance_registry):
    """Test that performance stats are accurately tracked."""
    executor = ToolExecutor(performance_registry)
    executor.reset_metrics()

    # Execute known number of tools
    for i in range(10):
        executor.execute("instant")
        executor.execute("echo", value=f"test{i}")

    stats = executor.get_performance_stats()

    # Verify stats
    assert stats["total_executions"] == 20
    assert stats["avg_execution_time_ms"] >= 0
    assert stats["avg_validation_time_ms"] >= 0
    assert stats["avg_overhead_ms"] >= 0

    # Average overhead should be under target
    assert stats["avg_overhead_ms"] < 50, (
        f"Average overhead {stats['avg_overhead_ms']:.2f}ms exceeds 50ms target"
    )


def test_concurrent_execution_overhead(performance_registry):
    """Test overhead doesn't increase with concurrent executions."""
    import asyncio

    executor = ToolExecutor(performance_registry)

    async def measure_concurrent():
        """Execute multiple tools concurrently."""
        tasks = [
            executor.execute_async("instant")
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(measure_concurrent())

    # All should succeed
    assert all(r.success for r in results)

    # Check overhead for each
    overhead_times = [
        r.metadata["metrics"]["validation_time_ms"]
        for r in results
    ]

    avg_overhead = statistics.mean(overhead_times)
    max_overhead = max(overhead_times)

    print(f"\nConcurrent Execution Overhead (10 tasks):")
    print(f"  Average: {avg_overhead:.3f}ms")
    print(f"  Maximum: {max_overhead:.3f}ms")

    # Concurrent execution shouldn't increase overhead
    assert avg_overhead < 50
    assert max_overhead < 100  # Even max should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
