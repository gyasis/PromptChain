"""
Performance testing fixtures.

Provides specialized fixtures for benchmarking and latency testing.
"""

import pytest
import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any
from pathlib import Path


@dataclass
class LatencyStats:
    """Statistical analysis of latency measurements."""

    measurements: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.measurements)

    @property
    def mean_ms(self) -> float:
        """Average latency in milliseconds."""
        return statistics.mean(self.measurements) if self.measurements else 0.0

    @property
    def median_ms(self) -> float:
        """Median latency in milliseconds."""
        return statistics.median(self.measurements) if self.measurements else 0.0

    @property
    def stdev_ms(self) -> float:
        """Standard deviation in milliseconds."""
        return statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum latency in milliseconds."""
        return min(self.measurements) if self.measurements else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum latency in milliseconds."""
        return max(self.measurements) if self.measurements else 0.0

    @property
    def p50_ms(self) -> float:
        """50th percentile (median)."""
        return self.percentile(50)

    @property
    def p95_ms(self) -> float:
        """95th percentile latency."""
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        """99th percentile latency."""
        return self.percentile(99)

    def percentile(self, p: float) -> float:
        """Calculate percentile."""
        if not self.measurements:
            return 0.0
        sorted_data = sorted(self.measurements)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def summary(self) -> Dict[str, float]:
        """Get complete summary statistics."""
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "stdev_ms": self.stdev_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


@pytest.fixture
def latency_tracker():
    """
    Track latency measurements with statistical analysis.

    Usage:
        tracker = latency_tracker()
        with tracker.measure():
            expensive_operation()

        stats = tracker.stats()
        assert stats.mean_ms < 100
    """

    class LatencyTracker:
        def __init__(self):
            self.measurements = []

        def measure(self):
            """Context manager for measuring latency."""
            return LatencyMeasurement(self)

        def add_measurement(self, duration_ms: float):
            """Add a latency measurement."""
            self.measurements.append(duration_ms)

        def stats(self) -> LatencyStats:
            """Get statistical summary."""
            return LatencyStats(measurements=self.measurements.copy())

        def reset(self):
            """Clear all measurements."""
            self.measurements.clear()

    class LatencyMeasurement:
        def __init__(self, tracker):
            self.tracker = tracker
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = (time.perf_counter() - self.start_time) * 1000
            self.tracker.add_measurement(elapsed)

    return LatencyTracker()


@pytest.fixture
def warmup_runner():
    """
    Run warmup iterations before benchmarking.

    Ensures JIT compilation, cache warming, etc.
    """

    def warmup(func: Callable, iterations: int = 10):
        """
        Run function multiple times for warmup.

        Args:
            func: Function to warm up
            iterations: Number of warmup iterations
        """
        for _ in range(iterations):
            try:
                func()
            except Exception:
                pass  # Ignore warmup errors

    return warmup


@pytest.fixture
def throughput_tester():
    """
    Measure operations per second.

    Usage:
        tester = throughput_tester()
        ops_per_sec = tester.measure(my_function, duration_seconds=1.0)
        assert ops_per_sec > 1000
    """

    class ThroughputTester:
        def measure(
            self,
            func: Callable,
            duration_seconds: float = 1.0,
            warmup: bool = True
        ) -> float:
            """
            Measure throughput (operations per second).

            Args:
                func: Function to benchmark
                duration_seconds: How long to run benchmark
                warmup: Run warmup iterations first

            Returns:
                Operations per second
            """
            # Warmup
            if warmup:
                for _ in range(10):
                    try:
                        func()
                    except Exception:
                        pass

            # Measure
            start = time.perf_counter()
            end_time = start + duration_seconds
            count = 0

            while time.perf_counter() < end_time:
                func()
                count += 1

            elapsed = time.perf_counter() - start
            return count / elapsed

    return ThroughputTester()


@pytest.fixture
def memory_tracker():
    """
    Track memory usage during operations.

    Note: Requires psutil to be installed.
    """
    try:
        import psutil
        import os

        class MemoryTracker:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.start_mb = None
                self.peak_mb = None

            def __enter__(self):
                self.start_mb = self.process.memory_info().rss / 1024 / 1024
                self.peak_mb = self.start_mb
                return self

            def __exit__(self, *args):
                current_mb = self.process.memory_info().rss / 1024 / 1024
                self.peak_mb = max(self.peak_mb, current_mb)

            @property
            def current_mb(self) -> float:
                """Current memory usage in MB."""
                return self.process.memory_info().rss / 1024 / 1024

            @property
            def increase_mb(self) -> float:
                """Memory increase since start in MB."""
                return self.current_mb - self.start_mb

        return MemoryTracker()

    except ImportError:
        # Return mock if psutil not available
        class MockMemoryTracker:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            current_mb = 0.0
            increase_mb = 0.0

        return MockMemoryTracker()


@pytest.fixture
def performance_report_generator(tmp_path):
    """
    Generate performance reports in multiple formats.

    Usage:
        generator = performance_report_generator()
        generator.add_benchmark("tool_name", stats)
        generator.save_report("json")
    """
    import json

    class PerformanceReportGenerator:
        def __init__(self, output_dir: Path):
            self.output_dir = output_dir
            self.benchmarks = {}

        def add_benchmark(self, name: str, stats: LatencyStats):
            """Add benchmark results."""
            self.benchmarks[name] = stats.summary()

        def save_report(self, format: str = "json") -> Path:
            """
            Save performance report.

            Args:
                format: Report format (json, csv, markdown)

            Returns:
                Path to saved report
            """
            if format == "json":
                report_path = self.output_dir / "performance_report.json"
                with open(report_path, "w") as f:
                    json.dump(self.benchmarks, f, indent=2)
                return report_path

            elif format == "csv":
                report_path = self.output_dir / "performance_report.csv"
                with open(report_path, "w") as f:
                    # Write header
                    headers = ["benchmark"] + list(
                        next(iter(self.benchmarks.values())).keys()
                    )
                    f.write(",".join(headers) + "\n")

                    # Write data
                    for name, stats in self.benchmarks.items():
                        values = [name] + [str(v) for v in stats.values()]
                        f.write(",".join(values) + "\n")
                return report_path

            elif format == "markdown":
                report_path = self.output_dir / "performance_report.md"
                with open(report_path, "w") as f:
                    f.write("# Performance Report\n\n")

                    for name, stats in self.benchmarks.items():
                        f.write(f"## {name}\n\n")
                        f.write("| Metric | Value (ms) |\n")
                        f.write("|--------|------------|\n")
                        for metric, value in stats.items():
                            f.write(f"| {metric} | {value:.2f} |\n")
                        f.write("\n")
                return report_path

            else:
                raise ValueError(f"Unknown format: {format}")

    return PerformanceReportGenerator(tmp_path)


@pytest.fixture
def performance_baseline():
    """
    Load and compare against performance baselines.

    Usage:
        baseline = performance_baseline()
        baseline.set("tool_name", mean_ms=50, p95_ms=100)
        assert baseline.check("tool_name", stats) is True
    """

    class PerformanceBaseline:
        def __init__(self):
            self.baselines = {}

        def set(self, name: str, **metrics):
            """Set performance baseline for a tool."""
            self.baselines[name] = metrics

        def check(self, name: str, stats: LatencyStats, tolerance: float = 1.1) -> bool:
            """
            Check if stats meet baseline (within tolerance).

            Args:
                name: Benchmark name
                stats: Measured statistics
                tolerance: Allowed deviation (1.1 = 10% slower ok)

            Returns:
                True if within baseline, False if regression
            """
            if name not in self.baselines:
                return True  # No baseline set

            baseline = self.baselines[name]
            actual = stats.summary()

            for metric, threshold in baseline.items():
                if metric in actual:
                    if actual[metric] > threshold * tolerance:
                        return False

            return True

        def report_regression(self, name: str, stats: LatencyStats) -> str:
            """Generate regression report."""
            if name not in self.baselines:
                return "No baseline to compare"

            baseline = self.baselines[name]
            actual = stats.summary()

            lines = [f"Performance regression for {name}:"]
            for metric, threshold in baseline.items():
                if metric in actual:
                    value = actual[metric]
                    pct_change = ((value - threshold) / threshold) * 100
                    if value > threshold:
                        lines.append(
                            f"  {metric}: {value:.2f}ms "
                            f"(baseline: {threshold:.2f}ms, +{pct_change:.1f}%)"
                        )

            return "\n".join(lines)

    return PerformanceBaseline()


@pytest.fixture
def load_generator():
    """
    Generate load for stress testing.

    Usage:
        generator = load_generator()
        generator.ramp_up(my_function, target_rps=100, duration=10)
    """
    import threading
    import queue

    class LoadGenerator:
        def ramp_up(
            self,
            func: Callable,
            target_rps: int,
            duration: float,
            warmup: float = 1.0
        ) -> Dict[str, Any]:
            """
            Gradually increase load to target requests per second.

            Args:
                func: Function to execute
                target_rps: Target requests per second
                duration: Test duration in seconds
                warmup: Warmup period in seconds

            Returns:
                Load test results with statistics
            """
            results = queue.Queue()
            stop_event = threading.Event()

            def worker():
                while not stop_event.is_set():
                    start = time.perf_counter()
                    try:
                        func()
                        elapsed = (time.perf_counter() - start) * 1000
                        results.put(("success", elapsed))
                    except Exception as e:
                        results.put(("error", str(e)))

            # Start workers
            workers = []
            for _ in range(target_rps):
                t = threading.Thread(target=worker, daemon=True)
                t.start()
                workers.append(t)
                time.sleep((warmup + duration) / target_rps)  # Ramp up

            # Wait for duration
            time.sleep(duration)

            # Stop workers
            stop_event.set()
            for t in workers:
                t.join(timeout=1.0)

            # Collect results
            successes = []
            errors = []
            while not results.empty():
                status, value = results.get()
                if status == "success":
                    successes.append(value)
                else:
                    errors.append(value)

            return {
                "total_requests": len(successes) + len(errors),
                "successful": len(successes),
                "failed": len(errors),
                "actual_rps": len(successes) / duration,
                "stats": LatencyStats(measurements=successes),
            }

    return LoadGenerator()
