# Performance Benchmarks for MLflow Observability

This document describes the comprehensive performance benchmarks for the MLflow observability package in PromptChain.

## Overview

The performance benchmarks are located in `tests/test_observability_performance.py` and validate all performance-related success criteria:

- **SC-002**: <0.1% overhead when disabled (ghost decorator)
- **SC-003**: <5ms per operation with background queue
- **SC-010**: 100+ metrics/second throughput
- **SC-001**: Metrics in UI within 5 seconds

## Running Benchmarks

### Run All Benchmarks

```bash
pytest tests/test_observability_performance.py -v
```

### Run Specific Benchmark

```bash
# Test disabled overhead (SC-002)
pytest tests/test_observability_performance.py::TestDisabledOverhead::test_benchmark_disabled_overhead -v

# Test enabled overhead (SC-003)
pytest tests/test_observability_performance.py::TestEnabledOverhead::test_benchmark_enabled_overhead -v

# Test queue throughput (SC-010)
pytest tests/test_observability_performance.py::TestQueueThroughput::test_benchmark_queue_throughput -v

# Test startup time (SC-001)
pytest tests/test_observability_performance.py::TestStartupTime::test_benchmark_startup_time -v
```

### Run with Detailed Output

```bash
pytest tests/test_observability_performance.py -v -s
```

The `-s` flag shows the detailed benchmark output with timing statistics.

## Benchmark Descriptions

### 1. Disabled Overhead (SC-002)

**File**: `TestDisabledOverhead::test_benchmark_disabled_overhead`

**Objective**: Verify <0.1% overhead when `PROMPTCHAIN_MLFLOW_ENABLED=false`

**Method**:
- Runs 1 million function call iterations
- Compares baseline (undecorated) vs ghost decorator (disabled)
- Runs 3 iterations and reports median
- Uses warm-up runs to eliminate JIT compilation effects

**Expected Output**:
```
BENCHMARK: benchmark_disabled_overhead
----------------------------------------
Iterations: 1,000,000
Baseline time: 0.123s
Decorated time: 0.123s
Overhead: 0.05%
Threshold: <0.1%
Status: PASS
```

### 2. Enabled Overhead (SC-003)

**File**: `TestEnabledOverhead::test_benchmark_enabled_overhead`

**Objective**: Verify <5ms per operation with background queue

**Method**:
- Runs 10,000 tracked LLM calls
- Measures average operation overhead
- Tests with background queue enabled
- Uses mocked MLflow to isolate queue performance

**Expected Output**:
```
BENCHMARK: benchmark_enabled_overhead
----------------------------------------
Operations: 10,000
Median overhead per op: 2.34ms
Threshold: <5ms
Status: PASS
```

### 3. Queue Throughput (SC-010)

**File**: `TestQueueThroughput::test_benchmark_queue_throughput`

**Objective**: Verify 100+ metrics/second throughput

**Method**:
- Submits 10,000 metrics rapidly
- Measures processing rate
- Verifies queue doesn't block main thread
- Runs 3 iterations and reports median throughput

**Expected Output**:
```
BENCHMARK: benchmark_queue_throughput
----------------------------------------
Operations: 10,000
Duration: 25.123s
Throughput: 398.05 ops/sec
Threshold: >=100 ops/sec
Status: PASS
```

### 4. Startup Time (SC-001)

**File**: `TestStartupTime::test_benchmark_startup_time`

**Objective**: Verify metrics appear in MLflow UI within 5 seconds

**Method**:
- Measures time from `init_mlflow()` to first metric logged
- Tests with mocked MLflow server for speed
- Runs 3 iterations and reports median
- Includes experiment setup and first metric flush

**Expected Output**:
```
BENCHMARK: benchmark_startup_time
----------------------------------------
Measured time: 245.67ms
Threshold: <5000ms
Status: PASS
```

### 5. Nested Run Overhead

**File**: `TestNestedRunOverhead::test_benchmark_nested_run_overhead`

**Objective**: Measure overhead of nested runs (session → LLM → tool)

**Method**:
- Compares 3-level nested run hierarchy vs flat structure
- Tests context switching overhead
- Runs 1,000 iterations per structure
- Informational benchmark (no hard threshold)

**Expected Output**:
```
BENCHMARK: benchmark_nested_run_overhead
----------------------------------------
Iterations: 1,000
Flat structure time: 0.123s
Nested structure time (3 levels): 0.156s
Overhead: 26.83%
Status: PASS (informational)
```

### 6. Concurrent Sessions

**File**: `TestConcurrentSessions::test_benchmark_concurrent_sessions`

**Objective**: Test performance with 10 concurrent sessions

**Method**:
- Runs 10 sessions concurrently using asyncio
- Measures context switching overhead
- Verifies ContextVars isolation doesn't degrade performance
- Compares concurrent vs sequential execution

**Expected Output**:
```
BENCHMARK: benchmark_concurrent_sessions
----------------------------------------
Sessions: 10
Operations per session: 100
Sequential time: 10.234s
Concurrent time: 1.567s
Speedup: 6.53x
Status: PASS (informational)
```

### 7. Large Parameter Logging

**File**: `TestLargeParameterLogging::test_benchmark_large_parameter_logging`

**Objective**: Test with large parameter payloads (10KB, 100KB, 1MB)

**Method**:
- Tests serialization overhead with large payloads
- Verifies background queue handles large payloads
- Measures log time vs payload size
- Runs 3 iterations per payload size

**Expected Output**:
```
BENCHMARK: benchmark_large_parameter_logging
----------------------------------------
10KB: 12.34ms
100KB: 45.67ms
1MB: 234.56ms
Status: PASS (informational)
```

## Benchmark Methodology

### Reproducibility

All benchmarks use the following methodology for reproducibility:

1. **Multiple Iterations**: Each benchmark runs 3 times
2. **Median Reporting**: Reports median values to eliminate outliers
3. **Warm-up Runs**: Includes warm-up iterations to eliminate JIT effects
4. **Mocked Dependencies**: Uses mocked MLflow to isolate performance

### Statistical Analysis

Benchmarks use Python's `statistics` module for:
- Median calculation (primary metric)
- Mean and standard deviation (when needed)
- Min/max values for range analysis

### Environment Variables

Benchmarks control environment via `patch.dict(os.environ, ...)`:

```python
# Disable tracking
{"PROMPTCHAIN_MLFLOW_ENABLED": "false"}

# Enable tracking with background queue
{
    "PROMPTCHAIN_MLFLOW_ENABLED": "true",
    "PROMPTCHAIN_MLFLOW_BACKGROUND": "true"
}
```

## Performance Thresholds

| Benchmark | Success Criteria | Threshold | Status |
|-----------|-----------------|-----------|--------|
| Disabled Overhead | SC-002 | <0.1% | Critical |
| Enabled Overhead | SC-003 | <5ms per op | Critical |
| Queue Throughput | SC-010 | ≥100 ops/sec | Critical |
| Startup Time | SC-001 | <5 seconds | Critical |
| Nested Run Overhead | Informational | <50% | Warning |
| Concurrent Sessions | Informational | Faster than sequential | Info |
| Large Payload | Informational | <1s for 1MB | Info |

## Utility Functions

The benchmark suite includes reusable utility functions:

### `measure_iterations(func, iterations)`

Measures execution time for N iterations of a function.

```python
time_taken = measure_iterations(my_function, 1_000_000)
```

### `calculate_overhead_pct(baseline_time, test_time)`

Calculates overhead percentage.

```python
overhead = calculate_overhead_pct(1.0, 1.05)  # Returns 5.0
```

### `measure_throughput(submit_func, count)`

Measures throughput in operations per second.

```python
throughput = measure_throughput(queue.submit, 10_000)
```

### `run_benchmark_iterations(benchmark_func, iterations=3)`

Runs benchmark multiple times and returns statistics.

```python
stats = run_benchmark_iterations(my_benchmark, iterations=3)
# Returns: {"min": 0.1, "max": 0.15, "median": 0.12, "mean": 0.12, "std": 0.02}
```

### `print_benchmark_result(name, iterations, baseline_time, test_time, threshold, unit="%")`

Prints formatted benchmark result.

### `print_throughput_result(name, count, duration, throughput, threshold)`

Prints formatted throughput result.

### `print_timing_result(name, time_ms, threshold_ms)`

Prints formatted timing result.

## Interpreting Results

### Success

All critical benchmarks pass their thresholds:
- ✅ Disabled overhead < 0.1%
- ✅ Enabled overhead < 5ms
- ✅ Queue throughput ≥ 100 ops/sec
- ✅ Startup time < 5 seconds

### Warnings

If any critical benchmark fails, investigate:
1. System load during benchmark
2. Environment configuration
3. MLflow server performance
4. Background queue settings

### Informational Benchmarks

Additional benchmarks provide insights but don't fail builds:
- Nested run overhead (acceptable up to 50%)
- Concurrent session speedup (should be faster than sequential)
- Large payload logging (should be <1s for 1MB)

## CI/CD Integration

### GitHub Actions

Add to `.github/workflows/performance.yml`:

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      - name: Run performance benchmarks
        run: |
          pytest tests/test_observability_performance.py -v -s
```

### Local Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running performance benchmarks..."
pytest tests/test_observability_performance.py --tb=short
if [ $? -ne 0 ]; then
    echo "Performance benchmarks failed!"
    exit 1
fi
```

## Troubleshooting

### Benchmark Failures

**Issue**: `AssertionError: Disabled overhead 0.15% exceeds 0.1% threshold`

**Solution**:
1. Ensure no other CPU-intensive processes running
2. Run benchmarks multiple times to verify consistency
3. Check if system is under load
4. Increase iteration count for more stable measurements

**Issue**: `Queue throughput below threshold`

**Solution**:
1. Verify background queue is enabled
2. Check mock operations are fast (no actual I/O)
3. Increase queue size if dropping operations
4. Review background thread startup time

**Issue**: `Startup time exceeds threshold`

**Solution**:
1. Verify MLflow mocking is working
2. Check network calls aren't hitting real server
3. Review experiment creation overhead
4. Optimize init_mlflow() implementation

### System Requirements

For accurate benchmarks:
- Python 3.8+
- 2+ CPU cores
- 4GB+ RAM
- Low system load (<50% CPU)
- No network I/O (mocked)

## Performance Tracking

### Baseline Measurements

Record baseline measurements for regression detection:

```bash
# Run benchmarks and save results
pytest tests/test_observability_performance.py -v -s > benchmark_results_$(date +%Y%m%d).txt
```

### Continuous Monitoring

Track performance over time:
1. Run benchmarks on every PR
2. Compare results to baseline
3. Flag regressions >10% degradation
4. Update baselines after verified improvements

## Contributing

When adding new benchmarks:

1. **Follow Naming Convention**: `test_benchmark_<feature>_<metric>`
2. **Include Documentation**: Add to this README
3. **Use Utilities**: Leverage existing utility functions
4. **Run 3 Iterations**: Report median values
5. **Add Assertions**: Include threshold-based assertions
6. **Print Results**: Use formatted output functions

Example:

```python
def test_benchmark_new_feature(self):
    """
    Benchmark new feature performance.

    Objective: Verify XYZ metric meets threshold
    """
    # Run benchmark 3 times
    results = []
    for _ in range(3):
        result = run_single_benchmark()
        results.append(result)

    median_result = statistics.median(results)

    # Print formatted result
    print_timing_result(
        name="benchmark_new_feature",
        time_ms=median_result,
        threshold_ms=100.0
    )

    # Assert threshold
    assert median_result < 100.0, (
        f"New feature {median_result:.2f}ms exceeds 100ms threshold"
    )
```

## References

- **SC-002**: Success Criteria - Disabled Overhead (<0.1%)
- **SC-003**: Success Criteria - Enabled Overhead (<5ms)
- **SC-010**: Success Criteria - Queue Throughput (≥100 ops/sec)
- **SC-001**: Success Criteria - Startup Time (<5 seconds)
- **FR-007**: Functional Requirement - Nested Run Tracking
- **FR-009**: Functional Requirement - Graceful Degradation
- **FR-010**: Functional Requirement - Background Queue
- **FR-016**: Functional Requirement - Exception Handling

---

**Last Updated**: 2026-01-10
**Maintainer**: PromptChain Team
**Version**: 1.0.0
