# CLI Tools Testing Guide

Comprehensive testing framework for Phase 11 CLI tools ecosystem.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Shared Fixtures](#shared-fixtures)
- [Testing Patterns](#testing-patterns)
- [Running Tests](#running-tests)
- [Writing New Tests](#writing-new-tests)
- [Performance Testing](#performance-testing)
- [Security Testing](#security-testing)

## Overview

This testing framework provides:

- **Reusable fixtures** for common testing scenarios
- **Test patterns** for different tool categories
- **Performance benchmarks** for execution speed
- **Security tests** for OWASP Top 10 attack vectors
- **Integration tests** for tool combinations

## Test Structure

```
tests/cli/tools/
├── conftest.py              # Shared fixtures
├── README.md                # This file
├── test_registry.py         # Tool registry tests (T115)
├── test_executor.py         # Tool executor tests (T116)
├── test_safety.py           # Safety validator tests (T117)
├── filesystem/              # File system tools
│   ├── conftest.py          # FS-specific fixtures
│   ├── test_read.py         # fs.read tests
│   ├── test_write.py        # fs.write tests
│   ├── test_edit.py         # fs.edit tests
│   └── test_search.py       # fs.search tests
├── code/                    # Code analysis tools
│   ├── test_grep.py
│   ├── test_symbols.py
│   └── test_ast_search.py
├── git/                     # Git operation tools
│   ├── test_status.py
│   ├── test_commit.py
│   └── test_diff.py
├── integration/             # Integration tests
│   ├── test_tool_chains.py  # Tool combination tests
│   └── test_agent_tools.py  # AgentChain integration
├── performance/             # Performance tests
│   ├── conftest.py          # Performance fixtures
│   ├── test_tool_latency.py
│   └── benchmarks.py
└── security/                # Security tests
    ├── conftest.py          # Security fixtures
    ├── test_path_traversal.py
    ├── test_command_injection.py
    └── test_resource_limits.py
```

## Shared Fixtures

Located in `conftest.py`, available to all tests.

### Core Infrastructure

**tool_registry**: Fresh tool registry for each test
```python
def test_example(tool_registry):
    @tool_registry.register(
        category="test",
        description="Test tool",
        parameters={"input": {"type": "string", "required": True}}
    )
    def my_tool(input: str) -> str:
        return f"Processed: {input}"

    tool = tool_registry.get("my_tool")
    assert tool is not None
```

**tool_executor**: Tool executor with registry
```python
def test_execution(tool_executor, tool_registry):
    # Register tool first
    result = tool_executor.execute("my_tool", input="test")
    assert result.success is True
```

**safety_validator**: Safety validator with temp project root
```python
def test_security(safety_validator):
    # Valid path
    path = safety_validator.validate_path("src/main.py")

    # Invalid path (should raise)
    with pytest.raises(SecurityError):
        safety_validator.validate_path("../../etc/passwd")
```

### Project Structure

**temp_project**: Temporary project with realistic structure
```python
def test_with_project(temp_project):
    # Project has src/, tests/, README.md
    assert (temp_project / "src" / "main.py").exists()
    assert (temp_project / "README.md").exists()
```

**sample_files**: Diverse sample files for testing
```python
def test_file_operations(sample_files):
    # Available files: python, text, config, markdown, large, binary
    python_file = sample_files["python"]
    content = python_file.read_text()
    assert "def hello" in content
```

**git_project**: Project with initialized Git repository
```python
def test_git_operations(git_project):
    # Git repo initialized with initial commit
    import subprocess
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=git_project,
        capture_output=True
    )
    assert b"Initial commit" in result.stdout
```

### Performance Testing

**performance_timer**: High-resolution timer
```python
def test_performance(performance_timer):
    with performance_timer() as timer:
        # Code to measure
        result = expensive_operation()

    assert timer.elapsed_ms < 100  # Must complete in <100ms
```

**benchmark_runner**: Statistical benchmarking
```python
def test_benchmark(benchmark_runner):
    results = benchmark_runner(
        lambda: my_function(),
        iterations=100
    )

    assert results.mean_ms < 50
    assert results.p95_ms < 100
    assert results.p99_ms < 150
```

### Security Testing

**attack_vectors**: Common attack patterns
```python
def test_security_validation(attack_vectors):
    for attack in attack_vectors["path_traversal"]:
        # Verify attack is blocked
        assert not is_safe_path(attack)
```

**security_validator**: Comprehensive security testing
```python
def test_attack_detection(security_validator):
    # Test against all attack categories
    assert security_validator.validate_against_attacks("normal/path.txt")
    assert not security_validator.validate_against_attacks("../../etc/passwd")
```

## Testing Patterns

### 1. Tool Registration Pattern

Test that tools can be registered and retrieved correctly.

```python
def test_tool_registration(tool_registry):
    """Test tool registration and retrieval."""

    @tool_registry.register(
        category="filesystem",
        description="Read file",
        parameters={"path": {"type": "string", "required": True}}
    )
    def fs_read(path: str) -> str:
        return f"Content of {path}"

    # Verify registration
    tool = tool_registry.get("fs_read")
    assert tool is not None
    assert tool.name == "fs_read"
    assert tool.category == "filesystem"
    assert "path" in tool.parameters

    # Verify in category listing
    fs_tools = tool_registry.list_tools(category="filesystem")
    assert "fs_read" in fs_tools
```

### 2. Tool Execution Pattern

Test tool execution with various inputs.

```python
def test_tool_execution(tool_executor, tool_registry):
    """Test successful tool execution."""

    @tool_registry.register(
        category="math",
        description="Multiply by two",
        parameters={"value": {"type": "integer", "required": True}}
    )
    def multiply_two(value: int) -> int:
        return value * 2

    # Execute tool
    result = tool_executor.execute("multiply_two", value=21)

    # Verify result
    assert result.success is True
    assert result.result == 42
    assert result.error is None
    assert result.execution_time_ms >= 0
```

### 3. Error Handling Pattern

Test error conditions and edge cases.

```python
def test_tool_error_handling(tool_executor, tool_registry):
    """Test tool handles errors gracefully."""

    @tool_registry.register(
        category="test",
        description="Divide numbers",
        parameters={
            "a": {"type": "integer", "required": True},
            "b": {"type": "integer", "required": True}
        }
    )
    def divide(a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    # Test normal execution
    result = tool_executor.execute("divide", a=10, b=2)
    assert result.success is True
    assert result.result == 5.0

    # Test error condition
    result = tool_executor.execute("divide", a=10, b=0)
    assert result.success is False
    assert "Division by zero" in result.error
```

### 4. Security Testing Pattern

Test security validations and attack prevention.

```python
def test_path_security(safety_validator, attack_vectors):
    """Test path traversal attacks are blocked."""

    for attack_path in attack_vectors["path_traversal"]:
        with pytest.raises(SecurityError, match="outside project root"):
            safety_validator.validate_path(attack_path)

    # Valid paths should work
    valid_path = safety_validator.validate_path("src/main.py")
    assert valid_path.name == "main.py"
```

### 5. Performance Testing Pattern

Test tool execution speed and efficiency.

```python
def test_tool_performance(tool_executor, tool_registry, performance_timer):
    """Test tool execution is fast."""

    @tool_registry.register(
        category="test",
        description="Simple operation",
        parameters={"input": {"type": "string", "required": True}}
    )
    def simple_tool(input: str) -> str:
        return input.upper()

    # Measure execution time
    with performance_timer() as timer:
        for _ in range(100):
            tool_executor.execute("simple_tool", input="test")

    avg_time_ms = timer.elapsed_ms / 100

    # Execution overhead should be minimal (<10ms per call)
    assert avg_time_ms < 10, f"Too slow: {avg_time_ms:.2f}ms average"
```

### 6. Integration Testing Pattern

Test multiple tools working together.

```python
def test_tool_chain(tool_executor, tool_registry, temp_project):
    """Test tools can be chained together."""

    # Register tools
    @tool_registry.register(
        category="filesystem",
        description="Read file",
        parameters={"path": {"type": "string", "required": True}}
    )
    def fs_read(path: str) -> str:
        return (temp_project / path).read_text()

    @tool_registry.register(
        category="text",
        description="Count lines",
        parameters={"text": {"type": "string", "required": True}}
    )
    def count_lines(text: str) -> int:
        return len(text.splitlines())

    # Execute chain: read file → count lines
    content = tool_executor.execute("fs_read", path="README.md").result
    line_count = tool_executor.execute("count_lines", text=content).result

    assert isinstance(line_count, int)
    assert line_count > 0
```

## Running Tests

### Run All Tool Tests
```bash
pytest tests/cli/tools/ -v
```

### Run Specific Category
```bash
# File system tests
pytest tests/cli/tools/filesystem/ -v

# Security tests
pytest tests/cli/tools/security/ -v

# Performance tests
pytest tests/cli/tools/performance/ -v
```

### Run with Coverage
```bash
pytest tests/cli/tools/ --cov=promptchain.cli.tools --cov-report=html
```

### Run Performance Benchmarks
```bash
pytest tests/cli/tools/performance/ -v --benchmark-only
```

### Run Security Tests Only
```bash
pytest tests/cli/tools/security/ -v -m security
```

## Writing New Tests

### Step 1: Identify Test Category

Determine which category your test belongs to:
- `filesystem/` - File operations
- `code/` - Code analysis
- `git/` - Git operations
- `integration/` - Multi-tool tests
- `performance/` - Speed/efficiency tests
- `security/` - Security validation

### Step 2: Create Test File

```python
# tests/cli/tools/filesystem/test_my_tool.py

import pytest

def test_my_tool_basic(tool_executor, tool_registry):
    """Test basic functionality."""
    # Register tool
    @tool_registry.register(
        category="filesystem",
        description="My new tool",
        parameters={"input": {"type": "string", "required": True}}
    )
    def my_tool(input: str) -> str:
        return f"Processed: {input}"

    # Test execution
    result = tool_executor.execute("my_tool", input="test")
    assert result.success is True
    assert "Processed" in result.result

def test_my_tool_edge_cases(tool_executor, tool_registry):
    """Test edge cases and errors."""
    # Test with empty input
    # Test with very large input
    # Test with invalid input
    pass

def test_my_tool_security(safety_validator):
    """Test security constraints."""
    # Test input validation
    # Test output sanitization
    pass

def test_my_tool_performance(benchmark_runner):
    """Test execution speed."""
    results = benchmark_runner(lambda: my_tool("test"), iterations=100)
    assert results.mean_ms < 50
```

### Step 3: Add Fixtures (if needed)

If your tests need category-specific fixtures, create a `conftest.py`:

```python
# tests/cli/tools/filesystem/conftest.py

import pytest

@pytest.fixture
def sample_text_file(temp_project):
    """Create a sample text file."""
    file_path = temp_project / "sample.txt"
    file_path.write_text("Sample content\nLine 2\nLine 3")
    return file_path
```

### Step 4: Document Test Coverage

Add docstrings explaining:
- What the test validates
- Why it's important
- Expected behavior
- Edge cases covered

## Performance Testing

### Performance Standards

All tools should meet these performance targets:

- **Execution overhead**: <10ms per call
- **File read**: <50ms for files <100KB
- **File write**: <100ms for files <100KB
- **Search operations**: <500ms for directories <1000 files
- **Git operations**: <200ms for basic commands

### Writing Performance Tests

```python
def test_tool_latency(tool_executor, benchmark_runner):
    """Test tool execution latency meets SLA."""

    # Run benchmark
    results = benchmark_runner(
        lambda: tool_executor.execute("my_tool", input="test"),
        iterations=100
    )

    # Verify performance
    assert results.mean_ms < 10, "Average latency too high"
    assert results.p95_ms < 20, "95th percentile too high"
    assert results.p99_ms < 50, "99th percentile too high"

    # Log results for tracking
    print(f"Performance: {results.mean_ms:.2f}ms avg, "
          f"{results.p95_ms:.2f}ms p95, {results.p99_ms:.2f}ms p99")
```

## Security Testing

### Security Standards

All tools must be tested against:

1. **Path Traversal**: Block `../../` and absolute paths outside project
2. **Command Injection**: Sanitize inputs in shell commands
3. **File Inclusion**: Validate file paths and types
4. **Code Injection**: Escape/validate code inputs
5. **Resource Exhaustion**: Limit file sizes, iterations, memory

### Writing Security Tests

```python
def test_tool_security(safety_validator, attack_vectors):
    """Test tool blocks common attacks."""

    # Test path traversal
    for attack in attack_vectors["path_traversal"]:
        with pytest.raises(SecurityError):
            my_tool_that_takes_path(attack)

    # Test command injection
    for attack in attack_vectors["command_injection"]:
        with pytest.raises(SecurityError):
            my_tool_that_runs_command(attack)

    # Test resource limits
    with pytest.raises(ResourceError):
        my_tool_with_large_input("A" * 10_000_000)  # 10MB
```

### Security Test Markers

Use pytest markers to categorize security tests:

```python
@pytest.mark.security
@pytest.mark.critical
def test_critical_security_flaw():
    """Test critical security vulnerability is fixed."""
    pass

@pytest.mark.security
@pytest.mark.owasp_top10
def test_owasp_vulnerability():
    """Test OWASP Top 10 vulnerability."""
    pass
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Use descriptive test names (test_tool_does_specific_thing)
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Edge Cases**: Test boundaries, empty inputs, large inputs
5. **Error Messages**: Verify error messages are helpful
6. **Performance**: Always test critical paths for speed
7. **Security**: Every tool that touches files/commands needs security tests
8. **Documentation**: Explain WHY, not just WHAT

## Test Coverage Goals

- **Unit tests**: 100% of tool functions
- **Integration tests**: All critical tool combinations
- **Security tests**: All attack vectors for each tool
- **Performance tests**: All performance-critical tools
- **Edge cases**: Empty, null, huge, malformed inputs

## Continuous Integration

Tests run automatically on:
- Every commit
- Pull requests
- Scheduled nightly builds

Performance and security tests must pass before merge.

## Troubleshooting

### Tests Fail in CI but Pass Locally

- Check file path differences (Windows vs Linux)
- Verify fixtures use `tmp_path` for isolation
- Check for timing issues (use `pytest-timeout`)

### Performance Tests are Flaky

- Run more iterations for statistical significance
- Use percentiles (p95, p99) instead of max
- Account for system load (CI may be slower)

### Security Tests Too Strict

- Ensure attack vectors are realistic
- Balance security vs usability
- Document why specific attacks are blocked

## Contributing

When adding new tests:

1. Follow existing patterns
2. Add fixtures to appropriate `conftest.py`
3. Document test purpose and coverage
4. Update this README with new patterns
5. Ensure tests are fast (<1s per test)

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
