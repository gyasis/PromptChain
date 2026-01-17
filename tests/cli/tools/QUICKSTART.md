# Testing Framework Quickstart Guide

Get started testing CLI tools in under 5 minutes.

## Prerequisites

```bash
# Ensure pytest is installed
pip install pytest pytest-asyncio pytest-mock
```

## Quick Validation

```bash
# Validate framework is set up correctly
python tests/cli/tools/validate_framework.py

# Should output: ✅ Framework Validation: PASSED
```

## Run Existing Tests

```bash
# Run all existing tests
pytest tests/cli/tools/ -v

# Run specific test file
pytest tests/cli/tools/test_registry.py -v

# Run with coverage
pytest tests/cli/tools/ --cov=promptchain.cli.tools --cov-report=html
```

## Write Your First Test

### 1. Choose Category

Tests are organized by tool category:
- `filesystem/` - File operations (read, write, edit, search)
- `code/` - Code analysis (grep, symbols, AST)
- `git/` - Git operations (status, commit, diff)
- `integration/` - Multi-tool workflows

### 2. Create Test File

Example: `tests/cli/tools/filesystem/test_read.py`

```python
"""Tests for fs.read tool (T118)."""

import pytest


class TestFsRead:
    """Test fs.read functionality."""

    def test_read_basic(self, tool_executor, tool_registry, sample_files):
        """Test reading a simple text file."""

        # Register the tool
        @tool_registry.register(
            category="filesystem",
            description="Read file contents",
            parameters={"path": {"type": "string", "required": True}}
        )
        def fs_read(path: str) -> str:
            """Read file from filesystem."""
            with open(path, 'r') as f:
                return f.read()

        # Execute the tool
        python_file = sample_files["python"]
        result = tool_executor.execute("fs_read", path=str(python_file))

        # Verify result
        assert result.success is True
        assert "def hello" in result.result  # sample_files["python"] contains this

    def test_read_nonexistent_file(self, tool_executor, tool_registry):
        """Test error handling for missing file."""

        @tool_registry.register(
            category="filesystem",
            description="Read file",
            parameters={"path": {"type": "string", "required": True}}
        )
        def fs_read(path: str) -> str:
            with open(path, 'r') as f:
                return f.read()

        # Should handle error gracefully
        result = tool_executor.execute("fs_read", path="/nonexistent/file.txt")

        assert result.success is False
        assert result.error_type in ["FileNotFoundError", "OSError"]

    def test_read_performance(self, tool_executor, tool_registry,
                             sample_files, benchmark_runner):
        """Test read performance meets SLA."""

        @tool_registry.register(
            category="filesystem",
            description="Read file",
            parameters={"path": {"type": "string", "required": True}}
        )
        def fs_read(path: str) -> str:
            with open(path, 'r') as f:
                return f.read()

        # Benchmark execution
        small_file = sample_files["text"]
        results = benchmark_runner(
            lambda: tool_executor.execute("fs_read", path=str(small_file)),
            iterations=100
        )

        # Should be fast (<50ms for small files)
        assert results.mean_ms < 50
        assert results.p95_ms < 100
```

### 3. Run Your Test

```bash
# Run your new test
pytest tests/cli/tools/filesystem/test_read.py -v

# Run with detailed output
pytest tests/cli/tools/filesystem/test_read.py -vv -s
```

## Common Fixtures

### Project Structure

```python
def test_with_temp_project(temp_project):
    """temp_project provides a realistic project structure."""
    # temp_project has:
    # - src/main.py
    # - tests/test_main.py
    # - README.md
    # - config.json

    assert (temp_project / "src" / "main.py").exists()
```

### Sample Files

```python
def test_with_samples(sample_files):
    """sample_files provides various file types."""
    # Available files:
    # - python: Python source with classes/functions
    # - text: Plain text file
    # - config: JSON configuration
    # - markdown: Markdown documentation
    # - large: Large file (10,000 lines)
    # - binary: Binary file

    python_code = sample_files["python"].read_text()
    assert "def hello" in python_code
```

### Performance Testing

```python
def test_performance(benchmark_runner):
    """Measure execution speed with statistics."""
    results = benchmark_runner(
        lambda: my_fast_function(),
        iterations=100
    )

    # Get detailed statistics
    print(f"Mean: {results.mean_ms:.2f}ms")
    print(f"P95: {results.p95_ms:.2f}ms")
    print(f"P99: {results.p99_ms:.2f}ms")

    assert results.mean_ms < 50
```

### Security Testing

```python
def test_security(attack_vectors, tool_executor):
    """Test against common attack vectors."""

    # Test path traversal attacks
    for attack in attack_vectors["path_traversal"]:
        result = tool_executor.execute("fs_read", path=attack)
        assert result.success is False  # Should be blocked

    # Test command injection
    for attack in attack_vectors["command_injection"]:
        result = tool_executor.execute("shell_exec", cmd=attack)
        assert result.success is False  # Should be blocked
```

## Test Organization

```python
"""
Tests for <tool_name> (<task_id>).

Tests cover:
- Basic functionality
- Edge cases
- Error handling
- Security validation
- Performance requirements
"""

import pytest


class TestBasicFunctionality:
    """Test normal operations."""

    def test_happy_path(self):
        """Test successful execution."""
        pass


class TestEdgeCases:
    """Test boundaries and special cases."""

    def test_empty_input(self):
        """Test with empty input."""
        pass

    def test_large_input(self):
        """Test with very large input."""
        pass


class TestErrorHandling:
    """Test error conditions."""

    def test_invalid_input(self):
        """Test with invalid input."""
        pass


class TestSecurity:
    """Test security constraints."""

    def test_path_traversal(self):
        """Test path traversal is blocked."""
        pass


class TestPerformance:
    """Test performance requirements."""

    def test_execution_speed(self):
        """Test meets performance SLA."""
        pass
```

## Useful Commands

```bash
# Run all tests
pytest tests/cli/tools/

# Run specific category
pytest tests/cli/tools/filesystem/

# Run with coverage report
pytest tests/cli/tools/ --cov=promptchain.cli.tools --cov-report=html
open htmlcov/index.html

# Run only security tests
pytest tests/cli/tools/security/

# Run only performance tests
pytest tests/cli/tools/performance/

# Run with verbose output
pytest tests/cli/tools/ -vv

# Run and show print statements
pytest tests/cli/tools/ -s

# Run specific test method
pytest tests/cli/tools/test_registry.py::TestToolRegistration::test_register_basic_tool

# Run tests in parallel (faster)
pytest tests/cli/tools/ -n auto
```

## Debugging Tests

```python
def test_debug_example(tool_executor):
    """Example with debugging."""

    # Add breakpoint
    import pdb; pdb.set_trace()

    # Or use pytest's builtin
    import pytest; pytest.set_trace()

    result = tool_executor.execute("my_tool", param="value")

    # Print detailed info
    print(f"Result: {result}")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Time: {result.execution_time_ms}ms")
```

## Common Patterns

### Testing Tool Registration

```python
def test_tool_registration(tool_registry):
    """Test tool can be registered."""

    @tool_registry.register(
        category="utility",
        description="Test tool",
        parameters={"input": {"type": "string", "required": True}}
    )
    def my_tool(input: str) -> str:
        return f"Processed: {input}"

    # Verify registration
    tool = tool_registry.get("my_tool")
    assert tool is not None
    assert tool.name == "my_tool"
```

### Testing Tool Execution

```python
def test_tool_execution(tool_executor, tool_registry):
    """Test tool execution."""

    @tool_registry.register(
        category="utility",
        description="Test",
        parameters={"value": {"type": "integer", "required": True}}
    )
    def double(value: int) -> int:
        return value * 2

    result = tool_executor.execute("double", value=21)

    assert result.success is True
    assert result.result == 42
```

### Testing Error Handling

```python
def test_error_handling(tool_executor, tool_registry):
    """Test error conditions."""

    @tool_registry.register(
        category="utility",
        description="Test",
        parameters={"value": {"type": "integer", "required": True}}
    )
    def divide_by_zero(value: int) -> float:
        return value / 0  # Will raise ZeroDivisionError

    result = tool_executor.execute("divide_by_zero", value=10)

    assert result.success is False
    assert result.error_type == "ZeroDivisionError"
```

## Next Steps

1. **Read Full Documentation**: `tests/cli/tools/README.md`
2. **See Examples**: Look at `test_registry.py`, `test_executor.py`
3. **Add Your Tests**: Follow the patterns above
4. **Run Validation**: `python tests/cli/tools/validate_framework.py`

## Getting Help

- **Full Testing Guide**: `tests/cli/tools/README.md`
- **Framework Summary**: `tests/cli/tools/TESTING_FRAMEWORK_SUMMARY.md`
- **Existing Tests**: Browse `test_*.py` files for examples
- **Fixtures**: Check `conftest.py` files for available fixtures

---

**Ready to Test!** Start with the simple example above, then gradually add more sophisticated tests using the fixtures and patterns provided.
