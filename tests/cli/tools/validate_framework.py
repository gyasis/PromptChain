#!/usr/bin/env python3
"""
Validation script for CLI Tools Testing Framework.

Verifies that all components are correctly set up and ready for use.
"""

import sys
from pathlib import Path
from typing import List, Tuple


def check_file_exists(file_path: Path) -> Tuple[bool, str]:
    """Check if file exists and return status message."""
    if file_path.exists():
        lines = len(file_path.read_text().splitlines()) if file_path.is_file() else 0
        return True, f"✅ {file_path.name} ({lines} lines)" if lines else f"✅ {file_path.name}"
    return False, f"❌ Missing: {file_path.name}"


def validate_framework() -> bool:
    """Validate the testing framework structure."""
    base_path = Path(__file__).parent
    all_passed = True
    results = []

    print("=" * 70)
    print("CLI Tools Testing Framework Validation")
    print("=" * 70)
    print()

    # Check core files
    print("📋 Core Files:")
    print("-" * 70)

    core_files = [
        base_path / "__init__.py",
        base_path / "conftest.py",
        base_path / "README.md",
        base_path / "TESTING_FRAMEWORK_SUMMARY.md",
    ]

    for file_path in core_files:
        passed, msg = check_file_exists(file_path)
        print(f"  {msg}")
        if not passed:
            all_passed = False
            results.append(msg)

    print()

    # Check test files
    print("🧪 Core Test Files:")
    print("-" * 70)

    test_files = [
        base_path / "test_registry.py",
        base_path / "test_executor.py",
        base_path / "test_safety.py",
    ]

    for file_path in test_files:
        passed, msg = check_file_exists(file_path)
        print(f"  {msg}")
        if not passed:
            all_passed = False
            results.append(msg)

    print()

    # Check category directories
    print("📁 Category Directories:")
    print("-" * 70)

    categories = {
        "filesystem": [
            "__init__.py",
            "conftest.py",
        ],
        "code": [
            "__init__.py",
        ],
        "git": [
            "__init__.py",
        ],
        "integration": [
            "__init__.py",
        ],
        "performance": [
            "__init__.py",
            "conftest.py",
        ],
        "security": [
            "__init__.py",
            "conftest.py",
        ],
    }

    for category, files in categories.items():
        category_path = base_path / category
        print(f"\n  {category}/")

        for file_name in files:
            file_path = category_path / file_name
            passed, msg = check_file_exists(file_path)
            print(f"    {msg}")
            if not passed:
                all_passed = False
                results.append(f"{category}/{msg}")

    print()

    # Check fixtures in conftest.py
    print("🔧 Fixture Validation:")
    print("-" * 70)

    conftest_path = base_path / "conftest.py"
    if conftest_path.exists():
        conftest_content = conftest_path.read_text()

        expected_fixtures = [
            "tool_registry",
            "tool_executor",
            "safety_validator",
            "temp_project",
            "sample_files",
            "git_project",
            "performance_timer",
            "benchmark_runner",
            "attack_vectors",
            "security_validator",
        ]

        for fixture in expected_fixtures:
            if f"def {fixture}(" in conftest_content:
                print(f"  ✅ {fixture}()")
            else:
                print(f"  ❌ Missing fixture: {fixture}()")
                all_passed = False
                results.append(f"Missing fixture: {fixture}()")

    print()

    # Validate filesystem fixtures
    print("📂 File System Fixtures:")
    print("-" * 70)

    fs_conftest = base_path / "filesystem" / "conftest.py"
    if fs_conftest.exists():
        fs_content = fs_conftest.read_text()

        fs_fixtures = [
            "file_tree",
            "large_file",
            "binary_file",
            "readonly_file",
            "symlink_file",
            "file_with_encoding",
        ]

        for fixture in fs_fixtures:
            if f"def {fixture}(" in fs_content:
                print(f"  ✅ {fixture}()")
            else:
                print(f"  ⚠️  Missing: {fixture}()")

    print()

    # Validate performance fixtures
    print("⚡ Performance Fixtures:")
    print("-" * 70)

    perf_conftest = base_path / "performance" / "conftest.py"
    if perf_conftest.exists():
        perf_content = perf_conftest.read_text()

        perf_fixtures = [
            "latency_tracker",
            "warmup_runner",
            "throughput_tester",
            "memory_tracker",
            "performance_baseline",
        ]

        for fixture in perf_fixtures:
            if f"def {fixture}(" in perf_content:
                print(f"  ✅ {fixture}()")
            else:
                print(f"  ⚠️  Missing: {fixture}()")

    print()

    # Validate security fixtures
    print("🔒 Security Fixtures:")
    print("-" * 70)

    sec_conftest = base_path / "security" / "conftest.py"
    if sec_conftest.exists():
        sec_content = sec_conftest.read_text()

        sec_fixtures = [
            "owasp_attack_vectors",
            "security_test_runner",
            "sandbox_executor",
            "permission_tester",
            "input_validator",
        ]

        for fixture in sec_fixtures:
            if f"def {fixture}(" in sec_content:
                print(f"  ✅ {fixture}()")
            else:
                print(f"  ⚠️  Missing: {fixture}()")

    print()

    # Statistics
    print("=" * 70)
    print("📊 Framework Statistics:")
    print("-" * 70)

    total_fixtures = 0
    for conftest_file in base_path.rglob("conftest.py"):
        content = conftest_file.read_text()
        fixtures_count = content.count("@pytest.fixture")
        total_fixtures += fixtures_count
        rel_path = conftest_file.relative_to(base_path)
        print(f"  {rel_path}: {fixtures_count} fixtures")

    print()
    print(f"  Total Fixtures: {total_fixtures}")

    # Count lines of code
    total_lines = 0
    for py_file in base_path.rglob("*.py"):
        if py_file.name != "__pycache__":
            lines = len(py_file.read_text().splitlines())
            total_lines += lines

    print(f"  Total Lines of Code: {total_lines:,}")

    # Count markdown docs
    md_files = list(base_path.rglob("*.md"))
    print(f"  Documentation Files: {len(md_files)}")

    print()

    # Final result
    print("=" * 70)
    if all_passed:
        print("✅ Framework Validation: PASSED")
        print()
        print("The testing framework is correctly set up and ready for use!")
        print()
        print("Next Steps:")
        print("  1. Run existing tests: pytest tests/cli/tools/test_registry.py -v")
        print("  2. Create tool tests using provided patterns")
        print("  3. Reference README.md for testing guidelines")
        return True
    else:
        print("❌ Framework Validation: FAILED")
        print()
        print("Issues found:")
        for result in results:
            print(f"  - {result}")
        print()
        print("Please fix the issues above and run validation again.")
        return False


if __name__ == "__main__":
    success = validate_framework()
    sys.exit(0 if success else 1)
