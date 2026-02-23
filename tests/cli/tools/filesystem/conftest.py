"""
File system tools testing fixtures.

Provides specialized fixtures for testing file operations.
"""

import pytest
from pathlib import Path
from typing import Dict, List


@pytest.fixture
def file_tree(temp_project):
    """
    Create a diverse file tree for testing search/navigation.

    Returns:
        Dict with file tree structure and paths
    """
    tree = {
        "root": temp_project,
        "files": [],
        "dirs": []
    }

    # Create nested directory structure
    dirs = [
        "src/utils",
        "src/models",
        "src/api",
        "tests/unit",
        "tests/integration",
        "docs/api",
        "docs/guides",
    ]

    for dir_path in dirs:
        full_path = temp_project / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        tree["dirs"].append(full_path)

    # Create various file types
    files = {
        "src/main.py": "def main(): pass",
        "src/utils/helpers.py": "def helper(): pass",
        "src/models/user.py": "class User: pass",
        "src/api/routes.py": "def route(): pass",
        "tests/unit/test_main.py": "def test_main(): pass",
        "tests/integration/test_api.py": "def test_api(): pass",
        "docs/api/reference.md": "# API Reference",
        "docs/guides/quickstart.md": "# Quick Start",
        "README.md": "# Project",
        "LICENSE": "MIT License",
        ".gitignore": "*.pyc\n__pycache__/",
    }

    for file_path, content in files.items():
        full_path = temp_project / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        tree["files"].append(full_path)

    return tree


@pytest.fixture
def large_file(temp_project):
    """Create a large file for testing performance and chunking."""
    file_path = temp_project / "large.txt"

    # Create 1MB file
    lines = [f"Line {i}: {'x' * 80}" for i in range(10_000)]
    file_path.write_text("\n".join(lines))

    return file_path


@pytest.fixture
def binary_file(temp_project):
    """Create a binary file for testing binary handling."""
    file_path = temp_project / "data.bin"

    # Create binary file with various byte values
    data = bytes(range(256)) * 100  # 25.6KB
    file_path.write_bytes(data)

    return file_path


@pytest.fixture
def readonly_file(temp_project):
    """Create a read-only file for testing permission handling."""
    file_path = temp_project / "readonly.txt"
    file_path.write_text("Read-only content")

    # Make read-only
    file_path.chmod(0o444)

    yield file_path

    # Cleanup: restore write permission
    file_path.chmod(0o644)


@pytest.fixture
def symlink_file(temp_project):
    """Create a symlink for testing symlink handling."""
    target = temp_project / "target.txt"
    target.write_text("Target content")

    link = temp_project / "link.txt"
    link.symlink_to(target)

    return {"target": target, "link": link}


@pytest.fixture
def file_with_encoding(temp_project):
    """Create files with different encodings."""
    files = {}

    # UTF-8 file
    files["utf8"] = temp_project / "utf8.txt"
    files["utf8"].write_text("Hello 世界 🌍", encoding="utf-8")

    # Latin-1 file
    files["latin1"] = temp_project / "latin1.txt"
    files["latin1"].write_text("Café résumé", encoding="latin-1")

    return files


@pytest.fixture
def nested_structure(temp_project):
    """
    Create deeply nested directory structure for testing recursion limits.

    Returns:
        Path to deepest directory
    """
    current = temp_project
    depth = 20  # Create 20 levels deep

    for i in range(depth):
        current = current / f"level_{i}"
        current.mkdir()

    # Add a file at the deepest level
    (current / "deep_file.txt").write_text("Found me!")

    return current


@pytest.fixture
def mixed_permissions_tree(temp_project):
    """Create directory tree with mixed permissions for testing."""
    # Readable directory
    readable = temp_project / "readable"
    readable.mkdir()
    (readable / "file.txt").write_text("Readable")

    # Writable directory
    writable = temp_project / "writable"
    writable.mkdir()
    (writable / "file.txt").write_text("Writable")
    writable.chmod(0o755)

    # Read-only directory
    readonly = temp_project / "readonly"
    readonly.mkdir()
    (readonly / "file.txt").write_text("Read-only")
    readonly.chmod(0o555)

    yield {
        "readable": readable,
        "writable": writable,
        "readonly": readonly
    }

    # Cleanup: restore permissions
    readonly.chmod(0o755)


@pytest.fixture
def file_change_tracker(temp_project):
    """Track file modifications for testing change detection."""
    import time
    from datetime import datetime

    class FileTracker:
        def __init__(self, root: Path):
            self.root = root
            self.snapshots = {}

        def snapshot(self, name: str):
            """Take a snapshot of file modification times."""
            self.snapshots[name] = {
                str(f.relative_to(self.root)): f.stat().st_mtime
                for f in self.root.rglob("*")
                if f.is_file()
            }

        def changes_since(self, name: str) -> List[str]:
            """Get list of files changed since snapshot."""
            if name not in self.snapshots:
                return []

            old_snapshot = self.snapshots[name]
            current = {
                str(f.relative_to(self.root)): f.stat().st_mtime
                for f in self.root.rglob("*")
                if f.is_file()
            }

            changed = []
            for path, mtime in current.items():
                if path not in old_snapshot or old_snapshot[path] < mtime:
                    changed.append(path)

            return changed

    return FileTracker(temp_project)


@pytest.fixture
def file_content_generator():
    """Generate various file contents for testing."""

    def generate(file_type: str, size: str = "small") -> str:
        """
        Generate file content.

        Args:
            file_type: Type of content (python, json, markdown, text)
            size: Size category (small, medium, large)

        Returns:
            Generated content string
        """
        sizes = {
            "small": 100,
            "medium": 1000,
            "large": 10000
        }
        lines = sizes.get(size, 100)

        if file_type == "python":
            return "\n".join([
                f"def function_{i}():",
                f"    '''Function {i} docstring.'''",
                f"    return {i}",
                ""
            ] * (lines // 4))

        elif file_type == "json":
            import json
            return json.dumps({
                f"key_{i}": f"value_{i}"
                for i in range(lines)
            }, indent=2)

        elif file_type == "markdown":
            return "\n".join([
                f"# Heading {i}",
                f"Content for section {i}",
                ""
            ] * (lines // 3))

        else:  # text
            return "\n".join([f"Line {i}" for i in range(lines)])

    return generate
