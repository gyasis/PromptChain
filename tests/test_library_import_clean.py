"""Test that library consumers can `import promptchain` without textual installed.

Regression test for v0.6.1 Bug 1: eager TUI import in promptchain/__init__.py
chain pulled `from textual.app import App` even for non-TUI consumers.
"""

import subprocess
import sys
import textwrap


def test_import_promptchain_without_textual():
    """Importing promptchain (and the public symbols) must not require textual.

    Simulates a clean library-consumer environment by stubbing textual + textual.app
    to ``None`` in sys.modules BEFORE importing promptchain. Any transitive
    `from textual.app import App` would raise ImportError on a None-stubbed module.
    """
    code = textwrap.dedent(
        """
        import sys

        # Make any `import textual` / `from textual.app import App` blow up
        # the way it would in a venv where textual is not installed.
        sys.modules["textual"] = None
        sys.modules["textual.app"] = None
        sys.modules["textual.widgets"] = None
        sys.modules["textual.containers"] = None

        import promptchain
        from promptchain import (
            PromptChain,
            PubSubBus,
            AsyncAgentInbox,
            PromptEngineer,
        )
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Library import failed without textual:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    assert "OK" in result.stdout
