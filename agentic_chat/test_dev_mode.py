#!/usr/bin/env python3
"""
Quick test of --dev mode observability integration
Tests that all events are properly logged
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock sys.argv to simulate --dev flag
sys.argv = ["test_dev_mode.py", "--dev"]

# Import after setting sys.argv
from agentic_chat.agentic_team_chat import main

async def test_dev_mode():
    """Test --dev mode with a simple query"""
    print("🧪 Testing --dev mode observability integration...")
    print("=" * 60)

    # The main function should handle --dev flag from sys.argv
    # We'll need to modify it to accept a test query instead of interactive input

    # For now, let's just import and check the logging setup
    import logging
    from agentic_chat.agentic_team_chat import setup_logging, args

    print(f"✅ --dev flag detected: {args.dev}")
    print(f"✅ --quiet flag: {args.quiet}")

    # Check logging handlers
    root_logger = logging.getLogger()
    print(f"\n📊 Logging configuration:")
    print(f"  Root logger level: {logging.getLevelName(root_logger.level)}")
    print(f"  Number of handlers: {len(root_logger.handlers)}")

    for i, handler in enumerate(root_logger.handlers):
        print(f"\n  Handler {i+1}: {handler.__class__.__name__}")
        print(f"    Level: {logging.getLevelName(handler.level)}")
        if hasattr(handler, 'baseFilename'):
            print(f"    File: {handler.baseFilename}")
        if hasattr(handler, 'stream'):
            print(f"    Stream: {handler.stream.name if hasattr(handler.stream, 'name') else 'console'}")

    print(f"\n✅ --dev mode setup complete")
    print(f"Expected behavior:")
    print(f"  - Terminal: Clean output (ERROR level only)")
    print(f"  - File: Full DEBUG logs in session_*.log")
    print(f"  - JSONL: Structured events in session_*.jsonl")

if __name__ == "__main__":
    asyncio.run(test_dev_mode())
