#!/usr/bin/env python3
"""
Pytest configuration for Research Agent tests

Handles common test setup, fixtures, and utilities for all test categories.
"""

import os
import sys
from pathlib import Path
import pytest

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Add project root to path for any direct imports
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Fixture providing the project root path"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")  
def test_data_path():
    """Fixture providing the test data path"""
    return Path(__file__).parent.parent / "test_data"

@pytest.fixture(scope="session")
def src_path():
    """Fixture providing the src path"""
    return Path(__file__).parent.parent / "src"

# Configure logging for tests
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose logging from some libraries during tests
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)