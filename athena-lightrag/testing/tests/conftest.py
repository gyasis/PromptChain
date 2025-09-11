#!/usr/bin/env python3
"""
Test Configuration for Athena LightRAG
======================================
Shared fixtures and configuration for all tests.

Author: PromptChain Team  
Date: 2025
"""

import asyncio
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment variables
os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)
os.environ["TESTING"] = "1"


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests requiring real components"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests with mocked dependencies"  
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "mcp: marks tests related to MCP server functionality"
    )
    config.addinivalue_line(
        "markers", 
        "reasoning: marks tests related to multi-hop reasoning"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    test_env = {
        'OPENAI_API_KEY': 'test-api-key-12345',
        'LIGHTRAG_WORKING_DIR': './tests/mock_athena_lightrag_db',
        'LOG_LEVEL': 'DEBUG',
        'MCP_TRANSPORT': 'stdio',
        'TESTING': '1'
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        yield test_env


@pytest.fixture 
def mock_lightrag_db(temp_dir):
    """Create a mock LightRAG database directory structure."""
    db_path = temp_dir / "mock_athena_lightrag_db"
    db_path.mkdir()
    
    # Create mock database files with realistic content
    mock_files = {
        "entities.json": {
            "medical_entities": ["patient", "appointment", "billing", "anesthesia"],
            "relationships": ["has_appointment", "requires_billing", "uses_anesthesia"]
        },
        "relationships.json": {
            "patient_appointment": {"type": "one_to_many", "strength": 0.9},
            "appointment_billing": {"type": "one_to_one", "strength": 0.8},
            "patient_anesthesia": {"type": "many_to_many", "strength": 0.7}
        },
        "graph_metadata.json": {
            "version": "1.0",
            "entity_count": 1000,
            "relationship_count": 2500,
            "last_updated": "2025-01-01T00:00:00Z"
        }
    }
    
    import json
    for filename, content in mock_files.items():
        (db_path / filename).write_text(json.dumps(content, indent=2))
    
    # Create some additional mock files
    (db_path / "vector_store.bin").write_bytes(b"mock vector data")
    (db_path / "index.db").write_text("mock database index")
    
    return str(db_path)


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses."""
    with patch('openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock chat completion
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Mock OpenAI response for testing purposes"))
        ]
        
        # Mock embeddings
        mock_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)  # Mock 1536-dimensional embedding
        ]
        
        yield mock_client


@pytest.fixture
def mock_lightrag():
    """Mock LightRAG instance with realistic responses.""" 
    with patch('athena_lightrag.core.LightRAG') as mock_lightrag_class:
        mock_rag = AsyncMock()
        mock_lightrag_class.return_value = mock_rag
        
        # Configure mock responses based on query patterns
        def mock_query_response(query, param=None):
            if "patient" in query.lower() and "appointment" in query.lower():
                return """
                The medical database contains several tables related to patient appointments:
                
                1. PATIENTS table - stores patient demographic information
                2. APPOINTMENTS table - tracks scheduled appointments 
                3. APPOINTMENT_STATUS table - manages appointment states
                4. PROVIDERS table - healthcare provider information
                5. SCHEDULES table - provider availability
                
                These tables are connected through foreign key relationships enabling
                comprehensive appointment management and patient tracking.
                """
            elif "anesthesia" in query.lower():
                return """
                Anesthesia workflows in the database involve several key components:
                
                1. ANESTHESIA_CASES table - individual anesthesia procedures
                2. ANESTHESIA_PROVIDERS table - anesthesiologists and CRNAs
                3. MEDICATIONS table - anesthetic drugs and dosages
                4. MONITORING table - vital signs during procedures
                5. BILLING_ANESTHESIA table - procedure billing codes
                
                The workflow connects patient care, provider assignments, medication
                administration, and billing processes in an integrated system.
                """
            elif "sql" in query.lower() or "query" in query.lower():
                return """
                Based on the database schema, here's an optimized SQL query:
                
                SELECT p.patient_id, p.first_name, p.last_name, 
                       a.appointment_date, a.appointment_time,
                       pr.provider_name, pr.specialty
                FROM patients p
                INNER JOIN appointments a ON p.patient_id = a.patient_id
                INNER JOIN providers pr ON a.provider_id = pr.provider_id  
                WHERE a.appointment_date >= CURDATE()
                  AND a.status = 'scheduled'
                ORDER BY a.appointment_date, a.appointment_time;
                
                This query efficiently retrieves upcoming scheduled appointments
                with patient and provider information using proper indexing.
                """
            else:
                return f"""
                Mock LightRAG response for query: {query}
                
                This response simulates the LightRAG knowledge graph retrieval
                for testing purposes. In a real implementation, this would return
                relevant information from the Athena medical database based on
                the query context and specified parameters.
                
                Query mode: {param.mode if param else 'default'}
                Context only: {param.only_need_context if param else False}
                """
        
        mock_rag.aquery.side_effect = mock_query_response
        mock_rag.initialize_storages.return_value = None
        
        yield mock_rag


@pytest.fixture
def mock_promptchain():
    """Mock PromptChain for multi-hop reasoning tests."""
    with patch('promptchain.PromptChain') as mock_chain_class:
        mock_chain = AsyncMock()
        mock_chain_class.return_value = mock_chain
        
        # Mock the process_prompt_async method
        def mock_process_prompt(prompt):
            if "multi-hop" in prompt.lower() or "reasoning" in prompt.lower():
                return """
                Multi-hop reasoning analysis completed:
                
                Step 1: Analyzed patient appointment workflows and identified key tables
                Step 2: Examined anesthesia case management and provider relationships  
                Step 3: Connected billing processes and revenue cycle integration
                Step 4: Synthesized comprehensive workflow understanding
                
                Final Answer: The anesthesia workflows connect to patient scheduling through
                the appointment system, which links to billing via procedure codes and 
                provider assignments. This creates an integrated healthcare delivery and
                revenue management system with full audit trails and compliance tracking.
                """
            else:
                return f"PromptChain processing result for: {prompt[:100]}..."
        
        mock_chain.process_prompt_async.side_effect = mock_process_prompt
        mock_chain.step_outputs = [
            "Step 1: Initial analysis and context gathering",
            "Step 2: Relationship mapping and workflow identification", 
            "Step 3: Integration analysis and synthesis",
            "Final: Comprehensive multi-hop reasoning result"
        ]
        
        # Mock tool registration
        mock_chain.register_tool_function = MagicMock()
        mock_chain.add_tools = MagicMock()
        
        yield mock_chain


@pytest.fixture
async def athena_instance(mock_lightrag_db, mock_env_vars, mock_lightrag):
    """Create a fully configured AthenaLightRAG instance for testing."""
    from athena_lightrag.core import AthenaLightRAG
    
    athena = AthenaLightRAG(
        working_dir=mock_lightrag_db,
        api_key="test-api-key",
        reasoning_model="gpt-4o-mini",
        max_reasoning_steps=3,
        enable_history_management=True
    )
    
    # Mark as initialized to skip setup
    athena.rag_initialized = True
    
    yield athena
    
    # Cleanup
    athena.active_reasoning_sessions.clear()


@pytest.fixture
def sample_query_results():
    """Sample query results for testing."""
    from athena_lightrag.core import QueryResult, ContextChunk
    from datetime import datetime, timezone
    
    basic_result = QueryResult(
        result="Sample query result about medical database tables",
        query_mode="hybrid", 
        context_only=False,
        query_id="test-query-123",
        timestamp=datetime.now(timezone.utc),
        performance_metrics={
            "execution_time_seconds": 1.23,
            "result_length": 150,
            "parameters": {"top_k": 60}
        },
        metadata={
            "query_text": "Sample test query",
            "query_mode": "hybrid"
        }
    )
    
    reasoning_result = QueryResult(
        result="Comprehensive multi-hop reasoning result",
        query_mode="hybrid",
        context_only=False, 
        query_id="test-reasoning-456",
        timestamp=datetime.now(timezone.utc),
        reasoning_steps=[
            "Step 1: Initial analysis",
            "Step 2: Context gathering", 
            "Step 3: Synthesis"
        ],
        accumulated_context="Accumulated context from multiple reasoning steps",
        context_chunks=[
            {
                "content": "First context chunk",
                "source_query": "Sub-query 1", 
                "reasoning_step": 1,
                "chunk_id": "chunk-1"
            }
        ],
        performance_metrics={
            "execution_time_seconds": 5.67,
            "reasoning_steps_count": 3,
            "context_chunks_count": 1
        },
        metadata={
            "initial_query": "Complex reasoning query",
            "context_accumulation_strategy": "incremental"
        }
    )
    
    return {
        "basic": basic_result,
        "reasoning": reasoning_result
    }


# Pytest plugins and hooks
def pytest_runtest_setup(item):
    """Setup for individual test runs.""" 
    # Add any per-test setup here
    pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown for individual test runs."""
    # Add any per-test cleanup here
    pass


# Test data and utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_mock_medical_queries():
        """Generate realistic medical database queries for testing."""
        return [
            "What tables contain patient demographic information?",
            "How are appointments scheduled and tracked?", 
            "What is the relationship between billing and insurance?",
            "How do anesthesia cases connect to surgical procedures?",
            "What audit trails exist for medication administration?",
            "How are provider credentials and specialties managed?",
            "What reporting capabilities exist for quality metrics?",
            "How is patient consent tracked and documented?"
        ]
    
    @staticmethod
    def generate_mock_sql_queries():
        """Generate sample SQL queries for testing SQL generation."""
        return [
            {
                "description": "Find all patients with upcoming appointments",
                "expected_tables": ["patients", "appointments"],
                "complexity": "basic"
            },
            {
                "description": "Complex multi-table join for anesthesia billing analysis", 
                "expected_tables": ["patients", "procedures", "anesthesia_cases", "billing"],
                "complexity": "advanced"
            },
            {
                "description": "Provider performance metrics with quality indicators",
                "expected_tables": ["providers", "appointments", "outcomes", "quality_metrics"], 
                "complexity": "intermediate"
            }
        ]


# Make test utilities available
test_data_generator = TestDataGenerator()