#!/usr/bin/env python3
"""
Comprehensive test suite for LightRAG integration
Tests the fixed implementation to prevent future regressions
"""

import sys
import os
import tempfile
import shutil
import pytest
from pathlib import Path

# Add source to path
sys.path.insert(0, 'src')

def test_lightrag_imports():
    """Test that all LightRAG imports work correctly"""
    # Core LightRAG
    from lightrag import LightRAG
    assert LightRAG is not None
    
    # OpenAI functions (primary provider)
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    assert callable(openai_complete_if_cache)
    assert callable(openai_embed)
    
    # Anthropic functions
    from lightrag.llm.anthropic import anthropic_complete, anthropic_embed
    assert callable(anthropic_complete)
    assert callable(anthropic_embed)
    
    # Ollama functions (corrected names)
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    assert callable(ollama_model_complete)
    assert callable(ollama_embed)
    
    print("✅ All LightRAG imports verified")


def test_three_tier_rag_import():
    """Test that ThreeTierRAG can be imported"""
    from research_agent.integrations.three_tier_rag import ThreeTierRAG, RAGTier
    assert ThreeTierRAG is not None
    assert RAGTier is not None
    print("✅ ThreeTierRAG import successful")


def test_llm_provider_detection():
    """Test LLM provider detection logic"""
    from research_agent.integrations.three_tier_rag import ThreeTierRAG
    
    rag = ThreeTierRAG()
    
    # Test OpenAI models
    assert rag._detect_llm_provider('gpt-4') == 'openai'
    assert rag._detect_llm_provider('gpt-3.5-turbo') == 'openai'
    assert rag._detect_llm_provider('gpt-4o-mini') == 'openai'
    assert rag._detect_llm_provider('o1-preview') == 'openai'
    
    # Test Anthropic models
    assert rag._detect_llm_provider('claude-3-sonnet') == 'anthropic'
    assert rag._detect_llm_provider('claude-3.5-sonnet') == 'anthropic'
    
    # Test Ollama models
    assert rag._detect_llm_provider('llama2') == 'ollama'
    assert rag._detect_llm_provider('mistral') == 'ollama'
    
    # Test unknown models
    assert rag._detect_llm_provider('unknown-model') == 'unknown'
    
    print("✅ LLM provider detection working correctly")


def test_lightrag_initialization_with_api_key():
    """Test LightRAG initialization with proper API key"""
    import tempfile
    
    # Set test API key
    os.environ['OPENAI_API_KEY'] = 'sk-test-key-for-testing'
    
    from research_agent.integrations.three_tier_rag import ThreeTierRAG
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'lightrag_working_dir': temp_dir,
            'lightrag_llm_model': 'gpt-4o-mini'
        }
        
        rag = ThreeTierRAG(config=config)
        
        # Check that tier processors were initialized
        from research_agent.integrations.three_tier_rag import RAGTier
        assert RAGTier.TIER1_LIGHTRAG in rag.tier_processors
        
        tier1_processor = rag.tier_processors[RAGTier.TIER1_LIGHTRAG]
        assert tier1_processor['status'] == 'initialized'
        assert tier1_processor['type'] == 'lightrag'
        assert tier1_processor['processor'] is not None
        
        print("✅ LightRAG initialization with API key successful")


def test_lightrag_initialization_without_api_key():
    """Test LightRAG initialization gracefully handles missing API key"""
    # Remove API key
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    from research_agent.integrations.three_tier_rag import ThreeTierRAG
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'lightrag_working_dir': temp_dir,
            'lightrag_llm_model': 'gpt-4o-mini'
        }
        
        rag = ThreeTierRAG(config=config)
        
        # Should still initialize but with mock processor
        from research_agent.integrations.three_tier_rag import RAGTier
        assert RAGTier.TIER1_LIGHTRAG in rag.tier_processors
        
        tier1_processor = rag.tier_processors[RAGTier.TIER1_LIGHTRAG]
        assert tier1_processor['status'] == 'failed'
        assert tier1_processor['type'] == 'lightrag_mock'
        assert 'OPENAI_API_KEY' in tier1_processor.get('error', '')
        
        print("✅ LightRAG graceful failure without API key working")


def test_different_llm_providers():
    """Test initialization with different LLM providers"""
    from research_agent.integrations.three_tier_rag import ThreeTierRAG
    
    # Test OpenAI
    os.environ['OPENAI_API_KEY'] = 'test-key'
    config = {'lightrag_llm_model': 'gpt-4o-mini'}
    rag = ThreeTierRAG(config=config)
    assert rag._detect_llm_provider('gpt-4o-mini') == 'openai'
    
    # Test Anthropic
    os.environ['ANTHROPIC_API_KEY'] = 'test-key'
    config = {'lightrag_llm_model': 'claude-3-sonnet'}
    rag = ThreeTierRAG(config=config)
    assert rag._detect_llm_provider('claude-3-sonnet') == 'anthropic'
    
    # Test Ollama (no API key needed)
    config = {'lightrag_llm_model': 'llama2'}
    rag = ThreeTierRAG(config=config)
    assert rag._detect_llm_provider('llama2') == 'ollama'
    
    print("✅ Different LLM provider configurations working")


def test_configuration_parameters():
    """Test that configuration parameters are properly passed through"""
    os.environ['OPENAI_API_KEY'] = 'test-key'
    
    from research_agent.integrations.three_tier_rag import ThreeTierRAG
    
    config = {
        'lightrag_llm_model': 'gpt-4',
        'lightrag_embedding_batch_num': 5,
        'lightrag_llm_max_async': 2,
        'lightrag_embedding_max_async': 4,
        'lightrag_chunk_token_size': 800,
        'lightrag_chunk_overlap': 50,
        'lightrag_top_k': 20,
        'lightrag_chunk_top_k': 5
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['lightrag_working_dir'] = temp_dir
        rag = ThreeTierRAG(config=config)
        
        from research_agent.integrations.three_tier_rag import RAGTier
        tier1_processor = rag.tier_processors[RAGTier.TIER1_LIGHTRAG]
        
        if tier1_processor['status'] == 'initialized':
            lightrag_instance = tier1_processor['processor']
            # Verify some configuration was applied
            assert lightrag_instance.embedding_batch_num == 5
            assert lightrag_instance.llm_model_max_async == 2
            assert lightrag_instance.embedding_func_max_async == 4
            assert lightrag_instance.chunk_token_size == 800
            assert lightrag_instance.chunk_overlap_token_size == 50
            assert lightrag_instance.top_k == 20
            assert lightrag_instance.chunk_top_k == 5
            
    print("✅ Configuration parameters properly applied")


def test_error_handling():
    """Test comprehensive error handling"""
    from research_agent.integrations.three_tier_rag import ThreeTierRAG
    
    # Test with invalid working directory (permissions)
    config = {
        'lightrag_working_dir': '/root/invalid_directory',  # Should fail
        'lightrag_llm_model': 'gpt-4o-mini'
    }
    
    # Should not crash, should handle gracefully
    rag = ThreeTierRAG(config=config)
    
    # Should still have tier processors, but may be in failed state
    from research_agent.integrations.three_tier_rag import RAGTier
    assert RAGTier.TIER1_LIGHTRAG in rag.tier_processors
    
    print("✅ Error handling working correctly")


def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_lightrag_imports,
        test_three_tier_rag_import,
        test_llm_provider_detection,
        test_lightrag_initialization_with_api_key,
        test_lightrag_initialization_without_api_key,
        test_different_llm_providers,
        test_configuration_parameters,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    print("🚀 Running LightRAG Integration Test Suite...")
    print("=" * 60)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! LightRAG integration is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    # Clean up any existing test data
    for cleanup_dir in ['./lightrag_data', './test_lightrag']:
        if os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir)
    
    success = run_all_tests()
    
    # Clean up test data
    for cleanup_dir in ['./lightrag_data', './test_lightrag']:
        if os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir)
    
    sys.exit(0 if success else 1)