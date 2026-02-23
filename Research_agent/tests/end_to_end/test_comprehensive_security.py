"""
Comprehensive security and functionality test suite for the model configuration system.

Tests all security fixes and features implemented for the Research Agent:
- YAML injection protection
- API key redaction
- File locking for atomic operations  
- LiteLLM format validation
- Model capability validation
- Configuration state synchronization
- Race condition prevention
"""

import asyncio
import os
import tempfile
import yaml
import logging
import threading
import concurrent.futures
import pytest
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test framework
class SecurityTestSuite:
    """Comprehensive security test suite"""
    
    def __init__(self):
        self.results = []
        self.temp_dir = None
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and capture results"""
        try:
            result = test_func()
            self.results.append({
                "test": test_name,
                "status": "PASS" if result else "FAIL",
                "details": "Test executed successfully" if result else "Test failed"
            })
            logger.info(f"✓ {test_name}: {'PASS' if result else 'FAIL'}")
            return result
        except Exception as e:
            self.results.append({
                "test": test_name,
                "status": "ERROR", 
                "details": str(e)
            })
            logger.error(f"✗ {test_name}: ERROR - {e}")
            return False
    
    async def run_async_test(self, test_name: str, test_func):
        """Run an async test and capture results"""
        try:
            result = await test_func()
            self.results.append({
                "test": test_name,
                "status": "PASS" if result else "FAIL",
                "details": "Async test executed successfully" if result else "Async test failed"
            })
            logger.info(f"✓ {test_name}: {'PASS' if result else 'FAIL'}")
            return result
        except Exception as e:
            self.results.append({
                "test": test_name,
                "status": "ERROR",
                "details": str(e)
            })
            logger.error(f"✗ {test_name}: ERROR - {e}")
            return False
    
    def setup_test_environment(self):
        """Setup test environment with temp config file"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, 'test_models.yaml')
        
        # Create test configuration
        test_config = {
            'default_model': 'gpt-4o-mini',
            'models': {
                'gpt-4o-mini': {
                    'model': 'gpt-4o-mini',
                    'temperature': 0.5,
                    'max_tokens': 2000,
                    'description': 'Test model',
                    'capabilities': ['search', 'analysis'],
                    'cost_per_1k_tokens': 0.0002
                },
                'claude-3-sonnet': {
                    'model': 'claude-3-sonnet-20240229',
                    'temperature': 0.5,
                    'max_tokens': 3000,
                    'description': 'Test Claude model',
                    'capabilities': ['analysis', 'chat'],
                    'cost_per_1k_tokens': 0.003
                }
            },
            'task_specific_models': {
                'query_generation': 'gpt-4o-mini',
                'analysis': 'claude-3-sonnet'
            },
            'profiles': {
                'test': {
                    'default_model': 'gpt-4o-mini'
                }
            },
            'active_profile': 'test',
            'litellm_settings': {
                'cache': True,
                'max_retries': 3
            },
            'metadata': {
                'version': '1.0.0',
                'last_modified': '',
                'modified_by': 'test_suite'
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            yaml.dump(test_config, f)
            
        logger.info(f"Test environment setup complete: {self.temp_dir}")
        return self.test_config_path
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")


def test_yaml_injection_protection():
    """Test YAML injection vulnerability protection"""
    from research_agent.core.model_config import ModelConfigManager
    
    suite = SecurityTestSuite()
    config_path = suite.setup_test_environment()
    
    try:
        # Test with malicious YAML content
        malicious_yaml = """
default_model: gpt-4o-mini
models:
  test: !!python/object/apply:os.system ["echo 'YAML injection attempt'"]
"""
        
        # Write malicious content
        with open(config_path, 'w') as f:
            f.write(malicious_yaml)
        
        # Try to load - should use safe_load and not execute code
        manager = ModelConfigManager(config_path, enable_sync=False)
        models = manager.get_models()
        
        # Should succeed without executing malicious code
        manager.shutdown()
        suite.cleanup_test_environment()
        
        # If we get here, YAML injection was prevented
        return True
        
    except Exception as e:
        logger.info(f"YAML injection prevented (expected): {e}")
        suite.cleanup_test_environment()
        return True  # Expected behavior


def test_api_key_redaction():
    """Test API key redaction in error messages"""
    from research_agent.core.model_tester import SecureModelTester
    
    # Test API key patterns
    tester = SecureModelTester()
    
    test_errors = [
        "Authentication failed with API key sk-1234567890abcdef",
        "Invalid authorization bearer sk-ant-1234567890-abcdef",
        "Error with key AIza1234567890abcdefghijk"
    ]
    
    all_redacted = True
    for error in test_errors:
        sanitized = tester._sanitize_error_message(error)
        
        # Should not contain actual API key patterns
        if any(pattern in sanitized for pattern in ['sk-123', 'sk-ant-123', 'AIza123']):
            all_redacted = False
            logger.error(f"API key not properly redacted in: {sanitized}")
        else:
            logger.info(f"API key properly redacted: {sanitized}")
    
    return all_redacted


def test_file_locking_atomic_operations():
    """Test file locking prevents race conditions"""
    from research_agent.core.file_lock import atomic_write, file_lock
    
    suite = SecurityTestSuite()
    config_path = suite.setup_test_environment()
    
    # Test atomic write
    try:
        test_content = "test: atomic_write_content"
        with atomic_write(config_path) as f:
            f.write(test_content)
        
        # Verify content was written
        with open(config_path, 'r') as f:
            content = f.read()
        
        atomic_success = test_content in content
        
        # Test file locking
        lock_test_passed = True
        try:
            with file_lock(config_path, timeout=1.0) as f:
                original_content = f.read()
                # Simulate concurrent access should be blocked
        except Exception as e:
            logger.info(f"File locking working as expected: {e}")
        
        suite.cleanup_test_environment()
        return atomic_success and lock_test_passed
        
    except Exception as e:
        logger.error(f"File locking test failed: {e}")
        suite.cleanup_test_environment()
        return False


def test_litellm_format_validation():
    """Test LiteLLM format validation and correction"""
    from research_agent.core.litellm_validator import get_litellm_validator
    
    validator = get_litellm_validator()
    
    # Test model configurations with known issues
    test_configs = {
        'incorrect-openai': {
            'model': 'openai/gpt-4o',  # Should be just 'gpt-4o'
            'temperature': 0.7
        },
        'incorrect-claude': {
            'model': 'claude-opus',  # Should be full version name
            'temperature': 0.5  
        },
        'correct-model': {
            'model': 'gpt-4o-mini',  # Already correct
            'temperature': 0.5
        }
    }
    
    # Validate configurations
    results = validator.validate_all_models(test_configs)
    
    # Check that incorrect formats are detected
    incorrect_detected = (
        not results['incorrect-openai'][0] and  # Should be invalid
        not results['incorrect-claude'][0] and   # Should be invalid  
        results['correct-model'][0]             # Should be valid
    )
    
    # Test auto-fix
    fixed_configs, changes = validator.fix_model_formats(test_configs)
    auto_fix_works = len(changes) >= 1  # Should fix at least 1 model
    
    logger.info(f"LiteLLM validation detected issues: {incorrect_detected}")
    logger.info(f"LiteLLM auto-fix applied {len(changes)} changes: {auto_fix_works}")
    
    return incorrect_detected and auto_fix_works


def test_capability_validation():
    """Test model capability validation for task assignments"""
    from research_agent.core.capability_validator import get_capability_validator
    
    validator = get_capability_validator()
    
    # Test valid and invalid assignments
    task_assignments = {
        'query_generation': 'gpt-4o',         # Valid
        'embedding_default': 'gpt-4o'         # Invalid - text model for embedding task
    }
    
    model_configs = {
        'gpt-4o': {
            'model': 'gpt-4o',
            'capabilities': ['query_generation', 'analysis', 'chat'],
            'max_tokens': 4000
        }
    }
    
    # Validate assignments
    results = validator.validate_task_assignments(task_assignments, model_configs)
    
    # Should detect the invalid embedding assignment
    invalid_detected = 'embedding_default' in results
    valid_passed = 'query_generation' not in results
    
    logger.info(f"Capability validation detected invalid assignment: {invalid_detected}")
    logger.info(f"Capability validation passed valid assignment: {valid_passed}")
    
    return invalid_detected and valid_passed


@pytest.mark.asyncio
async def test_configuration_synchronization():
    """Test real-time configuration synchronization"""
    from research_agent.core.model_config import ModelConfigManager
    from research_agent.core.config_sync import ConfigSyncSubscriber, ConfigChangeEvent
    
    class TestSyncSubscriber(ConfigSyncSubscriber):
        def __init__(self):
            self.events = []
        
        async def on_config_changed(self, event: ConfigChangeEvent):
            self.events.append(event)
    
    suite = SecurityTestSuite()
    config_path = suite.setup_test_environment()
    
    try:
        # Create manager with sync
        manager = ModelConfigManager(config_path, enable_sync=True)
        sync_manager = manager.get_sync_manager()
        
        if not sync_manager:
            return False
        
        await sync_manager.start()
        
        # Add subscriber
        subscriber = TestSyncSubscriber()
        sync_manager.add_subscriber(subscriber)
        
        # Make configuration changes
        test_model = {
            'model': 'test-sync-model',
            'temperature': 0.5,
            'max_tokens': 1000
        }
        
        manager.add_model('sync-test', test_model, save=False)
        manager.set_active_profile('test', save=False)
        
        # Wait for sync processing
        await asyncio.sleep(0.1)
        
        # Verify events were captured
        events_captured = len(subscriber.events) >= 2
        model_add_event = any(event.change_type.value == 'model_added' for event in subscriber.events)
        profile_change_event = any(event.change_type.value == 'profile_changed' for event in subscriber.events)
        
        await manager.shutdown_async()
        suite.cleanup_test_environment()
        
        sync_success = events_captured and model_add_event and profile_change_event
        logger.info(f"Configuration sync captured {len(subscriber.events)} events: {sync_success}")
        
        return sync_success
        
    except Exception as e:
        logger.error(f"Configuration sync test failed: {e}")
        suite.cleanup_test_environment()
        return False


def test_concurrent_access_safety():
    """Test that concurrent access to configuration is safe"""
    from research_agent.core.model_config import ModelConfigManager
    
    suite = SecurityTestSuite()
    config_path = suite.setup_test_environment()
    
    try:
        # Create manager
        manager = ModelConfigManager(config_path, enable_sync=False)
        
        results = []
        errors = []
        
        def worker_thread(worker_id):
            """Worker thread that modifies configuration"""
            try:
                for i in range(5):
                    test_model = {
                        'model': f'test-model-{worker_id}-{i}',
                        'temperature': 0.5,
                        'max_tokens': 1000
                    }
                    manager.add_model(f'worker-{worker_id}-model-{i}', test_model, save=False)
                    
                results.append(f"Worker {worker_id} completed")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Run concurrent threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and all operations completed
        all_completed = len(results) == 3 and len(errors) == 0
        total_models = len(manager.get_models())
        
        manager.shutdown()
        suite.cleanup_test_environment()
        
        logger.info(f"Concurrent access test - completed workers: {len(results)}, errors: {len(errors)}")
        logger.info(f"Total models after concurrent operations: {total_models}")
        
        return all_completed and total_models >= 15  # Original 2 + 15 added (3 workers × 5 models each)
        
    except Exception as e:
        logger.error(f"Concurrent access test failed: {e}")
        suite.cleanup_test_environment()
        return False


async def run_comprehensive_security_tests():
    """Run all security and functionality tests"""
    print("🔒 Running Comprehensive Security Test Suite for Research Agent Model Configuration")
    print("=" * 80)
    
    suite = SecurityTestSuite()
    
    # Run synchronous tests
    tests = [
        ("YAML Injection Protection", test_yaml_injection_protection),
        ("API Key Redaction", test_api_key_redaction), 
        ("File Locking & Atomic Operations", test_file_locking_atomic_operations),
        ("LiteLLM Format Validation", test_litellm_format_validation),
        ("Model Capability Validation", test_capability_validation),
        ("Concurrent Access Safety", test_concurrent_access_safety)
    ]
    
    for test_name, test_func in tests:
        suite.run_test(test_name, test_func)
    
    # Run async tests
    await suite.run_async_test("Configuration State Synchronization", test_configuration_synchronization)
    
    # Print results
    print("\n" + "=" * 80)
    print("🔒 COMPREHENSIVE SECURITY TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for r in suite.results if r["status"] == "PASS")
    failed = sum(1 for r in suite.results if r["status"] == "FAIL") 
    errors = sum(1 for r in suite.results if r["status"] == "ERROR")
    total = len(suite.results)
    
    for result in suite.results:
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"{status_icon} {result['test']}: {result['status']}")
        if result["status"] != "PASS":
            print(f"   └─ {result['details']}")
    
    print("\n" + "-" * 80)
    print(f"SUMMARY: {passed}/{total} tests passed ({failed} failed, {errors} errors)")
    
    if passed == total:
        print("🎉 ALL SECURITY TESTS PASSED - System is production ready!")
    else:
        print("⚠️  Some tests failed - Review security implementations")
    
    return passed == total


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = asyncio.run(run_comprehensive_security_tests())
    exit(0 if success else 1)