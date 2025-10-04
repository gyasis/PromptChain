"""
Comprehensive tests for the Model Configuration Management System

Tests the ModelConfigManager, validation, API endpoints, and integration
with the existing research system.
"""

import os
import pytest
import tempfile
import threading
import time
import yaml
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from typing import Dict, Any

# Imports now work with proper package structure

from research_agent.core.model_config import (
    ModelConfigManager, ModelInfo, ModelProfile, 
    get_model_config, integrate_with_research_config
)
from research_agent.core.model_validation import (
    ModelConfigValidator, ValidationSeverity, validate_model_config_file,
    format_validation_report
)
from research_agent.core.config import ResearchConfig


class TestModelInfo:
    """Test ModelInfo dataclass"""
    
    def test_model_info_creation(self):
        """Test creating ModelInfo with basic data"""
        model = ModelInfo(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4000,
            description="Test model",
            capabilities=["analysis", "synthesis"],
            cost_per_1k_tokens=0.01
        )
        
        assert model.model == "gpt-4o"
        assert model.provider == "openai"  # Auto-detected
        assert model.temperature == 0.7
        assert model.capabilities == ["analysis", "synthesis"]
    
    def test_provider_detection(self):
        """Test automatic provider detection"""
        test_cases = [
            ("gpt-4o", "openai"),
            ("claude-3-opus-20240229", "anthropic"),
            ("gemini/gemini-1.5-pro", "google"),
            ("ollama/llama3.2", "ollama"),
            ("unknown-model", "unknown")
        ]
        
        for model_name, expected_provider in test_cases:
            model = ModelInfo(model=model_name)
            assert model.provider == expected_provider


class TestModelConfigValidator:
    """Test model configuration validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = ModelConfigValidator()
        self.valid_config = {
            'default_model': 'gpt-4o-mini',
            'models': {
                'gpt-4o': {
                    'model': 'gpt-4o',
                    'temperature': 0.7,
                    'max_tokens': 4000,
                    'description': 'Test model',
                    'capabilities': ['analysis', 'synthesis'],
                    'cost_per_1k_tokens': 0.01
                },
                'claude-3-sonnet': {
                    'model': 'claude-3-sonnet-20240229',
                    'temperature': 0.5,
                    'max_tokens': 3000,
                    'capabilities': ['analysis'],
                    'cost_per_1k_tokens': 0.003
                }
            },
            'task_specific_models': {
                'query_generation': 'gpt-4o',
                'analysis': 'claude-3-sonnet'
            },
            'profiles': {
                'default': {
                    'default_model': 'gpt-4o',
                    'description': 'Default profile'
                }
            },
            'active_profile': 'default',
            'metadata': {
                'version': '1.0.0'
            }
        }
    
    def test_valid_config(self):
        """Test validation of a valid configuration"""
        is_valid, messages = self.validator.validate(self.valid_config)
        assert is_valid is True
        # Should have some info messages but no errors
        assert not any(msg.severity == ValidationSeverity.ERROR for msg in messages)
    
    def test_missing_required_sections(self):
        """Test validation fails for missing required sections"""
        invalid_config = {
            'models': {}
            # Missing task_specific_models and profiles
        }
        
        is_valid, messages = self.validator.validate(invalid_config)
        assert is_valid is False
        
        error_messages = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
        assert len(error_messages) >= 2  # At least missing sections
        
        error_paths = {msg.path for msg in error_messages}
        assert 'root.task_specific_models' in error_paths
        assert 'root.profiles' in error_paths
    
    def test_invalid_model_config(self):
        """Test validation of invalid model configuration"""
        invalid_config = self.valid_config.copy()
        invalid_config['models']['invalid-model'] = {
            'temperature': 3.0,  # Invalid: > 2.0
            'max_tokens': 0,     # Invalid: < 1
            'capabilities': ['invalid_capability']  # Invalid capability
            # Missing required 'model' field
        }
        
        is_valid, messages = self.validator.validate(invalid_config)
        assert is_valid is False
        
        error_messages = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
        assert len(error_messages) >= 3  # Missing model field, invalid temp, invalid max_tokens
    
    def test_cross_reference_validation(self):
        """Test validation of cross-references between sections"""
        invalid_config = self.valid_config.copy()
        invalid_config['default_model'] = 'nonexistent-model'
        invalid_config['task_specific_models']['query_generation'] = 'another-nonexistent'
        
        is_valid, messages = self.validator.validate(invalid_config)
        assert is_valid is False
        
        error_messages = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
        assert len(error_messages) >= 2
    
    def test_model_name_format_validation(self):
        """Test model name format validation"""
        invalid_config = self.valid_config.copy()
        invalid_config['models']['Invalid Model Name!'] = {
            'model': 'gpt-4o',
            'capabilities': ['analysis']
        }
        
        is_valid, messages = self.validator.validate(invalid_config)
        assert is_valid is False
        
        error_messages = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
        name_format_errors = [msg for msg in error_messages if 'Invalid model name format' in msg.message]
        assert len(name_format_errors) == 1
    
    def test_capabilities_validation(self):
        """Test capabilities validation"""
        config = self.valid_config.copy()
        config['models']['test-model'] = {
            'model': 'test-model',
            'capabilities': ['unknown_capability', 'analysis']
        }
        
        is_valid, messages = self.validator.validate(config)
        # Should be valid but with warnings
        assert is_valid is True
        
        warning_messages = [msg for msg in messages if msg.severity == ValidationSeverity.WARNING]
        capability_warnings = [msg for msg in warning_messages if 'Unknown capability' in msg.message]
        assert len(capability_warnings) == 1
    
    def test_ollama_model_validation(self):
        """Test Ollama-specific validation"""
        config = self.valid_config.copy()
        config['models']['ollama-model'] = {
            'model': 'ollama/llama3.2',
            'capabilities': ['analysis']
            # Missing api_base - should generate warning
        }
        
        is_valid, messages = self.validator.validate(config)
        assert is_valid is True
        
        warning_messages = [msg for msg in messages if msg.severity == ValidationSeverity.WARNING]
        ollama_warnings = [msg for msg in warning_messages if 'should specify api_base' in msg.message]
        assert len(ollama_warnings) == 1


class TestModelConfigManager:
    """Test ModelConfigManager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_models.yaml')
        
        # Create test configuration
        self.test_config = {
            'default_model': 'gpt-4o-mini',
            'models': {
                'gpt-4o': {
                    'model': 'gpt-4o',
                    'temperature': 0.7,
                    'max_tokens': 4000,
                    'description': 'Test GPT-4O model',
                    'capabilities': ['analysis', 'synthesis', 'chat'],
                    'cost_per_1k_tokens': 0.01
                },
                'claude-3-sonnet': {
                    'model': 'claude-3-sonnet-20240229',
                    'temperature': 0.5,
                    'max_tokens': 3000,
                    'description': 'Test Claude model',
                    'capabilities': ['analysis', 'multi_query'],
                    'cost_per_1k_tokens': 0.003
                },
                'local-llama': {
                    'model': 'ollama/llama3.2',
                    'api_base': 'http://localhost:11434',
                    'temperature': 0.6,
                    'max_tokens': 4000,
                    'capabilities': ['analysis', 'chat'],
                    'cost_per_1k_tokens': 0.0
                }
            },
            'task_specific_models': {
                'query_generation': 'gpt-4o',
                'analysis': 'claude-3-sonnet',
                'synthesis': 'gpt-4o',
                'chat_interface': 'local-llama'
            },
            'profiles': {
                'premium': {
                    'default_model': 'gpt-4o',
                    'description': 'Premium profile with best models',
                    'query_generation': 'gpt-4o',
                    'analysis': 'claude-3-sonnet'
                },
                'local': {
                    'default_model': 'local-llama',
                    'description': 'Local-only profile',
                    'query_generation': 'local-llama',
                    'analysis': 'local-llama'
                }
            },
            'active_profile': 'premium',
            'litellm_settings': {
                'cache': True,
                'max_retries': 3,
                'request_timeout': 120
            },
            'metadata': {
                'version': '1.0.0',
                'last_modified': '2025-01-13T00:00:00Z'
            }
        }
        
        # Write config file
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.config_file):
            os.unlink(self.config_file)
        os.rmdir(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test ModelConfigManager initialization"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        assert manager.config_path == self.config_file
        assert not manager.auto_reload
        assert len(manager.get_models()) == 3
    
    def test_get_models(self):
        """Test retrieving all models"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        models = manager.get_models()
        
        assert len(models) == 3
        assert 'gpt-4o' in models
        assert 'claude-3-sonnet' in models
        assert 'local-llama' in models
        
        gpt_model = models['gpt-4o']
        assert gpt_model.model == 'gpt-4o'
        assert gpt_model.provider == 'openai'
        assert gpt_model.temperature == 0.7
        assert 'analysis' in gpt_model.capabilities
    
    def test_get_model(self):
        """Test retrieving specific model"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        model = manager.get_model('claude-3-sonnet')
        assert model is not None
        assert model.model == 'claude-3-sonnet-20240229'
        assert model.provider == 'anthropic'
        
        # Test nonexistent model
        assert manager.get_model('nonexistent') is None
    
    def test_get_model_for_task(self):
        """Test task-specific model retrieval"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        # Test default profile task assignment
        analysis_model = manager.get_model_for_task('analysis')
        assert analysis_model is not None
        assert analysis_model.model == 'claude-3-sonnet-20240229'
        
        # Test profile-specific task assignment
        query_model = manager.get_model_for_task('query_generation', 'premium')
        assert query_model is not None
        assert query_model.model == 'gpt-4o'
        
        # Test fallback to global task assignment
        synthesis_model = manager.get_model_for_task('synthesis')
        assert synthesis_model is not None
        assert synthesis_model.model == 'gpt-4o'
    
    def test_add_model(self):
        """Test adding new model"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        new_model_config = {
            'model': 'gemini/gemini-1.5-pro',
            'temperature': 0.6,
            'max_tokens': 8000,
            'description': 'Test Gemini model',
            'capabilities': ['analysis', 'synthesis'],
            'cost_per_1k_tokens': 0.0035
        }
        
        manager.add_model('gemini-pro', new_model_config, save=False)
        
        models = manager.get_models()
        assert len(models) == 4
        assert 'gemini-pro' in models
        
        gemini_model = models['gemini-pro']
        assert gemini_model.model == 'gemini/gemini-1.5-pro'
        assert gemini_model.provider == 'google'
    
    def test_update_model(self):
        """Test updating existing model"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        update_config = {
            'model': 'gpt-4o',
            'temperature': 0.8,  # Changed
            'max_tokens': 6000,  # Changed
            'description': 'Updated GPT-4O model',  # Changed
            'capabilities': ['analysis', 'synthesis', 'chat'],
            'cost_per_1k_tokens': 0.015  # Changed
        }
        
        manager.update_model('gpt-4o', update_config, save=False)
        
        updated_model = manager.get_model('gpt-4o')
        assert updated_model.temperature == 0.8
        assert updated_model.max_tokens == 6000
        assert updated_model.description == 'Updated GPT-4O model'
        assert updated_model.cost_per_1k_tokens == 0.015
    
    def test_remove_model(self):
        """Test removing model"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        # Should not be able to remove model that's in use
        with pytest.raises(ValueError, match="currently in use"):
            manager.remove_model('gpt-4o', save=False)
        
        # Remove model not in use
        manager.remove_model('local-llama', save=False)
        
        models = manager.get_models()
        assert len(models) == 2
        assert 'local-llama' not in models
    
    def test_profiles(self):
        """Test profile management"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        profiles = manager.get_profiles()
        assert len(profiles) == 2
        assert 'premium' in profiles
        assert 'local' in profiles
        
        premium_profile = profiles['premium']
        assert premium_profile.default_model == 'gpt-4o'
        assert premium_profile.description == 'Premium profile with best models'
        
        # Test active profile
        assert manager.get_active_profile() == 'premium'
        
        # Test switching profile
        manager.set_active_profile('local', save=False)
        assert manager.get_active_profile() == 'local'
    
    def test_litellm_settings(self):
        """Test LiteLLM settings management"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        settings = manager.get_litellm_settings()
        assert settings['cache'] is True
        assert settings['max_retries'] == 3
        assert settings['request_timeout'] == 120
        
        # Update settings
        new_settings = {'max_retries': 5, 'request_timeout': 180}
        manager.update_litellm_settings(new_settings, save=False)
        
        updated_settings = manager.get_litellm_settings()
        assert updated_settings['max_retries'] == 5
        assert updated_settings['request_timeout'] == 180
        assert updated_settings['cache'] is True  # Should remain unchanged
    
    def test_validation(self):
        """Test configuration validation"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        # Test valid configuration
        is_valid, messages = manager.validate_config()
        assert is_valid is True
        
        # Test invalid configuration
        invalid_config = {'models': {}}  # Missing required sections
        is_valid, messages = manager.validate_config(invalid_config)
        assert is_valid is False
        assert len(messages) > 0
    
    @patch('research_agent.core.model_config.litellm')
    def test_model_connection_test(self, mock_litellm):
        """Test model connection testing"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Connection test successful."
        mock_litellm.completion.return_value = mock_response
        
        result = manager.test_model_connection('gpt-4o')
        assert result['success'] is True
        assert 'response_time' in result
        assert result['response'] == "Connection test successful."
        assert result['provider'] == 'openai'
        
        # Mock failure
        mock_litellm.completion.side_effect = Exception("Connection failed")
        
        result = manager.test_model_connection('gpt-4o')
        assert result['success'] is False
        assert result['error'] == "Connection failed"
    
    def test_thread_safety(self):
        """Test thread-safe operations"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        def add_models():
            for i in range(10):
                manager.add_model(f'test-model-{i}', {
                    'model': f'test-model-{i}',
                    'capabilities': ['test']
                }, save=False)
        
        # Run concurrent operations
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=add_models)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        models = manager.get_models()
        # Should have original 3 + 50 added models
        assert len(models) == 53


class TestResearchConfigIntegration:
    """Test integration with existing ResearchConfig"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_config_file = os.path.join(self.temp_dir, 'models.yaml')
        self.research_config_file = os.path.join(self.temp_dir, 'research_config.yaml')
        
        # Create test model config
        model_config = {
            'default_model': 'gpt-4o-mini',
            'models': {
                'gpt-4o-mini': {
                    'model': 'gpt-4o-mini',
                    'temperature': 0.5,
                    'max_tokens': 2000,
                    'capabilities': ['analysis'],
                    'cost_per_1k_tokens': 0.0002
                }
            },
            'task_specific_models': {
                'query_generation': 'gpt-4o-mini'
            },
            'profiles': {
                'default': {'default_model': 'gpt-4o-mini'}
            },
            'active_profile': 'default',
            'metadata': {'version': '1.0.0'}
        }
        
        with open(self.model_config_file, 'w') as f:
            yaml.dump(model_config, f)
        
        # Create minimal research config
        research_config = {
            'system': {
                'name': 'Test Research Agent',
                'version': '0.1.0'
            },
            'literature_search': {
                'enabled_sources': ['arxiv']
            },
            'three_tier_rag': {
                'execution_mode': 'sequential'
            },
            'promptchain': {
                'agents': {}
            },
            'research_session': {
                'max_iterations': 3
            }
        }
        
        with open(self.research_config_file, 'w') as f:
            yaml.dump(research_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        for file_path in [self.model_config_file, self.research_config_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)
        os.rmdir(self.temp_dir)
    
    @patch('research_agent.core.config.get_model_config')
    def test_integration(self, mock_get_model_config):
        """Test ResearchConfig integration with ModelConfigManager"""
        # Create mock model config manager
        mock_manager = Mock()
        mock_manager.get_model_for_task.return_value = ModelInfo(
            model='gpt-4o-mini',
            temperature=0.5,
            max_tokens=2000,
            capabilities=['analysis']
        )
        mock_get_model_config.return_value = mock_manager
        
        # Create ResearchConfig instance
        config = ResearchConfig(self.research_config_file)
        
        # Test model retrieval with fallback
        model_config = config.get_model_for_task_with_fallback('query_generation')
        assert model_config is not None
        assert 'name' in model_config
        assert 'params' in model_config
        
        # Test available models listing
        mock_manager.get_models.return_value = {
            'gpt-4o-mini': ModelInfo(model='gpt-4o-mini', provider='openai', capabilities=['analysis'])
        }
        
        available_models = config.list_available_models()
        assert 'gpt-4o-mini' in available_models
        assert available_models['gpt-4o-mini']['provider'] == 'openai'


class TestAPIEndpoints:
    """Test FastAPI endpoints for model management"""
    
    def setup_method(self):
        """Set up test client and fixtures"""
        # Create test FastAPI app with just the models router
        from fastapi import FastAPI
        from backend.api.models import router as models_router
        
        app = FastAPI()
        app.include_router(models_router, prefix="/api/models")
        
        self.client = TestClient(app)
        
        # Mock ModelConfigManager
        self.mock_manager = Mock()
        self.mock_models = {
            'gpt-4o': ModelInfo(
                model='gpt-4o',
                provider='openai',
                description='Test GPT-4O model',
                capabilities=['analysis', 'synthesis'],
                temperature=0.7,
                max_tokens=4000,
                cost_per_1k_tokens=0.01
            ),
            'claude-3-sonnet': ModelInfo(
                model='claude-3-sonnet-20240229',
                provider='anthropic',
                description='Test Claude model',
                capabilities=['analysis'],
                temperature=0.5,
                max_tokens=3000,
                cost_per_1k_tokens=0.003
            )
        }
        self.mock_manager.get_models.return_value = self.mock_models
    
    @patch('backend.api.models.get_model_config_manager')
    def test_list_models(self, mock_get_manager):
        """Test GET /api/models/ endpoint"""
        mock_get_manager.return_value = self.mock_manager
        
        response = self.client.get("/api/models/")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data['models']) == 2
        assert data['total_count'] == 2
        
        # Check model details
        gpt_model = next(m for m in data['models'] if m['name'] == 'gpt-4o')
        assert gpt_model['provider'] == 'openai'
        assert gpt_model['model'] == 'gpt-4o'
        assert 'analysis' in gpt_model['capabilities']
    
    @patch('backend.api.models.get_model_config_manager')
    def test_get_model(self, mock_get_manager):
        """Test GET /api/models/{model_id} endpoint"""
        mock_get_manager.return_value = self.mock_manager
        self.mock_manager.get_model.return_value = self.mock_models['gpt-4o']
        
        response = self.client.get("/api/models/gpt-4o")
        assert response.status_code == 200
        
        data = response.json()
        assert data['name'] == 'gpt-4o'
        assert data['provider'] == 'openai'
        assert data['temperature'] == 0.7
    
    @patch('backend.api.models.get_model_config_manager')
    def test_add_model(self, mock_get_manager):
        """Test POST /api/models/ endpoint"""
        mock_get_manager.return_value = self.mock_manager
        
        new_model = {
            'name': 'gemini-pro',
            'model': 'gemini/gemini-1.5-pro',
            'temperature': 0.6,
            'max_tokens': 8000,
            'description': 'Test Gemini model',
            'capabilities': ['analysis', 'synthesis'],
            'cost_per_1k_tokens': 0.0035
        }
        
        response = self.client.post("/api/models/", json=new_model)
        assert response.status_code == 200
        
        data = response.json()
        assert 'Model \'gemini-pro\' added successfully' in data['message']
        
        # Verify manager was called correctly
        self.mock_manager.add_model.assert_called_once()
        call_args = self.mock_manager.add_model.call_args
        assert call_args[0][0] == 'gemini-pro'  # model name
        assert call_args[0][1]['model'] == 'gemini/gemini-1.5-pro'
    
    @patch('backend.api.models.get_model_config_manager')
    def test_model_validation(self, mock_get_manager):
        """Test POST /api/models/validate endpoint"""
        mock_get_manager.return_value = self.mock_manager
        
        # Mock validation response
        from research_agent.core.model_validation import ValidationMessage, ValidationSeverity
        mock_messages = [
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                message='Test warning',
                path='test.path',
                suggestion='Test suggestion'
            )
        ]
        self.mock_manager.validate_config.return_value = (True, mock_messages)
        
        response = self.client.post("/api/models/validate")
        assert response.status_code == 200
        
        data = response.json()
        assert data['is_valid'] is True
        assert len(data['messages']) == 1
        assert data['summary']['warnings'] == 1
        
        message = data['messages'][0]
        assert message['message'] == 'Test warning'
        assert message['path'] == 'test.path'
        assert message['suggestion'] == 'Test suggestion'


class TestFileValidation:
    """Test file-based validation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_file = os.path.join(self.temp_dir, 'valid_models.yaml')
        self.invalid_file = os.path.join(self.temp_dir, 'invalid_models.yaml')
    
    def teardown_method(self):
        """Clean up test fixtures"""
        for file_path in [self.valid_file, self.invalid_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)
        os.rmdir(self.temp_dir)
    
    def test_validate_valid_file(self):
        """Test validation of valid configuration file"""
        valid_config = {
            'default_model': 'gpt-4o',
            'models': {
                'gpt-4o': {
                    'model': 'gpt-4o',
                    'temperature': 0.7,
                    'capabilities': ['analysis']
                }
            },
            'task_specific_models': {
                'analysis': 'gpt-4o'
            },
            'profiles': {
                'default': {'default_model': 'gpt-4o'}
            }
        }
        
        with open(self.valid_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        is_valid, messages = validate_model_config_file(self.valid_file)
        assert is_valid is True
        
        # Format report
        report = format_validation_report(messages, show_info=False)
        assert '✅' in report or '⚠️' in report  # Success or warnings only
    
    def test_validate_invalid_file(self):
        """Test validation of invalid configuration file"""
        invalid_config = {
            'models': {
                'invalid-model': {
                    'temperature': 3.0  # Invalid
                    # Missing required 'model' field
                }
            }
            # Missing required sections
        }
        
        with open(self.invalid_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        is_valid, messages = validate_model_config_file(self.invalid_file)
        assert is_valid is False
        assert len(messages) > 0
        
        # Check for error messages
        error_messages = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
        assert len(error_messages) > 0
        
        # Format report
        report = format_validation_report(messages)
        assert '❌' in report
        assert 'ERRORS' in report
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file"""
        nonexistent_file = os.path.join(self.temp_dir, 'nonexistent.yaml')
        
        is_valid, messages = validate_model_config_file(nonexistent_file)
        assert is_valid is False
        assert len(messages) == 1
        assert 'not found' in messages[0].message
    
    def test_validate_invalid_yaml(self):
        """Test validation of file with invalid YAML syntax"""
        with open(self.invalid_file, 'w') as f:
            f.write("invalid: yaml: syntax: [")
        
        is_valid, messages = validate_model_config_file(self.invalid_file)
        assert is_valid is False
        assert len(messages) == 1
        assert 'YAML parsing error' in messages[0].message


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations and concurrent access"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_models.yaml')
        
        test_config = {
            'default_model': 'gpt-4o-mini',
            'models': {
                'gpt-4o-mini': {
                    'model': 'gpt-4o-mini',
                    'capabilities': ['analysis']
                }
            },
            'task_specific_models': {},
            'profiles': {'default': {'default_model': 'gpt-4o-mini'}},
            'metadata': {'version': '1.0.0'}
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.config_file):
            os.unlink(self.config_file)
        os.rmdir(self.temp_dir)
    
    async def test_concurrent_model_operations(self):
        """Test concurrent model operations"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        async def add_models(start_id: int, count: int):
            """Add multiple models concurrently"""
            for i in range(count):
                model_name = f'test-model-{start_id + i}'
                model_config = {
                    'model': model_name,
                    'capabilities': ['test']
                }
                manager.add_model(model_name, model_config, save=False)
        
        # Run concurrent operations
        import asyncio
        await asyncio.gather(
            add_models(0, 10),
            add_models(10, 10),
            add_models(20, 10)
        )
        
        models = manager.get_models()
        assert len(models) == 31  # Original 1 + 30 added
    
    @patch('research_agent.core.model_config.litellm')
    async def test_concurrent_connection_tests(self, mock_litellm):
        """Test concurrent model connection tests"""
        manager = ModelConfigManager(self.config_file, auto_reload=False)
        
        # Mock successful responses with delay
        async def mock_completion(**kwargs):
            await asyncio.sleep(0.1)  # Simulate network delay
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test successful"
            return mock_response
        
        mock_litellm.completion = mock_completion
        
        # Run concurrent tests
        import asyncio
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, manager.test_model_connection, 'gpt-4o-mini'
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result['success'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])