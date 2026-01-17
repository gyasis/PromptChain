"""
Model Configuration Management System for Research Agent

Provides centralized model configuration management with file watching, 
thread-safe operations, and seamless integration with both CLI and web interfaces.
Uses LiteLLM format for unified model access across all providers.
"""

import os
import yaml
import json
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
from .model_validation import ModelConfigValidator, ValidationSeverity
from .file_lock import atomic_write, file_lock, ConfigurationLock
from .capability_validator import get_capability_validator
from .litellm_validator import get_litellm_validator, validate_litellm_config
from .config_sync import ConfigurationSyncManager, ConfigChangeType, setup_model_config_sync

# Import watchdog for file monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# Import LiteLLM for model validation
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Structured model information"""
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    cost_per_1k_tokens: float = 0.0
    api_base: Optional[str] = None
    embedding_dim: Optional[int] = None
    provider: Optional[str] = None
    
    def __post_init__(self):
        """Extract provider from model name if not specified"""
        if not self.provider:
            if self.model.startswith("gpt-") or self.model.startswith("text-"):
                self.provider = "openai"
            elif self.model.startswith("claude-"):
                self.provider = "anthropic"
            elif self.model.startswith("gemini/"):
                self.provider = "google"
            elif self.model.startswith("ollama/"):
                self.provider = "ollama"
            else:
                self.provider = "unknown"


@dataclass
class ModelProfile:
    """Model profile configuration"""
    name: str
    default_model: str
    task_assignments: Dict[str, str] = field(default_factory=dict)
    description: str = ""


class ModelConfigFileHandler(FileSystemEventHandler):
    """File system event handler for model configuration changes"""
    
    def __init__(self, callback: Callable):
        self.callback = callback
        self._last_modified = {}
        self._debounce_delay = 0.5  # 500ms debounce
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        now = time.time()
        
        # Debounce rapid file changes
        if file_path in self._last_modified:
            if now - self._last_modified[file_path] < self._debounce_delay:
                return
                
        self._last_modified[file_path] = now
        
        if file_path.endswith(('.yaml', '.yml')):
            logger.info(f"Model config file changed: {file_path}")
            try:
                self.callback()
            except Exception as e:
                logger.error(f"Error handling config file change: {e}")


class ModelConfigManager:
    """
    Centralized model configuration management system
    
    Features:
    - File-based configuration with auto-reload
    - Thread-safe operations for concurrent access
    - LiteLLM format support
    - Profile management
    - Model validation and testing
    - Integration with existing ResearchConfig
    """
    
    def __init__(self, config_path: Optional[str] = None, auto_reload: bool = True, enable_sync: bool = True):
        self.config_path = config_path or self._find_config_file()
        self.auto_reload = auto_reload
        self.enable_sync = enable_sync
        self._config_data = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._observer = None
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._reload_callbacks: List[Callable] = []
        self._validator = ModelConfigValidator()
        self._last_validation_messages = []
        
        # Setup synchronization manager if enabled
        self._sync_manager = None
        if enable_sync:
            try:
                self._sync_manager = setup_model_config_sync(self)
                logger.info("Configuration synchronization enabled")
            except Exception as e:
                logger.warning(f"Failed to setup configuration sync: {e}")
        
        # Load initial configuration
        self.reload()
        
        # Start file watching if enabled
        if auto_reload and WATCHDOG_AVAILABLE:
            self._start_file_watching()
    
    def _find_config_file(self) -> str:
        """Find the models.yaml configuration file"""
        possible_paths = [
            "./config/models.yaml",
            "./models.yaml",
            os.path.expanduser("~/.research_agent/models.yaml"),
            "/etc/research_agent/models.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Default to the first location
        return possible_paths[0]
    
    def _start_file_watching(self):
        """Start watching the configuration file for changes"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not available, file watching disabled")
            return
            
        try:
            config_dir = os.path.dirname(os.path.abspath(self.config_path))
            
            if not os.path.exists(config_dir):
                logger.warning(f"Config directory does not exist: {config_dir}")
                return
                
            event_handler = ModelConfigFileHandler(self._handle_config_reload)
            self._observer = Observer()
            self._observer.schedule(event_handler, config_dir, recursive=False)
            self._observer.start()
            
            logger.info(f"Started watching config directory: {config_dir}")
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
    
    def _handle_config_reload(self):
        """Handle configuration file reload with thread safety"""
        def reload_task():
            try:
                self.reload()
                # Notify callbacks
                for callback in self._reload_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in reload callback: {e}")
            except Exception as e:
                logger.error(f"Error reloading configuration: {e}")
        
        # Execute reload in thread pool to avoid blocking file watcher
        self._thread_pool.submit(reload_task)
    
    def add_reload_callback(self, callback: Callable):
        """Add callback to be called when configuration is reloaded"""
        self._reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable):
        """Remove reload callback"""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)
    
    def reload(self):
        """Reload configuration from file with thread safety and file locking"""
        with self._lock:
            try:
                if not os.path.exists(self.config_path):
                    logger.warning(f"Configuration file not found: {self.config_path}")
                    self._config_data = self._get_default_config()
                    return
                
                # Use file lock to prevent reading during writes
                with file_lock(self.config_path, timeout=5.0, exclusive=False) as f:
                    new_config = yaml.safe_load(f) or {}
                
                # Validate configuration structure
                self._validate_config(new_config)
                
                # Update configuration data
                self._config_data = new_config
                self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
                
                logger.info(f"Configuration reloaded from {self.config_path}")
                
            except FileNotFoundError:
                logger.error(f"Configuration file not found: {self.config_path}")
                self._config_data = self._get_default_config()
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML configuration: {e}")
                raise ValueError(f"Invalid YAML configuration: {e}")
            except Exception as e:
                logger.error(f"Error reloading configuration: {e}")
                raise
    
    def _validate_config(self, config_data: Dict[str, Any]):
        """Validate configuration structure using comprehensive validator"""
        is_valid, messages = self._validator.validate(config_data, strict=False)
        
        # Log validation messages
        for message in messages:
            if message.severity == ValidationSeverity.ERROR:
                logger.error(f"Config validation error: {message.path}: {message.message}")
            elif message.severity == ValidationSeverity.WARNING:
                logger.warning(f"Config validation warning: {message.path}: {message.message}")
            else:
                logger.info(f"Config validation info: {message.path}: {message.message}")
        
        # Perform LiteLLM format validation
        litellm_valid, litellm_issues = self._validate_litellm_formats(config_data)
        if not litellm_valid:
            is_valid = False
            for model_key, issue_data in litellm_issues.get("issues_by_model", {}).items():
                for issue in issue_data["issues"]:
                    logger.error(f"LiteLLM format error in model '{model_key}': {issue}")
        
        # Raise exception only for errors
        if not is_valid:
            error_messages = [
                msg.message for msg in messages 
                if msg.severity == ValidationSeverity.ERROR
            ]
            # Add LiteLLM errors
            if not litellm_valid:
                for model_key, issue_data in litellm_issues.get("issues_by_model", {}).items():
                    for issue in issue_data["issues"]:
                        error_messages.append(f"Model '{model_key}': {issue}")
            
            raise ValueError(f"Configuration validation failed: {'; '.join(error_messages)}")
        
        # Store validation messages for later access
        self._last_validation_messages = messages
    
    def _validate_litellm_formats(self, config_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate LiteLLM model formats in configuration.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        models = config_data.get("models", {})
        if not models:
            return True, {"all_valid": True, "total_models": 0}
        
        try:
            is_valid, report = validate_litellm_config(models)
            logger.info(f"LiteLLM format validation: {report['valid_models']}/{report['total_models']} models valid")
            
            if not is_valid:
                logger.warning(f"Found {report['invalid_models']} models with format issues")
                
            return is_valid, report
        except Exception as e:
            logger.error(f"Error validating LiteLLM formats: {e}")
            return False, {"error": str(e)}
    
    def fix_litellm_formats(self) -> List[str]:
        """
        Fix LiteLLM format issues in current configuration.
        
        Returns:
            List of changes made
        """
        with self._lock:
            models = self._config_data.get("models", {})
            if not models:
                return []
            
            validator = get_litellm_validator()
            fixed_models, changes = validator.fix_model_formats(models)
            
            if changes:
                # Update configuration with fixed models
                self._config_data["models"] = fixed_models
                self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
                self._config_data['metadata']['modified_by'] = 'litellm_format_fixer'
                
                # Log all changes
                for change in changes:
                    logger.info(f"LiteLLM format fix: {change}")
                
            return changes
    
    def _notify_sync(self, change_type: 'ConfigChangeType', affected_keys: List[str],
                    old_value: Optional[Any] = None, new_value: Optional[Any] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """Notify synchronization manager of configuration change"""
        if self._sync_manager and self.enable_sync:
            try:
                self._sync_manager.notify_change(change_type, affected_keys, old_value, new_value, metadata)
            except Exception as e:
                logger.error(f"Failed to notify configuration sync: {e}")
    
    def get_sync_manager(self) -> Optional['ConfigurationSyncManager']:
        """Get the synchronization manager instance"""
        return self._sync_manager
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not found"""
        return {
            'default_model': 'gpt-4o-mini',
            'models': {
                'gpt-4o-mini': {
                    'model': 'gpt-4o-mini',
                    'temperature': 0.5,
                    'max_tokens': 2000,
                    'description': 'Default fallback model',
                    'capabilities': ['search', 'analysis'],
                    'cost_per_1k_tokens': 0.0002
                }
            },
            'task_specific_models': {},
            'profiles': {
                'default': {
                    'default_model': 'gpt-4o-mini'
                }
            },
            'active_profile': 'default',
            'litellm_settings': {},
            'metadata': {
                'version': '1.0.0',
                'last_modified': datetime.now().isoformat(),
                'modified_by': 'system_default'
            }
        }
    
    def get_models(self) -> Dict[str, ModelInfo]:
        """Get all available models as ModelInfo objects"""
        with self._lock:
            models = {}
            for name, config in self._config_data.get('models', {}).items():
                models[name] = ModelInfo(**config)
            return models
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get specific model configuration"""
        with self._lock:
            model_config = self._config_data.get('models', {}).get(model_name)
            if model_config:
                return ModelInfo(**model_config)
            return None
    
    def get_model_for_task(self, task: str, profile: Optional[str] = None) -> Optional[ModelInfo]:
        """Get model assigned to a specific task"""
        with self._lock:
            # Use specified profile or active profile
            profile_name = profile or self._config_data.get('active_profile', 'default')
            
            # Check profile-specific task assignments first
            profiles = self._config_data.get('profiles', {})
            if profile_name in profiles:
                profile_config = profiles[profile_name]
                if task in profile_config:
                    model_name = profile_config[task]
                    return self.get_model(model_name)
            
            # Fall back to global task assignments
            task_models = self._config_data.get('task_specific_models', {})
            if task in task_models:
                model_name = task_models[task]
                return self.get_model(model_name)
            
            # Use default model
            default_model = self._config_data.get('default_model')
            if default_model:
                return self.get_model(default_model)
            
            return None
    
    def add_model(self, name: str, model_config: Dict[str, Any], save: bool = True):
        """Add a new model configuration"""
        with self._lock:
            # Validate model config
            if 'model' not in model_config:
                raise ValueError("Model configuration must include 'model' field")
            
            # Add to configuration
            if 'models' not in self._config_data:
                self._config_data['models'] = {}
            
            self._config_data['models'][name] = model_config
            self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
            self._config_data['metadata']['modified_by'] = 'api'
            
            # Notify synchronization
            self._notify_sync(ConfigChangeType.MODEL_ADDED, [f"models.{name}"], 
                             old_value=None, new_value=model_config)
            
            if save:
                self.save()
            
            logger.info(f"Added model: {name}")
    
    def update_model(self, name: str, model_config: Dict[str, Any], save: bool = True):
        """Update existing model configuration"""
        with self._lock:
            if name not in self._config_data.get('models', {}):
                raise ValueError(f"Model '{name}' not found")
            
            # Store old value for sync notification
            old_config = self._config_data['models'][name].copy()
            
            # Validate model config
            if 'model' not in model_config:
                raise ValueError("Model configuration must include 'model' field")
            
            # Update configuration
            self._config_data['models'][name] = model_config
            self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
            self._config_data['metadata']['modified_by'] = 'api'
            
            # Notify synchronization
            self._notify_sync(ConfigChangeType.MODEL_UPDATED, [f"models.{name}"], 
                             old_value=old_config, new_value=model_config)
            
            if save:
                self.save()
            
            logger.info(f"Updated model: {name}")
    
    def remove_model(self, name: str, save: bool = True):
        """Remove model configuration"""
        with self._lock:
            if name not in self._config_data.get('models', {}):
                raise ValueError(f"Model '{name}' not found")
            
            # Check if model is in use
            if self._is_model_in_use(name):
                raise ValueError(f"Cannot remove model '{name}' as it's currently in use")
            
            # Store old value for sync notification
            old_config = self._config_data['models'][name].copy()
            
            # Remove from configuration
            del self._config_data['models'][name]
            self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
            self._config_data['metadata']['modified_by'] = 'api'
            
            # Notify synchronization
            self._notify_sync(ConfigChangeType.MODEL_REMOVED, [f"models.{name}"], 
                             old_value=old_config, new_value=None)
            
            if save:
                self.save()
            
            logger.info(f"Removed model: {name}")
    
    def _is_model_in_use(self, model_name: str) -> bool:
        """Check if model is currently assigned to any task or profile"""
        # Check default model
        if self._config_data.get('default_model') == model_name:
            return True
        
        # Check task assignments
        task_models = self._config_data.get('task_specific_models', {})
        if model_name in task_models.values():
            return True
        
        # Check profile assignments
        profiles = self._config_data.get('profiles', {})
        for profile_config in profiles.values():
            if isinstance(profile_config, dict):
                if model_name in profile_config.values():
                    return True
        
        return False
    
    def get_profiles(self) -> Dict[str, ModelProfile]:
        """Get all available profiles"""
        with self._lock:
            profiles = {}
            for name, config in self._config_data.get('profiles', {}).items():
                if isinstance(config, dict):
                    profiles[name] = ModelProfile(
                        name=name,
                        default_model=config.get('default_model', ''),
                        task_assignments=config,
                        description=config.get('description', '')
                    )
            return profiles
    
    def get_active_profile(self) -> str:
        """Get currently active profile"""
        with self._lock:
            return self._config_data.get('active_profile', 'default')
    
    def set_active_profile(self, profile_name: str, save: bool = True):
        """Set active profile"""
        with self._lock:
            profiles = self._config_data.get('profiles', {})
            if profile_name not in profiles:
                raise ValueError(f"Profile '{profile_name}' not found")
            
            # Store old value for sync notification
            old_profile = self._config_data.get('active_profile')
            
            self._config_data['active_profile'] = profile_name
            self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
            self._config_data['metadata']['modified_by'] = 'api'
            
            # Notify synchronization
            self._notify_sync(ConfigChangeType.PROFILE_CHANGED, ["active_profile"], 
                             old_value=old_profile, new_value=profile_name)
            
            if save:
                self.save()
            
            logger.info(f"Set active profile: {profile_name}")
    
    def test_model_connection(self, model_name: str) -> Dict[str, Any]:
        """Test connection to a specific model"""
        model_info = self.get_model(model_name)
        if not model_info:
            return {"success": False, "error": f"Model '{model_name}' not found"}
        
        if not LITELLM_AVAILABLE:
            return {"success": False, "error": "LiteLLM not available for testing"}
        
        try:
            # Simple test with a basic prompt
            test_prompt = "Hello! Please respond with 'Connection test successful.'"
            
            # Prepare model parameters
            model_params = {
                "model": model_info.model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            # Add API base if specified (for local models)
            if model_info.api_base:
                model_params["api_base"] = model_info.api_base
            
            # Execute test
            start_time = time.time()
            response = litellm.completion(**model_params)
            end_time = time.time()
            
            return {
                "success": True,
                "response_time": round(end_time - start_time, 2),
                "response": response.choices[0].message.content if response.choices else "No response",
                "model": model_info.model,
                "provider": model_info.provider
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model_info.model,
                "provider": model_info.provider
            }
    
    def get_litellm_settings(self) -> Dict[str, Any]:
        """Get LiteLLM global settings"""
        with self._lock:
            return self._config_data.get('litellm_settings', {})
    
    def update_litellm_settings(self, settings: Dict[str, Any], save: bool = True):
        """Update LiteLLM global settings"""
        with self._lock:
            self._config_data['litellm_settings'] = {
                **self._config_data.get('litellm_settings', {}),
                **settings
            }
            self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
            self._config_data['metadata']['modified_by'] = 'api'
            
            if save:
                self.save()
    
    def get_config_data(self) -> Dict[str, Any]:
        """Get full configuration data (read-only copy)"""
        with self._lock:
            return json.loads(json.dumps(self._config_data))  # Deep copy
    
    def get_validation_messages(self) -> List[Any]:
        """Get validation messages from last configuration load"""
        return self._last_validation_messages.copy()
    
    def validate_config(self, config_data: Optional[Dict[str, Any]] = None, strict: bool = False) -> Tuple[bool, List[Any]]:
        """
        Validate configuration data
        
        Args:
            config_data: Configuration to validate (uses current config if None)
            strict: Whether to treat warnings as errors
            
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        if config_data is None:
            config_data = self.get_config_data()
        
        is_valid, messages = self._validator.validate(config_data, strict)
        
        # Add capability validation for task assignments
        capability_messages = self._validate_task_capabilities(config_data)
        messages.extend(capability_messages)
        
        # Update validity based on capability issues
        has_capability_errors = any(msg.get('level') == 'error' for msg in capability_messages)
        if has_capability_errors:
            is_valid = False
        
        return is_valid, messages
    
    def _validate_task_capabilities(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate task-model capability assignments.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            List of validation messages for capability issues
        """
        messages = []
        validator = get_capability_validator()
        
        task_assignments = config_data.get('task_specific_models', {})
        models = config_data.get('models', {})
        
        if not task_assignments or not models:
            return messages
        
        # Validate each task assignment
        validation_results = validator.validate_task_assignments(task_assignments, models)
        
        for task_name, issues in validation_results.items():
            model_key = task_assignments.get(task_name, 'unknown')
            model_name = models.get(model_key, {}).get('model', model_key)
            
            for issue in issues:
                # Determine severity based on issue type
                if any(keyword in issue.lower() for keyword in ['forbidden', 'lacks required', 'does not support']):
                    level = 'error'
                elif any(keyword in issue.lower() for keyword in ['below required', 'may not support']):
                    level = 'warning'
                else:
                    level = 'info'
                
                messages.append({
                    'level': level,
                    'message': f"Task '{task_name}' -> Model '{model_name}': {issue}",
                    'category': 'capability_validation',
                    'task': task_name,
                    'model': model_name,
                    'suggestion': self._get_capability_suggestion(task_name, models, validator)
                })
        
        return messages
    
    def _get_capability_suggestion(self, task_name: str, models: Dict[str, Dict[str, Any]], 
                                  validator) -> Optional[str]:
        """Get suggestion for better model assignment"""
        try:
            suggestion = validator.suggest_model_for_task(task_name, models)
            if suggestion:
                model_key, model_name, reason = suggestion
                return f"Consider using '{model_key}' ({model_name}): {reason}"
        except Exception:
            pass
        return None
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file with atomic operations and race condition protection"""
        save_path = path or self.config_path
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            try:
                # Update metadata
                self._config_data['metadata']['last_modified'] = datetime.now().isoformat()
                self._config_data['metadata']['modified_by'] = 'system'
                
                # Use atomic write to prevent race conditions and corruption
                with atomic_write(save_path, backup=True) as f:
                    yaml.dump(
                        self._config_data, 
                        f, 
                        default_flow_style=False, 
                        indent=2,
                        allow_unicode=True,
                        sort_keys=False
                    )
                
                logger.info(f"Configuration saved to {save_path}")
                
            except Exception as e:
                # Restore backup if save failed
                backup_path = f"{save_path}.backup"
                if os.path.exists(backup_path):
                    os.rename(backup_path, save_path)
                raise e
            else:
                # Remove backup on successful save
                backup_path = f"{save_path}.backup"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
    
    async def shutdown_async(self):
        """Async clean shutdown of the manager"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        # Shutdown synchronization manager
        if self._sync_manager:
            await self._sync_manager.stop()
        
        self._thread_pool.shutdown(wait=True)
        logger.info("ModelConfigManager async shutdown complete")
    
    def shutdown(self):
        """Clean shutdown of the manager"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        # Shutdown synchronization manager (sync version)
        if self._sync_manager:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new thread to run the async shutdown
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._sync_manager.stop())
                        future.result(timeout=5)
                else:
                    asyncio.run(self._sync_manager.stop())
            except Exception as e:
                logger.warning(f"Error shutting down sync manager: {e}")
        
        self._thread_pool.shutdown(wait=True)
        logger.info("ModelConfigManager shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def __str__(self) -> str:
        return f"ModelConfigManager(path={self.config_path}, models={len(self.get_models())})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Global instance for easy access
_model_config_instance = None
_instance_lock = threading.Lock()


def get_model_config(config_path: Optional[str] = None, auto_reload: bool = True) -> ModelConfigManager:
    """Get or create global ModelConfigManager instance"""
    global _model_config_instance
    
    with _instance_lock:
        if _model_config_instance is None or config_path is not None:
            _model_config_instance = ModelConfigManager(config_path, auto_reload)
        
        return _model_config_instance


def reload_model_config():
    """Reload global model configuration"""
    global _model_config_instance
    if _model_config_instance:
        _model_config_instance.reload()


# Integration helper for existing ResearchConfig
def integrate_with_research_config(research_config_instance, model_config_manager: Optional[ModelConfigManager] = None):
    """
    Integrate ModelConfigManager with existing ResearchConfig instance
    
    This allows the existing system to benefit from the new model management
    while maintaining backward compatibility.
    """
    if model_config_manager is None:
        model_config_manager = get_model_config()
    
    # Add method to get model for task
    def get_model_for_task(task: str, profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
        model_info = model_config_manager.get_model_for_task(task, profile)
        if model_info:
            return {
                "name": model_info.model,
                "params": {
                    "temperature": model_info.temperature,
                    "max_tokens": model_info.max_tokens
                },
                "api_base": model_info.api_base,
                "capabilities": model_info.capabilities
            }
        return None
    
    # Monkey patch the method onto the existing config instance
    research_config_instance.get_model_for_task = get_model_for_task
    
    # Add reload callback to sync changes
    def sync_callback():
        if hasattr(research_config_instance, 'reload'):
            research_config_instance.reload()
    
    model_config_manager.add_reload_callback(sync_callback)
    
    logger.info("ModelConfigManager integrated with ResearchConfig")
    return model_config_manager


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = ModelConfigManager()
    
    # Test basic operations
    print(f"Available models: {list(manager.get_models().keys())}")
    print(f"Active profile: {manager.get_active_profile()}")
    
    # Test model for task
    analysis_model = manager.get_model_for_task('react_analysis')
    if analysis_model:
        print(f"Analysis model: {analysis_model.model} ({analysis_model.provider})")
    
    # Test model connection (if LiteLLM is available)
    if LITELLM_AVAILABLE:
        test_result = manager.test_model_connection('gpt-4o-mini')
        print(f"Connection test: {test_result}")
    
    manager.shutdown()