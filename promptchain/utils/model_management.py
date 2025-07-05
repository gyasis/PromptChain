"""
Model Management Framework for PromptChain

This module provides a provider-agnostic model management system for efficient
VRAM usage with local LLM providers like Ollama, LocalAI, etc.

Key Features:
- Abstract ModelManager base class for extensibility
- Provider-specific implementations
- Integration hooks for PromptChain
- Configuration-driven model lifecycle management
- Backward compatibility with existing chains
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import httpx
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers"""
    OLLAMA = "ollama"
    LOCALAI = "localai"
    LLAMACPP = "llamacpp"
    # Add more providers as needed


@dataclass
class ModelInfo:
    """Model metadata and status information"""
    name: str
    provider: ModelProvider
    is_loaded: bool = False
    vram_usage: Optional[int] = None  # MB
    load_time: Optional[float] = None  # seconds
    last_used: Optional[float] = None  # timestamp
    parameters: Optional[Dict[str, Any]] = None


class ModelManagerError(Exception):
    """Base exception for model management errors"""
    pass


class ModelLoadError(ModelManagerError):
    """Raised when model loading fails"""
    pass


class ModelUnloadError(ModelManagerError):
    """Raised when model unloading fails"""
    pass


class AbstractModelManager(ABC):
    """
    Abstract base class for model managers.
    
    Defines the interface for provider-specific model management
    implementations. Each provider (Ollama, LocalAI, etc.) should
    implement this interface.
    """
    
    def __init__(self, provider: ModelProvider, base_url: str = None, **kwargs):
        self.provider = provider
        self.base_url = base_url
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.config = kwargs
        logger.info(f"Initialized {provider.value} model manager")
    
    @abstractmethod
    async def load_model_async(self, model_name: str, model_params: Optional[Dict] = None) -> ModelInfo:
        """
        Load a model asynchronously.
        
        Args:
            model_name: Name/identifier of the model
            model_params: Provider-specific parameters
            
        Returns:
            ModelInfo object with model metadata
            
        Raises:
            ModelLoadError: If model loading fails
        """
        pass
    
    @abstractmethod
    async def unload_model_async(self, model_name: str) -> bool:
        """
        Unload a model asynchronously.
        
        Args:
            model_name: Name/identifier of the model
            
        Returns:
            True if successfully unloaded, False otherwise
            
        Raises:
            ModelUnloadError: If model unloading fails
        """
        pass
    
    @abstractmethod
    async def is_model_loaded_async(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded.
        
        Args:
            model_name: Name/identifier of the model
            
        Returns:
            True if model is loaded, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_loaded_models_async(self) -> List[ModelInfo]:
        """
        Get list of currently loaded models.
        
        Returns:
            List of ModelInfo objects for loaded models
        """
        pass
    
    @abstractmethod
    async def health_check_async(self) -> bool:
        """
        Check if the model provider is healthy and reachable.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        pass
    
    # Synchronous wrappers for backward compatibility
    def load_model(self, model_name: str, model_params: Optional[Dict] = None) -> ModelInfo:
        """Synchronous wrapper for load_model_async"""
        return asyncio.run(self.load_model_async(model_name, model_params))
    
    def unload_model(self, model_name: str) -> bool:
        """Synchronous wrapper for unload_model_async"""
        return asyncio.run(self.unload_model_async(model_name))
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Synchronous wrapper for is_model_loaded_async"""
        return asyncio.run(self.is_model_loaded_async(model_name))
    
    def get_loaded_models(self) -> List[ModelInfo]:
        """Synchronous wrapper for get_loaded_models_async"""
        return asyncio.run(self.get_loaded_models_async())
    
    def health_check(self) -> bool:
        """Synchronous wrapper for health_check_async"""
        return asyncio.run(self.health_check_async())
    
    # Utility methods
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get cached model information"""
        return self.loaded_models.get(model_name)
    
    def get_provider_type(self) -> ModelProvider:
        """Get the provider type"""
        return self.provider
    
    async def cleanup_async(self):
        """Cleanup all loaded models (called on shutdown)"""
        logger.info(f"Cleaning up {self.provider.value} model manager...")
        for model_name in list(self.loaded_models.keys()):
            try:
                await self.unload_model_async(model_name)
            except Exception as e:
                logger.warning(f"Failed to unload {model_name} during cleanup: {e}")
    
    def cleanup(self):
        """Synchronous wrapper for cleanup_async"""
        asyncio.run(self.cleanup_async())


class ModelManagerFactory:
    """
    Factory for creating provider-specific model managers.
    
    Centralizes model manager creation and maintains registry
    of available providers.
    """
    
    _managers: Dict[ModelProvider, type] = {}
    
    @classmethod
    def register_manager(cls, provider: ModelProvider, manager_class: type):
        """Register a model manager class for a provider"""
        cls._managers[provider] = manager_class
        logger.info(f"Registered {provider.value} model manager: {manager_class.__name__}")
    
    @classmethod
    def create_manager(cls, provider: ModelProvider, **config) -> AbstractModelManager:
        """
        Create a model manager instance for the specified provider.
        
        Args:
            provider: The model provider enum
            **config: Provider-specific configuration
            
        Returns:
            AbstractModelManager instance
            
        Raises:
            ValueError: If provider is not supported
        """
        manager_class = cls._managers.get(provider)
        if not manager_class:
            available = list(cls._managers.keys())
            raise ValueError(f"Unsupported provider: {provider}. Available: {available}")
        
        return manager_class(provider=provider, **config)
    
    @classmethod
    def get_supported_providers(cls) -> List[ModelProvider]:
        """Get list of supported providers"""
        return list(cls._managers.keys())
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> AbstractModelManager:
        """
        Create a model manager from configuration dictionary.
        
        Args:
            config: Configuration with 'provider' key and provider-specific settings
            
        Returns:
            AbstractModelManager instance
        """
        provider_name = config.get('provider', 'ollama').lower()
        try:
            provider = ModelProvider(provider_name)
        except ValueError:
            raise ValueError(f"Invalid provider: {provider_name}")
        
        # Remove 'provider' from config before passing to manager
        manager_config = {k: v for k, v in config.items() if k != 'provider'}
        return cls.create_manager(provider, **manager_config)


# Configuration management
@dataclass
class ModelManagementConfig:
    """Configuration for model management system"""
    enabled: bool = False
    default_provider: ModelProvider = ModelProvider.OLLAMA
    auto_unload: bool = True  # Unload models after steps
    health_check_interval: int = 300  # seconds
    max_loaded_models: int = 2  # Maximum models to keep loaded
    provider_configs: Dict[ModelProvider, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.provider_configs is None:
            self.provider_configs = {
                ModelProvider.OLLAMA: {
                    'base_url': 'http://localhost:11434',
                    'timeout': 30
                }
            }


# Global configuration instance
_global_config = ModelManagementConfig()


def set_global_config(config: ModelManagementConfig):
    """Set the global model management configuration"""
    global _global_config
    _global_config = config
    logger.info("Updated global model management configuration")


def get_global_config() -> ModelManagementConfig:
    """Get the global model management configuration"""
    return _global_config


# Export key classes and functions
__all__ = [
    'AbstractModelManager',
    'ModelManagerFactory', 
    'ModelProvider',
    'ModelInfo',
    'ModelManagerError',
    'ModelLoadError',
    'ModelUnloadError',
    'ModelManagementConfig',
    'set_global_config',
    'get_global_config'
]