"""
Ollama Model Manager Implementation

Provides VRAM-efficient model management for Ollama provider,
including automatic model loading/unloading between chain steps.
"""

import time
import logging
from typing import Dict, Any, Optional, List
import httpx
import asyncio
from .model_management import (
    AbstractModelManager, 
    ModelProvider, 
    ModelInfo, 
    ModelLoadError, 
    ModelUnloadError,
    ModelManagerFactory
)

logger = logging.getLogger(__name__)


class OllamaModelManager(AbstractModelManager):
    """
    Ollama-specific model manager with VRAM optimization.
    
    Features:
    - Automatic model loading/unloading
    - VRAM usage tracking
    - Health monitoring
    - Robust error handling
    - Connection pooling
    """
    
    def __init__(self, provider: ModelProvider = ModelProvider.OLLAMA, 
                 base_url: str = "http://localhost:11434", 
                 timeout: int = 30,
                 max_retries: int = 3,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(provider, base_url, **kwargs)
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.client_config = {
            'timeout': httpx.Timeout(timeout),
            'follow_redirects': True,
            'limits': httpx.Limits(max_connections=10, max_keepalive_connections=5)
        }
        logger.info(f"Initialized Ollama manager: {base_url}")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make HTTP request to Ollama API with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(**self.client_config) as client:
                    response = await client.request(method, url, **kwargs)
                    return response
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == self.max_retries - 1:
                    raise ModelLoadError(f"Failed to connect to Ollama after {self.max_retries} attempts: {e}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise ModelLoadError("Exhausted all retry attempts")
    
    async def load_model_async(self, model_name: str, model_params: Optional[Dict] = None) -> ModelInfo:
        """
        Load an Ollama model with VRAM tracking.
        
        Args:
            model_name: Ollama model name (e.g., "llama2:7b", "mistral:latest", "ollama/mistral:latest")
            model_params: Additional model parameters
            
        Returns:
            ModelInfo with loading details
        """
        start_time = time.time()
        
        # Strip "ollama/" prefix if present for API calls
        actual_model_name = model_name.replace("ollama/", "") if model_name.startswith("ollama/") else model_name
        
        # Check if already loaded
        if await self.is_model_loaded_async(actual_model_name):
            model_info = self.loaded_models[actual_model_name]
            model_info.last_used = time.time()
            logger.info(f"Model {actual_model_name} already loaded")
            return model_info
        
        try:
            logger.info(f"Loading Ollama model: {actual_model_name}")
            
            # Get model info to determine max context length
            model_info_response = await self._make_request("POST", "/api/show", json={"name": actual_model_name})
            max_context_length = None
            
            if model_info_response.status_code == 200:
                model_data = model_info_response.json()
                # Try to extract context length from model info
                if 'model_info' in model_data:
                    context_info = model_data['model_info']
                    # Look for context length in various fields
                    if 'context_length' in context_info:
                        max_context_length = context_info['context_length']
                    elif 'max_sequence_length' in context_info:
                        max_context_length = context_info['max_sequence_length']
                # Also check parameters
                if 'parameters' in model_data:
                    params = model_data['parameters']
                    if 'num_ctx' in params:
                        max_context_length = int(params['num_ctx'])
                        
                if self.verbose and max_context_length:
                    logger.info(f"Model {actual_model_name} max context length: {max_context_length}")
            
            # Generate a small request to trigger model loading with optimal context settings
            load_options = {
                "num_predict": 1,  # Generate only 1 token
            }
            
            # Set maximum context length if detected
            if max_context_length:
                load_options["num_ctx"] = max_context_length
                if self.verbose:
                    logger.info(f"Setting model context length to maximum: {max_context_length}")
            
            response = await self._make_request(
                "POST", 
                "/api/generate",
                json={
                    "model": actual_model_name,  # Use actual model name without prefix
                    "prompt": "Hello",  # Minimal prompt to trigger loading
                    "stream": False,
                    "options": load_options
                }
            )
            
            if response.status_code != 200:
                raise ModelLoadError(f"Failed to load model {actual_model_name}: {response.text}")
            
            load_time = time.time() - start_time
            
            # Get model info for VRAM tracking
            vram_usage = await self._get_model_vram_usage(actual_model_name)
            
            model_info = ModelInfo(
                name=actual_model_name,  # Store actual name
                provider=self.provider,
                is_loaded=True,
                vram_usage=vram_usage,
                load_time=load_time,
                last_used=time.time(),
                parameters=model_params or {}
            )
            
            self.loaded_models[actual_model_name] = model_info  # Use actual name as key
            logger.info(f"Successfully loaded {actual_model_name} in {load_time:.2f}s (VRAM: {vram_usage}MB)")
            return model_info
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {actual_model_name}: {e}")
    
    async def unload_model_async(self, model_name: str) -> bool:
        """
        Unload an Ollama model to free VRAM.
        
        Uses Ollama's keep_alive=0 parameter to force unloading.
        
        Args:
            model_name: Name of the model to unload (with or without "ollama/" prefix)
            
        Returns:
            True if successfully unloaded
        """
        # Strip "ollama/" prefix if present for API calls
        actual_model_name = model_name.replace("ollama/", "") if model_name.startswith("ollama/") else model_name
        
        if not await self.is_model_loaded_async(actual_model_name):
            logger.warning(f"Model {actual_model_name} not loaded, nothing to unload")
            return True
        
        try:
            logger.info(f"Unloading Ollama model: {actual_model_name}")
            
            # Force unload using keep_alive=0
            response = await self._make_request(
                "POST",
                "/api/generate", 
                json={
                    "model": actual_model_name,  # Use actual name without prefix
                    "prompt": "",  # Empty prompt
                    "keep_alive": 0,  # Force unload immediately
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Unload request returned {response.status_code}: {response.text}")
            
            # Remove from our tracking
            if actual_model_name in self.loaded_models:
                del self.loaded_models[actual_model_name]
            
            # Verify unload by checking loaded models
            loaded_models = await self._get_ollama_loaded_models()
            is_still_loaded = any(m['name'] == actual_model_name for m in loaded_models)
            
            if is_still_loaded:
                logger.warning(f"Model {actual_model_name} may still be loaded in Ollama")
                return False
            
            logger.info(f"Successfully unloaded {actual_model_name}")
            return True
            
        except Exception as e:
            raise ModelUnloadError(f"Failed to unload model {actual_model_name}: {e}")
    
    async def is_model_loaded_async(self, model_name: str) -> bool:
        """Check if model is loaded in Ollama"""
        # Strip "ollama/" prefix if present for API calls
        actual_model_name = model_name.replace("ollama/", "") if model_name.startswith("ollama/") else model_name
        
        try:
            loaded_models = await self._get_ollama_loaded_models()
            return any(m['name'] == actual_model_name for m in loaded_models)
        except Exception as e:
            logger.warning(f"Failed to check if model {actual_model_name} is loaded: {e}")
            return actual_model_name in self.loaded_models  # Fallback to local tracking
    
    async def get_loaded_models_async(self) -> List[ModelInfo]:
        """Get list of models currently loaded in Ollama"""
        try:
            ollama_models = await self._get_ollama_loaded_models()
            
            # Update our local tracking with Ollama's state
            current_loaded = set()
            model_infos = []
            
            for model_data in ollama_models:
                model_name = model_data['name']
                current_loaded.add(model_name)
                
                # Get cached info or create new
                if model_name in self.loaded_models:
                    model_info = self.loaded_models[model_name]
                else:
                    model_info = ModelInfo(
                        name=model_name,
                        provider=self.provider,
                        is_loaded=True,
                        vram_usage=model_data.get('size', 0) // (1024 * 1024),  # Convert to MB
                        last_used=time.time()
                    )
                    self.loaded_models[model_name] = model_info
                
                model_infos.append(model_info)
            
            # Remove models that are no longer loaded
            to_remove = set(self.loaded_models.keys()) - current_loaded
            for model_name in to_remove:
                del self.loaded_models[model_name]
            
            return model_infos
            
        except Exception as e:
            logger.warning(f"Failed to get loaded models from Ollama: {e}")
            return list(self.loaded_models.values())
    
    async def health_check_async(self) -> bool:
        """Check if Ollama server is healthy"""
        try:
            response = await self._make_request("GET", "/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def _get_ollama_loaded_models(self) -> List[Dict]:
        """Get currently loaded models from Ollama /api/ps endpoint"""
        try:
            response = await self._make_request("GET", "/api/ps")
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            return []
        except Exception as e:
            logger.warning(f"Failed to get Ollama process status: {e}")
            return []
    
    async def _get_model_vram_usage(self, model_name: str) -> Optional[int]:
        """Estimate VRAM usage for a model"""
        try:
            # Get model details from Ollama
            response = await self._make_request("POST", "/api/show", json={"name": model_name})
            if response.status_code == 200:
                model_data = response.json()
                # Parse model size from modelfile or details
                # This is approximate - Ollama doesn't always provide exact VRAM usage
                details = model_data.get('details', {})
                parameter_size = details.get('parameter_size', 0)
                if parameter_size:
                    # Rough estimate: parameter size + some overhead
                    return int(parameter_size * 1.2 // (1024 * 1024))  # Convert to MB with overhead
            return None
        except Exception as e:
            logger.warning(f"Failed to get VRAM usage for {model_name}: {e}")
            return None
    
    async def get_available_models_async(self) -> List[str]:
        """Get list of available models that can be loaded"""
        try:
            response = await self._make_request("GET", "/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
            return []
    
    async def unload_all_models_async(self) -> bool:
        """Unload all currently loaded models"""
        try:
            loaded_models = await self.get_loaded_models_async()
            success = True
            
            for model_info in loaded_models:
                try:
                    await self.unload_model_async(model_info.name)
                except Exception as e:
                    logger.error(f"Failed to unload {model_info.name}: {e}")
                    success = False
            
            return success
        except Exception as e:
            logger.error(f"Failed to unload all models: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Synchronous wrapper for get_available_models_async"""
        return asyncio.run(self.get_available_models_async())
    
    def unload_all_models(self) -> bool:
        """Synchronous wrapper for unload_all_models_async"""
        return asyncio.run(self.unload_all_models_async())


# Register the Ollama manager with the factory
ModelManagerFactory.register_manager(ModelProvider.OLLAMA, OllamaModelManager)

logger.info("Registered OllamaModelManager with ModelManagerFactory")