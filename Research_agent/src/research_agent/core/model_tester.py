"""
Secure model testing functionality for Research Agent.

Provides secure model connection testing without exposing API keys or sensitive information
in error messages or responses.
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TestResult(Enum):
    SUCCESS = "success"
    AUTHENTICATION_ERROR = "auth_error"
    NETWORK_ERROR = "network_error"
    MODEL_ERROR = "model_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ModelTestResponse:
    """Secure model test response without sensitive information"""
    success: bool
    result: TestResult
    message: str
    response_time_ms: Optional[int] = None
    model_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "success": self.success,
            "result": self.result.value,
            "message": self.message,
            "response_time_ms": self.response_time_ms,
            "model_name": self.model_name
        }


class SecureModelTester:
    """
    Secure model testing with proper API key protection and sanitized error messages.
    
    Ensures no API keys, authentication details, or sensitive information
    is exposed in error messages or responses.
    """
    
    # Patterns to identify and sanitize sensitive information
    API_KEY_PATTERNS = [
        r'sk-[a-zA-Z0-9]{20,}',  # OpenAI API keys
        r'sk-ant-[a-zA-Z0-9-]{95,}',  # Anthropic API keys
        r'AIza[a-zA-Z0-9_-]{35,}',  # Google API keys
    ]
    
    # Sensitive keywords that should be removed from error messages
    SENSITIVE_KEYWORDS = [
        'api_key', 'apikey', 'authorization', 'bearer', 'token',
        'secret', 'credential', 'auth', 'password', 'key'
    ]
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def _sanitize_error_message(self, error_message: str) -> str:
        """
        Sanitize error message to remove API keys and sensitive information.
        
        Args:
            error_message: Raw error message from LiteLLM or provider
            
        Returns:
            Sanitized error message safe for display
        """
        sanitized = error_message
        
        # Remove API keys using patterns
        for pattern in self.API_KEY_PATTERNS:
            sanitized = re.sub(pattern, '***API_KEY_REDACTED***', sanitized, flags=re.IGNORECASE)
        
        # Remove sensitive keywords and their values
        for keyword in self.SENSITIVE_KEYWORDS:
            # Remove key=value patterns
            sanitized = re.sub(f'{keyword}["\']?[=:]["\']?[^\\s,}}]*', f'{keyword}=***REDACTED***', sanitized)
            # Remove Authorization header patterns
            sanitized = re.sub(f'{keyword}["\']?:\\s*["\']?[^\\s,}}]*', f'{keyword}: ***REDACTED***', sanitized)
        
        # Remove any remaining long alphanumeric strings that might be tokens
        sanitized = re.sub(r'\b[A-Za-z0-9]{20,}\b', '***TOKEN_REDACTED***', sanitized)
        
        return sanitized[:500]  # Limit length to prevent information leakage
    
    def _classify_error(self, error: Exception) -> TestResult:
        """
        Classify error type without exposing sensitive information.
        
        Args:
            error: Exception from model testing
            
        Returns:
            TestResult classification
        """
        error_str = str(error).lower()
        
        # Authentication errors
        if any(keyword in error_str for keyword in [
            'authentication', 'unauthorized', '401', 'invalid api key', 
            'api key', 'forbidden', '403', 'invalid_api_key'
        ]):
            return TestResult.AUTHENTICATION_ERROR
        
        # Network errors
        if any(keyword in error_str for keyword in [
            'connection', 'timeout', 'network', 'dns', 'unreachable',
            'connection refused', 'connection timeout'
        ]):
            return TestResult.NETWORK_ERROR
        
        # Model-specific errors
        if any(keyword in error_str for keyword in [
            'model not found', 'invalid model', 'model unavailable',
            'model does not exist', 'unsupported model'
        ]):
            return TestResult.MODEL_ERROR
        
        # Timeout errors
        if any(keyword in error_str for keyword in [
            'timeout', 'timed out', 'deadline exceeded'
        ]):
            return TestResult.TIMEOUT_ERROR
        
        return TestResult.UNKNOWN_ERROR
    
    def _get_safe_error_message(self, error: Exception, result: TestResult) -> str:
        """
        Generate safe error message for display.
        
        Args:
            error: Original exception
            result: Classified error type
            
        Returns:
            Safe error message for user display
        """
        # Provide generic but helpful error messages
        if result == TestResult.AUTHENTICATION_ERROR:
            return "Authentication failed. Please check your API key configuration."
        elif result == TestResult.NETWORK_ERROR:
            return "Network connection failed. Please check your internet connection and API endpoint."
        elif result == TestResult.MODEL_ERROR:
            return "Model not available. Please check the model name and provider settings."
        elif result == TestResult.TIMEOUT_ERROR:
            return f"Request timed out after {self.timeout} seconds. Please try again."
        else:
            # For unknown errors, provide sanitized message
            sanitized = self._sanitize_error_message(str(error))
            return f"Test failed: {sanitized}"
    
    async def test_model(self, model_config: Dict[str, Any]) -> ModelTestResponse:
        """
        Test model connection securely without exposing sensitive information.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            ModelTestResponse with sanitized information
        """
        import time
        import asyncio
        
        model_name = model_config.get('model', 'unknown')
        start_time = time.time()
        
        try:
            # Import LiteLLM here to avoid import issues
            import litellm
            
            # Prepare test request with minimal parameters
            test_params = {
                'model': model_name,
                'messages': [{'role': 'user', 'content': 'Test connection'}],
                'max_tokens': 5,
                'timeout': self.timeout
            }
            
            # Add API base if specified (for local models)
            if 'api_base' in model_config:
                test_params['api_base'] = model_config['api_base']
            
            # Test connection with timeout
            try:
                response = await asyncio.wait_for(
                    litellm.acompletion(**test_params),
                    timeout=self.timeout
                )
                
                response_time = int((time.time() - start_time) * 1000)
                
                logger.info(f"Model test successful: {model_name}")
                return ModelTestResponse(
                    success=True,
                    result=TestResult.SUCCESS,
                    message="Model connection successful",
                    response_time_ms=response_time,
                    model_name=model_name
                )
                
            except asyncio.TimeoutError:
                logger.warning(f"Model test timeout: {model_name}")
                return ModelTestResponse(
                    success=False,
                    result=TestResult.TIMEOUT_ERROR,
                    message=f"Connection timeout after {self.timeout} seconds",
                    model_name=model_name
                )
                
        except Exception as e:
            logger.error(f"Model test failed for {model_name}: {type(e).__name__}")
            
            # Classify error and generate safe message
            error_type = self._classify_error(e)
            safe_message = self._get_safe_error_message(e, error_type)
            
            response_time = int((time.time() - start_time) * 1000)
            
            return ModelTestResponse(
                success=False,
                result=error_type,
                message=safe_message,
                response_time_ms=response_time,
                model_name=model_name
            )
    
    def test_model_sync(self, model_config: Dict[str, Any]) -> ModelTestResponse:
        """
        Synchronous wrapper for model testing.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            ModelTestResponse with sanitized information
        """
        import asyncio
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.test_model(model_config))
                    return future.result(timeout=self.timeout + 5)
            else:
                return asyncio.run(self.test_model(model_config))
        except Exception as e:
            logger.error(f"Sync model test failed: {type(e).__name__}")
            return ModelTestResponse(
                success=False,
                result=TestResult.UNKNOWN_ERROR,
                message="Test execution failed",
                model_name=model_config.get('model', 'unknown')
            )


# Global tester instance
_model_tester = None


def get_model_tester(timeout: int = 30) -> SecureModelTester:
    """Get or create global model tester instance"""
    global _model_tester
    if _model_tester is None:
        _model_tester = SecureModelTester(timeout=timeout)
    return _model_tester


def test_model_connection(model_config: Dict[str, Any], timeout: int = 30) -> ModelTestResponse:
    """
    Convenience function to test model connection securely.
    
    Args:
        model_config: Model configuration dictionary
        timeout: Request timeout in seconds
        
    Returns:
        ModelTestResponse with sanitized information
    """
    tester = get_model_tester(timeout)
    return tester.test_model_sync(model_config)


def validate_model_config(model_config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate model configuration without testing connection.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(model_config, dict):
        return False, "Model configuration must be a dictionary"
    
    if 'model' not in model_config:
        return False, "Model configuration must include 'model' field"
    
    model_name = model_config.get('model', '')
    if not model_name or not isinstance(model_name, str):
        return False, "Model name must be a non-empty string"
    
    # Check for suspicious model names that might be injection attempts
    if any(char in model_name for char in ['<', '>', '&', '"', "'"]):
        return False, "Model name contains invalid characters"
    
    # Validate temperature if present
    if 'temperature' in model_config:
        temp = model_config['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            return False, "Temperature must be a number between 0 and 2"
    
    # Validate max_tokens if present
    if 'max_tokens' in model_config:
        tokens = model_config['max_tokens']
        if not isinstance(tokens, int) or tokens <= 0:
            return False, "Max tokens must be a positive integer"
    
    return True, "Configuration is valid"