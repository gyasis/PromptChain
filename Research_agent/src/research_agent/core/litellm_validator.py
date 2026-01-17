"""
LiteLLM model format validation and correction system.

Ensures all model configurations use correct LiteLLM format identifiers
to prevent runtime failures and provider API call issues.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LiteLLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    AZURE = "azure"
    BEDROCK = "bedrock"


@dataclass
class ModelFormatRule:
    """Rule for validating model format for a specific provider"""
    provider: ProviderType
    name_pattern: str  # Regex pattern
    correct_format: str  # Template for correct format
    examples: List[str]
    common_mistakes: Dict[str, str]  # mistake -> correction


class LiteLLMValidator:
    """
    Validates and corrects LiteLLM model format identifiers.
    
    Ensures all models use the correct format expected by LiteLLM
    to prevent runtime failures and API call errors.
    """
    
    def __init__(self):
        self._format_rules = self._define_format_rules()
        self._known_models = self._define_known_models()
    
    def _define_format_rules(self) -> Dict[ProviderType, ModelFormatRule]:
        """Define format validation rules for each provider"""
        return {
            ProviderType.OPENAI: ModelFormatRule(
                provider=ProviderType.OPENAI,
                name_pattern=r"^(gpt-|text-|davinci-|ada-|babbage-|curie-)",
                correct_format="model_name",  # Direct model name
                examples=["gpt-4o", "gpt-4o-mini", "text-embedding-3-large"],
                common_mistakes={
                    "openai/gpt-4o": "gpt-4o",
                    "openai:gpt-4o": "gpt-4o",
                    "gpt4": "gpt-4",
                    "gpt-4-turbo": "gpt-4-turbo-preview"
                }
            ),
            
            ProviderType.ANTHROPIC: ModelFormatRule(
                provider=ProviderType.ANTHROPIC,
                name_pattern=r"^claude-",
                correct_format="claude-model-version",
                examples=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                common_mistakes={
                    "anthropic/claude-3-opus": "claude-3-opus-20240229",
                    "claude-opus": "claude-3-opus-20240229",
                    "claude-sonnet": "claude-3-sonnet-20240229",
                    "claude-haiku": "claude-3-haiku-20240307"
                }
            ),
            
            ProviderType.GOOGLE: ModelFormatRule(
                provider=ProviderType.GOOGLE,
                name_pattern=r"^gemini/",
                correct_format="gemini/model-name",
                examples=["gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash", "gemini/gemini-pro"],
                common_mistakes={
                    "google/gemini-pro": "gemini/gemini-pro",
                    "gemini-pro": "gemini/gemini-pro",
                    "gemini-1.5-pro": "gemini/gemini-1.5-pro"
                }
            ),
            
            ProviderType.OLLAMA: ModelFormatRule(
                provider=ProviderType.OLLAMA,
                name_pattern=r"^ollama/",
                correct_format="ollama/model-name",
                examples=["ollama/llama3.2", "ollama/mistral-nemo:latest", "ollama/phi3:medium"],
                common_mistakes={
                    "llama3.2": "ollama/llama3.2",
                    "local/llama3": "ollama/llama3",
                    "ollama:llama3": "ollama/llama3"
                }
            )
        }
    
    def _define_known_models(self) -> Dict[str, Dict[str, Any]]:
        """Define known working model configurations"""
        return {
            # OpenAI Models
            "gpt-4o": {
                "provider": "openai",
                "type": "chat",
                "max_tokens": 4000,
                "supports_functions": True
            },
            "gpt-4o-mini": {
                "provider": "openai", 
                "type": "chat",
                "max_tokens": 2000,
                "supports_functions": True
            },
            "gpt-4": {
                "provider": "openai",
                "type": "chat",
                "max_tokens": 8000,
                "supports_functions": True
            },
            "text-embedding-3-large": {
                "provider": "openai",
                "type": "embedding",
                "embedding_dim": 3072
            },
            "text-embedding-3-small": {
                "provider": "openai",
                "type": "embedding", 
                "embedding_dim": 1536
            },
            
            # Anthropic Models  
            "claude-3-opus-20240229": {
                "provider": "anthropic",
                "type": "chat",
                "max_tokens": 4000,
                "supports_functions": True
            },
            "claude-3-sonnet-20240229": {
                "provider": "anthropic",
                "type": "chat", 
                "max_tokens": 3000,
                "supports_functions": True
            },
            "claude-3-haiku-20240307": {
                "provider": "anthropic",
                "type": "chat",
                "max_tokens": 2000,
                "supports_functions": True
            },
            
            # Google Models
            "gemini/gemini-1.5-pro": {
                "provider": "google",
                "type": "chat",
                "max_tokens": 8000,
                "supports_functions": True
            },
            "gemini/gemini-1.5-flash": {
                "provider": "google",
                "type": "chat",
                "max_tokens": 4000,
                "supports_functions": True
            },
            
            # Ollama Models (format examples)
            "ollama/llama3.2": {
                "provider": "ollama",
                "type": "chat",
                "max_tokens": 4000,
                "requires_api_base": True
            },
            "ollama/mistral-nemo:latest": {
                "provider": "ollama",
                "type": "chat", 
                "max_tokens": 8000,
                "requires_api_base": True
            },
            "ollama/phi3:medium": {
                "provider": "ollama",
                "type": "chat",
                "max_tokens": 2000,
                "requires_api_base": True
            }
        }
    
    def detect_provider(self, model_name: str) -> Optional[ProviderType]:
        """Detect provider from model name"""
        model_lower = model_name.lower()
        
        if model_lower.startswith("gpt-") or model_lower.startswith("text-"):
            return ProviderType.OPENAI
        elif model_lower.startswith("claude-"):
            return ProviderType.ANTHROPIC
        elif model_lower.startswith("gemini/"):
            return ProviderType.GOOGLE
        elif model_lower.startswith("ollama/"):
            return ProviderType.OLLAMA
        
        return None
    
    def validate_model_format(self, model_name: str, model_config: Dict[str, Any]) -> Tuple[bool, List[str], Optional[str]]:
        """
        Validate model format for LiteLLM compatibility.
        
        Args:
            model_name: Model identifier to validate
            model_config: Model configuration dict
            
        Returns:
            Tuple of (is_valid, issues, suggested_correction)
        """
        issues = []
        suggested_correction = None
        
        provider = self.detect_provider(model_name)
        if not provider:
            issues.append(f"Cannot detect provider for model: {model_name}")
            return False, issues, None
        
        # Check against format rules
        rule = self._format_rules.get(provider)
        if rule:
            # Check common mistakes
            if model_name in rule.common_mistakes:
                suggested_correction = rule.common_mistakes[model_name]
                issues.append(f"Incorrect format: use '{suggested_correction}' instead of '{model_name}'")
            
            # Validate against pattern
            import re
            if not re.match(rule.name_pattern, model_name):
                issues.append(f"Model name doesn't match expected pattern for {provider.value}")
        
        # Check against known working models
        if model_name in self._known_models:
            known_config = self._known_models[model_name]
            
            # Validate required fields for Ollama models
            if provider == ProviderType.OLLAMA:
                if "api_base" not in model_config and not model_config.get("api_base"):
                    issues.append("Ollama models require 'api_base' field (typically http://localhost:11434)")
        
        # Check for embedding model misuse
        if "embedding" in model_name.lower() and model_config.get("max_tokens"):
            issues.append("Embedding models should not have max_tokens parameter")
        
        return len(issues) == 0, issues, suggested_correction
    
    def validate_all_models(self, models_config: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[bool, List[str], Optional[str]]]:
        """
        Validate all models in configuration.
        
        Args:
            models_config: Complete models configuration
            
        Returns:
            Dict mapping model keys to validation results
        """
        results = {}
        
        for model_key, config in models_config.items():
            model_name = config.get("model", model_key)
            results[model_key] = self.validate_model_format(model_name, config)
        
        return results
    
    def fix_model_formats(self, models_config: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Automatically fix common model format issues.
        
        Args:
            models_config: Models configuration to fix
            
        Returns:
            Tuple of (fixed_config, list_of_changes_made)
        """
        fixed_config = models_config.copy()
        changes_made = []
        
        for model_key, config in fixed_config.items():
            model_name = config.get("model", model_key)
            is_valid, issues, correction = self.validate_model_format(model_name, config)
            
            if not is_valid and correction:
                # Apply correction
                config["model"] = correction
                changes_made.append(f"Fixed '{model_key}': {model_name} → {correction}")
                
                # Update provider-specific configs
                provider = self.detect_provider(correction)
                if provider == ProviderType.OLLAMA and "api_base" not in config:
                    config["api_base"] = "http://localhost:11434"
                    changes_made.append(f"Added api_base for Ollama model: {model_key}")
        
        return fixed_config, changes_made
    
    def get_format_recommendations(self) -> Dict[str, List[str]]:
        """Get format recommendations for each provider"""
        recommendations = {}
        
        for provider, rule in self._format_rules.items():
            recommendations[provider.value] = {
                "correct_format": rule.correct_format,
                "examples": rule.examples,
                "common_mistakes": rule.common_mistakes
            }
        
        return recommendations


# Global validator instance
_litellm_validator = None


def get_litellm_validator() -> LiteLLMValidator:
    """Get or create global LiteLLM validator instance"""
    global _litellm_validator
    if _litellm_validator is None:
        _litellm_validator = LiteLLMValidator()
    return _litellm_validator


def validate_litellm_config(models_config: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate entire model configuration.
    
    Args:
        models_config: Models configuration dictionary
        
    Returns:
        Tuple of (all_valid, validation_report)
    """
    validator = get_litellm_validator()
    results = validator.validate_all_models(models_config)
    
    all_valid = all(result[0] for result in results.values())
    
    validation_report = {
        "all_valid": all_valid,
        "total_models": len(results),
        "valid_models": sum(1 for result in results.values() if result[0]),
        "invalid_models": sum(1 for result in results.values() if not result[0]),
        "issues_by_model": {
            model_key: {"issues": issues, "correction": correction}
            for model_key, (valid, issues, correction) in results.items()
            if not valid
        }
    }
    
    return all_valid, validation_report


if __name__ == "__main__":
    # Test with some example configurations
    import yaml
    
    test_config = {
        "gpt-4o": {"model": "gpt-4o", "temperature": 0.7},
        "claude-opus": {"model": "claude-opus", "temperature": 0.5},  # Wrong format
        "gemini-pro": {"model": "gemini-pro", "temperature": 0.6},   # Wrong format
        "llama3": {"model": "llama3", "temperature": 0.7},           # Wrong format
    }
    
    validator = LiteLLMValidator()
    
    print("Validation Results:")
    results = validator.validate_all_models(test_config)
    for model_key, (valid, issues, correction) in results.items():
        print(f"  {model_key}: {'✓' if valid else '✗'}")
        if not valid:
            for issue in issues:
                print(f"    - {issue}")
            if correction:
                print(f"    → Suggested: {correction}")
    
    print("\nAuto-fixing:")
    fixed_config, changes = validator.fix_model_formats(test_config)
    for change in changes:
        print(f"  {change}")