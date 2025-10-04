"""
Model Configuration Validation System

Provides comprehensive validation for models.yaml configuration files
with detailed error reporting and schema enforcement.
"""

import os
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation message severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """Validation message with context"""
    severity: ValidationSeverity
    message: str
    path: str
    details: Optional[str] = None
    suggestion: Optional[str] = None


class ModelConfigValidator:
    """
    Comprehensive validator for model configuration files
    
    Validates structure, data types, constraints, and relationships
    in models.yaml configuration files.
    """
    
    def __init__(self):
        self.messages: List[ValidationMessage] = []
        self.supported_providers = {
            'openai', 'anthropic', 'google', 'ollama', 'mistral', 
            'cohere', 'huggingface', 'custom'
        }
        self.supported_capabilities = {
            'search', 'analysis', 'synthesis', 'multi_query', 'chat',
            'query_generation', 'routing', 'embedding', 'reasoning',
            'code_generation', 'translation', 'summarization'
        }
        self.required_tasks = {
            'query_generation', 'search_strategy', 'literature_search',
            'multi_query_coordination', 'react_analysis', 'synthesis',
            'chat_interface'
        }
        
    def validate(self, config: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[ValidationMessage]]:
        """
        Validate complete model configuration
        
        Args:
            config: Configuration dictionary to validate
            strict: Whether to treat warnings as errors
            
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        self.messages = []
        
        # Validate top-level structure
        self._validate_structure(config)
        
        # Validate models section
        if 'models' in config:
            self._validate_models(config['models'])
        
        # Validate profiles section
        if 'profiles' in config:
            self._validate_profiles(config['profiles'])
        
        # Validate task assignments
        if 'task_specific_models' in config:
            self._validate_task_assignments(config['task_specific_models'])
        
        # Validate LiteLLM settings
        if 'litellm_settings' in config:
            self._validate_litellm_settings(config['litellm_settings'])
        
        # Validate cross-references
        self._validate_cross_references(config)
        
        # Check completeness
        self._validate_completeness(config)
        
        # Determine if validation passed
        has_errors = any(msg.severity == ValidationSeverity.ERROR for msg in self.messages)
        has_warnings = any(msg.severity == ValidationSeverity.WARNING for msg in self.messages)
        
        is_valid = not has_errors and (not strict or not has_warnings)
        
        return is_valid, self.messages.copy()
    
    def _add_message(self, severity: ValidationSeverity, message: str, path: str, 
                     details: Optional[str] = None, suggestion: Optional[str] = None):
        """Add validation message"""
        self.messages.append(ValidationMessage(
            severity=severity,
            message=message,
            path=path,
            details=details,
            suggestion=suggestion
        ))
    
    def _validate_structure(self, config: Dict[str, Any]):
        """Validate top-level configuration structure"""
        required_sections = ['models', 'task_specific_models', 'profiles']
        
        for section in required_sections:
            if section not in config:
                self._add_message(
                    ValidationSeverity.ERROR,
                    f"Missing required section: {section}",
                    f"root.{section}",
                    suggestion=f"Add '{section}:' section to configuration"
                )
        
        # Validate optional sections
        if 'default_model' not in config:
            self._add_message(
                ValidationSeverity.WARNING,
                "No default model specified",
                "root.default_model",
                suggestion="Add 'default_model: model_name' to specify fallback model"
            )
        
        if 'active_profile' not in config:
            self._add_message(
                ValidationSeverity.WARNING,
                "No active profile specified",
                "root.active_profile",
                suggestion="Add 'active_profile: profile_name' to specify active profile"
            )
        
        # Validate metadata
        if 'metadata' in config:
            self._validate_metadata(config['metadata'])
    
    def _validate_metadata(self, metadata: Dict[str, Any]):
        """Validate metadata section"""
        if not isinstance(metadata, dict):
            self._add_message(
                ValidationSeverity.ERROR,
                "Metadata must be a dictionary",
                "metadata"
            )
            return
        
        if 'version' not in metadata:
            self._add_message(
                ValidationSeverity.WARNING,
                "No version specified in metadata",
                "metadata.version",
                suggestion="Add version field for tracking configuration changes"
            )
    
    def _validate_models(self, models: Dict[str, Any]):
        """Validate models section"""
        if not isinstance(models, dict):
            self._add_message(
                ValidationSeverity.ERROR,
                "Models section must be a dictionary",
                "models"
            )
            return
        
        if not models:
            self._add_message(
                ValidationSeverity.ERROR,
                "No models defined",
                "models",
                suggestion="Add at least one model configuration"
            )
            return
        
        for model_name, model_config in models.items():
            self._validate_model(model_name, model_config, f"models.{model_name}")
    
    def _validate_model(self, model_name: str, config: Dict[str, Any], path: str):
        """Validate individual model configuration"""
        if not isinstance(config, dict):
            self._add_message(
                ValidationSeverity.ERROR,
                f"Model '{model_name}' configuration must be a dictionary",
                path
            )
            return
        
        # Validate required fields
        required_fields = ['model']
        for field in required_fields:
            if field not in config:
                self._add_message(
                    ValidationSeverity.ERROR,
                    f"Missing required field: {field}",
                    f"{path}.{field}",
                    suggestion=f"Add '{field}' field with appropriate value"
                )
        
        # Validate model name format
        if not re.match(r'^[a-z0-9_-]+$', model_name):
            self._add_message(
                ValidationSeverity.ERROR,
                f"Invalid model name format: {model_name}",
                path,
                details="Model names can only contain lowercase letters, numbers, hyphens, and underscores",
                suggestion="Use format like 'gpt-4o' or 'claude_3_sonnet'"
            )
        
        # Validate model identifier
        if 'model' in config:
            self._validate_model_identifier(config['model'], f"{path}.model")
        
        # Validate numeric parameters
        self._validate_numeric_field(config, 'temperature', path, 0.0, 2.0, default=0.7)
        self._validate_numeric_field(config, 'max_tokens', path, 1, 32000, default=4000, field_type=int)
        self._validate_numeric_field(config, 'cost_per_1k_tokens', path, 0.0, float('inf'), default=0.0)
        
        # Validate capabilities
        if 'capabilities' in config:
            self._validate_capabilities(config['capabilities'], f"{path}.capabilities")
        
        # Validate description
        if 'description' in config:
            if not isinstance(config['description'], str):
                self._add_message(
                    ValidationSeverity.WARNING,
                    "Description should be a string",
                    f"{path}.description"
                )
            elif len(config['description']) > 500:
                self._add_message(
                    ValidationSeverity.WARNING,
                    "Description is very long (>500 characters)",
                    f"{path}.description",
                    suggestion="Consider shortening the description"
                )
        
        # Validate provider-specific fields
        self._validate_provider_specific(config, path)
    
    def _validate_model_identifier(self, model_id: str, path: str):
        """Validate LiteLLM model identifier format"""
        if not isinstance(model_id, str):
            self._add_message(
                ValidationSeverity.ERROR,
                "Model identifier must be a string",
                path
            )
            return
        
        if not model_id.strip():
            self._add_message(
                ValidationSeverity.ERROR,
                "Model identifier cannot be empty",
                path
            )
            return
        
        # Validate known provider patterns
        provider_patterns = {
            'openai': [r'^gpt-', r'^text-', r'^davinci', r'^curie', r'^babbage', r'^ada'],
            'anthropic': [r'^claude-'],
            'google': [r'^gemini/', r'^palm/'],
            'ollama': [r'^ollama/'],
            'mistral': [r'^mistral/', r'^open-mistral'],
            'cohere': [r'^cohere/', r'^command'],
        }
        
        # Check if model follows known patterns
        matches_pattern = False
        for provider, patterns in provider_patterns.items():
            for pattern in patterns:
                if re.match(pattern, model_id, re.IGNORECASE):
                    matches_pattern = True
                    break
            if matches_pattern:
                break
        
        if not matches_pattern:
            self._add_message(
                ValidationSeverity.WARNING,
                f"Model identifier '{model_id}' doesn't match known provider patterns",
                path,
                details="This may indicate a typo or unsupported model",
                suggestion="Verify the model identifier with LiteLLM documentation"
            )
    
    def _validate_numeric_field(self, config: Dict[str, Any], field_name: str, path: str, 
                                min_val: float, max_val: float, default: float = None, 
                                field_type: type = float):
        """Validate numeric configuration field"""
        if field_name not in config:
            if default is not None:
                self._add_message(
                    ValidationSeverity.INFO,
                    f"Using default value {default} for {field_name}",
                    f"{path}.{field_name}"
                )
            return
        
        value = config[field_name]
        
        # Type validation
        if not isinstance(value, (int, float)):
            self._add_message(
                ValidationSeverity.ERROR,
                f"{field_name} must be a number",
                f"{path}.{field_name}"
            )
            return
        
        # Convert to target type
        if field_type == int and not isinstance(value, int):
            if value.is_integer():
                value = int(value)
            else:
                self._add_message(
                    ValidationSeverity.ERROR,
                    f"{field_name} must be an integer",
                    f"{path}.{field_name}"
                )
                return
        
        # Range validation
        if value < min_val or value > max_val:
            self._add_message(
                ValidationSeverity.ERROR,
                f"{field_name} must be between {min_val} and {max_val}",
                f"{path}.{field_name}",
                details=f"Current value: {value}"
            )
    
    def _validate_capabilities(self, capabilities: List[str], path: str):
        """Validate model capabilities list"""
        if not isinstance(capabilities, list):
            self._add_message(
                ValidationSeverity.ERROR,
                "Capabilities must be a list",
                path
            )
            return
        
        if not capabilities:
            self._add_message(
                ValidationSeverity.WARNING,
                "No capabilities specified",
                path,
                suggestion="Add relevant capabilities like ['analysis', 'search', 'synthesis']"
            )
            return
        
        for capability in capabilities:
            if not isinstance(capability, str):
                self._add_message(
                    ValidationSeverity.ERROR,
                    f"Capability must be a string: {capability}",
                    path
                )
            elif capability not in self.supported_capabilities:
                self._add_message(
                    ValidationSeverity.WARNING,
                    f"Unknown capability: {capability}",
                    path,
                    details=f"Supported capabilities: {', '.join(sorted(self.supported_capabilities))}",
                    suggestion="Use standard capability names for better integration"
                )
    
    def _validate_provider_specific(self, config: Dict[str, Any], path: str):
        """Validate provider-specific configuration"""
        model_id = config.get('model', '')
        
        # Ollama models should have api_base
        if model_id.startswith('ollama/') or 'api_base' in config:
            if 'api_base' not in config:
                self._add_message(
                    ValidationSeverity.WARNING,
                    "Ollama models should specify api_base",
                    f"{path}.api_base",
                    suggestion="Add 'api_base: http://localhost:11434' for local Ollama"
                )
            elif config['api_base'] and not self._is_valid_url(config['api_base']):
                self._add_message(
                    ValidationSeverity.ERROR,
                    f"Invalid API base URL: {config['api_base']}",
                    f"{path}.api_base"
                )
        
        # Embedding models should have embedding_dim
        if 'embedding' in config.get('capabilities', []) or 'embed' in model_id.lower():
            if 'embedding_dim' not in config:
                self._add_message(
                    ValidationSeverity.WARNING,
                    "Embedding models should specify embedding_dim",
                    f"{path}.embedding_dim",
                    suggestion="Add dimension size like 'embedding_dim: 1536'"
                )
    
    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None
    
    def _validate_profiles(self, profiles: Dict[str, Any]):
        """Validate model profiles section"""
        if not isinstance(profiles, dict):
            self._add_message(
                ValidationSeverity.ERROR,
                "Profiles section must be a dictionary",
                "profiles"
            )
            return
        
        if not profiles:
            self._add_message(
                ValidationSeverity.WARNING,
                "No profiles defined",
                "profiles",
                suggestion="Add at least a 'default' profile"
            )
            return
        
        # Check for default profile
        if 'default' not in profiles:
            self._add_message(
                ValidationSeverity.WARNING,
                "No 'default' profile found",
                "profiles.default",
                suggestion="Add a default profile for fallback scenarios"
            )
        
        for profile_name, profile_config in profiles.items():
            self._validate_profile(profile_name, profile_config, f"profiles.{profile_name}")
    
    def _validate_profile(self, profile_name: str, config: Dict[str, Any], path: str):
        """Validate individual profile configuration"""
        if not isinstance(config, dict):
            self._add_message(
                ValidationSeverity.ERROR,
                f"Profile '{profile_name}' configuration must be a dictionary",
                path
            )
            return
        
        # Validate profile name
        if not re.match(r'^[a-z0-9_-]+$', profile_name):
            self._add_message(
                ValidationSeverity.WARNING,
                f"Profile name '{profile_name}' should use lowercase letters, numbers, and underscores",
                path
            )
        
        # Check for default_model
        if 'default_model' not in config:
            self._add_message(
                ValidationSeverity.WARNING,
                f"Profile '{profile_name}' missing default_model",
                f"{path}.default_model",
                suggestion="Add 'default_model' field"
            )
    
    def _validate_task_assignments(self, task_assignments: Dict[str, str]):
        """Validate task-specific model assignments"""
        if not isinstance(task_assignments, dict):
            self._add_message(
                ValidationSeverity.ERROR,
                "Task assignments must be a dictionary",
                "task_specific_models"
            )
            return
        
        # Check for required tasks
        missing_tasks = self.required_tasks - set(task_assignments.keys())
        if missing_tasks:
            self._add_message(
                ValidationSeverity.WARNING,
                f"Missing task assignments: {', '.join(sorted(missing_tasks))}",
                "task_specific_models",
                suggestion="Add assignments for all required research tasks"
            )
        
        # Validate task names
        for task in task_assignments.keys():
            if not re.match(r'^[a-z_]+$', task):
                self._add_message(
                    ValidationSeverity.WARNING,
                    f"Task name '{task}' should use lowercase letters and underscores",
                    f"task_specific_models.{task}"
                )
    
    def _validate_litellm_settings(self, settings: Dict[str, Any]):
        """Validate LiteLLM global settings"""
        if not isinstance(settings, dict):
            self._add_message(
                ValidationSeverity.ERROR,
                "LiteLLM settings must be a dictionary",
                "litellm_settings"
            )
            return
        
        # Validate specific settings
        numeric_settings = {
            'max_retries': (0, 10),
            'retry_delay': (0, 60),
            'request_timeout': (1, 600),
            'stream_timeout': (1, 1200),
            'rpm_limit': (1, 10000),
            'cache_ttl': (60, 86400)
        }
        
        for setting, (min_val, max_val) in numeric_settings.items():
            if setting in settings:
                value = settings[setting]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    self._add_message(
                        ValidationSeverity.ERROR,
                        f"Invalid {setting}: must be between {min_val} and {max_val}",
                        f"litellm_settings.{setting}",
                        details=f"Current value: {value}"
                    )
        
        # Validate boolean settings
        boolean_settings = ['cache', 'log_requests', 'log_responses', 'track_cost']
        for setting in boolean_settings:
            if setting in settings and not isinstance(settings[setting], bool):
                self._add_message(
                    ValidationSeverity.ERROR,
                    f"{setting} must be a boolean (true/false)",
                    f"litellm_settings.{setting}"
                )
    
    def _validate_cross_references(self, config: Dict[str, Any]):
        """Validate cross-references between sections"""
        models = config.get('models', {})
        task_assignments = config.get('task_specific_models', {})
        profiles = config.get('profiles', {})
        default_model = config.get('default_model')
        active_profile = config.get('active_profile')
        
        # Check default model exists
        if default_model and default_model not in models:
            self._add_message(
                ValidationSeverity.ERROR,
                f"Default model '{default_model}' not found in models section",
                "default_model",
                suggestion=f"Add '{default_model}' to models section or change default_model"
            )
        
        # Check active profile exists
        if active_profile and active_profile not in profiles:
            self._add_message(
                ValidationSeverity.ERROR,
                f"Active profile '{active_profile}' not found in profiles section",
                "active_profile",
                suggestion=f"Add '{active_profile}' to profiles section or change active_profile"
            )
        
        # Check task assignment models exist
        for task, model_name in task_assignments.items():
            if model_name not in models:
                self._add_message(
                    ValidationSeverity.ERROR,
                    f"Task '{task}' assigned to non-existent model '{model_name}'",
                    f"task_specific_models.{task}",
                    suggestion=f"Add '{model_name}' to models section or reassign task"
                )
        
        # Check profile assignments
        for profile_name, profile_config in profiles.items():
            if isinstance(profile_config, dict):
                for task, model_name in profile_config.items():
                    if task != 'description' and isinstance(model_name, str) and model_name not in models:
                        self._add_message(
                            ValidationSeverity.ERROR,
                            f"Profile '{profile_name}' assigns task '{task}' to non-existent model '{model_name}'",
                            f"profiles.{profile_name}.{task}",
                            suggestion=f"Add '{model_name}' to models section or reassign task"
                        )
    
    def _validate_completeness(self, config: Dict[str, Any]):
        """Validate configuration completeness"""
        models = config.get('models', {})
        
        # Check for diverse model capabilities
        all_capabilities = set()
        for model_config in models.values():
            if isinstance(model_config, dict) and 'capabilities' in model_config:
                all_capabilities.update(model_config['capabilities'])
        
        essential_capabilities = {'search', 'analysis', 'synthesis', 'chat'}
        missing_capabilities = essential_capabilities - all_capabilities
        
        if missing_capabilities:
            self._add_message(
                ValidationSeverity.WARNING,
                f"No models with essential capabilities: {', '.join(sorted(missing_capabilities))}",
                "models",
                suggestion="Add models with missing capabilities for full functionality"
            )
        
        # Check for cost diversity
        has_free_models = any(
            isinstance(model_config, dict) and model_config.get('cost_per_1k_tokens', 0) == 0
            for model_config in models.values()
        )
        
        if not has_free_models:
            self._add_message(
                ValidationSeverity.INFO,
                "No free models configured",
                "models",
                suggestion="Consider adding local/Ollama models to reduce costs"
            )


def validate_model_config_file(file_path: str, strict: bool = False) -> Tuple[bool, List[ValidationMessage]]:
    """
    Validate a model configuration file
    
    Args:
        file_path: Path to models.yaml file
        strict: Whether to treat warnings as errors
        
    Returns:
        Tuple of (is_valid, validation_messages)
    """
    try:
        import yaml
        
        if not os.path.exists(file_path):
            return False, [ValidationMessage(
                severity=ValidationSeverity.ERROR,
                message=f"Configuration file not found: {file_path}",
                path="file",
                suggestion="Create a models.yaml configuration file"
            )]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        validator = ModelConfigValidator()
        return validator.validate(config, strict)
        
    except yaml.YAMLError as e:
        return False, [ValidationMessage(
            severity=ValidationSeverity.ERROR,
            message=f"YAML parsing error: {e}",
            path="file",
            details="Fix YAML syntax errors",
            suggestion="Check indentation, quotes, and YAML structure"
        )]
    except Exception as e:
        return False, [ValidationMessage(
            severity=ValidationSeverity.ERROR,
            message=f"Validation error: {e}",
            path="system",
            details=str(e)
        )]


def format_validation_report(messages: List[ValidationMessage], show_info: bool = True) -> str:
    """
    Format validation messages into a human-readable report
    
    Args:
        messages: List of validation messages
        show_info: Whether to include info-level messages
        
    Returns:
        Formatted report string
    """
    if not messages:
        return "✅ Configuration validation passed with no issues."
    
    # Group messages by severity
    errors = [msg for msg in messages if msg.severity == ValidationSeverity.ERROR]
    warnings = [msg for msg in messages if msg.severity == ValidationSeverity.WARNING]
    infos = [msg for msg in messages if msg.severity == ValidationSeverity.INFO]
    
    report_lines = []
    
    # Summary
    if errors:
        report_lines.append(f"❌ Validation failed with {len(errors)} error(s)")
    elif warnings:
        report_lines.append(f"⚠️ Validation passed with {len(warnings)} warning(s)")
    else:
        report_lines.append("✅ Configuration validation passed")
    
    if warnings:
        report_lines.append(f"   {len(warnings)} warning(s) found")
    if infos and show_info:
        report_lines.append(f"   {len(infos)} info message(s)")
    
    report_lines.append("")
    
    # Error details
    if errors:
        report_lines.append("🚨 ERRORS (must fix):")
        for error in errors:
            report_lines.append(f"   • {error.path}: {error.message}")
            if error.details:
                report_lines.append(f"     Details: {error.details}")
            if error.suggestion:
                report_lines.append(f"     💡 Suggestion: {error.suggestion}")
        report_lines.append("")
    
    # Warning details
    if warnings:
        report_lines.append("⚠️ WARNINGS (recommended to fix):")
        for warning in warnings:
            report_lines.append(f"   • {warning.path}: {warning.message}")
            if warning.details:
                report_lines.append(f"     Details: {warning.details}")
            if warning.suggestion:
                report_lines.append(f"     💡 Suggestion: {warning.suggestion}")
        report_lines.append("")
    
    # Info details
    if infos and show_info:
        report_lines.append("ℹ️ INFO:")
        for info in infos:
            report_lines.append(f"   • {info.path}: {info.message}")
            if info.suggestion:
                report_lines.append(f"     💡 {info.suggestion}")
        report_lines.append("")
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Command-line validation tool
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python model_validation.py <models.yaml>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    is_valid, messages = validate_model_config_file(file_path, strict=False)
    
    print(format_validation_report(messages))
    
    if not is_valid:
        sys.exit(1)
    else:
        print("✅ Validation completed successfully!")