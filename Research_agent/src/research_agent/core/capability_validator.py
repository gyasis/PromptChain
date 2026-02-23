"""
Model capability validation system for Research Agent.

Ensures that models assigned to specific tasks have the required capabilities
to prevent runtime failures from incompatible model-task assignments.
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model type classifications"""
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    CHAT = "chat"
    COMPLETION = "completion"
    CODE_GENERATION = "code_generation"
    MULTIMODAL = "multimodal"


class TaskCapability(Enum):
    """Required capabilities for different research tasks"""
    QUERY_GENERATION = "query_generation"
    SEARCH = "search"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    MULTI_QUERY = "multi_query"
    CHAT = "chat"
    ROUTING = "routing"
    EMBEDDING = "embedding"
    CODE = "code"
    REASONING = "reasoning"


@dataclass
class CapabilityRequirement:
    """Defines capability requirements for a task"""
    task_name: str
    required_capabilities: Set[TaskCapability]
    forbidden_capabilities: Set[TaskCapability]
    min_max_tokens: Optional[int] = None
    requires_reasoning: bool = False
    requires_function_calling: bool = False


@dataclass  
class ModelCapabilities:
    """Defines what capabilities a model has"""
    model_name: str
    model_type: ModelType
    capabilities: Set[TaskCapability]
    max_tokens: Optional[int] = None
    supports_function_calling: bool = False
    supports_streaming: bool = False
    cost_tier: str = "medium"  # low, medium, high


class CapabilityValidator:
    """
    Validates model-task assignments based on capabilities.
    
    Prevents assigning incompatible models to tasks (e.g., embedding models
    for text generation, or weak models for complex reasoning).
    """
    
    def __init__(self):
        self._task_requirements = self._define_task_requirements()
        self._model_capabilities = self._define_model_capabilities()
    
    def _define_task_requirements(self) -> Dict[str, CapabilityRequirement]:
        """Define capability requirements for each research task"""
        return {
            "query_generation": CapabilityRequirement(
                task_name="query_generation",
                required_capabilities={TaskCapability.QUERY_GENERATION, TaskCapability.REASONING},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=2000,
                requires_reasoning=True
            ),
            
            "search_strategy": CapabilityRequirement(
                task_name="search_strategy",
                required_capabilities={TaskCapability.SEARCH, TaskCapability.ANALYSIS},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=1000,
                requires_reasoning=True
            ),
            
            "literature_search": CapabilityRequirement(
                task_name="literature_search",
                required_capabilities={TaskCapability.SEARCH},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=1000,
                requires_function_calling=True
            ),
            
            "multi_query_coordination": CapabilityRequirement(
                task_name="multi_query_coordination", 
                required_capabilities={TaskCapability.MULTI_QUERY, TaskCapability.ANALYSIS, TaskCapability.REASONING},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=3000,
                requires_reasoning=True
            ),
            
            "react_analysis": CapabilityRequirement(
                task_name="react_analysis",
                required_capabilities={TaskCapability.ANALYSIS, TaskCapability.REASONING},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=3000,
                requires_reasoning=True
            ),
            
            "synthesis": CapabilityRequirement(
                task_name="synthesis",
                required_capabilities={TaskCapability.SYNTHESIS, TaskCapability.REASONING},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=4000,
                requires_reasoning=True
            ),
            
            "chat_interface": CapabilityRequirement(
                task_name="chat_interface",
                required_capabilities={TaskCapability.CHAT},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=2000
            ),
            
            "embedding_default": CapabilityRequirement(
                task_name="embedding_default",
                required_capabilities={TaskCapability.EMBEDDING},
                forbidden_capabilities={TaskCapability.QUERY_GENERATION, TaskCapability.CHAT, 
                                       TaskCapability.ANALYSIS, TaskCapability.SYNTHESIS}
            ),
            
            "tier1_lightrag": CapabilityRequirement(
                task_name="tier1_lightrag",
                required_capabilities={TaskCapability.ANALYSIS, TaskCapability.REASONING},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=2000
            ),
            
            "tier2_paperqa2": CapabilityRequirement(
                task_name="tier2_paperqa2",
                required_capabilities={TaskCapability.ANALYSIS, TaskCapability.REASONING},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=3000,
                requires_reasoning=True
            ),
            
            "tier3_graphrag": CapabilityRequirement(
                task_name="tier3_graphrag",
                required_capabilities={TaskCapability.ANALYSIS, TaskCapability.REASONING},
                forbidden_capabilities={TaskCapability.EMBEDDING},
                min_max_tokens=2000,
                requires_reasoning=True
            )
        }
    
    def _define_model_capabilities(self) -> Dict[str, ModelCapabilities]:
        """Define capabilities for known models"""
        return {
            # OpenAI Models
            "gpt-4o": ModelCapabilities(
                model_name="gpt-4o",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.QUERY_GENERATION, TaskCapability.SEARCH, 
                             TaskCapability.ANALYSIS, TaskCapability.SYNTHESIS,
                             TaskCapability.MULTI_QUERY, TaskCapability.CHAT, 
                             TaskCapability.ROUTING, TaskCapability.REASONING},
                max_tokens=4000,
                supports_function_calling=True,
                supports_streaming=True,
                cost_tier="high"
            ),
            
            "gpt-4o-mini": ModelCapabilities(
                model_name="gpt-4o-mini",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.QUERY_GENERATION, TaskCapability.SEARCH,
                             TaskCapability.ROUTING, TaskCapability.CHAT, TaskCapability.ANALYSIS},
                max_tokens=2000,
                supports_function_calling=True,
                supports_streaming=True,
                cost_tier="low"
            ),
            
            "gpt-4": ModelCapabilities(
                model_name="gpt-4",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.ANALYSIS, TaskCapability.SYNTHESIS,
                             TaskCapability.CHAT, TaskCapability.REASONING},
                max_tokens=8000,
                supports_function_calling=True,
                cost_tier="high"
            ),
            
            # Anthropic Models
            "claude-3-opus-20240229": ModelCapabilities(
                model_name="claude-3-opus-20240229",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.ANALYSIS, TaskCapability.SYNTHESIS,
                             TaskCapability.MULTI_QUERY, TaskCapability.CHAT,
                             TaskCapability.REASONING},
                max_tokens=4000,
                supports_function_calling=True,
                cost_tier="high"
            ),
            
            "claude-3-sonnet-20240229": ModelCapabilities(
                model_name="claude-3-sonnet-20240229", 
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.ANALYSIS, TaskCapability.MULTI_QUERY,
                             TaskCapability.CHAT, TaskCapability.REASONING},
                max_tokens=3000,
                supports_function_calling=True,
                cost_tier="medium"
            ),
            
            "claude-3-haiku-20240307": ModelCapabilities(
                model_name="claude-3-haiku-20240307",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.SEARCH, TaskCapability.QUERY_GENERATION, TaskCapability.ROUTING},
                max_tokens=2000,
                supports_function_calling=True,
                cost_tier="low"
            ),
            
            # Google Models
            "gemini/gemini-1.5-pro": ModelCapabilities(
                model_name="gemini/gemini-1.5-pro",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.ANALYSIS, TaskCapability.SYNTHESIS,
                             TaskCapability.MULTI_QUERY, TaskCapability.CHAT, TaskCapability.REASONING},
                max_tokens=8000,
                supports_function_calling=True,
                cost_tier="medium"
            ),
            
            "gemini/gemini-1.5-flash": ModelCapabilities(
                model_name="gemini/gemini-1.5-flash",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.SEARCH, TaskCapability.QUERY_GENERATION, TaskCapability.ROUTING},
                max_tokens=4000,
                supports_function_calling=True,
                cost_tier="low"
            ),
            
            # Local/Ollama Models (more conservative capabilities)
            "ollama/llama3.2": ModelCapabilities(
                model_name="ollama/llama3.2",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.ANALYSIS, TaskCapability.SYNTHESIS, TaskCapability.CHAT},
                max_tokens=4000,
                cost_tier="low"
            ),
            
            "ollama/mistral-nemo:latest": ModelCapabilities(
                model_name="ollama/mistral-nemo:latest",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.ANALYSIS, TaskCapability.MULTI_QUERY, TaskCapability.CHAT},
                max_tokens=8000,
                cost_tier="low"
            ),
            
            "ollama/phi3:medium": ModelCapabilities(
                model_name="ollama/phi3:medium",
                model_type=ModelType.CHAT,
                capabilities={TaskCapability.SEARCH, TaskCapability.QUERY_GENERATION},
                max_tokens=2000,
                cost_tier="low"
            ),
            
            # Embedding Models
            "text-embedding-3-large": ModelCapabilities(
                model_name="text-embedding-3-large",
                model_type=ModelType.EMBEDDING,
                capabilities={TaskCapability.EMBEDDING},
                cost_tier="low"
            ),
            
            "text-embedding-3-small": ModelCapabilities(
                model_name="text-embedding-3-small",
                model_type=ModelType.EMBEDDING,
                capabilities={TaskCapability.EMBEDDING},
                cost_tier="low"
            ),
            
            "ollama/bge-m3:latest": ModelCapabilities(
                model_name="ollama/bge-m3:latest",
                model_type=ModelType.EMBEDDING,
                capabilities={TaskCapability.EMBEDDING},
                cost_tier="low"
            ),
            
            "ollama/mxbai-embed-large": ModelCapabilities(
                model_name="ollama/mxbai-embed-large",
                model_type=ModelType.EMBEDDING,
                capabilities={TaskCapability.EMBEDDING},
                cost_tier="low"
            )
        }
    
    def validate_model_for_task(self, model_name: str, task_name: str, 
                               model_config: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Validate if a model is suitable for a specific task.
        
        Args:
            model_name: Name of the model to validate
            task_name: Name of the task
            model_config: Optional model configuration for additional validation
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Get task requirements
        if task_name not in self._task_requirements:
            issues.append(f"Unknown task: {task_name}")
            return False, issues
        
        task_req = self._task_requirements[task_name]
        
        # Get model capabilities (try direct match first, then base model name)
        model_caps = None
        if model_name in self._model_capabilities:
            model_caps = self._model_capabilities[model_name]
        else:
            # Try to match base model name (e.g., "gpt-4o" from config key "gpt-4o")
            for known_model, caps in self._model_capabilities.items():
                if known_model in model_name or model_name in known_model:
                    model_caps = caps
                    break
        
        if not model_caps:
            issues.append(f"Unknown model capabilities: {model_name}")
            return False, issues
        
        # Check required capabilities
        missing_caps = task_req.required_capabilities - model_caps.capabilities
        if missing_caps:
            missing_names = [cap.value for cap in missing_caps]
            issues.append(f"Model lacks required capabilities: {', '.join(missing_names)}")
        
        # Check forbidden capabilities
        forbidden_present = task_req.forbidden_capabilities & model_caps.capabilities
        if forbidden_present:
            forbidden_names = [cap.value for cap in forbidden_present]
            issues.append(f"Model has forbidden capabilities for this task: {', '.join(forbidden_names)}")
        
        # Check token limits
        if task_req.min_max_tokens and model_caps.max_tokens:
            if model_caps.max_tokens < task_req.min_max_tokens:
                issues.append(f"Model max_tokens ({model_caps.max_tokens}) below required minimum ({task_req.min_max_tokens})")
        
        # Additional checks from model config
        if model_config:
            config_max_tokens = model_config.get('max_tokens')
            if config_max_tokens and task_req.min_max_tokens:
                if config_max_tokens < task_req.min_max_tokens:
                    issues.append(f"Configured max_tokens ({config_max_tokens}) below required minimum ({task_req.min_max_tokens})")
        
        # Check reasoning requirements
        if task_req.requires_reasoning and TaskCapability.REASONING not in model_caps.capabilities:
            issues.append(f"Task requires reasoning capability which model may not support well")
        
        # Check function calling requirements
        if task_req.requires_function_calling and not model_caps.supports_function_calling:
            issues.append(f"Task requires function calling which model does not support")
        
        return len(issues) == 0, issues
    
    def validate_task_assignments(self, task_assignments: Dict[str, str], 
                                 model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Validate all task assignments in a configuration.
        
        Args:
            task_assignments: Dict mapping task names to model names
            model_configs: Dict mapping model names to their configurations
            
        Returns:
            Dict mapping task names to lists of validation issues
        """
        validation_results = {}
        
        for task_name, model_key in task_assignments.items():
            # Get actual model name from config
            model_config = model_configs.get(model_key, {})
            model_name = model_config.get('model', model_key)
            
            is_valid, issues = self.validate_model_for_task(model_name, task_name, model_config)
            
            if not is_valid:
                validation_results[task_name] = issues
        
        return validation_results
    
    def get_compatible_models(self, task_name: str, 
                             available_models: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Get list of models compatible with a specific task.
        
        Args:
            task_name: Name of the task
            available_models: Dict of available model configurations
            
        Returns:
            List of (model_key, model_name) tuples for compatible models
        """
        compatible = []
        
        for model_key, model_config in available_models.items():
            model_name = model_config.get('model', model_key)
            is_valid, _ = self.validate_model_for_task(model_name, task_name, model_config)
            
            if is_valid:
                compatible.append((model_key, model_name))
        
        return compatible
    
    def suggest_model_for_task(self, task_name: str, 
                              available_models: Dict[str, Dict[str, Any]],
                              prefer_cost_tier: str = "medium") -> Optional[Tuple[str, str, str]]:
        """
        Suggest the best model for a specific task.
        
        Args:
            task_name: Name of the task
            available_models: Dict of available model configurations
            prefer_cost_tier: Preferred cost tier (low, medium, high)
            
        Returns:
            Tuple of (model_key, model_name, reason) or None if no suitable model
        """
        compatible = self.get_compatible_models(task_name, available_models)
        
        if not compatible:
            return None
        
        # Score models based on suitability
        scored_models = []
        for model_key, model_name in compatible:
            model_caps = self._model_capabilities.get(model_name)
            if not model_caps:
                continue
            
            score = 0
            
            # Prefer models with cost tier match
            if model_caps.cost_tier == prefer_cost_tier:
                score += 10
            
            # Bonus for function calling if task needs it
            task_req = self._task_requirements.get(task_name)
            if task_req and task_req.requires_function_calling and model_caps.supports_function_calling:
                score += 5
            
            # Bonus for reasoning if task needs it
            if task_req and task_req.requires_reasoning and TaskCapability.REASONING in model_caps.capabilities:
                score += 3
            
            # Prefer more capable models for complex tasks
            score += len(model_caps.capabilities)
            
            scored_models.append((score, model_key, model_name, model_caps))
        
        if not scored_models:
            return None
        
        # Return highest scored model
        scored_models.sort(reverse=True)
        best_score, best_key, best_name, best_caps = scored_models[0]
        
        reason = f"Best match: {best_caps.cost_tier} cost, {len(best_caps.capabilities)} capabilities"
        return best_key, best_name, reason


# Global validator instance
_capability_validator = None


def get_capability_validator() -> CapabilityValidator:
    """Get or create global capability validator instance"""
    global _capability_validator
    if _capability_validator is None:
        _capability_validator = CapabilityValidator()
    return _capability_validator