"""
Configuration management for Research Agent System

Handles loading and validation of YAML configuration with environment variable support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class SystemConfig:
    """System-level configuration"""
    name: str = "Research Agent System"
    version: str = "0.1.0"
    description: str = ""


@dataclass
class LiteratureSearchConfig:
    """Literature search configuration"""
    enabled_sources: list = field(default_factory=lambda: ["sci_hub", "arxiv", "pubmed"])
    sci_hub: Dict[str, Any] = field(default_factory=dict)
    arxiv: Dict[str, Any] = field(default_factory=dict)
    pubmed: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreeTierRAGConfig:
    """3-Tier RAG system configuration"""
    execution_mode: str = "sequential"
    tier1_lightrag: Dict[str, Any] = field(default_factory=dict)
    tier2_paperqa2: Dict[str, Any] = field(default_factory=dict)
    tier3_graphrag: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptChainConfig:
    """PromptChain integration configuration"""
    agent_chain: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, Any] = field(default_factory=dict)
    router: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSessionConfig:
    """Research session management configuration"""
    max_iterations: int = 5
    max_queries_per_iteration: int = 15
    min_papers_per_query: int = 5
    max_papers_total: int = 100
    completeness_threshold: float = 0.85
    gap_detection_threshold: float = 0.7
    citation_minimum: int = 3
    save_intermediate_results: bool = True
    session_timeout: int = 7200
    auto_save_interval: int = 300


@dataclass
class WebInterfaceConfig:
    """Web interface configuration"""
    fastapi: Dict[str, Any] = field(default_factory=dict)
    chainlit: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    console: Dict[str, Any] = field(default_factory=dict)
    file: Dict[str, Any] = field(default_factory=dict)
    jsonl: Dict[str, Any] = field(default_factory=dict)


class ResearchConfig:
    """Main configuration manager for Research Agent System"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config_data = {}
        
        # Load environment variables
        load_dotenv()
        
        # Load and validate configuration
        self.reload()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations"""
        possible_paths = [
            "./config/research_config.yaml",
            "./research_config.yaml",
            os.path.expanduser("~/.research_agent/config.yaml"),
            "/etc/research_agent/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Default to the first location
        return possible_paths[0]
    
    def reload(self):
        """Reload configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config_data = yaml.safe_load(f)
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Validate configuration
            self._validate_config()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_overrides = self.config_data.get('env_overrides', {})
        
        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(self.config_data, config_path, env_value)
    
    def _set_nested_value(self, data: dict, path: str, value: str):
        """Set a nested dictionary value using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _validate_config(self):
        """Validate configuration structure and required values"""
        required_sections = [
            'system', 'literature_search', 'three_tier_rag', 
            'promptchain', 'research_session'
        ]
        
        for section in required_sections:
            if section not in self.config_data:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """Set configuration value using dot notation"""
        self._set_nested_value(self.config_data, path, value)
    
    @property
    def system(self) -> SystemConfig:
        """Get system configuration"""
        data = self.config_data.get('system', {})
        return SystemConfig(**data)
    
    @property
    def literature_search(self) -> LiteratureSearchConfig:
        """Get literature search configuration"""
        data = self.config_data.get('literature_search', {})
        return LiteratureSearchConfig(**data)
    
    @property
    def three_tier_rag(self) -> ThreeTierRAGConfig:
        """Get 3-tier RAG configuration"""
        data = self.config_data.get('three_tier_rag', {})
        return ThreeTierRAGConfig(**data)
    
    @property
    def promptchain(self) -> PromptChainConfig:
        """Get PromptChain configuration"""
        data = self.config_data.get('promptchain', {})
        return PromptChainConfig(**data)
    
    @property
    def research_session(self) -> ResearchSessionConfig:
        """Get research session configuration"""
        data = self.config_data.get('research_session', {})
        return ResearchSessionConfig(**data)
    
    @property
    def web_interface(self) -> WebInterfaceConfig:
        """Get web interface configuration"""
        data = self.config_data.get('web_interface', {})
        return WebInterfaceConfig(**data)
    
    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration"""
        data = self.config_data.get('logging', {})
        return LoggingConfig(**data)
    
    def get_tier_config(self, tier: str, mode: str) -> Dict[str, Any]:
        """Get configuration for a specific tier and mode"""
        tier_data = self.config_data.get('three_tier_rag', {}).get(f'tier{tier}_{tier.lower()}', {})
        return tier_data.get(mode, {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        agents_config = self.config_data.get('promptchain', {}).get('agents', {})
        return agents_config.get(agent_name, {})
    
    def is_enabled(self, feature_path: str) -> bool:
        """Check if a feature is enabled"""
        return self.get(feature_path + '.enabled', False)
    
    def get_model_config(self, tier: str, mode: str) -> Dict[str, Any]:
        """Get model configuration for a tier and mode"""
        tier_config = self.get_tier_config(tier, mode)
        
        if mode == "ollama":
            return {
                "host": tier_config.get("host", "http://localhost:11434"),
                "llm_model": tier_config.get("llm_model"),
                "embedding_model": tier_config.get("embedding_model"),
                "api_base": tier_config.get("api_base", "http://localhost:11434")
            }
        else:  # cloud mode
            return {
                "llm_model": tier_config.get("llm_model"),
                "embedding_model": tier_config.get("embedding_model"),
                "temperature": tier_config.get("temperature", 0.1)
            }
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
    
    def __str__(self) -> str:
        return f"ResearchConfig(path={self.config_path})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Global configuration instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> ResearchConfig:
    """Get or create global configuration instance"""
    global _config_instance
    
    if _config_instance is None or config_path is not None:
        _config_instance = ResearchConfig(config_path)
    
    return _config_instance


def reload_config():
    """Reload global configuration"""
    global _config_instance
    if _config_instance:
        _config_instance.reload()