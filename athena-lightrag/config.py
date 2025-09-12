 the Path#!/usr/bin/env python3
"""
Athena LightRAG Configuration
============================
Configuration management and validation for Athena LightRAG system.

Author: Athena LightRAG System
Date: 2025-09-08
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

from exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

# Load environment variables - override system env vars with project .env
load_dotenv(override=True)

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db"
    validate_on_init: bool = True
    required_files: List[str] = field(default_factory=lambda: [
        "kv_store_full_entities.json",
        "kv_store_full_relations.json", 
        "kv_store_text_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "vdb_chunks.json"
    ])
    
    def validate(self) -> bool:
        """Validate database configuration."""
        if not self.validate_on_init:
            return True
            
        db_path = Path(self.working_dir)
        if not db_path.exists():
            raise ConfigurationError(
                f"Database directory does not exist: {self.working_dir}",
                config_field="working_dir"
            )
        
        missing_files = []
        for file_name in self.required_files:
            file_path = db_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"Missing database files: {missing_files}")
            # Don't raise error for missing files as they might be optional
        
        return True

@dataclass 
class LLMConfig:
    """LLM configuration settings."""
    api_key: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536
    max_async: int = 4
    enable_cache: bool = True
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: int = 300
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide it in LLMConfig.",
                config_field="api_key"
            )
        
        self.validate()
    
    def validate(self) -> bool:
        """Validate LLM configuration."""
        if self.max_async < 1 or self.max_async > 20:
            raise ValidationError(
                "max_async must be between 1 and 20",
                field_name="max_async",
                field_value=self.max_async,
                expected_type="int (1-20)"
            )
        
        if self.embedding_dim not in [384, 512, 768, 1024, 1536, 3072]:
            raise ValidationError(
                "embedding_dim must be a valid embedding dimension",
                field_name="embedding_dim", 
                field_value=self.embedding_dim,
                expected_type="int (384, 512, 768, 1024, 1536, 3072)"
            )
        
        if not 0 <= self.temperature <= 2:
            raise ValidationError(
                "temperature must be between 0 and 2",
                field_name="temperature",
                field_value=self.temperature,
                expected_type="float (0-2)"
            )
        
        return True

@dataclass
class QueryConfig:
    """Query configuration settings."""
    default_mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid"
    default_top_k: int = 60
    default_max_entity_tokens: int = 6000
    default_max_relation_tokens: int = 8000
    default_max_total_tokens: int = 30000
    default_response_type: str = "Multiple Paragraphs"
    enable_rerank: bool = True
    chunk_top_k: int = 20
    
    def validate(self) -> bool:
        """Validate query configuration."""
        if self.default_top_k < 1 or self.default_top_k > 1000:
            raise ValidationError(
                "default_top_k must be between 1 and 1000",
                field_name="default_top_k",
                field_value=self.default_top_k,
                expected_type="int (1-1000)"
            )
        
        token_fields = [
            ("default_max_entity_tokens", self.default_max_entity_tokens),
            ("default_max_relation_tokens", self.default_max_relation_tokens),
            ("default_max_total_tokens", self.default_max_total_tokens)
        ]
        
        for field_name, value in token_fields:
            if value < 100 or value > 50000:
                raise ValidationError(
                    f"{field_name} must be between 100 and 50000",
                    field_name=field_name,
                    field_value=value,
                    expected_type="int (100-50000)"
                )
        
        return True

@dataclass
class AgenticConfig:
    """Agentic reasoning configuration."""
    model_name: str = "openai/gpt-4"
    max_internal_steps: int = 8
    enable_verbose: bool = True
    enable_step_storage: bool = True
    reasoning_timeout_seconds: int = 600
    max_context_accumulation: int = 10
    confidence_threshold: float = 0.3
    
    def validate(self) -> bool:
        """Validate agentic configuration."""
        if self.max_internal_steps < 1 or self.max_internal_steps > 20:
            raise ValidationError(
                "max_internal_steps must be between 1 and 20",
                field_name="max_internal_steps",
                field_value=self.max_internal_steps,
                expected_type="int (1-20)"
            )
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValidationError(
                "confidence_threshold must be between 0 and 1",
                field_name="confidence_threshold",
                field_value=self.confidence_threshold,
                expected_type="float (0-1)"
            )
        
        return True

@dataclass
class MCPServerConfig:
    """MCP Server configuration."""
    server_name: str = "athena-lightrag"
    server_version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8000
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300
    enable_request_logging: bool = True
    
    def validate(self) -> bool:
        """Validate MCP server configuration."""
        if not 1000 <= self.port <= 65535:
            raise ValidationError(
                "port must be between 1000 and 65535",
                field_name="port",
                field_value=self.port,
                expected_type="int (1000-65535)"
            )
        
        if self.max_request_size < 1024 or self.max_request_size > 100 * 1024 * 1024:
            raise ValidationError(
                "max_request_size must be between 1KB and 100MB",
                field_name="max_request_size",
                field_value=self.max_request_size,
                expected_type="int (1024 - 104857600)"
            )
        
        return True

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_file_logging: bool = True
    log_file: str = "athena_lightrag.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    enable_structured_logging: bool = True
    
    def validate(self) -> bool:
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValidationError(
                f"level must be one of {valid_levels}",
                field_name="level",
                field_value=self.level,
                expected_type=f"str ({', '.join(valid_levels)})"
            )
        
        return True

@dataclass
class AthenaConfig:
    """Main Athena LightRAG configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    mcp_server: MCPServerConfig = field(default_factory=MCPServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> bool:
        """Validate all configuration sections."""
        try:
            self.database.validate()
            self.llm.validate()
            self.query.validate()
            self.agentic.validate()
            self.mcp_server.validate()
            self.logging.validate()
            return True
        except (ConfigurationError, ValidationError) as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "database": self.database.__dict__,
            "llm": {**self.llm.__dict__, "api_key": "***HIDDEN***"},  # Hide API key
            "query": self.query.__dict__,
            "agentic": self.agentic.__dict__,
            "mcp_server": self.mcp_server.__dict__,
            "logging": self.logging.__dict__
        }
    
    @classmethod
    def from_env(cls) -> 'AthenaConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("ATHENA_WORKING_DIR"):
            config.database.working_dir = os.getenv("ATHENA_WORKING_DIR")
        
        if os.getenv("ATHENA_MODEL_NAME"):
            config.llm.model_name = os.getenv("ATHENA_MODEL_NAME")
        
        if os.getenv("ATHENA_EMBEDDING_MODEL"):
            config.llm.embedding_model = os.getenv("ATHENA_EMBEDDING_MODEL")
        
        if os.getenv("ATHENA_MAX_ASYNC"):
            config.llm.max_async = int(os.getenv("ATHENA_MAX_ASYNC"))
        
        if os.getenv("ATHENA_DEFAULT_MODE"):
            config.query.default_mode = os.getenv("ATHENA_DEFAULT_MODE")
        
        if os.getenv("ATHENA_SERVER_PORT"):
            config.mcp_server.port = int(os.getenv("ATHENA_SERVER_PORT"))
        
        if os.getenv("ATHENA_LOG_LEVEL"):
            config.logging.level = os.getenv("ATHENA_LOG_LEVEL")
        
        # Validate the configuration
        config.validate()
        
        return config
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format
        )
        
        # Configure file logging if enabled
        if self.logging.enable_file_logging:
            from logging.handlers import RotatingFileHandler
            
            handler = RotatingFileHandler(
                self.logging.log_file,
                maxBytes=self.logging.log_max_size,
                backupCount=self.logging.log_backup_count
            )
            handler.setFormatter(logging.Formatter(self.logging.format))
            
            # Add handler to root logger
            logging.getLogger().addHandler(handler)


# Global configuration instance
_global_config: Optional[AthenaConfig] = None

def get_config() -> AthenaConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = AthenaConfig.from_env()
        _global_config.setup_logging()
    return _global_config

def set_config(config: AthenaConfig):
    """Set global configuration instance.""" 
    global _global_config
    config.validate()
    _global_config = config
    _global_config.setup_logging()

def reset_config():
    """Reset global configuration."""
    global _global_config
    _global_config = None