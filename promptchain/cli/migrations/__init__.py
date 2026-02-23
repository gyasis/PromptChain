"""Database migration module for PromptChain CLI.

This module contains database schema migrations for evolving the CLI
persistence layer while maintaining backward compatibility.

Available Migrations:
- v2_schema: Add orchestration support (AgentChain, MCP, workflows)
"""

from .v2_schema import (
    MIGRATION_VERSION,
    migrate_v1_to_v2,
    rollback_migration,
    verify_migration,
)

__all__ = [
    "MIGRATION_VERSION",
    "migrate_v1_to_v2",
    "verify_migration",
    "rollback_migration",
]
