"""
ChainBuilder - Agent-Facing API for Self-Writing Chains.

This module provides the interface that allows LLM agents to create,
modify, and evolve chains programmatically. It's designed to be
exposed as a tool that agents can call.

Key Features:
- Fluent builder API for chain construction
- Validation before save
- Version management (auto-increment)
- Template-based chain creation
- Tool functions for agent access

Usage by Agents:
    # Agent can create chains via tool calls
    result = chain_builder.create_chain(
        name="my-workflow",
        description="A custom workflow",
        steps=[
            {"type": "prompt", "content": "Analyze: {input}"},
            {"type": "chain", "chain_id": "validator:v1.0"},
            {"type": "prompt", "content": "Summarize: {input}"}
        ]
    )
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .chain_factory import ChainFactory, ChainNotFoundError
from .chain_models import (ChainDefinition, ChainMode, ChainStepDefinition,
                           Guardrails, StepType, ValidationResult)

logger = logging.getLogger(__name__)


class ChainBuilder:
    """Fluent builder for creating chain definitions.

    Provides a chainable API for building chains step by step:

        chain = (ChainBuilder("my-chain")
            .description("My custom chain")
            .add_prompt("Analyze: {input}")
            .add_chain("validator:v1.0")
            .add_prompt("Summarize: {input}")
            .build())

    Or use the static methods for direct tool access:

        result = ChainBuilder.create_chain(
            name="my-chain",
            steps=[{"type": "prompt", "content": "..."}]
        )
    """

    def __init__(
        self, name: str, version: str = "v1.0", mode: ChainMode = ChainMode.STRICT
    ):
        """Initialize builder.

        Args:
            name: Chain model name
            version: Version string (e.g., "v1.0")
            mode: Execution mode (strict or hybrid)
        """
        self._name = name
        self._version = version
        self._mode = mode
        self._description: Optional[str] = None
        self._llm_model = "openai/gpt-4.1-mini-2025-04-14"
        self._steps: List[ChainStepDefinition] = []
        self._tags: List[str] = []
        self._guardrails = Guardrails()
        self._created_by = "agent"

    # =========================================================================
    # Fluent Builder Methods
    # =========================================================================

    def description(self, desc: str) -> "ChainBuilder":
        """Set chain description."""
        self._description = desc
        return self

    def llm_model(self, model: str) -> "ChainBuilder":
        """Set LLM model for prompt steps."""
        self._llm_model = model
        return self

    def tags(self, *tags: str) -> "ChainBuilder":
        """Add tags to chain."""
        self._tags.extend(tags)
        return self

    def created_by(self, creator: str) -> "ChainBuilder":
        """Set creator (e.g., 'agent:gpt-4')."""
        self._created_by = creator
        return self

    def guardrails(
        self,
        max_steps: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_nested_depth: Optional[int] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ) -> "ChainBuilder":
        """Configure guardrails."""
        if max_steps is not None:
            self._guardrails.max_steps = max_steps
        if timeout_seconds is not None:
            self._guardrails.timeout_seconds = timeout_seconds
        if max_nested_depth is not None:
            self._guardrails.max_nested_depth = max_nested_depth
        if forbidden_patterns is not None:
            self._guardrails.forbidden_patterns = forbidden_patterns
        return self

    # =========================================================================
    # Step Adding Methods
    # =========================================================================

    def add_step(self, step: ChainStepDefinition) -> "ChainBuilder":
        """Add a pre-built step."""
        self._steps.append(step)
        return self

    def add_prompt(
        self,
        content: str,
        step_id: Optional[str] = None,
        prompt_id: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> "ChainBuilder":
        """Add a prompt step.

        Args:
            content: Prompt template (use {input} for input)
            step_id: Optional step ID (auto-generated if not provided)
            prompt_id: Optional PrePrompt ID to load instead of content
            strategy: Optional strategy (cot, react, etc.)
        """
        step_id = step_id or f"prompt_{len(self._steps) + 1}"
        step = ChainStepDefinition(
            id=step_id,
            type=StepType.PROMPT,
            content=content if not prompt_id else None,
            prompt_id=prompt_id,
            strategy=strategy,
        )
        self._steps.append(step)
        return self

    def add_chain(
        self, chain_ref: str, step_id: Optional[str] = None
    ) -> "ChainBuilder":
        """Add a nested chain step.

        Args:
            chain_ref: Chain reference (model:version or model)
            step_id: Optional step ID
        """
        step_id = step_id or f"chain_{len(self._steps) + 1}"
        step = ChainStepDefinition(id=step_id, type=StepType.CHAIN, chain_id=chain_ref)
        self._steps.append(step)
        return self

    def add_function(
        self, function_name: str, step_id: Optional[str] = None
    ) -> "ChainBuilder":
        """Add a function step.

        Args:
            function_name: Name of registered function
            step_id: Optional step ID
        """
        step_id = step_id or f"func_{len(self._steps) + 1}"
        step = ChainStepDefinition(
            id=step_id, type=StepType.FUNCTION, function_name=function_name
        )
        self._steps.append(step)
        return self

    def add_agentic(
        self,
        objective: str,
        step_id: Optional[str] = None,
        max_steps: int = 5,
        tools: Optional[List[str]] = None,
    ) -> "ChainBuilder":
        """Add an agentic step (hybrid mode only).

        Args:
            objective: Objective for the AgenticStepProcessor
            step_id: Optional step ID
            max_steps: Max internal reasoning steps
            tools: Allowed tools for agentic step
        """
        if self._mode != ChainMode.HYBRID:
            logger.warning("Agentic steps only work in hybrid mode. Switching mode.")
            self._mode = ChainMode.HYBRID

        step_id = step_id or f"agentic_{len(self._steps) + 1}"
        step = ChainStepDefinition(
            id=step_id,
            type=StepType.AGENTIC,
            objective=objective,
            max_steps=max_steps,
            tools=tools,
        )
        self._steps.append(step)
        return self

    # =========================================================================
    # Build Methods
    # =========================================================================

    def build(self) -> ChainDefinition:
        """Build the chain definition.

        Returns:
            ChainDefinition ready for saving or execution
        """
        if not self._steps:
            raise ValueError("Chain must have at least one step")

        return ChainDefinition(
            vin="",  # Auto-generated
            model=self._name,
            version=self._version,
            description=self._description,
            tags=self._tags,
            mode=self._mode,
            llm_model=self._llm_model,
            guardrails=self._guardrails,
            steps=self._steps,
            created_by=self._created_by,
        )

    def save(self, factory: Optional[ChainFactory] = None) -> str:
        """Build and save the chain.

        Args:
            factory: ChainFactory to use (creates default if None)

        Returns:
            Path where chain was saved
        """
        factory = factory or ChainFactory()
        chain_def = self.build()
        return factory.save(chain_def)

    # =========================================================================
    # Static Tool Methods (for Agent Access)
    # =========================================================================

    @staticmethod
    def create_chain(
        name: str,
        steps: List[Dict[str, Any]],
        description: Optional[str] = None,
        version: str = "v1.0",
        mode: str = "strict",
        llm_model: str = "openai/gpt-4.1-mini-2025-04-14",
        tags: Optional[List[str]] = None,
        created_by: str = "agent",
    ) -> Dict[str, Any]:
        """Create a new chain (tool function for agents).

        Args:
            name: Chain model name
            steps: List of step definitions:
                - {"type": "prompt", "content": "..."}
                - {"type": "chain", "chain_id": "model:version"}
                - {"type": "function", "function_name": "..."}
                - {"type": "agentic", "objective": "...", "max_steps": 5}
            description: Chain description
            version: Version string
            mode: "strict" or "hybrid"
            llm_model: LLM model for prompt steps
            tags: Optional tags
            created_by: Creator identifier

        Returns:
            Dict with success status, vin, and path
        """
        try:
            builder = ChainBuilder(name, version, ChainMode(mode))
            builder.description(description or f"Chain: {name}")
            builder.llm_model(llm_model)
            builder.created_by(created_by)

            if tags:
                builder.tags(*tags)

            # Add steps
            for step_data in steps:
                step_type = step_data.get("type", "prompt")

                if step_type == "prompt":
                    builder.add_prompt(
                        content=step_data.get("content", ""),
                        step_id=step_data.get("id"),
                        prompt_id=step_data.get("prompt_id"),
                        strategy=step_data.get("strategy"),
                    )
                elif step_type == "chain":
                    builder.add_chain(
                        chain_ref=step_data.get("chain_id", ""),
                        step_id=step_data.get("id"),
                    )
                elif step_type == "function":
                    builder.add_function(
                        function_name=step_data.get("function_name", ""),
                        step_id=step_data.get("id"),
                    )
                elif step_type == "agentic":
                    builder.add_agentic(
                        objective=step_data.get("objective", ""),
                        step_id=step_data.get("id"),
                        max_steps=step_data.get("max_steps", 5),
                        tools=step_data.get("tools"),
                    )

            # Build and save
            chain_def = builder.build()
            factory = ChainFactory()

            # Validate before save
            validation = factory.validate(chain_def)
            if not validation.passed:
                errors = [i.message for i in validation.issues if i.severity == "error"]
                return {
                    "success": False,
                    "error": f"Validation failed: {'; '.join(errors)}",
                }

            path = factory.save(chain_def)

            return {
                "success": True,
                "vin": chain_def.vin,
                "model": chain_def.model,
                "version": chain_def.version,
                "path": path,
                "steps_count": len(chain_def.steps),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def modify_chain(chain_ref: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing chain (tool function for agents).

        Args:
            chain_ref: Chain reference (model:version or model)
            modifications: Dict with fields to modify:
                - description: New description
                - add_steps: List of steps to add
                - remove_steps: List of step IDs to remove
                - update_steps: Dict of step_id -> updates
                - guardrails: Guardrail updates

        Returns:
            Dict with success status and new version info
        """
        try:
            factory = ChainFactory()
            chain_def = factory.resolve(chain_ref)

            # Apply modifications
            if "description" in modifications:
                chain_def.description = modifications["description"]

            if "add_steps" in modifications:
                for step_data in modifications["add_steps"]:
                    step = ChainStepDefinition(
                        id=step_data.get("id", f"step_{len(chain_def.steps) + 1}"),
                        type=StepType(step_data.get("type", "prompt")),
                        content=step_data.get("content"),
                        chain_id=step_data.get("chain_id"),
                        function_name=step_data.get("function_name"),
                        objective=step_data.get("objective"),
                    )
                    chain_def.steps.append(step)

            if "remove_steps" in modifications:
                remove_ids = set(modifications["remove_steps"])
                chain_def.steps = [s for s in chain_def.steps if s.id not in remove_ids]

            if "update_steps" in modifications:
                for step_id, updates in modifications["update_steps"].items():
                    for step in chain_def.steps:
                        if step.id == step_id:
                            for key, value in updates.items():
                                if hasattr(step, key):
                                    setattr(step, key, value)
                            break

            if "guardrails" in modifications:
                for key, value in modifications["guardrails"].items():
                    if hasattr(chain_def.guardrails, key):
                        setattr(chain_def.guardrails, key, value)

            # Increment version
            current_version = chain_def.version.lstrip("v")
            parts = current_version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            chain_def.version = "v" + ".".join(parts)

            # Generate new VIN
            chain_def.vin = ""  # Will be auto-generated

            # Validate
            validation = factory.validate(chain_def)
            if not validation.passed:
                errors = [i.message for i in validation.issues if i.severity == "error"]
                return {
                    "success": False,
                    "error": f"Validation failed: {'; '.join(errors)}",
                }

            # Save new version
            path = factory.save(chain_def)

            return {
                "success": True,
                "vin": chain_def.vin,
                "model": chain_def.model,
                "version": chain_def.version,
                "path": path,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def clone_chain(
        source_ref: str, new_name: str, modifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Clone an existing chain with optional modifications.

        Args:
            source_ref: Source chain reference
            new_name: Name for the cloned chain
            modifications: Optional modifications to apply

        Returns:
            Dict with success status and new chain info
        """
        try:
            factory = ChainFactory()
            source = factory.resolve(source_ref)

            # Create clone
            clone = ChainDefinition(
                vin="",  # Auto-generated
                model=new_name,
                version="v1.0",
                description=source.description,
                tags=source.tags.copy(),
                mode=source.mode,
                llm_model=source.llm_model,
                guardrails=Guardrails(**source.guardrails.model_dump()),
                steps=[ChainStepDefinition(**s.model_dump()) for s in source.steps],
                created_by="agent:clone",
            )

            # Apply modifications
            if modifications:
                if "description" in modifications:
                    clone.description = modifications["description"]
                if "tags" in modifications:
                    clone.tags = modifications["tags"]
                # Add more modification handling as needed

            path = factory.save(clone)

            return {
                "success": True,
                "vin": clone.vin,
                "model": clone.model,
                "version": clone.version,
                "path": path,
                "cloned_from": source_ref,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# =========================================================================
# Tool Definitions for Agent Registration
# =========================================================================


def get_chain_builder_tools() -> List[Dict[str, Any]]:
    """Get tool definitions for ChainBuilder methods.

    Returns tool schemas in OpenAI function format that can be
    registered with PromptChain or AgentChain.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "create_chain",
                "description": "Create a new chain workflow with specified steps. Chains are strict, guardrailed LLM workflows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Chain model name (e.g., 'my-workflow')",
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "prompt",
                                            "chain",
                                            "function",
                                            "agentic",
                                        ],
                                        "description": "Step type",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Prompt content (for prompt steps)",
                                    },
                                    "chain_id": {
                                        "type": "string",
                                        "description": "Chain reference (for chain steps)",
                                    },
                                    "function_name": {
                                        "type": "string",
                                        "description": "Function name (for function steps)",
                                    },
                                    "objective": {
                                        "type": "string",
                                        "description": "Objective (for agentic steps)",
                                    },
                                },
                                "required": ["type"],
                            },
                            "description": "List of steps in the chain",
                        },
                        "description": {
                            "type": "string",
                            "description": "Chain description",
                        },
                        "version": {
                            "type": "string",
                            "description": "Version string (e.g., 'v1.0')",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["strict", "hybrid"],
                            "description": "Execution mode",
                        },
                    },
                    "required": ["name", "steps"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "modify_chain",
                "description": "Modify an existing chain. Creates a new version with the modifications.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chain_ref": {
                            "type": "string",
                            "description": "Chain reference (model:version or model)",
                        },
                        "modifications": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "New description",
                                },
                                "add_steps": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                    "description": "Steps to add",
                                },
                                "remove_steps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Step IDs to remove",
                                },
                            },
                            "description": "Modifications to apply",
                        },
                    },
                    "required": ["chain_ref", "modifications"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "clone_chain",
                "description": "Clone an existing chain with a new name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_ref": {
                            "type": "string",
                            "description": "Source chain reference",
                        },
                        "new_name": {
                            "type": "string",
                            "description": "Name for the cloned chain",
                        },
                        "modifications": {
                            "type": "object",
                            "description": "Optional modifications to apply to clone",
                        },
                    },
                    "required": ["source_ref", "new_name"],
                },
            },
        },
    ]


def get_chain_builder_functions() -> Dict[str, Any]:
    """Get function mappings for tool execution.

    Returns dict mapping tool names to their implementations.
    """
    return {
        "create_chain": ChainBuilder.create_chain,
        "modify_chain": ChainBuilder.modify_chain,
        "clone_chain": ChainBuilder.clone_chain,
    }
