"""
Tool Registry System for PromptChain CLI

Provides decorator-based tool registration with metadata, parameter validation,
and discovery capabilities. Follows OpenAI function calling format for compatibility
with LiteLLM and existing PromptChain tool infrastructure.

Architecture:
- ToolMetadata: Complete tool specification (name, category, description, function, schema)
- ParameterSchema: Parameter definitions with type info and validation
- ToolRegistry: Central registry with decorator-based registration and lookup
"""

import functools
import inspect
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Standard tool categories for organization."""

    FILESYSTEM = "filesystem"
    SHELL = "shell"
    SESSION = "session"
    AGENT = "agent"
    CONTEXT = "context"
    ANALYSIS = "analysis"
    UTILITY = "utility"
    CUSTOM = "custom"
    COLLABORATION = "collaboration"


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""

    pass


class ToolValidationError(Exception):
    """Raised when tool parameter validation fails."""

    pass


class ToolNotFoundError(Exception):
    """Raised when requested tool is not found."""

    pass


@dataclass
class ParameterSchema:
    """
    Parameter schema following OpenAI function calling format.

    Attributes:
        name: Parameter name
        type: JSON schema type (string, number, integer, boolean, object, array)
        description: Parameter description
        required: Whether parameter is required
        default: Default value if not required
        enum: Allowed values (for string/number types)
        properties: Nested schema for object types
        items: Item schema for array types
    """

    name: str
    type: str  # "string", "number", "integer", "boolean", "object", "array"
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None
    properties: Optional[Dict[str, "ParameterSchema"]] = None
    items: Optional["ParameterSchema"] = None

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function parameter schema format.

        Returns:
            Dictionary compatible with OpenAI function calling format
        """
        schema: Dict[str, Any] = {"type": self.type, "description": self.description}

        if self.enum is not None:
            schema["enum"] = self.enum

        if self.default is not None:
            schema["default"] = self.default

        if self.properties is not None:
            schema["properties"] = {
                k: v.to_openai_schema() for k, v in self.properties.items()
            }

        if self.items is not None:
            schema["items"] = self.items.to_openai_schema()

        return schema

    def validate_value(self, value: Any) -> None:
        """
        Validate value against parameter schema.

        Args:
            value: Value to validate

        Raises:
            ToolValidationError: If validation fails
        """
        # Type validation
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)),
            "integer": lambda v: isinstance(v, int),
            "boolean": lambda v: isinstance(v, bool),
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
        }

        validator = type_validators.get(self.type)
        if validator and not validator(value):
            raise ToolValidationError(
                f"Parameter '{self.name}' must be of type '{self.type}', got {type(value).__name__}"
            )

        # Enum validation
        if self.enum is not None and value not in self.enum:
            raise ToolValidationError(
                f"Parameter '{self.name}' must be one of {self.enum}, got {value}"
            )

        # Nested validation for objects
        if self.type == "object" and self.properties:
            if not isinstance(value, dict):
                raise ToolValidationError(
                    f"Parameter '{self.name}' must be an object/dict"
                )
            for prop_name, prop_schema in self.properties.items():
                if prop_schema.required and prop_name not in value:
                    raise ToolValidationError(
                        f"Required property '{prop_name}' missing from '{self.name}'"
                    )
                if prop_name in value:
                    prop_schema.validate_value(value[prop_name])

        # Array item validation
        if self.type == "array" and self.items:
            if not isinstance(value, list):
                raise ToolValidationError(
                    f"Parameter '{self.name}' must be an array/list"
                )
            for item in value:
                self.items.validate_value(item)


@dataclass
class ToolMetadata:
    """
    Complete tool metadata and specification.

    Attributes:
        name: Tool name (unique identifier)
        category: Tool category for organization
        description: Tool description
        parameters: Parameter schemas
        function: Callable function to execute
        tags: Additional tags for discovery
        examples: Usage examples
        allowed_agents: List of agent names that can use this tool (None = all agents)
        capabilities: List of capability tags for agent routing (FR-001 to FR-005)
    """

    name: str
    category: Union[ToolCategory, str]
    description: str
    parameters: Dict[str, ParameterSchema]
    function: Callable
    tags: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)
    # Multi-agent communication extensions (003-multi-agent-communication)
    allowed_agents: Optional[List[str]] = None  # None means all agents can use
    capabilities: List[str] = field(
        default_factory=list
    )  # e.g., ["file_read", "code_search"]

    def __post_init__(self):
        """Normalize category to ToolCategory enum."""
        if isinstance(self.category, str):
            try:
                self.category = ToolCategory(self.category)
            except ValueError:
                self.category = ToolCategory.CUSTOM

    def get_required_parameters(self) -> List[str]:
        """
        Get list of required parameter names.

        Returns:
            List of required parameter names
        """
        return [name for name, param in self.parameters.items() if param.required]

    def get_optional_parameters(self) -> List[str]:
        """
        Get list of optional parameter names.

        Returns:
            List of optional parameter names
        """
        return [name for name, param in self.parameters.items() if not param.required]

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling schema format.

        Returns:
            Dictionary compatible with OpenAI function calling format
        """
        properties = {
            name: param.to_openai_schema() for name, param in self.parameters.items()
        }

        required = self.get_required_parameters()

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required if required else [],
                },
            },
        }

    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate parameters against schema.

        Args:
            params: Parameters to validate

        Raises:
            ToolValidationError: If validation fails
        """
        # Check required parameters
        required = self.get_required_parameters()
        missing = [name for name in required if name not in params]

        if missing:
            raise ToolValidationError(
                f"Missing required parameters for tool '{self.name}': {missing}"
            )

        # Validate each provided parameter
        for name, value in params.items():
            if name not in self.parameters:
                logger.warning(
                    f"Unknown parameter '{name}' for tool '{self.name}', ignoring"
                )
                continue

            self.parameters[name].validate_value(value)

    def __call__(self, **kwargs) -> Any:
        """
        Execute tool with parameter validation.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool execution result

        Raises:
            ToolValidationError: If parameters are invalid
        """
        self.validate_parameters(kwargs)
        return self.function(**kwargs)


class ToolRegistry:
    """
    Central registry for CLI tools with decorator-based registration.

    Features:
    - Decorator-based registration (@registry.register())
    - Tool lookup by name or category
    - Parameter schema validation
    - OpenAI function calling format compatibility
    - Tag-based discovery

    Example:
        registry = ToolRegistry()

        @registry.register(
            category="filesystem",
            description="Read file contents",
            parameters={
                "path": {"type": "string", "required": True, "description": "File path"},
                "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"}
            }
        )
        def fs_read(path: str, encoding: str = "utf-8") -> str:
            with open(path, encoding=encoding) as f:
                return f.read()
    """

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, ToolMetadata] = {}
        self._category_index: Dict[ToolCategory, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._capability_index: Dict[str, Set[str]] = {}  # capability -> tool names

        # Initialize category index
        for category in ToolCategory:
            self._category_index[category] = set()

    def _dict_to_param_schema(self, name: str, spec: Dict[str, Any]) -> ParameterSchema:
        """
        Recursively convert a dict specification to ParameterSchema.

        Args:
            name: Parameter name
            spec: Dict specification with type, description, etc.

        Returns:
            ParameterSchema object with nested schemas properly converted
        """
        # Recursively convert nested properties
        properties = None
        if spec.get("properties"):
            properties = {}
            for prop_name, prop_spec in spec["properties"].items():
                if isinstance(prop_spec, dict):
                    properties[prop_name] = self._dict_to_param_schema(
                        prop_name, prop_spec
                    )
                elif isinstance(prop_spec, ParameterSchema):
                    properties[prop_name] = prop_spec

        # Recursively convert items schema
        items = None
        if spec.get("items"):
            items_spec = spec["items"]
            if isinstance(items_spec, dict):
                items = self._dict_to_param_schema("items", items_spec)
            elif isinstance(items_spec, ParameterSchema):
                items = items_spec

        return ParameterSchema(
            name=name,
            type=spec.get("type", "string"),
            description=spec.get("description", ""),
            required=spec.get("required", False),
            default=spec.get("default"),
            enum=spec.get("enum"),
            properties=properties,
            items=items,
        )

    def register(
        self,
        category: Union[ToolCategory, str],
        description: str,
        parameters: Optional[Dict[str, Union[Dict[str, Any], ParameterSchema]]] = None,
        tags: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        name: Optional[str] = None,
        allowed_agents: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
    ) -> Callable:
        """
        Decorator for registering tools.

        Args:
            category: Tool category (ToolCategory enum or string)
            description: Tool description
            parameters: Parameter schemas (dict or ParameterSchema objects)
            tags: Additional tags for discovery
            examples: Usage examples
            name: Override tool name (defaults to function name)
            allowed_agents: List of agent names that can use this tool (None = all agents)
            capabilities: List of capability tags for agent routing (FR-001 to FR-005)

        Returns:
            Decorator function

        Raises:
            ToolRegistrationError: If registration fails

        Example:
            @registry.register(
                category="filesystem",
                description="Read file contents",
                parameters={
                    "path": {"type": "string", "required": True, "description": "File path"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                capabilities=["file_read", "io"]
            )
            def fs_read(path: str, encoding: str = "utf-8") -> str:
                ...
        """

        def decorator(func: Callable) -> Callable:
            # Determine tool name
            tool_name = name or func.__name__

            # Check for duplicate registration
            if tool_name in self._tools:
                raise ToolRegistrationError(f"Tool '{tool_name}' is already registered")

            # Process parameter schemas
            param_schemas = {}
            if parameters:
                for param_name, param_spec in parameters.items():
                    if isinstance(param_spec, ParameterSchema):
                        param_schemas[param_name] = param_spec
                    elif isinstance(param_spec, dict):
                        # Convert dict to ParameterSchema (recursively handles nested schemas)
                        param_schemas[param_name] = self._dict_to_param_schema(
                            param_name, param_spec
                        )
                    else:
                        raise ToolRegistrationError(
                            f"Invalid parameter spec for '{param_name}': must be dict or ParameterSchema"
                        )

            # Create tool metadata
            metadata = ToolMetadata(
                name=tool_name,
                category=category,
                description=description,
                parameters=param_schemas,
                function=func,
                tags=set(tags or []),
                examples=examples or [],
                allowed_agents=allowed_agents,
                capabilities=capabilities or [],
            )

            # Register tool
            self._tools[tool_name] = metadata

            # Update category index
            # __post_init__ guarantees category is ToolCategory after construction
            category_key = cast(ToolCategory, metadata.category)
            self._category_index[category_key].add(tool_name)

            # Update tag index
            for tag in metadata.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(tool_name)

            # Update capability index (FR-001 to FR-005)
            for cap in metadata.capabilities:
                if cap not in self._capability_index:
                    self._capability_index[cap] = set()
                self._capability_index[cap].add(tool_name)

            if isinstance(metadata.category, ToolCategory):
                logger.debug(
                    f"Registered tool: {tool_name} (category: {metadata.category.value})"
                )
            else:
                logger.debug(
                    f"Registered tool: {tool_name} (category: {metadata.category})"
                )

            # Return wrapped function with metadata attached
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper._tool_metadata = metadata  # type: ignore[attr-defined]
            return wrapper

        return decorator

    def get(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Get tool metadata by name.

        Args:
            tool_name: Tool name

        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._tools.get(tool_name)

    def get_by_category(self, category: Union[ToolCategory, str]) -> List[ToolMetadata]:
        """
        Get all tools in a category.

        Args:
            category: Tool category (ToolCategory enum or string)

        Returns:
            List of ToolMetadata for tools in category
        """
        # Normalize category
        if isinstance(category, str):
            try:
                category = ToolCategory(category)
            except ValueError:
                category = ToolCategory.CUSTOM

        tool_names = self._category_index.get(category, set())
        return [self._tools[name] for name in tool_names]

    def get_by_tags(
        self, tags: List[str], match_all: bool = False
    ) -> List[ToolMetadata]:
        """
        Get tools by tags.

        Args:
            tags: List of tags to search for
            match_all: If True, tool must have all tags. If False, any tag matches.

        Returns:
            List of matching ToolMetadata
        """
        if not tags:
            return []

        matching_tools: set[str] = set()

        if match_all:
            # Tool must have all tags
            for tag in tags:
                tool_names = self._tag_index.get(tag, set())
                if not matching_tools:
                    matching_tools = tool_names.copy()
                else:
                    matching_tools &= tool_names
        else:
            # Tool must have at least one tag
            for tag in tags:
                tool_names = self._tag_index.get(tag, set())
                matching_tools |= tool_names

        return [self._tools[name] for name in matching_tools]

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """
        List all categories with registered tools.

        Returns:
            List of category names
        """
        return [cat.value for cat, tools in self._category_index.items() if tools]

    def list_tags(self) -> List[str]:
        """
        List all tags in use.

        Returns:
            List of tag names
        """
        return list(self._tag_index.keys())

    def list_capabilities(self) -> List[str]:
        """
        List all capabilities in use.

        Returns:
            List of capability names (FR-001 to FR-005)
        """
        return list(self._capability_index.keys())

    def discover_capabilities(
        self,
        agent_name: Optional[str] = None,
        capability_filter: Optional[List[str]] = None,
    ) -> List[ToolMetadata]:
        """
        Discover tools available to an agent, optionally filtered by capabilities.

        This is the core method for agent capability discovery (FR-001 to FR-005).
        Enables intelligent routing by allowing agents to discover what tools
        they have access to and what capabilities are available.

        Args:
            agent_name: Name of the agent to discover tools for. If None, returns
                       all tools (for system-level discovery)
            capability_filter: Optional list of capabilities to filter by.
                              If provided, only returns tools with at least one
                              of the specified capabilities

        Returns:
            List of ToolMetadata objects available to the agent with matching capabilities

        Example:
            # Discover all file-related tools for the "worker" agent
            tools = registry.discover_capabilities(
                agent_name="worker",
                capability_filter=["file_read", "file_write"]
            )
        """
        matching_tools = []

        for tool in self._tools.values():
            # Check agent access (None = all agents can access)
            if agent_name is not None and tool.allowed_agents is not None:
                if agent_name not in tool.allowed_agents:
                    continue

            # Check capability filter
            if capability_filter:
                # Tool must have at least one matching capability
                if not any(cap in tool.capabilities for cap in capability_filter):
                    continue

            matching_tools.append(tool)

        return matching_tools

    def get_by_capability(self, capability: str) -> List[ToolMetadata]:
        """
        Get all tools with a specific capability.

        Args:
            capability: Capability name to search for

        Returns:
            List of ToolMetadata for tools with the capability
        """
        tool_names = self._capability_index.get(capability, set())
        return [self._tools[name] for name in tool_names]

    def get_agent_tools(self, agent_name: str) -> List[ToolMetadata]:
        """
        Get all tools accessible to a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of ToolMetadata accessible to the agent
        """
        return [
            tool
            for tool in self._tools.values()
            if tool.allowed_agents is None or agent_name in tool.allowed_agents
        ]

    def get_openai_schemas(
        self, category: Optional[Union[ToolCategory, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get OpenAI function calling schemas for all tools or a specific category.

        Args:
            category: Optional category filter

        Returns:
            List of OpenAI function calling schema dictionaries
        """
        if category:
            tools = self.get_by_category(category)
        else:
            tools = list(self._tools.values())

        return [tool.to_openai_schema() for tool in tools]

    def execute(self, tool_name: str, **params) -> Any:
        """
        Execute a tool by name with parameter validation.

        Args:
            tool_name: Tool name
            **params: Tool parameters

        Returns:
            Tool execution result

        Raises:
            ToolNotFoundError: If tool not found
            ToolValidationError: If parameters are invalid
        """
        tool = self.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found in registry")

        return tool(**params)

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool.

        Args:
            tool_name: Tool name

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name not in self._tools:
            return False

        tool = self._tools[tool_name]

        # Remove from tools
        del self._tools[tool_name]

        # Remove from category index
        # __post_init__ guarantees category is ToolCategory after construction
        self._category_index[cast(ToolCategory, tool.category)].discard(tool_name)

        # Remove from tag index
        for tag in tool.tags:
            self._tag_index[tag].discard(tool_name)
            if not self._tag_index[tag]:
                del self._tag_index[tag]

        # Remove from capability index (FR-001 to FR-005)
        for cap in tool.capabilities:
            self._capability_index[cap].discard(tool_name)
            if not self._capability_index[cap]:
                del self._capability_index[cap]

        logger.debug(f"Unregistered tool: {tool_name}")
        return True

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._tag_index.clear()
        self._capability_index.clear()

        for category in ToolCategory:
            self._category_index[category] = set()

        logger.debug("Cleared all registered tools")

    def export_metadata(self) -> Dict[str, Any]:
        """
        Export registry metadata (excludes function objects).

        Returns:
            Dictionary with registry metadata
        """
        return {
            "tools": {
                name: {
                    "category": cast(ToolCategory, tool.category).value,
                    "description": tool.description,
                    "parameters": {
                        pname: {
                            "type": param.type,
                            "description": param.description,
                            "required": param.required,
                            "default": param.default,
                            "enum": param.enum,
                        }
                        for pname, param in tool.parameters.items()
                    },
                    "tags": list(tool.tags),
                    "examples": tool.examples,
                    "allowed_agents": tool.allowed_agents,
                    "capabilities": tool.capabilities,
                }
                for name, tool in self._tools.items()
            },
            "statistics": {
                "total_tools": len(self._tools),
                "categories": {
                    cat.value: len(tools)
                    for cat, tools in self._category_index.items()
                    if tools
                },
                "total_tags": len(self._tag_index),
                "total_capabilities": len(self._capability_index),
            },
        }
