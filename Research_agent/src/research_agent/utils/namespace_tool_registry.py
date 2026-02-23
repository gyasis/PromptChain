"""
Namespace-Aware Tool Registry

Provides conflict-free tool registration with automatic namespacing and validation.
Prevents tool name conflicts between agents while maintaining clean architecture.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import inspect

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Information about a registered tool"""
    name: str
    original_name: str
    namespace: str
    function: Callable
    schema: Dict[str, Any]
    agent_id: str
    registration_time: datetime
    usage_count: int = 0
    
    def get_namespaced_name(self) -> str:
        """Get the full namespaced name"""
        return f"{self.namespace}_{self.original_name}"


@dataclass
class NamespaceRegistry:
    """Registry for managing namespaced tools"""
    tools: Dict[str, ToolInfo] = field(default_factory=dict)
    namespaces: Dict[str, Set[str]] = field(default_factory=dict)
    conflicts_prevented: int = 0
    total_registrations: int = 0
    
    def register_namespace(self, namespace: str) -> None:
        """Register a new namespace"""
        if namespace not in self.namespaces:
            self.namespaces[namespace] = set()
            logger.debug(f"Registered namespace: {namespace}")


class NamespaceToolRegistry:
    """
    Namespace-aware tool registration system that prevents conflicts
    while maintaining functionality and clean API.
    """
    
    def __init__(self, enable_validation: bool = True, log_conflicts: bool = True):
        self.registry = NamespaceRegistry()
        self.enable_validation = enable_validation
        self.log_conflicts = log_conflicts
        self._function_registry: Dict[str, Callable] = {}
        self._schema_registry: Dict[str, Dict[str, Any]] = {}
        
        logger.info("NamespaceToolRegistry initialized")
    
    def generate_namespace(self, agent_class_name: str, agent_id: Optional[str] = None) -> str:
        """
        Generate consistent namespace for an agent
        
        Args:
            agent_class_name: Name of the agent class
            agent_id: Optional unique ID for the agent instance
            
        Returns:
            Namespace string (e.g., 'synthesis', 'multiquery', 'querygen')
        """
        # Convert class name to clean namespace
        namespace = agent_class_name.lower()
        
        # Handle common agent naming patterns
        if namespace.endswith('agent'):
            namespace = namespace[:-5]  # Remove 'agent' suffix
        elif namespace.endswith('coordinator'):
            namespace = namespace[:-11]  # Remove 'coordinator' suffix
        elif namespace.endswith('processor'):
            namespace = namespace[:-9]  # Remove 'processor' suffix
        elif namespace.endswith('generator'):
            namespace = namespace[:-9]  # Remove 'generator' suffix
            namespace = namespace + 'gen'  # Shorten to 'gen'
        
        # Clean up common patterns
        namespace = namespace.replace('_', '').replace('-', '')
        
        # Add agent ID if provided for uniqueness
        if agent_id:
            namespace = f"{namespace}_{agent_id[:6]}"  # Limit ID to 6 chars
            
        self.registry.register_namespace(namespace)
        return namespace
    
    def register_tool(
        self,
        tool_function: Callable,
        tool_schema: Dict[str, Any],
        namespace: str,
        agent_id: str,
        original_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Register a tool with namespace protection
        
        Args:
            tool_function: The function to register
            tool_schema: OpenAI function schema
            namespace: Namespace for the tool
            agent_id: ID of the agent registering the tool
            original_name: Original tool name (extracted from schema if None)
            
        Returns:
            Tuple of (namespaced_name, original_name)
        """
        self.registry.total_registrations += 1
        
        # Extract original name from schema or function
        if original_name is None:
            original_name = tool_schema.get('function', {}).get('name', tool_function.__name__)
        
        # Generate namespaced name
        namespaced_name = f"{namespace}_{original_name}"
        
        # Check for conflicts
        if namespaced_name in self.registry.tools:
            existing_tool = self.registry.tools[namespaced_name]
            if existing_tool.agent_id != agent_id:
                self.registry.conflicts_prevented += 1
                if self.log_conflicts:
                    logger.warning(
                        f"Tool conflict prevented: {namespaced_name} "
                        f"(agents: {existing_tool.agent_id} vs {agent_id})"
                    )
                # Generate unique name with hash
                conflict_hash = hashlib.md5(f"{agent_id}_{original_name}".encode()).hexdigest()[:6]
                namespaced_name = f"{namespace}_{original_name}_{conflict_hash}"
        
        # Create updated schema with namespaced name
        namespaced_schema = self._create_namespaced_schema(tool_schema, namespaced_name)
        
        # Register tool
        tool_info = ToolInfo(
            name=namespaced_name,
            original_name=original_name,
            namespace=namespace,
            function=tool_function,
            schema=namespaced_schema,
            agent_id=agent_id,
            registration_time=datetime.now()
        )
        
        self.registry.tools[namespaced_name] = tool_info
        self.registry.namespaces[namespace].add(namespaced_name)
        
        # Store in registries for easy access
        self._function_registry[namespaced_name] = tool_function
        self._schema_registry[namespaced_name] = namespaced_schema
        
        logger.debug(f"Registered tool: {namespaced_name} (from {original_name}) for agent {agent_id}")
        
        return namespaced_name, original_name
    
    def _create_namespaced_schema(self, original_schema: Dict[str, Any], namespaced_name: str) -> Dict[str, Any]:
        """Create namespaced version of tool schema"""
        schema = original_schema.copy()
        
        # Update the function name in schema
        if 'function' in schema:
            schema['function'] = schema['function'].copy()
            schema['function']['name'] = namespaced_name
            
            # Update description to include namespace info
            original_desc = schema['function'].get('description', '')
            schema['function']['description'] = f"{original_desc} [namespace: {namespaced_name.split('_')[0]}]"
        
        return schema
    
    def register_multiple_tools(
        self,
        tools_data: List[Tuple[Callable, Dict[str, Any]]],
        namespace: str,
        agent_id: str
    ) -> Dict[str, str]:
        """
        Register multiple tools at once
        
        Args:
            tools_data: List of (function, schema) tuples
            namespace: Namespace for all tools
            agent_id: ID of the agent registering tools
            
        Returns:
            Mapping of original_name -> namespaced_name
        """
        name_mapping = {}
        
        for tool_function, tool_schema in tools_data:
            namespaced_name, original_name = self.register_tool(
                tool_function, tool_schema, namespace, agent_id
            )
            name_mapping[original_name] = namespaced_name
        
        logger.info(f"Registered {len(tools_data)} tools for agent {agent_id} in namespace '{namespace}'")
        return name_mapping
    
    def get_tools_for_chain(self, namespace: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
        """
        Get tools ready for PromptChain registration
        
        Args:
            namespace: Optional namespace to filter tools
            
        Returns:
            Tuple of (schemas_list, functions_dict)
        """
        if namespace:
            # Get tools for specific namespace
            tool_names = self.registry.namespaces.get(namespace, set())
            tools = [self.registry.tools[name] for name in tool_names]
        else:
            # Get all tools
            tools = list(self.registry.tools.values())
        
        schemas = [tool.schema for tool in tools]
        functions = {tool.name: tool.function for tool in tools}
        
        return schemas, functions
    
    def get_namespace_tools(self, namespace: str) -> List[ToolInfo]:
        """Get all tools for a specific namespace"""
        tool_names = self.registry.namespaces.get(namespace, set())
        return [self.registry.tools[name] for name in tool_names]
    
    def validate_no_conflicts(self) -> bool:
        """Validate that no tool name conflicts exist"""
        tool_names = list(self.registry.tools.keys())
        unique_names = set(tool_names)
        
        has_conflicts = len(tool_names) != len(unique_names)
        
        if has_conflicts:
            logger.error(f"Tool name conflicts detected: {len(tool_names)} total, {len(unique_names)} unique")
            return False
        
        logger.info(f"Tool validation passed: {len(unique_names)} unique tools registered")
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        namespace_stats = {}
        for namespace, tools in self.registry.namespaces.items():
            namespace_stats[namespace] = len(tools)
        
        return {
            'total_tools': len(self.registry.tools),
            'total_namespaces': len(self.registry.namespaces),
            'conflicts_prevented': self.registry.conflicts_prevented,
            'total_registrations': self.registry.total_registrations,
            'namespace_distribution': namespace_stats,
            'validation_enabled': self.enable_validation
        }
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all tools from a specific namespace"""
        if namespace not in self.registry.namespaces:
            return 0
        
        tool_names = list(self.registry.namespaces[namespace])
        count = len(tool_names)
        
        # Remove tools
        for tool_name in tool_names:
            del self.registry.tools[tool_name]
            if tool_name in self._function_registry:
                del self._function_registry[tool_name]
            if tool_name in self._schema_registry:
                del self._schema_registry[tool_name]
        
        # Clear namespace
        self.registry.namespaces[namespace].clear()
        
        logger.info(f"Cleared {count} tools from namespace '{namespace}'")
        return count
    
    def reset_registry(self) -> None:
        """Reset the entire registry"""
        self.registry = NamespaceRegistry()
        self._function_registry.clear()
        self._schema_registry.clear()
        logger.info("Tool registry reset")
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry data for debugging"""
        export_data = {
            'stats': self.get_registry_stats(),
            'tools': {},
            'namespaces': dict(self.registry.namespaces)
        }
        
        # Export tool info (without functions for serializability)
        for name, tool_info in self.registry.tools.items():
            export_data['tools'][name] = {
                'original_name': tool_info.original_name,
                'namespace': tool_info.namespace,
                'agent_id': tool_info.agent_id,
                'registration_time': tool_info.registration_time.isoformat(),
                'usage_count': tool_info.usage_count,
                'schema': tool_info.schema
            }
        
        return export_data


class NamespaceToolMixin:
    """
    Mixin class to add namespace-aware tool registration to agents
    """
    
    def __init_namespace_tools__(
        self,
        tool_registry: NamespaceToolRegistry,
        agent_id: Optional[str] = None
    ):
        """Initialize namespace tools for an agent"""
        self.tool_registry = tool_registry
        self.agent_id = agent_id or f"{self.__class__.__name__.lower()}_{id(self)}"
        self.namespace = tool_registry.generate_namespace(
            self.__class__.__name__,
            agent_id
        )
        self._registered_tools: Dict[str, str] = {}
        
        logger.debug(f"Initialized namespace tools for {self.agent_id} with namespace '{self.namespace}'")
    
    def register_namespaced_tool(
        self,
        tool_function: Callable,
        tool_schema: Dict[str, Any]
    ) -> str:
        """Register a single tool with namespace protection"""
        namespaced_name, original_name = self.tool_registry.register_tool(
            tool_function, tool_schema, self.namespace, self.agent_id
        )
        self._registered_tools[original_name] = namespaced_name
        return namespaced_name
    
    def register_namespaced_tools(
        self,
        tools_data: List[Tuple[Callable, Dict[str, Any]]]
    ) -> Dict[str, str]:
        """Register multiple tools with namespace protection"""
        name_mapping = self.tool_registry.register_multiple_tools(
            tools_data, self.namespace, self.agent_id
        )
        self._registered_tools.update(name_mapping)
        return name_mapping
    
    def get_my_tools(self) -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
        """Get tools registered by this agent"""
        return self.tool_registry.get_tools_for_chain(self.namespace)
    
    def apply_tools_to_chain(self, chain) -> None:
        """Apply this agent's tools to a PromptChain"""
        schemas, functions = self.get_my_tools()
        
        # Register functions
        for name, func in functions.items():
            chain.register_tool_function(func)
        
        # Add schemas
        if schemas:
            chain.add_tools(schemas)
        
        logger.info(f"Applied {len(schemas)} tools from namespace '{self.namespace}' to PromptChain")


# Global registry instance for shared use
_global_tool_registry = NamespaceToolRegistry()


def get_global_tool_registry() -> NamespaceToolRegistry:
    """Get the global tool registry instance"""
    return _global_tool_registry


def reset_global_registry() -> None:
    """Reset the global registry (useful for testing)"""
    global _global_tool_registry
    _global_tool_registry.reset_registry()
    logger.info("Global tool registry reset")