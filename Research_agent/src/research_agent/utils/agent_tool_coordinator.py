"""
Agent Tool Coordinator

Provides a centralized system for coordinating tool registration across all research agents.
Ensures no conflicts while maintaining clean agent interfaces.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass
from datetime import datetime

from .namespace_tool_registry import NamespaceToolRegistry, get_global_tool_registry
from ..agents.synthesis_agent import SynthesisAgent
from ..integrations.multi_query_coordinator import MultiQueryCoordinator

logger = logging.getLogger(__name__)


@dataclass
class AgentToolConfig:
    """Configuration for agent tool registration"""
    agent_class: str
    namespace: str
    agent_id: str
    tool_count: int
    registration_time: datetime
    config_hash: str


class AgentToolCoordinator:
    """
    Coordinates tool registration across all research agents to prevent conflicts
    """
    
    def __init__(self, tool_registry: Optional[NamespaceToolRegistry] = None):
        self.tool_registry = tool_registry or get_global_tool_registry()
        self.registered_agents: Dict[str, AgentToolConfig] = {}
        self.agent_instances: Dict[str, Any] = {}
        
        logger.info("AgentToolCoordinator initialized")
    
    def register_synthesis_agent(
        self,
        config: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> SynthesisAgent:
        """
        Register and initialize a SynthesisAgent with namespaced tools
        
        Args:
            config: Agent configuration
            agent_id: Optional unique identifier for the agent
            
        Returns:
            Initialized SynthesisAgent with conflict-free tools
        """
        agent_id = agent_id or f"synthesis_{len(self.registered_agents)}"
        config_with_id = {**config, 'agent_id': agent_id}
        
        # Create agent with namespace tool support
        agent = SynthesisAgent(config_with_id, self.tool_registry)
        
        # Record registration
        agent_config = AgentToolConfig(
            agent_class="SynthesisAgent",
            namespace=agent.namespace,
            agent_id=agent_id,
            tool_count=len(agent._registered_tools),
            registration_time=datetime.now(),
            config_hash=self._hash_config(config)
        )
        
        self.registered_agents[agent_id] = agent_config
        self.agent_instances[agent_id] = agent
        
        logger.info(f"Registered SynthesisAgent '{agent_id}' with namespace '{agent.namespace}' ({agent_config.tool_count} tools)")
        
        return agent
    
    def register_multiquery_coordinator(
        self,
        config: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> MultiQueryCoordinator:
        """
        Register and initialize a MultiQueryCoordinator with namespaced tools
        
        Args:
            config: Agent configuration
            agent_id: Optional unique identifier for the agent
            
        Returns:
            Initialized MultiQueryCoordinator with conflict-free tools
        """
        agent_id = agent_id or f"multiquery_{len(self.registered_agents)}"
        
        # Convert ResearchConfig to dict if necessary
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config if isinstance(config, dict) else {}
            
        config_with_id = {**config_dict, 'agent_id': agent_id}
        
        # Create agent with namespace tool support
        agent = MultiQueryCoordinator(config_with_id, self.tool_registry)
        
        # Record registration
        agent_config = AgentToolConfig(
            agent_class="MultiQueryCoordinator",
            namespace=agent.namespace,
            agent_id=agent_id,
            tool_count=len(agent._registered_tools),
            registration_time=datetime.now(),
            config_hash=self._hash_config(config)
        )
        
        self.registered_agents[agent_id] = agent_config
        self.agent_instances[agent_id] = agent
        
        logger.info(f"Registered MultiQueryCoordinator '{agent_id}' with namespace '{agent.namespace}' ({agent_config.tool_count} tools)")
        
        return agent
    
    def register_agent_by_type(
        self,
        agent_type: str,
        config: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> Any:
        """
        Register an agent by type name
        
        Args:
            agent_type: Type of agent ('synthesis', 'multiquery', etc.)
            config: Agent configuration
            agent_id: Optional unique identifier
            
        Returns:
            Initialized agent instance
        """
        agent_type = agent_type.lower()
        
        if agent_type in ['synthesis', 'synthesis_agent']:
            return self.register_synthesis_agent(config, agent_id)
        elif agent_type in ['multiquery', 'multi_query_coordinator']:
            return self.register_multiquery_coordinator(config, agent_id)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Get a registered agent by ID"""
        return self.agent_instances.get(agent_id)
    
    def list_registered_agents(self) -> List[AgentToolConfig]:
        """Get list of all registered agents"""
        return list(self.registered_agents.values())
    
    def get_namespace_distribution(self) -> Dict[str, List[str]]:
        """Get distribution of agents across namespaces"""
        namespace_dist = {}
        for agent_id, config in self.registered_agents.items():
            namespace = config.namespace
            if namespace not in namespace_dist:
                namespace_dist[namespace] = []
            namespace_dist[namespace].append(agent_id)
        
        return namespace_dist
    
    def validate_no_conflicts(self) -> Dict[str, Any]:
        """
        Validate that no tool conflicts exist across all registered agents
        
        Returns:
            Validation report
        """
        # Validate registry
        registry_valid = self.tool_registry.validate_no_conflicts()
        
        # Get registry stats
        stats = self.tool_registry.get_registry_stats()
        
        # Check for namespace conflicts
        namespace_dist = self.get_namespace_distribution()
        namespace_conflicts = {
            ns: agents for ns, agents in namespace_dist.items() 
            if len(agents) > 1
        }
        
        validation_report = {
            'registry_valid': registry_valid,
            'total_agents': len(self.registered_agents),
            'total_tools': stats['total_tools'],
            'conflicts_prevented': stats['conflicts_prevented'],
            'namespace_conflicts': namespace_conflicts,
            'namespace_distribution': namespace_dist,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Log results
        if registry_valid and not namespace_conflicts:
            logger.info(f"✅ Tool validation passed: {stats['total_tools']} tools across {len(self.registered_agents)} agents")
        else:
            logger.warning(f"⚠️ Tool validation issues: registry_valid={registry_valid}, namespace_conflicts={bool(namespace_conflicts)}")
        
        return validation_report
    
    def cleanup_agent(self, agent_id: str) -> bool:
        """
        Cleanup a specific agent and its tools
        
        Args:
            agent_id: Agent to cleanup
            
        Returns:
            Success status
        """
        if agent_id not in self.registered_agents:
            logger.warning(f"Agent '{agent_id}' not found for cleanup")
            return False
        
        try:
            # Get agent config
            agent_config = self.registered_agents[agent_id]
            
            # Clear tools from registry
            tools_cleared = self.tool_registry.clear_namespace(agent_config.namespace)
            
            # Remove agent references
            del self.registered_agents[agent_id]
            if agent_id in self.agent_instances:
                del self.agent_instances[agent_id]
            
            logger.info(f"Cleaned up agent '{agent_id}': removed {tools_cleared} tools from namespace '{agent_config.namespace}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup agent '{agent_id}': {e}")
            return False
    
    def cleanup_all_agents(self) -> int:
        """
        Cleanup all registered agents
        
        Returns:
            Number of agents cleaned up
        """
        agent_ids = list(self.registered_agents.keys())
        cleaned_count = 0
        
        for agent_id in agent_ids:
            if self.cleanup_agent(agent_id):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} agents")
        return cleaned_count
    
    def get_coordination_report(self) -> Dict[str, Any]:
        """Get comprehensive coordination report"""
        validation = self.validate_no_conflicts()
        registry_stats = self.tool_registry.get_registry_stats()
        
        return {
            'coordinator_summary': {
                'total_agents_registered': len(self.registered_agents),
                'agent_types': list(set(config.agent_class for config in self.registered_agents.values())),
                'coordination_status': 'healthy' if validation['registry_valid'] else 'issues_detected'
            },
            'tool_registry_stats': registry_stats,
            'validation_results': validation,
            'agent_details': [
                {
                    'agent_id': agent_id,
                    'agent_class': config.agent_class,
                    'namespace': config.namespace,
                    'tool_count': config.tool_count,
                    'registration_time': config.registration_time.isoformat()
                }
                for agent_id, config in self.registered_agents.items()
            ]
        }
    
    def _hash_config(self, config: Any) -> str:
        """Generate hash of configuration for tracking"""
        import hashlib
        import json
        
        # Convert ResearchConfig to dict if necessary
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config if isinstance(config, dict) else {}
        
        # Remove non-hashable items
        clean_config = {k: v for k, v in config_dict.items() if isinstance(v, (str, int, float, bool, list, dict))}
        config_str = json.dumps(clean_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


# Global coordinator instance
_global_coordinator = AgentToolCoordinator()


def get_global_coordinator() -> AgentToolCoordinator:
    """Get the global agent tool coordinator"""
    return _global_coordinator


def register_research_agents(**agents_config) -> Dict[str, Any]:
    """
    Convenience function to register multiple research agents at once
    
    Args:
        **agents_config: Agent configurations by type
        
    Returns:
        Dictionary of registered agent instances
    """
    coordinator = get_global_coordinator()
    registered_agents = {}
    
    for agent_type, config in agents_config.items():
        try:
            agent = coordinator.register_agent_by_type(agent_type, config)
            registered_agents[agent_type] = agent
            logger.info(f"Successfully registered {agent_type} agent")
        except Exception as e:
            logger.error(f"Failed to register {agent_type} agent: {e}")
    
    # Validate all registrations
    validation = coordinator.validate_no_conflicts()
    if not validation['registry_valid']:
        logger.warning("Some tool conflicts detected during batch registration")
    
    return registered_agents


def reset_global_coordination() -> None:
    """Reset the global coordination system (useful for testing)"""
    global _global_coordinator
    _global_coordinator.cleanup_all_agents()
    _global_coordinator = AgentToolCoordinator()
    logger.info("Global agent coordination reset")