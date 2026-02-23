"""
Mental Model Tools for Agent Self-Understanding.

Implements US7 Mental Models Integration tools.
Enables agents to query and update their mental models.
"""

import json
from typing import Any, Dict, List, Optional

from ..registry import tool_registry, ToolCategory


def get_my_capabilities(session_manager: Any) -> Dict[str, Any]:
    """
    Get the current agent's capabilities and specializations.

    Args:
        session_manager: Session manager with mental model state

    Returns:
        Dict with agent's mental model information
    """
    from ...models import MentalModel, MentalModelManager, create_default_model

    # Get current agent name
    agent_name = "default"
    if hasattr(session_manager, 'session') and session_manager.session:
        if hasattr(session_manager.session, 'current_agent'):
            agent_name = session_manager.session.current_agent or "default"

    # Get or create mental model manager
    if not hasattr(session_manager, '_mental_model_manager'):
        session_manager._mental_model_manager = MentalModelManager()

    model = session_manager._mental_model_manager.get_or_create(agent_name)

    # If no specializations, create default
    if not model.specializations:
        from ...models import SpecializationType
        model = create_default_model(agent_name, [SpecializationType.GENERAL])
        session_manager._mental_model_manager.update(model)

    return {
        "agent_name": model.agent_name,
        "specializations": [
            {
                "type": s.specialization.value,
                "proficiency": s.proficiency,
                "capabilities": s.related_capabilities,
                "experience": s.experience_count
            }
            for s in model.specializations
        ],
        "known_agents": model.known_agents,
        "fitness_areas": [s.specialization.value for s in model.specializations if s.proficiency >= 0.5]
    }


def discover_capable_agents(
    session_manager: Any,
    required_capabilities: List[str]
) -> Dict[str, Any]:
    """
    Discover agents capable of handling specific capabilities.

    Args:
        session_manager: Session manager with mental model state
        required_capabilities: List of required capability names

    Returns:
        Dict with capable agents and their fitness scores
    """
    from ...models import MentalModelManager

    if not hasattr(session_manager, '_mental_model_manager'):
        session_manager._mental_model_manager = MentalModelManager()

    manager = session_manager._mental_model_manager

    # Score all known agents
    agent_scores = []
    for agent_name in manager.list_agents():
        model = manager.get(agent_name)
        if model:
            score = model.calculate_task_fitness(required_capabilities)
            agent_scores.append({
                "agent": agent_name,
                "fitness_score": round(score, 3),
                "matching_capabilities": [
                    cap for cap in required_capabilities
                    if any(cap in s.related_capabilities for s in model.specializations)
                ]
            })

    # Sort by fitness score
    agent_scores.sort(key=lambda x: x["fitness_score"], reverse=True)

    best_agent = manager.find_best_agent(required_capabilities)

    return {
        "required_capabilities": required_capabilities,
        "best_agent": best_agent,
        "all_agents": agent_scores,
        "total_agents_checked": len(agent_scores)
    }


def update_specialization(
    session_manager: Any,
    specialization_type: str,
    proficiency: Optional[float] = None,
    capabilities: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Update an agent's specialization.

    Args:
        session_manager: Session manager
        specialization_type: Type of specialization to update
        proficiency: New proficiency level (0.0-1.0)
        capabilities: New capabilities list

    Returns:
        Dict with updated specialization info
    """
    from ...models import MentalModelManager, SpecializationType

    # Get current agent
    agent_name = "default"
    if hasattr(session_manager, 'session') and session_manager.session:
        if hasattr(session_manager.session, 'current_agent'):
            agent_name = session_manager.session.current_agent or "default"

    if not hasattr(session_manager, '_mental_model_manager'):
        session_manager._mental_model_manager = MentalModelManager()

    model = session_manager._mental_model_manager.get_or_create(agent_name)

    # Find or create specialization
    try:
        spec_type = SpecializationType(specialization_type)
    except ValueError:
        return {
            "success": False,
            "error": f"Unknown specialization type: {specialization_type}",
            "valid_types": [s.value for s in SpecializationType]
        }

    spec = model.get_specialization(spec_type)
    if spec:
        if proficiency is not None:
            spec.proficiency = max(0.0, min(1.0, proficiency))
        if capabilities is not None:
            spec.related_capabilities = capabilities
    else:
        spec = model.add_specialization(
            spec_type,
            proficiency=proficiency or 0.5,
            capabilities=capabilities or []
        )

    session_manager._mental_model_manager.update(model)

    return {
        "success": True,
        "agent": agent_name,
        "specialization": spec.to_dict()
    }


def record_task_experience(
    session_manager: Any,
    task_type: str,
    success: bool,
    capabilities_used: List[str]
) -> Dict[str, Any]:
    """
    Record task completion for learning.

    Args:
        session_manager: Session manager
        task_type: Type of task completed
        success: Whether task was successful
        capabilities_used: Capabilities that were used

    Returns:
        Dict with updated learning state
    """
    from ...models import MentalModelManager

    agent_name = "default"
    if hasattr(session_manager, 'session') and session_manager.session:
        if hasattr(session_manager.session, 'current_agent'):
            agent_name = session_manager.session.current_agent or "default"

    if not hasattr(session_manager, '_mental_model_manager'):
        session_manager._mental_model_manager = MentalModelManager()

    model = session_manager._mental_model_manager.get_or_create(agent_name)
    model.record_task_outcome(task_type, success, capabilities_used)

    # Update specialization experience
    for spec in model.specializations:
        if any(cap in spec.related_capabilities for cap in capabilities_used):
            spec.record_experience(success)

    session_manager._mental_model_manager.update(model)

    return {
        "success": True,
        "agent": agent_name,
        "task_recorded": {
            "type": task_type,
            "success": success,
            "capabilities": capabilities_used
        },
        "tasks_in_history": len(model.task_history)
    }


def share_capabilities(
    session_manager: Any,
    target_agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Share this agent's capabilities with other agents.

    Args:
        session_manager: Session manager
        target_agent: Specific agent to share with (None = broadcast to all)

    Returns:
        Dict with sharing results
    """
    from ...models import MentalModelManager

    agent_name = "default"
    if hasattr(session_manager, 'session') and session_manager.session:
        if hasattr(session_manager.session, 'current_agent'):
            agent_name = session_manager.session.current_agent or "default"

    if not hasattr(session_manager, '_mental_model_manager'):
        session_manager._mental_model_manager = MentalModelManager()

    model = session_manager._mental_model_manager.get_or_create(agent_name)

    # Collect all capabilities
    all_capabilities = []
    for spec in model.specializations:
        all_capabilities.extend(spec.related_capabilities)
    all_capabilities = list(set(all_capabilities))  # Dedupe

    if target_agent:
        target_model = session_manager._mental_model_manager.get_or_create(target_agent)
        target_model.learn_about_agent(agent_name, all_capabilities)
        session_manager._mental_model_manager.update(target_model)
        shared_with = [target_agent]
    else:
        session_manager._mental_model_manager.broadcast_agent_discovery(agent_name, all_capabilities)
        shared_with = [a for a in session_manager._mental_model_manager.list_agents() if a != agent_name]

    return {
        "success": True,
        "agent": agent_name,
        "capabilities_shared": all_capabilities,
        "shared_with": shared_with
    }


# Register tools with the registry
@tool_registry.register(
    category=ToolCategory.COLLABORATION,
    description="Get the current agent's capabilities and specializations",
    parameters={},
    tags=["mental-model", "capabilities", "self-awareness"],
    capabilities=["mental_model", "introspection"]
)
def get_my_capabilities_tool() -> str:
    """Get my capabilities - wrapper for session injection."""
    from ...session_manager import SessionManager
    session_manager = SessionManager.get_instance()
    result = get_my_capabilities(session_manager)
    return json.dumps(result, indent=2)


@tool_registry.register(
    category=ToolCategory.COLLABORATION,
    description="Discover agents capable of handling specific capabilities",
    parameters={
        "required_capabilities": {"type": "array", "items": {"type": "string"}, "description": "List of required capability names", "required": True}
    },
    tags=["mental-model", "discovery", "routing"],
    capabilities=["mental_model", "agent_discovery"]
)
def discover_capable_agents_tool(required_capabilities: List[str]) -> str:
    """Discover capable agents - wrapper for session injection."""
    from ...session_manager import SessionManager
    session_manager = SessionManager.get_instance()
    result = discover_capable_agents(session_manager, required_capabilities)
    return json.dumps(result, indent=2)


@tool_registry.register(
    category=ToolCategory.COLLABORATION,
    description="Update an agent's specialization",
    parameters={
        "specialization_type": {"type": "string", "description": "Type of specialization", "required": True},
        "proficiency": {"type": "number", "description": "Proficiency level 0.0-1.0", "required": False},
        "capabilities": {"type": "array", "items": {"type": "string"}, "description": "List of capabilities", "required": False}
    },
    tags=["mental-model", "learning", "specialization"],
    capabilities=["mental_model", "self_improvement"]
)
def update_specialization_tool(
    specialization_type: str,
    proficiency: Optional[float] = None,
    capabilities: Optional[List[str]] = None
) -> str:
    """Update specialization - wrapper for session injection."""
    from ...session_manager import SessionManager
    session_manager = SessionManager.get_instance()
    result = update_specialization(session_manager, specialization_type, proficiency, capabilities)
    return json.dumps(result, indent=2)


@tool_registry.register(
    category=ToolCategory.COLLABORATION,
    description="Record task completion for learning",
    parameters={
        "task_type": {"type": "string", "description": "Type of task completed", "required": True},
        "success": {"type": "boolean", "description": "Whether task was successful", "required": True},
        "capabilities_used": {"type": "array", "items": {"type": "string"}, "description": "Capabilities used", "required": True}
    },
    tags=["mental-model", "learning", "experience"],
    capabilities=["mental_model", "learning"]
)
def record_task_experience_tool(
    task_type: str,
    success: bool,
    capabilities_used: List[str]
) -> str:
    """Record experience - wrapper for session injection."""
    from ...session_manager import SessionManager
    session_manager = SessionManager.get_instance()
    result = record_task_experience(session_manager, task_type, success, capabilities_used)
    return json.dumps(result, indent=2)


@tool_registry.register(
    category=ToolCategory.COLLABORATION,
    description="Share this agent's capabilities with other agents",
    parameters={
        "target_agent": {"type": "string", "description": "Specific agent to share with (None = broadcast)", "required": False}
    },
    tags=["mental-model", "sharing", "collaboration"],
    capabilities=["mental_model", "broadcasting"]
)
def share_capabilities_tool(target_agent: Optional[str] = None) -> str:
    """Share capabilities - wrapper for session injection."""
    from ...session_manager import SessionManager
    session_manager = SessionManager.get_instance()
    result = share_capabilities(session_manager, target_agent)
    return json.dumps(result, indent=2)
