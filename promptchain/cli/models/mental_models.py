"""
Mental Models for Agent Capability Understanding.

Implements US7 Mental Models Integration (T076-T120) - CRITICAL
Enables agents to understand and reason about their own and other agents' capabilities.

Mental models define:
- Agent specializations (what an agent is good at)
- Capability requirements (what tools/capabilities a task needs)
- Agent compatibility scoring (matching tasks to agents)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import json
import uuid


class SpecializationType(str, Enum):
    """Types of agent specializations."""
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    FILE_OPERATIONS = "file_operations"
    SEARCH_DISCOVERY = "search_discovery"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    DATA_PROCESSING = "data_processing"
    API_INTEGRATION = "api_integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UI_UX = "ui_ux"
    DEVOPS = "devops"
    GENERAL = "general"


@dataclass
class AgentSpecialization:
    """
    Defines an agent's area of expertise.

    Attributes:
        specialization: Type of specialization
        proficiency: Skill level 0.0-1.0
        related_capabilities: Tool capabilities this specialization can use
        experience_count: Number of tasks completed in this specialization
    """
    specialization: SpecializationType
    proficiency: float = 0.5  # 0.0-1.0 scale
    related_capabilities: List[str] = field(default_factory=list)
    experience_count: int = 0

    def __post_init__(self):
        if not 0.0 <= self.proficiency <= 1.0:
            raise ValueError(f"Proficiency must be 0.0-1.0, got {self.proficiency}")

    def record_experience(self, success: bool = True):
        """Record task completion, adjusting proficiency."""
        self.experience_count += 1
        if success:
            # Increase proficiency slightly with diminishing returns
            self.proficiency = min(1.0, self.proficiency + (1.0 - self.proficiency) * 0.1)
        else:
            # Decrease slightly on failure
            self.proficiency = max(0.0, self.proficiency - 0.05)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specialization": self.specialization.value,
            "proficiency": self.proficiency,
            "related_capabilities": self.related_capabilities,
            "experience_count": self.experience_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpecialization":
        return cls(
            specialization=SpecializationType(data["specialization"]),
            proficiency=data.get("proficiency", 0.5),
            related_capabilities=data.get("related_capabilities", []),
            experience_count=data.get("experience_count", 0)
        )


@dataclass
class MentalModel:
    """
    Agent's mental model - understanding of capabilities and specializations.

    Attributes:
        agent_name: Name of the agent this model belongs to
        specializations: List of agent's specializations
        known_agents: Dict of other agents and their understood capabilities
        task_history: Recent task outcomes for learning
        created_at: Model creation timestamp
        updated_at: Last update timestamp
    """
    agent_name: str
    specializations: List[AgentSpecialization] = field(default_factory=list)
    known_agents: Dict[str, List[str]] = field(default_factory=dict)  # agent -> capabilities
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def add_specialization(
        self,
        spec_type: SpecializationType,
        proficiency: float = 0.5,
        capabilities: Optional[List[str]] = None
    ) -> AgentSpecialization:
        """Add a new specialization to this agent."""
        spec = AgentSpecialization(
            specialization=spec_type,
            proficiency=proficiency,
            related_capabilities=capabilities or []
        )
        self.specializations.append(spec)
        self.updated_at = datetime.now().timestamp()
        return spec

    def get_specialization(self, spec_type: SpecializationType) -> Optional[AgentSpecialization]:
        """Get specialization by type."""
        for spec in self.specializations:
            if spec.specialization == spec_type:
                return spec
        return None

    def learn_about_agent(self, agent_name: str, capabilities: List[str]):
        """Learn about another agent's capabilities."""
        self.known_agents[agent_name] = capabilities
        self.updated_at = datetime.now().timestamp()

    def record_task_outcome(
        self,
        task_type: str,
        success: bool,
        capabilities_used: List[str]
    ):
        """Record a task outcome for learning."""
        self.task_history.append({
            "task_type": task_type,
            "success": success,
            "capabilities_used": capabilities_used,
            "timestamp": datetime.now().timestamp()
        })
        # Keep last 100 tasks
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
        self.updated_at = datetime.now().timestamp()

    def calculate_task_fitness(self, required_capabilities: List[str]) -> float:
        """
        Calculate how well this agent fits a task based on capabilities.

        Returns fitness score 0.0-1.0
        """
        if not required_capabilities:
            return 0.5  # Default for unspecified tasks

        # Get all capabilities from specializations
        my_capabilities: Set[str] = set()
        for spec in self.specializations:
            my_capabilities.update(spec.related_capabilities)

        # Calculate overlap
        matched = len(set(required_capabilities) & my_capabilities)
        total = len(required_capabilities)

        return matched / total if total > 0 else 0.5

    def suggest_agent_for_task(self, required_capabilities: List[str]) -> Optional[str]:
        """Suggest the best known agent for a task."""
        best_agent = None
        best_score = 0.0

        for agent_name, agent_caps in self.known_agents.items():
            matched = len(set(required_capabilities) & set(agent_caps))
            score = matched / len(required_capabilities) if required_capabilities else 0
            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent if best_score > 0.3 else None  # Threshold for suggestion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "specializations": [s.to_dict() for s in self.specializations],
            "known_agents": self.known_agents,
            "task_history": self.task_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MentalModel":
        return cls(
            agent_name=data["agent_name"],
            specializations=[AgentSpecialization.from_dict(s) for s in data.get("specializations", [])],
            known_agents=data.get("known_agents", {}),
            task_history=data.get("task_history", []),
            created_at=data.get("created_at", datetime.now().timestamp()),
            updated_at=data.get("updated_at", datetime.now().timestamp())
        )

    def __repr__(self) -> str:
        specs = ", ".join(s.specialization.value for s in self.specializations)
        return f"MentalModel(agent={self.agent_name}, specs=[{specs}], known_agents={len(self.known_agents)})"


class MentalModelManager:
    """
    Manages mental models for all agents in a session.

    Provides centralized access to agent mental models with
    caching and persistence support.
    """

    def __init__(self):
        self._models: Dict[str, MentalModel] = {}

    def get_or_create(self, agent_name: str) -> MentalModel:
        """Get existing or create new mental model for agent."""
        if agent_name not in self._models:
            self._models[agent_name] = MentalModel(agent_name=agent_name)
        return self._models[agent_name]

    def get(self, agent_name: str) -> Optional[MentalModel]:
        """Get mental model if exists."""
        return self._models.get(agent_name)

    def update(self, model: MentalModel):
        """Update/store a mental model."""
        self._models[model.agent_name] = model

    def find_best_agent(self, required_capabilities: List[str]) -> Optional[str]:
        """Find the best agent across all models for given capabilities."""
        best_agent = None
        best_score = 0.0

        for agent_name, model in self._models.items():
            score = model.calculate_task_fitness(required_capabilities)
            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent if best_score > 0.3 else None

    def broadcast_agent_discovery(self, agent_name: str, capabilities: List[str]):
        """Broadcast an agent's capabilities to all other agents' models."""
        for model in self._models.values():
            if model.agent_name != agent_name:
                model.learn_about_agent(agent_name, capabilities)

    def list_agents(self) -> List[str]:
        """List all agents with mental models."""
        return list(self._models.keys())

    def export_all(self) -> Dict[str, Any]:
        """Export all models for persistence."""
        return {name: model.to_dict() for name, model in self._models.items()}

    def import_all(self, data: Dict[str, Any]):
        """Import models from persistence."""
        for name, model_data in data.items():
            self._models[name] = MentalModel.from_dict(model_data)

    def clear(self):
        """Clear all models."""
        self._models.clear()


# Default specialization mappings
DEFAULT_SPECIALIZATION_CAPABILITIES = {
    SpecializationType.CODE_ANALYSIS: ["code_search", "ripgrep_search", "file_read"],
    SpecializationType.CODE_GENERATION: ["file_write", "file_edit", "code_generation"],
    SpecializationType.FILE_OPERATIONS: ["file_read", "file_write", "file_edit", "file_delete", "directory_list", "directory_create"],
    SpecializationType.SEARCH_DISCOVERY: ["ripgrep_search", "file_search", "code_search"],
    SpecializationType.TESTING: ["terminal_execute", "test_runner"],
    SpecializationType.DOCUMENTATION: ["file_read", "file_write", "documentation"],
    SpecializationType.DEBUGGING: ["code_search", "file_read", "terminal_execute", "debugging"],
    SpecializationType.ARCHITECTURE: ["code_analysis", "file_read", "architecture"],
    SpecializationType.DATA_PROCESSING: ["file_read", "file_write", "data_processing"],
    SpecializationType.API_INTEGRATION: ["api_call", "http_request"],
    SpecializationType.SECURITY: ["security_scan", "code_analysis"],
    SpecializationType.PERFORMANCE: ["profiling", "performance_analysis"],
    SpecializationType.UI_UX: ["ui_generation", "component_creation"],
    SpecializationType.DEVOPS: ["terminal_execute", "deployment", "ci_cd"],
    SpecializationType.GENERAL: ["file_read", "file_write", "terminal_execute"]
}


def create_default_model(agent_name: str, specializations: Optional[List[SpecializationType]] = None) -> MentalModel:
    """
    Create a mental model with default specializations.

    Args:
        agent_name: Name of the agent
        specializations: List of specialization types (defaults to GENERAL)

    Returns:
        Configured MentalModel instance
    """
    model = MentalModel(agent_name=agent_name)

    specs = specializations or [SpecializationType.GENERAL]
    for spec_type in specs:
        capabilities = DEFAULT_SPECIALIZATION_CAPABILITIES.get(spec_type, [])
        model.add_specialization(spec_type, proficiency=0.5, capabilities=capabilities)

    return model
