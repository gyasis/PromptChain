"""
ChainFactory Data Models

Data classes for the ChainFactory system - strict, guardrailed chains
that can be versioned, nested, and managed via the VIN/Model system.

VIN = Unique chain instance identifier (like a car VIN)
Model = Chain template type (like a car model)
Version = Chain evolution (like model year)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime
from enum import Enum
import uuid
import hashlib


class StepType(str, Enum):
    """Types of steps in a chain."""
    PROMPT = "prompt"       # Execute a prompt template
    CHAIN = "chain"         # Call another chain (nested)
    FUNCTION = "function"   # Call a Python function
    AGENTIC = "agentic"     # AgenticStepProcessor invocation (hybrid mode only)


class ChainMode(str, Enum):
    """Execution modes for chains."""
    STRICT = "strict"       # Pure chain mode - no agentic steps
    HYBRID = "hybrid"       # Mixed mode - allows agentic steps


class ChainStepDefinition(BaseModel):
    """Definition of a single step in a chain.

    Different from models.ChainStep which captures execution results.
    This defines what a step SHOULD do, not what it DID.
    """
    id: str = Field(description="Unique step identifier within chain")
    type: StepType = Field(description="Type of step")

    # For prompt steps
    prompt_id: Optional[str] = Field(default=None, description="PrePrompt ID to load")
    content: Optional[str] = Field(default=None, description="Inline prompt content")
    strategy: Optional[str] = Field(default=None, description="Prompting strategy (cot, react, etc)")

    # For chain steps
    chain_id: Optional[str] = Field(default=None, description="Chain reference (model:version or VIN)")

    # For function steps
    function_name: Optional[str] = Field(default=None, description="Registered function name")

    # For agentic steps (hybrid mode only)
    objective: Optional[str] = Field(default=None, description="Objective for AgenticStepProcessor")
    max_steps: Optional[int] = Field(default=5, description="Max internal steps for agentic")
    tools: Optional[List[str]] = Field(default=None, description="Allowed tools for agentic step")

    # Common
    description: Optional[str] = Field(default=None, description="Human-readable step description")

    @field_validator('id', mode='before')
    @classmethod
    def generate_id_if_empty(cls, v):
        """Generate step ID if not provided."""
        if not v:
            return f"step_{uuid.uuid4().hex[:8]}"
        return v


class Guardrails(BaseModel):
    """Safety guardrails for chain execution."""
    max_steps: int = Field(default=10, description="Maximum number of steps allowed")
    allowed_tools: Optional[List[str]] = Field(default=None, description="Whitelisted tools (None = all)")
    forbidden_patterns: List[str] = Field(
        default_factory=lambda: ["rm -rf", "DROP TABLE", "DELETE FROM", "--no-preserve-root"],
        description="Patterns that will cause chain rejection"
    )
    timeout_seconds: int = Field(default=300, description="Maximum execution time")
    max_nested_depth: int = Field(default=5, description="Maximum chain nesting depth")
    require_approval_for: Optional[List[str]] = Field(
        default=None,
        description="Step types requiring human approval"
    )


class ChainDefinition(BaseModel):
    """Complete chain definition - the core data structure.

    This is what gets saved to disk and versioned. Contains all
    information needed to execute the chain.
    """
    # Identity (VIN System)
    vin: str = Field(description="Unique chain instance ID (VIN)")
    model: str = Field(description="Chain template type (model name)")
    version: str = Field(description="Chain version (v1.0, v1.1, etc)")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(default="user", description="Creator: 'user' or 'agent:model-name'")
    description: Optional[str] = Field(default=None, description="What this chain does")
    tags: List[str] = Field(default_factory=list)

    # Execution config
    mode: ChainMode = Field(default=ChainMode.STRICT, description="Execution mode")
    llm_model: str = Field(
        default="openai/gpt-4.1-mini-2025-04-14",
        description="LLM model for prompt steps"
    )
    guardrails: Guardrails = Field(default_factory=Guardrails)

    # The chain itself
    steps: List[ChainStepDefinition] = Field(description="Ordered list of steps")

    # Input/Output schema
    inputs: List[str] = Field(default_factory=lambda: ["input"], description="Expected input names")
    outputs: List[str] = Field(default_factory=lambda: ["output"], description="Output names")

    # Execution state (not persisted)
    is_temp: bool = Field(default=False, exclude=True, description="Ephemeral chain flag")

    @field_validator('vin', mode='before')
    @classmethod
    def generate_vin_if_empty(cls, v, info):
        """Generate VIN if not provided."""
        if not v:
            model_name = info.data.get('model', 'unknown')
            version = info.data.get('version', 'v1')
            unique_id = uuid.uuid4().hex[:12]
            return f"chain_{model_name}_{unique_id}_{version}"
        return v

    @field_validator('version', mode='before')
    @classmethod
    def normalize_version(cls, v):
        """Ensure version starts with 'v'."""
        if v and not v.startswith('v'):
            return f"v{v}"
        return v or "v1.0"

    def content_hash(self) -> str:
        """Generate hash of chain content for change detection."""
        content = f"{self.model}:{self.version}:{len(self.steps)}"
        for step in self.steps:
            content += f":{step.type}:{step.prompt_id or step.content or step.chain_id or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ValidationIssue(BaseModel):
    """Single validation issue."""
    severity: Literal["error", "warning", "info"] = "error"
    step_id: Optional[str] = None
    message: str
    suggestion: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of chain validation."""
    passed: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)
    guardrails_checked: bool = True

    def add_error(self, message: str, step_id: Optional[str] = None, suggestion: Optional[str] = None):
        """Add error and mark as failed."""
        self.passed = False
        self.issues.append(ValidationIssue(
            severity="error",
            step_id=step_id,
            message=message,
            suggestion=suggestion
        ))

    def add_warning(self, message: str, step_id: Optional[str] = None):
        """Add warning (doesn't fail validation)."""
        self.issues.append(ValidationIssue(
            severity="warning",
            step_id=step_id,
            message=message
        ))


class ChainManifest(BaseModel):
    """Metadata for a chain model (stored in manifest.json).

    Contains info about all versions of a particular chain model.
    """
    name: str = Field(description="Chain model name")
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    latest: str = Field(description="Latest version (e.g., 'v2.0')")
    versions: List[str] = Field(default_factory=list, description="All available versions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_version(self, version: str):
        """Add a new version to the manifest."""
        if version not in self.versions:
            self.versions.append(version)
            self.versions.sort(key=lambda v: [int(x) for x in v.lstrip('v').split('.')])
        self.latest = version
        self.updated_at = datetime.utcnow()


class ChainRegistryEntry(BaseModel):
    """Single entry in the chain registry."""
    model: str
    versions: List[str]
    latest: str
    tags: List[str] = Field(default_factory=list)
    path: str = Field(description="Relative path to chain model directory")


class ChainRegistry(BaseModel):
    """Master registry of all chains (registry.json).

    Provides quick lookup of available chains without scanning filesystem.
    """
    models: Dict[str, ChainRegistryEntry] = Field(default_factory=dict)
    by_vin: Dict[str, str] = Field(
        default_factory=dict,
        description="VIN to file path mapping"
    )
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def register_chain(self, chain_def: ChainDefinition, file_path: str):
        """Register a chain in the registry."""
        model = chain_def.model
        version = chain_def.version

        if model not in self.models:
            self.models[model] = ChainRegistryEntry(
                model=model,
                versions=[version],
                latest=version,
                tags=chain_def.tags,
                path=model
            )
        else:
            entry = self.models[model]
            if version not in entry.versions:
                entry.versions.append(version)
            entry.latest = version
            entry.tags = list(set(entry.tags + chain_def.tags))

        self.by_vin[chain_def.vin] = file_path
        self.updated_at = datetime.utcnow()

    def get_latest_version(self, model: str) -> Optional[str]:
        """Get latest version for a model."""
        if model in self.models:
            return self.models[model].latest
        return None

    def resolve_chain_path(self, reference: str) -> Optional[str]:
        """Resolve chain reference to file path.

        Supports:
        - VIN: "chain_abc123_v1.0"
        - Model:version: "query-optimizer:v1.0"
        - Model (latest): "query-optimizer"
        """
        # Check if it's a VIN
        if reference in self.by_vin:
            return self.by_vin[reference]

        # Check if it's model:version
        if ':' in reference:
            model, version = reference.split(':', 1)
            if model in self.models:
                return f"{model}/{version}.json"

        # Check if it's just a model name (use latest)
        if reference in self.models:
            latest = self.models[reference].latest
            return f"{reference}/{latest}.json"

        return None


class ChainExecutionRecord(BaseModel):
    """Record of a chain execution (for analytics/history)."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    vin: str
    input_hash: str = Field(description="Hash of input for cache lookup")
    output: Optional[str] = None
    execution_time_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    steps_executed: int = 0


class ChainCall:
    """Marker class for chain invocation in PromptChain instructions.

    When PromptChain encounters a ChainCall instruction, it delegates
    execution to ChainExecutor instead of processing it as a prompt.

    Usage in PromptChain:
        chain = PromptChain(
            models=["openai/gpt-4.1-mini-2025-04-14"],
            instructions=[
                "Analyze: {input}",
                ChainCall("query-optimizer:v1.0"),  # Executes chain
                "Summarize: {input}"
            ]
        )
    """

    def __init__(self, chain_ref: str):
        """Initialize ChainCall.

        Args:
            chain_ref: Chain reference (model:version, model, or VIN)
        """
        self.chain_ref = chain_ref

        # Parse reference
        if ':' in chain_ref:
            parts = chain_ref.split(':', 1)
            self.model = parts[0]
            self.version = parts[1]
        else:
            self.model = chain_ref
            self.version = "latest"

    def __repr__(self):
        return f"ChainCall({self.chain_ref})"

    def __str__(self):
        return f"chain:{self.chain_ref}"


class TaskOutput(BaseModel):
    """Task output matching tasks.md format.

    Used by TaskBuilder to generate chains that align with
    the project's task management structure.
    """
    task_id: str = Field(description="T001, T002, etc.")
    parallel: bool = Field(default=False, description="[P] marker")
    story: Optional[str] = Field(default=None, description="US1, US2, etc.")
    description: str
    chain_id: Optional[str] = Field(default=None, description="chain:model:version")
    phase: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    execution: Literal["strict", "agentic"] = "strict"
