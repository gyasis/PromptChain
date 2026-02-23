"""
ChainFactory - Factory for creating, versioning, and managing strict chains.

The ChainFactory implements the VIN/Model system for chains:
- VIN: Unique chain instance identifier (like a car VIN)
- Model: Chain template type (like a car model)
- Version: Chain evolution (like model year)

Directory Structure:
~/.promptchain/
├── chains/               # Saved/permanent chains
│   ├── registry.json     # Master registry
│   ├── {model}/
│   │   ├── manifest.json
│   │   └── v1.0.json
└── chains_temp/          # Ephemeral chains (auto-cleanup)
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import logging

from .chain_models import (
    ChainDefinition,
    ChainStepDefinition,
    ChainManifest,
    ChainRegistry,
    ChainMode,
    StepType,
    Guardrails,
    ValidationResult,
    ValidationIssue
)

logger = logging.getLogger(__name__)


class ChainFactoryError(Exception):
    """Base exception for ChainFactory errors."""
    pass


class ChainNotFoundError(ChainFactoryError):
    """Chain not found in registry."""
    pass


class ChainValidationError(ChainFactoryError):
    """Chain failed validation."""
    pass


class ChainFactory:
    """Factory for creating, versioning, and managing strict chains.

    Usage:
        factory = ChainFactory()
        chain = factory.create("query-optimizer")  # Latest version
        result = chain.execute("my input")

        # Or create temp chain
        temp = factory.create_temp([
            {"type": "prompt", "content": "Analyze: {input}"},
            {"type": "prompt", "content": "Summarize: {input}"}
        ])

        # Save as permanent
        vin = factory.save(temp, model="my-analyzer", version="v1.0")
    """

    DEFAULT_BASE_PATH = os.path.expanduser("~/.promptchain")
    CHAINS_DIR = "chains"
    CHAINS_TEMP_DIR = "chains_temp"
    REGISTRY_FILE = "registry.json"

    def __init__(
        self,
        base_path: Optional[str] = None,
        auto_setup: bool = True,
        default_model: str = "openai/gpt-4.1-mini-2025-04-14"
    ):
        """Initialize ChainFactory.

        Args:
            base_path: Base path for chain storage (default: ~/.promptchain)
            auto_setup: Automatically create directory structure
            default_model: Default LLM model for chains
        """
        self.base_path = Path(base_path or self.DEFAULT_BASE_PATH)
        self.chains_path = self.base_path / self.CHAINS_DIR
        self.temp_path = self.base_path / self.CHAINS_TEMP_DIR
        self.registry_path = self.chains_path / self.REGISTRY_FILE
        self.default_model = default_model

        self._registry: Optional[ChainRegistry] = None

        if auto_setup:
            self._setup_directories()

    def _setup_directories(self):
        """Create directory structure if it doesn't exist."""
        self.chains_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)

        # Create registry if doesn't exist
        if not self.registry_path.exists():
            self._save_registry(ChainRegistry())
            logger.info(f"Created chain registry at {self.registry_path}")

    @property
    def registry(self) -> ChainRegistry:
        """Get the chain registry (lazy loaded)."""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self) -> ChainRegistry:
        """Load registry from disk."""
        if not self.registry_path.exists():
            return ChainRegistry()
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            return ChainRegistry.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}. Creating new.")
            return ChainRegistry()

    def _save_registry(self, registry: Optional[ChainRegistry] = None):
        """Save registry to disk."""
        registry = registry or self.registry
        with open(self.registry_path, 'w') as f:
            json.dump(registry.model_dump(mode='json'), f, indent=2, default=str)
        self._registry = registry

    # =========================================================================
    # Core Methods
    # =========================================================================

    def create(
        self,
        model: str,
        version: str = "latest"
    ) -> ChainDefinition:
        """Create a chain from a saved model.

        Args:
            model: Chain model name (e.g., "query-optimizer")
            version: Version to load ("latest", "v1.0", etc.)

        Returns:
            ChainDefinition ready for execution

        Raises:
            ChainNotFoundError: If model/version not found
        """
        # Resolve "latest" to actual version
        if version == "latest":
            version = self.registry.get_latest_version(model)
            if not version:
                raise ChainNotFoundError(f"No versions found for model: {model}")

        # Load from file
        chain_path = self.chains_path / model / f"{version}.json"
        if not chain_path.exists():
            raise ChainNotFoundError(f"Chain not found: {model}:{version}")

        try:
            with open(chain_path, 'r') as f:
                data = json.load(f)
            chain = ChainDefinition.model_validate(data)
            logger.debug(f"Loaded chain: {chain.vin}")
            return chain
        except Exception as e:
            raise ChainFactoryError(f"Failed to load chain {model}:{version}: {e}")

    def create_temp(
        self,
        steps: List[Dict[str, Any]],
        model_name: str = "temp",
        description: Optional[str] = None,
        guardrails: Optional[Guardrails] = None,
        llm_model: Optional[str] = None,
        mode: ChainMode = ChainMode.STRICT
    ) -> ChainDefinition:
        """Create an ephemeral chain (not saved to disk).

        Args:
            steps: List of step definitions
            model_name: Name for the temp chain
            description: Optional description
            guardrails: Custom guardrails (uses defaults if None)
            llm_model: LLM model to use
            mode: Execution mode (strict or hybrid)

        Returns:
            ChainDefinition ready for execution
        """
        # Convert step dicts to ChainStepDefinition
        parsed_steps = []
        for i, step in enumerate(steps):
            step_copy = step.copy()
            if 'id' not in step_copy:
                step_copy['id'] = f"step_{i + 1}"
            if 'type' not in step_copy:
                # Infer type from content
                if 'chain_id' in step_copy:
                    step_copy['type'] = StepType.CHAIN
                elif 'function_name' in step_copy:
                    step_copy['type'] = StepType.FUNCTION
                elif 'objective' in step_copy:
                    step_copy['type'] = StepType.AGENTIC
                else:
                    step_copy['type'] = StepType.PROMPT
            parsed_steps.append(ChainStepDefinition.model_validate(step_copy))

        # Generate temp VIN
        temp_id = uuid.uuid4().hex[:12]
        vin = f"temp_{model_name}_{temp_id}"

        chain = ChainDefinition(
            vin=vin,
            model=model_name,
            version="temp",
            description=description or "Temporary chain",
            mode=mode,
            llm_model=llm_model or self.default_model,
            guardrails=guardrails or Guardrails(),
            steps=parsed_steps,
            is_temp=True
        )

        # Optionally save to temp directory for debugging
        temp_file = self.temp_path / f"{vin}.json"
        with open(temp_file, 'w') as f:
            json.dump(chain.model_dump(mode='json'), f, indent=2, default=str)

        logger.debug(f"Created temp chain: {vin}")
        return chain

    def save(
        self,
        chain: ChainDefinition,
        model: Optional[str] = None,
        version: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """Save a chain to permanent storage.

        Args:
            chain: ChainDefinition to save
            model: Model name (uses chain.model if None)
            version: Version (auto-increments if None)
            description: Override chain description

        Returns:
            VIN of the saved chain
        """
        model = model or chain.model
        if model == "temp":
            raise ChainFactoryError("Cannot save with model name 'temp'. Provide a model name.")

        # Determine version
        if version:
            if not version.startswith('v'):
                version = f"v{version}"
        else:
            # Auto-increment version
            existing = self.list_versions(model)
            if existing:
                latest = existing[-1]
                parts = latest.lstrip('v').split('.')
                parts[-1] = str(int(parts[-1]) + 1)
                version = 'v' + '.'.join(parts)
            else:
                version = "v1.0"

        # Generate new VIN
        vin = f"chain_{model}_{uuid.uuid4().hex[:12]}_{version}"

        # Create model directory
        model_dir = self.chains_path / model
        model_dir.mkdir(parents=True, exist_ok=True)

        # Update chain with new identity
        saved_chain = chain.model_copy(update={
            'vin': vin,
            'model': model,
            'version': version,
            'description': description or chain.description,
            'is_temp': False,
            'created_at': datetime.utcnow()
        })

        # Save chain file
        chain_file = model_dir / f"{version}.json"
        with open(chain_file, 'w') as f:
            json.dump(saved_chain.model_dump(mode='json'), f, indent=2, default=str)

        # Update/create manifest
        self._update_manifest(model, saved_chain)

        # Update registry
        self.registry.register_chain(saved_chain, str(chain_file.relative_to(self.chains_path)))
        self._save_registry()

        # Clean up temp file if exists
        if chain.is_temp:
            temp_file = self.temp_path / f"{chain.vin}.json"
            if temp_file.exists():
                temp_file.unlink()

        logger.info(f"Saved chain: {vin} at {chain_file}")
        return vin

    def save_from_dict(
        self,
        chain_def: Dict[str, Any],
        model: str,
        version: Optional[str] = None
    ) -> str:
        """Save a chain from a dictionary definition.

        Args:
            chain_def: Dictionary chain definition
            model: Model name
            version: Version (auto-increments if None)

        Returns:
            VIN of saved chain
        """
        # Parse into ChainDefinition
        chain = ChainDefinition.model_validate(chain_def)
        return self.save(chain, model=model, version=version)

    def _update_manifest(self, model: str, chain: ChainDefinition):
        """Update or create manifest for a chain model."""
        model_dir = self.chains_path / model
        manifest_path = model_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = ChainManifest.model_validate(json.load(f))
        else:
            manifest = ChainManifest(
                name=model,
                description=chain.description,
                tags=chain.tags,
                latest=chain.version,
                versions=[chain.version]
            )

        manifest.add_version(chain.version)

        with open(manifest_path, 'w') as f:
            json.dump(manifest.model_dump(mode='json'), f, indent=2, default=str)

    # =========================================================================
    # Registry Methods
    # =========================================================================

    def list_models(self) -> List[str]:
        """List all available chain models."""
        return list(self.registry.models.keys())

    def list_chains(self) -> Dict[str, Dict[str, Any]]:
        """List all chains with their metadata.

        Returns:
            Dict mapping model names to their info (versions, latest, tags)
        """
        result = {}
        for name, entry in self.registry.models.items():
            result[name] = {
                "latest": entry.latest,
                "versions": entry.versions,
                "tags": entry.tags,
                "path": entry.path
            }
        return result

    def list_versions(self, model: str) -> List[str]:
        """List all versions of a model."""
        if model in self.registry.models:
            return sorted(self.registry.models[model].versions)
        return []

    def get_by_vin(self, vin: str) -> ChainDefinition:
        """Get a chain by its VIN.

        Args:
            vin: Unique chain identifier

        Returns:
            ChainDefinition

        Raises:
            ChainNotFoundError: If VIN not found
        """
        path = self.registry.by_vin.get(vin)
        if not path:
            raise ChainNotFoundError(f"VIN not found: {vin}")

        chain_path = self.chains_path / path
        if not chain_path.exists():
            raise ChainNotFoundError(f"Chain file missing for VIN: {vin}")

        with open(chain_path, 'r') as f:
            data = json.load(f)
        return ChainDefinition.model_validate(data)

    def resolve(self, reference: str) -> ChainDefinition:
        """Resolve any chain reference to ChainDefinition.

        Supports:
        - VIN: "chain_abc123_v1.0"
        - model:version: "query-optimizer:v1.0"
        - model (latest): "query-optimizer"

        Args:
            reference: Chain reference string

        Returns:
            ChainDefinition
        """
        # Check if VIN
        if reference.startswith("chain_") or reference.startswith("temp_"):
            return self.get_by_vin(reference)

        # Check model:version
        if ':' in reference:
            model, version = reference.split(':', 1)
            return self.create(model, version)

        # Must be model name (use latest)
        return self.create(reference, "latest")

    def exists(self, model: str, version: str = "latest") -> bool:
        """Check if a chain exists.

        Args:
            model: Chain model name
            version: Version to check

        Returns:
            True if chain exists
        """
        if version == "latest":
            return model in self.registry.models

        if model in self.registry.models:
            return version in self.registry.models[model].versions

        return False

    # =========================================================================
    # Management Methods
    # =========================================================================

    def delete(self, model: str, version: Optional[str] = None, force: bool = False) -> bool:
        """Delete a chain or all versions of a model.

        Args:
            model: Chain model name
            version: Specific version to delete (None = all)
            force: Skip confirmation

        Returns:
            True if deleted
        """
        if model not in self.registry.models:
            return False

        if version:
            # Delete specific version
            chain_path = self.chains_path / model / f"{version}.json"
            if chain_path.exists():
                chain_path.unlink()

            # Update registry
            entry = self.registry.models[model]
            if version in entry.versions:
                entry.versions.remove(version)

            # Remove VINs for this version
            vins_to_remove = [
                vin for vin, path in self.registry.by_vin.items()
                if f"{model}/{version}" in path
            ]
            for vin in vins_to_remove:
                del self.registry.by_vin[vin]

            # If no versions left, remove model entirely
            if not entry.versions:
                del self.registry.models[model]
                shutil.rmtree(self.chains_path / model, ignore_errors=True)

            self._save_registry()
            logger.info(f"Deleted chain version: {model}:{version}")
        else:
            # Delete entire model
            if not force:
                logger.warning(f"Use force=True to delete all versions of {model}")
                return False

            shutil.rmtree(self.chains_path / model, ignore_errors=True)

            # Clean registry
            del self.registry.models[model]
            self.registry.by_vin = {
                vin: path for vin, path in self.registry.by_vin.items()
                if not path.startswith(f"{model}/")
            }
            self._save_registry()
            logger.info(f"Deleted all versions of model: {model}")

        return True

    def fork(
        self,
        model: str,
        version: str,
        new_version: str,
        new_model: Optional[str] = None
    ) -> str:
        """Fork a chain to create a new version.

        Args:
            model: Source model name
            version: Source version
            new_version: Target version
            new_model: Optional new model name (fork to different model)

        Returns:
            VIN of forked chain
        """
        source = self.create(model, version)
        target_model = new_model or model

        return self.save(
            source,
            model=target_model,
            version=new_version,
            description=f"Forked from {model}:{version}"
        )

    def cleanup_temp(self, max_age_hours: int = 24) -> int:
        """Clean up old temp chains.

        Args:
            max_age_hours: Delete temps older than this

        Returns:
            Number of files deleted
        """
        from datetime import timedelta

        count = 0
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        for temp_file in self.temp_path.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(temp_file.stat().st_mtime)
                if mtime < cutoff:
                    temp_file.unlink()
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to clean {temp_file}: {e}")

        if count:
            logger.info(f"Cleaned up {count} temp chains")
        return count

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, chain: ChainDefinition) -> ValidationResult:
        """Validate a chain definition.

        Checks:
        - Step configuration is valid
        - Guardrails are respected
        - No forbidden patterns
        - Nested chains exist

        Args:
            chain: ChainDefinition to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        # Check step count
        if len(chain.steps) > chain.guardrails.max_steps:
            result.add_error(
                f"Chain has {len(chain.steps)} steps, max allowed is {chain.guardrails.max_steps}",
                suggestion="Reduce steps or increase guardrails.max_steps"
            )

        # Check each step
        for step in chain.steps:
            # Validate step type configuration
            if step.type == StepType.PROMPT:
                if not step.prompt_id and not step.content:
                    result.add_error(
                        "Prompt step missing both prompt_id and content",
                        step_id=step.id
                    )

            elif step.type == StepType.CHAIN:
                if not step.chain_id:
                    result.add_error(
                        "Chain step missing chain_id",
                        step_id=step.id
                    )
                else:
                    # Check if nested chain exists
                    try:
                        self.resolve(step.chain_id)
                    except ChainNotFoundError:
                        result.add_warning(
                            f"Nested chain not found: {step.chain_id}",
                            step_id=step.id
                        )

            elif step.type == StepType.AGENTIC:
                if chain.mode == ChainMode.STRICT:
                    result.add_error(
                        "Agentic steps not allowed in strict mode",
                        step_id=step.id,
                        suggestion="Change mode to 'hybrid' or use chain/prompt steps"
                    )

            # Check for forbidden patterns
            content_to_check = step.content or step.objective or ""
            for pattern in chain.guardrails.forbidden_patterns:
                if pattern.lower() in content_to_check.lower():
                    result.add_error(
                        f"Forbidden pattern detected: '{pattern}'",
                        step_id=step.id
                    )

        return result

    # =========================================================================
    # Info Methods
    # =========================================================================

    def info(self, model: str, version: str = "latest") -> Dict[str, Any]:
        """Get detailed info about a chain.

        Args:
            model: Chain model name
            version: Version

        Returns:
            Dictionary with chain details
        """
        chain = self.create(model, version)
        return {
            "vin": chain.vin,
            "model": chain.model,
            "version": chain.version,
            "description": chain.description,
            "mode": chain.mode.value,
            "llm_model": chain.llm_model,
            "step_count": len(chain.steps),
            "steps": [
                {"id": s.id, "type": s.type.value, "description": s.description}
                for s in chain.steps
            ],
            "guardrails": chain.guardrails.model_dump(),
            "created_at": chain.created_at.isoformat(),
            "created_by": chain.created_by,
            "tags": chain.tags
        }

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for chains by name or tags.

        Args:
            query: Search string for model names/descriptions
            tags: Tags to filter by

        Returns:
            List of matching chain summaries
        """
        results = []

        for model_name, entry in self.registry.models.items():
            # Check name match
            if query and query.lower() not in model_name.lower():
                continue

            # Check tag match
            if tags and not any(tag in entry.tags for tag in tags):
                continue

            results.append({
                "model": model_name,
                "latest": entry.latest,
                "versions": entry.versions,
                "tags": entry.tags
            })

        return results
