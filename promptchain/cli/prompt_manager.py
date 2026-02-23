"""
Prompt management system for CLI agents.

This module provides the PromptManager class that bridges PromptChain's
PrePrompt system with the CLI, enabling prompt discovery, loading, and
template management for agents.
"""

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from promptchain.utils.preprompt import PrePrompt
from promptchain.utils.prompt_loader import list_available_prompts
from promptchain.cli.models.prompt import Prompt, Strategy, AgentTemplate

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompts, strategies, and templates for CLI agents.

    This class provides a high-level interface to PromptChain's prompt
    infrastructure, enabling:
    - Discovery and listing of available prompts
    - Loading prompts with optional strategy application
    - Saving custom prompts and strategies
    - Managing agent templates

    Attributes:
        prompts_dir: Root directory for user prompts (~/.promptchain/prompts/)
        agents_dir: Directory for agent-specific prompts
        strategies_dir: Directory for strategy templates
        custom_dir: Directory for custom user prompts
        templates_dir: Directory for agent templates
        preprompt: PrePrompt instance for prompt loading
    """

    def __init__(self, prompts_dir: Path):
        """Initialize PromptManager.

        Args:
            prompts_dir: Root directory for prompts (typically ~/.promptchain/prompts/)
        """
        self.prompts_dir = Path(prompts_dir)
        self.agents_dir = self.prompts_dir / "agents"
        self.strategies_dir = self.prompts_dir / "strategies"
        self.custom_dir = self.prompts_dir / "custom"
        self.templates_dir = self.prompts_dir.parent / "templates" / "agents"

        # Create directory structure if it doesn't exist
        self._ensure_directories()

        # Initialize PrePrompt with custom directories
        # Note: PrePrompt looks for strategies in its standard_strategy_dir
        # which defaults to library path. We need to manually handle
        # user strategies from self.strategies_dir
        self.preprompt = PrePrompt(
            additional_prompt_dirs=[
                str(self.agents_dir),
                str(self.custom_dir)
            ]
        )

        logger.info(f"PromptManager initialized with prompts_dir: {self.prompts_dir}")

    def _ensure_directories(self):
        """Create prompts directory structure if it doesn't exist."""
        for directory in [
            self.prompts_dir,
            self.agents_dir,
            self.strategies_dir,
            self.custom_dir,
            self.templates_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def list_prompts(self, category: Optional[str] = None, search: Optional[str] = None) -> List[Prompt]:
        """List all available prompts, optionally filtered.

        Args:
            category: Filter by category (e.g., 'agents', 'custom')
            search: Search term to filter prompt IDs and descriptions

        Returns:
            List of Prompt objects
        """
        all_prompts = []

        # Manually discover prompts from our directories
        # (prompt_loader has path resolution issues)
        dirs_to_scan = {
            'agents': self.agents_dir,
            'custom': self.custom_dir
        }

        for cat, dir_path in dirs_to_scan.items():
            # Filter by category if specified
            if category and cat != category:
                continue

            if not dir_path.exists():
                continue

            # Scan for .md files
            for prompt_file in dir_path.glob("*.md"):
                try:
                    content = prompt_file.read_text(encoding='utf-8')

                    # Extract description (first non-empty line after title)
                    description = None
                    lines = content.split('\n')
                    for line in lines[1:]:  # Skip title
                        if line.strip() and not line.startswith('#'):
                            description = line.strip()
                            break

                    # Create Prompt object
                    prompt_id = prompt_file.stem
                    prompt = Prompt(
                        id=prompt_id,
                        content=content,
                        category=cat,
                        description=description,
                        path=str(prompt_file),
                        strategies=[s.id for s in self.list_strategies()]
                    )

                    # Filter by search term if specified
                    if search:
                        search_lower = search.lower()
                        if search_lower not in prompt.id.lower() and \
                           (not prompt.description or search_lower not in prompt.description.lower()):
                            continue

                    all_prompts.append(prompt)

                except Exception as e:
                    logger.warning(f"Could not load prompt {prompt_file}: {e}")
                    continue

        return sorted(all_prompts, key=lambda p: (p.category, p.id))

    def load_prompt(self, prompt_id: str, strategy: Optional[str] = None) -> str:
        """Load prompt content using PrePrompt system.

        Args:
            prompt_id: Prompt identifier (filename without extension)
            strategy: Optional strategy ID to apply

        Returns:
            Full prompt text with strategy prepended if specified

        Raises:
            FileNotFoundError: If prompt or strategy not found
        """
        try:
            # Load base prompt
            base_content = self.preprompt.load(prompt_id)

            # If strategy requested, manually prepend it
            # (PrePrompt looks for strategies in library path, we have them in user dir)
            if strategy:
                strategy_obj = None
                for s in self.list_strategies():
                    if s.id == strategy:
                        strategy_obj = s
                        break

                if not strategy_obj:
                    raise FileNotFoundError(f"Strategy '{strategy}' not found")

                # Prepend strategy prompt
                combined = f"{strategy_obj.prompt}\n\n{base_content}"
                logger.debug(f"Loaded prompt '{prompt_id}' with strategy '{strategy}'")
                return combined

            logger.debug(f"Loaded prompt: {prompt_id}")
            return base_content

        except FileNotFoundError as e:
            logger.error(f"Prompt not found: {prompt_id}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_id}: {e}")
            raise

    def save_prompt(
        self,
        prompt_id: str,
        content: str,
        category: str = "custom",
        description: Optional[str] = None
    ) -> Path:
        """Save a new prompt to the specified category.

        Args:
            prompt_id: Unique identifier for the prompt
            content: Prompt text content
            category: Category directory (default: 'custom')
            description: Optional description (added as comment at top)

        Returns:
            Path to saved prompt file
        """
        # Determine target directory
        if category == "agents":
            category_dir = self.agents_dir
        elif category == "custom":
            category_dir = self.custom_dir
        else:
            # Create custom category directory
            category_dir = self.prompts_dir / category

        category_dir.mkdir(exist_ok=True)

        # Add description as comment if provided
        if description and not content.startswith("#"):
            content = f"# {prompt_id.title()}\n\n{description}\n\n{content}"

        # Save prompt file
        prompt_path = category_dir / f"{prompt_id}.md"
        prompt_path.write_text(content, encoding='utf-8')

        # Rescan prompts to update cache
        self.preprompt._scan_all_prompt_dirs()

        logger.info(f"Saved prompt '{prompt_id}' to {prompt_path}")
        return prompt_path

    def list_strategies(self) -> List[Strategy]:
        """List all available strategies.

        Returns:
            List of Strategy objects
        """
        strategies = []

        if not self.strategies_dir.exists():
            return strategies

        for strategy_file in self.strategies_dir.glob("*.json"):
            try:
                data = json.loads(strategy_file.read_text())
                strategy = Strategy(
                    id=strategy_file.stem,
                    name=data.get('name', strategy_file.stem),
                    prompt=data.get('prompt', ''),
                    description=data.get('description')
                )
                strategies.append(strategy)
            except Exception as e:
                logger.warning(f"Could not load strategy {strategy_file}: {e}")

        return sorted(strategies, key=lambda s: s.id)

    def save_strategy(
        self,
        strategy_id: str,
        name: str,
        prompt: str,
        description: Optional[str] = None
    ) -> Path:
        """Save a new strategy.

        Args:
            strategy_id: Unique identifier for the strategy
            name: Display name
            prompt: Strategy instruction text
            description: Optional description

        Returns:
            Path to saved strategy file
        """
        strategy_data = {
            "name": name,
            "prompt": prompt,
            "description": description
        }

        strategy_path = self.strategies_dir / f"{strategy_id}.json"
        strategy_path.write_text(
            json.dumps(strategy_data, indent=2),
            encoding='utf-8'
        )

        logger.info(f"Saved strategy '{strategy_id}' to {strategy_path}")
        return strategy_path

    def list_templates(self, tag: Optional[str] = None) -> List[AgentTemplate]:
        """List all agent templates, optionally filtered by tag.

        Args:
            tag: Filter templates by tag

        Returns:
            List of AgentTemplate objects
        """
        templates = []

        if not self.templates_dir.exists():
            return templates

        for template_file in self.templates_dir.glob("*.json"):
            try:
                data = json.loads(template_file.read_text())
                template = AgentTemplate.from_dict(data)

                # Filter by tag if specified
                if tag and tag not in template.tags:
                    continue

                templates.append(template)
            except Exception as e:
                logger.warning(f"Could not load template {template_file}: {e}")

        return sorted(templates, key=lambda t: t.name)

    def save_template(self, template: AgentTemplate) -> Path:
        """Save an agent template.

        Args:
            template: AgentTemplate object to save

        Returns:
            Path to saved template file
        """
        template_filename = template.name.lower().replace(' ', '-') + '.json'
        template_path = self.templates_dir / template_filename

        template_path.write_text(
            json.dumps(template.to_dict(), indent=2),
            encoding='utf-8'
        )

        logger.info(f"Saved template '{template.name}' to {template_path}")
        return template_path

    def get_template(self, name: str) -> Optional[AgentTemplate]:
        """Get a specific template by name.

        Args:
            name: Template name (case-insensitive)

        Returns:
            AgentTemplate if found, None otherwise
        """
        name_lower = name.lower()
        for template in self.list_templates():
            if template.name.lower() == name_lower:
                return template
        return None

    def _load_prompt_content(self, path: str) -> str:
        """Load prompt file content.

        Args:
            path: Path to prompt file

        Returns:
            File content as string
        """
        return Path(path).read_text(encoding='utf-8')

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get a specific prompt by ID.

        Args:
            prompt_id: Prompt identifier

        Returns:
            Prompt object if found, None otherwise
        """
        prompts = self.list_prompts()
        for prompt in prompts:
            if prompt.id == prompt_id:
                return prompt
        return None

    def prompt_exists(self, prompt_id: str) -> bool:
        """Check if a prompt exists.

        Args:
            prompt_id: Prompt identifier

        Returns:
            True if prompt exists, False otherwise
        """
        return self.get_prompt(prompt_id) is not None
