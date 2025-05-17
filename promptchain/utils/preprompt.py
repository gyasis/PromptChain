# promptchain/utils/preprompt.py
import os
import json
import glob
from typing import Optional, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Determine the directory where this file (preprompt.py) is located
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the standard relative paths for prompts and strategies relative to utils/
# Go up TWO levels from utils to reach the project root, then into prompts
_STANDARD_PROMPT_DIR = os.path.join(_UTILS_DIR, "../../prompts") 
_STANDARD_STRATEGY_DIR = os.path.join(_STANDARD_PROMPT_DIR, "strategies")
# Define common prompt file extensions
_PROMPT_EXTENSIONS = ('.txt', '.md', '.xml', '.json')

class PrePrompt:
    """
    Manages loading base prompts from standard and/or additional directories,
    and optionally prepending strategy prompts from a standard location.

    Prompts are identified by their filename without extension (promptID).
    If the same promptID exists in multiple directories, prompts from
    additional_prompt_dirs are prioritized over the standard_prompt_dir.
    Strategies are identified by their JSON filename without extension (strategyID)
    and are only loaded from the standard strategy directory.
    """
    def __init__(self, additional_prompt_dirs: Optional[List[str]] = None):
        """
        Initializes the PrePrompt manager.

        Args:
            additional_prompt_dirs: An optional list of absolute or relative paths
                                      to directories containing custom prompts.
        """
        self.standard_prompt_dir = os.path.normpath(_STANDARD_PROMPT_DIR)
        self.standard_strategy_dir = os.path.normpath(_STANDARD_STRATEGY_DIR)
        self.additional_prompt_dirs = [os.path.normpath(d) for d in additional_prompt_dirs] if additional_prompt_dirs else []

        self._prompts: Dict[str, str] = {} # Cache: {promptID: full_path}
        self._strategies: Dict[str, str] = {} # Cache for strategies: {strategyID: strategy_prompt_text}

        self._validate_dirs()
        self._scan_all_prompt_dirs() # Populate the prompt cache

    def _validate_dirs(self):
        """Checks if configured directories exist."""
        # Only warn about missing standard prompt dir if NO additional dirs were given
        if not os.path.isdir(self.standard_prompt_dir):
            if not self.additional_prompt_dirs:
                logger.warning(
                    f"Standard prompt directory not found or not a directory at: "
                    f"{self.standard_prompt_dir}. Loading standard prompts by ID might fail."
                )
            # else: # If additional dirs are present, don't warn about standard one
            #     logger.debug(f"Standard prompt dir not found, but additional dirs provided: {self.standard_prompt_dir}")
        else:
            logger.info(f"PrePrompt: Using standard prompts dir: {self.standard_prompt_dir}")

        # Only warn about missing standard strategy dir if NO additional dirs were given
        # (Strategies are currently ONLY loaded from standard dir, so maybe always warn? Let's keep the conditional logic for consistency)
        if not os.path.isdir(self.standard_strategy_dir):
            if not self.additional_prompt_dirs:
                logger.warning(f"Standard strategy directory not found at: {self.standard_strategy_dir}. Strategies cannot be loaded by ID.")
            # else:
            #     logger.debug(f"Standard strategy dir not found, but additional prompt dirs provided: {self.standard_strategy_dir}")
        else:
            logger.info(f"PrePrompt: Using standard strategies dir: {self.standard_strategy_dir}")

        valid_additional_dirs = []
        for i, dir_path in enumerate(self.additional_prompt_dirs):
             if not os.path.isdir(dir_path):
                 logger.warning(f"Additional prompt directory #{i+1} not found or not a directory: {dir_path}. Skipping.")
             else:
                 logger.info(f"PrePrompt: Added additional prompts dir: {dir_path}")
                 valid_additional_dirs.append(dir_path)
        self.additional_prompt_dirs = valid_additional_dirs


    def _scan_prompt_dir(self, directory: str):
        """Scans a single directory for prompt files and adds them to the cache if not already present."""
        if not os.path.isdir(directory):
            return # Skip if directory doesn't exist

        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                # Ensure it's a file and has a valid extension
                if os.path.isfile(item_path):
                    name, ext = os.path.splitext(item)
                    if ext.lower() in _PROMPT_EXTENSIONS:
                        promptID = name
                        # Add to cache only if the ID doesn't exist yet
                        # This respects the prioritization order (additional first)
                        if promptID not in self._prompts:
                            self._prompts[promptID] = item_path
                            logger.debug(f"Found prompt '{promptID}' at {item_path}")
        except OSError as e:
             logger.error(f"Error scanning directory {directory}: {e}")

    def _scan_all_prompt_dirs(self):
        """Scans all configured prompt directories (additional first, then standard)."""
        self._prompts = {} # Reset cache before scan
        logger.debug("Scanning for prompts...")

        # 1. Scan additional directories (prioritized)
        for dir_path in self.additional_prompt_dirs:
            self._scan_prompt_dir(dir_path)

        # 2. Scan standard directory (only add if ID not found in additional dirs)
        self._scan_prompt_dir(self.standard_prompt_dir)

        logger.info(f"Scan complete. Found {len(self._prompts)} unique prompt IDs.")


    def list_prompt_ids(self) -> List[str]:
        """Lists available unique prompt IDs found across all configured directories."""
        return sorted(list(self._prompts.keys()))

    def list_strategy_ids(self) -> List[str]:
        """Lists available strategy IDs (json filenames without extension) from the standard directory."""
        # Strategies are still only loaded from the standard directory
        if not self.standard_strategy_dir or not os.path.isdir(self.standard_strategy_dir): return []
        ids = []
        try:
            for item in os.listdir(self.standard_strategy_dir):
                item_path = os.path.join(self.standard_strategy_dir, item)
                if os.path.isfile(item_path) and item.endswith(".json"):
                    name, _ = os.path.splitext(item)
                    ids.append(name)
            return sorted(ids)
        except OSError as e:
            logger.error(f"Error listing standard strategies in {self.standard_strategy_dir}: {e}")
            return []

    def _find_prompt_file(self, promptID: str) -> Optional[str]:
        """Finds the first file matching the promptID in the standard prompt directory."""
        if not self.standard_prompt_dir: return None # Cannot search if dir wasn't found
        if os.path.sep in promptID or (os.path.altsep and os.path.altsep in promptID):
             raise ValueError(f"promptID '{promptID}' cannot contain path separators.")
        search_pattern = os.path.join(self.standard_prompt_dir, f"{promptID}.*")
        matching_files = glob.glob(search_pattern)
        # Filter out directories just in case glob picks them up somehow
        files_only = [f for f in matching_files if os.path.isfile(f)]
        return files_only[0] if files_only else None

    def _load_strategy_prompt(self, strategy: str) -> str:
        """Loads the prompt text from a strategy JSON file in the standard directory."""
        # Check cache first
        if strategy in self._strategies:
            return self._strategies[strategy]

        # Ensure standard strategy dir exists before proceeding
        if not self.standard_strategy_dir or not os.path.isdir(self.standard_strategy_dir):
             raise FileNotFoundError(
                 f"Cannot load strategy '{strategy}' because the standard strategy "
                 f"directory was not found or configured correctly: {self.standard_strategy_dir}"
             )

        if os.path.sep in strategy or (os.path.altsep and os.path.altsep in strategy):
             raise ValueError(f"strategy '{strategy}' cannot contain path separators.")

        strategy_file = os.path.join(self.standard_strategy_dir, f"{strategy}.json")
        if not os.path.isfile(strategy_file):
            raise FileNotFoundError(
                f"Strategy file not found for strategy '{strategy}': {strategy_file}"
            )

        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON for strategy '{strategy}' from {strategy_file}: {e}"
            )
        except Exception as e:
            raise IOError(f"Error reading strategy file {strategy_file}: {e}")

        if "prompt" not in strategy_data or not isinstance(strategy_data["prompt"], str):
            raise ValueError(
                f"Strategy JSON for '{strategy}' ({strategy_file}) "
                "must contain a 'prompt' key with a string value."
            )

        # Cache the loaded strategy prompt
        strategy_text = strategy_data["prompt"]
        self._strategies[strategy] = strategy_text
        logger.debug(f"Loaded and cached strategy '{strategy}'")
        return strategy_text

    def load(self, promptID_with_strategy: str) -> str:
        """
        Loads the base prompt by ID from configured directories and optionally
        prepends a strategy prompt from the standard strategy directory.

        Searches for `promptID` first in `additional_prompt_dirs`, then in the
        `standard_prompt_dir`. Uses the first match found.

        The input string can be just the `promptID` or `promptID:strategyID`.

        Args:
            promptID_with_strategy: The identifier string, potentially including
                                    a strategy suffix (e.g., "my_prompt" or
                                    "my_prompt:summarize").

        Returns:
            The combined prompt text as a string.

        Raises:
            FileNotFoundError: If the specified `promptID` is not found in any
                               configured directory, or if the specified
                               `strategyID` is not found in the standard
                               strategy directory.
            ValueError: If the input format is wrong or strategy JSON is invalid.
            IOError: If there's an error reading the prompt or strategy file.
        """
        # --- Parse promptID and optional strategy ---
        if not isinstance(promptID_with_strategy, str) or not promptID_with_strategy:
             raise ValueError("Input must be a non-empty string.")
        parts = promptID_with_strategy.split(':', 1)
        promptID = parts[0]
        strategy: Optional[str] = parts[1] if len(parts) > 1 else None

        if not promptID:
             raise ValueError("promptID cannot be empty.")
        if strategy == "": # Disallow empty strategy name like "my_prompt:"
             raise ValueError("Strategy name cannot be empty if ':' separator is used.")

        # --- Load Base Prompt from Cache ---
        prompt_file_path = self._prompts.get(promptID) # Look up in the cache
        if not prompt_file_path:
            # Prompt ID not found during initial scan
            all_dirs = self.additional_prompt_dirs + [self.standard_prompt_dir]
            searched_dirs = [d for d in all_dirs if os.path.isdir(d)] # Only list dirs that actually exist
            raise FileNotFoundError(
                f"Prompt ID '{promptID}' not found in any configured directory during scan. Searched: {searched_dirs}"
            )
        if not os.path.isfile(prompt_file_path):
             # This shouldn't happen if scan worked, but check anyway
             raise FileNotFoundError(
                f"Prompt file for ID '{promptID}' was found during scan but is now missing at: {prompt_file_path}"
             )

        logger.debug(f"Loading base prompt for ID '{promptID}' from: {prompt_file_path}")
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                base_prompt_content = f.read()
        except Exception as e:
            raise IOError(f"Error reading prompt file {prompt_file_path}: {e}")

        # --- Load and Prepend Strategy (if provided) ---
        if strategy:
            logger.debug(f"Loading strategy '{strategy}' for prompt '{promptID}'")
            try:
                strategy_prompt = self._load_strategy_prompt(strategy)
                # Prepend strategy prompt. Adding two newlines for separation.
                combined_prompt = f"{strategy_prompt}\n\n{base_prompt_content}"
                return combined_prompt
            except (FileNotFoundError, ValueError, IOError) as e:
                # Add context to the error and re-raise
                raise type(e)(f"Failed to load strategy '{strategy}' while processing prompt '{promptID}': {e}") from e
        else:
            # No strategy, return only base prompt content
            return base_prompt_content

    def parse_id(self, instruction_string: str) -> Tuple[str, Optional[str]]:
        """Parses an instruction string into promptID and optional strategyID."""
        if not isinstance(instruction_string, str):
             raise ValueError("Input must be a string.")
        parts = instruction_string.split(':', 1)
        promptID = parts[0]
        strategyID: Optional[str] = parts[1] if len(parts) > 1 else None
        # Basic validation
        if not promptID:
             raise ValueError("Prompt ID cannot be empty in instruction string.")
        if strategyID == "":
             raise ValueError("Strategy ID cannot be empty if ':' separator is used.")
        return promptID, strategyID 