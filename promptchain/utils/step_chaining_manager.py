"""
Step Chaining Manager for MCP Tool Hijacker
===========================================

This module manages step-to-step data flow in PromptChain, enabling
dynamic parameter passing from previous step outputs to hijacker tools.

Key Features:
- Store step outputs for later use
- Create template variables from step outputs
- Parse JSON outputs for key extraction
- Support both indexed (step_1) and named (search_step) references
"""

from typing import Any, Dict, List, Optional
import json
from .json_output_parser import JSONOutputParser, CommonExtractions


class StepChainingManager:
    """
    Manages step outputs and template variable creation for PromptChain.
    
    Enables dynamic parameter passing between steps using template syntax:
    - {previous.results[0].id} - from immediately previous step
    - {step_1.metadata.title} - from specific step number  
    - {search_step.first_result} - from named step
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize step chaining manager.
        
        Args:
            verbose: Enable debug logging
        """
        self.verbose = verbose
        self.json_parser = JSONOutputParser(verbose=verbose)
        
        # Storage for step outputs
        self.step_outputs: Dict[int, Any] = {}  # step_index -> output
        self.named_outputs: Dict[str, Any] = {}  # step_name -> output
        
        # Track current step for "previous" references
        self.current_step = 0
    
    def store_step_output(self, step_index: int, output: Any, step_name: Optional[str] = None):
        """
        Store output from a completed step.
        
        Args:
            step_index: Index of the step (1-based to match PromptChain)
            output: The step's output data
            step_name: Optional name for the step
        """
        self.step_outputs[step_index] = output
        
        if step_name:
            self.named_outputs[step_name] = output
        
        if self.verbose:
            print(f"📦 Stored step {step_index} output{' (' + step_name + ')' if step_name else ''}: {type(output)}")
    
    def create_template_vars_for_step(self, current_step_index: int) -> Dict[str, Any]:
        """
        Create template variables available for the current step.
        
        Args:
            current_step_index: Index of the current step being processed
            
        Returns:
            Dictionary of template variables for hijacker use
        """
        template_vars = {}
        
        # Add "previous" - output from immediately previous step
        if current_step_index > 1:
            previous_output = self.step_outputs.get(current_step_index - 1)
            if previous_output is not None:
                template_vars.update(
                    CommonExtractions.create_template_vars(previous_output, "previous")
                )
        
        # Add numbered step references: step_1, step_2, etc.
        for step_num, output in self.step_outputs.items():
            if step_num < current_step_index:  # Only previous steps
                step_vars = CommonExtractions.create_template_vars(output, f"step_{step_num}")
                template_vars.update(step_vars)
        
        # Add named step references
        for step_name, output in self.named_outputs.items():
            name_vars = CommonExtractions.create_template_vars(output, step_name)
            template_vars.update(name_vars)
        
        if self.verbose and template_vars:
            print(f"🔗 Created template vars for step {current_step_index}: {list(template_vars.keys())}")
        
        return template_vars
    
    def parse_step_output_for_chaining(self, output: Any, parse_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Parse step output and extract values for easier chaining.
        
        Args:
            output: Raw step output
            parse_config: Optional parsing configuration
            
        Returns:
            Processed output (enhanced with extracted values if configured)
        """
        if parse_config is None:
            return output
        
        # If output is a string that looks like JSON, try to parse it
        parsed_output = output
        if isinstance(output, str) and output.strip().startswith(('{', '[')):
            try:
                parsed_output = json.loads(output)
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON
        
        # Apply extraction rules from parse_config
        if isinstance(parse_config, dict) and "extractions" in parse_config:
            extractions = parse_config["extractions"]
            if isinstance(extractions, dict):
                extracted = self.json_parser.extract_multiple(
                    parsed_output, 
                    extractions, 
                    parse_config.get("defaults", {})
                )
                
                # Enhance output with extracted values
                if isinstance(parsed_output, dict):
                    parsed_output["_extracted"] = extracted
                else:
                    # Wrap non-dict outputs
                    parsed_output = {
                        "original": parsed_output,
                        "_extracted": extracted
                    }
        
        return parsed_output
    
    def get_step_output(self, step_reference: str) -> Any:
        """
        Get output from a specific step using various reference formats.
        
        Args:
            step_reference: Reference format (e.g., "previous", "step_1", "search_step")
            
        Returns:
            Step output or None if not found
        """
        if step_reference == "previous":
            if self.current_step > 1:
                return self.step_outputs.get(self.current_step - 1)
        elif step_reference.startswith("step_"):
            try:
                step_num = int(step_reference[5:])
                return self.step_outputs.get(step_num)
            except ValueError:
                pass
        else:
            # Named step
            return self.named_outputs.get(step_reference)
        
        return None
    
    def clear_outputs(self):
        """Clear all stored step outputs."""
        self.step_outputs.clear()
        self.named_outputs.clear()
        self.current_step = 0
    
    def get_available_references(self) -> Dict[str, List[str]]:
        """Get list of available step references for debugging."""
        return {
            "numbered_steps": [f"step_{num}" for num in self.step_outputs.keys()],
            "named_steps": list(self.named_outputs.keys()),
            "previous_available": self.current_step > 1
        }


# Convenience functions for common chaining patterns
def create_deeplake_extraction_config() -> Dict[str, Any]:
    """Create common extraction config for DeepLake responses."""
    return {
        "extractions": {
            "first_id": "results[0].id",
            "first_text": "results[0].text", 
            "first_title": "results[0].metadata.title",
            "result_count": "results",
            "all_ids": "results",  # Will need custom processing
        },
        "defaults": {
            "first_id": None,
            "first_text": "",
            "first_title": "Unknown",
            "result_count": [],
        }
    }


def create_template_example_for_hijacker() -> Dict[str, str]:
    """Example template configuration for hijacker tools."""
    return {
        # Use previous step's first result ID as document_id parameter
        "document_id": "{previous_first_id}",
        
        # Use previous step's result count to set n_results
        "n_results": "{previous.results|5}",  # Default to 5 if missing
        
        # Use a specific step's output
        "query": "{search_step.query}",
        
        # Combine values
        "title": "Analysis of: {previous_first_title}"
    }