# T050: Create AgenticStepProcessor from YAML Configs

## Objective
Extend YAML translator to create AgenticStepProcessor instances from agentic_step configurations in agent instruction chains.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/config/yaml_translator.py` (extend this)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (target class)
- `/home/gyasis/Documents/code/PromptChain/tests/cli/unit/test_yaml_agentic_step_config.py` (tests from T048)

## Requirements

### Implementation Location
`/home/gyasis/Documents/code/PromptChain/promptchain/cli/config/yaml_translator.py`

### Changes Needed

1. **Add AgenticStepProcessor Import**
   ```python
   from promptchain.utils.agentic_step_processor import AgenticStepProcessor
   ```

2. **Extend `_parse_instructions()` Method**
   - Detect `agentic_step` dict in instruction list
   - Extract configuration fields
   - Create AgenticStepProcessor instance
   - Insert into instruction chain

3. **Configuration Field Mapping**
   ```python
   YAML Field              → AgenticStepProcessor Parameter
   ─────────────────────────────────────────────────────────
   objective               → objective (required)
   max_internal_steps      → max_internal_steps (required)
   model_name              → model_name (required, or inherit from agent)
   history_mode            → history_mode (optional, default: "progressive")
   temperature             → temperature (optional, default: 0.7)
   clarification_attempts  → clarification_attempts (optional, default: 1)
   store_detailed_steps    → store_detailed_steps (optional, default: False)
   ```

4. **Model Inheritance Logic**
   ```python
   # If agentic_step.model_name not specified, inherit from parent agent
   if "model_name" not in agentic_config:
       agentic_config["model_name"] = agent_config.get("model", "openai/gpt-4")
   ```

5. **Validation Requirements**
   - Raise ValueError if `objective` missing
   - Raise ValueError if `max_internal_steps` missing or invalid (≤0)
   - Raise ValueError if `history_mode` invalid
   - Provide clear error messages with field name and expected type

### Example YAML Patterns to Support

**Pattern 1: Minimal Config**
```yaml
agents:
  researcher:
    model: "openai/gpt-4"
    instructions:
      - "Prepare: {input}"
      - agentic_step:
          objective: "Research task"
          max_internal_steps: 5
      - "Summarize: {input}"
```

**Pattern 2: Full Config**
```yaml
agents:
  analyst:
    instructions:
      - agentic_step:
          objective: "Complex analysis"
          max_internal_steps: 8
          model_name: "anthropic/claude-3-opus-20240229"
          history_mode: "progressive"
          temperature: 0.7
          clarification_attempts: 2
          store_detailed_steps: true
```

**Pattern 3: Multiple Agentic Steps**
```yaml
agents:
  workflow:
    instructions:
      - "Init: {input}"
      - agentic_step:
          objective: "Phase 1"
          max_internal_steps: 3
      - "Intermediate: {input}"
      - agentic_step:
          objective: "Phase 2"
          max_internal_steps: 5
      - "Final: {input}"
```

### Implementation Approach

```python
def _parse_instructions(self, instructions_config: list, agent_model: str = None) -> list:
    """Parse instruction list, converting agentic_step dicts to AgenticStepProcessor."""
    parsed_instructions = []

    for instruction in instructions_config:
        if isinstance(instruction, dict) and "agentic_step" in instruction:
            # Extract agentic_step config
            agentic_config = instruction["agentic_step"]

            # Validate required fields
            if "objective" not in agentic_config:
                raise ValueError("agentic_step missing required field: objective")
            if "max_internal_steps" not in agentic_config:
                raise ValueError("agentic_step missing required field: max_internal_steps")

            # Model inheritance
            if "model_name" not in agentic_config:
                agentic_config["model_name"] = agent_model or "openai/gpt-4"

            # Create AgenticStepProcessor
            processor = AgenticStepProcessor(
                objective=agentic_config["objective"],
                max_internal_steps=agentic_config["max_internal_steps"],
                model_name=agentic_config["model_name"],
                history_mode=agentic_config.get("history_mode", "progressive"),
                temperature=agentic_config.get("temperature", 0.7),
                clarification_attempts=agentic_config.get("clarification_attempts", 1),
                store_detailed_steps=agentic_config.get("store_detailed_steps", False)
            )
            parsed_instructions.append(processor)

        elif isinstance(instruction, str):
            parsed_instructions.append(instruction)

        else:
            raise TypeError(f"Invalid instruction type: {type(instruction)}")

    return parsed_instructions
```

### Success Criteria
- All T048 unit tests pass
- YAML configs correctly create AgenticStepProcessor instances
- Model inheritance works when model_name omitted
- Validation errors provide clear messages
- No regressions in existing YAML parsing

## Validation
1. Run unit tests: `pytest tests/cli/unit/test_yaml_agentic_step_config.py -v`
2. Manual YAML test with sample config
3. Verify instruction chain construction

## Deliverable
- Extended `yaml_translator.py` with agentic_step support
- All validation and error handling implemented
- Documentation of supported YAML patterns
