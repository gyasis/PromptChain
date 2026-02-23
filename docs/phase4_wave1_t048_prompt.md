# T048: Unit Test - AgenticStepProcessor YAML Config Translation

## Objective
Create unit tests verifying YAML translator correctly converts agentic_step configurations to AgenticStepProcessor instances.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/config/yaml_translator.py` (translator to extend)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (target class)
- `/home/gyasis/Documents/code/PromptChain/tests/cli/unit/test_yaml_agentic_config.py` (existing pattern)

## Requirements

### Test File Location
`/home/gyasis/Documents/code/PromptChain/tests/cli/unit/test_yaml_agentic_step_config.py`

### Test Cases

1. **test_agentic_step_minimal_config**
   ```yaml
   agentic_step:
     objective: "Research and analyze"
     max_internal_steps: 5
     model_name: "openai/gpt-4"
   ```
   - Verify AgenticStepProcessor instance created
   - Assert required fields populated correctly
   - Validate default values for optional fields

2. **test_agentic_step_full_config**
   ```yaml
   agentic_step:
     objective: "Complex multi-hop reasoning"
     max_internal_steps: 8
     model_name: "anthropic/claude-3-opus-20240229"
     history_mode: "progressive"
     temperature: 0.7
     clarification_attempts: 2
     store_detailed_steps: true
   ```
   - Verify all fields translated correctly
   - Assert optional parameters set properly
   - Validate complex configuration handling

3. **test_agentic_step_in_instruction_chain**
   ```yaml
   agents:
     researcher:
       instructions:
         - "Prepare research strategy: {input}"
         - agentic_step:
             objective: "Conduct research"
             max_internal_steps: 6
             model_name: "openai/gpt-4"
         - "Synthesize findings: {input}"
   ```
   - Verify mixed instruction chain parsing
   - Assert agentic_step converted to AgenticStepProcessor
   - Validate instruction order preserved

4. **test_agentic_step_validation_errors**
   - Missing required field (objective) → ValueError
   - Invalid max_internal_steps (0, negative) → ValueError
   - Invalid model_name (empty string) → ValueError
   - Invalid history_mode → ValueError

5. **test_agentic_step_default_model_inheritance**
   ```yaml
   agents:
     researcher:
       model: "openai/gpt-4"
       instructions:
         - agentic_step:
             objective: "Research task"
             max_internal_steps: 5
             # model_name omitted - should inherit from agent
   ```
   - Verify model inheritance when not specified
   - Assert AgenticStepProcessor uses agent's model

### Success Criteria
- All 5 test cases pass
- Invalid configs raise ValueError with clear messages
- Tests run in <3 seconds
- YAML translator handles all edge cases

### Example Test Structure
```python
import pytest
import yaml
from promptchain.cli.config.yaml_translator import YAMLTranslator
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

def test_agentic_step_minimal_config():
    """YAML translator creates AgenticStepProcessor from minimal config."""
    yaml_config = """
    agents:
      test_agent:
        model: "openai/gpt-4"
        instructions:
          - agentic_step:
              objective: "Test objective"
              max_internal_steps: 3
              model_name: "openai/gpt-4"
    """

    config = yaml.safe_load(yaml_config)
    translator = YAMLTranslator(config)
    agent_chain = translator.build_agent_chain()

    # Extract AgenticStepProcessor from agent instructions
    agent = agent_chain.agents["test_agent"]
    processor = agent.instructions[0]

    assert isinstance(processor, AgenticStepProcessor)
    assert processor.objective == "Test objective"
    assert processor.max_internal_steps == 3
    assert processor.model_name == "openai/gpt-4"
```

## Validation
Run: `pytest tests/cli/unit/test_yaml_agentic_step_config.py -v`
Expected: 5 passed in <3s

## Deliverable
- Test file with 5 passing unit tests
- Validation for all config combinations
- Clear error messages for invalid configs
