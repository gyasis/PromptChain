# T045: Contract Test - Instruction Chain Validation

## Objective
Create contract test ensuring instruction chains with AgenticStepProcessor instances maintain proper type contracts and validation.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (existing processor)
- `/home/gyasis/Documents/code/PromptChain/tests/cli/contract/test_instruction_chain_contract.py` (reference pattern)
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/config/yaml_translator.py` (YAML translator)

## Requirements

### Test File Location
`/home/gyasis/Documents/code/PromptChain/tests/cli/contract/test_agentic_instruction_chain.py`

### Test Cases
1. **test_instruction_chain_accepts_agentic_step_processor**
   - Verify instruction list can contain AgenticStepProcessor instances
   - Validate mixed instructions (strings + AgenticStepProcessor + functions)
   - Assert no type errors during chain construction

2. **test_agentic_step_processor_config_contract**
   - Validate required fields: objective, max_internal_steps, model_name
   - Verify optional fields: history_mode, tool_access, temperature
   - Assert proper defaults when fields omitted

3. **test_instruction_chain_type_validation**
   - Test invalid instruction types (int, dict, etc.) raise TypeError
   - Verify AgenticStepProcessor instance validation
   - Assert clear error messages for contract violations

4. **test_agentic_step_processor_in_promptchain**
   - Create PromptChain with AgenticStepProcessor in instructions
   - Validate chain accepts processor without errors
   - Assert processor maintains internal state isolation

### Success Criteria
- All 4 test cases pass
- Contract violations raise TypeError with descriptive messages
- Tests run in <5 seconds
- 100% code coverage for instruction validation logic

### Example Test Structure
```python
import pytest
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.promptchaining import PromptChain

def test_instruction_chain_accepts_agentic_step_processor():
    """Contract: Instruction chains must accept AgenticStepProcessor instances."""
    processor = AgenticStepProcessor(
        objective="Test objective",
        max_internal_steps=3,
        model_name="openai/gpt-4"
    )

    instructions = [
        "First prompt: {input}",
        processor,  # AgenticStepProcessor instance
        "Final prompt: {input}"
    ]

    # Should not raise TypeError
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=instructions,
        verbose=False
    )

    assert len(chain.instructions) == 3
    assert isinstance(chain.instructions[1], AgenticStepProcessor)
```

## Validation
Run: `pytest tests/cli/contract/test_agentic_instruction_chain.py -v`
Expected: 4 passed in <5s

## Deliverable
- Test file with 4 passing contract tests
- Clear error messages for contract violations
- Documentation of instruction chain type contracts
