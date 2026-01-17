# T049: Implement Instruction Chain Processing for AgenticStepProcessor

## Objective
Modify `_build_instruction_chain()` method in PromptChain to properly execute AgenticStepProcessor instances within instruction chains.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/promptchaining.py` (modify this)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (integrate with)
- `/home/gyasis/Documents/code/PromptChain/tests/cli/contract/test_agentic_instruction_chain.py` (contract tests from T045)

## Current State Analysis

### Existing Instruction Processing (lines ~500-600)
```python
async def _build_instruction_chain(self, instructions):
    """Process instruction chain: strings → LLM, functions → execute, ..."""
    for instruction in instructions:
        if isinstance(instruction, str):
            # LLM call
            result = await self._call_llm(instruction)
        elif callable(instruction):
            # Function execution
            result = instruction(self.current_input)
        # Need to add: AgenticStepProcessor handling
```

## Requirements

### Changes to `promptchain/utils/promptchaining.py`

1. **Add AgenticStepProcessor Import**
   ```python
   from promptchain.utils.agentic_step_processor import AgenticStepProcessor
   ```

2. **Extend `_build_instruction_chain()` Method**

   Add handling for AgenticStepProcessor instances:

   ```python
   async def _build_instruction_chain(self, instructions):
       """Process instruction chain with AgenticStepProcessor support."""
       results = []

       for idx, instruction in enumerate(instructions):
           if isinstance(instruction, str):
               # Existing: LLM call
               result = await self._execute_llm_instruction(instruction)
               results.append(result)

           elif callable(instruction) and not isinstance(instruction, AgenticStepProcessor):
               # Existing: Function call
               result = instruction(self.current_input)
               results.append(result)

           elif isinstance(instruction, AgenticStepProcessor):
               # NEW: AgenticStepProcessor execution
               result = await self._execute_agentic_step(instruction, idx)
               results.append(result)

           else:
               raise TypeError(f"Invalid instruction type at index {idx}: {type(instruction)}")

       return results
   ```

3. **Add `_execute_agentic_step()` Helper Method**

   ```python
   async def _execute_agentic_step(
       self,
       processor: AgenticStepProcessor,
       step_index: int
   ) -> str:
       """Execute AgenticStepProcessor within instruction chain.

       Args:
           processor: AgenticStepProcessor instance
           step_index: Index in instruction chain (for logging)

       Returns:
           Final synthesis result from processor
       """
       if self.verbose:
           print(f"\n{'='*60}")
           print(f"AGENTIC STEP {step_index + 1}: {processor.objective}")
           print(f"Max Internal Steps: {processor.max_internal_steps}")
           print(f"{'='*60}\n")

       # Pass current input to processor
       # Processor handles multi-hop reasoning internally
       result = await processor.process_async(self.current_input)

       # Update current_input for next instruction
       self.current_input = result

       # Store detailed steps if enabled
       if processor.store_detailed_steps and hasattr(processor, 'step_history'):
           if not hasattr(self, 'agentic_step_details'):
               self.agentic_step_details = []
           self.agentic_step_details.append({
               'step_index': step_index,
               'objective': processor.objective,
               'internal_steps': processor.step_history,
               'final_result': result
           })

       return result
   ```

4. **Tool Sharing with AgenticStepProcessor**

   Ensure processor can access chain's registered tools:

   ```python
   async def _execute_agentic_step(self, processor, step_index):
       """Execute agentic step with tool access."""
       # Share registered tools with processor
       if self.registered_functions:
           processor.register_tools(self.registered_functions)

       # Share tool schemas
       if self.tools:
           processor.add_tool_schemas(self.tools)

       # Execute processor
       result = await processor.process_async(self.current_input)
       return result
   ```

5. **Progress Callback Support**

   Add callback mechanism for TUI progress display:

   ```python
   def set_agentic_progress_callback(self, callback):
       """Set callback for agentic step progress updates.

       Callback signature: callback(step_num: int, total_steps: int, message: str)
       """
       self.agentic_progress_callback = callback

   async def _execute_agentic_step(self, processor, step_index):
       """Execute with progress callbacks."""
       # Set processor callback to forward to chain callback
       if hasattr(self, 'agentic_progress_callback'):
           processor.set_progress_callback(self.agentic_progress_callback)

       result = await processor.process_async(self.current_input)
       return result
   ```

### Integration Points

**With ExecutionHistoryManager** (T053):
- Log agentic step start/completion
- Track internal reasoning steps
- Include in conversation history

**With TUI** (T052):
- Progress callback triggers widget updates
- Stream internal step outputs
- Display completion status

**With Error Handling** (T055):
- Catch max_steps exhaustion
- Handle tool call failures
- Propagate errors to chain level

### Success Criteria
- AgenticStepProcessor executes within instruction chains
- Results flow correctly to next instruction
- Tools shared between chain and processor
- Progress callbacks work (for TUI integration)
- No regressions in existing instruction processing
- All T045 contract tests pass

## Validation

1. **Unit Test**: Direct chain execution
   ```python
   processor = AgenticStepProcessor(
       objective="Test reasoning",
       max_internal_steps=3,
       model_name="openai/gpt-4"
   )

   chain = PromptChain(
       models=["openai/gpt-4"],
       instructions=[
           "Prepare: {input}",
           processor,
           "Finalize: {input}"
       ]
   )

   result = await chain.process_prompt_async("Test input")
   assert result is not None
   ```

2. **Run Contract Tests**: `pytest tests/cli/contract/test_agentic_instruction_chain.py -v`

3. **Integration Test**: Full workflow with tools
   ```python
   # T046 will test this
   ```

## Deliverable
- Modified `promptchaining.py` with AgenticStepProcessor support
- Progress callback mechanism implemented
- Tool sharing between chain and processor
- All contract tests passing
