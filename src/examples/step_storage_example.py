from src.utils.promptchaining import PromptChain

def step_storage_example():
    """Example showing step output storage without full history"""
    
    chain = PromptChain(
        models=[
            "openai/gpt-4",
            "anthropic/claude-3-sonnet-20240229"
        ],
        instructions=[
            "Write initial analysis of: {input}",
            "Create detailed summary of: {input}"
        ],
        full_history=False,  # Don't return full history
        store_steps=True     # But store steps for later access
    )

    # Process the prompt - only get final result
    final_result = chain.process_prompt("AI impact on healthcare")
    print("Final Result:", final_result)

    # Later, access specific step outputs if needed
    initial_analysis = chain.get_step_output(1)
    print("\nInitial Analysis:", initial_analysis['output'])
    
    detailed_summary = chain.get_step_output(2)
    print("\nDetailed Summary:", detailed_summary['output'])

    # Access all stored steps
    print("\nAll Steps:")
    for step_num in range(len(chain.step_outputs)):
        step_data = chain.get_step_output(step_num)
        print(f"\nStep {step_num}:")
        print(f"Type: {step_data['type']}")
        print(f"Output: {step_data['output'][:100]}...")  # Show first 100 chars 