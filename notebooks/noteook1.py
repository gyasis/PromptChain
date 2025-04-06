# %%
from promptchain import PromptChain

# %%
from promptchain.utils.prompt_engineer import PromptEngineer

# Initialize engineer
engineer = PromptEngineer(
    max_iterations=3,
    use_human_evaluation=False,
    verbose=True
)

# Create specialized prompt
task = """Create a prompt for an AI agent that helps users analyze financial data.
The agent should:
1. Extract key metrics
2. Identify trends
3. Provide actionable insights
4. Format output as a structured report"""

optimized_prompt = engineer.create_specialized_prompt(task)
# %%
from promptchain.utils.prompt_engineer import PromptEngineer

# Initialize engineer
engineer = PromptEngineer(
    max_iterations=3,
    use_human_evaluation=False,
    verbose=True
)

# Create specialized prompt
task = """Create a chain of thought methodology where the agent whill do internal thinking recongnitoin of the task, taskes to comple and working through thelist ainternal and then output the respos
"""

optimized_prompt = engineer.create_specialized_prompt(task)
# %%
