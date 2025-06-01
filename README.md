# PromptChain

A flexible prompt engineering library designed to create, refine, and specialize prompts for LLM applications and agent frameworks. PromptChain excels at creating specialized prompts that can be injected into agent frameworks and building sophisticated LLM processing pipelines.

## Primary Use Cases

### 🎯 Agent Framework Integration
Create specialized prompts for agent frameworks like AutoGen, LangChain, and CrewAI:

```python
from promptchain import PromptChain

# Generate optimized prompts for agents
prompt_engineer = PromptChain(
    models=["anthropic/claude-3-sonnet-20240229", "openai/gpt-4"],
    instructions=[
        "Analyze this agent task: {input}",
        "Create a specialized prompt with examples and constraints: {input}"
    ]
)

specialized_prompt = prompt_engineer.process_prompt("Research and summarize AI papers")

# Use with any agent framework
from autogen import AssistantAgent
agent = AssistantAgent(
    name="researcher", 
    system_message=specialized_prompt,
    llm_config={"config_list": [...]}
)
```

### ⚙️ LLM Processing Chains
Chain multiple LLM calls and functions for complex workflows:

```python
# Sequential processing with multiple models
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Extract key information from: {input}",
        "Create detailed analysis: {input}",
        "Generate actionable insights: {input}"
    ]
)

result = chain.process_prompt("Your content here...")
```

### 🔧 Custom Function Integration
Combine LLMs with custom processing functions:

```python
def validate_output(text: str) -> str:
    # Custom validation logic
    return f"Validated: {text}"

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Generate content: {input}",
        validate_output,  # Custom function
        "Finalize based on validation: {input}"
    ]
)
```

## Installation

```bash
# From PyPI (coming soon)
pip install promptchain

# From GitHub
pip install git+https://github.com/gyasis/promptchain.git

# Development installation
git clone https://github.com/gyasis/promptchain.git
cd promptchain
pip install -e .
```

## Quick Start

1. **Set up your environment:**
```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

2. **Simple example:**
```python
from promptchain import PromptChain

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze and improve this text: {input}"]
)

result = chain.process_prompt("Write about quantum computing")
print(result)
```

## Key Features

- **Flexible Chain Construction**: Create sequences of LLM calls and functions
- **Multiple Model Support**: Use different models (GPT-4, Claude, etc.) in the same chain  
- **Function Injection**: Insert custom processing between LLM calls
- **Advanced Prompt Engineering**: Systematic prompt optimization and testing
- **Agent Framework Integration**: Generate specialized prompts for any agent system
- **Async Support**: Full async/await support for modern applications
- **Memory Management**: State persistence and conversation history

## Documentation

📚 **[Complete Documentation Index](docs/index.md)** - Start here for comprehensive guides

### Quick Links
- **[Quick Start Guide](docs/promptchain_quickstart_multistep.md)** - Get up and running in minutes
- **[Complete Framework Guide](docs/promptchain_guide.md)** - Comprehensive usage documentation  
- **[Prompt Engineering Guide](docs/prompt_engineer.md)** - Advanced prompt optimization
- **[Agent Integration Examples](docs/AgentChain_Usage.md)** - Using with agent frameworks
- **[Function Integration](docs/registering_functions_to_chain.md)** - Custom function examples
- **[Advanced Techniques](docs/chainbreaker_guide.md)** - Chain breakers, circular processing, etc.

## Requirements

- Python 3.8+
- litellm
- python-dotenv

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for model integration
- OpenAI and Anthropic for LLM APIs