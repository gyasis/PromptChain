# Registering Functions to the PromptChain

This document provides a detailed explanation of how to register functions to the `PromptChain` for both tool calling and function injection. It covers the structure of the functions, the necessary steps for registration, and examples to illustrate the process.

## Function Structure

### Tool Functions
Tool functions are Python functions that can be called by the LLM during the execution of a prompt chain. These functions must:
- Accept specific parameters as defined in their corresponding tool schema.
- Return a string output that the LLM can use to generate a response.

#### Example Tool Function
```python
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Gets the current weather for a specified location.

    Args:
        location: The city and state/country (e.g., "San Francisco, CA").
        unit: The temperature unit ("celsius" or "fahrenheit"). Defaults to "celsius".

    Returns:
        A string describing the current weather conditions.
    """
    # Example implementation
    return f"The weather in {location} is sunny with a temperature of 25°C."
```

### Function Injection
Function injection involves directly calling a Python function within the prompt chain without LLM intervention. These functions are typically used for specific, predefined tasks.

#### Example Function Injection
```python
def calculate_sum(a: int, b: int) -> int:
    """
    Calculates the sum of two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b
```

## Steps to Register Functions to the PromptChain

### 1. Define the Tool Schema
Create a tool schema that describes the function's purpose, parameters, and any other relevant details. This schema is used by the LLM to understand when and how to call the function.

#### Example Tool Schema
```python
weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather conditions for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state or country, e.g., San Francisco, CA or London, UK",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
```

### 2. Register the Tool Function
Use the `register_tool_function` method of the `PromptChain` to register the Python function that implements the tool. The function's `__name__` must match the `name` in the tool schema.

#### Example Registration
```python
chain = PromptChain(models=["openai/gpt-4o"], instructions=["..."])
chain.add_tools([weather_tool_schema])
chain.register_tool_function(get_current_weather)
```

### 3. Execute the Prompt Chain
Run the prompt chain with an input that may trigger the tool call. The LLM will decide when to call the registered functions based on the input and the tool schemas.

#### Example Execution
```python
initial_input = "What's the weather like in London right now?"
final_output = chain.process_prompt(initial_input)
print(final_output)
```

## Conclusion
By following these steps, you can effectively register and utilize functions within the `PromptChain`. This allows for dynamic and context-aware decision-making by the LLM, enhancing the flexibility and functionality of your prompt chains. 
noteId: "944f409014ea11f0a459a9c802cb6e4a"
tags: []

---

 