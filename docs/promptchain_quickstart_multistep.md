---
noteId: "52c288303ea211f0a612f1c6d62fea30"
tags: []

---

# PromptChain: Multi-Step Chain Quickstart

PromptChain enables you to build a sequence of processing steps—each can be a prompt template or a Python function. The output of each step is passed as input to the next, allowing for complex, composable workflows.

---

## Multi-Step Flow Example

Suppose you want to:
1. Summarize user input.
2. Extract keywords from the summary.
3. Generate a follow-up question based on those keywords.
4. Format the final output.

Here's how you can do it:

```python
from promptchain.utils.promptchaining import PromptChain

# Step 1: Summarize the input
summarize_prompt = "Summarize the following text in one sentence:\n{input}"

# Step 2: Extract keywords
keywords_prompt = "Extract 3 keywords from this summary:\n{input}"

# Step 3: Generate a follow-up question
question_prompt = "Write a follow-up question based on these keywords:\n{input}"

# Step 4: Format the output (as a function)
def format_output(text: str) -> str:
    return f"Here is your follow-up question:\n{text}"

# Create the PromptChain with 4 steps
chain = PromptChain(
    models=[{
        "name": "ollama/mistral-small:22B",
        "params": {"api_base": "http://localhost:4000/v1"}
    }],
    instructions=[
        summarize_prompt,      # Step 1: Summarize
        keywords_prompt,       # Step 2: Extract keywords
        question_prompt,       # Step 3: Generate question
        format_output          # Step 4: Format output (function)
    ],
    full_history=True,
    store_steps=True,
    verbose=False
)

# Run the chain
user_input = "The Eiffel Tower is one of the most famous landmarks in Paris, attracting millions of visitors every year."
result = chain.process_prompt(user_input)
print(result)
```

---

## How the Flow Works

1. **Summarize**: The user input is summarized.
2. **Extract Keywords**: The summary is used to extract keywords.
3. **Generate Question**: A follow-up question is generated from the keywords.
4. **Format Output**: The final output is formatted by a Python function.

---

## Output Example

```
Here is your follow-up question:
What makes the Eiffel Tower a popular tourist attraction in Paris?
```

---

**Tip:**  
You can mix and match prompt templates and Python functions in any order. Each step receives the previous step's output as its input. 
noteId: "203b1a90374911f0a58c35ceb4dbead4"
tags: []

---

 