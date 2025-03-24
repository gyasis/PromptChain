# Web Search Tutorial: Dynamic Head-Tail Injection

This tutorial demonstrates how to use the DynamicChainBuilder with head-tail technique to decompose a single search query into multiple search questions, execute them, and then synthesize the results.

## Overview

We'll build a dynamic search system that:

1. Takes a complex search query
2. Uses a head chain to decompose it into multiple search questions
3. Injects function-based tail chains for each question
4. Conducts web searches for each question
5. Synthesizes results into a final summary

## Step 1: Set Up Dependencies

```python
import requests
from typing import List, Dict
import json
from promptchain.utils.promptchaining import DynamicChainBuilder, PromptChain

# Simple web search function using a search API
def web_search(query: str, num_results: int = 3) -> Dict:
    """
    Perform a web search using a search API.
    Returns search results as a dictionary.
    """
    try:
        # Replace with your preferred search API
        api_key = "YOUR_SEARCH_API_KEY" 
        search_url = "https://api.searchapi.com/v1/search"
        
        params = {
            "api_key": api_key,
            "q": query,
            "num": num_results
        }
        
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Search failed with status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
```

## Step 2: Create Search Result Processor

```python
def process_search_results(results: Dict) -> str:
    """
    Process and format search results into readable text.
    """
    if "error" in results:
        return f"Search error: {results['error']}"
    
    if "results" not in results or not results["results"]:
        return "No results found."
    
    formatted = []
    for i, result in enumerate(results["results"], 1):
        title = result.get("title", "No title")
        snippet = result.get("snippet", "No description")
        url = result.get("link", "No URL")
        
        formatted.append(f"Result {i}:\nTitle: {title}\nSummary: {snippet}\nURL: {url}\n")
    
    return "\n".join(formatted)
```

## Step 3: Create Search Function for Chain Injection

```python
def create_search_function(memory_namespace: str = "search_results"):
    """
    Creates a search function that can be injected into a chain.
    Stores results in memory for later retrieval.
    """
    def search_function(query: str, builder: DynamicChainBuilder = None) -> str:
        """
        Executes a web search and returns formatted results.
        Optionally stores results in builder's memory if provided.
        """
        # Clean the query
        query = query.strip()
        if query.lower().startswith("search for:"):
            query = query[len("search for:"):].strip()
            
        # Perform search
        results = web_search(query)
        formatted_results = process_search_results(results)
        
        # Store in memory if builder provided
        if builder:
            search_key = f"search_{hash(query) % 10000}"
            builder.store_memory(search_key, formatted_results, namespace=memory_namespace)
            return f"Completed search for: {query}\nResults stored with key: {search_key}\n\n{formatted_results}"
        
        return formatted_results
    
    return search_function
```

## Step 4: Create Base Templates

```python
# Template for head chain (question decomposition)
HEAD_TEMPLATE = """
Task Decomposition Framework
============================
Your task: {instruction}
Input context: {input}

Guidelines:
- Break down the complex query into 5 specific search questions
- Each question should explore a different aspect
- Questions should be clear and specific for web searches
- Format as a numbered list
"""

# Template for synthesis chain
SYNTHESIS_TEMPLATE = """
Information Synthesis Framework
==============================
Your task: {instruction}
Research findings: {input}

Guidelines:
- Synthesize all search results into a comprehensive answer
- Highlight key insights from different sources
- Address contradictions or different perspectives
- Provide a balanced, well-structured response
"""
```

## Step 5: Build the Dynamic Search System

```python
def build_search_system():
    """
    Build a complete search system using head-tail technique.
    """
    # Create builder with head-tail technique
    builder = DynamicChainBuilder(
        base_model="openai/gpt-4",
        base_instruction=HEAD_TEMPLATE,
        technique="head-tail"
    )
    
    # Create search function
    search_func = create_search_function(memory_namespace="search_results")
    
    # Create head chain for decomposing the question
    head_chain = builder.create_chain(
        "question_decomposer",
        """Analyze the complex search query and break it down into 5 specific search questions.
        Each question should focus on a different aspect of the overall query.
        Ensure questions are specific enough for web search engines.
        Return the questions as a numbered list.""",
        execution_mode="serial",
        group="search_system"
    )
    
    # Create the summary chain
    summary_chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Synthesize the following search results into a comprehensive answer: {input}"
        ],
        store_steps=True
    )
    
    # Register summary chain
    builder.chain_registry["summarizer"] = {
        "chain": summary_chain,
        "execution_mode": "serial",
        "group": "search_system",
        "dependencies": ["question_decomposer"],
        "status": "created"
    }
    
    # Function to dynamically inject search functions based on generated questions
    def inject_search_functions(questions_output: str) -> None:
        """
        Parse the list of questions and inject a search function for each one.
        """
        # Parse questions from output
        questions = []
        for line in questions_output.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.lower().startswith(("- ", "* "))):
                # Remove leading numbers or bullet points
                cleaned = line.lstrip("0123456789.-* ")
                questions.append(cleaned)
        
        # Limit to 5 questions
        questions = questions[:5]
        
        # Create and inject search function chain for each question
        for i, question in enumerate(questions):
            # Create a custom search function for this specific question
            def make_search_function(q):
                return lambda x: search_func(q)
            
            # Create the search chain with the function
            search_chain_id = f"search_{i+1}"
            search_chain = PromptChain(
                models=["openai/gpt-4"],  # Won't be used for function step
                instructions=[
                    f"Search for: {question}",
                    make_search_function(question)  # Inject the search function
                ],
                store_steps=True
            )
            
            # Register the search chain
            builder.chain_registry[search_chain_id] = {
                "chain": search_chain,
                "execution_mode": "parallel",
                "group": "search_system",
                "dependencies": ["question_decomposer"],
                "status": "created"
            }
            
            # Update summarizer dependencies to include this search
            builder.chain_registry["summarizer"]["dependencies"].append(search_chain_id)
    
    return builder, inject_search_functions
```

## Step 6: Execute the Search System

```python
async def execute_search_system(query: str):
    """
    Execute the complete search system with the given query.
    """
    # Build the search system
    builder, inject_search_functions = build_search_system()
    
    # Execute the head chain to decompose the question
    questions_output = await builder.execute_chain("question_decomposer", query)
    print("Generated search questions:")
    print(questions_output)
    
    # Inject search function chains based on generated questions
    inject_search_functions(questions_output)
    
    # Execute all search chains in parallel
    async def parallel_executor(chain_inputs):
        """Custom parallel executor for search queries"""
        import asyncio
        
        async def execute_single(chain_id, input_data):
            # Execute chain synchronously but in parallel tasks
            return chain_id, await asyncio.to_thread(builder.execute_chain, chain_id, input_data)
        
        # Create tasks for all chains
        tasks = [
            asyncio.create_task(execute_single(chain_id, input_data))
            for chain_id, input_data in chain_inputs
        ]
        
        # Gather results
        results = await asyncio.gather(*tasks)
        return dict(results)
    
    # Find all search chains
    search_chains = [
        chain_id for chain_id in builder.chain_registry
        if chain_id.startswith("search_")
    ]
    
    # Execute search chains in parallel
    search_inputs = [(chain_id, query) for chain_id in search_chains]
    search_results = await parallel_executor(search_inputs)
    
    # Combine all search results for the summarizer
    combined_results = "\n\n".join([
        f"Results for Question {i+1}:\n{result}"
        for i, (_, result) in enumerate(sorted(search_results.items()))
    ])
    
    # Execute the summarizer chain
    final_summary = await builder.execute_chain("summarizer", combined_results)
    
    return {
        "questions": questions_output,
        "search_results": search_results,
        "summary": final_summary
    }
```

## Step 7: Example Usage

```python
import asyncio

async def main():
    # Complex search query that needs decomposition
    complex_query = """
    What are the environmental and economic impacts of electric vehicles compared 
    to internal combustion engines, considering manufacturing, usage, and disposal?
    """
    
    results = await execute_search_system(complex_query)
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(results["summary"])

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

## Example Output

The system will generate 5 search questions like:

```
1. What are the environmental impacts of manufacturing electric vehicles vs. internal combustion engines?
2. How does the carbon footprint of operating electric vehicles compare to gasoline cars over their lifetime?
3. What economic costs are associated with electric vehicle production compared to traditional vehicles?
4. How do disposal and recycling processes differ between electric and internal combustion vehicles?
5. What are the total cost of ownership differences between electric and gasoline vehicles?
```

Each question will be searched individually, and the results will be synthesized into a comprehensive final summary.

## Key Components Explained

### Head Chain (Question Decomposition)
The head chain analyzes a complex query and breaks it down into multiple specific search questions.

### Function-Based Tail Chains
Each search question gets its own function-based chain that:
1. Takes the question
2. Executes a web search
3. Formats the results

### Parallel Execution
Search functions are executed in parallel for efficiency.

### Dynamic Injection
The system dynamically:
1. Generates questions with the head chain
2. Creates and injects custom search function chains
3. Updates dependencies for proper execution flow

### Final Synthesis
A summarizer chain takes all search results and synthesizes them into a comprehensive answer.

## Advanced Options

### Memory Integration
```python
# Store search results in memory
builder.store_memory("last_search_query", complex_query)
builder.store_memory("last_search_summary", results["summary"])

# Create a memory-enabled search chain
memory_chain = builder.create_memory_chain(
    "search_memory",
    namespace="search_history"
)
```

### Custom Search APIs
The web search function can be replaced with any API:
- Google Custom Search
- Bing Search
- DuckDuckGo
- Specialized APIs (academic, news, etc.)

### Result Filtering
Add filtering steps to improve result quality:
```python
def filter_search_results(results: Dict, min_relevance: float = 0.5) -> Dict:
    """Filter search results by relevance score"""
    if "results" in results:
        results["results"] = [
            r for r in results["results"] 
            if r.get("relevance_score", 0) >= min_relevance
        ]
    return results
``` 
noteId: "4e897f90088211f09aa1cf0704e2b01f"
tags: []

---

 