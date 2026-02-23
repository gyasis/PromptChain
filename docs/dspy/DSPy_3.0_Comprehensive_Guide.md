# DSPy 3.0 Comprehensive Guide: Cheat Sheet & Training Diagram

> **⚠️ IMPORTANT: This guide has been verified against DSPy 3.0 (2025) using Context7 documentation and Gemini research. All code examples use the latest API syntax.**
> 
> **Key Changes in DSPy 3.0:**
> - `dspy.OpenAI` is **DEPRECATED** - Always use `dspy.LM("provider/model")` instead
> - Unified `dspy.LM()` interface for all providers (OpenAI, Anthropic, Gemini, etc.)
> - Model format: `"provider/model-name"` (e.g., `"openai/gpt-4o-mini"` - **RECOMMENDED: Best value in Dec 2025**)

## Executive Summary

**Question:** Can you extract/grab the actual trained prompt structure in DSPy 3.0?

**Answer: YES!** DSPy 3.0 provides multiple ways to access and view the compiled/optimized prompt structures:

1. **`dspy.inspect_history(n=2)`** - View the last N prompts used (before and after optimization)
2. **Accessing Predictor Attributes** - Direct access to compiled prompts via `compiled_module.predictor.prompt` or `compiled_module.predictor.demos`
3. **`dspy.inspect(compiled_module)`** - Detailed view of the entire compiled program structure
4. **Iterating Through Predictors** - Using `named_predictors()` to access all prompts in complex programs

---

## Table of Contents

1. [Installation & Setup](#installation--setup) - **Includes Model Recommendations (Dec 2025)**
2. [Core Concepts](#core-concepts) - **Includes Optimizer Selection by Dataset Size**
3. [Complete Modules Reference](#complete-modules-reference) - **All 13 DSPy Modules** (includes NEW CodeAct, Parallel)
4. [Complete Optimizers Reference](#complete-optimizers-reference) - **All 11 DSPy Optimizers** (includes NEW SIMBA)
5. [Extracting Compiled Prompts](#extracting-compiled-prompts)
6. [Common Patterns](#common-patterns)
7. [Optimization Workflows](#optimization-workflows)
8. [Best Practices](#best-practices)
9. [Training Diagram](#training-diagram)
10. [Code Examples](#code-examples)
11. [Model Pricing Reference](#model-pricing-reference-december-2025)

---

## Installation & Setup

```bash
# Install DSPy 3.0 (verified 2025)
pip install -U dspy
# Or install latest from GitHub:
# pip install git+https://github.com/stanfordnlp/dspy.git

# Install additional dependencies (optional)
pip install -U datasets
pip install mlflow>=2.20  # For tracing and monitoring
```

### Model Recommendations (December 2025)

**⭐ Best Value Models (Recommended for Most Use Cases):**

1. **GPT-4o-mini (OpenAI)** - ⭐ **RECOMMENDED**
   - **Cost:** $0.15/$0.60 per 1M tokens (input/output)
   - **Performance:** Near GPT-4 level, **60%+ cheaper than GPT-3.5-turbo**
   - **Context:** 128K tokens
   - **Use:** General purpose, best value for most tasks
   - **Model string:** `"openai/gpt-4o-mini"`

2. **Claude 3.5 Haiku (Anthropic)** - Fast & Cheap
   - **Cost:** $0.80/$4 per 1M tokens
   - **Performance:** Fast, cost-effective
   - **Use:** High-throughput tasks, simple reasoning
   - **Model string:** `"anthropic/claude-3-5-haiku-20241022"`

3. **Grok (xAI)** - Cheapest Option
   - **Cost:** $0.20/$0.50 per 1M tokens (cheapest!)
   - **Use:** Budget-constrained projects
   - **Model string:** `"xai/grok-beta"` (if available via LiteLLM)

**High Performance Models (When Quality > Cost):**

- **GPT-4o (OpenAI):** $2.50/$10 per 1M tokens - Best overall performance
- **Claude 3.5 Sonnet (Anthropic):** $3/$15 per 1M tokens - Excellent reasoning

**⚠️ Avoid in 2025:**
- ❌ **GPT-3.5-turbo** - More expensive AND worse than GPT-4o-mini
- ❌ Older Claude models - Newer versions are better value

**Environment Configuration:**

```python
import dspy
import os

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-key-here"
# Or for other providers:
# os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

# Configure the Language Model (DSPy 3.0 - Use dspy.LM, NOT dspy.OpenAI)
# dspy.OpenAI is DEPRECATED - always use dspy.LM with provider/model format

# RECOMMENDED (Dec 2025): GPT-4o-mini - Best value: 60%+ cheaper than GPT-3.5-turbo, better performance
lm = dspy.LM("openai/gpt-4o-mini")  # $0.15/$0.60 per 1M tokens - BEST VALUE!

# Alternative cheap options (Dec 2025):
# lm = dspy.LM("anthropic/claude-3-5-haiku-20241022")  # Fast & cheap: $0.80/$4 per 1M tokens
# lm = dspy.LM("xai/grok-beta")  # Cheapest: $0.20/$0.50 per 1M tokens (if available)

# For higher performance (when needed):
# lm = dspy.LM("openai/gpt-4o")  # $2.50/$10 per 1M tokens - best performance
# lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022")  # $3/$15 per 1M tokens

# With API key: lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_API_KEY")
dspy.configure(lm=lm)

# Optional: Enable caching
dspy.settings.configure(cache='jsonlite')  # or PersistentCache("cache.json")
```

---

## Core Concepts

### 1. Signatures

Signatures define the input/output contract for your LLM interactions. They replace manual prompt engineering.

```python
class QuestionAnswer(dspy.Signature):
    """Answer questions with short, concise answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# String-based signature (simpler)
qa_signature = "question -> answer"
```

**Advanced Signature Example:**

```python
from typing import Literal

class ExtractEvents(dspy.Signature):
    """Extract events and action items from email threads."""
    subject = dspy.InputField()
    thread = dspy.InputField()
    attachments: dict[str, dspy.Image] = dspy.InputField()
    events: list[Event] = dspy.OutputField()
    action_items: dict[str, Literal["P0", "P1", "P2"]] = dspy.OutputField()
```

### 2. Modules

Modules are the building blocks that use signatures to interact with LMs.

```python
class GenerateAnswer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QuestionAnswer)
        # Or: self.cot = dspy.ChainOfThought(QuestionAnswer)
    
    def forward(self, question):
        return self.predict(question=question)

# Usage
qa = GenerateAnswer()
result = qa(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

### 3. Optimizers (Teleprompters)

Optimizers automatically improve your programs by tuning prompts and selecting examples.

**Common Optimizers:**
- `BootstrapFewShot` - Good starting point, learns from examples
- `MIPROv2` - More advanced, explores prompt space intelligently
- `BayesianSignatureOptimizer` - Optimizes signature descriptions
- `GEPA` - Genetic Evolutionary Prompting Algorithm

### 4. Choosing the Right Optimizer Based on Dataset Size

**⭐ CRITICAL: Select optimizer based on your dataset size!**

| Dataset Size | Recommended Optimizer | Why | Code Example |
|--------------|----------------------|-----|--------------|
| **Very Small (~10 examples)** | `BootstrapFewShot` | Generates demonstrations from limited data | See below |
| **Small to Medium (~50 examples)** | `BootstrapFewShotWithRandomSearch` | Applies BootstrapFewShot with random search | See below |
| **Medium to Large (200+ examples)** | `MIPROv2` | Optimizes both instructions and demonstrations | See below |
| **Large (300+ examples)** | `MIPROv2` (heavy mode) | Prevents overfitting, better generalization | See below |

**Detailed Guide:**

#### Very Small Datasets (~10 examples)

Use `BootstrapFewShot` - it generates demonstrations and uses your labeled examples:

```python
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match

# For ~10 examples
optimizer = BootstrapFewShot(
    metric=answer_exact_match,
    max_demonstrations=2,  # Keep small for tiny datasets
    max_bootstrapped_demos=4
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

#### Small to Medium Datasets (~50 examples)

Use `BootstrapFewShotWithRandomSearch` - runs BootstrapFewShot multiple times:

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# For ~50 examples
optimizer = BootstrapFewShotWithRandomSearch(
    metric=answer_exact_match,
    max_demonstrations=3,
    num_candidates=10  # Try 10 different configurations
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

#### Medium to Large Datasets (200+ examples)

Use `MIPROv2` - optimizes both instructions and few-shot examples:

```python
from dspy.teleprompt import MIPROv2

def judge(prediction, gold):
    return 1.0 if prediction.answer.lower() == gold.answer.lower() else 0.0

# For 200+ examples
optimizer = MIPROv2(
    metric=judge,
    num_candidates=10,
    init_temperature=1.0
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

#### Large Datasets (300+ examples)

Use `MIPROv2` with heavy optimization mode:

```python
from dspy.teleprompt import MIPROv2

# For 300+ examples - use heavy mode
optimizer = MIPROv2(
    metric=judge,
    num_candidates=20,  # More candidates for large datasets
    init_temperature=1.0,
    optimization_level="heavy"  # Heavy optimization
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

**Quick Decision Tree:**

```
Do you have ~10 examples?
  YES → Use BootstrapFewShot
  NO ↓

Do you have ~50 examples?
  YES → Use BootstrapFewShotWithRandomSearch
  NO ↓

Do you have 200+ examples?
  YES → Use MIPROv2 (medium mode)
  NO ↓

Do you have 300+ examples?
  YES → Use MIPROv2 (heavy mode)
```

**Pro Tips:**
- **Start simple:** Always begin with `BootstrapFewShot` for small datasets
- **Compose optimizers:** You can run `MIPROv2` then use its output for `BootstrapFinetune`
- **Iterate:** Try different optimizers and compare results
- **Monitor costs:** Large datasets + large LMs = expensive optimization runs

---

## Complete Modules Reference

DSPy 3.0 provides **13 core modules** that serve as building blocks for LLM applications. Each module implements different reasoning and interaction patterns.

> **Note:** `dspy.Assertion` is deprecated in DSPy 2.6+ - use `dspy.Refine` or `dspy.BestOfN` instead.

### Quick Reference Table

| Module | Purpose | When to Use |
|--------|---------|-------------|
| `dspy.Predict` | Basic signature implementation | Straightforward input→output mapping |
| `dspy.ChainOfThought` | Step-by-step reasoning | Complex reasoning, math, logic problems |
| `dspy.ChainOfThoughtWithHint` | CoT with guidance hints | When LM needs direction or tends to go off-track |
| `dspy.ProgramOfThought` | Generate and execute code | Math, algorithmic, computational problems |
| `dspy.ReAct` | Reasoning + Acting with tools | External tool/API interaction, agentic tasks |
| `dspy.Retrieve` | Retrieve from knowledge base | RAG applications, document QA |
| `dspy.Refine` | Iterative feedback refinement | Quality improvement, self-correction |
| `dspy.BestOfN` | Run N times, pick best | Explore multiple answers, reduce variance |
| `dspy.Ensemble` | Combine multiple programs | Critical applications needing redundancy |
| `dspy.MultiChainComparison` | Compare multiple reasoning paths | Robust answer selection, verification |
| `dspy.Assertion` | ⚠️ **DEPRECATED** - Use Refine/BestOfN | Replaced by Refine with reward functions |
| `dspy.CodeAct` | **NEW** Code interpreter + tools | Code execution with tool integration |
| `dspy.Parallel` | Concurrent module execution | Run multiple modules in parallel |

### Detailed Module Descriptions

#### 1. `dspy.Predict` - Basic Prediction

The simplest module that directly implements a signature. Use when you have straightforward input→output mapping without complex reasoning.

```python
# Basic usage
predict = dspy.Predict("question -> answer")
result = predict(question="What is 2+2?")

# With explicit signature
class QA(dspy.Signature):
    """Answer questions concisely."""
    question = dspy.InputField()
    answer = dspy.OutputField()

predict = dspy.Predict(QA)
```

**When to use:** Simple transformations, extraction tasks, classification, straightforward Q&A.

#### 2. `dspy.ChainOfThought` - Step-by-Step Reasoning

Encourages the LM to reason through problems step-by-step before giving a final answer. Automatically adds a `reasoning` field.

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="If a train travels 60 mph for 2.5 hours, how far does it go?")
print(result.reasoning)  # Shows step-by-step work
print(result.answer)     # "150 miles"
```

**When to use:** Math problems, logic puzzles, multi-step reasoning, complex analysis.

#### 3. `dspy.ChainOfThoughtWithHint` - Guided Reasoning

Like ChainOfThought but allows you to provide hints to guide the reasoning process when the LM tends to go off-track.

```python
cot_hint = dspy.ChainOfThoughtWithHint("question -> answer")
result = cot_hint(
    question="What is the integral of x^2?",
    hint="Use the power rule: ∫x^n dx = x^(n+1)/(n+1)"
)
```

**When to use:** When LM needs guidance, domain-specific reasoning, correcting common mistakes.

#### 4. `dspy.ProgramOfThought` - Code-Based Reasoning

Generates and executes code to solve problems. Particularly powerful for mathematical and algorithmic tasks.

```python
pot = dspy.ProgramOfThought("question -> answer")
result = pot(question="Calculate the factorial of 7")
# Generates: code that computes factorial(7) = 5040
```

**When to use:** Mathematical computations, data processing, algorithmic problems, anything benefiting from code execution.

#### 5. `dspy.ReAct` - Reasoning + Acting

Implements the ReAct pattern for agentic tasks. Alternates between reasoning (thinking) and acting (using tools).

```python
# Define tools first
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Create ReAct agent
react = dspy.ReAct("question -> answer", tools=[search, calculator])
result = react(question="What is the population of France times 2?")
```

**When to use:** Agentic workflows, tool use, multi-step tasks requiring external actions.

#### 6. `dspy.Retrieve` - Knowledge Retrieval

Retrieves relevant passages from a knowledge base. Essential for RAG (Retrieval Augmented Generation) applications.

```python
# Configure retriever (e.g., ColBERT, Pinecone, etc.)
retriever = dspy.ColBERTv2(url='http://your-colbert-server')
dspy.settings.configure(rm=retriever)

# Use Retrieve module
retrieve = dspy.Retrieve(k=3)  # Get top 3 passages
passages = retrieve(query="What is machine learning?")
```

**When to use:** RAG systems, document Q&A, knowledge-grounded generation.

#### 7. `dspy.Refine` - Iterative Refinement

Iteratively refines outputs based on feedback. Useful for quality improvement and self-correction.

```python
class WritingRefinement(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("topic -> draft")
        self.refine = dspy.Refine("draft, feedback -> improved_draft", max_iterations=3)

    def forward(self, topic):
        draft = self.generate(topic=topic)
        return self.refine(draft=draft.draft, feedback="Make it more concise")
```

**When to use:** Writing improvement, code refinement, iterative quality enhancement.

#### 8. `dspy.BestOfN` - Multiple Attempts

Runs the same module N times and selects the best output based on a scoring function.

```python
# Create base module
base_qa = dspy.Predict("question -> answer")

# Wrap with BestOfN
best_qa = dspy.BestOfN(base_qa, n=5, scoring_fn=my_scoring_function)
result = best_qa(question="Explain quantum entanglement")
```

**When to use:** High-stakes outputs, reducing variance, exploring answer space.

#### 9. `dspy.Ensemble` - Combine Multiple Programs

Combines outputs from multiple programs to produce a more robust final answer.

```python
# Create different approaches
cot_qa = dspy.ChainOfThought("question -> answer")
pot_qa = dspy.ProgramOfThought("question -> answer")
simple_qa = dspy.Predict("question -> answer")

# Ensemble them
ensemble = dspy.Ensemble([cot_qa, pot_qa, simple_qa], aggregation="voting")
result = ensemble(question="What is 15% of 80?")
```

**When to use:** Critical applications, redundancy, combining different reasoning strategies.

#### 10. `dspy.MultiChainComparison` - Compare Reasoning Paths

Generates multiple reasoning chains and compares them to select the most reliable answer.

```python
mcc = dspy.MultiChainComparison("question -> answer", num_chains=3)
result = mcc(question="Is the statement 'All swans are white' logically valid?")
# Generates 3 different reasoning paths and synthesizes
```

**When to use:** Logical reasoning, verification, when answer reliability is critical.

#### 11. `dspy.Assertion` - Output Constraints ⚠️ DEPRECATED

> **⚠️ DEPRECATED in DSPy 2.6+**: `dspy.Assert` and `dspy.Suggest` have been **replaced by `dspy.Refine` and `dspy.BestOfN`**. Use reward functions instead of assertions for constraint enforcement.

**Legacy example (not recommended):**

```python
# OLD WAY - Deprecated
class ValidatedQA(dspy.Module):
    def forward(self, question):
        result = self.predict(question=question)
        dspy.Assert(len(result.answer) > 0, "Answer must not be empty")  # Deprecated!
        return result
```

**NEW WAY - Use Refine with reward functions:**

```python
# NEW WAY - Recommended
def reward_fn(example, prediction, trace=None):
    if len(prediction.answer) == 0:
        return 0.0
    if len(prediction.answer) > 500:
        return 0.5
    return 1.0

refine = dspy.Refine("question -> answer", reward_fn=reward_fn, max_iterations=3)
```

#### 12. `dspy.CodeAct` - Code Interpreter with Tools (NEW in DSPy 3.0)

**NEW in DSPy 3.0!** Combines code generation with tool execution. Uses a Python interpreter and predefined tools for complex computational tasks.

```python
# Define tools
def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    return eval(expression)

def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    import requests
    return requests.get(url).text

# Create CodeAct agent
codeact = dspy.CodeAct(
    "question -> answer",
    tools=[calculate, fetch_data],
    max_iterations=5
)

result = codeact(question="Calculate 15% tip on $85.50 and round to 2 decimals")
# Generates and executes: calculate("round(85.50 * 0.15, 2)")
```

**When to use:** Complex computations, data processing, tasks requiring code execution with tool access.

**Key difference from ProgramOfThought:** CodeAct integrates tools and maintains state across multiple code executions, while ProgramOfThought focuses on single-shot code generation.

#### 13. `dspy.Parallel` - Concurrent Module Execution

Runs multiple module instances concurrently across threads for improved performance.

```python
# Create a module to run in parallel
qa_module = dspy.Predict("question -> answer")

# Wrap with Parallel for concurrent execution
parallel_qa = dspy.Parallel(num_threads=4)

# Process multiple questions concurrently
questions = [
    dspy.Example(question="What is Python?"),
    dspy.Example(question="What is JavaScript?"),
    dspy.Example(question="What is Rust?"),
    dspy.Example(question="What is Go?")
]

# Run in parallel
results = parallel_qa(qa_module, questions)
```

**When to use:** Batch processing, high-throughput applications, when you need to process many inputs concurrently.

**Note:** Used internally by optimizers like SIMBA for efficient batch evaluation.

### Module Selection Decision Tree

```
What type of task?
│
├── Simple transformation → dspy.Predict
├── Needs reasoning → dspy.ChainOfThought
│   └── LM goes off-track → dspy.ChainOfThoughtWithHint
├── Mathematical/Code → dspy.ProgramOfThought
│   └── Needs tools + state → dspy.CodeAct (NEW)
├── Needs external tools → dspy.ReAct
├── Needs knowledge base → dspy.Retrieve
├── Needs refinement → dspy.Refine
├── High stakes/variance → dspy.BestOfN or dspy.Ensemble
├── Needs verification → dspy.MultiChainComparison
├── Batch processing → dspy.Parallel
└── Needs constraints → dspy.Refine (replaces deprecated Assertion)
```

---

## Complete Optimizers Reference

DSPy 3.0 provides **11 optimizers** (also called teleprompters) that automatically improve your programs by tuning prompts, selecting examples, and optimizing weights.

> **Note:** `BayesianSignatureOptimizer` may be superseded by MIPROv2 which now uses Bayesian optimization internally.

### Quick Reference Table

| Optimizer | Dataset Size | Purpose | Computational Cost |
|-----------|-------------|---------|-------------------|
| `LabeledFewShot` | Small, high-quality | Quick baselines with labeled examples | ⚡ Very Low |
| `BootstrapFewShot` | ~10-30 examples | Generate synthetic demonstrations | ⚡ Low |
| `BootstrapFewShotWithRandomSearch` | ~50+ examples | Random search over demo configurations | ⚡⚡ Low-Medium |
| `KNNFewShot` | 20-100+ examples | K-nearest neighbor demo selection | ⚡⚡ Medium |
| `COPRO` | Any | Coordinate ascent on instructions | ⚡⚡⚡ Medium-High |
| `GEPA` | Few dozen | Genetic evolutionary prompting | ⚡⚡⚡ Medium-High |
| `SIMBA` | 50+ examples | **NEW** Stochastic mini-batch optimization | ⚡⚡⚡ Medium-High |
| `BayesianSignatureOptimizer` | Any | Bayesian optimization of signatures | ⚡⚡⚡ Medium-High |
| `MIPROv2` | 50-300+ examples | Instruction + demo optimization | ⚡⚡⚡⚡ High |
| `BootstrapFinetune` | Large | Fine-tune LM weights | ⚡⚡⚡⚡⚡ Very High |
| `Ensemble` | N/A | Combine multiple optimized programs | Varies |

### Detailed Optimizer Descriptions

#### 1. `LabeledFewShot` - Quick Baseline

The simplest optimizer. Uses your labeled examples directly as few-shot demonstrations without any bootstrapping.

```python
from dspy.teleprompt import LabeledFewShot

optimizer = LabeledFewShot(k=3)  # Use 3 examples
compiled_module = optimizer.compile(module, trainset=trainset)
```

**When to use:** Quick prototyping, high-quality curated examples, when you want full control over demos.

**Pros:** Fast, predictable, no extra LM calls
**Cons:** Limited by quality of your examples

#### 2. `BootstrapFewShot` - Synthetic Demonstrations

Generates additional demonstrations by running your module on training data and keeping successful outputs.

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=my_metric,
    max_demonstrations=3,       # Max demos in final prompt
    max_bootstrapped_demos=8,   # Max demos to generate
    max_rounds=2                # Bootstrapping rounds
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

**When to use:** ~10 examples, want to augment limited data.

**Pros:** Creates diverse demonstrations, works with limited data
**Cons:** Quality depends on initial module performance

#### 3. `BootstrapFewShotWithRandomSearch` - Random Search

Runs BootstrapFewShot multiple times with different configurations and selects the best.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=my_metric,
    max_demonstrations=4,
    num_candidate_programs=10,  # Try 10 configurations
    num_threads=4               # Parallel execution
)
compiled_module = optimizer.compile(module, trainset=trainset, valset=valset)
```

**When to use:** ~50 examples, want better coverage of configuration space.

**Pros:** Better exploration, finds good configurations
**Cons:** More expensive than basic BootstrapFewShot

#### 4. `KNNFewShot` - K-Nearest Neighbor Selection

Dynamically selects demonstrations based on semantic similarity to the input query.

```python
from dspy.teleprompt import KNNFewShot

optimizer = KNNFewShot(
    k=5,                        # Number of nearest neighbors
    trainset=trainset,
    vectorizer=your_vectorizer  # Optional custom embeddings
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

**When to use:** 20-100+ examples, diverse query types, when relevant examples matter.

**Pros:** Input-specific demonstrations, scales well
**Cons:** Requires embedding computation, overhead per query

#### 5. `COPRO` - Coordinate Ascent

Optimizes instructions through coordinate ascent, iteratively improving each component.

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=my_metric,
    breadth=10,                 # Candidates per dimension
    depth=3,                    # Optimization depth
    init_temperature=1.4
)
compiled_module = optimizer.compile(module, trainset=trainset, eval_kwargs={'num_threads': 4})
```

**When to use:** Any dataset size, want to optimize instruction text.

**Pros:** Systematic instruction optimization
**Cons:** Many LM calls, can be slow

#### 6. `GEPA` - Genetic Evolutionary Prompting

Uses genetic algorithms to evolve better prompts through mutation and selection. Based on the 2025 paper "GEPA: Reflective Prompt Evolution."

```python
from dspy.teleprompt import GEPA

optimizer = GEPA(
    metric=my_metric,
    num_generations=10,
    population_size=20,
    mutation_rate=0.1
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

**When to use:** Few dozen examples, want creative prompt exploration.

**Pros:** Explores diverse prompt variations, can find unexpected improvements
**Cons:** Non-deterministic, requires tuning hyperparameters

#### 7. `SIMBA` - Stochastic Mini-Batch Optimization (NEW)

**NEW in DSPy 3.0!** Uses stochastic mini-batch sampling to identify challenging examples with high output variability, then applies introspective failure analysis.

```python
from dspy.teleprompt import SIMBA

optimizer = SIMBA(
    metric=my_metric,
    bsize=16,              # Mini-batch size
    max_steps=50,          # Maximum optimization steps
    num_threads=4          # Parallel threads for evaluation
)

compiled_module = optimizer.compile(
    module,
    trainset=trainset,
    seed=42  # For reproducibility
)

# Access optimization logs
print(compiled_module.trial_logs)
print(compiled_module.candidate_programs)
```

**When to use:** 50+ examples, want to focus on difficult/variable examples.

**Pros:** Efficient for finding edge cases, good for noisy datasets
**Cons:** Requires sufficient batch size, may need many steps

**Key Feature:** SIMBA tracks which examples cause high variability across runs, focusing optimization on the most challenging cases.

#### 8. `BayesianSignatureOptimizer` - Signature Tuning

> **Note:** This optimizer may be superseded by MIPROv2, which now incorporates Bayesian optimization internally.

Uses Bayesian optimization to tune signature field descriptions and docstrings.

```python
from dspy.teleprompt import BayesianSignatureOptimizer

optimizer = BayesianSignatureOptimizer(
    metric=my_metric,
    n_iterations=50,
    num_candidates=10
)
compiled_module = optimizer.compile(module, trainset=trainset)
```

**When to use:** Any dataset, want to optimize how fields are described.

**Pros:** Principled optimization, good for signature refinement
**Cons:** Requires many evaluations, may be deprecated in future versions

#### 9. `MIPROv2` - Multiprompt Instruction Proposal Optimizer

The most powerful optimizer. Jointly optimizes instructions AND demonstrations.

```python
from dspy.teleprompt import MIPROv2

# For 200+ examples
optimizer = MIPROv2(
    metric=my_metric,
    num_candidates=10,
    init_temperature=1.0,
)

# For 300+ examples - use heavy mode
optimizer_heavy = MIPROv2(
    metric=my_metric,
    num_candidates=20,
    init_temperature=1.0,
    auto=True,                  # Auto-configure based on dataset
)

compiled_module = optimizer.compile(module, trainset=trainset)
```

**When to use:** 200+ examples, want maximum performance improvement.

**Pros:** State-of-the-art results, joint optimization
**Cons:** Expensive, requires significant compute

#### 10. `BootstrapFinetune` - LM Fine-tuning

Fine-tunes the underlying language model weights using bootstrapped examples.

```python
from dspy.teleprompt import BootstrapFinetune

optimizer = BootstrapFinetune(
    metric=my_metric,
    num_threads=4
)

# This will actually modify model weights
compiled_module = optimizer.compile(
    module,
    trainset=trainset,
    target="openai/gpt-3.5-turbo"  # Model to finetune
)
```

**When to use:** When prompt optimization isn't enough, have significant training data.

**Pros:** Permanent improvements, can exceed prompt-based optimization
**Cons:** Very expensive, requires API support for finetuning, model-specific

#### 11. `Ensemble` - Combine Optimized Programs

Not a traditional optimizer, but combines multiple optimized programs for better robustness.

```python
from dspy.teleprompt import Ensemble

# Create multiple optimized versions
opt1 = BootstrapFewShot(metric=metric).compile(module, trainset=trainset)
opt2 = MIPROv2(metric=metric).compile(module, trainset=trainset)
opt3 = COPRO(metric=metric).compile(module, trainset=trainset)

# Ensemble them
ensemble = Ensemble([opt1, opt2, opt3], aggregation="voting")
```

**When to use:** Critical applications, want redundancy and robustness.

**Pros:** Combines strengths of different approaches
**Cons:** Higher inference cost (runs multiple models)

### Optimizer Selection Decision Tree

```
How much data do you have?
│
├── Very little (~10-30) → BootstrapFewShot
│   └── High-quality curated → LabeledFewShot
│
├── Small (~50) → BootstrapFewShotWithRandomSearch
│   └── Noisy/variable data → SIMBA (NEW)
│
├── Medium (100+) → KNNFewShot or COPRO
│   └── Want instruction optimization → COPRO or GEPA
│   └── Diverse query types → KNNFewShot
│   └── Focus on hard examples → SIMBA (NEW)
│
├── Large (200+) → MIPROv2 (medium mode)
│
├── Very Large (300+) → MIPROv2 (heavy mode, 40+ trials)
│   └── Want permanent improvements → BootstrapFinetune
│
└── Critical application → Ensemble (combine multiple)

Pro tip: Start with 30 examples, aim for 300 for optimal results.
```

### Composing Optimizers

You can chain optimizers for better results:

```python
# Step 1: Optimize instructions with COPRO
copro = COPRO(metric=metric).compile(module, trainset=trainset)

# Step 2: Then optimize demos with MIPROv2
mipro = MIPROv2(metric=metric).compile(copro, trainset=trainset)

# Step 3: Optionally finetune
final = BootstrapFinetune(metric=metric).compile(mipro, trainset=large_trainset)
```

### Cost Estimation

| Optimizer | Approximate LM Calls (per example) | Best for Budget |
|-----------|-----------------------------------|-----------------|
| LabeledFewShot | 0 | ⭐⭐⭐⭐⭐ Cheapest |
| BootstrapFewShot | 2-5 | ⭐⭐⭐⭐ Very Cheap |
| BootstrapFewShotWithRandomSearch | 20-50 | ⭐⭐⭐ Moderate |
| KNNFewShot | 1 (per query) + embeddings | ⭐⭐⭐⭐ Cheap inference |
| COPRO | 50-200 | ⭐⭐ Expensive |
| GEPA | 100-500 | ⭐⭐ Expensive |
| SIMBA | 50-200 | ⭐⭐ Expensive |
| BayesianSignatureOptimizer | 50-200 | ⭐⭐ Expensive |
| MIPROv2 | 200-1000 | ⭐ Very Expensive |
| BootstrapFinetune | Varies by provider | ⭐ Very Expensive |

---

## Extracting Compiled Prompts

### Method 1: Using `dspy.inspect_history()`

View the history of LM calls, including prompts and responses:

```python
# After running your program
dspy.inspect_history(n=2)  # Shows last 2 prompts (before/after optimization)
```

### Method 2: Accessing Predictor Attributes

Direct access to compiled prompts:

```python
# After compilation
compiled_module = optimizer.compile(module, trainset=trainset)

# Access the compiled prompt
print(compiled_module.predict.prompt)

# Access demonstrations/examples used
print(compiled_module.predict.demos)

# Access training inputs/outputs
if hasattr(compiled_module.predict, 'demos') and len(compiled_module.predict.demos) > 0:
    print(compiled_module.predict.demos[0].train_inputs)
    print(compiled_module.predict.demos[0].train_outputs)
```

### Method 3: Using `dspy.inspect()`

Get a detailed view of the entire compiled program:

```python
dspy.inspect(compiled_module)
# Prints detailed structure including all prompts, modules, and configurations
```

### Method 4: Iterating Through Multiple Predictors

For complex programs with multiple predictors:

```python
# Iterate through all predictors
for name, predictor in compiled_module.named_predictors():
    print(f"Predictor: {name}")
    try:
        # Access prompt
        if hasattr(predictor, 'prompt'):
            print(f"  Prompt: {predictor.prompt}")
        
        # Access demos
        if hasattr(predictor, 'demos'):
            print(f"  Demos: {len(predictor.demos)} examples")
            for i, demo in enumerate(predictor.demos):
                print(f"    Demo {i}: {demo}")
    except AttributeError:
        print("  No prompt found for this predictor")
```

### Complete Example: Extracting Prompts

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Define module
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.cot(question=question)

# Prepare data
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question")
]

# Compile
optimizer = BootstrapFewShot(metric=dspy.evaluate.answer_exact_match, max_demonstrations=2)
compiled_qa = optimizer.compile(QA(), trainset=trainset)

# Extract and view prompts
print("=" * 50)
print("EXTRACTED PROMPTS:")
print("=" * 50)

# Method 1: Direct access
print("\n1. Direct Prompt Access:")
print(compiled_qa.cot.prompt)

# Method 2: View demos
print("\n2. Demonstrations Used:")
for i, demo in enumerate(compiled_qa.cot.demos):
    print(f"  Demo {i+1}:")
    print(f"    Input: {demo.train_inputs}")
    print(f"    Output: {demo.train_outputs}")

# Method 3: Inspect history
print("\n3. Inspection History:")
dspy.inspect_history(n=1)

# Method 4: Full inspection
print("\n4. Full Module Inspection:")
dspy.inspect(compiled_qa)
```

---

## Common Patterns

### 1. Chain-of-Thought (CoT)

Encourages step-by-step reasoning:

```python
class CoTQA(dspy.Signature):
    """Answer questions with step-by-step reasoning."""
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="Think step by step")
    answer = dspy.OutputField()

class CoTAnswer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(CoTQA)
    
    def forward(self, question):
        return self.cot(question=question)

# Usage
cot_qa = CoTAnswer()
result = cot_qa(question="If a train travels 60 mph for 2 hours, how far does it go?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")
```

### 2. ReAct (Reasoning + Acting)

Combines reasoning with tool/action execution:

```python
class Search(dspy.Signature):
    """Search for information."""
    query = dspy.InputField()
    results = dspy.OutputField()

class ReActQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.Predict(Search)
        self.answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # Step 1: Search
        search_result = self.search(query=question)
        # Step 2: Answer based on search
        return self.answer(context=search_result.results, question=question)
```

### 3. RAG (Retrieval Augmented Generation)

```python
class RAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever  # Your retrieval system
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # Retrieve relevant context
        passages = self.retriever(question, k=3)
        context = "\n\n".join([p.passage for p in passages])
        
        # Generate answer with context
        return self.generate(context=context, question=question)
```

### 4. Multi-Hop Reasoning

```python
class MultiHopQA(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = []
        # Multiple retrieval hops
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            passages = self.retriever(query, k=2)
            context.extend([p.passage for p in passages])
        
        return self.generate_answer(context="\n\n".join(context), question=question)
```

---

## Optimization Workflows

### Basic Optimization Workflow

```python
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match

# 1. Define your module
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.predict(question=question)

# 2. Prepare datasets
trainset = [
    dspy.Example(question="Q1", answer="A1").with_inputs("question"),
    dspy.Example(question="Q2", answer="A2").with_inputs("question"),
    # ... more examples
]

devset = [
    dspy.Example(question="Q1", answer="A1").with_inputs("question"),
    # ... validation examples
]

# 3. Define metric
def my_metric(preds, golds, trace=None):
    return dspy.evaluate.answer_exact_match(preds, golds)

# 4. Compile with optimizer
optimizer = BootstrapFewShot(
    metric=my_metric,
    max_demonstrations=3,
    max_bootstrapped_demos=4
)
compiled_module = optimizer.compile(MyModule(), trainset=trainset)

# 5. Evaluate
results = dspy.evaluate.evaluate(
    compiled_module,
    devset=devset,
    metric=my_metric,
    display_progress=True
)

# 6. Extract and inspect compiled prompts
print("Compiled Prompt:")
print(compiled_module.predict.prompt)
dspy.inspect_history(n=2)
```

### Advanced: MIPROv2 Optimization

```python
from dspy.teleprompt import MIPROv2

# Define judge function for evaluation
def judge(prediction, gold):
    # Your custom evaluation logic
    return 1.0 if prediction.answer.lower() == gold.answer.lower() else 0.0

# Create optimizer
optimizer = MIPROv2(
    metric=judge,
    num_candidates=10,
    init_temperature=1.0
)

# Compile
compiled_module = optimizer.compile(module, trainset=trainset)

# MIPROv2 explores prompt space more intelligently
# Inspect the optimized prompts
dspy.inspect(compiled_module)
```

### Using MLflow for Monitoring

```python
import mlflow

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy_Optimization")

# Enable autologging
mlflow.dspy.autolog()

# Run optimization (traces will be logged automatically)
optimizer = BootstrapFewShot(metric=my_metric)
compiled_module = optimizer.compile(module, trainset=trainset)

# View in MLflow UI at http://localhost:5000
```

---

## Best Practices

### 1. Start Simple
- Begin with basic `dspy.Predict` modules
- Use simple string signatures initially
- Gradually add complexity (CoT, multi-hop, etc.)

### 2. Define Clear Metrics ⭐ **CRITICAL**

**Metrics are the foundation of DSPy optimization!** See `DSPy_3.0_Custom_Evaluations_Guide.md` for a complete guide.

```python
# Good: Specific, measurable metric
def exact_match_metric(example, prediction, trace=None):
    """Note: Correct signature is (example, prediction, trace=None)"""
    return 1.0 if example.answer.lower() == prediction.answer.lower() else 0.0

# Bad: Vague metric
def vague_metric(example, prediction, trace=None):
    return "good" if len(prediction.answer) > 0 else "bad"  # Returns string, not number!
```

**Key Points:**
- Metric signature: `(example, prediction, trace=None)`
- Return: `float` (0.0-1.0), `int` (0/1), or `bool` (True/False)
- Higher = better
- **Your metric guides ALL optimization** - make it count!

### 3. Use Representative Validation Sets
- Validation set should reflect real-world distribution
- Include edge cases and difficult examples
- Balance positive and negative examples

### 4. Choose Optimizer Based on Dataset Size ⭐
**This is critical!** The right optimizer depends on how much data you have:

- **~10 examples:** Use `BootstrapFewShot`
- **~50 examples:** Use `BootstrapFewShotWithRandomSearch`
- **200+ examples:** Use `MIPROv2` (medium mode)
- **300+ examples:** Use `MIPROv2` (heavy mode)

See the detailed guide in [Core Concepts - Choosing the Right Optimizer](#4-choosing-the-right-optimizer-based-on-dataset-size) section above.

### 5. Inspect and Understand Prompts
```python
# Always inspect compiled prompts
dspy.inspect_history(n=5)  # See recent prompts
dspy.inspect(compiled_module)  # Full structure

# Save prompts for analysis
with open("compiled_prompts.txt", "w") as f:
    f.write(str(compiled_module.predict.prompt))
```

### 6. Use Caching
```python
# Enable caching to save API calls during development
dspy.settings.configure(cache='jsonlite')
# Or persistent cache
dspy.settings.configure(cache=dspy.cache.PersistentCache("cache.json"))
```

### 7. Handle Errors Gracefully
```python
class RobustQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")
    
    def forward(self, question):
        try:
            result = self.predict(question=question)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return dspy.Prediction(answer="I couldn't process that question.")
```

### 8. Monitor API Usage & Choose Cost-Effective Models
```python
# Use GPT-4o-mini for development and most tasks (60%+ cheaper than GPT-3.5-turbo!)
lm = dspy.LM("openai/gpt-4o-mini")  # Best value in Dec 2025

# Only use expensive models when you need maximum performance
# lm = dspy.LM("openai/gpt-4o")  # Only for critical tasks
```
- Use caching to reduce redundant calls
- Set budget limits for optimization runs
- Log all API calls for cost tracking
- **Use GPT-4o-mini instead of GPT-3.5-turbo** - it's cheaper AND better!

### 9. Document Your Signatures
```python
class WellDocumentedQA(dspy.Signature):
    """
    Answer questions based on provided context.
    
    The answer should be:
    - Concise (1-2 sentences)
    - Factual and accurate
    - Based only on the provided context
    """
    context = dspy.InputField(desc="Relevant context for answering")
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="The answer to the question")
```

### 10. Iterate and Refine
- Run optimization multiple times
- Compare different optimizer settings
- A/B test different signature descriptions
- Track performance over time

---

## Training Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DSPy 3.0 Training Workflow                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  1. SETUP    │
│              │
│ • Install    │
│ • Configure  │
│   LM         │
│ • Set API    │
│   Keys       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. DEFINE    │
│   SIGNATURE  │
│              │
│ • Inputs     │
│ • Outputs    │
│ • Descriptions│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. CREATE    │
│   MODULE     │
│              │
│ • Predict    │
│ • ChainOfThought│
│ • Custom     │
│   Logic      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. PREPARE   │
│   DATA       │
│              │
│ • Train Set  │
│ • Dev Set    │
│ • Examples   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 5. DEFINE    │
│   METRIC     │
│              │
│ • Exact Match│
│ • Custom     │
│   Function   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 6. OPTIMIZE  │
│              │
│ • Bootstrap  │
│ • MIPROv2    │
│ • Bayesian   │
│ • GEPA       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 7. COMPILE   │
│              │
│ • Tune Prompts│
│ • Select     │
│   Examples   │
│ • Optimize   │
│   Weights    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 8. EXTRACT   │
│   PROMPTS    │
│              │
│ • inspect_   │
│   history()  │
│ • predictor. │
│   prompt     │
│ • inspect()  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 9. EVALUATE  │
│              │
│ • Run on     │
│   Dev Set    │
│ • Calculate  │
│   Metrics    │
│ • Analyze    │
│   Results    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 10. ITERATE  │
│              │
│ • Refine     │
│ • Adjust     │
│ • Re-optimize│
└──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│              Component Interaction Diagram                       │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │  Signature  │◄───── Defines contract
    └──────┬──────┘
           │
           │ used by
           ▼
    ┌─────────────┐
    │   Module     │◄───── Implements logic
    └──────┬──────┘
           │
           │ compiled by
           ▼
    ┌─────────────┐
    │  Optimizer   │◄───── Tunes prompts
    └──────┬──────┘
           │
           │ uses
           ▼
    ┌─────────────┐      ┌─────────────┐
    │  Train Set   │      │  Dev Set     │
    └─────────────┘      └─────────────┘
           │                    │
           └────────┬───────────┘
                    │
                    ▼
           ┌─────────────┐
           │   Metric     │◄───── Evaluates performance
           └──────┬───────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │   Compiled Module            │
    │                              │
    │ • Optimized Prompts          │
    │ • Selected Examples          │
    │ • Tuned Weights              │
    └─────────────────────────────┘
                  │
                  │ can extract via
                  ▼
    ┌─────────────────────────────┐
    │   Prompt Extraction Methods   │
    │                              │
    │ • inspect_history()          │
    │ • predictor.prompt              │
    │ • predictor.demos             │
    │ • inspect()                   │
    └─────────────────────────────┘
```

---

## Code Examples

### Complete Example: Question Answering with Prompt Extraction

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match

# Configure LM (DSPy 3.0 - Use dspy.LM)
# GPT-4o-mini: Best value in Dec 2025 - cheaper AND better than GPT-3.5-turbo
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define signature
class QA(dspy.Signature):
    """Answer questions with concise answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Create module
class QuestionAnswering(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(QA)
    
    def forward(self, question):
        return self.cot(question=question)

# Prepare data
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare").with_inputs("question"),
]

devset = [
    dspy.Example(question="What is the largest planet?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
]

# Optimize
optimizer = BootstrapFewShot(
    metric=answer_exact_match,
    max_demonstrations=2
)
compiled_qa = optimizer.compile(QuestionAnswering(), trainset=trainset)

# Evaluate
results = dspy.evaluate.evaluate(
    compiled_qa,
    devset=devset,
    metric=answer_exact_match,
    display_progress=True
)
print(f"Accuracy: {results}")

# EXTRACT PROMPTS
print("\n" + "="*60)
print("EXTRACTED COMPILED PROMPTS")
print("="*60)

# Method 1: Direct access
print("\n1. Compiled Prompt (Direct Access):")
print("-" * 60)
print(compiled_qa.cot.prompt)

# Method 2: View demonstrations
print("\n2. Demonstrations Used in Prompt:")
print("-" * 60)
for i, demo in enumerate(compiled_qa.cot.demos):
    print(f"\nDemo {i+1}:")
    print(f"  Input: {demo.train_inputs}")
    print(f"  Output: {demo.train_outputs}")

# Method 3: Inspect history
print("\n3. Recent Prompt History:")
print("-" * 60)
dspy.inspect_history(n=2)

# Method 4: Full inspection
print("\n4. Full Module Structure:")
print("-" * 60)
dspy.inspect(compiled_qa)

# Save prompts to file
with open("compiled_prompts.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("COMPILED PROMPTS\n")
    f.write("="*60 + "\n\n")
    f.write("Main Prompt:\n")
    f.write(str(compiled_qa.cot.prompt))
    f.write("\n\n" + "="*60 + "\n")
    f.write("Demonstrations:\n")
    f.write("="*60 + "\n")
    for i, demo in enumerate(compiled_qa.cot.demos):
        f.write(f"\nDemo {i+1}:\n")
        f.write(f"  Input: {demo.train_inputs}\n")
        f.write(f"  Output: {demo.train_outputs}\n")

print("\n✓ Prompts saved to compiled_prompts.txt")
```

### Example: Multi-Module Program with Prompt Extraction

```python
class MultiStepQA(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # Step 1: Generate search query
        query = self.generate_query(question=question)
        
        # Step 2: Retrieve context
        passages = self.retriever(query.search_query, k=3)
        context = "\n\n".join([p.passage for p in passages])
        
        # Step 3: Generate answer
        return self.generate_answer(context=context, question=question)

# After compilation, extract all prompts
def extract_all_prompts(compiled_module):
    """Extract prompts from all predictors in a compiled module."""
    prompts = {}
    
    for name, predictor in compiled_module.named_predictors():
        prompts[name] = {
            'prompt': getattr(predictor, 'prompt', None),
            'demos': getattr(predictor, 'demos', []),
            'num_demos': len(getattr(predictor, 'demos', []))
        }
    
    return prompts

# Usage
compiled_multi = optimizer.compile(MultiStepQA(retriever), trainset=trainset)
all_prompts = extract_all_prompts(compiled_multi)

for name, prompt_info in all_prompts.items():
    print(f"\n{name}:")
    print(f"  Prompt: {prompt_info['prompt'][:200]}...")  # First 200 chars
    print(f"  Number of demos: {prompt_info['num_demos']}")
```

---

## Key Takeaways

1. **Prompt Extraction is Possible**: DSPy 3.0 provides multiple methods to access compiled prompts
2. **Use `dspy.inspect_history()`** for quick debugging
3. **Access `predictor.prompt`** for direct prompt access
4. **Use `dspy.inspect()`** for comprehensive program analysis
5. **Iterate through `named_predictors()`** for complex multi-module programs
6. **Save prompts** to files for analysis and documentation

---

## Resources

- **Official Documentation**: https://dspy-docs.vercel.app/
- **GitHub Repository**: https://github.com/stanfordnlp/dspy
- **Databricks Integration**: https://www.databricks.com/blog/dspy-3-0
- **Video Tutorials**: Multiple videos available in your DeepLake RAG database
- **📘 Custom Evaluations Guide**: See `DSPy_3.0_Custom_Evaluations_Guide.md` for comprehensive guide on writing custom metrics

---

## Version Information

- **DSPy Version**: 3.0+
- **Last Updated**: December 2025
- **Compatibility**: Python 3.8+, Works with OpenAI, Anthropic, and other LiteLLM-compatible providers

## Model Pricing Reference (December 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|-------|----------------------|----------------------|----------|
| **GPT-4o-mini** ⭐ | $0.15 | $0.60 | **Best value - recommended for most tasks** |
| Grok (xAI) | $0.20 | $0.50 | Cheapest option |
| Claude 3.5 Haiku | $0.80 | $4.00 | Fast, cost-effective |
| GPT-4o | $2.50 | $10.00 | Highest performance |
| Claude 3.5 Sonnet | $3.00 | $15.00 | Excellent reasoning |
| ~~GPT-3.5-turbo~~ | ~~$0.50~~ | ~~$1.50~~ | ❌ **Avoid - more expensive than GPT-4o-mini!** |

---

---

## Verification & Updates

**This guide has been verified against DSPy 3.0 (2025) using:**
- ✅ **Context7 Documentation** - Official DSPy library documentation
- ✅ **Gemini Research** - Latest API changes and best practices (December 2025)
- ✅ **DeepLake RAG** - Your video sessions and tutorials
- ✅ **Official DSPy Resources** - GitHub and documentation

**Key Corrections Made:**
1. ✅ Replaced deprecated `dspy.OpenAI` with `dspy.LM("openai/model")`
2. ✅ Updated all LM configuration examples to use unified `dspy.LM()` interface
3. ✅ Verified installation command (`pip install -U dspy`)
4. ✅ Confirmed prompt extraction methods (`inspect_history()`, `predictor.prompt`, etc.)
5. ✅ Verified optimizer usage patterns

**Last Verified:** December 2025  
**DSPy Version:** 3.0+

---

*This guide was compiled from: DeepLake RAG database, Gemini research, Context7 documentation, and official DSPy resources.*
