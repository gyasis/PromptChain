# PromptChain Benchmarking Framework PRD

## Status: PRD for Future Implementation

## Overview

A comprehensive testing and benchmarking framework that leverages the existing `promptchain query --dev` command to:
- Run questions from datasets with full observability
- Capture outputs, logs, and execution details automatically
- Score results against expected outputs
- Save logs for analysis
- Test across all strategy modes and 14 agentic pattern pillars
- Generate performance and accuracy reports

## Key Integration Point: Existing Query Command

The framework uses the existing `promptchain query` command which:
- Takes a query string and executes it
- Supports `--dev` flag for full logging/observability
- Saves logs to `~/.promptchain/sessions/{session}/debug_{timestamp}.log`
- Returns response to stdout
- Supports `--session`, `--agent`, `--model` options

## Test Configuration Matrix

### Strategy Configurations (21 Total)

| Router Mode | Execution Mode | Supervisor | Total |
|-------------|----------------|------------|-------|
| router | pipeline | none | 1 |
| router | round-robin | none | 1 |
| router | broadcast | none | 1 |
| router | pipeline | checkpoint | 1 |
| router | round-robin | checkpoint | 1 |
| router | broadcast | checkpoint | 1 |
| router | pipeline | full | 1 |
| ... | ... | ... | ... |
| **3 router modes** | **4 execution modes** | **6 supervisor configs** | **21** |

**Router Modes:**
1. `router` - Dynamic agent selection via LLM
2. `pipeline` - Sequential agent execution
3. `round-robin` - Cyclic agent execution

**Execution Modes:**
1. `pipeline` - Sequential step execution
2. `round-robin` - Cyclic step execution
3. `broadcast` - Parallel execution with synthesis
4. `hybrid` - Mix of sequential and parallel

**Supervisor Configurations:**
1. `none` - No supervision
2. `checkpoint` - Checkpoint-based recovery
3. `full` - Full supervision with intervention
4. `critique` - Self-critique loop
5. `parallel-vote` - Multiple agents vote on result
6. `hierarchical` - Manager-worker pattern

### 14 Agentic Pattern Pillars

Each pattern represents a distinct architectural approach:

| ID | Pattern | Description |
|----|---------|-------------|
| P01 | Single Agent | One agent, direct execution |
| P02 | Router-Based | LLM routes to specialized agents |
| P03 | Pipeline | Sequential multi-agent flow |
| P04 | Broadcast | Parallel execution, synthesized results |
| P05 | Hierarchical | Manager coordinates worker agents |
| P06 | Debate/Adversarial | Agents argue, reach consensus |
| P07 | Reflection | Self-critique and improvement loop |
| P08 | Tool-Augmented | Agent with external tool access |
| P09 | Memory-Augmented | Agent with persistent memory/RAG |
| P10 | Planner-Executor | Separate planning and execution agents |
| P11 | Critique-Revise | Critic agent reviews, revises output |
| P12 | Ensemble | Multiple agents, aggregated results |
| P13 | Speculative | Parallel speculative execution |
| P14 | Supervisor | Central supervisor coordinates all |

## Architecture Components

### 1. DatasetLoader

```python
class DatasetLoader:
    """Loads and manages test datasets."""

    def load(self, dataset_path: str) -> List[TestCase]:
        """Load dataset from JSON/CSV/JSONL."""

    def get_test_cases(self,
                       filter_tags: List[str] = None,
                       limit: int = None) -> List[TestCase]:
        """Get filtered test cases."""

@dataclass
class TestCase:
    id: str
    input_query: str
    expected_output: str  # For comparison
    expected_keywords: List[str]  # Must contain these
    category: str  # "reasoning", "code", "factual", etc.
    difficulty: str  # "easy", "medium", "hard"
    tags: List[str]
```

### 2. TestRunner

```python
class TestRunner:
    """Executes tests using promptchain query command."""

    async def run_test(self,
                       test_case: TestCase,
                       strategy: StrategyConfig,
                       pattern: str) -> TestResult:
        """Run single test with unique session."""

        # Create unique session per test
        session_id = f"bench_{test_case.id}_{strategy.name}_{pattern}_{timestamp}"

        # Execute via CLI with --dev for full logging
        cmd = f"promptchain query \"{test_case.input_query}\" "
              f"--session {session_id} "
              f"--dev "
              f"--model {strategy.model}"

        result = await self.execute_command(cmd)

        return TestResult(
            test_case_id=test_case.id,
            session_id=session_id,
            strategy=strategy.name,
            pattern=pattern,
            output=result.stdout,
            execution_time_ms=result.duration,
            log_path=f"~/.promptchain/sessions/{session_id}/debug_*.log"
        )

    async def run_batch(self,
                        test_cases: List[TestCase],
                        strategies: List[StrategyConfig],
                        patterns: List[str],
                        parallel: int = 4) -> BatchResult:
        """Run batch of tests, optionally in parallel."""
```

### 3. LogParser

```python
class LogParser:
    """Parses debug logs from test runs."""

    def parse(self, log_path: str) -> ExecutionLog:
        """Parse debug log file into structured data."""

    def extract_metrics(self, log: ExecutionLog) -> Metrics:
        """Extract performance metrics from log."""
        return Metrics(
            total_tokens=log.total_tokens,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            llm_calls=log.llm_call_count,
            tool_calls=log.tool_call_count,
            reasoning_steps=log.step_count,
            errors=log.error_count
        )
```

### 4. Evaluator

```python
class Evaluator:
    """Scores test outputs against expected results."""

    def evaluate(self,
                 test_case: TestCase,
                 result: TestResult) -> EvaluationScore:
        """Score a single test result."""

        scores = {
            "keyword_match": self._keyword_score(test_case, result),
            "semantic_similarity": self._semantic_score(test_case, result),
            "llm_judge": self._llm_judge_score(test_case, result),
            "execution_success": 1.0 if result.success else 0.0
        }

        return EvaluationScore(
            test_case_id=test_case.id,
            scores=scores,
            overall=self._compute_overall(scores),
            passed=scores["overall"] >= self.pass_threshold
        )

    def _llm_judge_score(self, test_case: TestCase, result: TestResult) -> float:
        """Use LLM to judge output quality (expensive but accurate)."""
        prompt = f"""
        Question: {test_case.input_query}
        Expected Answer: {test_case.expected_output}
        Actual Answer: {result.output}

        Score the actual answer from 0.0 to 1.0 based on correctness and completeness.
        """
        # Call judge LLM
```

### 5. ReportGenerator

```python
class ReportGenerator:
    """Generates benchmark reports."""

    def generate(self,
                 batch_result: BatchResult,
                 evaluations: List[EvaluationScore]) -> BenchmarkReport:
        """Generate comprehensive report."""

    def to_markdown(self, report: BenchmarkReport) -> str:
        """Export as markdown."""

    def to_json(self, report: BenchmarkReport) -> dict:
        """Export as JSON for programmatic access."""

    def to_csv(self, report: BenchmarkReport) -> str:
        """Export as CSV for spreadsheet analysis."""

@dataclass
class BenchmarkReport:
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int

    # Per-strategy breakdown
    strategy_results: Dict[str, StrategyMetrics]

    # Per-pattern breakdown
    pattern_results: Dict[str, PatternMetrics]

    # Per-category breakdown
    category_results: Dict[str, CategoryMetrics]

    # Top/bottom performers
    best_strategy: str
    best_pattern: str
    worst_strategy: str
    worst_pattern: str

    # Cost analysis
    total_tokens: int
    estimated_cost: float

    # Detailed results
    all_results: List[TestResultWithScore]
```

## CLI Interface

### Main Benchmark Command

```bash
# Run full benchmark suite
promptchain benchmark --dataset ./datasets/reasoning.json

# Run with specific strategies
promptchain benchmark --dataset ./datasets/code.json \
    --strategies router,pipeline,broadcast \
    --patterns P01,P02,P05

# Run single pattern across all strategies
promptchain benchmark --dataset ./datasets/factual.json \
    --pattern P03 \
    --all-strategies

# Run single strategy across all patterns
promptchain benchmark --dataset ./datasets/reasoning.json \
    --strategy router-pipeline-full \
    --all-patterns

# Quick test with subset
promptchain benchmark --dataset ./datasets/all.json \
    --limit 10 \
    --quick  # Use fastest model

# Full matrix test (21 strategies x 14 patterns = 294 configs per test case!)
promptchain benchmark --dataset ./datasets/comprehensive.json \
    --full-matrix \
    --parallel 8
```

### Command Options

```
Options:
  --dataset PATH          Path to test dataset (required)
  --strategies LIST       Comma-separated strategy names
  --patterns LIST         Comma-separated pattern IDs (P01-P14)
  --all-strategies        Test all 21 strategy configurations
  --all-patterns          Test all 14 patterns
  --full-matrix           Test all strategies x all patterns
  --limit N               Limit test cases (for quick testing)
  --parallel N            Parallel test execution (default: 4)
  --model MODEL           Override model for all tests
  --output PATH           Output report path (default: ./benchmark_report.md)
  --format FORMAT         Report format: markdown, json, csv (default: markdown)
  --quick                 Use fastest model for quick iteration
  --judge-model MODEL     Model for LLM-as-judge evaluation
  --verbose               Show progress details
```

### Example Workflow

```bash
# 1. Create dataset
cat > datasets/reasoning.json << 'EOF'
[
  {
    "id": "R001",
    "input_query": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "expected_output": "No, we cannot conclude that some roses fade quickly. While all roses are flowers, the statement only says SOME flowers fade quickly - those could be non-rose flowers.",
    "expected_keywords": ["cannot conclude", "some", "logical fallacy"],
    "category": "reasoning",
    "difficulty": "medium",
    "tags": ["logic", "syllogism"]
  },
  {
    "id": "R002",
    "input_query": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "expected_output": "The ball costs $0.05 (5 cents). If the ball is X, the bat is X+$1. So X + (X+$1) = $1.10, meaning 2X = $0.10, so X = $0.05.",
    "expected_keywords": ["$0.05", "5 cents", "0.05"],
    "category": "reasoning",
    "difficulty": "easy",
    "tags": ["math", "puzzle"]
  }
]
EOF

# 2. Run benchmark with specific configuration
promptchain benchmark --dataset datasets/reasoning.json \
    --strategies router-pipeline-none,router-broadcast-critique \
    --patterns P01,P07,P11 \
    --output results/reasoning_benchmark.md

# 3. View results
cat results/reasoning_benchmark.md
```

## Dataset Format

### JSON Format

```json
[
  {
    "id": "unique_test_id",
    "input_query": "The question or task to execute",
    "expected_output": "The ideal response (for comparison)",
    "expected_keywords": ["word1", "word2"],
    "category": "reasoning|code|factual|creative|analysis",
    "difficulty": "easy|medium|hard",
    "tags": ["tag1", "tag2"],
    "metadata": {
      "source": "MMLU",
      "subject": "physics"
    }
  }
]
```

### JSONL Format (for large datasets)

```jsonl
{"id": "T001", "input_query": "...", "expected_output": "...", "category": "reasoning"}
{"id": "T002", "input_query": "...", "expected_output": "...", "category": "code"}
```

### CSV Format

```csv
id,input_query,expected_output,expected_keywords,category,difficulty,tags
T001,"What is 2+2?","4","4,four",reasoning,easy,"math,arithmetic"
T002,"Write a hello world in Python","print('Hello, World!')","print,Hello",code,easy,"python,basic"
```

## Report Output Format

### Markdown Report Example

```markdown
# PromptChain Benchmark Report

**Date:** 2025-11-30 14:30:00
**Dataset:** reasoning.json (50 test cases)
**Strategies Tested:** 5
**Patterns Tested:** 4

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 1000 |
| Passed | 847 |
| Failed | 153 |
| Pass Rate | 84.7% |
| Total Tokens | 2,450,000 |
| Estimated Cost | $12.35 |

## Best Performers

| Category | Winner | Score |
|----------|--------|-------|
| Best Strategy | router-pipeline-critique | 91.2% |
| Best Pattern | P07 (Reflection) | 89.5% |
| Best Category | factual | 95.1% |

## Strategy Breakdown

| Strategy | Pass Rate | Avg Tokens | Avg Time |
|----------|-----------|------------|----------|
| router-pipeline-none | 78.5% | 1,200 | 2.3s |
| router-pipeline-critique | 91.2% | 2,400 | 4.1s |
| router-broadcast-full | 85.3% | 3,100 | 5.2s |

## Pattern Breakdown

| Pattern | Pass Rate | Best Strategy |
|---------|-----------|---------------|
| P01 Single Agent | 72.1% | router-pipeline-critique |
| P07 Reflection | 89.5% | router-pipeline-full |
| P11 Critique-Revise | 88.2% | router-broadcast-critique |

## Category Breakdown

| Category | Pass Rate | Hardest Pattern |
|----------|-----------|-----------------|
| reasoning | 81.2% | P01 |
| code | 87.5% | P03 |
| factual | 95.1% | P01 |
| analysis | 79.8% | P02 |

## Failed Tests

| Test ID | Category | Strategy | Pattern | Error |
|---------|----------|----------|---------|-------|
| R023 | reasoning | router-pipeline-none | P01 | Incorrect logic |
| C045 | code | router-broadcast-full | P03 | Syntax error |
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `promptchain/benchmark/` module
- [ ] Implement DatasetLoader with JSON/JSONL/CSV support
- [ ] Implement TestRunner with CLI integration
- [ ] Create unique session-per-test mechanism

### Phase 2: Execution & Logging
- [ ] Implement LogParser for debug log parsing
- [ ] Extract metrics from execution logs
- [ ] Add parallel test execution support
- [ ] Implement timeout and retry mechanisms

### Phase 3: Evaluation
- [ ] Implement keyword matching scorer
- [ ] Add semantic similarity scorer (embeddings)
- [ ] Implement LLM-as-judge scorer
- [ ] Create composite scoring system

### Phase 4: Reporting
- [ ] Implement ReportGenerator
- [ ] Add markdown report output
- [ ] Add JSON/CSV export
- [ ] Create visualization support (charts)

### Phase 5: CLI Integration
- [ ] Add `promptchain benchmark` command
- [ ] Implement all CLI options
- [ ] Add progress reporting
- [ ] Support configuration files

### Phase 6: Optimization
- [ ] Add caching for repeated queries
- [ ] Implement incremental benchmarking
- [ ] Add cost tracking and budgets
- [ ] Create benchmark comparison tools

## File Structure

```
promptchain/
├── benchmark/
│   ├── __init__.py
│   ├── loader.py          # DatasetLoader
│   ├── runner.py          # TestRunner
│   ├── parser.py          # LogParser
│   ├── evaluator.py       # Evaluator
│   ├── reporter.py        # ReportGenerator
│   ├── models.py          # Data classes
│   └── strategies.py      # Strategy configurations
├── cli/
│   └── commands/
│       └── benchmark.py   # CLI command
└── datasets/              # Example datasets
    ├── reasoning.json
    ├── code.json
    └── factual.json
```

## Dependencies

- `click` - CLI framework (existing)
- `asyncio` - Parallel execution (existing)
- `pandas` - Data handling for reports
- `sentence-transformers` - Semantic similarity (optional)
- `matplotlib` / `plotly` - Visualization (optional)

## Cost Considerations

Full matrix testing is expensive:
- 21 strategies x 14 patterns = 294 configurations
- Per 100 test cases = 29,400 test runs
- Estimated tokens per run: ~2,000
- Total tokens: ~58.8M tokens
- Estimated cost varies by model (gpt-4.1-mini is recommended for screening)

Recommended approach:
1. Start with quick runs (P01 only, 3 strategies)
2. Identify promising configurations
3. Run full matrix only on final candidates
4. Use cheaper models for initial screening
