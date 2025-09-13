# MCP Tool Hijacker Examples Location

The MCP Tool Hijacker examples have been organized and are located at:

```
examples/mcp_hijacker_demos/
├── README.md                           # Complete guide and overview
├── simple_hijacker_test.py            # Quick start and basic concept
├── hijacker_deeplake_example.py       # Main DeepLake RAG integration
├── example_hijacker_deeplake_demo.py   # Comprehensive demo
├── working_hijacker_demo.py           # Real-world integration
├── real_hijacker_demo.py              # Performance comparison
└── quick_hijacker_demo.py             # Fast setup guide
```

## Quick Access

**Primary example to run:**
```bash
cd examples/mcp_hijacker_demos/
python hijacker_deeplake_example.py
```

**For your specific use case (DeepLake RAG with neural network optimization):**
```bash
python simple_hijacker_test.py
```

## Key Features Demonstrated

✅ **Sub-100ms tool execution** (vs 500-2000ms traditional)  
✅ **Step-to-step parameter chaining** (NEW!) - dynamic parameters from previous outputs  
✅ **JSON output parsing** (NEW!) - extract specific keys for next steps  
✅ **Template syntax** - `{previous.results[0].id}`, `{step_1.metadata.title}`  
✅ **Static parameter management** (set once, reuse everywhere)  
✅ **Batch processing** with concurrency control  
✅ **Performance monitoring** and statistics  
✅ **DeepLake RAG integration** for context retrieval  
✅ **Non-breaking integration** with existing PromptChain  

## Documentation

- **API Documentation**: `docs/mcp_tool_hijacker.md` (🆕 **Fully Enhanced with Step Chaining!**)
- **Examples Guide**: `examples/mcp_hijacker_demos/README.md`
- **Step Chaining Demo**: `examples/mcp_hijacker_demos/step_chaining_example.py` (🆕)
- **Pull Request**: https://github.com/gyasis/PromptChain/pull/1

## Usage Patterns

### Basic Hijacker Usage
```python
chain = PromptChain(
    models=["openai/gpt-4o-mini"],
    instructions=["Analyze: {input}"],
    enable_mcp_hijacker=True  # 🔑 Enable hijacker
)

async with chain:
    # Set static params once
    chain.mcp_hijacker.set_static_params(
        "mcp__deeplake__retrieve_context",
        n_results="5",
        include_embeddings=False
    )
    
    # Lightning-fast execution
    result = await chain.mcp_hijacker.call_tool(
        "mcp__deeplake__retrieve_context",
        query="neural network optimization"
    )
```

### Step Chaining Usage (NEW!)
```python
# Step 1: Search
result1 = await hijacker.call_tool("search_tool", query="optimization")
hijacker.store_step_output(1, result1, "search_step")

# Step 2: Use Step 1's output as input
hijacker.param_manager.set_parameter_template(
    "detail_tool",
    "document_id", 
    "{previous.results[0].id}"  # Dynamic parameter from previous step
)

result2 = await hijacker.call_tool_with_chaining(
    "detail_tool", 
    current_step=2
)
```

---

**Note**: The `examples/` directory may be gitignored locally, but the hijacker implementation and documentation are fully tracked in the repository.