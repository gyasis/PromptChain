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
✅ **Static parameter management** (set once, reuse everywhere)  
✅ **Batch processing** with concurrency control  
✅ **Performance monitoring** and statistics  
✅ **DeepLake RAG integration** for context retrieval  
✅ **Non-breaking integration** with existing PromptChain  

## Documentation

- **API Documentation**: `docs/mcp_tool_hijacker.md`
- **Examples Guide**: `examples/mcp_hijacker_demos/README.md`
- **Pull Request**: https://github.com/gyasis/PromptChain/pull/1

## Usage Pattern

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

---

**Note**: The `examples/` directory may be gitignored locally, but the hijacker implementation and documentation are fully tracked in the repository.