#!/usr/bin/env python3
"""
Quick test of visual chat system with Candle library query
"""

import asyncio
from visual_output import ChatVisualizer

# Sample Candle library markdown response (from previous research)
candle_markdown = """# Candle Library - Overview

**Candle** is a minimalist machine learning framework for Rust, developed by Hugging Face. Think of it as "PyTorch for Rust" - a lightweight alternative focused on performance, simplicity, and serverless deployment.

## Purpose

- **Primary Goal**: Enable ML inference and training in Rust with minimal dependencies
- **Target Use Cases**:
  - Production ML deployments
  - Serverless/edge computing
  - Performance-critical applications
  - WASM-compatible ML models

## Key Features

### 1. **Pure Rust Implementation**
- No Python dependencies required
- CPU and GPU (CUDA/Metal) support
- Compile to WebAssembly

### 2. **Minimal & Fast**
- Lightweight compared to PyTorch/TensorFlow
- Lower memory footprint
- Fast compilation times

### 3. **PyTorch-like API**
- Familiar tensor operations
- Automatic differentiation
- Model building abstractions

## Basic Example

```rust
use candle_core::{Tensor, Device, DType};

fn main() -> Result<()> {
    // Create a tensor on CPU
    let device = Device::Cpu;

    // Create a 2x3 tensor
    let tensor = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;

    // Basic operations
    let sum = tensor.sum_all()?;
    let doubled = (tensor * 2.0)?;

    println!("Sum: {:?}", sum.to_scalar::<f32>()?);
    Ok(())
}
```

## Quick Comparison

| Feature | Candle | PyTorch |
|---------|--------|---------|
| Language | Rust | Python/C++ |
| Size | ~5-10 MB | ~500+ MB |
| Startup | Fast | Slower |
| WASM Support | ✅ Yes | ❌ No |

## When to Use Candle

- ✅ Building production ML services in Rust
- ✅ Need minimal dependencies and fast startup
- ✅ Deploying to edge/serverless/WASM
- ❌ Research/experimentation (PyTorch still better)

**Bottom Line**: Candle brings practical machine learning to Rust, prioritizing performance and deployment over research flexibility.
"""

async def demo_visual_chat():
    viz = ChatVisualizer()

    # Demo header
    viz.render_header(
        "AGENTIC CHAT TEAM",
        "6-Agent Collaborative System with Multi-Hop Reasoning"
    )

    # Demo team roster
    agents = {
        "research": {},
        "analysis": {},
        "coding": {},
        "terminal": {},
        "documentation": {},
        "synthesis": {}
    }
    viz.render_team_roster(agents)

    # Demo commands
    viz.render_commands_help()

    # Simulate user input
    print("\n")
    viz.render_user_input("What is Rust's Candle library?")

    # Simulate thinking
    viz.render_thinking_indicator("research")
    print()

    # Simulate agent response with rich markdown
    viz.render_agent_response("research", candle_markdown)

    # Demo stats
    viz.render_stats({
        "Session Duration": "2 minutes",
        "Total Queries": 1,
        "History Size": "5,240 / 180,000 tokens",
        "History Usage": "2.9%"
    })

    # Demo success message
    viz.render_system_message("Query completed successfully! Rich markdown rendering active.", "success")

if __name__ == "__main__":
    asyncio.run(demo_visual_chat())
