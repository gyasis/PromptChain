#!/usr/bin/env python3
"""
Focused Bug Analysis for Athena LightRAG MCP Server
====================================================
Quick analysis of critical bugs identified during adversarial testing.
Focuses on high-confidence, high-severity issues for immediate remediation.

Author: Adversarial Bug Hunter Agent
Date: 2025-09-08
"""

import asyncio
import json
import logging
import traceback
import sys
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

# Add parent directories to path
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena_mcp_server import create_manual_mcp_server

logger = logging.getLogger(__name__)

@dataclass
class CriticalBug:
    """Critical bug report in required JSON format."""
    title: str
    severity: str
    confidence: str
    category: str
    location: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    root_cause_hypothesis: str
    blast_radius: str
    fix_suggestion: List[str]
    next_actions: List[str]
    links: List[str]
    labels: List[str]

async def analyze_critical_bugs():
    """Analyze critical bugs found during testing."""
    bugs = []
    
    # ============= BUG 1: AgenticStepProcessor Parameter Mismatch =============
    bugs.append(CriticalBug(
        title="AgenticStepProcessor parameter incompatibility breaks multi-hop reasoning",
        severity="Blocker",
        confidence="High",
        category="Contract",
        location=[{
            "file": "agentic_lightrag.py",
            "line": 382,
            "symbol": "AgenticStepProcessor"
        }],
        evidence={
            "type": "RuntimeTrace",
            "details": "AgenticStepProcessor.__init__() got an unexpected keyword argument 'additional_instructions'",
            "repro_steps": [
                "Call lightrag_multi_hop_reasoning with any parameters",
                "AgenticStepProcessor constructor fails with unexpected keyword",
                "Multi-hop reasoning completely broken"
            ]
        },
        root_cause_hypothesis="API drift between agentic_lightrag.py and AgenticStepProcessor constructor. Code uses 'additional_instructions' parameter that doesn't exist in AgenticStepProcessor.__init__",
        blast_radius="Complete failure of multi-hop reasoning tool - core MCP functionality broken",
        fix_suggestion=[
            "Check AgenticStepProcessor constructor signature",
            "Remove 'additional_instructions' parameter or map to correct parameter",
            "Update create_multi_hop_processor() method in agentic_lightrag.py"
        ],
        next_actions=[
            "Examine AgenticStepProcessor.__init__ signature",
            "Test corrected parameter usage",
            "Verify multi-hop reasoning works after fix"
        ],
        links=[
            "/home/gyasis/Documents/code/PromptChain/athena-lightrag/agentic_lightrag.py:382"
        ],
        labels=["API_DRIFT", "CONSTRUCTOR_MISMATCH", "BLOCKER"]
    ))
    
    # ============= BUG 2: Token Limit Boundary Violations =============
    bugs.append(CriticalBug(
        title="Token limit boundary validation allows negative and dangerous values",
        severity="Major",
        confidence="High", 
        category="I/O",
        location=[{
            "file": "athena_mcp_server.py", 
            "line": 189,
            "symbol": "_handle_local_query"
        }],
        evidence={
            "type": "Static",
            "details": "No input validation for max_entity_tokens, max_relation_tokens parameters",
            "repro_steps": [
                "Send lightrag_local_query with max_entity_tokens=-1",
                "Send lightrag_global_query with max_relation_tokens=2^31", 
                "Observe acceptance of invalid values"
            ]
        },
        root_cause_hypothesis="MCP tool handlers accept Pydantic-validated parameters but don't enforce business logic constraints on token limits",
        blast_radius="Potential resource exhaustion, unexpected behavior with extreme values, possible integer overflow",
        fix_suggestion=[
            "Add input validation in tool handlers: validate 1 <= tokens <= 100000",
            "Handle integer overflow gracefully",
            "Return error responses for invalid ranges"
        ],
        next_actions=[
            "Add validation to all token parameter handlers",
            "Test boundary conditions after fix",
            "Document valid token ranges"
        ],
        links=[
            "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_mcp_server.py"
        ],
        labels=["INPUT_VALIDATION", "BOUNDARY_VIOLATION"]
    ))
    
    # ============= BUG 3: Performance Degradation & Resource Exhaustion =============
    bugs.append(CriticalBug(
        title="Severe performance degradation under normal load causes timeout failures",
        severity="Critical",
        confidence="High",
        category="Integration",
        location=[{
            "file": "lightrag_core.py",
            "line": 125,
            "symbol": "LightRAG.__init__"
        }],
        evidence={
            "type": "RuntimeTrace",
            "details": "Test suite timeout at 2m 0.0s during basic parameter testing",
            "repro_steps": [
                "Run adversarial_test_suite.py",
                "Observe 2-minute timeout during lightweight parameter tests",
                "Multiple LightRAG reinitializations visible in logs"
            ]
        },
        root_cause_hypothesis="Each MCP tool call creates new LightRAG instance, loading full vector databases (~1839+3035+1282 vectors) causing severe initialization overhead",
        blast_radius="MCP server unusable under production load, timeout failures, resource exhaustion",
        fix_suggestion=[
            "Implement singleton pattern for LightRAG instances",
            "Cache initialized LightRAG objects across tool calls", 
            "Lazy loading of vector databases",
            "Connection pooling for database access"
        ],
        next_actions=[
            "Profile memory and CPU usage during tool calls",
            "Implement LightRAG instance caching",
            "Benchmark performance improvement after caching"
        ],
        links=[
            "/home/gyasis/Documents/code/PromptChain/athena-lightrag/lightrag_core.py",
            "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_mcp_server.py"
        ],
        labels=["PERFORMANCE_CRITICAL", "RESOURCE_EXHAUSTION", "TIMEOUT"]
    ))
    
    # ============= BUG 4: Missing Rerank Model Configuration =============
    bugs.append(CriticalBug(
        title="Missing rerank model configuration degrades query quality",
        severity="Major",
        confidence="High",
        category="Logic",
        location=[{
            "file": "lightrag_core.py",
            "line": 125,
            "symbol": "LightRAG.__init__"
        }],
        evidence={
            "type": "RuntimeTrace", 
            "details": "Rerank is enabled but no rerank_model_func provided. Reranking will be skipped.",
            "repro_steps": [
                "Execute any LightRAG query",
                "Observe rerank warning in logs",
                "Query results lack proper ranking"
            ]
        },
        root_cause_hypothesis="LightRAG initialization enables reranking by default but no rerank_model_func is provided in configuration",
        blast_radius="Degraded query result quality, suboptimal context retrieval for multi-hop reasoning",
        fix_suggestion=[
            "Add rerank_model_func to LightRAG initialization",
            "Or explicitly disable reranking with enable_rerank=False",
            "Use cross-encoder reranking model for better results"
        ],
        next_actions=[
            "Research compatible reranking models",
            "Implement rerank function or disable feature",
            "Test query quality with/without reranking"
        ],
        links=[
            "/home/gyasis/Documents/code/PromptChain/athena-lightrag/lightrag_core.py:125"
        ],
        labels=["CONFIGURATION_MISSING", "QUALITY_DEGRADATION"]
    ))
    
    return bugs

async def test_specific_failures():
    """Test specific failure scenarios quickly."""
    try:
        server = create_manual_mcp_server()
        issues = []
        
        # Test 1: Multi-hop reasoning failure
        try:
            result = await server.call_tool("lightrag_multi_hop_reasoning", {
                "query": "test query"
            })
            if not result.get("success"):
                issues.append(f"Multi-hop reasoning failed: {result.get('error')}")
        except Exception as e:
            issues.append(f"Multi-hop reasoning exception: {str(e)}")
        
        # Test 2: Token boundary validation
        try:
            result = await server.call_tool("lightrag_local_query", {
                "query": "test",
                "max_entity_tokens": -1
            })
            if result.get("success"):
                issues.append("Negative token limit accepted")
        except Exception:
            # Expected to fail
            pass
        
        # Test 3: Performance - simple query should complete quickly  
        import time
        start = time.time()
        try:
            result = await asyncio.wait_for(
                server.call_tool("lightrag_local_query", {"query": "test"}),
                timeout=10.0
            )
            elapsed = time.time() - start
            if elapsed > 5.0:
                issues.append(f"Simple query took {elapsed:.1f}s - performance issue")
        except asyncio.TimeoutError:
            issues.append("Simple query timeout - severe performance issue")
        
        return issues
        
    except Exception as e:
        return [f"Setup failed: {str(e)}"]

async def main():
    """Generate focused bug report."""
    print("=" * 80)
    print("FOCUSED BUG ANALYSIS - Athena LightRAG MCP Server")
    print("=" * 80)
    
    # Analyze critical bugs
    bugs = await analyze_critical_bugs()
    
    # Test specific failures
    print("\nTesting specific failure scenarios...")
    issues = await test_specific_failures()
    
    print(f"\n=== EXECUTIVE SUMMARY ===")
    print(f"Critical Bugs Identified: {len(bugs)}")
    print(f"Runtime Issues Confirmed: {len(issues)}")
    
    for issue in issues:
        print(f"  ⚠️  {issue}")
    
    print(f"\n=== CRITICAL BUG REPORTS (JSON FORMAT) ===")
    for i, bug in enumerate(bugs, 1):
        print(f"\n--- Bug Report #{i} ---")
        print(json.dumps(bug.__dict__, indent=2))
    
    # Save bug reports
    bugs_data = [bug.__dict__ for bug in bugs]
    bug_file = Path("critical_bugs.json")
    bug_file.write_text(json.dumps(bugs_data, indent=2))
    
    print(f"\n=== REMEDIATION PRIORITY ===")
    blocker_bugs = [b for b in bugs if b.severity == "Blocker"]
    critical_bugs = [b for b in bugs if b.severity == "Critical"]
    
    print(f"IMMEDIATE ACTION REQUIRED:")
    print(f"  🔴 Blocker: {len(blocker_bugs)} (system unusable)")
    print(f"  🟠 Critical: {len(critical_bugs)} (major functionality broken)")
    
    print(f"\nNext Steps:")
    print(f"1. Fix AgenticStepProcessor parameter mismatch (BLOCKER)")
    print(f"2. Implement LightRAG instance caching (CRITICAL performance)")
    print(f"3. Add token validation (MAJOR security/stability)")
    print(f"4. Configure rerank model (MAJOR quality)")
    
    print(f"\nBug reports saved to: {bug_file.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())