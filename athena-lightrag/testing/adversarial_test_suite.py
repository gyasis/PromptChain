#!/usr/bin/env python3
"""
Adversarial Test Suite for Athena LightRAG MCP Server
======================================================
Comprehensive adversarial testing for MCP tools with PromptChain → LightRAG integration.
Tests parameter boundaries, error conditions, security vulnerabilities, and performance limits.

Author: Adversarial Bug Hunter Agent
Date: 2025-09-08
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import random
import string

# Add parent directories to path
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena_mcp_server import (
    AthenaMCPServer, ManualMCPServer, create_athena_mcp_server, 
    create_manual_mcp_server, LocalQueryParams, GlobalQueryParams,
    HybridQueryParams, ContextExtractParams, MultiHopReasoningParams,
    SQLGenerationParams
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Result of an adversarial test."""
    test_id: str
    category: str
    severity: str  # Blocker|Critical|Major|Minor|Trivial
    confidence: str  # High|Medium|Low
    passed: bool
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    reproduction_steps: List[str] = field(default_factory=list)
    fix_suggestion: List[str] = field(default_factory=list)
    execution_time: float = 0.0

@dataclass 
class BugReport:
    """Adversarial bug report in required JSON format."""
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
    links: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

class AdversarialTestSuite:
    """Comprehensive adversarial testing for Athena MCP Server."""
    
    def __init__(self, working_dir: Optional[str] = None):
        """Initialize test suite."""
        self.working_dir = working_dir or "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"
        self.test_results: List[TestResult] = []
        self.bug_reports: List[BugReport] = []
        self.server: Optional[ManualMCPServer] = None
        
    async def setup(self):
        """Setup test environment."""
        try:
            self.server = create_manual_mcp_server(working_dir=self.working_dir)
            logger.info("Test environment setup complete")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    async def teardown(self):
        """Cleanup test environment."""
        logger.info("Test environment teardown complete")
    
    # ============= CATEGORY I3.1: Integration Testing =============
    
    async def test_promptchain_orchestration(self) -> TestResult:
        """Test PromptChain orchestration with LightRAG tools."""
        test_id = "I3.1.001"
        start_time = time.time()
        
        try:
            # Test multi-hop reasoning which uses PromptChain internally
            params = {
                "query": "Complex query requiring PromptChain orchestration",
                "objective": "Test PromptChain integration",
                "max_steps": 3
            }
            
            result = await self.server.call_tool("lightrag_multi_hop_reasoning", params)
            
            # Validate PromptChain integration
            if not result.get("success"):
                return TestResult(
                    test_id=test_id,
                    category="Integration",
                    severity="Critical",
                    confidence="High",
                    passed=False,
                    error=f"PromptChain orchestration failed: {result.get('error')}",
                    details=result,
                    execution_time=time.time() - start_time
                )
            
            # Check for reasoning steps (indicates PromptChain worked)
            if not result.get("reasoning_steps"):
                return TestResult(
                    test_id=test_id,
                    category="Integration",
                    severity="Major",
                    confidence="High",
                    passed=False,
                    error="No reasoning steps generated - PromptChain may not be executing",
                    details=result,
                    execution_time=time.time() - start_time
                )
            
            return TestResult(
                test_id=test_id,
                category="Integration",
                severity="",
                confidence="High",
                passed=True,
                details={"reasoning_steps": len(result.get("reasoning_steps", []))},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                category="Integration",
                severity="Blocker",
                confidence="High",
                passed=False,
                error=str(e),
                details={"traceback": traceback.format_exc()},
                execution_time=time.time() - start_time
            )
    
    async def test_tool_parameter_validation(self) -> TestResult:
        """Test parameter validation for all MCP tools."""
        test_id = "I3.1.002"
        start_time = time.time()
        errors = []
        
        test_cases = [
            # Missing required parameters
            ("lightrag_local_query", {}, "Missing 'query' parameter"),
            ("lightrag_global_query", {"top_k": 10}, "Missing 'query' parameter"),
            
            # Invalid parameter types
            ("lightrag_local_query", {"query": 123}, "Invalid query type"),
            ("lightrag_hybrid_query", {"query": "test", "top_k": "not_a_number"}, "Invalid top_k type"),
            
            # Invalid enum values
            ("lightrag_context_extract", {"query": "test", "mode": "invalid_mode"}, "Invalid mode enum"),
        ]
        
        for tool_name, params, expected_error in test_cases:
            try:
                result = await self.server.call_tool(tool_name, params)
                if result.get("success"):
                    errors.append(f"{tool_name}: Expected validation error for {expected_error}")
            except Exception as e:
                # Expected behavior - parameter validation should fail
                pass
        
        if errors:
            return TestResult(
                test_id=test_id,
                category="Integration",
                severity="Major",
                confidence="High",
                passed=False,
                error="Parameter validation bypass detected",
                details={"validation_failures": errors},
                execution_time=time.time() - start_time
            )
        
        return TestResult(
            test_id=test_id,
            category="Integration",
            severity="",
            confidence="High",
            passed=True,
            execution_time=time.time() - start_time
        )
    
    # ============= CATEGORY I3.2: Multi-Hop Reasoning Validation =============
    
    async def test_reasoning_depth_limits(self) -> TestResult:
        """Test multi-hop reasoning with different depth parameters."""
        test_id = "I3.2.001"
        start_time = time.time()
        
        depth_tests = [
            (1, "Minimal depth"),
            (3, "Default depth"),
            (8, "Maximum safe depth"),
            (20, "Excessive depth - should be limited"),
        ]
        
        results = []
        for depth, description in depth_tests:
            try:
                params = {
                    "query": f"Test query for {description}",
                    "max_steps": depth
                }
                
                result = await self.server.call_tool("lightrag_multi_hop_reasoning", params)
                results.append({
                    "depth": depth,
                    "description": description,
                    "success": result.get("success"),
                    "steps_executed": len(result.get("reasoning_steps", [])),
                    "tokens_used": result.get("total_tokens_used", 0)
                })
                
            except Exception as e:
                results.append({
                    "depth": depth,
                    "description": description,
                    "error": str(e)
                })
        
        # Check for depth limit enforcement
        excessive_test = results[-1]
        if excessive_test.get("steps_executed", 0) > 10:
            return TestResult(
                test_id=test_id,
                category="Logic",
                severity="Major",
                confidence="High",
                passed=False,
                error="No upper limit on reasoning depth - potential infinite loop risk",
                details={"depth_results": results},
                reproduction_steps=[
                    "Call lightrag_multi_hop_reasoning with max_steps=20",
                    "Observe unbounded reasoning steps"
                ],
                fix_suggestion=["Implement hard limit on max_steps (e.g., 10)"],
                execution_time=time.time() - start_time
            )
        
        return TestResult(
            test_id=test_id,
            category="Logic",
            severity="",
            confidence="High",
            passed=True,
            details={"depth_results": results},
            execution_time=time.time() - start_time
        )
    
    async def test_circular_reasoning_detection(self) -> TestResult:
        """Test detection of circular reasoning patterns."""
        test_id = "I3.2.002"
        start_time = time.time()
        
        # Query designed to potentially cause circular reasoning
        circular_query = "What causes X which is caused by Y which is caused by X?"
        
        try:
            params = {
                "query": circular_query,
                "objective": "Detect circular dependencies",
                "max_steps": 5
            }
            
            result = await self.server.call_tool("lightrag_multi_hop_reasoning", params)
            
            # Check if reasoning steps show repetition
            steps = result.get("reasoning_steps", [])
            if len(steps) > len(set(steps)):
                return TestResult(
                    test_id=test_id,
                    category="Logic",
                    severity="Major",
                    confidence="Medium",
                    passed=False,
                    error="Circular reasoning pattern detected",
                    details={"repeated_steps": steps},
                    reproduction_steps=[
                        f"Execute query: {circular_query}",
                        "Observe repeated reasoning steps"
                    ],
                    fix_suggestion=["Implement circular dependency detection", "Add visited state tracking"],
                    execution_time=time.time() - start_time
                )
            
            return TestResult(
                test_id=test_id,
                category="Logic",
                severity="",
                confidence="High",
                passed=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                category="Logic",
                severity="Critical",
                confidence="High",
                passed=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    # ============= CATEGORY I3.3: Parameter Boundary Testing =============
    
    async def test_token_limit_boundaries(self) -> TestResult:
        """Test token limit parameters at boundaries."""
        test_id = "I3.3.001"
        start_time = time.time()
        issues = []
        
        boundary_tests = [
            ("lightrag_local_query", {"query": "test", "max_entity_tokens": -1}, "Negative tokens"),
            ("lightrag_local_query", {"query": "test", "max_entity_tokens": 0}, "Zero tokens"),
            ("lightrag_local_query", {"query": "test", "max_entity_tokens": 999999}, "Excessive tokens"),
            ("lightrag_global_query", {"query": "test", "max_relation_tokens": 2**31}, "Integer overflow"),
        ]
        
        for tool_name, params, test_type in boundary_tests:
            try:
                result = await self.server.call_tool(tool_name, params)
                
                # Check for proper boundary handling
                if params.get("max_entity_tokens", 0) < 0 and result.get("success"):
                    issues.append(f"{test_type}: Accepted negative token limit")
                
                if params.get("max_entity_tokens", 1) == 0 and result.get("result"):
                    issues.append(f"{test_type}: Generated output with zero token limit")
                    
            except Exception as e:
                # Some boundaries should cause errors
                if "overflow" in test_type.lower() and "overflow" not in str(e).lower():
                    issues.append(f"{test_type}: No overflow protection")
        
        if issues:
            return TestResult(
                test_id=test_id,
                category="I/O",
                severity="Major",
                confidence="High",
                passed=False,
                error="Token limit boundary violations",
                details={"issues": issues},
                reproduction_steps=[
                    "Set max_entity_tokens to negative value",
                    "Set max_relation_tokens to 2^31",
                    "Observe improper handling"
                ],
                fix_suggestion=[
                    "Add input validation for token limits",
                    "Enforce reasonable bounds (e.g., 1-100000)",
                    "Handle integer overflow gracefully"
                ],
                execution_time=time.time() - start_time
            )
        
        return TestResult(
            test_id=test_id,
            category="I/O",
            severity="",
            confidence="High",
            passed=True,
            execution_time=time.time() - start_time
        )
    
    async def test_string_injection_attacks(self) -> TestResult:
        """Test for SQL injection and command injection vulnerabilities."""
        test_id = "I3.3.002"
        start_time = time.time()
        vulnerabilities = []
        
        injection_payloads = [
            "'; DROP TABLE patients; --",
            "' OR '1'='1",
            "$(rm -rf /)",
            "` && cat /etc/passwd",
            "../../../etc/passwd",
            "\\x00\\x01\\x02",
            "<script>alert('XSS')</script>",
        ]
        
        for payload in injection_payloads:
            try:
                # Test SQL generation tool
                sql_result = await self.server.call_tool("lightrag_sql_generation", {
                    "natural_query": payload,
                    "include_explanation": False
                })
                
                if sql_result.get("success") and any(danger in str(sql_result.get("sql", "")).lower() 
                                                     for danger in ["drop", "delete", "exec", "system"]):
                    vulnerabilities.append(f"SQL injection: {payload[:50]}...")
                
                # Test other query tools
                query_result = await self.server.call_tool("lightrag_local_query", {
                    "query": payload
                })
                
                if query_result.get("success") and any(danger in str(query_result.get("result", "")).lower()
                                                       for danger in ["passwd", "shadow", "script"]):
                    vulnerabilities.append(f"Query injection: {payload[:50]}...")
                    
            except Exception as e:
                # Errors are expected for malicious inputs
                pass
        
        if vulnerabilities:
            return TestResult(
                test_id=test_id,
                category="SecurityAdj",
                severity="Critical",
                confidence="High",
                passed=False,
                error="Injection vulnerabilities detected",
                details={"vulnerabilities": vulnerabilities},
                reproduction_steps=[
                    "Send SQL injection payload to lightrag_sql_generation",
                    "Send command injection to query tools",
                    "Check for unsanitized execution"
                ],
                fix_suggestion=[
                    "Implement input sanitization",
                    "Use parameterized queries",
                    "Escape special characters",
                    "Validate against injection patterns"
                ],
                execution_time=time.time() - start_time
            )
        
        return TestResult(
            test_id=test_id,
            category="SecurityAdj",
            severity="",
            confidence="High", 
            passed=True,
            execution_time=time.time() - start_time
        )
    
    # ============= CATEGORY I3.4: Performance Testing =============
    
    async def test_concurrent_tool_calls(self) -> TestResult:
        """Test concurrent execution of multiple tool calls."""
        test_id = "I3.4.001"
        start_time = time.time()
        
        async def make_concurrent_calls(num_calls: int):
            """Make concurrent tool calls."""
            tasks = []
            for i in range(num_calls):
                tool = random.choice(["lightrag_local_query", "lightrag_global_query", "lightrag_hybrid_query"])
                params = {"query": f"Concurrent test query {i}"}
                tasks.append(self.server.call_tool(tool, params))
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        try:
            # Test with increasing concurrency
            concurrency_levels = [5, 10, 20]
            performance_issues = []
            
            for level in concurrency_levels:
                level_start = time.time()
                results = await make_concurrent_calls(level)
                level_time = time.time() - level_start
                
                errors = [r for r in results if isinstance(r, Exception)]
                failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
                
                if errors or failed:
                    performance_issues.append({
                        "concurrency": level,
                        "errors": len(errors),
                        "failures": len(failed),
                        "time": level_time
                    })
                
                # Check for linear vs exponential time growth
                if level > 5 and level_time > level * 2:
                    performance_issues.append({
                        "issue": "Non-linear performance degradation",
                        "concurrency": level,
                        "time": level_time
                    })
            
            if performance_issues:
                return TestResult(
                    test_id=test_id,
                    category="Integration",
                    severity="Major",
                    confidence="High",
                    passed=False,
                    error="Concurrency handling issues detected",
                    details={"issues": performance_issues},
                    reproduction_steps=[
                        "Execute 20 concurrent tool calls",
                        "Observe errors or performance degradation"
                    ],
                    fix_suggestion=[
                        "Implement connection pooling",
                        "Add rate limiting",
                        "Use async locks for shared resources"
                    ],
                    execution_time=time.time() - start_time
                )
            
            return TestResult(
                test_id=test_id,
                category="Integration",
                severity="",
                confidence="High",
                passed=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                category="Integration",
                severity="Critical",
                confidence="High",
                passed=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def test_memory_pressure(self) -> TestResult:
        """Test behavior under memory pressure with large contexts."""
        test_id = "I3.4.002"
        start_time = time.time()
        
        # Generate large query to stress memory
        large_query = " ".join([f"complex medical term {i}" for i in range(1000)])
        
        try:
            params = {
                "query": large_query,
                "max_entity_tokens": 50000,
                "max_relation_tokens": 50000,
                "top_k": 200
            }
            
            result = await self.server.call_tool("lightrag_hybrid_query", params)
            
            # Check for memory-related issues
            if not result.get("success"):
                error_msg = str(result.get("error", "")).lower()
                if any(mem_err in error_msg for mem_err in ["memory", "oom", "allocation"]):
                    return TestResult(
                        test_id=test_id,
                        category="I/O",
                        severity="Major",
                        confidence="High",
                        passed=False,
                        error="Memory management issue under pressure",
                        details={"error": result.get("error")},
                        reproduction_steps=[
                            "Create query with 1000+ terms",
                            "Set max tokens to 50000+",
                            "Execute hybrid query"
                        ],
                        fix_suggestion=[
                            "Implement memory usage monitoring",
                            "Add query size limits",
                            "Stream large results instead of loading all"
                        ],
                        execution_time=time.time() - start_time
                    )
            
            return TestResult(
                test_id=test_id,
                category="I/O",
                severity="",
                confidence="High",
                passed=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            if "memory" in str(e).lower():
                return TestResult(
                    test_id=test_id,
                    category="I/O",
                    severity="Critical",
                    confidence="High",
                    passed=False,
                    error=f"Memory exception: {str(e)}",
                    execution_time=time.time() - start_time
                )
            raise
    
    # ============= CATEGORY I3.5: Edge Cases and Error Recovery =============
    
    async def test_empty_and_null_inputs(self) -> TestResult:
        """Test handling of empty, null, and edge case inputs."""
        test_id = "I3.5.001"
        start_time = time.time()
        issues = []
        
        edge_cases = [
            ("", "Empty string"),
            (" ", "Whitespace only"),
            ("\n\t\r", "Whitespace characters"),
            ("a" * 10000, "Very long query"),
            ("🎭🔬💊", "Unicode/emoji"),
            ("\x00\x01", "Control characters"),
        ]
        
        for query, description in edge_cases:
            try:
                result = await self.server.call_tool("lightrag_local_query", {"query": query})
                
                if query in ["", " "] and result.get("success"):
                    issues.append(f"{description}: Should reject empty queries")
                
                if "\x00" in query and result.get("success"):
                    issues.append(f"{description}: Should sanitize control characters")
                    
            except Exception as e:
                # Some edge cases should cause errors
                if description == "Very long query" and "timeout" not in str(e).lower():
                    issues.append(f"{description}: No timeout protection")
        
        if issues:
            return TestResult(
                test_id=test_id,
                category="EdgeCase",
                severity="Major",
                confidence="High",
                passed=False,
                error="Edge case handling failures",
                details={"issues": issues},
                reproduction_steps=[
                    "Send empty string query",
                    "Send control characters",
                    "Send 10000 character query"
                ],
                fix_suggestion=[
                    "Validate non-empty queries",
                    "Strip/sanitize control characters",
                    "Implement query size limits"
                ],
                execution_time=time.time() - start_time
            )
        
        return TestResult(
            test_id=test_id,
            category="EdgeCase",
            severity="",
            confidence="High",
            passed=True,
            execution_time=time.time() - start_time
        )
    
    async def test_error_recovery_mechanisms(self) -> TestResult:
        """Test error recovery and retry mechanisms."""
        test_id = "I3.5.002"
        start_time = time.time()
        
        # Simulate various failure scenarios
        failure_scenarios = [
            ("lightrag_local_query", {"query": "test", "top_k": -1}, "Invalid parameter recovery"),
            ("lightrag_multi_hop_reasoning", {"query": "test", "max_steps": 0}, "Zero steps recovery"),
        ]
        
        recovery_issues = []
        for tool, params, scenario in failure_scenarios:
            try:
                # First call with bad params
                result1 = await self.server.call_tool(tool, params)
                
                # Try valid call immediately after
                valid_params = {"query": "valid test query"}
                result2 = await self.server.call_tool(tool, valid_params)
                
                if not result2.get("success"):
                    recovery_issues.append(f"{scenario}: Failed to recover after error")
                    
            except Exception as e:
                recovery_issues.append(f"{scenario}: Exception during recovery test: {str(e)}")
        
        if recovery_issues:
            return TestResult(
                test_id=test_id,
                category="Integration",
                severity="Major",
                confidence="Medium",
                passed=False,
                error="Error recovery mechanism failures",
                details={"issues": recovery_issues},
                reproduction_steps=[
                    "Send invalid request",
                    "Send valid request immediately after",
                    "Check if system recovered properly"
                ],
                fix_suggestion=[
                    "Implement proper error state cleanup",
                    "Add retry mechanisms with backoff",
                    "Ensure errors don't corrupt state"
                ],
                execution_time=time.time() - start_time
            )
        
        return TestResult(
            test_id=test_id,
            category="Integration",
            severity="",
            confidence="High",
            passed=True,
            execution_time=time.time() - start_time
        )
    
    # ============= Main Test Execution =============
    
    async def run_all_tests(self) -> Tuple[List[TestResult], List[BugReport]]:
        """Run all adversarial tests."""
        await self.setup()
        
        test_methods = [
            # Integration tests
            self.test_promptchain_orchestration,
            self.test_tool_parameter_validation,
            
            # Multi-hop reasoning
            self.test_reasoning_depth_limits,
            self.test_circular_reasoning_detection,
            
            # Parameter boundaries
            self.test_token_limit_boundaries,
            self.test_string_injection_attacks,
            
            # Performance
            self.test_concurrent_tool_calls,
            self.test_memory_pressure,
            
            # Edge cases
            self.test_empty_and_null_inputs,
            self.test_error_recovery_mechanisms,
        ]
        
        logger.info(f"Running {len(test_methods)} adversarial tests...")
        
        for test_method in test_methods:
            try:
                result = await test_method()
                self.test_results.append(result)
                
                # Generate bug report for failures
                if not result.passed and result.severity in ["Blocker", "Critical", "Major"]:
                    bug_report = self._create_bug_report(result)
                    self.bug_reports.append(bug_report)
                
                status = "✓" if result.passed else "✗"
                logger.info(f"{status} {result.test_id}: {test_method.__name__}")
                
            except Exception as e:
                logger.error(f"Test {test_method.__name__} crashed: {e}")
                self.test_results.append(TestResult(
                    test_id=f"CRASH_{test_method.__name__}",
                    category="Integration",
                    severity="Blocker",
                    confidence="High",
                    passed=False,
                    error=str(e),
                    details={"traceback": traceback.format_exc()}
                ))
        
        await self.teardown()
        return self.test_results, self.bug_reports
    
    def _create_bug_report(self, result: TestResult) -> BugReport:
        """Create bug report from test result."""
        return BugReport(
            title=f"[{result.category}] {result.error or 'Test failure'}",
            severity=result.severity,
            confidence=result.confidence,
            category=result.category,
            location=[{
                "file": "athena_mcp_server.py",
                "line": 0,
                "symbol": result.test_id
            }],
            evidence={
                "type": "TestFail",
                "details": result.details,
                "repro_steps": result.reproduction_steps
            },
            root_cause_hypothesis=f"Test {result.test_id} identified issue in {result.category}",
            blast_radius=f"Affects MCP tool reliability - severity: {result.severity}",
            fix_suggestion=result.fix_suggestion or ["Investigate and fix identified issue"],
            next_actions=["Run focused test on specific failure", "Implement suggested fixes"],
            labels=["ADVERSARIAL_TEST", result.category.upper()]
        )
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        failed = total_tests - passed
        
        report = f"""
{'='*80}
ADVERSARIAL TEST REPORT - Athena LightRAG MCP Server
{'='*80}

EXECUTIVE SUMMARY
-----------------
Total Tests: {total_tests}
Passed: {passed} ({passed/total_tests*100:.1f}%)
Failed: {failed} ({failed/total_tests*100:.1f}%)
Critical Bugs: {len([b for b in self.bug_reports if b.severity == "Critical"])}
Major Bugs: {len([b for b in self.bug_reports if b.severity == "Major"])}

TEST RESULTS BY CATEGORY
------------------------
"""
        
        # Group results by category
        categories = {}
        for result in self.test_results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        for category, results in categories.items():
            passed_cat = sum(1 for r in results if r.passed)
            report += f"\n{category}:\n"
            report += f"  Passed: {passed_cat}/{len(results)}\n"
            for r in results:
                status = "✓" if r.passed else f"✗ [{r.severity}]"
                report += f"  {status} {r.test_id}: {r.error or 'Passed'}\n"
        
        # Bug reports
        if self.bug_reports:
            report += f"\n{'='*80}\nBUG REPORTS (JSON Format)\n{'='*80}\n"
            for bug in self.bug_reports:
                report += json.dumps(bug.__dict__, indent=2) + "\n\n"
        
        # Performance metrics
        report += f"\n{'='*80}\nPERFORMANCE METRICS\n{'='*80}\n"
        avg_time = sum(r.execution_time for r in self.test_results) / len(self.test_results)
        report += f"Average test execution time: {avg_time:.3f}s\n"
        report += f"Slowest test: {max(self.test_results, key=lambda r: r.execution_time).test_id}\n"
        
        return report


async def main():
    """Main entry point for adversarial testing."""
    suite = AdversarialTestSuite()
    
    try:
        results, bugs = await suite.run_all_tests()
        
        # Generate and print report
        report = suite.generate_report()
        print(report)
        
        # Save report to file
        report_path = Path("adversarial_test_report.txt")
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")
        
        # Save bug reports as JSON
        if bugs:
            bugs_path = Path("bug_reports.json")
            bugs_data = [bug.__dict__ for bug in bugs]
            bugs_path.write_text(json.dumps(bugs_data, indent=2))
            logger.info(f"Bug reports saved to {bugs_path}")
        
        # Exit with error code if critical bugs found
        critical_bugs = [b for b in bugs if b.severity in ["Blocker", "Critical"]]
        if critical_bugs:
            logger.error(f"Found {len(critical_bugs)} critical bugs!")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())