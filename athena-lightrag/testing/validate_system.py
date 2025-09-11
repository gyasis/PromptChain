#!/usr/bin/env python3
"""
Athena LightRAG System Validation Script
========================================
Comprehensive validation script to test the complete system functionality
including database connectivity, MCP server functionality, and multi-hop reasoning.

This script can be used for:
- Development validation
- Production health checks  
- CI/CD pipeline testing
- System troubleshooting

Author: PromptChain Team
Date: 2025
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from athena_lightrag.core import (
    query_athena_basic,
    query_athena_multi_hop,
    get_athena_database_info,
    athena_context
)
from athena_lightrag.server import (
    query_athena,
    query_athena_reasoning,
    get_database_status,
    generate_sql_query,
    get_query_mode_help
)


class ValidationResult:
    """Container for validation test results."""
    
    def __init__(self, test_name: str, success: bool, message: str, 
                 execution_time: float = 0.0, details: Optional[Dict] = None):
        self.test_name = test_name
        self.success = success
        self.message = message
        self.execution_time = execution_time
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "message": self.message,
            "execution_time": self.execution_time,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class SystemValidator:
    """Comprehensive system validation orchestrator."""
    
    def __init__(self, verbose: bool = True, log_level: str = "INFO"):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Test configuration
        self.test_queries = {
            "basic": [
                "What tables are related to patient appointments?",
                "Describe the anesthesia case management system",
                "How is billing information structured in the database?",
                "What are the main collector categories?"
            ],
            "reasoning": [
                "How do anesthesia workflows connect to patient scheduling and billing systems?",
                "What is the complete patient journey from appointment to billing?",
                "How are quality metrics tracked across the care delivery process?",
                "What audit trails exist for medication administration and billing?"
            ],
            "sql_generation": [
                "Find all patients with upcoming anesthesia procedures",
                "Generate a report of provider performance metrics",
                "Create a query for billing reconciliation analysis",
                "List all high-risk patients with recent procedures"
            ]
        }
    
    def log_progress(self, message: str, level: str = "INFO"):
        """Log progress with optional verbose output."""
        getattr(self.logger, level.lower())(message)
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {message}")
    
    async def run_validation_test(self, test_name: str, test_func, *args, **kwargs) -> ValidationResult:
        """Run a single validation test with timing and error handling."""
        self.log_progress(f"Starting test: {test_name}")
        start_time = time.time()
        
        try:
            result = await test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            validation_result = ValidationResult(
                test_name=test_name,
                success=True,
                message=f"Test passed in {execution_time:.2f}s",
                execution_time=execution_time,
                details={"result": str(result)[:500]}  # Truncate long results
            )
            
            self.log_progress(f"✅ {test_name} - PASSED ({execution_time:.2f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Test failed: {str(e)}"
            
            validation_result = ValidationResult(
                test_name=test_name,
                success=False,
                message=error_msg,
                execution_time=execution_time,
                details={
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            
            self.log_progress(f"❌ {test_name} - FAILED ({execution_time:.2f}s): {error_msg}", "ERROR")
        
        self.results.append(validation_result)
        return validation_result
    
    async def test_environment_setup(self) -> bool:
        """Test that the environment is properly configured.""" 
        self.log_progress("Testing environment setup...")
        
        # Check required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Check database path
        db_path = os.getenv('LIGHTRAG_WORKING_DIR', './athena_lightrag_db')
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        return True
    
    async def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity and basic info retrieval."""
        self.log_progress("Testing database connectivity...")
        
        info = await get_athena_database_info(return_raw=True)
        
        if not info.get('database_exists'):
            raise RuntimeError("Database does not exist or is not accessible")
        
        return info
    
    async def test_basic_queries(self) -> List[Dict[str, Any]]:
        """Test basic query functionality with various modes."""
        self.log_progress("Testing basic queries...")
        
        results = []
        modes = ["local", "global", "hybrid", "naive"]
        
        for mode in modes:
            for query in self.test_queries["basic"][:2]:  # Test first 2 queries per mode
                result = await query_athena_basic(
                    query=query,
                    mode=mode,
                    top_k=30,
                    return_full_result=False
                )
                
                if not result or len(result) < 10:
                    raise ValueError(f"Insufficient response for query '{query}' in mode '{mode}'")
                
                results.append({
                    "query": query,
                    "mode": mode,
                    "result_length": len(result),
                    "success": True
                })
        
        return results
    
    async def test_multi_hop_reasoning(self) -> List[Dict[str, Any]]:
        """Test multi-hop reasoning functionality."""
        self.log_progress("Testing multi-hop reasoning...")
        
        results = []
        strategies = ["incremental", "comprehensive", "focused"]
        
        for strategy in strategies:
            for query in self.test_queries["reasoning"][:1]:  # Test 1 query per strategy
                result = await query_athena_multi_hop(
                    query=query,
                    context_strategy=strategy,
                    max_steps=3,
                    return_full_result=True
                )
                
                if isinstance(result, dict):
                    final_result = result.get("result", "")
                    reasoning_steps = result.get("reasoning_steps", [])
                    context_chunks = result.get("context_chunks", [])
                    
                    if not final_result or len(final_result) < 50:
                        raise ValueError(f"Insufficient reasoning result for query '{query}'")
                    
                    if len(reasoning_steps) == 0:
                        raise ValueError(f"No reasoning steps recorded for query '{query}'")
                    
                    results.append({
                        "query": query,
                        "strategy": strategy,
                        "result_length": len(final_result),
                        "reasoning_steps_count": len(reasoning_steps),
                        "context_chunks_count": len(context_chunks),
                        "success": True
                    })
                else:
                    # Handle string result (fallback mode)
                    if not result or len(result) < 50:
                        raise ValueError(f"Insufficient reasoning result for query '{query}'")
                    
                    results.append({
                        "query": query,
                        "strategy": strategy,
                        "result_length": len(result),
                        "success": True
                    })
        
        return results
    
    async def test_mcp_server_tools(self) -> Dict[str, Any]:
        """Test MCP server tool functionality."""
        self.log_progress("Testing MCP server tools...")
        
        # Test basic query tool
        basic_result = await query_athena(
            query="What are the main patient tables?",
            mode="hybrid",
            top_k=50
        )
        
        if not basic_result or "error" in basic_result.lower():
            raise ValueError("MCP basic query tool failed")
        
        # Test reasoning query tool
        reasoning_result = await query_athena_reasoning(
            query="How are patient appointments connected to billing?",
            context_strategy="incremental",
            max_reasoning_steps=3
        )
        
        if not reasoning_result or "error" in reasoning_result.lower():
            raise ValueError("MCP reasoning query tool failed")
        
        # Test database status tool
        status_result = await get_database_status(
            include_performance_stats=True,
            return_raw_data=False
        )
        
        if not status_result or "error" in status_result.lower():
            raise ValueError("MCP database status tool failed")
        
        # Test SQL generation tool
        sql_result = await generate_sql_query(
            natural_language_query="Find patients with recent appointments",
            target_database_type="mysql",
            include_validation=True
        )
        
        if not sql_result or "error" in sql_result.lower():
            raise ValueError("MCP SQL generation tool failed")
        
        return {
            "basic_query_length": len(basic_result),
            "reasoning_query_length": len(reasoning_result), 
            "status_result_length": len(status_result),
            "sql_result_length": len(sql_result),
            "all_tools_working": True
        }
    
    async def test_sql_generation_capabilities(self) -> List[Dict[str, Any]]:
        """Test SQL generation with various database types."""
        self.log_progress("Testing SQL generation capabilities...")
        
        results = []
        db_types = ["mysql", "postgresql", "sqlite"]
        optimization_levels = ["basic", "intermediate", "advanced"]
        
        for db_type in db_types:
            for opt_level in optimization_levels[:2]:  # Test first 2 optimization levels
                for query in self.test_queries["sql_generation"][:1]:  # Test 1 query per combo
                    result = await generate_sql_query(
                        natural_language_query=query,
                        target_database_type=db_type,
                        optimization_level=opt_level,
                        include_validation=True,
                        return_explanation=True
                    )
                    
                    if not result or len(result) < 100:
                        raise ValueError(f"Insufficient SQL generation result for '{query}'")
                    
                    # Check for SQL keywords to verify actual SQL was generated
                    sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "INSERT", "UPDATE", "DELETE"]
                    has_sql = any(keyword in result.upper() for keyword in sql_keywords)
                    
                    if not has_sql:
                        raise ValueError(f"Generated result does not appear to contain SQL for '{query}'")
                    
                    results.append({
                        "query": query,
                        "database_type": db_type,
                        "optimization_level": opt_level,
                        "result_length": len(result),
                        "contains_sql": has_sql,
                        "success": True
                    })
        
        return results
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks and timing."""
        self.log_progress("Running performance benchmarks...")
        
        # Benchmark basic queries
        basic_times = []
        for _ in range(3):
            start_time = time.time()
            await query_athena_basic("What are the appointment tables?", mode="hybrid")
            basic_times.append(time.time() - start_time)
        
        # Benchmark reasoning queries
        reasoning_times = []
        for _ in range(2):
            start_time = time.time()
            await query_athena_multi_hop(
                "How do billing and scheduling connect?",
                context_strategy="incremental",
                max_steps=3
            )
            reasoning_times.append(time.time() - start_time)
        
        return {
            "basic_query_avg_time": sum(basic_times) / len(basic_times),
            "basic_query_min_time": min(basic_times),
            "basic_query_max_time": max(basic_times),
            "reasoning_query_avg_time": sum(reasoning_times) / len(reasoning_times),
            "reasoning_query_min_time": min(reasoning_times), 
            "reasoning_query_max_time": max(reasoning_times),
            "performance_acceptable": all(t < 30 for t in basic_times + reasoning_times)
        }
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run the complete validation suite."""
        self.log_progress("Starting full system validation...")
        start_time = time.time()
        
        # Define test suite
        test_suite = [
            ("Environment Setup", self.test_environment_setup),
            ("Database Connectivity", self.test_database_connectivity),
            ("Basic Queries", self.test_basic_queries),
            ("Multi-hop Reasoning", self.test_multi_hop_reasoning),
            ("MCP Server Tools", self.test_mcp_server_tools),
            ("SQL Generation", self.test_sql_generation_capabilities),
            ("Performance Benchmarks", self.test_performance_benchmarks)
        ]
        
        # Run tests
        for test_name, test_func in test_suite:
            await self.run_validation_test(test_name, test_func)
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        passed_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Create summary
        summary = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_execution_time": total_time,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "overall_status": "PASSED" if success_rate >= 0.8 else "FAILED",
            "test_results": [r.to_dict() for r in self.results]
        }
        
        self.log_progress(f"Validation complete! Overall status: {summary['overall_status']}")
        self.log_progress(f"Success rate: {success_rate:.1%} ({passed_tests}/{total_tests} tests passed)")
        self.log_progress(f"Total execution time: {total_time:.2f} seconds")
        
        return summary
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save validation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results_{timestamp}.json"
        
        summary = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.success),
            "test_results": [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log_progress(f"Results saved to {filename}")
        return filename


async def main():
    """Main validation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Athena LightRAG System Validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--save-results", "-s", action="store_true", help="Save results to file")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick validation (subset of tests)")
    
    args = parser.parse_args()
    
    # Create validator
    validator = SystemValidator(verbose=args.verbose, log_level=args.log_level)
    
    try:
        # Run validation
        if args.quick:
            # Quick validation - just basic tests
            await validator.run_validation_test("Environment Setup", validator.test_environment_setup)
            await validator.run_validation_test("Database Connectivity", validator.test_database_connectivity)
            await validator.run_validation_test("Basic Query Test", 
                lambda: query_athena_basic("What are the main patient tables?"))
        else:
            # Full validation suite
            results = await validator.run_full_validation()
            
            # Print summary
            print("\n" + "="*60)
            print(f"VALIDATION SUMMARY - {results['overall_status']}")
            print("="*60)
            print(f"Tests: {results['passed_tests']}/{results['total_tests']} passed")
            print(f"Success Rate: {results['success_rate']:.1%}")
            print(f"Execution Time: {results['total_execution_time']:.2f}s")
            print("="*60)
        
        # Save results if requested
        if args.save_results:
            validator.save_results()
        
        # Exit with appropriate code
        passed_tests = sum(1 for r in validator.results if r.success)
        total_tests = len(validator.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:
            print("\n✅ Validation PASSED - System is ready for use!")
            sys.exit(0)
        else:
            print("\n❌ Validation FAILED - Please check errors above")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nValidation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())