#!/usr/bin/env python3
"""
Environment Validation Script for Athena LightRAG MCP Server
===========================================================

This script validates that the UV-managed development environment is
properly configured with all required dependencies.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class EnvironmentValidator:
    """Comprehensive environment validation for Athena LightRAG project"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.results = {
            "environment_status": "unknown",
            "python_info": {},
            "dependency_status": {},
            "import_tests": {},
            "mcp_functionality": {},
            "issues": [],
            "recommendations": []
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run comprehensive environment validation"""
        print("🔍 Starting Athena LightRAG Environment Validation...")
        print("=" * 60)
        
        # Core environment checks
        self._validate_python_environment()
        self._validate_uv_setup()
        self._validate_dependencies()
        self._test_imports()
        self._test_mcp_functionality()
        
        # Generate final assessment
        self._generate_assessment()
        
        return self.results
    
    def _validate_python_environment(self):
        """Validate Python interpreter and virtual environment"""
        print("\n📋 Python Environment Validation")
        print("-" * 40)
        
        # Python version and path
        python_path = sys.executable
        python_version = sys.version
        virtual_env = os.environ.get('VIRTUAL_ENV', 'Not activated')
        
        self.results["python_info"] = {
            "executable": python_path,
            "version": python_version,
            "virtual_env": virtual_env,
            "is_uv_venv": ".venv" in python_path
        }
        
        print(f"✓ Python executable: {python_path}")
        print(f"✓ Python version: {python_version.split()[0]}")
        print(f"✓ Virtual environment: {virtual_env}")
        
        # Check if we're in UV virtual environment
        if ".venv" in python_path:
            print("✅ Running in UV virtual environment")
        else:
            self.results["issues"].append("Not running in UV virtual environment")
            print("⚠️  WARNING: Not running in UV virtual environment")
    
    def _validate_uv_setup(self):
        """Validate UV installation and project setup"""
        print("\n🚀 UV Setup Validation")
        print("-" * 40)
        
        # Check UV availability
        try:
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
            uv_version = result.stdout.strip()
            print(f"✓ UV version: {uv_version}")
        except FileNotFoundError:
            self.results["issues"].append("UV not installed or not in PATH")
            print("❌ UV not found")
            return
        
        # Check project files
        required_files = ['pyproject.toml', 'uv.lock', '.venv']
        for file_path in required_files:
            if (self.project_root / file_path).exists():
                print(f"✓ {file_path} found")
            else:
                self.results["issues"].append(f"Missing {file_path}")
                print(f"❌ {file_path} missing")
    
    def _validate_dependencies(self):
        """Validate that all dependencies are installed"""
        print("\n📦 Dependency Validation")
        print("-" * 40)
        
        critical_deps = [
            ("fastmcp", "fastmcp"),
            ("lightrag-hku", "lightrag"), 
            ("mcp", "mcp"),
            ("promptchain", "promptchain"),
            ("openai", "openai"),
            ("litellm", "litellm"),
            ("pydantic", "pydantic"),
            ("rich", "rich")
        ]
        
        for dep_name, import_name in critical_deps:
            try:
                __import__(import_name)
                self.results["dependency_status"][dep_name] = "installed"
                print(f"✓ {dep_name}")
            except ImportError:
                self.results["dependency_status"][dep_name] = "missing"
                self.results["issues"].append(f"Dependency {dep_name} not available")
                print(f"❌ {dep_name}")
    
    def _test_imports(self):
        """Test critical imports and their functionality"""
        print("\n🧪 Import Tests")
        print("-" * 40)
        
        import_tests = {
            "fastmcp": self._test_fastmcp_import,
            "lightrag": self._test_lightrag_import,
            "mcp": self._test_mcp_import,
            "promptchain": self._test_promptchain_import
        }
        
        for name, test_func in import_tests.items():
            try:
                test_func()
                self.results["import_tests"][name] = "success"
                print(f"✓ {name} import test passed")
            except Exception as e:
                self.results["import_tests"][name] = f"failed: {str(e)}"
                self.results["issues"].append(f"{name} import test failed: {str(e)}")
                print(f"❌ {name} import test failed: {str(e)}")
    
    def _test_fastmcp_import(self):
        """Test FastMCP functionality"""
        from mcp.server.fastmcp.server import FastMCP
        
        # Test server creation
        mcp = FastMCP("Test Server")
        
        @mcp.tool()
        def test_tool(query: str) -> dict:
            return {"result": f"Test: {query}"}
        
        assert mcp.name == "Test Server"
    
    def _test_lightrag_import(self):
        """Test LightRAG functionality"""
        import lightrag
        from lightrag import LightRAG
        
        # Test that we can create a LightRAG instance (without actual initialization)
        assert hasattr(lightrag, '__version__')
    
    def _test_mcp_import(self):
        """Test MCP package functionality"""
        import mcp
        from mcp.server import Server
        
        # Test basic MCP server functionality
        assert hasattr(mcp, 'server')
    
    def _test_promptchain_import(self):
        """Test PromptChain functionality"""
        import promptchain
        from promptchain import PromptChain
        from promptchain.utils.mcp_helpers import MCPHelper
        
        # Test that core classes are available
        assert PromptChain is not None
        assert MCPHelper is not None
    
    def _test_mcp_functionality(self):
        """Test MCP server functionality"""
        print("\n🌐 MCP Functionality Tests")
        print("-" * 40)
        
        try:
            # Test that we can create a basic MCP server structure
            from mcp.server.fastmcp.server import FastMCP
            
            server = FastMCP("Athena Test Server")
            
            @server.tool()
            def test_functionality(query: str) -> dict:
                return {"status": "working", "query": query}
            
            self.results["mcp_functionality"]["server_creation"] = "success"
            print("✓ MCP server creation test passed")
            
        except Exception as e:
            self.results["mcp_functionality"]["server_creation"] = f"failed: {str(e)}"
            self.results["issues"].append(f"MCP server creation failed: {str(e)}")
            print(f"❌ MCP server creation failed: {str(e)}")
    
    def _generate_assessment(self):
        """Generate final environment assessment"""
        print("\n📊 Environment Assessment")
        print("=" * 60)
        
        total_issues = len(self.results["issues"])
        
        if total_issues == 0:
            self.results["environment_status"] = "healthy"
            print("🎉 ENVIRONMENT STATUS: HEALTHY")
            print("✅ All validations passed! Environment is ready for development.")
        elif total_issues <= 2:
            self.results["environment_status"] = "minor_issues"
            print("⚠️  ENVIRONMENT STATUS: MINOR ISSUES")
            print("🔧 Some minor issues detected but environment is largely functional.")
        else:
            self.results["environment_status"] = "needs_repair"
            print("🚨 ENVIRONMENT STATUS: NEEDS REPAIR")
            print("❌ Multiple issues detected. Environment repair required.")
        
        if self.results["issues"]:
            print(f"\n🐛 Issues Detected ({len(self.results['issues'])}):")
            for i, issue in enumerate(self.results["issues"], 1):
                print(f"   {i}. {issue}")
        
        # Generate recommendations
        self._generate_recommendations()
        
        if self.results["recommendations"]:
            print(f"\n💡 Recommendations ({len(self.results['recommendations'])}):")
            for i, rec in enumerate(self.results["recommendations"], 1):
                print(f"   {i}. {rec}")
    
    def _generate_recommendations(self):
        """Generate environment repair recommendations"""
        recommendations = []
        
        # UV environment issues
        if any("UV" in issue for issue in self.results["issues"]):
            recommendations.append("Ensure UV is properly installed: curl -LsSf https://astral.sh/uv/install.sh | sh")
        
        # Virtual environment issues
        if any("virtual environment" in issue for issue in self.results["issues"]):
            recommendations.append("Activate UV virtual environment: source .venv/bin/activate")
            recommendations.append("Or use improved activation script: source activate_env.sh")
        
        # Dependency issues
        if any("Dependency" in issue for issue in self.results["issues"]):
            recommendations.append("Reinstall dependencies: uv sync --locked")
            recommendations.append("If that fails, try: uv sync --refresh")
        
        # Import issues
        if any("import test failed" in issue for issue in self.results["issues"]):
            recommendations.append("Check for package conflicts: uv pip list")
            recommendations.append("Clear Python cache: find . -name __pycache__ -type d -exec rm -rf {} +")
        
        # Environment variable issues (conda interference)
        if not self.results["python_info"].get("is_uv_venv", False):
            recommendations.append("Clear conda interference: unset _CONDA_PYTHON_SYSCONFIGDATA_NAME CONDA_PYTHON_EXE")
            recommendations.append("Use clean environment: env PYTHONPATH=\"\" PATH=\".venv/bin:$PATH\" python")
        
        self.results["recommendations"] = recommendations


def main():
    """Main validation entry point"""
    validator = EnvironmentValidator()
    results = validator.validate_all()
    
    # Save detailed results to file
    results_file = Path("environment_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results["environment_status"] == "healthy":
        print("\n🚀 Environment ready for development!")
        sys.exit(0)
    elif results["environment_status"] == "minor_issues":
        print("\n⚠️  Environment has minor issues but is usable.")
        sys.exit(1)
    else:
        print("\n🔧 Environment needs repair before development.")
        sys.exit(2)


if __name__ == "__main__":
    main()