"""Helper functions and mock tools for tutorial demonstrations.

This module provides mock tools, scenario builders, and MLflow helpers
for the full stack AgenticStepProcessor tutorial with MLflow observability.
"""

from typing import List, Dict, Any
import random
import json


# ============================================================================
# MOCK TOOLS - Simulate real tool execution for demonstrations
# ============================================================================

def search_codebase(query: str) -> str:
    """Simulate searching a codebase.

    Returns realistic mock search results with file paths and code snippets.
    """
    mock_results = {
        "authentication": [
            {"file": "src/auth/login.py", "line": 45, "snippet": "def authenticate_user(username, password):"},
            {"file": "src/auth/jwt_handler.py", "line": 12, "snippet": "def generate_jwt_token(user_id):"},
            {"file": "src/middleware/auth_middleware.py", "line": 23, "snippet": "def verify_token(token):"}
        ],
        "database": [
            {"file": "src/models/user.py", "line": 8, "snippet": "class User(db.Model):"},
            {"file": "src/db/connection.py", "line": 15, "snippet": "def get_db_connection():"},
            {"file": "src/queries/user_queries.py", "line": 34, "snippet": "def fetch_user_by_id(user_id):"}
        ],
        "api": [
            {"file": "src/api/routes.py", "line": 67, "snippet": "@app.route('/api/users', methods=['GET'])"},
            {"file": "src/api/handlers.py", "line": 89, "snippet": "def handle_user_request(request):"},
        ]
    }

    # Select results based on query keywords
    results = []
    for keyword, matches in mock_results.items():
        if keyword.lower() in query.lower():
            results.extend(matches)

    if not results:
        results = random.choice(list(mock_results.values()))

    output = f"Found {len(results)} matches for '{query}':\n\n"
    for r in results:
        output += f"• {r['file']}:{r['line']}\n  {r['snippet']}\n\n"

    return output


def read_file(path: str) -> str:
    """Simulate reading a file.

    Returns mock file content based on the file path.
    """
    mock_files = {
        "src/auth/login.py": '''"""Authentication module for user login."""
import bcrypt
from datetime import datetime
from src.db.connection import get_db_connection
from src.auth.jwt_handler import generate_jwt_token

def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user with username and password.

    Returns:
        dict: {"success": bool, "token": str, "user_id": int}
    """
    db = get_db_connection()
    user = db.query("SELECT * FROM users WHERE username = ?", (username,))

    if not user:
        return {"success": False, "error": "User not found"}

    if not bcrypt.checkpw(password.encode(), user['password_hash']):
        return {"success": False, "error": "Invalid password"}

    token = generate_jwt_token(user['id'])
    db.log_login(user['id'], datetime.now())

    return {
        "success": True,
        "token": token,
        "user_id": user['id']
    }
''',
        "src/auth/jwt_handler.py": '''"""JWT token generation and validation."""
import jwt
from datetime import datetime, timedelta
from config import SECRET_KEY, TOKEN_EXPIRY_HOURS

def generate_jwt_token(user_id: int) -> str:
    """Generate JWT token for authenticated user."""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token: str) -> dict:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return {"valid": True, "user_id": payload['user_id']}
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token expired"}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "Invalid token"}
'''
    }

    content = mock_files.get(path, f"# Mock content for {path}\n# File not found in mock data")
    return f"=== Content of {path} ===\n\n{content}"


def analyze_code(code: str) -> str:
    """Simulate code analysis.

    Returns mock analysis results with potential issues and recommendations.
    """
    issues = [
        "⚠️ Missing input validation on line 15",
        "⚠️ Potential SQL injection vulnerability in query construction",
        "⚠️ Hardcoded credentials detected (line 45)",
        "✅ Proper error handling implemented",
        "✅ Uses prepared statements for database queries",
        "💡 Consider adding rate limiting to prevent brute force attacks"
    ]

    selected_issues = random.sample(issues, k=random.randint(3, 5))

    analysis = "📊 Code Analysis Results:\n\n"
    analysis += "\n".join(selected_issues)
    analysis += "\n\n🔍 Overall Security Score: 7.5/10"

    return analysis


def delete_file(path: str) -> str:
    """DANGEROUS operation for CoVe demonstration.

    This should be blocked by Chain of Verification in the tutorial.
    """
    return f"⚠️ DANGEROUS: Attempting to delete {path}"


def write_report(content: str) -> str:
    """Simulate writing a report.

    Returns success message.
    """
    word_count = len(content.split())
    return f"✅ Report written successfully ({word_count} words)"


# ============================================================================
# SCENARIO BUILDERS - Create realistic test scenarios
# ============================================================================

def create_research_scenario() -> Dict[str, Any]:
    """Create a research scenario with tools and objective.

    Returns a scenario dict for testing Phase 1-4 features.
    """
    return {
        "objective": "Research authentication patterns in the codebase and identify security vulnerabilities",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_codebase",
                    "description": "Search through project files for code patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (e.g., 'authentication', 'database', 'api')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to read"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_code",
                    "description": "Analyze code for security issues and best practices",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to analyze"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ],
        "tool_functions": {
            "search_codebase": search_codebase,
            "read_file": read_file,
            "analyze_code": analyze_code
        }
    }


def create_production_scenario() -> Dict[str, Any]:
    """Create a safety-critical scenario for demonstrating Phase 3 features.

    Includes dangerous operations that should be blocked by CoVe.
    """
    return {
        "objective": "Analyze and cleanup authentication code (includes dangerous operations)",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_codebase",
                    "description": "Search through project files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file (DANGEROUS)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_report",
                    "description": "Write analysis report",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"}
                        },
                        "required": ["content"]
                    }
                }
            }
        ],
        "tool_functions": {
            "search_codebase": search_codebase,
            "delete_file": delete_file,
            "write_report": write_report
        }
    }


def create_multi_agent_scenario() -> Dict[str, Any]:
    """Create a multi-agent orchestration scenario.

    Demonstrates different agents with different feature configurations.
    """
    return {
        "tasks": [
            {
                "description": "Research authentication patterns",
                "expected_agent": "researcher",
                "objective": "Find all authentication-related code"
            },
            {
                "description": "Analyze findings for vulnerabilities",
                "expected_agent": "analyzer",
                "objective": "Identify security issues"
            },
            {
                "description": "Write comprehensive security report",
                "expected_agent": "writer",
                "objective": "Document findings and recommendations"
            }
        ]
    }


# ============================================================================
# MLFLOW HELPERS - Setup and visualization utilities
# ============================================================================

def setup_mlflow_tracking(experiment_name: str) -> None:
    """Configure MLflow tracking for tutorial.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    try:
        import mlflow
        import os

        # Set tracking URI
        os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)

        print(f"✅ MLflow tracking configured")
        print(f"   Experiment: {experiment_name}")
        print(f"   Tracking URI: file:./mlruns")
        print(f"\n💡 Launch MLflow UI: mlflow ui")
        print(f"   Then visit: http://localhost:5000")

    except ImportError:
        print("❌ MLflow not installed. Run: pip install mlflow")


def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    """Compare metrics across multiple MLflow runs.

    Args:
        run_ids: List of MLflow run IDs to compare

    Returns:
        Comparison data as dictionary
    """
    try:
        import mlflow
        import pandas as pd

        data = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            data.append({
                "run_id": run_id[:8],
                "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
                **run.data.metrics
            })

        df = pd.DataFrame(data)
        return df.to_dict('records')

    except ImportError:
        return {"error": "MLflow or pandas not installed"}


def visualize_blackboard_evolution(snapshots: List[Dict]) -> None:
    """Visualize Blackboard state evolution over time.

    Args:
        snapshots: List of Blackboard state snapshots
    """
    print("\n📊 Blackboard Evolution:")
    print("=" * 60)

    for i, snapshot in enumerate(snapshots):
        print(f"\n🔄 Iteration {i + 1}:")
        print(f"   Facts: {len(snapshot.get('facts_discovered', {}))}")
        print(f"   Observations: {len(snapshot.get('observations', []))}")
        print(f"   Plan steps: {len(snapshot.get('current_plan', []))}")

        if i < 3:  # Show detail for first 3
            print(f"   Latest fact: {list(snapshot.get('facts_discovered', {}).keys())[-1] if snapshot.get('facts_discovered') else 'None'}")


def visualize_tao_execution(tao_log: List[Dict]) -> None:
    """Visualize TAO (Think-Act-Observe) execution phases.

    Args:
        tao_log: List of TAO phase executions
    """
    print("\n🧠 TAO Execution Log:")
    print("=" * 60)

    for i, phase in enumerate(tao_log[:5]):  # Show first 5
        phase_name = phase['phase']
        data = phase['data']

        if phase_name == "THINK":
            print(f"\n💭 THINK (Step {i//3 + 1}):")
            print(f"   Reasoning: {data.get('summary', 'N/A')[:80]}...")
        elif phase_name == "ACT":
            print(f"   🎬 ACT: {data.get('tool_name', 'unknown')}()")
        elif phase_name == "OBSERVE":
            print(f"   👁️  OBSERVE: {data.get('summary', 'N/A')[:80]}...")


# ============================================================================
# MOCK LLM RUNNER - For testing without real API calls
# ============================================================================

class MockLLMRunner:
    """Mock LLM runner for tutorial demonstrations without API costs."""

    def __init__(self, scenario_type: str = "research"):
        self.scenario_type = scenario_type
        self.call_count = 0

    async def __call__(self, messages: List[Dict], **kwargs) -> Dict:
        """Simulate LLM response based on scenario."""
        self.call_count += 1

        # Simulate different responses based on iteration
        if self.call_count <= 3:
            # Tool calls
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{self.call_count}",
                            "type": "function",
                            "function": {
                                "name": "search_codebase",
                                "arguments": json.dumps({"query": "authentication"})
                            }
                        }]
                    }
                }]
            }
        else:
            # Final response
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Analysis complete. Found 3 authentication files with proper JWT implementation.",
                        "tool_calls": []
                    }
                }]
            }
