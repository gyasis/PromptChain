"""
Main UseCase Validator - converts plain English YAML specs to executable Python tests
"""
import os
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any
from jinja2 import Template
from src.utils.api_client import APIClient
from src.utils.test_data import TestDataManager
from src.utils.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class UseCaseValidator:
    """Converts plain English use case specifications into executable Python tests."""

    def __init__(self):
        self.api_client = APIClient()
        self.test_data = TestDataManager()
        self.db_manager = DatabaseManager()
        self.generated_tests = []

    def load_usecase(self, filepath: str) -> Dict[str, Any]:
        """Load and parse a use case YAML file."""
        try:
            with open(filepath, 'r') as file:
                usecase = yaml.safe_load(file)
            logger.info(f"Loaded use case: {usecase.get('name', 'Unknown')}")
            return usecase
        except Exception as e:
            logger.error(f"Failed to load use case from {filepath}: {e}")
            raise

    def generate_test_code(self, usecase: Dict[str, Any]) -> str:
        """Convert use case specification to executable Python test code."""
        template = Template('''
import pytest
import time
from src.utils.api_client import APIClient
from src.utils.test_data import TestDataManager
from src.utils.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

@pytest.mark.{{ tags | join(' @pytest.mark.') }}
class Test{{ class_name }}:
    """{{ description }}"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup test environment and handle cleanup."""
        self.api_client = APIClient()
        self.test_data = TestDataManager()
        self.db_manager = DatabaseManager()
        self.created_resources = []

        # SETUP PHASE
        logger.info("=== SETUP PHASE ===")
        {% for setup_step in setup %}
        # {{ setup_step }}
        {{ self.convert_setup_step(setup_step) }}
        {% endfor %}

        yield

        # CLEANUP PHASE
        logger.info("=== CLEANUP PHASE ===")
        {% for cleanup_step in cleanup %}
        # {{ cleanup_step }}
        {{ self.convert_cleanup_step(cleanup_step) }}
        {% endfor %}

    def test_{{ test_method_name }}(self):
        """{{ description }}"""
        logger.info("=== EXECUTION PHASE ===")
        
        {% for step in steps %}
        # Step {{ loop.index }}: {{ step.step }}
        logger.info("Step {{ loop.index }}: {{ step.step }}")
        {{ self.convert_test_step(step) }}
        
        {% endfor %}

        # VALIDATION PHASE
        logger.info("=== VALIDATION PHASE ===")
        {% for validation in validations %}
        # {{ validation }}
        {{ self.convert_validation(validation) }}
        {% endfor %}

        logger.info("✅ Use case completed successfully")

        {% if error_scenarios %}
        {% for scenario in error_scenarios %}
        def test_{{ scenario | slugify }}(self):
            """Test error scenario: {{ scenario }}"""
            {{ self.convert_error_scenario(scenario) }}
        {% endfor %}
        {% endif %}
        ''')

        context = {
            'class_name': self._to_class_name(usecase['name']),
            'test_method_name': self._to_method_name(usecase['name']),
            'description': usecase['description'],
            'tags': usecase.get('tags', []),
            'setup': usecase.get('setup', []),
            'steps': usecase.get('steps', []),
            'validations': usecase.get('validations', []),
            'cleanup': usecase.get('cleanup', []),
            'error_scenarios': usecase.get('error_scenarios', [])
        }

        return template.render(**context)

    def convert_setup_step(self, step: str) -> str:
        """Convert setup instruction to Python code."""
        step_lower = step.lower()
        
        if "generate" in step_lower and "user data" in step_lower:
            return "self.test_user = self.test_data.generate_user_data()"
        elif "create test user" in step_lower:
            return """
response = self.api_client.post("/auth/register", self.test_data.generate_user_data())
self.test_user_id = response.json()['userId']
self.created_resources.append(('user', self.test_user_id))
"""
        elif "create test product" in step_lower:
            return """
product_data = self.test_data.generate_product_data()
response = self.api_client.post("/products", product_data)
self.test_product_id = response.json()['productId']
self.created_resources.append(('product', self.test_product_id))
"""
        elif "ensure" in step_lower and "doesn't exist" in step_lower:
            return "# Verification that test data doesn't exist - handled by unique generation"
        else:
            return f"# TODO: Implement setup step: {step}"

    def convert_test_step(self, step: Dict[str, Any]) -> str:
        """Convert test step to Python code."""
        action = step['action'].lower()
        expect = step.get('expect', '')
        
        if action.startswith('post to'):
            endpoint = self._extract_endpoint(action)
            return self._generate_post_request(endpoint, step)
        elif action.startswith('get'):
            endpoint = self._extract_endpoint(action)
            return self._generate_get_request(endpoint, step)
        elif action.startswith('put to'):
            endpoint = self._extract_endpoint(action)
            return self._generate_put_request(endpoint, step)
        elif action.startswith('delete'):
            endpoint = self._extract_endpoint(action)
            return self._generate_delete_request(endpoint, step)
        else:
            return f"# TODO: Implement action: {action}"

    def _generate_post_request(self, endpoint: str, step: Dict[str, Any]) -> str:
        """Generate POST request code."""
        data_source = self._determine_data_source(step)
        auth_header = self._determine_auth_header(step)
        
        code = f"""
{data_source}
response = self.api_client.post("{endpoint}", data{auth_header})
"""
        
        if 'expect' in step:
            code += self._generate_assertions(step['expect'], 'response')
        
        return code

    def _generate_get_request(self, endpoint: str, step: Dict[str, Any]) -> str:
        """Generate GET request code."""
        auth_header = self._determine_auth_header(step)
        
        code = f'response = self.api_client.get("{endpoint}"{auth_header})\n'
        
        if 'expect' in step:
            code += self._generate_assertions(step['expect'], 'response')
        
        return code

    def _extract_endpoint(self, action: str) -> str:
        """Extract API endpoint from action description."""
        # Look for patterns like "/auth/register", "/users/{userId}", etc.
        match = re.search(r'(/[a-zA-Z0-9/_{}]+)', action)
        if match:
            endpoint = match.group(1)
            # Replace placeholders with actual variables
            endpoint = endpoint.replace('{userId}', '{self.test_user_id}')
            endpoint = endpoint.replace('{orderId}', '{self.test_order_id}')
            endpoint = endpoint.replace('{productId}', '{self.test_product_id}')
            return endpoint
        return "/unknown"

    def _determine_data_source(self, step: Dict[str, Any]) -> str:
        """Determine what data to send with the request."""
        action = step['action'].lower()
        
        if 'user email, password' in action:
            return "data = self.test_user"
        elif 'profile data' in action:
            return "data = self.test_data.generate_profile_data()"
        elif 'order' in action:
            return "data = self.test_data.generate_order_data()"
        elif step.get('data'):
            return f"# Data: {step['data']}\ndata = self.test_data.generate_custom_data()"
        else:
            return "data = {}"

    def _determine_auth_header(self, step: Dict[str, Any]) -> str:
        """Determine if authentication headers are needed."""
        action = step['action'].lower()
        
        if 'auth token' in action or 'using auth' in action:
            return ', headers={"Authorization": f"Bearer {self.auth_token}"}'
        elif 'api key' in action:
            return ', headers={"X-API-Key": self.api_key}'
        else:
            return ''

    def _generate_assertions(self, expect: str, response_var: str) -> str:
        """Generate assertion code from expectation description."""
        expect_lower = expect.lower()
        assertions = []
        
        if 'status' in expect_lower:
            status_match = re.search(r'status (\d+)', expect_lower)
            if status_match:
                status = status_match.group(1)
                assertions.append(f'assert {response_var}.status_code == {status}, f"Expected {status}, got {{{response_var}.status_code}}"')
        
        if 'contains userid' in expect_lower:
            assertions.append(f'assert "userId" in {response_var}.json(), "Response missing userId"')
        
        if 'contains token' in expect_lower:
            assertions.append(f'assert "token" in {response_var}.json(), "Response missing token"')
            assertions.append(f'self.auth_token = {response_var}.json()["token"]')
        
        if 'email verification confirmed' in expect_lower:
            assertions.append(f'assert {response_var}.json().get("verified") == True, "Email not verified"')
        
        return '\n'.join(assertions) + '\n'

    def convert_validation(self, validation: str) -> str:
        """Convert validation description to assertion code."""
        validation_lower = validation.lower()
        
        if 'user email matches' in validation_lower:
            return 'assert user["email"] == self.test_user["email"], "Email mismatch"'
        elif 'status is active' in validation_lower:
            return 'assert user.get("status") == "active", "User should be active"'
        elif 'email is marked as verified' in validation_lower:
            return 'assert user.get("emailVerified") == True, "Email should be verified"'
        elif 'appears in database' in validation_lower:
            return '''
db_user = self.db_manager.get_user_by_id(self.test_user_id)
assert db_user is not None, "User not found in database"
'''
        else:
            return f'# TODO: Implement validation: {validation}'

    def convert_cleanup_step(self, step: str) -> str:
        """Convert cleanup instruction to Python code."""
        step_lower = step.lower()
        
        if 'delete' in step_lower and 'user' in step_lower:
            return '''
for resource_type, resource_id in self.created_resources:
    if resource_type == 'user':
        try:
            self.api_client.delete(f"/users/{resource_id}")
        except Exception as e:
            logger.warning(f"Cleanup failed for user {resource_id}: {e}")
'''
        elif 'remove' in step_lower and 'database' in step_lower:
            return '''
if hasattr(self, 'test_user_id'):
    self.db_manager.cleanup_user_data(self.test_user_id)
'''
        else:
            return f'# TODO: Implement cleanup: {step}'

    def _to_class_name(self, name: str) -> str:
        """Convert use case name to Python class name."""
        return ''.join(word.capitalize() for word in re.findall(r'\w+', name))

    def _to_method_name(self, name: str) -> str:
        """Convert use case name to Python method name."""
        return '_'.join(re.findall(r'\w+', name.lower()))

    def execute_usecase(self, usecase_file: str) -> Dict[str, Any]:
        """Load, generate, and execute a use case test."""
        logger.info(f"Executing use case: {usecase_file}")
        
        # Load the use case specification
        usecase = self.load_usecase(usecase_file)
        
        # Generate Python test code
        test_code = self.generate_test_code(usecase)
        
        # Save generated code for debugging
        output_file = f"generated_tests/test_{Path(usecase_file).stem}.py"
        os.makedirs("generated_tests", exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(test_code)
        
        logger.info(f"Generated test code saved to: {output_file}")
        
        # Execute the test using pytest
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", output_file, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        return {
            "usecase_name": usecase['name'],
            "status": "PASSED" if result.returncode == 0 else "FAILED",
            "output": result.stdout,
            "errors": result.stderr,
            "generated_code": test_code
        }



