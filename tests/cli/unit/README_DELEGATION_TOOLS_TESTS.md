# Delegation Tools Tests - Quick Reference

## Test File Location
`/home/gyasis/Documents/code/PromptChain/tests/cli/unit/test_delegation_tools.py`

## Running Tests

### Run All Tests
```bash
python -m pytest tests/cli/unit/test_delegation_tools.py -v
```

### Run Specific Test Class
```bash
# Test delegate_task function
python -m pytest tests/cli/unit/test_delegation_tools.py::TestDelegateTask -v

# Test get_pending_tasks function
python -m pytest tests/cli/unit/test_delegation_tools.py::TestGetPendingTasks -v

# Test update_task_status function
python -m pytest tests/cli/unit/test_delegation_tools.py::TestUpdateTaskStatus -v

# Test request_help function
python -m pytest tests/cli/unit/test_delegation_tools.py::TestRequestHelp -v
```

### Run Single Test
```bash
python -m pytest tests/cli/unit/test_delegation_tools.py::TestDelegateTask::test_delegate_task_success_low_priority -v
```

### Run with Detailed Output
```bash
python -m pytest tests/cli/unit/test_delegation_tools.py -vv --tb=long
```

### Run in Parallel (if pytest-xdist installed)
```bash
python -m pytest tests/cli/unit/test_delegation_tools.py -n auto
```

## Test Statistics

- **Total Tests**: 34
- **Test Classes**: 6
- **Success Rate**: 100% (34/34 PASSED)
- **Average Duration**: ~130ms per test
- **Total Suite Time**: ~4-6 seconds

## Test Coverage Breakdown

### delegate_task (9 tests)
- All priority levels (low, medium, high)
- Context preservation
- Error handling (self-delegation, empty description, invalid priority)
- Response formatting

### get_pending_tasks (7 tests)
- Empty queue handling
- Single and multiple task display
- Agent filtering
- Status filtering (pending only)
- Description truncation
- Error handling

### update_task_status (7 tests)
- Status transitions (pending → in_progress → completed/failed)
- Result and error message attachment
- Invalid task ID handling
- Invalid status handling
- Optional fields

### request_help (7 tests)
- Capability matching
- Broadcast fallback
- Agent exclusion
- Priority handling
- Context preservation
- Task format validation

### request_help_tool (2 tests)
- Wrapper functionality
- Error handling

### Session Manager Injection (2 tests)
- Dependency injection pattern
- Error when uninitialized

## Key Features

### Mock-Based Testing
- No database dependencies
- Fast execution
- Isolated tests
- Predictable behavior

### Comprehensive Coverage
- Success paths
- Error paths
- Edge cases
- Validation logic
- Response formatting

### Test Quality
- Clear naming conventions
- Arrange-Act-Assert pattern
- Isolated test cases
- Comprehensive assertions

## Dependencies

```python
import pytest
from unittest.mock import Mock, MagicMock, patch
from promptchain.cli.tools.library import delegation_tools
from promptchain.cli.models.task import Task, TaskPriority, TaskStatus
```

## Common Test Patterns

### Basic Test Structure
```python
def test_function_scenario(self, mock_session_manager):
    # Arrange
    # ... setup test data ...

    # Act
    result = delegation_tools.function_name(...)

    # Assert
    assert "expected" in result
    # ... verify state ...
```

### Error Testing
```python
def test_function_error_case(self, mock_session_manager):
    # Act
    result = delegation_tools.function_name(invalid_input)

    # Assert
    assert "Error:" in result
    # Verify no side effects
```

### Mocking External Dependencies
```python
@patch('promptchain.cli.tools.library.delegation_tools.registry')
def test_with_registry_mock(self, mock_registry, mock_session_manager):
    mock_registry.discover_capabilities.return_value = [...]
    # ... test logic ...
```

## Troubleshooting

### Import Errors
If you see import errors, ensure the project root is in your Python path:
```bash
export PYTHONPATH=/home/gyasis/Documents/code/PromptChain:$PYTHONPATH
```

### Pytest Not Found
Install pytest:
```bash
pip install pytest pytest-mock
```

### Mock Issues
If mocks aren't working, verify the autouse fixture is active:
```python
@pytest.fixture(autouse=True)
def inject_session_manager(mock_session_manager):
    delegation_tools.set_session_manager(mock_session_manager)
    yield
    delegation_tools._session_manager = None
```

## Next Steps

### Adding New Tests
1. Create test method in appropriate class
2. Use clear, descriptive name
3. Follow Arrange-Act-Assert pattern
4. Add assertions for both result and state
5. Run test to verify

### Modifying Existing Tests
1. Identify affected test(s)
2. Update assertions
3. Run full test suite to check for regressions
4. Update documentation if behavior changes

## Related Documentation

- **Test Summary**: `TEST_DELEGATION_TOOLS_SUMMARY.md`
- **Source Code**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tools/library/delegation_tools.py`
- **Task Model**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/models/task.py`

---

**Last Updated**: 2025-11-28
**Maintainer**: Test Automation Specialist
