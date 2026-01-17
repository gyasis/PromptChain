# Blackboard Tools Test Guide

## Quick Reference

### Running Tests

```bash
# Run all blackboard tool tests
python -m pytest tests/cli/unit/test_blackboard_tools.py -v

# Run specific test class
python -m pytest tests/cli/unit/test_blackboard_tools.py::TestWriteToBlackboard -v

# Run specific test
python -m pytest tests/cli/unit/test_blackboard_tools.py::TestWriteToBlackboard::test_write_dict_value -v

# Run with output capture disabled (see print statements)
python -m pytest tests/cli/unit/test_blackboard_tools.py -v -s
```

### Expected Results
```
44 tests, all passing
Execution time: ~4-7 seconds
```

## Test Structure

### 7 Test Classes, 44 Total Tests

1. **TestWriteToBlackboard** (12 tests)
   - Data types: string, dict, list, numeric, boolean, None, nested
   - Validation: empty key, whitespace key, empty written_by
   - Updates: version increments

2. **TestReadFromBlackboard** (9 tests)
   - Data types: all types from write tests
   - Edge cases: non-existent keys, after deletion
   - Metadata: version tracking, author tracking

3. **TestListBlackboardKeys** (6 tests)
   - States: empty, single, multiple keys
   - Sorting: alphabetical order
   - Mutations: after deletion

4. **TestDeleteBlackboardEntry** (5 tests)
   - Basic: delete existing, delete non-existent
   - Integration: read after delete, list after delete
   - Idempotency: delete twice

5. **TestSessionManagerIntegration** (6 tests)
   - Initialization: set/get session manager
   - Error handling: all operations without SM

6. **TestJSONSerializationRoundTrip** (3 tests)
   - Dict, list, nested structures
   - Verifies JSON serialization works correctly

7. **TestVersionTracking** (3 tests)
   - Initial version: starts at 1
   - Increments: v1 → v2 → v3
   - Independence: each key has own version

## Writing New Tests

### Test Pattern Template

```python
def test_feature_description(self, mock_session_manager):
    """Test that feature does X when Y happens."""
    # Arrange: Set up test data
    test_key = "test_key"
    test_value = "test_value"

    # Act: Execute the function
    result = blackboard_tools.write_to_blackboard(
        key=test_key,
        value=test_value,
        written_by="test_agent"
    )

    # Assert: Verify results
    assert "expected_string" in result
    mock_session_manager.write_blackboard.assert_called_once()
```

### Fixture Usage

```python
# Automatic fixture (runs for all tests)
@pytest.fixture(autouse=True)
def setup_session_manager(mock_session_manager):
    blackboard_tools.set_session_manager(mock_session_manager)
    yield
    blackboard_tools._session_manager = None

# Manual fixture (use when needed)
def test_something(mock_session_manager):
    # mock_session_manager available here
    pass
```

## Common Test Scenarios

### Testing Data Types

```python
# String
blackboard_tools.write_to_blackboard("key", "value", "agent")

# Dict
blackboard_tools.write_to_blackboard("key", {"a": 1}, "agent")

# List
blackboard_tools.write_to_blackboard("key", [1, 2, 3], "agent")

# Numeric
blackboard_tools.write_to_blackboard("key", 42, "agent")
blackboard_tools.write_to_blackboard("key", 3.14, "agent")

# Boolean
blackboard_tools.write_to_blackboard("key", True, "agent")

# None
blackboard_tools.write_to_blackboard("key", None, "agent")
```

### Testing Validation

```python
# Empty key
result = blackboard_tools.write_to_blackboard("", "value", "agent")
assert result == "Error: Key cannot be empty"

# Whitespace key
result = blackboard_tools.write_to_blackboard("   ", "value", "agent")
assert result == "Error: Key cannot be empty"

# Empty written_by
result = blackboard_tools.write_to_blackboard("key", "value", "")
assert result == "Error: written_by cannot be empty"
```

### Testing Version Tracking

```python
# First write (v1)
blackboard_tools.write_to_blackboard("key", "v1", "agent1")
result = blackboard_tools.read_from_blackboard("key")
assert "(v1, by agent1)" in result

# Update (v2)
blackboard_tools.write_to_blackboard("key", "v2", "agent2")
result = blackboard_tools.read_from_blackboard("key")
assert "(v2, by agent2)" in result
```

### Testing CRUD Operations

```python
# Create
blackboard_tools.write_to_blackboard("key", "value", "agent")

# Read
result = blackboard_tools.read_from_blackboard("key")
assert "value" in result

# Update
blackboard_tools.write_to_blackboard("key", "new_value", "agent2")

# Delete
result = blackboard_tools.delete_blackboard_entry("key")
assert "deleted" in result

# Verify deletion
result = blackboard_tools.read_from_blackboard("key")
assert "No entry found" in result
```

## Mock Session Manager

### Interface

```python
mock_sm._blackboard_storage  # Internal storage dict

# Methods
mock_sm.write_blackboard(key, value, written_by) -> BlackboardEntry
mock_sm.read_blackboard(key) -> Optional[BlackboardEntry]
mock_sm.list_blackboard_keys() -> List[str]
mock_sm.delete_blackboard_entry(key) -> bool
```

### Behavior

- **write_blackboard**: Creates new entry (v1) or updates existing (v++)
- **read_blackboard**: Returns entry or None
- **list_blackboard_keys**: Returns list of keys (can be empty)
- **delete_blackboard_entry**: Returns True if deleted, False if not found

### Accessing Mock Storage

```python
# Direct access for verification
assert "my_key" in mock_session_manager._blackboard_storage
entry = mock_session_manager._blackboard_storage["my_key"]
assert entry.version == 2
```

## Assertion Patterns

### String Assertions

```python
# Exact match
assert result == "Blackboard is empty"

# Contains
assert "Blackboard entry 'key' written" in result
assert "(v2, by agent)" in result

# Multiple conditions
assert "key1" in result
assert "key2" in result
assert "Blackboard keys (2)" in result
```

### Mock Assertions

```python
# Called once with specific args
mock_session_manager.write_blackboard.assert_called_once_with(
    key="test",
    value="value",
    written_by="agent"
)

# Not called
mock_session_manager.write_blackboard.assert_not_called()

# Called with any args
assert mock_session_manager.read_blackboard.called
```

## Edge Cases Covered

### Input Validation
- Empty strings (key, written_by)
- Whitespace-only strings
- None values (valid for value, invalid for key/written_by)

### State Transitions
- Write → Read → Update → Read → Delete → Read
- List empty → Write → List → Delete → List empty

### Concurrent Operations
- Multiple writes to same key (version tracking)
- Multiple keys with independent versions
- Delete while other keys exist

### Error Conditions
- Session manager not initialized
- Non-existent keys
- Idempotent operations (delete twice)

## Testing Best Practices

### 1. Arrange-Act-Assert Pattern
```python
# Arrange: Set up test data
key = "test_key"

# Act: Execute function
result = blackboard_tools.write_to_blackboard(key, "value", "agent")

# Assert: Verify outcome
assert "written" in result
```

### 2. One Assertion Per Concern
```python
# Good: Test one thing
def test_write_returns_success_message(self, mock_session_manager):
    result = blackboard_tools.write_to_blackboard("k", "v", "a")
    assert "written" in result

# Good: Test another thing separately
def test_write_calls_session_manager(self, mock_session_manager):
    blackboard_tools.write_to_blackboard("k", "v", "a")
    mock_session_manager.write_blackboard.assert_called_once()
```

### 3. Descriptive Test Names
```python
# Good: Describes what and when
def test_read_non_existent_key_returns_not_found(self, mock_session_manager):
    pass

# Bad: Vague
def test_read(self, mock_session_manager):
    pass
```

### 4. Test Independence
```python
# Each test should work in isolation
# Don't rely on test execution order
# Use fixtures to ensure clean state
```

### 5. Clear Failure Messages
```python
# Good: Context in assertion
assert len(keys) == 3, f"Expected 3 keys, got {len(keys)}: {keys}"

# Good: Use equality for better pytest output
assert result == "Blackboard is empty"  # Shows diff on failure
```

## Troubleshooting

### Import Errors
```python
# If you see import errors, check:
1. Python path includes project root
2. __init__.py files exist
3. Dependencies installed (pytest)
```

### Mock Not Working
```python
# Verify fixture is used
def test_something(mock_session_manager):  # ✅ Correct
    pass

def test_something():  # ❌ Missing fixture
    pass
```

### Session Manager Not Initialized
```python
# Make sure autouse fixture is working
@pytest.fixture(autouse=True)  # ✅ Auto-runs
def setup_session_manager(mock_session_manager):
    blackboard_tools.set_session_manager(mock_session_manager)
    yield
    blackboard_tools._session_manager = None
```

### Version Not Incrementing
```python
# Make sure you're reading the updated value
blackboard_tools.write_to_blackboard("k", "v1", "a1")
blackboard_tools.write_to_blackboard("k", "v2", "a2")  # Update
result = blackboard_tools.read_from_blackboard("k")  # Read AFTER update
assert "(v2" in result
```

## Adding New Tests

### Checklist
- [ ] Test name describes what and when
- [ ] Docstring explains the test
- [ ] Uses mock_session_manager fixture
- [ ] Follows Arrange-Act-Assert pattern
- [ ] Makes clear assertions
- [ ] Tests one concern
- [ ] Handles edge cases
- [ ] Independent of other tests

### Example: Adding a New Test

```python
def test_write_updates_timestamp(self, mock_session_manager):
    """Test that updating an entry updates the timestamp."""
    # Arrange
    import time

    # Act: Write initial
    blackboard_tools.write_to_blackboard("key", "v1", "agent")
    entry1 = mock_session_manager._blackboard_storage["key"]
    time1 = entry1.written_at

    # Wait a bit
    time.sleep(0.1)

    # Act: Update
    blackboard_tools.write_to_blackboard("key", "v2", "agent")
    entry2 = mock_session_manager._blackboard_storage["key"]
    time2 = entry2.written_at

    # Assert: Timestamp updated
    assert time2 > time1
```

## Summary

- **44 comprehensive tests** covering all blackboard tool functions
- **Mock-based** for fast, isolated testing
- **100% function coverage** of public API
- **All edge cases** tested (validation, errors, state transitions)
- **JSON serialization** verified for all data types
- **Version tracking** fully tested
- **Clear patterns** for extending the test suite
