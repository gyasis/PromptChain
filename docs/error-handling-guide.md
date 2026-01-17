# PromptChain CLI Error Handling Guide

## Overview

The PromptChain CLI implements comprehensive error handling with graceful crash recovery, user-friendly messages, and automatic retry mechanisms (Task T141).

## Architecture

### Core Components

1. **ErrorHandler** (`promptchain/cli/error_handler.py`)
   - Centralized error classification and handling
   - Auto-retry with exponential backoff
   - Error tracking and history management
   - User-friendly message generation

2. **ErrorLogger** (`promptchain/cli/utils/error_logger.py`)
   - Session-specific JSONL error logging (T143)
   - Structured error data with metadata
   - Error history retrieval

3. **Global Exception Handler** (`PromptChainApp.on_exception`)
   - Catches all unhandled exceptions in Textual app
   - Graceful recovery without crashing
   - Auto-saves session before potential crash

## Error Categories

The error handler classifies errors into specific categories for targeted handling:

### API_AUTH
**Description**: API authentication/authorization failures
**Severity**: ERROR
**Retryable**: No

**Common Causes**:
- Missing API key environment variable
- Invalid API key
- Expired credentials

**User Message Example**:
```
API Authentication Failed

The API key for your model is missing or invalid.

Set environment variable:
  export OPENAI_API_KEY=your_key

Add your API keys to a .env file or export as environment variables.
```

**Recovery Hint**: Check .env file and verify API key is valid

### API_RATE_LIMIT
**Description**: Rate limiting from API provider
**Severity**: WARNING
**Retryable**: Yes (with exponential backoff)

**Common Causes**:
- Too many requests in short period
- Exceeded API tier limits

**User Message Example**:
```
Rate Limit Exceeded

You've hit the API rate limit. The request will be retried automatically.

Tips:
  • Automatic retry in progress (with exponential backoff)
  • Use a different model with higher limits
  • Upgrade your API plan for higher rate limits
```

**Recovery Hint**: Wait for automatic retry or switch to different model

### API_TIMEOUT
**Description**: Request timeout
**Severity**: WARNING
**Retryable**: Yes

**Common Causes**:
- Slow API response
- Large request/response
- Network congestion

**Recovery Hint**: Wait for automatic retry or simplify request

### NETWORK
**Description**: Network connectivity issues
**Severity**: ERROR
**Retryable**: Yes

**Common Causes**:
- No internet connection
- API server down
- Firewall blocking
- DNS resolution failure

**Recovery Hint**: Verify internet connection and API server status

### FILE_IO
**Description**: File system operations
**Severity**: ERROR
**Retryable**: No

**Common Causes**:
- File not found
- Directory not found
- Invalid file path

**User Message Example**:
```
File Error

Could not access: test.txt

Current directory: /home/user/project

Check the file path and try again.
Use @path/to/file syntax for file references.
```

### PERMISSION
**Description**: Permission denied
**Severity**: ERROR
**Retryable**: No

**Recovery Hint**: Check permissions with `ls -l`

### VALIDATION
**Description**: Input validation failures
**Severity**: ERROR
**Retryable**: No

**Common Causes**:
- Invalid model name
- Malformed input
- Constraint violations

### SESSION
**Description**: Session management errors
**Severity**: ERROR
**Retryable**: No

**Common Causes**:
- Corrupted session data
- Database access issues

### AGENT
**Description**: Agent initialization/execution
**Severity**: ERROR
**Retryable**: No

### MCP
**Description**: MCP server communication
**Severity**: WARNING
**Retryable**: No

**Recovery Hint**: Check MCP server configuration and restart if needed

## Auto-Retry Mechanism

### Configuration

```python
RetryConfig(
    max_attempts=3,           # Maximum retry attempts
    base_delay=1.0,           # Initial delay (seconds)
    max_delay=30.0,          # Maximum delay cap
    exponential_base=2.0,    # Exponential backoff multiplier
    retryable_categories=[   # Which errors to retry
        ErrorCategory.API_RATE_LIMIT,
        ErrorCategory.API_TIMEOUT,
        ErrorCategory.NETWORK,
    ]
)
```

### Exponential Backoff

Delays between retries grow exponentially:

- Attempt 1: No delay
- Attempt 2: 1.0s delay
- Attempt 3: 2.0s delay
- Attempt 4: 4.0s delay
- ...up to max_delay

### Example Usage

```python
# Automatically retries on transient errors
async def _generate_response():
    return await active_chain.process_prompt_async(input)

response = await error_handler.handle_with_retry(
    _generate_response,
    "generating response from agent"
)
```

## Error Tracking

### In-Memory History

```python
# Track error
error_context = error_handler.classify_error(error, "operation")
error_handler.track_error(error_context)

# Retrieve recent errors
recent_errors = error_handler.get_recent_errors(count=10)

# Filter by category
network_errors = error_handler.get_recent_errors(
    count=10,
    category=ErrorCategory.NETWORK
)
```

### JSONL Session Logs (T143)

Errors are automatically logged to `~/.promptchain/sessions/<session-id>/errors.jsonl`:

```json
{
  "timestamp": "2025-01-18T10:30:45.123456",
  "error_type": "ConnectionError",
  "error_message": "Failed to connect to API",
  "context": "generating response",
  "metadata": {
    "category": "network",
    "severity": "error",
    "user_message": "...",
    "recovery_hint": "..."
  },
  "stack_trace": "Traceback (most recent call last):\n..."
}
```

## Global Exception Handler

The `PromptChainApp.on_exception` method catches all unhandled exceptions:

```python
def on_exception(self, event) -> None:
    """Global exception handler for Textual app."""
    error = event.exception

    # Handle error with comprehensive handler
    error_msg = self._handle_error(error, "application error")

    # Display in chat view
    chat_view.add_message(error_msg)

    # Save session state before potential crash
    if self.session:
        self.session_manager.save_session(self.session)

    # Prevent app crash - allow continued operation
    event.prevent_default()
```

**Features**:
- Catches all unhandled exceptions
- Displays user-friendly error message
- Auto-saves session before crash
- Allows app to continue running
- Graceful degradation

## Debug Mode

Enable debug mode for detailed error information:

```bash
# Environment variable
export PROMPTCHAIN_DEBUG=1

# Or in config.json
{
  "debug_mode": true
}
```

**Debug Mode Effects**:
- Includes full stack traces in error messages
- Logs tracebacks to JSONL session logs
- Shows internal error details

## Error Message Structure

### User-Facing Message

```
[Severity Level] Error Title

Brief description of what went wrong.

Common causes:
  • Cause 1
  • Cause 2
  • Cause 3

Additional helpful information...

Recovery suggestion:
  Steps to resolve the issue
```

### Metadata

All error messages include structured metadata:

```python
{
    "error": True,
    "error_type": "ConnectionError",
    "error_category": "network",
    "error_severity": "error",
    "error_message": "Failed to connect",
    "operation": "API call",
    "recovery_hint": "Check internet connection",
    "timestamp": 1705574445.123,
    "traceback": "..." (if debug_mode)
}
```

## Error Recovery Strategies

### Automatic Recovery

1. **Auto-Retry**: Transient errors (rate limits, timeouts, network) are automatically retried
2. **Exponential Backoff**: Delays increase to avoid overwhelming services
3. **Session Preservation**: Session state saved before potential crash
4. **Graceful Degradation**: MCP errors allow operation to continue without tools

### Manual Recovery

Error messages include actionable recovery hints:

- **API Auth Errors**: Set environment variable, check .env file
- **Rate Limits**: Wait for retry, use different model, upgrade plan
- **File Errors**: Verify file path, check permissions
- **Model Errors**: Check model name spelling, use different model
- **Network Errors**: Check internet connection, verify API status

## Testing

Comprehensive test suite in `tests/cli/test_error_handler.py`:

- Error classification for all categories
- Message generation with/without debug mode
- Retry mechanism with exponential backoff
- Error tracking and history
- Recovery hint generation
- Metadata population

Run tests:
```bash
pytest tests/cli/test_error_handler.py -v
```

## Best Practices

### For Users

1. **Check Error Messages**: Read the full error message and recovery hints
2. **Enable Debug Mode**: For persistent issues, enable debug mode for details
3. **Review Error Logs**: Check `errors.jsonl` for detailed error history
4. **Report Issues**: Include error log excerpts when reporting bugs

### For Developers

1. **Use Error Handler**: Always use `error_handler.classify_error()` for errors
2. **Provide Context**: Include operation context in error classification
3. **Add Recovery Hints**: Provide actionable suggestions for users
4. **Test Error Paths**: Write tests for error scenarios
5. **Log Metadata**: Include relevant metadata for debugging

## Example Scenarios

### Scenario 1: Rate Limit with Auto-Retry

```
User sends message → LLM call → Rate limit error (429)
↓
Error handler classifies as API_RATE_LIMIT
↓
Shows user: "Rate Limit Exceeded - retrying automatically..."
↓
Wait 1 second → Retry #1 → Still rate limited
↓
Wait 2 seconds → Retry #2 → Success!
↓
Display response to user
```

### Scenario 2: Missing API Key

```
User sends message → LLM call → Authentication error
↓
Error handler classifies as API_AUTH
↓
Shows user:
  "API Authentication Failed
   Set environment variable: OPENAI_API_KEY=your_key
   Recovery: Check .env file and verify API key"
↓
Error logged to errors.jsonl
↓
App continues running (no crash)
```

### Scenario 3: File Not Found

```
User: "@nonexistent.txt explain this"
↓
File loader tries to read → FileNotFoundError
↓
Error handler classifies as FILE_IO
↓
Shows user:
  "File Error
   Could not access: nonexistent.txt
   Current directory: /home/user/project
   Recovery: Verify file path and check permissions"
↓
App continues (doesn't crash on bad file reference)
```

### Scenario 4: Global Exception (Unhandled)

```
Internal bug causes unhandled exception
↓
Textual catches via PromptChainApp.on_exception()
↓
Error handler classifies and displays message
↓
Session auto-saved before potential crash
↓
User sees: "Unexpected Error - session saved, you can continue"
↓
App continues running (prevented crash)
```

## Configuration

### Retry Configuration

Customize retry behavior in `PromptChainApp.__init__`:

```python
self.error_handler = ErrorHandler(
    retry_config=RetryConfig(
        max_attempts=5,      # More retries
        base_delay=2.0,      # Longer initial delay
        max_delay=60.0,      # Higher delay cap
    ),
    debug_mode=config.debug_mode,
    session_id=session.id,
    sessions_dir=sessions_dir,
)
```

### Debug Mode

Enable in config.json:

```json
{
  "debug_mode": true,
  "default_model": "openai/gpt-4",
  ...
}
```

Or environment variable:

```bash
export PROMPTCHAIN_DEBUG=1
```

## Troubleshooting

### Error Logs Not Created

**Cause**: Session directory doesn't exist or no write permissions
**Solution**: Check `~/.promptchain/sessions/<session-id>/` exists and is writable

### Retries Not Working

**Cause**: Error not in `retryable_categories`
**Solution**: Check `RetryConfig.retryable_categories` includes the error category

### Debug Mode Not Showing Tracebacks

**Cause**: Debug mode not enabled
**Solution**: Set `PROMPTCHAIN_DEBUG=1` or `config.debug_mode = true`

### Session Not Saving on Error

**Cause**: Exception in save logic itself
**Solution**: Check `errors.jsonl` for save-related errors

## Related Tasks

- **T141**: Global error handler implementation ✓
- **T142**: Specific error handlers (API, network, file, etc.) ✓
- **T143**: Error logging to JSONL session logs ✓
- **T144**: Auto-retry mechanism with exponential backoff ✓
- **T145**: User-friendly error messages ✓
- **T146**: Recovery hints and suggestions ✓

## Future Enhancements

1. **Error Analytics**: Aggregate error statistics across sessions
2. **Smart Recovery**: Automatic fallback to alternative models on errors
3. **Error Notifications**: Desktop notifications for critical errors
4. **Error Patterns**: Detect and warn about recurring error patterns
5. **Custom Recovery Actions**: User-defined recovery scripts
