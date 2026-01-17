# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Security Features

PromptChain CLI implements multiple security layers to protect against common vulnerabilities:

### 1. YAML Injection Prevention (OWASP A03:2021)

**Protection:**
- Uses `yaml.safe_load()` exclusively (prevents arbitrary code execution)
- Schema validation with whitelisted fields
- Depth limits to prevent YAML bombs
- Size limits to prevent DoS attacks
- Pattern matching to block code execution attempts

**Example Attack Blocked:**
```yaml
# ❌ BLOCKED: Attempt to execute Python code
!!python/object/apply:os.system
args: ['rm -rf /']

# ❌ BLOCKED: Code execution in string
agents:
  malicious:
    instruction_chain:
      - "__import__('os').system('ls')"
```

**Configuration:**
```python
from promptchain.cli.security.yaml_validator import YAMLValidator

validator = YAMLValidator(
    max_file_size=1024 * 1024,  # 1MB
    max_depth=10,
    max_string_length=10000,
    max_collection_size=100,
)
```

### 2. Path Traversal Prevention (OWASP A01:2021)

**Protection:**
- Path validation against base directory
- Null byte injection detection
- Symlink resolution with boundary checks
- Whitelist of allowed system paths

**Example Attack Blocked:**
```python
# ❌ BLOCKED: Directory traversal
session_name = "../../../etc/passwd"

# ❌ BLOCKED: Null byte injection
filename = "safe.txt\x00../../etc/shadow"

# ❌ BLOCKED: Access to system directories
working_dir = "/etc/shadow"
```

**Safe Usage:**
```python
from promptchain.cli.security.input_sanitizer import PathValidator

validator = PathValidator(base_dir="/home/user/.promptchain")
safe_path = validator.validate_path("sessions/my-session")
# Returns: /home/user/.promptchain/sessions/my-session
```

### 3. Input Sanitization

**Protection:**
- Name validation (alphanumeric + dash/underscore only)
- Model name whitelist (openai/, anthropic/, ollama/, gemini/)
- Instruction chain validation (no code execution patterns)
- String length limits
- Control character stripping

**Example Attack Blocked:**
```python
# ❌ BLOCKED: Invalid characters
agent_name = "agent@#$%"

# ❌ BLOCKED: Model not in whitelist
model = "malicious/backdoor"

# ❌ BLOCKED: Code execution in instruction
instruction = "eval('__import__(\"os\").system(\"whoami\")')"
```

**Safe Usage:**
```python
from promptchain.cli.security.input_sanitizer import InputSanitizer

sanitizer = InputSanitizer()

# Validate inputs
agent_name = sanitizer.validate_name("my-agent")
model = sanitizer.validate_model_name("openai/gpt-4")
instructions = sanitizer.validate_instruction_chain([
    "Analyze: {input}",
    {"type": "agentic_step", "objective": "Research"}
])
```

### 4. SQL Injection Prevention

**Protection:**
- Parameterized queries exclusively (no string concatenation)
- SQLite with PRAGMA foreign_keys enabled
- Input validation before database operations

**Safe Pattern:**
```python
# ✅ SAFE: Parameterized query
conn.execute(
    "SELECT * FROM sessions WHERE name = ?",
    (session_name,)
)

# ❌ UNSAFE: String concatenation (not used in codebase)
# query = f"SELECT * FROM sessions WHERE name = '{session_name}'"
```

### 5. Defense in Depth

PromptChain implements multiple validation layers:

1. **YAML Layer:** Schema validation, safe loading
2. **Input Layer:** Sanitization, whitelist validation
3. **Path Layer:** Directory traversal prevention
4. **Database Layer:** Parameterized queries
5. **Session Layer:** Path validation, name sanitization

## Security Best Practices

### For Users

1. **YAML Configuration Files:**
   - Only load YAML from trusted sources
   - Review configuration files before use
   - Avoid copying YAML configs from untrusted sources

2. **Session Storage:**
   - Use default session directory (`~/.promptchain/sessions/`)
   - Avoid custom session directories in sensitive locations
   - Regularly review session contents

3. **MCP Server Configuration:**
   - Only configure MCP servers from trusted sources
   - Review command and arguments before enabling auto_connect
   - Monitor MCP server connections

4. **Working Directories:**
   - Use project-specific working directories
   - Avoid setting working directory to system folders
   - Verify working directory before execution

### For Developers

1. **Adding New Features:**
   - Use `InputSanitizer` for all user inputs
   - Use `PathValidator` for all file paths
   - Use `YAMLValidator` for configuration files
   - Never use `yaml.load()`, always use `yaml.safe_load()`

2. **Database Operations:**
   - Always use parameterized queries
   - Never concatenate user input into SQL
   - Validate inputs before database operations

3. **File Operations:**
   - Validate paths before file I/O
   - Use `PathValidator.validate_path()` with base_dir
   - Check file permissions before operations

4. **Code Review Checklist:**
   - [ ] All user inputs validated
   - [ ] All file paths validated
   - [ ] YAML loaded with `safe_load()`
   - [ ] SQL queries parameterized
   - [ ] No code execution in user input
   - [ ] Error messages don't leak sensitive info

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability:

### DO:
- Report via GitHub Security Advisories (preferred)
- Email security@promptchain.dev (if GitHub unavailable)
- Provide detailed reproduction steps
- Include affected versions
- Suggest fix if possible

### DON'T:
- Publicly disclose the vulnerability before fix
- Test vulnerability on production systems
- Attempt to access other users' data

### Response Timeline:
- **24 hours:** Initial acknowledgment
- **7 days:** Severity assessment and triage
- **30 days:** Fix development and testing
- **90 days:** Public disclosure (coordinated)

### Severity Levels:

**Critical:** Remote code execution, arbitrary file access, data breach
- Response: Immediate patch, security advisory, CVE assignment

**High:** Authentication bypass, privilege escalation, injection attacks
- Response: Priority patch within 7 days, security advisory

**Medium:** DoS, information disclosure, security misconfiguration
- Response: Patch within 30 days, release notes

**Low:** Low-impact information disclosure, minor security improvements
- Response: Patch in next release, changelog

## Security Updates

Subscribe to security updates:
- GitHub Watch: Releases only
- Security Advisories: https://github.com/yourusername/PromptChain/security/advisories
- Changelog: Track `[SECURITY]` tags

## Acknowledgments

We thank the following researchers for responsible disclosure:

- *Your name could be here! Report vulnerabilities responsibly.*

## Security Testing

### Running Security Tests

```bash
# Run all security tests
pytest tests/cli/security/ -v

# Run specific test categories
pytest tests/cli/security/test_yaml_injection.py -v
pytest tests/cli/security/test_input_sanitization.py -v

# Run with coverage
pytest tests/cli/security/ --cov=promptchain.cli.security --cov-report=html
```

### Security Checklist for New Features

- [ ] All user inputs validated with `InputSanitizer`
- [ ] All file paths validated with `PathValidator`
- [ ] YAML configs validated with `YAMLValidator`
- [ ] SQL queries use parameterized binding
- [ ] No shell command injection vectors
- [ ] No template injection vectors
- [ ] Error messages don't leak sensitive data
- [ ] Security tests added for new attack vectors
- [ ] Documentation updated with security notes

## Compliance

PromptChain security features align with:

- **OWASP Top 10 2021:**
  - A01:2021 - Broken Access Control (path validation)
  - A03:2021 - Injection (YAML, SQL, code injection prevention)
  - A04:2021 - Insecure Design (defense in depth)
  - A05:2021 - Security Misconfiguration (secure defaults)

- **CWE Common Weaknesses:**
  - CWE-22: Path Traversal
  - CWE-89: SQL Injection
  - CWE-94: Code Injection
  - CWE-502: Deserialization of Untrusted Data
  - CWE-611: XML External Entities (YAML equivalent)

## License

Security features are part of PromptChain and covered by the project license.

## Contact

- Security Issues: security@promptchain.dev
- General Issues: GitHub Issues
- Documentation: https://github.com/yourusername/PromptChain/docs
