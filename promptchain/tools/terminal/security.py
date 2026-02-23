"""
Security Guardrails Module for Terminal Tool

Provides comprehensive security validation for terminal commands including:
- Destructive command detection and blocking
- Installation command permission requests
- User permission system with timeouts
- Configurable security policies

Author: PromptChain Team
License: MIT
"""

import re
import platform
import time
from typing import Dict, List, Optional, Tuple


class SecurityGuardrails:
    """Security manager for terminal command validation and permission handling"""
    
    # Destructive command patterns - these are blocked by default
    DESTRUCTIVE_PATTERNS = [
        # File system destruction
        r'^rm\s+-[rf]+\s+/',  # rm -rf on root paths
        r'^rm\s+-[rf]+\s+\*',  # rm -rf with wildcards  
        r'^rm\s+-[rf]+\s+~',  # rm -rf on home directory
        r'^sudo\s+rm',  # sudo rm commands
        
        # System level dangerous commands
        r'^dd\s+.*of=/dev/',  # dd to device files
        r'^format\s+',  # format commands
        r'>\s*/dev/(?:sda|sdb|sdc|nvme)',  # redirecting to block devices
        r'^mkfs',  # filesystem creation
        r'^fdisk',  # disk partitioning
        r'^parted',  # disk partitioning
        r'^gdisk',  # GPT partitioning
        
        # System control
        r'^shutdown',  # system shutdown
        r'^reboot',  # system reboot
        r'^halt',  # system halt
        r'^poweroff',  # power off system
        
        # Process manipulation
        r'^kill\s+-9\s+1\b',  # kill init process
        r'^killall\s+-9\s+',  # aggressive kill all
        
        # Fork bombs and malicious patterns  
        r':\(\)\{.*\|.*&\s*\};',  # classic fork bomb
        r'^:\(\)\{:\|:&\};:',  # fork bomb variant
        r'while\s+true.*do.*done',  # infinite loops
        
        # Dangerous permissions
        r'^chmod\s+777\s+/',  # world-writable root directories
        r'^chmod\s+666\s+/',  # world-writable root files
        r'^chown\s+.*\s+/',  # ownership changes on root
        
        # Network/system exposure  
        r'^iptables\s+-F',  # flush firewall rules
        r'^ufw\s+disable',  # disable firewall
        
        # Crypto/security
        r'^openssl\s+.*-out\s+/dev/',  # crypto output to devices
        r'^gpg\s+--delete-secret-key',  # delete GPG keys
    ]
    
    # Installation command patterns - require permission
    INSTALLATION_PATTERNS = [
        # Python package managers
        r'^pip\s+install',
        r'^pip3\s+install', 
        r'^poetry\s+add',
        r'^pipenv\s+install',
        r'^conda\s+install',
        
        # Node.js package managers
        r'^npm\s+install',
        r'^yarn\s+add',
        r'^pnpm\s+add',
        
        # System package managers
        r'^apt\s+install',
        r'^apt-get\s+install',
        r'^yum\s+install',
        r'^dnf\s+install',
        r'^zypper\s+install',
        r'^pacman\s+-S',
        r'^brew\s+install',
        r'^pkg\s+install',
        
        # Language-specific installers
        r'^gem\s+install',
        r'^cargo\s+install',
        r'^go\s+install',
        r'^stack\s+install',
        r'^cabal\s+install',
        
        # Container/VM installation
        r'^docker\s+pull',
        r'^docker\s+run.*--privileged',
        r'^vagrant\s+box\s+add',
    ]
    
    # File modification patterns that need caution
    CAUTION_PATTERNS = [
        # File operations that might be risky
        r'^rm\s+',  # any rm command
        r'^rmdir\s+',  # remove directories
        r'^mv\s+.*\s+/',  # moving files to root directories
        r'^cp\s+.*\s+/',  # copying to root directories
        
        # Permission/ownership changes
        r'^chmod\s+',  # permission changes
        r'^chown\s+',  # ownership changes
        r'^chgrp\s+',  # group changes
        
        # File editing/modification
        r'>\s+[^>]',  # file redirection (overwrite)
        r'^sed\s+-i',  # in-place file editing
        r'^awk\s+.*>\s+',  # awk with output redirection
        
        # System configuration
        r'^systemctl\s+',  # systemd service control
        r'^service\s+',  # service control
        r'^crontab\s+',  # cron job editing
        
        # Network configuration
        r'^ifconfig\s+',  # network interface config
        r'^ip\s+',  # network configuration
        r'^netsh\s+',  # Windows network config
        
        # Archive operations on system paths
        r'^tar\s+.*\s+/',  # extracting archives to root
        r'^unzip\s+.*\s+/',  # unzip to root directories
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize security guardrails with optional custom configuration"""
        self.config = config or {}
        
        # Security settings
        self.require_permission = self.config.get('require_permission', True)
        self.block_destructive = self.config.get('block_destructive', True)
        self.log_commands = self.config.get('log_all_commands', True)
        
        # Custom patterns
        self.custom_blocked = self.config.get('custom_blocked_patterns', [])
        self.custom_caution = self.config.get('custom_caution_patterns', [])
        self.whitelist = self.config.get('whitelist_patterns', [])
        
        # Permission settings
        self.permission_timeout = self.config.get('permission_timeout', 30)
        self.auto_approve = self.config.get('auto_approve', False)
        
        # Compile patterns for better performance
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        self.compiled_destructive = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in (self.DESTRUCTIVE_PATTERNS + self.custom_blocked)
        ]
        self.compiled_installation = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INSTALLATION_PATTERNS
        ]
        self.compiled_caution = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (self.CAUTION_PATTERNS + self.custom_caution)
        ]
        self.compiled_whitelist = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.whitelist
        ]
        
    def check_command(self, command: str) -> Tuple[bool, str, str]:
        """
        Check if a command is safe to execute
        
        Args:
            command: The command string to validate
            
        Returns:
            Tuple of (is_safe, risk_level, reason)
            risk_level: 'safe', 'caution', 'dangerous', 'blocked'
        """
        if not command.strip():
            return True, 'safe', 'Empty command'
            
        # Check whitelist first - if whitelisted, it's safe
        for pattern in self.compiled_whitelist:
            if pattern.search(command):
                return True, 'safe', 'Whitelisted command'
        
        # Check for destructive commands - these are blocked
        for pattern in self.compiled_destructive:
            if pattern.search(command):
                if self.block_destructive:
                    return False, 'blocked', f'Destructive command detected: {pattern.pattern}'
                return False, 'dangerous', f'Dangerous command pattern: {pattern.pattern}'
        
        # Check for installation commands - these need permission
        for pattern in self.compiled_installation:
            if pattern.search(command):
                return False, 'caution', 'Installation command requires permission'
        
        # Check for caution patterns - potentially risky
        for pattern in self.compiled_caution:
            if pattern.search(command):
                return True, 'caution', f'Potentially risky command: {pattern.pattern}'
        
        # Additional heuristic checks
        risk_level, reason = self._heuristic_analysis(command)
        
        return True, risk_level, reason
    
    def _heuristic_analysis(self, command: str) -> Tuple[str, str]:
        """Perform heuristic analysis for additional security checks"""
        
        # Check for suspicious patterns
        suspicious_indicators = [
            ('&& rm', 'Command chaining with rm'),
            ('; rm', 'Command chaining with rm'),
            ('`rm', 'Command substitution with rm'),
            ('\\$\\(rm', 'Command substitution with rm'),
            ('> /dev/', 'Output redirection to device'),
            ('curl.*\\|.*sh', 'Pipe curl to shell'),
            ('wget.*\\|.*sh', 'Pipe wget to shell'),
            ('base64.*decode', 'Base64 decoding (possible obfuscation)'),
            ('eval.*', 'Dynamic code evaluation'),
            ('exec.*', 'Process execution'),
        ]
        
        for indicator, reason in suspicious_indicators:
            if re.search(indicator, command, re.IGNORECASE):
                return 'caution', reason
                
        # Check command length (very long commands might be obfuscated)
        if len(command) > 500:
            return 'caution', 'Unusually long command'
            
        # Check for multiple command chaining
        chain_indicators = ['&&', '||', ';', '|']
        chain_count = sum(command.count(indicator) for indicator in chain_indicators)
        if chain_count > 3:
            return 'caution', 'Complex command chaining detected'
            
        return 'safe', 'No suspicious patterns detected'
    
    def request_permission(
        self, 
        command: str, 
        risk_level: str, 
        reason: str, 
        timeout: Optional[int] = None
    ) -> bool:
        """
        Request user permission for risky commands
        
        Args:
            command: The command to execute
            risk_level: The risk level ('safe', 'caution', 'dangerous', 'blocked')
            reason: Reason for the risk assessment
            timeout: Timeout in seconds for user response
            
        Returns:
            True if permission granted, False otherwise
        """
        if not self.require_permission:
            return True
            
        if risk_level == 'safe':
            return True
            
        if risk_level == 'blocked':
            return False
            
        # Auto-approve if configured (for testing)
        if self.auto_approve:
            return True
            
        timeout = timeout or self.permission_timeout
        
        # Display permission prompt
        self._display_permission_prompt(command, risk_level, reason)
        
        try:
            response = self._get_user_response(timeout)
            granted = response.lower() in ['yes', 'y', 'allow', 'proceed']
            
            if granted:
                print("✅ Permission granted - proceeding with command execution")
            else:
                print("❌ Permission denied - command will not be executed")
                
            return granted
            
        except Exception as e:
            print(f"\n❌ Error getting permission response: {e}")
            print("Defaulting to DENY for safety")
            return False
    
    def _display_permission_prompt(self, command: str, risk_level: str, reason: str):
        """Display a formatted permission prompt"""
        # Risk level styling
        risk_colors = {
            'caution': '🟡',
            'dangerous': '🔴',
            'blocked': '⛔'
        }
        
        icon = risk_colors.get(risk_level, '⚠️')
        
        print("\n" + "="*70)
        print(f"{icon} TERMINAL COMMAND PERMISSION REQUEST")
        print("="*70)
        print(f"Command: {command}")
        print(f"Risk Level: {risk_level.upper()}")
        print(f"Reason: {reason}")
        
        if risk_level == 'dangerous':
            print("\n🔴 WARNING: This command could potentially harm your system!")
            print("   Only proceed if you fully understand what it will do.")
            
        elif risk_level == 'caution':
            print(f"\n🟡 CAUTION: This command will modify your system.")
            print("   Review the command carefully before proceeding.")
            
        print("\nOptions:")
        print("  • Type 'yes' or 'y' to allow and execute the command")
        print("  • Type 'no' or 'n' to deny and cancel execution")  
        print("  • Press Enter or wait for timeout to deny")
        print(f"\nTimeout: {self.permission_timeout} seconds")
        print("-" * 70)
        print("Your choice: ", end='', flush=True)
    
    def _get_user_response(self, timeout: int) -> str:
        """
        Get user response with timeout support
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            User response string
            
        Raises:
            TimeoutError: If user doesn't respond within timeout
        """
        try:
            if platform.system() == 'Windows':
                # Windows doesn't support select, use simple input
                # Note: This won't have timeout on Windows
                return input().strip()
            else:
                # Unix-like systems with timeout support
                import select
                import sys
                
                ready, _, _ = select.select([sys.stdin], [], [], timeout)
                if ready:
                    response = sys.stdin.readline().strip()
                    return response
                else:
                    print(f"\n⏰ Timeout ({timeout}s) - permission denied")
                    raise TimeoutError("User response timeout")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted - permission denied")
            return "no"
        except Exception as e:
            print(f"\n❌ Input error: {e}")
            raise
    
    def is_safe_command(self, command: str) -> bool:
        """
        Quick safety check - returns True only if command is completely safe
        
        Args:
            command: Command to check
            
        Returns:
            True if safe, False if needs permission or is blocked
        """
        is_safe, risk_level, _ = self.check_command(command)
        return is_safe and risk_level == 'safe'
    
    def get_blocked_patterns(self) -> List[str]:
        """Get list of all blocked command patterns"""
        return self.DESTRUCTIVE_PATTERNS + self.custom_blocked
    
    def add_custom_pattern(self, pattern: str, pattern_type: str = 'caution'):
        """
        Add a custom security pattern
        
        Args:
            pattern: Regex pattern to add
            pattern_type: 'blocked', 'caution', or 'whitelist'
        """
        if pattern_type == 'blocked':
            self.custom_blocked.append(pattern)
        elif pattern_type == 'caution':
            self.custom_caution.append(pattern)
        elif pattern_type == 'whitelist':
            self.whitelist.append(pattern)
        
        # Re-compile patterns
        self._compile_patterns()