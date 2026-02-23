"""
Terminal Session Manager for PromptChain Terminal Tool

Provides persistent terminal sessions with named identifiers that maintain:
- Environment variables across commands
- Working directory persistence  
- Environment activation state (NVM, conda, etc.)
- Command history per session
- Session switching capabilities

This solves the core limitation where each command ran in a new subprocess,
losing all state between commands.

Author: PromptChain Team
License: MIT
"""

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import pty
import select
import signal


@dataclass
class SessionState:
    """State of a terminal session"""
    session_id: str
    name: str
    working_directory: str
    environment_variables: Dict[str, str] = field(default_factory=dict)
    active_environment: Optional[Dict[str, str]] = None
    command_history: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    process_pid: Optional[int] = None
    master_fd: Optional[int] = None
    slave_fd: Optional[int] = None


class PersistentTerminalSession:
    """A persistent terminal session that maintains state across commands"""
    
    def __init__(self, session_id: str, name: str, working_directory: str = None, 
                 environment_variables: Dict[str, str] = None):
        self.session_id = session_id
        self.name = name
        self.working_directory = working_directory or os.getcwd()
        self.environment_variables = environment_variables or {}
        self.command_history = []
        self.created_at = time.time()
        self.last_accessed = time.time()
        
        # PTY for persistent shell
        self.master_fd = None
        self.slave_fd = None
        self.process = None
        self.shell_ready = False
        self._setup_persistent_shell()
        
    def _setup_persistent_shell(self):
        """Setup a persistent shell process using PTY"""
        try:
            # Create pseudo-terminal
            self.master_fd, self.slave_fd = pty.openpty()
            
            # Prepare environment
            env = os.environ.copy()
            env.update(self.environment_variables)
            env['PS1'] = f'[{self.name}]$ '  # Custom prompt
            
            # Start persistent bash process
            self.process = subprocess.Popen(
                ['/bin/bash', '--login', '-i'],
                stdin=self.slave_fd,
                stdout=self.slave_fd,
                stderr=self.slave_fd,
                env=env,
                cwd=self.working_directory,
                preexec_fn=os.setsid
            )
            
            # Close slave fd in parent (child process will use it)
            os.close(self.slave_fd)
            self.slave_fd = None
            
            # Set non-blocking mode for master
            import fcntl
            fcntl.fcntl(self.master_fd, fcntl.F_SETFL, os.O_NONBLOCK)
            
            # Wait for shell to be ready
            time.sleep(0.5)
            self._drain_initial_output()
            self.shell_ready = True
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to setup persistent shell: {e}")
    
    def _drain_initial_output(self):
        """Drain initial shell output/prompts"""
        try:
            while True:
                ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                if not ready:
                    break
                try:
                    data = os.read(self.master_fd, 1024)
                    if not data:
                        break
                except OSError:
                    break
        except:
            pass
    
    def execute_command(self, command: str, timeout: int = 60) -> Tuple[str, int]:
        """
        Execute a command in the persistent session
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (output, return_code)
        """
        if not self.shell_ready or not self.master_fd:
            raise RuntimeError("Session not ready or closed")
        
        self.last_accessed = time.time()
        self.command_history.append(command)
        
        try:
            # Simple approach: execute command and capture stdout/stderr via redirection
            import tempfile
            import uuid
            
            # Create temporary files for output capture
            temp_id = str(uuid.uuid4())[:8]
            stdout_file = f"/tmp/promptchain_out_{temp_id}"
            stderr_file = f"/tmp/promptchain_err_{temp_id}"
            
            # Execute command with output redirection and return code capture
            exec_cmd = (f"({command}) > {stdout_file} 2> {stderr_file}; "
                       f"echo $? > {stdout_file}.rc\n")
            
            os.write(self.master_fd, exec_cmd.encode('utf-8'))
            
            # Wait for command to complete by checking for the return code file
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if return code file exists
                check_cmd = f"test -f {stdout_file}.rc && echo 'DONE'\n"
                os.write(self.master_fd, check_cmd.encode('utf-8'))
                
                # Read any output to keep the pipe clear
                try:
                    ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                    if ready:
                        data = os.read(self.master_fd, 4096).decode('utf-8', errors='replace')
                        if 'DONE' in data:
                            break
                except OSError:
                    pass
                
                time.sleep(0.1)
            else:
                # Cleanup on timeout
                cleanup_cmd = f"rm -f {stdout_file} {stderr_file} {stdout_file}.rc\n"
                os.write(self.master_fd, cleanup_cmd.encode('utf-8'))
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            
            # Read the results from the temp files
            read_stdout_cmd = f"cat {stdout_file}\n"
            read_stderr_cmd = f"cat {stderr_file}\n"
            read_rc_cmd = f"cat {stdout_file}.rc\n"
            cleanup_cmd = f"rm -f {stdout_file} {stderr_file} {stdout_file}.rc\n"
            
            # Execute read commands and capture their output
            os.write(self.master_fd, read_stdout_cmd.encode('utf-8'))
            time.sleep(0.1)
            
            # Read stdout content
            stdout_content = ""
            start_read = time.time()
            while time.time() - start_read < 5:
                ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                if ready:
                    try:
                        data = os.read(self.master_fd, 4096).decode('utf-8', errors='replace')
                        if data and not data.isspace():
                            # Simple extraction: everything before a prompt-like line
                            lines = data.split('\n')
                            for line in lines:
                                cleaned = line.strip()
                                if cleaned and not self._is_prompt_line(cleaned) and 'cat ' not in cleaned:
                                    stdout_content += cleaned + '\n'
                            break
                    except OSError:
                        pass
                else:
                    break
            
            # Read return code
            os.write(self.master_fd, read_rc_cmd.encode('utf-8'))
            time.sleep(0.1)
            
            return_code = 0
            start_read = time.time()
            while time.time() - start_read < 2:
                ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                if ready:
                    try:
                        data = os.read(self.master_fd, 4096).decode('utf-8', errors='replace')
                        if data:
                            # Extract return code
                            lines = data.split('\n')
                            for line in lines:
                                cleaned = line.strip()
                                if cleaned.isdigit():
                                    return_code = int(cleaned)
                                    break
                            break
                    except OSError:
                        pass
                else:
                    break
            
            # Cleanup
            os.write(self.master_fd, cleanup_cmd.encode('utf-8'))
            
            return stdout_content.strip(), return_code
            
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {e}")
    
    def _clean_terminal_output(self, line: str) -> str:
        """Clean terminal control sequences from output line"""
        import re
        
        # Remove common terminal control sequences
        line = re.sub(r'\x1b\[[?]?[0-9;]*[a-zA-Z]', '', line)  # ANSI escape sequences
        line = re.sub(r'\x1b\[[\?]?[0-9]+[hlm]', '', line)     # Terminal mode changes
        line = line.replace('\x1b[?2004l', '').replace('\x1b[?2004h', '')  # Bracketed paste
        line = line.replace('\r', '').strip()  # Remove carriage returns
        
        return line
    
    def _is_prompt_line(self, line: str) -> bool:
        """Check if line is a shell prompt"""
        # Common prompt patterns
        prompt_patterns = [
            r'.*@.*:.*\$\s*$',  # user@host:path$
            r'.*\$\s*$',        # simple $
            r'.*#\s*$',         # root #
            r'^\s*$'            # empty line
        ]
        
        import re
        for pattern in prompt_patterns:
            if re.match(pattern, line):
                return True
        return False
    
    def _process_command_output(self, raw_output: str, original_command: str) -> str:
        """Process raw terminal output to extract just the command result"""
        if not raw_output:
            return ""
        
        # Remove any leftover markers first
        import re
        output = re.sub(r'__PROMPTCHAIN_END_[a-f0-9_]+__\d*', '', raw_output)
        
        # Split into lines and process
        lines = output.split('\n')
        result_lines = []
        skip_next_prompt = False
        
        for line in lines:
            # Clean terminal control sequences
            cleaned_line = self._clean_terminal_output(line)
            
            # Skip empty lines
            if not cleaned_line.strip():
                continue
            
            # Skip the command echo (exact match)
            if cleaned_line.strip() == original_command.strip():
                skip_next_prompt = True  # Skip the next prompt that usually follows
                continue
            
            # Skip command with semicolon and echo (our command format)
            if '; echo' in cleaned_line and original_command.split()[0] in cleaned_line:
                skip_next_prompt = True
                continue
            
            # Skip prompt lines
            if self._is_prompt_line(cleaned_line):
                if skip_next_prompt:
                    skip_next_prompt = False
                continue
            
            # Skip lines that are clearly command echoes with path resolution
            if (('/usr/bin/' in cleaned_line or '/bin/' in cleaned_line) and 
                len(cleaned_line.split()) > 2):  # Path + command + args
                continue
                
            # Skip lines that contain our command but are clearly echoes
            cmd_first_word = original_command.strip().split()[0]
            if (cmd_first_word in cleaned_line and 
                (cleaned_line.startswith(cmd_first_word) or cleaned_line.endswith(cmd_first_word))):
                # This looks like a command echo, skip it
                continue
            
            # This should be actual command output
            result_lines.append(cleaned_line)
        
        # Join and clean up the result
        result = '\n'.join(result_lines).strip()
        
        # Remove any remaining control sequences or artifacts
        result = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', result)  # ANSI sequences
        result = re.sub(r'\r+', '', result)  # Carriage returns
        
        return result
    
    def get_current_directory(self) -> str:
        """Get current working directory of the session"""
        try:
            output, _ = self.execute_command("pwd")
            return output.strip()
        except:
            return self.working_directory
    
    def get_environment_variable(self, var_name: str) -> Optional[str]:
        """Get environment variable from the session"""
        try:
            output, return_code = self.execute_command(f"echo ${var_name}")
            if return_code == 0:
                return output.strip() if output.strip() else None
        except:
            pass
        return None
    
    def set_environment_variable(self, var_name: str, value: str):
        """Set environment variable in the session"""
        try:
            self.execute_command(f"export {var_name}='{value}'")
            self.environment_variables[var_name] = value
        except Exception as e:
            raise RuntimeError(f"Failed to set environment variable: {e}")
    
    def change_directory(self, directory: str):
        """Change working directory in the session"""
        try:
            output, return_code = self.execute_command(f"cd '{directory}'")
            if return_code != 0:
                raise RuntimeError(f"Failed to change directory: {output}")
            self.working_directory = self.get_current_directory()
        except Exception as e:
            raise RuntimeError(f"Failed to change directory: {e}")
    
    def activate_nvm_version(self, version: str) -> bool:
        """Activate specific NVM version in the session"""
        try:
            # Source NVM and activate version
            commands = [
                "source ~/.nvm/nvm.sh",
                f"nvm use {version}"
            ]
            
            for cmd in commands:
                output, return_code = self.execute_command(cmd)
                if return_code != 0:
                    return False
            
            # Verify activation
            output, return_code = self.execute_command("node --version")
            if return_code == 0 and version in output:
                return True
                
        except:
            pass
        return False
    
    def activate_conda_environment(self, env_name: str) -> bool:
        """Activate conda environment in the session"""
        try:
            # Initialize conda and activate environment
            commands = [
                "eval \"$(conda shell.bash hook)\"",
                f"conda activate {env_name}"
            ]
            
            for cmd in commands:
                output, return_code = self.execute_command(cmd)
                if return_code != 0:
                    return False
            
            return True
        except:
            pass
        return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "working_directory": self.get_current_directory(),
            "environment_variables": self.environment_variables,
            "command_history_length": len(self.command_history),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "shell_ready": self.shell_ready,
            "process_alive": self.process.poll() is None if self.process else False
        }
    
    def close(self):
        """Close the persistent session"""
        self._cleanup()
    
    def _cleanup(self):
        """Cleanup session resources"""
        try:
            if self.process and self.process.poll() is None:
                # Gracefully terminate
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    
            if self.master_fd:
                os.close(self.master_fd)
                self.master_fd = None
                
            if self.slave_fd:
                os.close(self.slave_fd)
                self.slave_fd = None
                
        except Exception:
            pass
        
        self.shell_ready = False


class SessionManager:
    """Manages multiple named terminal sessions"""
    
    def __init__(self, max_sessions: int = 10):
        self.max_sessions = max_sessions
        self.sessions: Dict[str, PersistentTerminalSession] = {}
        self.session_names: Dict[str, str] = {}  # name -> session_id mapping
        self.current_session_id: Optional[str] = None
        
    def create_session(self, name: str, working_directory: str = None, 
                      environment_variables: Dict[str, str] = None) -> str:
        """
        Create a new named terminal session
        
        Args:
            name: Human-readable name for the session
            working_directory: Initial working directory
            environment_variables: Initial environment variables
            
        Returns:
            Session ID
        """
        # Check if name already exists
        if name in self.session_names:
            raise ValueError(f"Session with name '{name}' already exists")
        
        # Enforce session limit
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest_id = min(self.sessions.keys(), 
                          key=lambda sid: self.sessions[sid].last_accessed)
            self.close_session(oldest_id)
        
        # Create new session
        session_id = str(uuid.uuid4())
        session = PersistentTerminalSession(
            session_id=session_id,
            name=name,
            working_directory=working_directory,
            environment_variables=environment_variables
        )
        
        self.sessions[session_id] = session
        self.session_names[name] = session_id
        
        # Set as current if it's the first session
        if not self.current_session_id:
            self.current_session_id = session_id
            
        return session_id
    
    def get_session(self, identifier: str) -> Optional[PersistentTerminalSession]:
        """
        Get session by ID or name
        
        Args:
            identifier: Session ID or name
            
        Returns:
            Session instance or None
        """
        # Try as session ID first
        if identifier in self.sessions:
            return self.sessions[identifier]
        
        # Try as session name
        if identifier in self.session_names:
            session_id = self.session_names[identifier]
            return self.sessions.get(session_id)
        
        return None
    
    def switch_session(self, identifier: str) -> bool:
        """
        Switch to a different session
        
        Args:
            identifier: Session ID or name to switch to
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(identifier)
        if session:
            self.current_session_id = session.session_id
            return True
        return False
    
    def get_current_session(self) -> Optional[PersistentTerminalSession]:
        """Get the currently active session"""
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with their info"""
        return [
            {
                **session.get_session_info(),
                "is_current": session.session_id == self.current_session_id
            }
            for session in self.sessions.values()
        ]
    
    def close_session(self, identifier: str) -> bool:
        """
        Close and remove a session
        
        Args:
            identifier: Session ID or name
            
        Returns:
            True if closed, False if not found
        """
        session = self.get_session(identifier)
        if not session:
            return False
        
        # Close the session
        session.close()
        
        # Remove from tracking
        session_id = session.session_id
        session_name = session.name
        
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        if session_name in self.session_names:
            del self.session_names[session_name]
        
        # Update current session if needed
        if self.current_session_id == session_id:
            # Switch to another session if available
            if self.sessions:
                self.current_session_id = next(iter(self.sessions.keys()))
            else:
                self.current_session_id = None
        
        return True
    
    def close_all_sessions(self):
        """Close all sessions"""
        for session in list(self.sessions.values()):
            session.close()
        
        self.sessions.clear()
        self.session_names.clear()
        self.current_session_id = None
    
    def get_session_by_name(self, name: str) -> Optional[PersistentTerminalSession]:
        """Get session by name (convenience method)"""
        return self.get_session(name)
    
    def rename_session(self, old_identifier: str, new_name: str) -> bool:
        """
        Rename a session
        
        Args:
            old_identifier: Current session ID or name
            new_name: New name for the session
            
        Returns:
            True if successful, False otherwise
        """
        if new_name in self.session_names:
            return False  # Name already exists
        
        session = self.get_session(old_identifier)
        if not session:
            return False
        
        # Update name mappings
        old_name = session.name
        if old_name in self.session_names:
            del self.session_names[old_name]
        
        session.name = new_name
        self.session_names[new_name] = session.session_id
        
        return True