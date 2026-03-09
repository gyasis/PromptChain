"""
Environment Management Module for Terminal Tool

Provides intelligent detection and activation of development environments:
- Conda/Miniconda environments
- Python virtual environments (venv, virtualenv)
- Node Version Manager (NVM)
- Custom environment managers

Author: PromptChain Team
License: MIT
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EnvironmentManager:
    """Manages development environment detection and activation"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize environment manager

        Args:
            config: Configuration dictionary with environment settings
        """
        self.config = config or {}
        self.detected_envs: Dict[str, Any] = {}
        self.active_env: Optional[Tuple[str, Optional[str]]] = None
        self.env_vars_backup = os.environ.copy()

        # Configuration options
        self.auto_activate = self.config.get("auto_activate_env", False)
        self.prefer_env = self.config.get(
            "prefer_environment", None
        )  # 'conda', 'venv', 'nvm'
        self.create_if_missing = self.config.get("create_if_missing", False)
        self.custom_paths = self.config.get("custom_environment_paths", {})

    def detect_environments(self) -> Dict[str, Any]:
        """
        Detect available development environments

        Returns:
            Dictionary of detected environments with their configurations
        """
        envs = {}

        # Check for Conda environments
        conda_envs = self._detect_conda()
        if conda_envs:
            envs["conda"] = conda_envs

        # Check for Python virtual environments
        venv_info = self._detect_python_venv()
        if venv_info:
            envs["python_venv"] = venv_info

        # Check for Node Version Manager
        nvm_info = self._detect_nvm()
        if nvm_info:
            envs["nvm"] = nvm_info

        # Check for active virtualenv
        virtualenv_info = self._detect_active_virtualenv()
        if virtualenv_info:
            envs["active_virtualenv"] = virtualenv_info

        # Check for Poetry environments
        poetry_info = self._detect_poetry()
        if poetry_info:
            envs["poetry"] = poetry_info

        # Check for Pipenv environments
        pipenv_info = self._detect_pipenv()
        if pipenv_info:
            envs["pipenv"] = pipenv_info

        # Check custom environment paths
        custom_envs = self._detect_custom_environments()
        if custom_envs:
            envs.update(custom_envs)

        self.detected_envs = envs
        return envs

    def _detect_conda(self) -> Optional[Dict[str, Any]]:
        """Detect Conda/Miniconda environments"""
        # Check if conda is available
        conda_exec = shutil.which("conda")
        if not conda_exec:
            # Check common conda locations
            common_paths = [
                Path.home() / "anaconda3" / "bin" / "conda",
                Path.home() / "miniconda3" / "bin" / "conda",
                Path("/opt/anaconda3/bin/conda"),
                Path("/opt/miniconda3/bin/conda"),
                Path("/usr/local/anaconda3/bin/conda"),
                Path("/usr/local/miniconda3/bin/conda"),
            ]

            for path in common_paths:
                if path.exists():
                    conda_exec = str(path)
                    break
            else:
                return None

        try:
            # Get conda info
            result = subprocess.run(
                [conda_exec, "info", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                conda_info = json.loads(result.stdout)

                # Get list of environments
                env_result = subprocess.run(
                    [conda_exec, "env", "list", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                envs_info = {}
                if env_result.returncode == 0:
                    env_data = json.loads(env_result.stdout)
                    for env_path in env_data.get("envs", []):
                        env_name = Path(env_path).name
                        envs_info[env_name] = env_path

                return {
                    "type": "conda",
                    "executable": conda_exec,
                    "version": conda_info.get("conda_version", "unknown"),
                    "default_env": conda_info.get("default_env", "base"),
                    "active_env": conda_info.get("active_env", None),
                    "environments": envs_info,
                    "conda_info": conda_info,
                }

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass

        return None

    def _detect_python_venv(self) -> Optional[Dict[str, Any]]:
        """Detect Python virtual environments"""
        venv_indicators = ["venv", ".venv", "env", ".env", "virtualenv"]

        detected_venvs = []

        # Check current directory and parent directories
        current_path = Path.cwd()
        for _ in range(3):  # Check up to 3 parent directories
            for venv_name in venv_indicators:
                venv_path = current_path / venv_name
                if self._is_valid_venv(venv_path):
                    detected_venvs.append(
                        {
                            "name": venv_name,
                            "path": str(venv_path),
                            "python_executable": self._get_venv_python(venv_path),
                            "active": False,
                        }
                    )
            current_path = current_path.parent
            if current_path == current_path.parent:  # Reached root
                break

        if detected_venvs:
            return {"type": "python_venv", "environments": detected_venvs}

        return None

    def _is_valid_venv(self, path: Path) -> bool:
        """Check if a path contains a valid Python virtual environment"""
        if not path.exists() or not path.is_dir():
            return False

        # Check for venv indicators
        indicators = [
            path / "pyvenv.cfg",
            path / "bin" / "activate",  # Unix
            path / "Scripts" / "activate.bat",  # Windows
            path / "bin" / "python",  # Unix
            path / "Scripts" / "python.exe",  # Windows
        ]

        return any(indicator.exists() for indicator in indicators)

    def _get_venv_python(self, venv_path: Path) -> Optional[str]:
        """Get the Python executable path for a virtual environment"""
        # Unix-like systems
        unix_python = venv_path / "bin" / "python"
        if unix_python.exists():
            return str(unix_python)

        # Windows
        windows_python = venv_path / "Scripts" / "python.exe"
        if windows_python.exists():
            return str(windows_python)

        return None

    def _detect_nvm(self) -> Optional[Dict[str, Any]]:
        """Detect Node Version Manager (NVM)"""
        # Check NVM directory
        nvm_dir = Path.home() / ".nvm"
        if not nvm_dir.exists():
            # Check NVM_DIR environment variable
            nvm_dir_env = os.environ.get("NVM_DIR")
            if nvm_dir_env:
                nvm_dir = Path(nvm_dir_env)

        if not nvm_dir.exists():
            return None

        # Get NVM script path
        nvm_script = nvm_dir / "nvm.sh"
        if not nvm_script.exists():
            return None

        # Get installed Node versions
        versions_dir = nvm_dir / "versions" / "node"
        versions = []
        if versions_dir.exists():
            versions = [d.name for d in versions_dir.iterdir() if d.is_dir()]

        # Try to get current version
        current_version = None
        try:
            # Check if node is available and get version
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                current_version = result.stdout.strip()
        except:
            pass

        return {
            "type": "nvm",
            "nvm_dir": str(nvm_dir),
            "script_path": str(nvm_script),
            "installed_versions": versions,
            "current_version": current_version,
            "has_nvmrc": (Path.cwd() / ".nvmrc").exists(),
        }

    def _detect_active_virtualenv(self) -> Optional[Dict[str, Any]]:
        """Detect currently active virtualenv/venv"""
        virtual_env = os.environ.get("VIRTUAL_ENV")
        if virtual_env:
            return {
                "type": "active_virtualenv",
                "path": virtual_env,
                "name": Path(virtual_env).name,
                "python_executable": (os.environ.get("VIRTUAL_ENV") or "")
                + "/bin/python",
                "activated": True,
            }
        return None

    def _detect_poetry(self) -> Optional[Dict[str, Any]]:
        """Detect Poetry environment"""
        # Check if poetry is available
        if not shutil.which("poetry"):
            return None

        # Check for pyproject.toml
        pyproject_path = Path.cwd() / "pyproject.toml"
        if not pyproject_path.exists():
            return None

        try:
            # Get poetry env info
            result = subprocess.run(
                ["poetry", "env", "info", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=Path.cwd(),
            )

            if result.returncode == 0:
                poetry_info = json.loads(result.stdout)
                return {
                    "type": "poetry",
                    "project_dir": str(Path.cwd()),
                    "pyproject_path": str(pyproject_path),
                    "venv_path": poetry_info.get("path"),
                    "python_executable": poetry_info.get("executable"),
                    "is_venv": poetry_info.get("is_venv", False),
                }

        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        return None

    def _detect_pipenv(self) -> Optional[Dict[str, Any]]:
        """Detect Pipenv environment"""
        # Check if pipenv is available
        if not shutil.which("pipenv"):
            return None

        # Check for Pipfile
        pipfile_path = Path.cwd() / "Pipfile"
        if not pipfile_path.exists():
            return None

        try:
            # Get pipenv venv location
            result = subprocess.run(
                ["pipenv", "--venv"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=Path.cwd(),
            )

            if result.returncode == 0:
                venv_path = result.stdout.strip()
                return {
                    "type": "pipenv",
                    "project_dir": str(Path.cwd()),
                    "pipfile_path": str(pipfile_path),
                    "venv_path": venv_path,
                    "python_executable": str(Path(venv_path) / "bin" / "python"),
                }

        except subprocess.TimeoutExpired:
            pass

        return None

    def _detect_custom_environments(self) -> Dict[str, Any]:
        """Detect custom environments from configuration"""
        custom_envs = {}

        for env_name, env_config in self.custom_paths.items():
            env_path = Path(env_config.get("path", ""))
            if env_path.exists():
                custom_envs[f"custom_{env_name}"] = {
                    "type": "custom",
                    "name": env_name,
                    "path": str(env_path),
                    "activation_command": env_config.get("activation_command"),
                    "executable": env_config.get("executable"),
                }

        return custom_envs

    def activate_environment(
        self, env_type: str, env_name: Optional[str] = None
    ) -> bool:
        """
        Activate a development environment

        Args:
            env_type: Type of environment ('conda', 'venv', 'nvm', etc.)
            env_name: Name/identifier of the specific environment

        Returns:
            True if activation successful, False otherwise
        """
        try:
            if env_type == "conda":
                return self._activate_conda(env_name)
            elif env_type == "python_venv":
                return self._activate_python_venv(env_name)
            elif env_type == "nvm":
                return self._activate_nvm(env_name)
            elif env_type == "poetry":
                return self._activate_poetry()
            elif env_type == "pipenv":
                return self._activate_pipenv()
            elif env_type.startswith("custom_"):
                return self._activate_custom(env_type, env_name)

        except Exception as e:
            print(f"Failed to activate environment {env_type}: {e}")

        return False

    def _activate_conda(self, env_name: Optional[str] = None) -> bool:
        """Activate conda environment"""
        conda_info = self.detected_envs.get("conda")
        if not conda_info:
            return False

        env_name = env_name or conda_info.get("default_env", "base")

        # Get environment path
        env_path = conda_info["environments"].get(env_name)
        if not env_path:
            return False

        # Modify environment variables to activate conda env
        bin_dir = Path(env_path) / "bin"
        if bin_dir.exists():
            # Update PATH
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}:{current_path}"

            # Set conda environment variables
            os.environ["CONDA_DEFAULT_ENV"] = env_name
            os.environ["CONDA_PREFIX"] = env_path

            self.active_env = ("conda", env_name)
            return True

        return False

    def _activate_python_venv(self, venv_name: Optional[str] = None) -> bool:
        """Activate Python virtual environment"""
        venv_info = self.detected_envs.get("python_venv")
        if not venv_info:
            return False

        # Find the venv to activate
        target_venv = None
        if venv_name:
            for venv in venv_info["environments"]:
                if venv["name"] == venv_name:
                    target_venv = venv
                    break
        else:
            # Use first available venv
            target_venv = (
                venv_info["environments"][0] if venv_info["environments"] else None
            )

        if not target_venv:
            return False

        venv_path = Path(target_venv["path"])

        # Update PATH to include venv bin directory
        bin_dir = venv_path / "bin"
        scripts_dir = venv_path / "Scripts"  # Windows

        if bin_dir.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}:{current_path}"
            os.environ["VIRTUAL_ENV"] = str(venv_path)
        elif scripts_dir.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{scripts_dir};{current_path}"
            os.environ["VIRTUAL_ENV"] = str(venv_path)
        else:
            return False

        self.active_env = ("python_venv", target_venv["name"])
        return True

    def _activate_nvm(self, version: Optional[str] = None) -> bool:
        """Activate Node Version Manager version"""
        nvm_info = self.detected_envs.get("nvm")
        if not nvm_info:
            return False

        # If no version specified, check for .nvmrc or use latest
        if not version:
            nvmrc_path = Path.cwd() / ".nvmrc"
            if nvmrc_path.exists():
                version = nvmrc_path.read_text().strip()
            elif nvm_info["installed_versions"]:
                # Use latest version (assuming semver-like ordering)
                version = sorted(nvm_info["installed_versions"])[-1]
            else:
                return False

        # Check if version is installed
        if version not in nvm_info["installed_versions"]:
            return False

        # Update PATH to use the specific Node version
        node_bin = Path(nvm_info["nvm_dir"]) / "versions" / "node" / version / "bin"
        if node_bin.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{node_bin}:{current_path}"

            self.active_env = ("nvm", version)
            return True

        return False

    def _activate_poetry(self) -> bool:
        """Activate Poetry environment"""
        poetry_info = self.detected_envs.get("poetry")
        if not poetry_info or not poetry_info.get("venv_path"):
            return False

        venv_path = Path(poetry_info["venv_path"])
        bin_dir = venv_path / "bin"

        if bin_dir.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}:{current_path}"
            os.environ["VIRTUAL_ENV"] = str(venv_path)

            self.active_env = ("poetry", str(venv_path))
            return True

        return False

    def _activate_pipenv(self) -> bool:
        """Activate Pipenv environment"""
        pipenv_info = self.detected_envs.get("pipenv")
        if not pipenv_info or not pipenv_info.get("venv_path"):
            return False

        venv_path = Path(pipenv_info["venv_path"])
        bin_dir = venv_path / "bin"

        if bin_dir.exists():
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}:{current_path}"
            os.environ["VIRTUAL_ENV"] = str(venv_path)

            self.active_env = ("pipenv", str(venv_path))
            return True

        return False

    def _activate_custom(self, env_type: str, env_name: Optional[str] = None) -> bool:
        """Activate custom environment"""
        env_info = self.detected_envs.get(env_type)
        if not env_info:
            return False

        # Execute custom activation command if provided
        activation_cmd = env_info.get("activation_command")
        if activation_cmd:
            try:
                subprocess.run(activation_cmd, shell=True, check=True)
                self.active_env = (env_type, env_name)
                return True
            except subprocess.CalledProcessError:
                return False

        return False

    def auto_detect_and_activate(self) -> Optional[Tuple[str, Optional[str]]]:
        """
        Automatically detect and activate the best available environment

        Returns:
            Tuple of (env_type, env_name) if successful, None otherwise
        """
        # First detect all environments
        self.detect_environments()

        # Priority order based on configuration or defaults
        priority_order = []
        if self.prefer_env:
            priority_order.append(self.prefer_env)

        # Default priority: conda > poetry > pipenv > python_venv > nvm
        default_priority = ["conda", "poetry", "pipenv", "python_venv", "nvm"]
        for env_type in default_priority:
            if env_type not in priority_order:
                priority_order.append(env_type)

        # Try to activate environments in priority order
        for env_type in priority_order:
            if env_type in self.detected_envs:
                if self.activate_environment(env_type):
                    return self.active_env

        return None

    def restore_environment(self):
        """Restore original environment variables"""
        os.environ.clear()
        os.environ.update(self.env_vars_backup)
        self.active_env = None

    def get_active_environment(self) -> Optional[Tuple[str, Optional[str]]]:
        """Get currently active environment"""
        return self.active_env

    def get_environment_info(self, env_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific environment type"""
        return self.detected_envs.get(env_type)

    def list_available_environments(self) -> List[str]:
        """Get list of all available environment types"""
        return list(self.detected_envs.keys())
