"""Native PromptChain Docker code-execution component.

A lean, dependency-free (subprocess around the ``docker`` CLI) executor for running
LLM-generated code in a sandboxed container. Adopts the AG2/AutoGen
``DockerCommandLineCodeExecutor`` *principles* — explicit lifecycle, single work_dir
mount, per-run timeout, explicit image policy, language-aware execution, structured
result, container-only — WITHOUT taking the ``autogen`` dependency, and adds
first-class hardening flags AG2 doesn't expose (``--network none``, mem/cpu/pids
limits, ``--read-only`` + tmpfs, non-root ``--user``).

Usable as a PromptChain ``Callable`` step or a registered tool. See PromptChain
issue #6 and the PRD ``promptchain_native_docker_executor_2026-06-19``.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    """Structured execution result (principle: caller decides pass/fail)."""
    exit_code: int
    output: str           # combined stdout + stderr
    timed_out: bool = False

    def __bool__(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


# language -> (file extension, interpreter argv)
_LANG_RUN = {
    "python": ("py", ["python"]), "py": ("py", ["python"]),
    "bash": ("sh", ["bash"]), "sh": ("sh", ["sh"]), "shell": ("sh", ["sh"]),
}
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.S)


def extract_code_blocks(text: str):
    """Return ``[(lang, code)]`` from markdown fenced blocks (lang defaults to 'python')."""
    return [((lang or "python").lower(), code.strip()) for lang, code in _FENCE.findall(text or "")]


class DockerExecutor:
    """Run code in a sandboxed Docker container via the ``docker`` CLI (no docker-py dep).

    Per-run ephemeral container (``docker run --rm``); the work_dir is the single bind
    mount. Use as a context manager to auto-clean a temp work_dir::

        with DockerExecutor(image="python:3-slim", network="none", timeout=60) as ex:
            res = ex.run_code_blocks(llm_output)   # -> ExecResult(exit_code, output)
    """

    def __init__(self, image: str = "python:3-slim", work_dir: str | None = None,
                 timeout: int = 120, network: str | None = "none",
                 memory: str | None = None, cpus: float | None = None,
                 pids_limit: int | None = None, read_only: bool = False,
                 user: str | None = None, auto_remove: bool = True,
                 extra_run_args: list | None = None, docker_bin: str = "docker"):
        self.image = image
        self.timeout = timeout
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.read_only = read_only
        self.user = user
        self.auto_remove = auto_remove
        self.extra_run_args = list(extra_run_args or [])   # AG2's container_create_kwargs analog
        self.docker_bin = docker_bin
        self._tmp = None
        if work_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="pc_docker_")
            work_dir = self._tmp.name
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

    # --- lifecycle (parity with AG2's start/stop) ---
    def __enter__(self) -> "DockerExecutor":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    @staticmethod
    def available(docker_bin: str = "docker") -> bool:
        """True iff a docker daemon is reachable (so callers can fall back gracefully)."""
        try:
            return subprocess.run([docker_bin, "info"], capture_output=True, timeout=10).returncode == 0
        except Exception:
            return False

    # --- the sandboxed run ---
    def _run_argv(self) -> list:
        argv = [self.docker_bin, "run", "-i"]
        if self.auto_remove:
            argv.append("--rm")
        argv += ["-v", f"{self.work_dir}:/work", "-w", "/work"]
        if self.network is not None:
            argv += ["--network", self.network]
        if self.memory:
            argv += ["--memory", str(self.memory)]
        if self.cpus:
            argv += ["--cpus", str(self.cpus)]
        if self.pids_limit:
            argv += ["--pids-limit", str(self.pids_limit)]
        if self.read_only:
            argv += ["--read-only", "--tmpfs", "/tmp"]
        if self.user:
            argv += ["--user", str(self.user)]
        argv += self.extra_run_args
        return argv

    def run(self, command: str) -> ExecResult:
        """Run a shell ``command`` string inside the sandbox; return an ExecResult."""
        argv = self._run_argv() + [self.image, "sh", "-c", command]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
            return ExecResult(p.returncode, (p.stdout or "") + (p.stderr or ""))
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "") + (e.stderr or "") if (e.stdout or e.stderr) else ""
            return ExecResult(124, partial + f"\n[timeout after {self.timeout}s]", timed_out=True)

    def write_file(self, relpath: str, content: str) -> str:
        """Write ``content`` to ``relpath`` inside the work_dir (creating parent dirs).

        Lets a caller stage a target file / tests / fixtures before ``run``. Returns
        the absolute path written. ``relpath`` is confined to the work_dir.
        """
        dest = os.path.abspath(os.path.join(self.work_dir, relpath))
        if not dest.startswith(self.work_dir + os.sep) and dest != self.work_dir:
            raise ValueError(f"relpath escapes work_dir: {relpath!r}")
        os.makedirs(os.path.dirname(dest) or self.work_dir, exist_ok=True)
        with open(dest, "w") as f:
            f.write(content)
        return dest

    def run_code_blocks(self, blocks) -> ExecResult:
        """Write code blocks to the work_dir and run them in order, stopping at the first
        non-zero exit. ``blocks`` is ``[(lang, code)]`` or raw markdown text."""
        if isinstance(blocks, str):
            blocks = extract_code_blocks(blocks)
        ran = 0
        last = ExecResult(0, "(no code blocks)")
        for i, (lang, code) in enumerate(blocks):
            if lang not in _LANG_RUN:
                continue   # skip non-runnable blocks (dockerfile, text, json, yaml, ...)
            ext, interp = _LANG_RUN[lang]
            fname = f"block_{i}.{ext}"
            with open(os.path.join(self.work_dir, fname), "w") as f:
                f.write(code)
            cmd = " ".join(shlex.quote(x) for x in interp + [fname])
            last = self.run(cmd)
            ran += 1
            if last.exit_code != 0:
                break
        if ran == 0:
            return ExecResult(127, "no runnable code block (only non-runnable languages found)")
        return last
