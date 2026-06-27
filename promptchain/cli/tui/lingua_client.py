"""Client for the persistent local LLMLingua-2 compression worker.

Spawns ``~/.local/bin/lingua-worker`` (which runs in the isolated uv venv and
keeps the BERT model warm), serializes requests over JSON-lines, and degrades
GRACEFULLY to a no-op when the worker isn't available — so promptchain never
imports torch/llmlingua and a missing/broken venv just disables compression.

Blocking: ``compress()`` blocks on the worker. The first call also blocks while
the model loads — callers should pre-warm (``ensure()``) off the UI thread and
run ``compress()`` off the UI thread too.
"""
import json
import os
import subprocess
import threading
from typing import Dict, Optional

WORKER = os.path.expanduser("~/.local/bin/lingua-worker")


class LinguaCompressor:
    def __init__(self, worker_path: str = WORKER):
        self._worker_path = worker_path
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._ready = False
        self._dead = False  # failed to start — don't keep retrying

    def _ensure(self) -> bool:
        if self._ready:
            return True
        if self._dead or not os.path.exists(self._worker_path):
            self._dead = True
            return False
        try:
            self._proc = subprocess.Popen(
                [self._worker_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            line = self._proc.stdout.readline()  # blocks until model loads
            info = json.loads(line) if line else {}
            if info.get("ready"):
                self._ready = True
                return True
        except Exception:
            pass
        self._dead = True
        return False

    def ensure(self) -> bool:
        """Pre-warm the worker (load the model). Call off the UI thread."""
        with self._lock:
            return self._ensure()

    def available(self) -> bool:
        return self._ready or (not self._dead and os.path.exists(self._worker_path))

    def compress(
        self, text: str, rate: float = 0.5, target_token: Optional[int] = None
    ) -> Optional[Dict]:
        """Compress ``text``. Returns
        ``{"compressed","origin_tokens","compressed_tokens","ratio"}`` or None
        (caller falls back to the original text). Blocking — run off the UI thread.
        """
        if not text or not text.strip():
            return None
        with self._lock:
            if not self._ensure() or self._proc is None:
                return None
            try:
                req: Dict = {"text": text}
                if target_token:
                    req["target_token"] = int(target_token)
                else:
                    req["rate"] = float(rate)
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if not line:
                    self._ready = False
                    self._dead = True
                    return None
                resp = json.loads(line)
                return None if resp.get("error") else resp
            except Exception:
                self._ready = False
                self._dead = True
                return None

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
            self._ready = False
