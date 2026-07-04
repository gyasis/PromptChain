"""GovernorClient — atelier-governor memory-admission PLUGIN for PromptChain (issue #37).

DEV/EXPERIMENT branch — NOT merged to main. This is a CUSTOM plugin for the atelier governor (the
Mac Studio unified-memory hub), not a core PromptChain feature. The callers accept a GENERIC,
duck-typed `governor` hook — any object exposing `.lease(model, est_gb) -> lease` (a context manager
with `.release()`). GovernorClient is ONE implementation. `governor=None` => ungoverned, runs anywhere
(dev box, cloud host). So PromptChain stays portable and the governor is a plugin, not a dependency.

Why: MLXCaller/LlamaCppCaller load models IN-PROCESS on the shared Mac Studio; without reserving that
memory the governor could stack a TTS/ASR sidecar on top past the ~55 GB swap cliff. This does the
manual admit/release the governor documents for direct-backend callers.
"""
import json
import time
import uuid
import urllib.request


class _Lease:
    """Held memory reservation. `.release()` drops it; also a context manager."""

    def __init__(self, client, job_id, grant):
        self.client = client
        self.job_id = job_id
        self.grant = grant
        self.lease_id = grant.get("lease_id")
        self.reserved_gb = grant.get("reserved_gb")
        self.granted = bool(grant.get("grant"))

    def release(self):
        try:
            return self.client.release(job_id=self.job_id)
        except Exception:
            return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class GovernorClient:
    """Client for the atelier governor's manual memory gate (POST /admit · POST /release)."""

    def __init__(self, base_url="http://192.168.0.159:8799", timeout=15, poll_s=3.0, max_wait_s=600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s

    def _req(self, method, path, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def admit(self, model, est_gb, job_id=None, backend="auto", wait=True):
        """Reserve est_gb for `model`. Grants immediately, or queues under pressure and polls to grant
        (never evicts a running model). Returns the grant dict (with job_id); if it can't grant within
        max_wait_s, returns the last (ungranted) response — the caller decides whether to proceed."""
        job_id = job_id or f"pc-{uuid.uuid4().hex[:8]}"
        waited = 0.0
        while True:
            r = self._req("POST", "/admit", {"job_id": job_id, "model": model, "backend": backend, "est_gb": est_gb})
            r["job_id"] = job_id
            if r.get("grant") or not wait or waited >= self.max_wait_s:
                return r
            time.sleep(self.poll_s)
            waited += self.poll_s

    def release(self, job_id=None, lease_id=None):
        return self._req("POST", "/release", {"job_id": job_id} if job_id else {"lease_id": lease_id})

    def budget(self):
        return self._req("GET", "/budget")

    def pressure(self):
        return self._req("GET", "/pressure")

    def lease(self, model, est_gb, job_id=None, backend="auto"):
        """Admit now (waiting if queued) and return a _Lease to hold + release. The duck-typed hook
        the callers expect: `governor.lease(model, est_gb)` -> object with `.release()`."""
        grant = self.admit(model, est_gb, job_id=job_id, backend=backend)
        return _Lease(self, grant["job_id"], grant)
