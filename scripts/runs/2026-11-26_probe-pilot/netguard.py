"""netguard — stdlib-only network allow-list. Import it FIRST, before any promptchain import.

``promptchain/__init__`` imports litellm, which can fetch its cost map from the internet, so
the guard has to be installed first and lives outside the package. Only the (host, port)
pairs in ALLOWED may be connected to; loopback is NOT allowed (127.0.0.1:37400 is the
LiteLLM gateway that proxies paid providers, and raw ollama :11434 is forbidden).
Hostname lookups are blocked unless the name is in ALLOWED.
"""
import os
import socket

ALLOWED = {("192.168.0.159", 8799)}
_HOSTS = {h for h, _ in ALLOWED}
blocked: list = []

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LITELLM_TELEMETRY="False", DO_NOT_TRACK="1")
_SECRET_SUFFIXES = ("_API_KEY", "_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_ACCESS_KEY_ID", "_SECRET_ACCESS_KEY")
_SECRET_PREFIXES = ("AWS_", "AZURE_", "GOOGLE_APPLICATION", "OPENAI", "ANTHROPIC", "GEMINI", "MINIMAX", "KIMI",
                    "MOONSHOT", "OLLAMA_API", "HF_", "GH_", "GITHUB_")
for _k in list(os.environ):
    if _k.upper().endswith(_SECRET_SUFFIXES) or _k.upper().startswith(_SECRET_PREFIXES):
        if _k not in ("HF_HUB_OFFLINE",):
            os.environ.pop(_k)


def _check(addr):
    if isinstance(addr, tuple) and len(addr) >= 2 and (addr[0], addr[1]) not in ALLOWED:
        blocked.append(tuple(addr[:2]))
        raise RuntimeError(f"NETWORK BLOCKED {addr[:2]}")


_connect, _connect_ex, _getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo


def _guard_connect(self, addr, *a, **k):
    _check(addr)
    return _connect(self, addr, *a, **k)


def _guard_connect_ex(self, addr, *a, **k):
    _check(addr)
    return _connect_ex(self, addr, *a, **k)


def _guard_getaddrinfo(host, *a, **k):
    if isinstance(host, (str, bytes)):
        h = host.decode() if isinstance(host, bytes) else host
        if h not in _HOSTS and not h.replace(".", "").isdigit():
            blocked.append((h, "dns"))
            raise RuntimeError(f"NETWORK BLOCKED dns {h}")
    return _getaddrinfo(host, *a, **k)


socket.socket.connect = _guard_connect
socket.socket.connect_ex = _guard_connect_ex
socket.getaddrinfo = _guard_getaddrinfo
