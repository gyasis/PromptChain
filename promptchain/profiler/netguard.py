"""netguard — importing this module blocks every outbound socket except an allow-list.

For probe runs outside pytest: provider keys are dropped from the environment and only
loopback plus the Atelier Mac Studio (``PROBE_ALLOW_HOSTS``, default 192.168.0.159) may
be connected to. Import it before anything that could reach a provider.
"""
import os
import socket

ALLOWED = {"127.0.0.1", "localhost", "::1"} | set(
    h for h in os.environ.get("PROBE_ALLOW_HOSTS", "192.168.0.159").split(",") if h)
blocked: list = []

for _k in list(os.environ):
    if _k.endswith("_API_KEY") or _k.endswith("_TOKEN"):
        os.environ.pop(_k)

_orig_connect = socket.socket.connect
_orig_getaddrinfo = socket.getaddrinfo


def _connect(self, addr, *a, **k):
    if isinstance(addr, tuple) and addr and addr[0] not in ALLOWED:
        blocked.append(addr)
        raise RuntimeError(f"NETWORK BLOCKED {addr}")
    return _orig_connect(self, addr, *a, **k)


def _getaddrinfo(host, *a, **k):
    if isinstance(host, str) and host not in ALLOWED and not host.replace(".", "").isdigit():
        blocked.append((host, "dns"))
        raise RuntimeError(f"NETWORK BLOCKED dns {host}")
    return _orig_getaddrinfo(host, *a, **k)


socket.socket.connect = _connect
socket.getaddrinfo = _getaddrinfo
