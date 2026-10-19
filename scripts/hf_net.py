#!/usr/bin/env python
"""Shared HTTP layer for peek_checkpoint.py / fetch_minimal.py.

Why this exists: the bare `urllib.request.urlopen` traceback is useless when the failure is
environmental. In practice the three things that actually break these scripts are

  1. a corporate / VPN proxy that answers `502 Bad Gateway` for CONNECT tunneling
     -> `Tunnel connection failed: 502 Bad Gateway` (looks like "the site is down")
  2. huggingface.co being unreachable (mainland China) while `HF_ENDPOINT` is unset
  3. a transient 5xx that would succeed on retry

So: retry transient errors, let `--no-proxy` bypass the proxy, and turn any leftover failure
into a message that says which of the three it is and what to type next.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request

USER_AGENT = "hf-model-standalone/1.0"


class FetchError(RuntimeError):
    """Network failure carrying a human-actionable explanation."""

    def __init__(self, url: str, reason: str, hint: str):
        super().__init__(f"{reason}\n  url: {url}\n{hint}")
        self.url = url
        self.reason = reason


def _proxy_env():
    return {k: os.environ[k] for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy") if os.environ.get(k)}


def _opener(use_proxy: bool):
    if use_proxy:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _hint(url: str, reason: str, endpoint: str | None) -> str:
    lines = []
    low = reason.lower()
    if "tunnel connection failed" in low or "proxy" in low or "502" in low:
        env = _proxy_env()
        if env:
            shown = ", ".join(f"{k}={v}" for k, v in env.items())
            lines.append(f"  your shell has a proxy set ({shown}) and it is refusing the tunnel.")
            lines.append("  try:  --no-proxy      (bypass it)")
    if endpoint and "huggingface.co" in endpoint:
        lines.append("  huggingface.co is unreachable from many networks; mirror it:")
        lines.append("        --endpoint https://hf-mirror.com")
    lines.append("  offline alternative: pass a local path (a dir with config.json / *.safetensors)")
    return "\n".join(lines)


def http_get(url, headers=None, timeout=60, retries=2, use_proxy=True, endpoint=None):
    """GET with retry on transient failures; raises FetchError with a fix-it hint."""
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and "huggingface.co" in url:
        hdrs["Authorization"] = f"Bearer {token}"

    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with _opener(use_proxy).open(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # 404/401/403 will never succeed on retry: fail fast
            if e.code < 500 or attempt == retries:
                raise FetchError(url, f"HTTP {e.code} {e.reason}", _hint(url, str(e), endpoint)) from e
            last = e
        except urllib.error.URLError as e:
            reason = str(getattr(e, "reason", e))
            last = e
            if attempt == retries:
                raise FetchError(url, f"URLError: {reason}", _hint(url, reason, endpoint)) from e
        time.sleep(1.5 * (attempt + 1))
    raise FetchError(url, str(last), _hint(url, str(last), endpoint))


def add_net_args(ap, default_endpoint: str):
    ap.add_argument(
        "--endpoint",
        default=default_endpoint,
        help=f"hub endpoint (default: {default_endpoint}; use https://hf-mirror.com if huggingface.co is blocked)",
    )
    ap.add_argument(
        "--no-proxy",
        action="store_true",
        help="ignore HTTPS_PROXY/HTTP_PROXY (fixes 'Tunnel connection failed: 502 Bad Gateway')",
    )
