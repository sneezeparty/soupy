"""Tiny stdlib HTTP wrapper.

Exists so validators have one place to mock and so the installer can run
on a fresh clone without `requests` installed.
"""

from __future__ import annotations

import json as _json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


class HttpError(Exception):
    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status
        self.body = body
        self.url = url


def get_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Tuple[int, Any]:
    """GET a URL and parse the response as JSON. Returns (status, parsed_body).

    Non-2xx responses still return (status, body) — callers decide whether
    to treat them as failures. Network-level errors raise `HttpError`.
    """
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.getcode() or 0
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        status = e.code
    except urllib.error.URLError as e:
        raise HttpError(0, str(e.reason), url) from e
    except (TimeoutError, OSError) as e:
        raise HttpError(0, str(e), url) from e

    try:
        body = _json.loads(raw) if raw else None
    except _json.JSONDecodeError:
        body = raw
    return status, body


def get_text(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Tuple[int, str]:
    """GET a URL and return the raw body text. Network errors raise HttpError."""
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode() or 0, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, raw
    except urllib.error.URLError as e:
        raise HttpError(0, str(e.reason), url) from e
    except (TimeoutError, OSError) as e:
        raise HttpError(0, str(e), url) from e
