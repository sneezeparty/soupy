"""A stand-in for LM Studio's REST API, enough of it for the panel's Model tab.

``FakeLMStudio`` gives the tests an aiohttp application that lists models and
loads and unloads them in memory, with the request and response shapes from
LM Studio's ``/api/v1`` documentation. Run as a script it serves the same
thing on a port, for trying the panel by hand without touching a real
LM Studio:

    python tests/fake_lm_studio.py --port 15112 [--models listing.json] [--fail-load KEY]

``listing.json`` is a saved ``GET /api/v1/models`` response.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Iterable, List, Optional

from aiohttp import web

GB = 1024**3


def model(key, *, type="llm", size_gb=1.0, max_context=131072, vision=True, loaded_context=None, quant="Q4_K_M"):
    entry = {
        "type": type,
        "publisher": key.split("/")[0] if "/" in key else "",
        "key": key,
        "display_name": key.split("/")[-1],
        "quantization": {"name": quant, "bits_per_weight": 4},
        "size_bytes": int(size_gb * GB),
        "params_string": None,
        "loaded_instances": [],
        "max_context_length": max_context,
        "format": "gguf",
    }
    if type == "llm":
        entry["capabilities"] = {"vision": vision, "trained_for_tool_use": True}
    if loaded_context:
        entry["loaded_instances"].append({"id": key, "config": {"context_length": loaded_context}})
    return entry


# Shaped like the maintainer's machine: one model does chat and images, a
# small embedding model sits beside it, and bigger models wait on disk.
DEFAULT_MODELS = [
    model("google/gemma-4-12b", size_gb=7.0, loaded_context=28160),
    model("google/gemma-4-31b", size_gb=26.9, max_context=262144, quant="4bit"),
    model("llama-3.2-dark-champion", size_gb=6.6, vision=False, quant="Q2_K"),
    model("text-embedding-qwen3-embedding-0.6b", type="embedding", size_gb=0.6, max_context=32768, loaded_context=8192),
    model("text-embedding-nomic-embed-text-v1.5", type="embedding", size_gb=0.1, max_context=2048),
]


class FakeLMStudio:
    """The fake server and its state. ``app`` is the aiohttp application to serve.

    ``calls`` holds each request as ``(method, path, body)`` and ``auth`` its
    Authorization header, in the order they arrived. Pass ``timeline`` to have
    each request's last path segment appended to a list the caller shares with
    other recorders.
    """

    def __init__(
        self,
        models: Optional[List[dict]] = None,
        *,
        fail_load: Iterable[str] = (),
        fail_unload: Iterable[str] = (),
        token: str = "",
        load_seconds: float = 0.0,
        native: bool = True,
        timeline: Optional[list] = None,
    ) -> None:
        self.models = copy.deepcopy(DEFAULT_MODELS if models is None else models)
        self.calls: List[tuple] = []
        self.auth: List[str] = []
        self.fail_load, self.fail_unload = set(fail_load), set(fail_unload)
        self.token = token
        self.load_seconds = load_seconds
        self.timeline = timeline
        self.app = web.Application(middlewares=[self._record])
        if native:
            self.app.router.add_get("/api/v1/models", self._native_models)
            self.app.router.add_post("/api/v1/models/load", self._load)
            self.app.router.add_post("/api/v1/models/unload", self._unload)
        self.app.router.add_get("/v1/models", self._openai_models)

    def loaded(self) -> dict:
        """``{instance id: context length}`` for everything loaded right now."""
        return {i["id"]: i["config"]["context_length"] for m in self.models for i in m["loaded_instances"]}

    @staticmethod
    def _error(status: int, message: str) -> web.Response:
        return web.json_response({"error": {"message": message, "type": "invalid_request"}}, status=status)

    @web.middleware
    async def _record(self, request: web.Request, handler):
        body = await request.json() if request.can_read_body else None
        self.calls.append((request.method, request.path, body))
        self.auth.append(request.headers.get("Authorization", ""))
        if self.timeline is not None:
            self.timeline.append(request.path.rsplit("/", 1)[-1])
        if self.token and request.headers.get("Authorization") != f"Bearer {self.token}":
            return self._error(401, "Invalid API token")
        return await handler(request)

    def _find(self, key: str) -> Optional[dict]:
        return next((m for m in self.models if m["key"] == key), None)

    async def _native_models(self, _request: web.Request) -> web.Response:
        return web.json_response({"models": self.models})

    async def _openai_models(self, _request: web.Request) -> web.Response:
        loaded = [{"id": i["id"], "object": "model"} for m in self.models for i in m["loaded_instances"]]
        return web.json_response({"object": "list", "data": loaded})

    async def _load(self, request: web.Request) -> web.Response:
        body = await request.json()
        entry = self._find(str(body.get("model", "")))
        if entry is None:
            return self._error(404, f"Model {body.get('model')!r} not found")
        if self.load_seconds:
            await asyncio.sleep(self.load_seconds)
        if entry["key"] in self.fail_load:
            return self._error(500, "Failed to load model: insufficient system resources")
        context = int(body.get("context_length") or 4096)
        taken = {i["id"] for i in entry["loaded_instances"]}
        instance_id, n = entry["key"], 2
        while instance_id in taken:  # LM Studio numbers a second copy of a loaded model
            instance_id, n = f"{entry['key']}:{n}", n + 1
        entry["loaded_instances"].append({"id": instance_id, "config": {"context_length": context}})
        answer = {
            "type": entry["type"],
            "instance_id": instance_id,
            "load_time_seconds": self.load_seconds,
            "status": "loaded",
        }
        if body.get("echo_load_config"):
            answer["load_config"] = {"context_length": context, "flash_attention": True}
        return web.json_response(answer)

    async def _unload(self, request: web.Request) -> web.Response:
        body = await request.json()
        instance_id = str(body.get("instance_id", ""))
        if instance_id in self.fail_unload:
            return self._error(500, "Failed to unload model")
        for entry in self.models:
            for inst in entry["loaded_instances"]:
                if inst["id"] == instance_id:
                    entry["loaded_instances"].remove(inst)
                    return web.json_response({"instance_id": instance_id})
        return self._error(404, f"No loaded model instance {instance_id!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=15112)
    parser.add_argument("--models", type=Path, help="a saved GET /api/v1/models response")
    parser.add_argument("--fail-load", action="append", default=[], metavar="KEY")
    parser.add_argument("--load-seconds", type=float, default=2.0)
    parser.add_argument("--openai-only", action="store_true", help="serve /v1/models only, like a non-LM Studio server")
    args = parser.parse_args()
    listing = json.loads(args.models.read_text())["models"] if args.models else None
    server = FakeLMStudio(
        listing, fail_load=args.fail_load, load_seconds=args.load_seconds, native=not args.openai_only
    )
    web.run_app(server.app, host="127.0.0.1", port=args.port)
