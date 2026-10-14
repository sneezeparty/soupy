"""LM Studio's model list and its load and unload calls, for the dashboard's Model tab.

LM Studio's own REST API (``/api/v1``) says which models are downloaded, which
are loaded and with what context length, and loads or unloads one on request.
Any other OpenAI-compatible server only answers ``/v1/models``. For those the
panel saves the model name and leaves loading to the server.

A switch is planned first (:func:`build_plan`, no network) and then carried out
(:func:`apply_plan`). The dashboard shows the plan in its confirm dialog, so
what the dialog says is what happens.

Kept apart from ``web.app`` so the tests can use it without starting the panel.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

import aiohttp

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_CHAT_BUDGET = 16000
# The bot's own floor for CONTEXT_WINDOW_TOKENS (soupy/settings.py).
MIN_CHAT_BUDGET = 512
MIN_CONTEXT_LENGTH = 512
# Chat prompts are sized from character counts, so a budget right at the loaded
# context can overflow it. The profile builder leaves the same margin
# (USER_PROFILE_CONTEXT_SAFETY).
HEADROOM = 0.85

LIST_TIMEOUT = 10
UNLOAD_TIMEOUT = 60
# A large model read from a cold disk takes minutes.
LOAD_TIMEOUT = 600

_TRUE = ("1", "true", "yes", "y", "on")


class LMStudioError(Exception):
    """The server could not be reached or refused a request."""


class PlanError(Exception):
    """The requested switch can't be done as asked. The message is shown to the user."""


@dataclass
class Settings:
    """What ``.env-stable`` says about models."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    chat: str = ""
    vision_model: str = ""  # VISION_MODEL as written; empty means "same as chat"
    vision_enabled: bool = False
    embedding: str = ""
    chat_budget: int = DEFAULT_CHAT_BUDGET
    saved_list: List[str] = field(default_factory=list)

    @property
    def vision(self) -> str:
        """The model that describes images, or "" when vision is off."""
        if not self.vision_enabled:
            return ""
        return self.vision_model or self.chat


def _int(raw: object, default: int) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def settings_from_env(kv: Mapping[str, str]) -> Settings:
    saved = [m.strip() for m in re.split(r"[\n,]", kv.get("AVAILABLE_MODELS", "")) if m.strip()]
    return Settings(
        base_url=(kv.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip(),
        api_key=(kv.get("OPENAI_API_KEY") or kv.get("LOCAL_KEY") or "").strip(),
        chat=kv.get("LOCAL_CHAT", "").strip(),
        vision_model=kv.get("VISION_MODEL", "").strip(),
        vision_enabled=kv.get("ENABLE_VISION", "").strip().lower() in _TRUE,
        embedding=kv.get("RAG_EMBEDDING_MODEL", "").strip(),
        chat_budget=_int(kv.get("CONTEXT_WINDOW_TOKENS", ""), DEFAULT_CHAT_BUDGET),
        saved_list=saved,
    )


def api_root(base_url: str) -> str:
    """``http://host:1234/v1`` as ``http://host:1234``."""
    return re.sub(r"/v1$", "", (base_url or DEFAULT_BASE_URL).rstrip("/"))


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def parse_native(data: Mapping) -> List[dict]:
    """``GET /api/v1/models`` as the rows the dashboard uses."""
    rows: List[dict] = []
    for m in data.get("models") or data.get("data") or []:
        key = str(m.get("key") or m.get("id") or "")
        if not key:
            continue
        caps = m.get("capabilities") or {}
        instances = [
            {"id": str(i.get("id") or key), "context_length": (i.get("config") or {}).get("context_length")}
            for i in (m.get("loaded_instances") or [])
        ]
        rows.append(
            {
                "key": key,
                "type": m.get("type") or "llm",
                "name": m.get("display_name") or "",
                "quant": (m.get("quantization") or {}).get("name") or "",
                "size_bytes": int(m.get("size_bytes") or 0),
                "max_context": m.get("max_context_length"),
                "vision": caps.get("vision") if "vision" in caps else None,
                "instances": instances,
            }
        )
    rows.sort(key=lambda r: r["key"].lower())
    return rows


async def _read_json(resp: aiohttp.ClientResponse) -> Mapping:
    try:
        data = await resp.json(content_type=None)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


async def _fail(resp: aiohttp.ClientResponse, what: str) -> None:
    """Raise with LM Studio's own explanation, which it sends as JSON or as plain text."""
    body = (await resp.text()).strip()
    detail = body
    try:
        parsed = json.loads(body)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict):
            detail = str(err.get("message") or body)
        elif err:
            detail = str(err)
        elif isinstance(parsed, dict) and parsed.get("message"):
            detail = str(parsed["message"])
    except ValueError:
        pass
    raise LMStudioError(f"{what} (HTTP {resp.status}): {detail[:300]}" if detail else f"{what} (HTTP {resp.status})")


async def list_models(base_url: str, api_key: str = "") -> dict:
    """``{"native", "models"}``. ``native`` is False for a server without LM Studio's own API."""
    root = api_root(base_url)
    timeout = aiohttp.ClientTimeout(total=LIST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(headers=_headers(api_key), timeout=timeout) as session:
            async with session.get(f"{root}/api/v1/models") as resp:
                if resp.status == 200:
                    return {"native": True, "models": parse_native(await _read_json(resp))}
                if resp.status in (401, 403):
                    await _fail(resp, "LM Studio refused the request")
            async with session.get(f"{root}/v1/models") as resp:
                if resp.status != 200:
                    await _fail(resp, "The model server refused the request")
                data = await _read_json(resp)
    except LMStudioError:
        raise
    except aiohttp.ClientConnectorError as exc:
        raise LMStudioError(f"Nothing is answering at {root}. Check that the model server is running.") from exc
    except asyncio.TimeoutError as exc:
        raise LMStudioError(f"The model server at {root} did not answer within {LIST_TIMEOUT} seconds.") from exc
    except Exception as exc:
        raise LMStudioError(f"Could not reach the model server at {root}: {exc or type(exc).__name__}") from exc
    rows = [
        {
            "key": str(m.get("id")),
            "type": None,
            "name": "",
            "quant": "",
            "size_bytes": 0,
            "max_context": None,
            "vision": None,
            "instances": [],
        }
        for m in data.get("data") or []
        if m.get("id")
    ]
    rows.sort(key=lambda r: r["key"].lower())
    return {"native": False, "models": rows}


def find(models: List[dict], name: str) -> Optional[dict]:
    """The model a configured name refers to: its key, or the id of one of its loaded instances."""
    if not name:
        return None
    for m in models:
        if m["key"] == name:
            return m
    for m in models:
        if any(i["id"] == name for i in m["instances"]):
            return m
    return None


def describe(listing: Mapping, settings: Settings) -> List[dict]:
    """The listing with what Soupy uses each model for."""
    models = listing.get("models") or []
    used: Dict[str, List[str]] = {}
    for role, name in (("chat", settings.chat), ("vision", settings.vision), ("embeddings", settings.embedding)):
        m = find(models, name)
        if m is not None:
            used.setdefault(m["key"], []).append(role)
    out = []
    for m in models:
        first = m["instances"][0] if m["instances"] else None
        out.append(
            {
                **m,
                "roles": used.get(m["key"], []),
                "loaded": bool(m["instances"]),
                "loaded_context": first["context_length"] if first else None,
            }
        )
    return out


def _whole(raw: object, what: str) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        raise PlanError(f"{what} must be a whole number.") from None


def build_plan(
    listing: Mapping,
    settings: Settings,
    *,
    model: str,
    context_length: object,
    chat_budget: object,
    vision_follows: bool = False,
    bot_running: bool = False,
) -> dict:
    """Work out what switching to ``model`` takes: what to unload, load, and write to ``.env-stable``.

    Pure: nothing is contacted or changed. Raises :class:`PlanError` when the
    request can't be done as asked.
    """
    model = str(model or "").strip()
    if not model:
        raise PlanError("Pick a model.")
    budget = _whole(chat_budget, "The chat prompt budget")
    if budget < MIN_CHAT_BUDGET:
        raise PlanError(f"The chat prompt budget can't be under {MIN_CHAT_BUDGET} tokens.")

    native = bool(listing.get("native"))
    models: List[dict] = list(listing.get("models") or [])
    warnings: List[str] = []
    unload: List[dict] = []
    load: Optional[dict] = None
    keep: List[dict] = []
    context: Optional[int] = None

    vision_before = settings.vision
    vision_after = model if (settings.vision_enabled and vision_follows) else vision_before

    if native:
        target = find(models, model)
        if target is None:
            raise PlanError(f"LM Studio has no model named {model}.")
        if target["type"] == "embedding":
            raise PlanError(f"{model} is an embedding model. It can't write replies.")
        context = _whole(context_length, "The context length")
        if context < MIN_CONTEXT_LENGTH:
            raise PlanError(f"The context length can't be under {MIN_CONTEXT_LENGTH} tokens.")
        if target["max_context"] and context > int(target["max_context"]):
            raise PlanError(f"{target['key']} takes at most {int(target['max_context']):,} tokens of context.")
        if budget > context:
            raise PlanError(
                f"The chat prompt budget ({budget:,}) is larger than the context length ({context:,}). "
                "A long conversation would overflow the model."
            )
        if budget > context * HEADROOM:
            warnings.append(
                f"The chat prompt budget is {round(100 * budget / context)}% of the context length. "
                "Soupy sizes prompts from character counts, so leave some room."
            )
        if settings.vision_enabled and vision_after == model and target["vision"] is False:
            warnings.append(f"LM Studio doesn't list {target['key']} as able to read images.")

        # Reload the target unless one instance of it is already loaded as asked.
        as_asked = len(target["instances"]) == 1 and target["instances"][0]["context_length"] == context
        gone = set()
        if not as_asked:
            for inst in target["instances"]:
                unload.append(_instance(target, inst, "reloaded with the new context length"))
                gone.add(inst["id"])
            load = {"key": target["key"], "context_length": context, "size_bytes": target["size_bytes"]}

        # Then whatever the bot stops using.
        needed = {target["key"]}
        for name in (vision_after, settings.embedding):
            m = find(models, name)
            if m is not None:
                needed.add(m["key"])
        for name, reason in ((settings.chat, "no longer the chat model"), (vision_before, "no longer reads images")):
            m = find(models, name)
            if m is None or m["key"] in needed:
                continue
            for inst in m["instances"]:
                if inst["id"] not in gone:
                    unload.append(_instance(m, inst, reason))
                    gone.add(inst["id"])

        roles = {r["key"]: r["roles"] for r in describe(listing, _after(settings, model, vision_after))}
        for m in models:
            for inst in m["instances"]:
                if inst["id"] not in gone:
                    keep.append({**_instance(m, inst, ""), "roles": roles.get(m["key"], [])})
    else:
        warnings.append(
            "This server doesn't take load requests. The name is saved, and the server loads the model "
            "when the bot first asks for it."
            if listing.get("reachable", True)
            else "The model server can't be reached, so nothing is loaded. The name is saved for when it is back."
        )

    env: Dict[str, str] = {}
    if model != settings.chat:
        env["LOCAL_CHAT"] = model
    if budget != settings.chat_budget:
        env["CONTEXT_WINDOW_TOKENS"] = str(budget)
    if settings.vision_enabled:
        # The bot reads VISION_MODEL and falls back to LOCAL_CHAT, so an empty
        # VISION_MODEL moves with the chat model by itself.
        implied = settings.vision_model or model
        if vision_after != implied:
            env["VISION_MODEL"] = vision_after

    noop = not env and not unload and load is None
    loaded_after = sum(k["size_bytes"] for k in keep) + (load["size_bytes"] if load else 0)
    return {
        "model": model,
        "context_length": context,
        "chat_budget": budget,
        "native": native,
        "unload": unload,
        "load": load,
        "keep": keep,
        "loaded_bytes_after": loaded_after,
        "env": env,
        "vision_after": vision_after,
        "bot_running": bool(bot_running),
        "noop": noop,
        "warnings": warnings,
    }


def _instance(model: Mapping, inst: Mapping, reason: str) -> dict:
    return {
        "id": inst["id"],
        "key": model["key"],
        "context_length": inst["context_length"],
        "size_bytes": model["size_bytes"],
        "reason": reason,
    }


def _after(settings: Settings, model: str, vision_after: str) -> Settings:
    """The settings as they will read once the plan's env changes are written."""
    return Settings(
        base_url=settings.base_url,
        api_key=settings.api_key,
        chat=model,
        vision_model=vision_after,
        vision_enabled=settings.vision_enabled,
        embedding=settings.embedding,
        chat_budget=settings.chat_budget,
        saved_list=settings.saved_list,
    )


async def _unload(session: aiohttp.ClientSession, root: str, instance_id: str) -> None:
    timeout = aiohttp.ClientTimeout(total=UNLOAD_TIMEOUT)
    async with session.post(f"{root}/api/v1/models/unload", json={"instance_id": instance_id}, timeout=timeout) as r:
        if r.status != 200:
            await _fail(r, f"LM Studio could not unload {instance_id}")


async def _load(session: aiohttp.ClientSession, root: str, key: str, context_length: Optional[int]) -> Mapping:
    body: Dict[str, object] = {"model": key, "echo_load_config": True}
    if context_length:
        body["context_length"] = int(context_length)
    timeout = aiohttp.ClientTimeout(total=LOAD_TIMEOUT)
    async with session.post(f"{root}/api/v1/models/load", json=body, timeout=timeout) as r:
        if r.status != 200:
            await _fail(r, f"LM Studio could not load {key}")
        return await _read_json(r)


BotCall = Callable[[], Awaitable[Tuple[bool, str]]]


async def apply_plan(
    plan: Mapping,
    settings: Settings,
    *,
    stop_bot: BotCall,
    start_bot: BotCall,
    write_env: Callable[[Dict[str, str]], None],
    log: Callable[[str], None] = lambda _msg: None,
) -> dict:
    """Carry out a plan from :func:`build_plan`. Returns ``{"ok", "message", "steps"}``.

    The bot is stopped first so it can't call a model that is half way out of
    memory. If LM Studio fails part way, whatever was unloaded is loaded again,
    ``.env-stable`` is left alone, and the bot starts as it was.
    """
    steps: List[str] = []

    def note(text: str) -> None:
        steps.append(text)
        log(text)

    was_running = bool(plan.get("bot_running"))
    if was_running:
        await stop_bot()
        note("Stopped the bot.")

    root = api_root(settings.base_url)
    unloaded: List[Mapping] = []
    applied_context = None
    try:
        async with aiohttp.ClientSession(headers=_headers(settings.api_key)) as session:
            try:
                for inst in plan.get("unload") or []:
                    await _unload(session, root, inst["id"])
                    unloaded.append(inst)
                    note(f"Unloaded {inst['id']}.")
                target = plan.get("load")
                if target:
                    note(f"Loading {target['key']} at {int(target['context_length']):,} tokens.")
                    result = await _load(session, root, target["key"], target["context_length"])
                    applied_context = (result.get("load_config") or {}).get("context_length")
                    note(f"Loaded {target['key']}.")
            except Exception as exc:
                reason = str(exc) if isinstance(exc, LMStudioError) else f"{type(exc).__name__}: {exc}"
                note(f"Failed: {reason}")
                stranded = []
                for inst in unloaded:
                    try:
                        await _load(session, root, inst["key"], inst["context_length"])
                        note(f"Loaded {inst['key']} again.")
                    except Exception as again:
                        stranded.append(inst["key"])
                        note(f"Could not load {inst['key']} again: {again}")
                if was_running:
                    await start_bot()
                    note("Started the bot.")
                if stranded:
                    tail = f"{', '.join(stranded)} is no longer loaded. Load it in LM Studio by hand."
                elif unloaded:
                    tail = "The models that were unloaded are loaded again."
                else:
                    tail = "Nothing was unloaded."
                if not reason.endswith((".", "!", "?")):
                    reason += "."
                return {"ok": False, "message": f"{reason} {tail} No settings were changed.", "steps": steps}
    except Exception as exc:  # the session itself could not be made
        if was_running:
            await start_bot()
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}", "steps": steps}

    env = dict(plan.get("env") or {})
    if env:
        write_env(env)
        note("Saved " + ", ".join(sorted(env)) + ".")

    message = f"{plan['model']} is the chat model."
    if plan.get("load"):
        message = f"Loaded {plan['model']} at {int(plan['load']['context_length']):,} tokens."
    elif not plan.get("native") and "LOCAL_CHAT" in env:
        message = f"Saved {plan['model']} as the chat model."
    elif "LOCAL_CHAT" not in env and env:
        message = "Saved."
    if applied_context and plan.get("load") and int(applied_context) != int(plan["load"]["context_length"]):
        message += f" LM Studio applied a context length of {int(applied_context):,} instead."

    if was_running:
        ok, detail = await start_bot()
        if ok:
            note("Started the bot.")
            message += " The bot restarted."
        else:
            note(f"The bot did not start: {detail}")
            return {"ok": False, "message": f"{message} The bot did not start again: {detail}", "steps": steps}
    else:
        message += " The bot is stopped and uses it when it starts."
    return {"ok": True, "message": message, "steps": steps}
