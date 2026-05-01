"""Step 4 — LM Studio.

Probe `GET {base}/models`, list whatever's loaded, and ask the user to
pick the chat model and the embedding model (must be different). If
vision was selected in step 1, also pick the vision model from the same
list.
"""

from __future__ import annotations

from typing import Dict, List

from .. import validators


DEFAULT_BASE_URL = "http://localhost:1234/v1"


def _probe(base_url: str, ui) -> List[str]:
    while True:
        ui.step(f"probing {base_url}/models")
        result = validators.lm_studio_models(base_url)
        if result.ok:
            ui.ok(result.message)
            return list(result.data)
        ui.fail(result.message)
        choice = ui.retry_skip_abort("How do you want to handle this?")
        if choice == "retry":
            continue
        if choice == "skip":
            ui.warn("continuing without a verified model list — you'll fill these in by hand")
            return []
        raise SystemExit("aborted at LM Studio probe")


def _pick(label: str, models: List[str], ui, *, default: str = "") -> str:
    if not models:
        return ui.prompt(label, default=default or None)
    options = [(m, m) for m in models]
    return ui.prompt_choice(label, options, default=default if default in models else None)


def _prompt_base_url(state: Dict, ui) -> str:
    return ui.prompt(
        "LM Studio base URL",
        default=state.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
    )


def run(state: Dict, ui) -> Dict:
    ui.header("Step 4 — LM Studio")
    ui.info("Make sure LM Studio is running with a chat model AND an embedding model loaded.")
    ui.hr()

    base_url = _prompt_base_url(state, ui)
    models = _probe(base_url, ui)

    # If only one model is loaded, walk the user through loading the second one.
    if len(models) == 1:
        ui.warn("only one model is loaded — Soupy needs both a chat model and an embedding model")
        ui.info("load the missing one in LM Studio (Models tab -> load), then continue")
        while len(models) < 2:
            try:
                input("press enter to re-probe... ")
            except EOFError:
                ui.warn("stdin closed; continuing without verifying both models are loaded")
                break
            models = _probe(base_url, ui)
            if not models:
                break

    chat_default = state.get("LOCAL_CHAT", "")
    embed_default = state.get("RAG_EMBEDDING_MODEL", "")

    chat_model = _pick("Chat model id:", models, ui, default=chat_default)
    embed_options = [m for m in models if m != chat_model] or models
    embed_model = _pick(
        "Embedding model id (must differ from chat model):",
        embed_options,
        ui,
        default=embed_default,
    )
    if chat_model == embed_model:
        ui.warn("chat and embedding model are the same id — RAG will misbehave; fix this in LM Studio later")

    out: Dict = {
        "OPENAI_BASE_URL": base_url,
        "LOCAL_CHAT": chat_model,
        "RAG_EMBEDDING_MODEL": embed_model,
    }

    if state.get("vision"):
        vision_default = state.get("VISION_MODEL", "")
        vision_model = _pick("Vision model id:", models, ui, default=vision_default)
        out["ENABLE_VISION"] = "true"
        out["VISION_MODEL"] = vision_model

    return out
