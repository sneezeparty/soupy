"""Tests for web.services.lm_studio: reading the model list, planning a switch, carrying it out.

The plan is a pure function, so most cases need no server. The ones that carry
a plan out talk to ``tests/fake_lm_studio.py`` over real HTTP.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestServer

from tests import fake_lm_studio as fake
from web.services import lm_studio

GEMMA12 = "google/gemma-4-12b"
GEMMA31 = "google/gemma-4-31b"
LLAMA = "llama-3.2-dark-champion"
EMBED = "text-embedding-qwen3-embedding-0.6b"

# The maintainer's setup: one model for chat and images, loaded with more
# context than the chat budget uses.
ENV = {
    "OPENAI_BASE_URL": "http://localhost:1234/v1",
    "LOCAL_KEY": "lm-studio",
    "LOCAL_CHAT": GEMMA12,
    "VISION_MODEL": GEMMA12,
    "ENABLE_VISION": "true",
    "RAG_EMBEDDING_MODEL": EMBED,
    "CONTEXT_WINDOW_TOKENS": "16000",
}


def settings(**changes):
    return lm_studio.settings_from_env({**ENV, **changes})


def listing(models=None):
    return {"native": True, "models": lm_studio.parse_native({"models": models or fake.DEFAULT_MODELS})}


def plan(env=None, models=None, **request):
    request.setdefault("model", GEMMA12)
    request.setdefault("context_length", 28160)
    request.setdefault("chat_budget", 16000)
    return lm_studio.build_plan(listing(models), settings(**(env or {})), **request)


def ids(rows):
    return [r["id"] for r in rows]


# ---- Settings and the listing ------------------------------------------------


def test_settings_read_the_env_the_way_the_bot_does():
    s = settings()
    assert (s.chat, s.vision, s.embedding, s.chat_budget) == (GEMMA12, GEMMA12, EMBED, 16000)
    assert s.api_key == "lm-studio"
    assert settings(OPENAI_API_KEY="preferred").api_key == "preferred"
    # An empty VISION_MODEL means the chat model reads images.
    assert settings(VISION_MODEL="").vision == GEMMA12
    assert settings(ENABLE_VISION="false").vision == ""
    assert settings(CONTEXT_WINDOW_TOKENS="").chat_budget == lm_studio.DEFAULT_CHAT_BUDGET
    assert settings(AVAILABLE_MODELS="a, b\nc,,").saved_list == ["a", "b", "c"]


@pytest.mark.parametrize(
    "base_url, root",
    [
        ("http://localhost:1234/v1", "http://localhost:1234"),
        ("http://localhost:1234/v1/", "http://localhost:1234"),
        ("http://host/v1proxy/v1", "http://host/v1proxy"),
        ("http://localhost:1234", "http://localhost:1234"),
        ("", "http://localhost:1234"),
    ],
)
def test_api_root_strips_only_the_trailing_v1(base_url, root):
    assert lm_studio.api_root(base_url) == root


def test_listing_rows_carry_what_the_dashboard_shows():
    rows = {m["key"]: m for m in listing()["models"]}
    gemma = rows[GEMMA12]
    assert gemma["type"] == "llm" and gemma["vision"] is True
    assert gemma["size_bytes"] == 7 * fake.GB and gemma["max_context"] == 131072
    assert gemma["instances"] == [{"id": GEMMA12, "context_length": 28160}]
    assert rows[GEMMA31]["instances"] == []
    assert rows[EMBED]["type"] == "embedding" and rows[EMBED]["vision"] is None
    assert rows[LLAMA]["vision"] is False


def test_describe_says_what_each_model_is_used_for():
    rows = {m["key"]: m for m in lm_studio.describe(listing(), settings())}
    assert rows[GEMMA12]["roles"] == ["chat", "vision"]
    assert (rows[GEMMA12]["loaded"], rows[GEMMA12]["loaded_context"]) == (True, 28160)
    assert rows[EMBED]["roles"] == ["embeddings"]
    assert (rows[GEMMA31]["roles"], rows[GEMMA31]["loaded"], rows[GEMMA31]["loaded_context"]) == ([], False, None)
    assert lm_studio.describe(listing(), settings(ENABLE_VISION="false"))[0]["roles"] == ["chat"]


def test_a_configured_name_can_be_an_instance_id():
    models = [fake.model(GEMMA12, loaded_context=8192)]
    models[0]["loaded_instances"][0]["id"] = "soupy-chat"
    rows = lm_studio.parse_native({"models": models})
    assert lm_studio.find(rows, "soupy-chat")["key"] == GEMMA12
    assert lm_studio.find(rows, "nothing") is None


# ---- Planning ----------------------------------------------------------------


def test_asking_for_what_is_already_so_changes_nothing():
    p = plan()
    assert p["noop"] is True
    assert (p["unload"], p["load"], p["env"]) == ([], None, {})
    assert ids(p["keep"]) == [GEMMA12, EMBED]


def test_a_new_budget_alone_touches_only_the_env():
    p = plan(chat_budget=20000)
    assert p["env"] == {"CONTEXT_WINDOW_TOKENS": "20000"}
    assert (p["unload"], p["load"], p["noop"]) == ([], None, False)


def test_a_new_context_length_reloads_the_model_even_when_it_also_reads_images():
    """The old switch protected the vision model from unloading, so reloading a
    model that did both jobs left the first copy in memory and loaded a second."""
    p = plan(context_length=16384)
    assert ids(p["unload"]) == [GEMMA12]
    assert p["load"] == {"key": GEMMA12, "context_length": 16384, "size_bytes": 7 * fake.GB}
    assert p["env"] == {}
    assert ids(p["keep"]) == [EMBED]


def test_switching_with_vision_following_unloads_the_old_model():
    p = plan(model=GEMMA31, vision_follows=True)
    assert ids(p["unload"]) == [GEMMA12]
    assert p["load"]["key"] == GEMMA31
    assert p["env"] == {"LOCAL_CHAT": GEMMA31, "VISION_MODEL": GEMMA31}
    assert ids(p["keep"]) == [EMBED]
    assert p["loaded_bytes_after"] == int(26.9 * fake.GB) + int(0.6 * fake.GB)


def test_switching_without_vision_following_keeps_the_old_model_for_images():
    p = plan(model=GEMMA31)
    assert p["unload"] == []
    assert p["env"] == {"LOCAL_CHAT": GEMMA31}
    kept = {k["id"]: k["roles"] for k in p["keep"]}
    assert kept == {GEMMA12: ["vision"], EMBED: ["embeddings"]}
    assert p["vision_after"] == GEMMA12


def test_an_empty_vision_model_follows_by_itself_or_gets_pinned():
    follows = plan(env={"VISION_MODEL": ""}, model=GEMMA31, vision_follows=True)
    assert follows["env"] == {"LOCAL_CHAT": GEMMA31}
    assert ids(follows["unload"]) == [GEMMA12]

    stays = plan(env={"VISION_MODEL": ""}, model=LLAMA)
    assert stays["env"] == {"LOCAL_CHAT": LLAMA, "VISION_MODEL": GEMMA12}
    assert stays["unload"] == []


def test_vision_settings_are_left_alone_when_vision_is_off():
    p = plan(env={"ENABLE_VISION": "false"}, model=GEMMA31, vision_follows=True)
    assert p["env"] == {"LOCAL_CHAT": GEMMA31}
    assert ids(p["unload"]) == [GEMMA12]


def test_a_separate_vision_model_is_unloaded_once_nothing_uses_it():
    models = [*fake.DEFAULT_MODELS, fake.model("qwen/vision", size_gb=5, loaded_context=8192)]
    p = plan(env={"VISION_MODEL": "qwen/vision"}, models=models, model=GEMMA31, vision_follows=True)
    assert sorted(ids(p["unload"])) == [GEMMA12, "qwen/vision"]
    assert p["env"] == {"LOCAL_CHAT": GEMMA31, "VISION_MODEL": GEMMA31}


def test_a_target_already_loaded_as_asked_is_not_loaded_again():
    models = [
        m if m["key"] != GEMMA31 else fake.model(GEMMA31, size_gb=26.9, loaded_context=28160)
        for m in fake.DEFAULT_MODELS
    ]
    p = plan(models=models, model=GEMMA31, vision_follows=True)
    assert p["load"] is None
    assert ids(p["unload"]) == [GEMMA12]


def test_every_copy_of_the_target_is_replaced_by_one():
    twice = fake.model(GEMMA12, size_gb=7, loaded_context=28160)
    twice["loaded_instances"].append({"id": GEMMA12 + ":2", "config": {"context_length": 28160}})
    p = plan(models=[twice, *fake.DEFAULT_MODELS[1:]])
    assert ids(p["unload"]) == [GEMMA12, GEMMA12 + ":2"]
    assert p["load"]["context_length"] == 28160


def test_models_soupy_does_not_use_are_left_loaded():
    models = [*fake.DEFAULT_MODELS, fake.model("someone/elses", size_gb=3, loaded_context=4096)]
    p = plan(models=models, model=GEMMA31, vision_follows=True)
    kept = {k["id"]: k["roles"] for k in p["keep"]}
    assert kept == {EMBED: ["embeddings"], "someone/elses": []}


@pytest.mark.parametrize(
    "request_, says",
    [
        ({"model": ""}, "Pick a model"),
        ({"model": "no/such-model"}, "no model named"),
        ({"model": EMBED}, "embedding model"),
        ({"context_length": 200000}, "at most 131,072"),
        ({"context_length": 100}, "can't be under"),
        ({"context_length": "lots"}, "whole number"),
        ({"context_length": None}, "whole number"),
        ({"chat_budget": 30000}, "larger than the context length"),
        ({"chat_budget": 100}, "can't be under"),
        ({"chat_budget": ""}, "whole number"),
    ],
)
def test_requests_that_cannot_work_are_refused(request_, says):
    with pytest.raises(lm_studio.PlanError) as err:
        plan(**request_)
    assert says in str(err.value)


def test_a_budget_near_the_context_length_is_warned_about():
    assert plan(chat_budget=16000)["warnings"] == []
    assert plan(chat_budget=int(28160 * 0.85))["warnings"] == []
    tight = plan(chat_budget=27000)
    assert len(tight["warnings"]) == 1 and "96%" in tight["warnings"][0]


def test_a_model_that_cannot_read_images_is_warned_about_when_vision_follows():
    assert any("read images" in w for w in plan(model=LLAMA, vision_follows=True)["warnings"])
    assert plan(model=LLAMA)["warnings"] == []


def test_a_server_without_the_load_api_only_gets_the_name_saved():
    other = {"native": False, "reachable": True, "models": [{"key": "llama3", "type": None, "instances": []}]}
    p = lm_studio.build_plan(other, settings(), model="llama3", context_length=None, chat_budget=16000)
    assert (p["unload"], p["load"], p["context_length"]) == ([], None, None)
    assert p["env"] == {"LOCAL_CHAT": "llama3"}
    assert "doesn't take load requests" in p["warnings"][0]

    down = {"native": False, "reachable": False, "models": []}
    p = lm_studio.build_plan(down, settings(), model=GEMMA31, context_length=None, chat_budget=16000)
    assert "can't be reached" in p["warnings"][0]


# ---- Carrying a plan out -----------------------------------------------------


def run_switch(server, *, env=None, bot_running=True, start_ok=True, **request):
    """Plan a switch against the fake server, then apply it.

    Returns ``(result, timeline, written)``. The timeline is everything that
    happened while the plan was applied, bot calls and LM Studio calls in one
    list, in order.
    """
    timeline = server.timeline = []
    written = {}

    async def stop():
        timeline.append("stop")
        return True, "Bot stopped"

    async def start():
        timeline.append("start")
        return (True, "Bot started") if start_ok else (False, "Entrypoint not found")

    def write_env(updates):
        timeline.append("env")
        written.update(updates)

    async def scenario():
        async with TestServer(server.app) as http:
            s = settings(OPENAI_BASE_URL=str(http.make_url("/v1")), **(env or {}))
            found = await lm_studio.list_models(s.base_url, s.api_key)
            request.setdefault("context_length", 28160)
            request.setdefault("chat_budget", 16000)
            p = lm_studio.build_plan(found, s, bot_running=bot_running, **request)
            timeline.clear()
            return await lm_studio.apply_plan(p, s, stop_bot=stop, start_bot=start, write_env=write_env)

    return asyncio.run(scenario()), timeline, written


def list_models(server, api_key=""):
    async def scenario():
        async with TestServer(server.app) as http:
            return await lm_studio.list_models(str(http.make_url("/v1")), api_key)

    return asyncio.run(scenario())


def test_list_models_reads_the_native_api_with_the_configured_key():
    server = fake.FakeLMStudio(token="lm-studio")
    found = list_models(server, "lm-studio")
    assert found["native"] is True
    assert len(found["models"]) == len(fake.DEFAULT_MODELS)
    assert server.auth == ["Bearer lm-studio"]


def test_list_models_reports_a_refused_token():
    with pytest.raises(lm_studio.LMStudioError) as err:
        list_models(fake.FakeLMStudio(token="right"), "wrong")
    assert "HTTP 401" in str(err.value) and "Invalid API token" in str(err.value)


def test_list_models_falls_back_to_the_openai_list():
    found = list_models(fake.FakeLMStudio(native=False))
    assert found["native"] is False
    assert [m["key"] for m in found["models"]] == [GEMMA12, EMBED]


def test_list_models_says_when_the_server_is_down():
    with pytest.raises(lm_studio.LMStudioError) as err:
        asyncio.run(lm_studio.list_models("http://127.0.0.1:9/v1"))
    assert str(err.value) == "Nothing is answering at http://127.0.0.1:9. Check that the model server is running."


def test_a_switch_stops_the_bot_swaps_the_model_saves_and_restarts():
    """The bot is down before LM Studio is touched, and comes back after the env is written."""
    server = fake.FakeLMStudio(token="lm-studio")
    result, timeline, written = run_switch(server, model=GEMMA31, vision_follows=True)
    assert result["ok"] is True, result
    assert timeline == ["stop", "unload", "load", "env", "start"]
    assert server.loaded() == {GEMMA31: 28160, EMBED: 8192}
    assert written == {"LOCAL_CHAT": GEMMA31, "VISION_MODEL": GEMMA31}
    assert "Loaded google/gemma-4-31b at 28,160 tokens" in result["message"]
    assert set(server.auth) == {"Bearer lm-studio"}
    load_call = [body for _method, path, body in server.calls if path.endswith("/load")][0]
    assert (load_call["model"], load_call["context_length"]) == (GEMMA31, 28160)


def test_a_reload_leaves_one_copy_at_the_new_context_length():
    server = fake.FakeLMStudio()
    result, timeline, written = run_switch(server, model=GEMMA12, context_length=16384)
    assert result["ok"] is True, result
    assert server.loaded() == {GEMMA12: 16384, EMBED: 8192}
    assert timeline == ["stop", "unload", "load", "start"]
    assert written == {}


def test_a_failed_load_puts_the_old_model_back_and_changes_nothing():
    server = fake.FakeLMStudio(fail_load=[GEMMA31])
    result, timeline, written = run_switch(server, model=GEMMA31, vision_follows=True)
    assert result["ok"] is False
    assert "insufficient system resources" in result["message"]
    assert "loaded again" in result["message"] and "No settings were changed" in result["message"]
    assert server.loaded() == {GEMMA12: 28160, EMBED: 8192}
    assert written == {}
    # unload old, fail to load new, load old again, bot back up
    assert timeline == ["stop", "unload", "load", "load", "start"]


def test_a_failed_unload_stops_before_anything_is_loaded():
    server = fake.FakeLMStudio(fail_unload=[GEMMA12])
    result, timeline, written = run_switch(server, model=GEMMA31, vision_follows=True)
    assert result["ok"] is False
    assert "Nothing was unloaded" in result["message"]
    assert server.loaded() == {GEMMA12: 28160, EMBED: 8192}
    assert written == {}
    assert timeline == ["stop", "unload", "start"]


def test_a_model_that_cannot_be_put_back_is_named():
    server = fake.FakeLMStudio(fail_load=[GEMMA12])
    result, _timeline, written = run_switch(server, model=GEMMA12, context_length=16384)
    assert result["ok"] is False
    assert f"{GEMMA12} is no longer loaded" in result["message"]
    assert written == {}


def test_a_stopped_bot_stays_stopped():
    result, timeline, written = run_switch(fake.FakeLMStudio(), model=GEMMA31, vision_follows=True, bot_running=False)
    assert result["ok"] is True
    assert timeline == ["unload", "load", "env"]
    assert "The bot is stopped" in result["message"]
    assert written["LOCAL_CHAT"] == GEMMA31


def test_a_bot_that_does_not_come_back_is_reported():
    result, _timeline, written = run_switch(fake.FakeLMStudio(), model=GEMMA31, vision_follows=True, start_ok=False)
    assert result["ok"] is False
    assert "did not start again: Entrypoint not found" in result["message"]
    # The switch itself went through.
    assert written["LOCAL_CHAT"] == GEMMA31


def test_a_budget_change_never_calls_lm_studio():
    result, timeline, written = run_switch(fake.FakeLMStudio(), model=GEMMA12, chat_budget=20000)
    assert result["ok"] is True
    assert timeline == ["stop", "env", "start"]
    assert written == {"CONTEXT_WINDOW_TOKENS": "20000"}
