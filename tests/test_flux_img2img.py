"""Image-to-image on the Flux backend: which pipeline a request takes, and what strength means.

An image plus a prompt runs Klein edit (the prompt is an instruction) unless a
strength is given, which picks noise-mix img2img (the prompt describes the
result). The 🖼️ button and ``/flux`` with an attachment follow the same rule.

The Discord side is driven with stand-ins and a fake Flux server; no model runs
here. ``flux_server.py`` is imported with its ``.env-stable`` loading switched
off so the suite never reads the real configuration.
"""

from __future__ import annotations

import asyncio
import math
import sys

import pytest

from tests.flux_fakes import (
    RESULT,
    SOURCE,
    FakeInteraction,
    FakeMessage,
    details,
    import_flux_server,
    load_cog,
    run_item,
)


@pytest.fixture
def flux(monkeypatch, tmp_path):
    return load_cog(monkeypatch, tmp_path)


def submit(cog, *, prompt, strength, edit_available=True, source_image=None, source_url="http://unused/source.png"):
    """Fill in the 🖼️ dialog and submit it. Returns ``(interaction, queued items)``."""

    async def scenario():
        modal = cog.FluxImg2ImgModal(source_image, 0.5, source_url, edit_available=edit_available)
        modal.prompt_input._value = prompt
        modal.strength_input._value = strength
        interaction = FakeInteraction(responded=False)
        await modal.on_submit(interaction)
        return interaction, cog.bot.sd_queue.items

    return asyncio.run(scenario())


# ---- which pipeline -----------------------------------------------------------


def test_button_with_blank_strength_edits_by_instruction(flux, monkeypatch):
    interaction, queued = submit(flux, prompt="Make it a theme park", strength="")
    (note,) = interaction.response.messages
    assert note.startswith("🛠️ Editing the image...")
    (item,) = queued
    assert item["strength"] is None

    def build(base):
        return {**item, "image_url": f"{base}/source.png"}

    server, interaction = run_item(flux, monkeypatch, build)
    ((path, fields),) = server.calls
    assert path == "/flux_edit"
    assert fields["prompt"] == "Make it a theme park"
    assert fields["image"] == SOURCE
    assert fields["steps"] == "6"  # the edit pipeline's own steps, not FLUX_STEPS
    assert (fields["width"], fields["height"]) == ("1392", "752")
    assert "strength" not in fields and "negative_prompt" not in fields
    assert "⚡ Edit" in details(interaction) and "🎚️" not in details(interaction)
    assert [a["source"] for a in flux.archived] == ["flux-edit"]


def test_button_with_a_strength_runs_noise_mix(flux, monkeypatch):
    interaction, queued = submit(flux, prompt="a theme park at night", strength="0.6")
    (note,) = interaction.response.messages
    assert note.startswith("🛠️ Running img2img at strength 0.6...")
    (item,) = queued
    assert item["strength"] == 0.6

    server, interaction = run_item(flux, monkeypatch, lambda base: {**item, "image_url": f"{base}/source.png"})
    ((path, fields),) = server.calls
    assert path == "/flux_img2img"
    assert (fields["strength"], fields["steps"]) == ("0.6", "4")
    assert "negative_prompt" not in fields  # Flux has none; the server ignored it
    assert "⚡ Img2Img" in details(interaction) and "🎚️ 0.6" in details(interaction)
    assert [a["source"] for a in flux.archived] == ["flux-img2img"]


def test_without_the_edit_pipeline_the_button_blends_at_the_default(flux, monkeypatch):
    monkeypatch.setitem(flux.settings.__dict__, "flux_edit_enabled", False)

    async def build_modal():
        return flux.FluxImg2ImgModal(None, 0.5, "http://unused/source.png", edit_available=False)

    modal = asyncio.run(build_modal())
    assert modal.strength_input.required and modal.strength_input.default == "0.5"

    _, queued = submit(flux, prompt="a theme park", strength="", edit_available=False)
    (item,) = queued
    assert item["strength"] == 0.5
    server, _ = run_item(flux, monkeypatch, lambda base: {**item, "image_url": f"{base}/source.png"})
    assert [path for path, _ in server.calls] == ["/flux_img2img"]


@pytest.mark.parametrize("strength, path", [(None, "/flux_edit"), (0.3, "/flux_img2img"), (0.0, "/flux_img2img")])
def test_flux_command_with_an_image_follows_the_same_rule(flux, monkeypatch, strength, path):
    """``/flux`` used to ignore its strength option whenever the edit pipeline was on."""

    def build(base):
        return {
            "type": "flux",
            "interaction": FakeInteraction(),
            "description": "make this anime",
            "size": "wide",
            "seed": 7,
            "image_url": f"{base}/source.png",
            "strength": strength,
        }

    server, _ = run_item(flux, monkeypatch, build)
    ((called, fields),) = server.calls
    assert called == path
    if strength is not None:
        assert fields["strength"] == str(strength)


def test_flux_command_with_an_image_and_no_edit_pipeline_uses_the_default_strength(flux, monkeypatch):
    monkeypatch.setitem(flux.settings.__dict__, "flux_edit_enabled", False)

    def build(base):
        return {
            "type": "flux",
            "interaction": FakeInteraction(),
            "description": "a watercolour",
            "seed": 7,
            "image_url": f"{base}/source.png",
            "strength": None,
        }

    server, _ = run_item(flux, monkeypatch, build)
    ((path, fields),) = server.calls
    assert (path, fields["strength"]) == ("/flux_img2img", "0.5")


def test_text_to_image_is_untouched(flux, monkeypatch):
    def build(_base):
        return {"type": "flux", "interaction": FakeInteraction(), "description": "a cat", "size": "wide", "seed": 7}

    server, interaction = run_item(flux, monkeypatch, build)
    ((path, fields),) = server.calls
    assert path == "/flux" and "image" not in fields and "negative_prompt" not in fields
    assert "⚡ Flux" in details(interaction)


# ---- where the picture comes from ---------------------------------------------


def test_saved_source_file_is_used_without_a_download(flux, monkeypatch, tmp_path):
    (tmp_path / "flux_sources").mkdir()
    (tmp_path / "flux_sources" / "saved.png").write_bytes(SOURCE)

    def build(_base):
        return {
            "type": "flux",
            "interaction": FakeInteraction(),
            "action": "img2img",
            "prompt": "add an alien",
            "strength": None,
            "seed": 7,
            "source_image": "flux_sources/saved.png",
            "image_url": "http://127.0.0.1:9/never-fetched.png",
        }

    server, _ = run_item(flux, monkeypatch, build)
    ((path, fields),) = server.calls
    assert path == "/flux_edit" and fields["image"] == SOURCE


def test_missing_source_file_falls_back_to_the_attachment(flux, monkeypatch):
    def build(base):
        return {
            "type": "flux",
            "interaction": FakeInteraction(),
            "action": "img2img",
            "prompt": "add an alien",
            "strength": None,
            "seed": 7,
            "source_image": "flux_sources/deleted.png",
            "image_url": f"{base}/source.png",
        }

    server, _ = run_item(flux, monkeypatch, build)
    ((path, fields),) = server.calls
    assert path == "/flux_edit" and fields["image"] == SOURCE


def test_a_request_about_a_picture_never_becomes_text_to_image(flux, monkeypatch):
    """With the source gone, the instruction alone would generate something unrelated."""

    def build(_base):
        return {
            "type": "flux",
            "interaction": FakeInteraction(),
            "action": "img2img",
            "prompt": "Make it a theme park",
            "strength": None,
            "seed": 7,
            "source_image": "images/deleted.png",
        }

    server, interaction = run_item(flux, monkeypatch, build)
    assert server.calls == []
    ((message, kwargs),) = interaction.followup.sent
    assert "Couldn't load the image" in message and "file" not in kwargs


def test_button_hands_the_dialog_the_archived_picture_and_the_attachment(flux, tmp_path):
    """The displayed picture is read from the archive; nothing copies it to flux_sources."""

    async def click(filename):
        view = flux.FluxRemixView("a cat", 1392, 752, 7)
        interaction = FakeInteraction(FakeMessage("https://cdn.example/result.png", filename), responded=False)
        await view.img2img_button.callback(interaction)
        return interaction.response.modal

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "123456_acat.png").write_bytes(SOURCE)
    modal = asyncio.run(click("123456_acat.png"))
    assert (modal.source_image, modal.source_url) == ("images/123456_acat.png", "https://cdn.example/result.png")
    assert modal.edit_available is True
    assert modal.strength_input.required is False and modal.strength_input.default is None

    # Deleted from the archive (the panel's Media tab can do that): the attachment is left.
    modal = asyncio.run(click("999999_gone.png"))
    assert (modal.source_image, modal.source_url) == (None, "https://cdn.example/result.png")


def test_an_upload_is_saved_once_and_named_as_the_result_source(flux, monkeypatch, tmp_path):
    def build(base):
        return {
            "type": "flux",
            "interaction": FakeInteraction(),
            "description": "make this anime",
            "seed": 7,
            "image_url": f"{base}/source.png",
            "strength": None,
        }

    run_item(flux, monkeypatch, build)
    run_item(flux, monkeypatch, build)
    (saved,) = (tmp_path / "flux_sources").iterdir()
    assert saved.read_bytes() == SOURCE
    assert [a["extra"]["source_image"] for a in flux.archived] == [f"flux_sources/{saved.name}"] * 2
    # A result is not copied anywhere but the archive.
    assert len(list((tmp_path / "images").glob("*.png"))) == 2


@pytest.mark.parametrize(
    "ref", ["images/../../.env-stable", "logs/soupy.log", "/etc/passwd", "images/", "flux_sources/a/b.png", "", None]
)
def test_a_source_reference_stays_inside_the_two_media_folders(flux, ref):
    assert flux._source_path(ref) is None and flux._load_source(ref) is None


# ---- the dialog ---------------------------------------------------------------


@pytest.mark.parametrize("strength", ["strong", "1.5", "-0.2"])
def test_dialog_rejects_a_bad_strength(flux, strength):
    interaction, queued = submit(flux, prompt="add an alien", strength=strength)
    assert queued == []
    (message,) = interaction.response.messages
    assert "Strength must be" in message


def test_dialog_and_command_text_fit_discords_limits(flux):
    async def build():
        return [flux.FluxImg2ImgModal(None, 0.5, "u", edit_available=flag).to_dict() for flag in (True, False)]

    for modal in asyncio.run(build()):
        assert len(modal["title"]) <= 45
        for row in modal["components"]:
            (field,) = row["components"]
            assert len(field["label"]) <= 45, field["label"]
            assert len(field.get("placeholder", "")) <= 100, field["placeholder"]
    options = flux.flux.to_dict(flux.bot.tree)["options"]
    assert [option["name"] for option in options] == ["description", "size", "image", "strength", "seed", "fancy"]
    for option in options:
        assert 1 <= len(option["description"]) <= 100, option["name"]


# ---- the server's strength schedule -------------------------------------------


@pytest.fixture(scope="module")
def server():
    """flux_server.py, imported without letting it read ``.env-stable``."""
    return import_flux_server()


def mflux_start_step(total: int, image_strength: float) -> int:
    """mflux's own rule (models/common/config/config.py, ``init_time_step``)."""
    return max(1, int(total * max(0.0, min(1.0, image_strength))))


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 20, 50])
def test_schedule_always_runs_the_steps_asked_for(server, steps):
    for hundredths in range(0, 101):
        total, image_strength = server._img2img_schedule(steps, hundredths / 100)
        start = mflux_start_step(total, image_strength)
        assert 0.0 < image_strength < 1.0
        assert total - start == steps, (steps, hundredths, total, start)


def test_schedule_starts_later_the_lower_the_strength(server):
    """Position on the noise schedule: 0 is pure noise, 1 is the untouched image."""
    positions = []
    for hundredths in range(5, 101, 5):
        total, image_strength = server._img2img_schedule(4, hundredths / 100)
        positions.append(mflux_start_step(total, image_strength) / total)
    assert positions == sorted(positions, reverse=True)
    # Passing 4 steps straight to mflux left three settings across this range.
    assert len(set(positions)) >= 10
    assert math.isclose(positions[9], 0.5)  # strength 0.5 starts half-way


def test_schedule_start_noise_rises_with_strength_on_the_real_schedule(server):
    """Against mflux's own scheduler, so a change in its rounding or schedule shows up here."""
    pytest.importorskip("mflux")
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    def start_noise(strength: float) -> float:
        total, image_strength = server._img2img_schedule(4, strength)
        config = Config(
            model_config=ModelConfig.flux2_klein_4b(),
            num_inference_steps=total,
            height=752,
            width=1392,
            guidance=1.0,
            image_path="unused.png",
            image_strength=image_strength,
        )
        assert total - config.init_time_step == 4
        return float(config.scheduler.sigmas[config.init_time_step])

    noise = [start_noise(s) for s in (0.1, 0.25, 0.5, 0.6, 0.7, 0.8)]
    assert noise == sorted(noise) and len(set(noise)) == len(noise)
    assert noise[0] < 0.35 and 0.7 < noise[2] < 0.8 and noise[-1] > 0.9


def test_img2img_endpoint_sends_the_stretched_schedule_to_mflux(server, monkeypatch):
    from fastapi.testclient import TestClient

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return RESULT

    monkeypatch.setattr(server, "_generate", fake_generate)
    response = TestClient(server.app).post(
        "/flux_img2img",
        data={"prompt": "a theme park", "steps": "4", "strength": "0.6", "width": "1392", "height": "752", "seed": "7"},
        files={"image": ("source.png", SOURCE, "image/png")},
    )
    assert response.status_code == 200 and response.content == RESULT
    assert seen["steps"] == 7  # 4 denoising steps at the end of a 7-step schedule
    assert mflux_start_step(seen["steps"], seen["strength"]) == 3
    assert (seen["width"], seen["height"]) == (1392, 752)


# ---- the server's housekeeping ----------------------------------------------------


def test_every_picture_names_the_model_that_made_it(server, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_generate", lambda **_kwargs: RESULT)
    monkeypatch.setattr(server, "_generate_edit", lambda **_kwargs: RESULT)
    client = TestClient(server.app)
    upload = {"image": ("source.png", SOURCE, "image/png")}
    text = client.post("/flux", data={"prompt": "a cat"})
    blend = client.post("/flux_img2img", data={"prompt": "a cat", "strength": "0.5"}, files=upload)
    edit = client.post("/flux_edit", data={"prompt": "add a hat"}, files=upload)
    assert [r.status_code for r in (text, blend, edit)] == [200, 200, 200]
    assert (text.headers["X-Flux-Model"], blend.headers["X-Flux-Model"]) == (server.FLUX_MODEL, server.FLUX_MODEL)
    assert edit.headers["X-Flux-Model"] == server.FLUX_EDIT_MODEL


def test_a_failed_run_returns_the_reason(server, monkeypatch):
    from fastapi.testclient import TestClient

    def refuse(**_kwargs):
        raise RuntimeError("[metal] buffer too large")

    monkeypatch.setattr(server, "_generate", refuse)
    response = TestClient(server.app).post("/flux", data={"prompt": "a cat"})
    assert response.status_code == 500 and response.json() == {"detail": "[metal] buffer too large"}


def test_the_mlx_cache_cap_is_set_once_before_a_model_loads(server, monkeypatch):
    calls = []

    class FakeCore:
        @staticmethod
        def set_cache_limit(limit):
            calls.append(limit)
            return 60 * 1024**3

    import types

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = FakeCore
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", FakeCore)
    monkeypatch.setattr(server, "_model", None)
    monkeypatch.setattr(server, "_construct_mflux", lambda: "the model")

    monkeypatch.setattr(server, "_cache_limit_applied", False)
    monkeypatch.setattr(server, "FLUX_CACHE_LIMIT_GB", 2.0)
    assert server._load_model() == "the model"
    server._apply_cache_limit()
    assert calls == [2 * 1024**3]

    # 0 leaves MLX's own default in place.
    calls.clear()
    monkeypatch.setattr(server, "_cache_limit_applied", False)
    monkeypatch.setattr(server, "FLUX_CACHE_LIMIT_GB", 0.0)
    server._apply_cache_limit()
    assert calls == []


def test_health_reports_the_cap_and_stays_out_of_the_access_log(server):
    import logging

    from fastapi.testclient import TestClient

    health = TestClient(server.app).get("/health").json()
    assert health["status"] == "ok" and health["cache_limit_gb"] == server.FLUX_CACHE_LIMIT_GB

    def passes(message):
        record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, message, None, None)
        return all(f.filter(record) for f in logging.getLogger("uvicorn.access").filters)

    assert not passes('127.0.0.1:5 - "GET /health HTTP/1.1" 200 OK')
    assert passes('127.0.0.1:5 - "POST /flux HTTP/1.1" 200 OK')
