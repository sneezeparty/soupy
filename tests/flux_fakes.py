"""Stand-ins for testing the Flux cog without Discord, the Flux server or the real media folder.

``load_cog`` loads ``soupy.cogs.flux`` onto the test bot and points its side
effects at a temporary folder. ``FakeFlux`` answers the three generation
endpoints over real HTTP and records what it was sent. The ``Fake*`` Discord
objects carry only what the cog touches.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import sys

from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image

import soupy_remastered_stablediffusion as bot_module  # noqa: E402 (conftest sets env first)


def png(width=64, height=48, colour=(200, 30, 30)) -> bytes:
    """A real PNG, big enough to pass the cog's "is this an image" check."""
    image = Image.effect_noise((width, height), 64).convert("RGB")
    image.paste(colour, (0, 0, width // 2, height // 2))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


SOURCE = png(1392, 752)
RESULT = png(96, 64, (30, 30, 200))


class FakeFlux:
    """Stands in for flux_server.py: records each generation request and returns a PNG."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        # Set to (status, json body) to make the next requests fail.
        self.failure: tuple[int, dict] | None = None
        # aiohttp refuses bodies over 1 MB by default, and a 1 MP PNG is bigger.
        self.app = web.Application(client_max_size=64 * 1024**2)
        self.app.router.add_get("/source.png", self._source)
        for path in ("/flux", "/flux_img2img", "/flux_edit"):
            self.app.router.add_post(path, self._generate)

    async def _source(self, _request: web.Request) -> web.Response:
        return web.Response(body=SOURCE, content_type="image/png")

    async def _generate(self, request: web.Request) -> web.Response:
        fields = {}
        for name, value in (await request.post()).items():
            fields[name] = value.file.read() if isinstance(value, web.FileField) else value
        self.calls.append((request.path, fields))
        if self.failure:
            status, body = self.failure
            return web.json_response(body, status=status)
        return web.Response(body=RESULT, content_type="image/png", headers={"X-Flux-Model": "test-model"})


class FakeResponse:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.modal = None
        self._done = False

    def is_done(self) -> bool:
        return self._done

    async def send_message(self, content=None, **_kwargs) -> None:
        self._done = True
        self.messages.append(content)

    async def send_modal(self, modal) -> None:
        self._done = True
        self.modal = modal

    async def defer(self, **_kwargs) -> None:
        self._done = True


class FakeFollowup:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    async def send(self, content=None, **kwargs) -> None:
        self.sent.append((content, kwargs))


class FakeChannel:
    id = 222

    @contextlib.asynccontextmanager
    async def typing(self):
        yield


class FakeUser:
    id = 111
    mention = "<@111>"

    def __str__(self) -> str:
        return "tester"


class FakeGuild:
    id = 333


class FakeAttachment:
    def __init__(self, url: str, filename: str = "", width: int = 0, height: int = 0) -> None:
        self.url = url
        self.filename = filename
        self.width = width
        self.height = height


class FakeEmbed:
    def __init__(self, description: str) -> None:
        self.description = description


class FakeMessage:
    """A posted result: one attachment and, optionally, its two embeds."""

    def __init__(self, attachment_url: str, filename: str = "", *, embeds=(), width: int = 0, height: int = 0) -> None:
        self.attachments = [FakeAttachment(attachment_url, filename, width, height)]
        self.embeds = [FakeEmbed(text) for text in embeds]


class FakeInteraction:
    def __init__(self, message=None, *, responded=True) -> None:
        self.response = FakeResponse()
        self.response._done = responded
        self.followup = FakeFollowup()
        self.channel = FakeChannel()
        self.user = FakeUser()
        self.guild = FakeGuild()
        self.guild_id = FakeGuild.id
        self.message = message
        self.deleted_original = 0

    async def delete_original_response(self) -> None:
        self.deleted_original += 1


class FakeQueue:
    def __init__(self) -> None:
        self.items: list[dict] = []

    async def put(self, item) -> None:
        self.items.append(item)

    def qsize(self) -> int:
        return 0


def load_cog(monkeypatch, tmp_path):
    """The Flux cog with its side effects pointed away from the real bot and media folder.

    The archive is real and lives under ``tmp_path``, so a result can be
    posted and then found again by a click. ``cog.archived`` also lists what
    each result was archived with.
    """

    async def load():
        if "soupy.cogs.sd" not in bot_module.bot.extensions:
            await bot_module.bot.load_extension("soupy.cogs.sd")
        if "soupy.cogs.flux" not in bot_module.bot.extensions:
            await bot_module.bot.load_extension("soupy.cogs.flux")

    asyncio.run(load())
    cog = sys.modules["soupy.cogs.flux"]
    sd = sys.modules["soupy.cogs.sd"]

    def media_dirs():
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / "thumbs").mkdir(exist_ok=True)
        return tmp_path

    async def no_stat(*_args, **_kwargs):
        return None

    cog.archived = []

    def archive(data, **kwargs):
        cog.archived.append(kwargs)
        sd.archive_image_bytes(data, **kwargs)

    monkeypatch.setattr(cog, "_ensure_media_dirs", media_dirs)
    monkeypatch.setattr(sd, "_ensure_media_dirs", media_dirs)
    monkeypatch.setattr(cog, "_archive_image_bytes", archive)
    monkeypatch.setattr(cog, "archive_sent_message", lambda **_kwargs: None)
    monkeypatch.setattr(cog, "increment_user_stat", no_stat)
    monkeypatch.setattr(cog, "_workload", cog._Workload())
    monkeypatch.setattr(cog.bot, "sd_queue", FakeQueue())
    # Owners skip the per-user cooldown, which a test clicking five buttons would hit.
    monkeypatch.setattr(bot_module, "OWNER_IDS", [FakeUser.id])
    for name, value in {
        "flux_enabled": True,
        "flux_img2img_url": "",
        "flux_edit_url": "",
        "flux_edit_enabled": True,
        "flux_model": "klein",
        "flux_edit_model": "flux2-klein-4b",
        "flux_steps": 4,
        "flux_guidance": 1.0,
        "flux_default_strength": 0.5,
        "flux_edit_steps": 6,
        "flux_edit_guidance": 1.0,
    }.items():
        monkeypatch.setitem(cog.settings.__dict__, name, value)
    return cog


def run_item(cog, monkeypatch, build_item, *, failure=None):
    """Queue-consumer stand-in: run one queue item against a fake Flux server.

    ``build_item(base_url)`` returns the item, so a test can point an attachment
    URL at the fake server. Returns ``(server, interaction)``.
    """

    async def scenario():
        server = FakeFlux()
        server.failure = failure
        async with TestServer(server.app) as http:
            base = str(http.make_url("")).rstrip("/")
            monkeypatch.setitem(cog.settings.__dict__, "flux_server_url", base)
            item = build_item(base)
            await cog.process_flux_image(item)
            return server, item["interaction"]

    return asyncio.run(scenario())


def posted(interaction) -> dict:
    """The result message an interaction ended with: its file name, embeds and view."""
    ((_content, kwargs),) = interaction.followup.sent
    return {
        "filename": kwargs["file"].filename,
        "prompt_embed": kwargs["embeds"][0].description,
        "details": kwargs["embeds"][1].description,
        "view": kwargs["view"],
    }


def details(interaction) -> str:
    """The second embed of the posted result: seed, action, time, strength."""
    return posted(interaction)["details"]


def as_message(result: dict, *, width=1392, height=752, url="https://cdn.example/result.png") -> FakeMessage:
    """The Discord message for a posted result, as a later click would see it."""
    return FakeMessage(
        url, result["filename"], embeds=(result["prompt_embed"], result["details"]), width=width, height=height
    )


def labels(view) -> list[str]:
    return [child.label for child in view.children]


def import_flux_server():
    """flux_server.py, imported without letting it read ``.env-stable``.

    Skips the test where the Flux extras are not installed: the server's form
    endpoints need python-multipart, which ``requirements.txt`` leaves out.
    """
    import pytest

    try:
        import python_multipart  # noqa: F401
    except ImportError:
        pytest.importorskip("multipart", reason="flux_server.py needs python-multipart (a Flux extra)")
    import dotenv

    real = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *_args, **_kwargs: False
    try:
        sys.modules.pop("flux_server", None)
        return importlib.import_module("flux_server")
    finally:
        dotenv.load_dotenv = real
