"""The buttons under a Flux result, and what a result remembers about how it was made.

A click after a restart is handled by a blank template view, so everything a
button needs has to come from the message and the archive row. These tests
post a result through a fake Flux server, rebuild the message a later click
would see, and click with a blank view.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from tests.flux_fakes import (
    SOURCE,
    FakeAttachment,
    FakeInteraction,
    FakeMessage,
    as_message,
    import_flux_server,
    labels,
    load_cog,
    posted,
    run_item,
)

WIDE = (1392, 752)
TALL = (752, 1392)


@pytest.fixture
def flux(monkeypatch, tmp_path):
    return load_cog(monkeypatch, tmp_path)


def generate(cog, monkeypatch, **item) -> dict:
    """Run one queue item and return the posted result."""
    built = {"type": "flux", "seed": 7, **item}

    def build(base):
        if built.get("image_url") == "UPLOAD":
            built["image_url"] = f"{base}/source.png"
        return {**built, "interaction": FakeInteraction()}

    _server, interaction = run_item(cog, monkeypatch, build)
    return posted(interaction)


def click(cog, button: str, message, *, view=None):
    """Click a button on ``message``. Returns ``(interaction, items queued by the click)``."""

    async def scenario():
        blank = view or cog.FluxRemixView(prompt="", width=1024, height=1024, seed=0)
        interaction = FakeInteraction(message, responded=False)
        before = len(cog.bot.sd_queue.items)
        await getattr(blank, button).callback(interaction)
        return interaction, cog.bot.sd_queue.items[before:]

    return asyncio.run(scenario())


def run_queued(cog, monkeypatch, item, **kwargs):
    return run_item(cog, monkeypatch, lambda _base: item, **kwargs)


# ---- a picture made from another picture --------------------------------------


def test_remix_on_an_edited_picture_redoes_the_edit(flux, monkeypatch):
    """It used to run the instruction as a text prompt, which draws something unrelated."""
    edited = generate(flux, monkeypatch, description="make it a theme park", image_url="UPLOAD", strength=None)

    interaction, (item,) = click(flux, "remix_button", as_message(edited))
    assert interaction.response.messages[0].startswith("🛠️ Redoing the change with a new seed...")
    assert item["prompt"] == "make it a theme park" and item["strength"] is None and item["seed"] != 7
    assert item["source_image"].startswith("flux_sources/")

    server, interaction = run_queued(flux, monkeypatch, item)
    ((path, fields),) = server.calls
    assert path == "/flux_edit" and fields["image"] == SOURCE and fields["prompt"] == "make it a theme park"
    # The redo starts from the same source, so it can be redone again.
    assert flux.archived[-1]["extra"]["source_image"] == item["source_image"]


def test_remix_on_a_blended_picture_keeps_its_strength(flux, monkeypatch):
    blended = generate(flux, monkeypatch, description="a theme park at night", image_url="UPLOAD", strength=0.6)

    _, (item,) = click(flux, "remix_button", as_message(blended))
    assert item["strength"] == 0.6
    server, _ = run_queued(flux, monkeypatch, item)
    ((path, fields),) = server.calls
    assert path == "/flux_img2img" and fields["strength"] == "0.6" and fields["image"] == SOURCE


def test_an_edited_picture_has_no_fancy_or_shape_buttons(flux, monkeypatch):
    edited = generate(flux, monkeypatch, description="add an alien", image_url="UPLOAD", strength=None)
    assert labels(edited["view"]) == ["✏️ Edit prompt", "🎲 Remix", "🖼️ Change image", "🎨 R-Fancy", "🔤 R-Keyword"]

    # Edits posted before this rule still show them. They explain instead of drawing nonsense.
    for button in ("fancy_button", "wide_button", "tall_button", "square_button"):
        interaction, queued = click(flux, button, as_message(edited))
        assert queued == []
        assert "made by changing another one" in interaction.response.messages[0]


def test_edit_prompt_on_an_edited_picture_redoes_it_with_a_new_instruction(flux, monkeypatch):
    edited = generate(flux, monkeypatch, description="add an alien", image_url="UPLOAD", strength=None)

    interaction, _ = click(flux, "edit_button", as_message(edited))
    modal = interaction.response.modal
    assert modal.title == "✏️ Redo from the previous image" and modal.shape_select is None
    assert modal.image_description.default == "add an alien"

    async def submit():
        modal.image_description._value = "add a robot"
        modal.seed_input._value = "99"
        again = FakeInteraction(responded=False)
        await modal.on_submit(again)
        return flux.bot.sd_queue.items[-1]

    item = asyncio.run(submit())
    assert (item["prompt"], item["seed"], item["strength"]) == ("add a robot", 99, None)
    server, _ = run_queued(flux, monkeypatch, item)
    ((path, fields),) = server.calls
    assert path == "/flux_edit" and fields["image"] == SOURCE and fields["prompt"] == "add a robot"


def test_changing_an_edited_picture_starts_from_that_picture(flux, monkeypatch, tmp_path):
    """A is edited into B. Changing B gives C from B, never from A again."""
    b = generate(flux, monkeypatch, description="make it a theme park", image_url="UPLOAD", strength=None)
    picture_b = (tmp_path / "images" / b["filename"]).read_bytes()
    assert picture_b != SOURCE

    interaction, _ = click(flux, "img2img_button", as_message(b))
    modal = interaction.response.modal
    assert modal.source_image == f"images/{b['filename']}"

    async def submit():
        modal.prompt_input._value = "add fireworks"
        modal.strength_input._value = ""
        await modal.on_submit(FakeInteraction(responded=False))
        return flux.bot.sd_queue.items[-1]

    item = asyncio.run(submit())
    server, interaction = run_item(flux, monkeypatch, lambda base: {**item, "image_url": f"{base}/source.png"})
    ((path, fields),) = server.calls
    assert path == "/flux_edit" and fields["image"] == picture_b and fields["prompt"] == "add fireworks"
    # C remembers B, so redoing C starts from B as well.
    assert flux.archived[-1]["extra"]["source_image"] == f"images/{b['filename']}"

    # With B deleted from the archive, the attachment under the button is still B.
    (tmp_path / "images" / b["filename"]).unlink()
    interaction, _ = click(flux, "img2img_button", as_message(b, url="https://cdn.example/b.png"))
    assert (interaction.response.modal.source_image, interaction.response.modal.source_url) == (
        None,
        "https://cdn.example/b.png",
    )


def test_an_edit_from_before_sources_were_recorded_says_so(flux, tmp_path):
    (tmp_path / "images").mkdir()
    row = {"filename": "111111_makeitpixar.png", "prompt": "make it pixar", "seed": 5, "source": "flux-edit"}
    (tmp_path / "images" / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    message = FakeMessage(
        "https://cdn.example/x.png",
        "111111_makeitpixar.png",
        embeds=("**Prompt:** make it pixar", "🌱 5 ⚡ Edit ⏱️ 80.00s 📋 1"),
        width=1392,
        height=752,
    )
    for button in ("remix_button", "edit_button"):
        interaction, queued = click(flux, button, message)
        assert queued == [] and interaction.response.modal is None
        assert "can't be redone" in interaction.response.messages[0]


def test_a_live_view_knows_its_recipe_without_the_archive(flux, monkeypatch, tmp_path):
    """Archiving is best-effort. A click before the next restart still has the view's own fields."""
    edited = generate(flux, monkeypatch, description="add an alien", image_url="UPLOAD", strength=None)
    (tmp_path / "images" / "index.jsonl").unlink()

    _, (item,) = click(flux, "remix_button", as_message(edited), view=edited["view"])
    assert item["source_image"].startswith("flux_sources/") and item["strength"] is None


# ---- Fancy, Remix and the shape buttons on a text-to-image picture -------------


@pytest.fixture
def fancy_calls(flux, monkeypatch):
    """Replace the LLM rewrite with one that is easy to recognise. Lists what it was asked to rewrite."""
    calls = []

    async def rewrite(prompt):
        calls.append(prompt)
        return f"FANCY {len(calls)}: {prompt}", 1.5

    monkeypatch.setattr(sys.modules["soupy.cogs.sd"], "build_flux_fancy_prompt", rewrite)
    return calls


def test_fancy_shows_the_typed_prompt_and_rewrites_from_it_again(flux, monkeypatch, fancy_calls):
    plain = generate(flux, monkeypatch, description="a cat", size="wide")
    _, (item,) = click(flux, "fancy_button", as_message(plain))
    _, interaction = run_queued(flux, monkeypatch, item)
    first = posted(interaction)
    assert first["prompt_embed"] == "**Original:** a cat\n\n**Prompt:** FANCY 1: a cat"
    assert flux.archived[-1]["extra"]["original_prompt"] == "a cat"
    assert "⚡ Fancy" in first["details"]

    # A second Fancy used to rewrite the first rewrite.
    _, (item,) = click(flux, "fancy_button", as_message(first))
    _, interaction = run_queued(flux, monkeypatch, item)
    assert fancy_calls == ["a cat", "a cat"]
    assert posted(interaction)["prompt_embed"] == "**Original:** a cat\n\n**Prompt:** FANCY 2: a cat"


def test_the_original_follows_a_fancy_prompt_through_remix_and_reshape(flux, monkeypatch, fancy_calls):
    _, (item,) = click(flux, "fancy_button", as_message(generate(flux, monkeypatch, description="a cat", size="wide")))
    _, interaction = run_queued(flux, monkeypatch, item)
    fancy = posted(interaction)

    for button in ("remix_button", "tall_button"):
        _, (item,) = click(flux, button, as_message(fancy))
        assert (item["prompt"], item["original"]) == ("FANCY 1: a cat", "a cat")
        _, interaction = run_queued(flux, monkeypatch, item)
        assert posted(interaction)["prompt_embed"].startswith("**Original:** a cat\n\n")


def test_the_original_is_read_from_the_message_when_the_archive_row_is_gone(flux, monkeypatch, tmp_path, fancy_calls):
    _, (item,) = click(flux, "fancy_button", as_message(generate(flux, monkeypatch, description="a cat", size="wide")))
    _, interaction = run_queued(flux, monkeypatch, item)
    fancy = posted(interaction)
    (tmp_path / "images" / "index.jsonl").unlink()

    _, (item,) = click(flux, "fancy_button", as_message(fancy))
    assert item["original"] == "a cat"


@pytest.mark.parametrize("button", ["remix_button", "fancy_button"])
def test_remix_and_fancy_keep_the_clicked_pictures_shape(flux, monkeypatch, button):
    """Both used to come out wide whatever was clicked."""
    tall = generate(flux, monkeypatch, description="a lighthouse", size="tall")
    _, (item,) = click(flux, button, as_message(tall, width=TALL[0], height=TALL[1]))
    assert (item["width"], item["height"]) == TALL


def test_shape_buttons_keep_the_seed_and_random_ones_stay_wide(flux, monkeypatch):
    plain = generate(flux, monkeypatch, description="a lighthouse", size="tall")
    message = as_message(plain, width=TALL[0], height=TALL[1])

    _, (item,) = click(flux, "wide_button", message)
    assert (item["width"], item["height"], item["seed"], item["action_name"]) == (*WIDE, 7, "Wide")
    _, (item,) = click(flux, "square_button", message)
    assert (item["width"], item["height"], item["seed"]) == (1024, 1024, 7)
    for button in ("random_fancy_button", "random_keyword_button"):
        _, (item,) = click(flux, button, message)
        assert (item["action"], item["width"], item["height"]) == ("random", *WIDE)


# ---- /flux fancy:true ------------------------------------------------------------


def flux_command(cog, **options):
    async def scenario():
        interaction = FakeInteraction(responded=False)
        args = {"size": None, "image": None, "strength": None, "seed": None, "fancy": None, **options}
        await cog.flux.callback(interaction, **args)
        return cog.bot.sd_queue.items[-1]

    return asyncio.run(scenario())


def test_flux_command_can_write_the_fancy_prompt_first(flux, monkeypatch, fancy_calls):
    """One generation, where it used to take a plain one and then a click on Fancy."""
    monkeypatch.setitem(flux.settings.__dict__, "flux_server_url", "http://127.0.0.1:9")
    item = flux_command(flux, description="a cat", fancy=True, seed=42)
    assert (item["action"], item["prompt"], item["seed"]) == ("fancy", "a cat", 42)
    assert (item["width"], item["height"]) == WIDE

    _, interaction = run_queued(flux, monkeypatch, item)
    assert posted(interaction)["prompt_embed"] == "**Original:** a cat\n\n**Prompt:** FANCY 1: a cat"
    assert interaction.deleted_original == 0  # a slash command's reply is the picture itself


def test_flux_command_ignores_fancy_with_an_image(flux, monkeypatch):
    monkeypatch.setitem(flux.settings.__dict__, "flux_server_url", "http://127.0.0.1:9")
    upload = FakeAttachment("https://cdn.example/photo.jpg", "photo.jpg")
    item = flux_command(flux, description="make it a theme park", image=upload, fancy=True)
    assert "action" not in item and item["image_url"] == "https://cdn.example/photo.jpg"


# ---- the note a click gets ------------------------------------------------------


def test_a_click_is_told_its_place_and_the_note_goes_when_the_picture_arrives(flux, monkeypatch):
    plain = as_message(generate(flux, monkeypatch, description="a cat", size="wide"))

    first, (item_one,) = click(flux, "remix_button", plain)
    second, (item_two,) = click(flux, "remix_button", plain)
    assert first.response.messages == ["🛠️ Remixing..."]
    assert second.response.messages == ["🛠️ Remixing... 1 ahead of you."]

    _, done = run_queued(flux, monkeypatch, item_one)
    assert done is first and first.deleted_original == 1 and second.deleted_original == 0
    # The second click is now next, and a third only has it ahead.
    third, _ = click(flux, "remix_button", plain)
    assert third.response.messages == ["🛠️ Remixing... 1 ahead of you."]


def test_the_note_also_goes_when_the_server_fails_and_the_reason_is_shown(flux, monkeypatch):
    plain = as_message(generate(flux, monkeypatch, description="a cat", size="wide"))
    interaction, (item,) = click(flux, "remix_button", plain)

    failure = (500, {"detail": "[metal] Attempting to allocate 21 GB, more than the maximum allowed buffer size"})
    server, _ = run_queued(flux, monkeypatch, item, failure=failure)
    ((message, kwargs),) = interaction.followup.sent
    assert message.startswith("❌ Flux server error (HTTP 500): [metal] Attempting to allocate 21 GB")
    assert kwargs.get("ephemeral") is True and interaction.deleted_original == 1
    assert flux._workload.running is None and flux._workload.pending == []


def test_the_count_ahead_includes_the_running_job(flux):
    load = flux._Workload()
    first, ahead = load.add()
    assert ahead == 0
    load.start(first)
    assert load.add()[1] == 1
    assert load.add()[1] == 2
    load.finish(first)
    assert load.add()[1] == 2


# ---- what the archive keeps -----------------------------------------------------


def test_archive_row_records_what_made_the_picture(flux, monkeypatch, tmp_path):
    generate(flux, monkeypatch, description="a cat", size="wide")
    generate(flux, monkeypatch, description="add an alien", image_url="UPLOAD", strength=None)
    generate(flux, monkeypatch, description="a cat at night", image_url="UPLOAD", strength=0.6)
    text, edit, blend = [json.loads(line) for line in (tmp_path / "images" / "index.jsonl").read_text().splitlines()]

    assert (text["source"], text["action"], text["model"], text["steps"], text["guidance"]) == (
        "flux",
        "Flux",
        "test-model",
        4,
        1.0,
    )
    assert "strength" not in text and "source_image" not in text and "original_prompt" not in text
    assert text["seconds"] >= 0

    assert (edit["source"], edit["action"], edit["steps"]) == ("flux-edit", "Edit", 6)
    assert edit["source_image"].startswith("flux_sources/") and "strength" not in edit

    assert (blend["source"], blend["strength"], blend["source_image"]) == ("flux-img2img", 0.6, edit["source_image"])


def test_a_filename_survives_discord_whatever_the_prompt_says(flux, monkeypatch):
    """A click finds the archive row by the attachment's name, so the name has to come back unchanged."""
    edited = generate(flux, monkeypatch, description="añade un gato 猫 café!", image_url="UPLOAD", strength=None)
    assert edited["filename"].isascii() and edited["filename"].endswith("_aadeungatocaf.png")
    _, (item,) = click(flux, "remix_button", as_message(edited))
    assert item["source_image"].startswith("flux_sources/")


def test_an_archived_picture_is_found_by_its_filename(flux, tmp_path):
    sd = sys.modules["soupy.cogs.sd"]
    (tmp_path / "images").mkdir()
    rows = [
        {"filename": "1_a.png", "prompt": "first"},
        {"filename": "2_b.png", "prompt": 'mentions "1_a.png" in passing'},
        "not json at all",
        {"filename": "1_a.png", "prompt": "the same name, written later"},
    ]
    text = "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows)
    (tmp_path / "images" / "index.jsonl").write_text(text + "\n", encoding="utf-8")

    assert sd.find_archived_image("1_a.png")["prompt"] == "the same name, written later"
    assert sd.find_archived_image("2_b.png")["prompt"].startswith("mentions")
    assert sd.find_archived_image("3_c.png") is None and sd.find_archived_image("") is None


# ---- labels and dialogs -----------------------------------------------------------


def test_every_button_says_what_it_does(flux):
    async def build():
        return flux.FluxRemixView(prompt="a cat", width=1392, height=752, seed=7)

    view = asyncio.run(build())
    assert labels(view) == [
        "✏️ Edit prompt",
        "🪄 Fancy",
        "🎲 Remix",
        "🖼️ Change image",
        "🎨 R-Fancy",
        "🔤 R-Keyword",
        "↔️ Wide",
        "↕️ Tall",
        "🟦 Square",
    ]
    assert all(any(ch.isascii() and ch.isalpha() for ch in label) for label in labels(view))
    assert all(child.style.name != "danger" for child in view.children)  # red reads as "delete"
    # Buttons on every picture already posted are matched by these. They must not change.
    assert [child.custom_id for child in view.children] == [
        "fluxgen_edit_button",
        "fluxgen_fancy_button",
        "fluxgen_remix_button",
        "fluxgen_img2img_button",
        "fluxgen_rfancy_button",
        "fluxgen_rkeyword_button",
        "fluxgen_wide_button",
        "fluxgen_tall_button",
        "fluxgen_square_button",
    ]


def edit_dialog(cog, *, width, height, **kwargs):
    async def build():
        return cog.FluxEditModal("a cat", width, height, 7, **kwargs)

    return asyncio.run(build())


def test_edit_prompt_dialog_offers_shapes_in_place_of_typed_sizes(flux):
    modal = edit_dialog(flux, width=752, height=1392)
    prompt, shape, seed = modal.to_dict()["components"]
    assert [row["type"] for row in (prompt, shape, seed)] == [18, 18, 18]  # Discord's "label" wrapper
    assert (prompt["component"]["type"], shape["component"]["type"], seed["component"]["type"]) == (4, 3, 4)
    options = shape["component"]["options"]
    assert [(o["value"], o["default"]) for o in options] == [("wide", False), ("tall", True), ("square", False)]
    assert options[0]["label"] == "Wide (1392×752)"

    # A size that is none of the three (older pictures were rounded up to 1408x768) can be kept.
    odd = edit_dialog(flux, width=1408, height=768).to_dict()["components"][1]["component"]["options"]
    assert [(o["value"], o["default"]) for o in odd][0] == ("keep", True) and odd[0][
        "label"
    ] == "Same as now (1408×768)"


@pytest.mark.parametrize(
    "chosen, size",
    [(["wide"], WIDE), (["tall"], TALL), (["square"], (1024, 1024)), (["keep"], (1408, 768)), ([], (1408, 768))],
)
def test_edit_prompt_dialog_runs_the_chosen_shape(flux, chosen, size):
    async def submit():
        modal = flux.FluxEditModal("a cat", 1408, 768, 7, original="cat")
        modal.shape_select._values = chosen
        modal.image_description._value = "a cat"
        modal.seed_input._value = ""
        interaction = FakeInteraction(responded=False)
        await modal.on_submit(interaction)
        return interaction, flux.bot.sd_queue.items[-1]

    interaction, item = asyncio.run(submit())
    assert (item["width"], item["height"]) == size
    assert (item["action_name"], item["original"]) == ("Reprompt", "cat") and item["seed"] != 7  # blank seed: random
    assert interaction.response.messages[0].startswith("🛠️ Generating...")


def test_edit_prompt_dialog_drops_the_original_when_the_prompt_changes(flux):
    async def submit():
        modal = flux.FluxEditModal("FANCY: a cat", 1392, 752, 7, original="a cat")
        modal.shape_select._values = ["wide"]
        modal.image_description._value = "a dog"
        modal.seed_input._value = "7"
        await modal.on_submit(FakeInteraction(responded=False))
        return flux.bot.sd_queue.items[-1]

    item = asyncio.run(submit())
    assert (item["prompt"], item["original"], item["seed"]) == ("a dog", None, 7)


def test_dialogs_fit_discords_limits(flux):
    for modal in (
        edit_dialog(flux, width=1408, height=768),
        edit_dialog(flux, width=1392, height=752, from_image=True, source_image="images/x.png"),
        edit_dialog(flux, width=1392, height=752, from_image=True, source_image="images/x.png", strength=0.5),
    ):
        data = modal.to_dict()
        assert len(data["title"]) <= 45 and len(data["components"]) <= 5
        for row in data["components"]:
            assert 1 <= len(row["label"]) <= 45 and len(row.get("description") or "") <= 100
            for option in row["component"].get("options", []):
                assert 1 <= len(option["label"]) <= 100


# ---- FLUX_ENABLED -----------------------------------------------------------------


def test_with_flux_off_the_command_is_not_registered_and_old_buttons_say_so(flux, monkeypatch):
    monkeypatch.setitem(flux.settings.__dict__, "flux_enabled", False)

    class Tree:
        def __init__(self):
            self.added = []

        def add_command(self, command):
            self.added.append(command)

    class Bot:
        def __init__(self):
            self.tree = Tree()
            self.views = []

        def add_view(self, view):
            self.views.append(view)

    bot = Bot()

    async def scenario():
        await flux.setup(bot)
        interaction = FakeInteraction(responded=False)
        return await bot.views[0].interaction_check(interaction), interaction

    allowed, interaction = asyncio.run(scenario())
    assert bot.tree.added == [] and len(bot.views) == 1
    assert allowed is False and "turned off" in interaction.response.messages[0]


# ---- sizes ------------------------------------------------------------------------


SIZES = [
    (1392, 752),
    (752, 1392),
    (1024, 1024),
    (4032, 3024),
    (3024, 4032),
    (4000, 256),
    (640, 480),
    (1408, 768),
    (17, 9),
]


@pytest.mark.parametrize("size", SIZES)
def test_a_source_size_is_fitted_to_what_klein_can_run(flux, size):
    width, height = flux._clamp_src_dims(*size)
    assert width % 16 == 0 and height % 16 == 0 and width >= 16 and height >= 16
    assert width * height <= 1024 * 1024 and max(width, height) <= 1536
    if min(size) >= 256:
        # Same shape, to within the rounding to 16.
        assert abs(width / height - size[0] / size[1]) < 0.08 * size[0] / size[1]


def test_sizes_that_already_fit_are_left_alone(flux):
    for size in ((1392, 752), (752, 1392), (1024, 1024), (640, 480)):
        assert flux._clamp_src_dims(*size) == size
    assert flux._clamp_src_dims(4032, 3024) == (1168, 880)
    assert flux._clamp_src_dims(0, 500) == (1024, 1024)


@pytest.mark.parametrize("size", SIZES)
def test_the_bot_and_the_server_fit_sizes_the_same_way(flux, size):
    """The bot shows and archives the size it expects; the server decides. They have to agree."""
    server = import_flux_server()
    assert flux._clamp_src_dims(*size) == server._fit_within(*size, max_pixels=1024 * 1024, max_dim=1536)


# ---- reading a posted result back --------------------------------------------------


def test_view_state_is_read_back_from_a_posted_result():
    sd = sys.modules["soupy.cogs.sd"]
    read = sd._view_state_from_message

    plain = FakeMessage(
        "u", "f.png", embeds=("**Prompt:** a cat\nin a hat", "🌱 4242 ⚡ Flux ⏱️ 34.10s 📋 1"), width=1392, height=752
    )
    assert read(plain) == {"prompt": "a cat\nin a hat", "seed": 4242, "width": 1392, "height": 752, "original": None}

    fancy = FakeMessage("u", "f.png", embeds=("**Original:** a cat\n\n**Prompt:** A sleek tabby", "🌱 1 ⚡ Fancy"))
    assert (read(fancy)["prompt"], read(fancy)["original"]) == ("A sleek tabby", "a cat")

    random = FakeMessage("u", "f.png", embeds=("**Selected Terms:** cat, hat\n\n**Prompt:** A cat in a hat", "🌱 1"))
    assert (read(random)["prompt"], read(random)["original"]) == ("A cat in a hat", None)

    outpaint = FakeMessage("u", "f.png", embeds=("**Prompt:** a beach\n**Direction:** both", "🌱 9"))
    assert read(outpaint)["prompt"] == "a beach"

    bare = FakeMessage("u", "f.png")
    assert read(bare) == {"prompt": None, "seed": None, "width": None, "height": None, "original": None}
    assert read(None)["prompt"] is None
