"""Tests for the wizard's control flow: prompts, resume, re-runs, dry runs, handoff.

Every test here pins a way the installer used to fail a real install or
re-run. Steps are stubbed or driven through ``NonInteractiveUI``; nothing
touches the network, a real venv, or the repo's own ``.env-stable``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import install
from installer import env_writer
from installer import state as state_mod
from installer.steps import s01_goals, s02_environment, s05_optional, s06_write_config, s08_handoff
from installer.ui import UI, NonInteractiveUI, WizardAbort

FIXTURE = Path(__file__).parent / "fixtures" / "env-stable.example.fixture"

GOALS_OFF = {"image_gen_mode": "none", "bluesky": False, "vision": False, "daily_posts": False}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    """A scratch repo root: the wizard's paths point here instead of the checkout."""
    example = tmp_path / ".env-stable.example"
    example.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    for module in (install, s06_write_config):
        monkeypatch.setattr(module, "EXAMPLE_PATH", example)
        monkeypatch.setattr(module, "TARGET_PATH", tmp_path / ".env-stable")
    monkeypatch.setattr(install, "STATE_PATH", tmp_path / ".install-state.json")
    return tmp_path


@pytest.fixture
def steps_seen(monkeypatch) -> list:
    """Replace every step with a recorder: (step id, state as the step saw it)."""
    seen: list = []

    def recorder(sid: str):
        def step(state, ui):
            seen.append((sid, dict(state)))
            return {}

        return step

    monkeypatch.setattr(install, "STEPS", [(sid, label, recorder(sid)) for sid, label, _fn in install.STEPS])
    return seen


def _answer_with(monkeypatch, *lines: str) -> None:
    """Feed `lines` to input(), then behave like a closed stdin."""
    pending = list(lines)

    def fake_input(_prompt: str = "") -> str:
        if not pending:
            raise EOFError
        return pending.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_blank_enter_at_a_choice_asks_again(monkeypatch):
    # A stray Enter at the model picker used to end the whole wizard.
    _answer_with(monkeypatch, "", "2")
    assert UI().prompt_choice("Chat model id:", [("a", "a"), ("b", "b")]) == "b"


def test_closed_stdin_at_a_choice_aborts(monkeypatch):
    _answer_with(monkeypatch)
    with pytest.raises(WizardAbort):
        UI().prompt_choice("Chat model id:", [("a", "a"), ("b", "b")])


def test_closed_stdin_at_a_choice_takes_the_default(monkeypatch):
    _answer_with(monkeypatch)
    assert UI().prompt_choice("Image generation:", [("a", "a"), ("b", "b")], default="b") == "b"


def test_closed_stdin_at_a_required_prompt_aborts(monkeypatch):
    # This one looped on "value required" forever.
    _answer_with(monkeypatch)
    with pytest.raises(WizardAbort):
        UI().prompt("Primary Discord server (guild) id")


# ---------------------------------------------------------------------------
# --resume and --dry-run
# ---------------------------------------------------------------------------


def _save_progress(completed: list, **extra) -> None:
    state_mod.save(
        install.STATE_PATH,
        {"DISCORD_TOKEN": "real-token", "OWNER_IDS": "1", **GOALS_OFF, "__completed_steps__": completed, **extra},
    )


def test_resume_asks_again_for_a_secret_it_cannot_recover(repo, steps_seen):
    # The state file holds a placeholder where the token was. Resuming past the
    # Discord step used to carry that placeholder into .env-stable as the token.
    _save_progress(["goals", "environment", "discord"])

    assert install.main(["--resume"]) == 0

    ran = [sid for sid, _state in steps_seen]
    assert ran == ["discord", "lm_studio", "optional", "write_config", "verify", "handoff"]
    for _sid, state in steps_seen:
        assert state.get("DISCORD_TOKEN") != state_mod.PLACEHOLDER


def test_resume_reads_a_secret_back_from_the_config_it_wrote(repo, steps_seen):
    install.TARGET_PATH.write_text("DISCORD_TOKEN=real-token\n", encoding="utf-8")
    _save_progress(
        ["goals", "environment", "discord", "lm_studio", "optional", "write_config"],
        __config_written__=True,
    )

    assert install.main(["--resume"]) == 0

    assert [sid for sid, _state in steps_seen] == ["verify", "handoff"]
    assert steps_seen[0][1]["DISCORD_TOKEN"] == "real-token"


def test_resume_does_not_trust_a_placeholder_sitting_in_the_config(repo, steps_seen):
    # A config already damaged by the old bug must not be read back as the token.
    install.TARGET_PATH.write_text(f'DISCORD_TOKEN="{state_mod.PLACEHOLDER}"\n', encoding="utf-8")
    _save_progress(
        ["goals", "environment", "discord", "lm_studio", "optional", "write_config"],
        __config_written__=True,
    )

    assert install.main(["--resume"]) == 0

    assert [sid for sid, _state in steps_seen] == ["discord", "verify", "handoff"]


def test_resume_with_only_the_handoff_left_asks_for_nothing(repo, steps_seen):
    _save_progress(["goals", "environment", "discord", "lm_studio", "optional", "write_config", "verify"])

    assert install.main(["--resume"]) == 0

    assert [sid for sid, _state in steps_seen] == ["handoff"]


def test_placeholder_is_never_written_as_a_value():
    updates = s06_write_config._gather_updates(
        {"DISCORD_TOKEN": state_mod.PLACEHOLDER, "BLUESKY_APP_PASSWORD": state_mod.PLACEHOLDER, "OWNER_IDS": "1"}
    )
    assert "DISCORD_TOKEN" not in updates
    assert "BLUESKY_APP_PASSWORD" not in updates
    assert updates["OWNER_IDS"] == "1"


def test_dry_run_leaves_no_state_behind(repo, steps_seen):
    # A dry run used to record every step as completed, so a following
    # --resume said "nothing to do" with nothing installed.
    assert install.main(["--dry-run"]) == 0
    assert len(steps_seen) == len(install.STEPS)
    assert not install.STATE_PATH.exists()

    steps_seen.clear()
    assert install.main(["--resume"]) == 0
    assert len(steps_seen) == len(install.STEPS)
    assert not steps_seen[0][1].get("__dry_run__")


def test_resume_ignores_the_completed_steps_of_an_old_dry_run(repo, steps_seen):
    state_mod.save(
        install.STATE_PATH,
        {"__dry_run__": True, "__completed_steps__": [sid for sid, _l, _f in install.STEPS], "OWNER_IDS": "1"},
    )

    assert install.main(["--resume"]) == 0

    assert len(steps_seen) == len(install.STEPS)
    assert steps_seen[0][1]["OWNER_IDS"] == "1"


def test_too_old_a_python_is_refused_before_any_question(repo, steps_seen, monkeypatch, capsys):
    monkeypatch.setattr(install, "MIN_PYTHON", (99, 0))
    assert install.main([]) == 2
    assert steps_seen == []
    assert "Python 99.0 or newer" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Re-running over an existing config
# ---------------------------------------------------------------------------

EXISTING = """\
DISCORD_TOKEN=real-token
OWNER_IDS=111111111111111111
GUILD_ID=222222222222222222
MAX_TOKENS=4321
BEHAVIOUR="a persona
tuned by hand"
BLUESKY_HANDLE=yourname.bsky.social
BLUESKY_AUTO_REPLY=true
ENABLE_VISION=true
DAILY_POST_ENABLED=true
DAILY_POST_CHANNELS="{\\"333333333333333333\\": \\"tech news\\"}"
FLUX_ENABLED=true
MY_OWN_SETTING=keep-me
"""


def test_rerun_keeps_what_the_wizard_does_not_ask_about(repo):
    # Rendering from the example reset MAX_TOKENS and the persona to their
    # defaults and dropped hand-added keys.
    install.TARGET_PATH.write_text(EXISTING, encoding="utf-8")

    out = s06_write_config.run({**GOALS_OFF, "DISCORD_TOKEN": "new-token"}, NonInteractiveUI({}))

    assert out == {"__config_written__": True}
    written = env_writer.parse_existing(install.TARGET_PATH)
    assert written["DISCORD_TOKEN"] == "new-token"
    assert written["MAX_TOKENS"] == "4321"
    assert written["BEHAVIOUR"] == "a persona\ntuned by hand"
    assert written["MY_OWN_SETTING"] == "keep-me"
    assert len(list(repo.glob(".env-stable.bak.*"))) == 1


def test_rerun_switches_off_what_was_declined(repo):
    install.TARGET_PATH.write_text(EXISTING, encoding="utf-8")

    s06_write_config.run(dict(GOALS_OFF), NonInteractiveUI({}))

    written = env_writer.parse_existing(install.TARGET_PATH)
    for key in ("ENABLE_VISION", "DAILY_POST_ENABLED", "FLUX_ENABLED", "BLUESKY_AUTO_REPLY"):
        assert written[key] == "false", key


def test_flux_setup_has_the_panel_start_the_server_unless_told_not_to(repo):
    answers = {"flux server bind host": "127.0.0.1", "flux server port": "4942"}
    fresh = s05_optional._local_flux({}, NonInteractiveUI(answers))
    assert (fresh["FLUX_ENABLED"], fresh["FLUX_SERVER_AUTOSTART"]) == ("true", "true")
    # A re-run over a config that says no keeps the no.
    kept = s05_optional._local_flux({"FLUX_SERVER_AUTOSTART": "false"}, NonInteractiveUI(answers))
    assert kept["FLUX_SERVER_AUTOSTART"] == "false"
    declined = s05_optional._local_flux({}, NonInteractiveUI({**answers, "have the web panel start": "n"}))
    assert declined["FLUX_SERVER_AUTOSTART"] == "false"

    install.TARGET_PATH.write_text(EXISTING, encoding="utf-8")
    s06_write_config.run({**GOALS_OFF, "image_gen_mode": "local_flux", **declined}, NonInteractiveUI({}))
    written = env_writer.parse_existing(install.TARGET_PATH)
    assert (written["FLUX_ENABLED"], written["FLUX_SERVER_AUTOSTART"]) == ("true", "false")


def test_rerun_with_nothing_to_change_writes_nothing(repo):
    install.TARGET_PATH.write_text(EXISTING, encoding="utf-8")
    s06_write_config.run(dict(GOALS_OFF), NonInteractiveUI({}))
    after_first = install.TARGET_PATH.read_text(encoding="utf-8")

    out = s06_write_config.run(dict(GOALS_OFF), NonInteractiveUI({}))

    # Still counts as written: a later --resume reads the secrets back from it.
    assert out == {"__config_written__": True}
    assert install.TARGET_PATH.read_text(encoding="utf-8") == after_first
    assert len(list(repo.glob(".env-stable.bak.*"))) == 1


def test_first_install_still_renders_from_the_example(repo):
    s06_write_config.run({**GOALS_OFF, "DISCORD_TOKEN": "new-token"}, NonInteractiveUI({}))

    text = install.TARGET_PATH.read_text(encoding="utf-8")
    assert "DISCORD_TOKEN=new-token" in text
    assert "# Comment that should survive untouched." in text
    assert not list(repo.glob(".env-stable.bak.*"))


def test_reconfigure_starts_from_the_existing_answers(repo, steps_seen, monkeypatch):
    install.TARGET_PATH.write_text(EXISTING, encoding="utf-8")
    _answer_with(monkeypatch, "1")  # "reconfigure" at the existing-install prompt

    assert install.main([]) == 0

    state = steps_seen[0][1]
    assert state["DISCORD_TOKEN"] == "real-token"
    assert state["OWNER_IDS"] == "111111111111111111"
    # Still the template's placeholder, so not an answer.
    assert "BLUESKY_HANDLE" not in state
    assert state["bluesky"] is False
    assert state["vision"] is True
    assert state["daily_posts"] is True
    assert state["image_gen_mode"] == "local_flux"
    assert json.loads(state["DAILY_POST_CHANNELS"]) == {"333333333333333333": "tech news"}
    # The on/off keys are re-derived from the goals; a stale "true" here would
    # make a feature just declined look enabled to the verify step.
    assert "ENABLE_VISION" not in state


def test_verify_only_reads_the_config_and_keeps_the_state_file(repo, steps_seen, monkeypatch):
    install.TARGET_PATH.write_text(EXISTING, encoding="utf-8")
    _answer_with(monkeypatch, "2")  # "verify-only"

    assert install.main([]) == 0

    assert [sid for sid, _state in steps_seen] == ["verify"]
    state = steps_seen[0][1]
    assert state["DISCORD_TOKEN"] == "real-token"
    assert state["ENABLE_VISION"] == "true"
    # Not a wizard key: the whole file used to be copied into the state.
    assert "BEHAVIOUR" not in state
    assert not install.STATE_PATH.exists()


# ---------------------------------------------------------------------------
# Environment step
# ---------------------------------------------------------------------------


def test_requirements_txt_is_installed_unless_the_lock_is_asked_for():
    ui = NonInteractiveUI({})
    assert s02_environment._bot_requirements(False, ui).name == "requirements.txt"
    assert s02_environment._bot_requirements(True, ui).name == "requirements.lock"


def test_a_venv_directory_without_pip_is_built_again(tmp_path, monkeypatch):
    # `python -m venv` on Debian/Ubuntu without python3-venv fails after making
    # the directory. The next run used to see the directory and skip creation.
    calls: list = []
    monkeypatch.setattr(s02_environment.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    venv = tmp_path / ".venv"
    venv.mkdir()

    s02_environment._create_venv(venv, NonInteractiveUI({}))
    assert calls == [[sys.executable, "-m", "venv", str(venv)]]

    pip = s02_environment._venv_pip(venv)
    pip.parent.mkdir(parents=True)
    pip.write_text("")
    s02_environment._create_venv(venv, NonInteractiveUI({}))
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("os_name", "machine", "python", "offered"),
    [
        ("darwin", "arm64", (3, 13), ["none", "local_flux", "local_mps", "remote"]),
        ("darwin", "x86_64", (3, 12), ["none", "remote"]),
        ("linux", "x86_64", (3, 12), ["none", "local_cuda", "remote"]),
        ("win32", "AMD64", (3, 11), ["none", "local_cuda", "remote"]),
        ("linux", "x86_64", (3, 13), ["none", "remote"]),
    ],
)
def test_only_backends_this_machine_can_run_are_offered(os_name, machine, python, offered):
    options, notes = s01_goals.image_gen_options(os_name, machine, python)
    assert [value for value, _label in options] == offered
    # Hidden for the Python version is the one case the user can fix, so it is explained.
    assert bool(notes) == (os_name != "darwin" and python > s01_goals.CUDA_MAX_PYTHON)


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------


def test_handoff_launches_with_the_venv_python(tmp_path, monkeypatch):
    # The wizard runs on the system Python, which on a fresh clone has no
    # uvicorn. Launching run_all.py with it failed for every new user.
    venv = tmp_path / ".venv"
    python = s02_environment.venv_python(venv)
    python.parent.mkdir(parents=True)
    python.write_text("")
    (tmp_path / "run_all.py").write_text("")
    monkeypatch.setattr(s08_handoff, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(s08_handoff, "BOT_VENV", venv)
    launched: dict = {}
    monkeypatch.setattr(s08_handoff.os, "chdir", lambda path: launched.update(cwd=path))
    monkeypatch.setattr(s08_handoff.os, "execvp", lambda file, argv: launched.update(file=file, argv=argv))

    s08_handoff.run({"image_gen_mode": "none"}, NonInteractiveUI({"launch": True}))

    assert launched["file"] == str(python)
    assert launched["argv"] == [str(python), str(tmp_path / "run_all.py")]
    assert launched["file"] != sys.executable
    assert launched["cwd"] == tmp_path


def test_handoff_does_not_launch_when_declined(tmp_path, monkeypatch):
    monkeypatch.setattr(s08_handoff, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(s08_handoff, "BOT_VENV", tmp_path / ".venv")
    monkeypatch.setattr(s08_handoff.os, "execvp", lambda *a: pytest.fail("launched anyway"))

    assert s08_handoff.run({"image_gen_mode": "none"}, NonInteractiveUI({"launch": False})) == {"__done__": True}
