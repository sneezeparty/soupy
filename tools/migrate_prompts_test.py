"""Tests for tools/migrate_prompts.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import migrate_prompts as mp  # noqa: E402


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch):
    """Build a tiny fake repo: prompts/ dir + .env-stable. Patch REPO_ROOT
    so the tool resolves files inside the tmp tree."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "behaviour.default.txt").write_text("DEFAULT BEHAVIOUR", encoding="utf-8")
    (prompts / "nineball.default.txt").write_text("DEFAULT 9BALL", encoding="utf-8")
    (prompts / "fancy.default.txt").write_text("DEFAULT FANCY", encoding="utf-8")
    (prompts / "behaviour_search.default.txt").write_text("DEFAULT SEARCH", encoding="utf-8")
    (prompts / "behaviour_daily_post.default.txt").write_text("DEFAULT DAILY", encoding="utf-8")
    (prompts / "randomprompt.default.txt").write_text("DEFAULT RANDOM", encoding="utf-8")
    (prompts / "sd_negative_prompt.default.txt").write_text("DEFAULT NEG", encoding="utf-8")

    env = tmp_path / ".env-stable"
    monkeypatch.setattr(mp, "REPO_ROOT", tmp_path)
    return tmp_path, prompts, env


def test_dry_run_does_not_modify(fake_repo):
    tmp, prompts, env = fake_repo
    env.write_text(
        'BEHAVIOUR="my custom personality"\n' "9BALL=DEFAULT 9BALL\n" "NOT_A_PROMPT=keep\n",
        encoding="utf-8",
    )
    rc = mp.main(["--env", str(env), "--prompts", str(prompts), "--dry-run"])
    assert rc == 0
    assert 'BEHAVIOUR="my custom personality"' in env.read_text()
    assert not (prompts / "behaviour.txt").exists()


def test_migrates_custom_value_to_file(fake_repo):
    tmp, prompts, env = fake_repo
    env.write_text(
        'BEHAVIOUR="my custom personality"\n' "OWNER_IDS=12345\n",
        encoding="utf-8",
    )
    rc = mp.main(["--env", str(env), "--prompts", str(prompts)])
    assert rc == 0
    assert (prompts / "behaviour.txt").exists()
    assert "my custom personality" in (prompts / "behaviour.txt").read_text()
    # Original env line is commented out
    new_env = env.read_text()
    assert '# BEHAVIOUR="my custom personality"' in new_env
    assert "(migrated to prompts/behaviour.txt" in new_env
    # Other env vars untouched
    assert "OWNER_IDS=12345" in new_env


def test_skips_default_value(fake_repo):
    tmp, prompts, env = fake_repo
    env.write_text(
        "BEHAVIOUR=DEFAULT BEHAVIOUR\n",
        encoding="utf-8",
    )
    rc = mp.main(["--env", str(env), "--prompts", str(prompts)])
    assert rc == 0
    # No file written (would just duplicate the default)
    assert not (prompts / "behaviour.txt").exists()
    # Env line commented out
    assert "# BEHAVIOUR=DEFAULT BEHAVIOUR" in env.read_text()


def test_skips_blank_value(fake_repo):
    tmp, prompts, env = fake_repo
    env.write_text(
        "9BALL=\n" "BEHAVIOUR=\n",
        encoding="utf-8",
    )
    rc = mp.main(["--env", str(env), "--prompts", str(prompts)])
    assert rc == 0
    assert not (prompts / "nineball.txt").exists()


def test_does_not_overwrite_existing_user_file(fake_repo):
    tmp, prompts, env = fake_repo
    (prompts / "behaviour.txt").write_text("USER ALREADY EDITED THIS", encoding="utf-8")
    env.write_text(
        'BEHAVIOUR="env-version"\n',
        encoding="utf-8",
    )
    rc = mp.main(["--env", str(env), "--prompts", str(prompts)])
    assert rc == 0
    # User file untouched
    assert (prompts / "behaviour.txt").read_text() == "USER ALREADY EDITED THIS"
    # Env var line still there (not commented) so user can reconcile
    assert 'BEHAVIOUR="env-version"' in env.read_text()


def test_creates_backup(fake_repo):
    tmp, prompts, env = fake_repo
    env.write_text('BEHAVIOUR="x"\n', encoding="utf-8")
    rc = mp.main(["--env", str(env), "--prompts", str(prompts)])
    assert rc == 0
    backups = list(tmp.glob(".env-stable.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == 'BEHAVIOUR="x"\n'


def test_handles_multiline_quoted_value(fake_repo):
    tmp, prompts, env = fake_repo
    env.write_text(
        'BEHAVIOUR="line one\nline two\nline three"\n',
        encoding="utf-8",
    )
    rc = mp.main(["--env", str(env), "--prompts", str(prompts)])
    assert rc == 0
    written = (prompts / "behaviour.txt").read_text()
    assert "line one" in written
    assert "line two" in written
    assert "line three" in written


def test_idempotent_rerun(fake_repo):
    tmp, prompts, env = fake_repo
    env.write_text('BEHAVIOUR="custom"\n', encoding="utf-8")
    mp.main(["--env", str(env), "--prompts", str(prompts)])
    # Second run: env var is now commented out, no file should be re-created
    snapshot_env = env.read_text()
    snapshot_file = (prompts / "behaviour.txt").read_text()
    rc = mp.main(["--env", str(env), "--prompts", str(prompts)])
    assert rc == 0
    # Env stayed the same; one new backup added
    assert env.read_text() == snapshot_env
    assert (prompts / "behaviour.txt").read_text() == snapshot_file
