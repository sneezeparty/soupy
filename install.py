#!/usr/bin/env python3
"""Soupy interactive installer.

Run `python3 install.py` from a fresh clone (`python install.py` on
Windows). The wizard validates inputs as it goes, generates `.env-stable`,
and offers to hand off to `run_all.py`. Importable on a clean machine —
only stdlib at module level (plus optional `colorama` for colour, with a
graceful fallback).

Flags:
  --dry-run         walk the prompts without creating venvs / installing
                    packages / writing files.
  --resume          skip steps already completed in `.install-state.json`.
  --minimal         skip step 5 entirely (no Bluesky / SD / daily posts);
                    still prompts for Discord and LM Studio.
  --locked          install the pinned versions in `requirements.lock`
                    instead of resolving `requirements.txt`.
  --non-interactive --config <path>
                    read every answer from a JSON file instead of stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from installer import state as state_mod
from installer.env_writer import parse_existing
from installer.steps import STEPS
from installer.steps.s06_write_config import WIZARD_KEYS
from installer.ui import UI, NonInteractiveUI, WizardAbort, wizard_command

REPO_ROOT = Path(__file__).resolve().parent
STATE_PATH = REPO_ROOT / ".install-state.json"
EXAMPLE_PATH = REPO_ROOT / ".env-stable.example"
TARGET_PATH = REPO_ROOT / ".env-stable"

MIN_PYTHON = (3, 10)

# Secrets are scrubbed from the state file, so a resumed run has to get them
# back: from the `.env-stable` an earlier session wrote, or else by running
# the step that asks for them again.
SECRET_STEPS = {"DISCORD_TOKEN": "discord", "BLUESKY_APP_PASSWORD": "optional"}

# Derived from the step-1 goals on every run. An old value must not ride
# along in state, or a feature just declined would still look switched on.
GOAL_SWITCHES = ("ENABLE_VISION", "DAILY_POST_ENABLED", "FLUX_ENABLED")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="install.py",
        description="Interactive installer for Soupy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="walk through prompts but don't create venvs, install packages, or write .env-stable",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="read .install-state.json and skip completed steps",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="skip optional integrations (step 5)",
    )
    parser.add_argument(
        "--locked",
        action="store_true",
        help="install the pinned requirements.lock instead of requirements.txt",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="read every answer from --config (JSON) instead of stdin",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="path to a JSON file of answers for --non-interactive mode",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete .install-state.json before starting",
    )
    return parser.parse_args(argv)


def _existing_install_action(ui: UI) -> str:
    ui.info(f"{TARGET_PATH} already exists.")
    return ui.prompt_choice(
        "What would you like to do?",
        [
            ("reconfigure", "reconfigure (re-run the wizard; settings it doesn't ask about are kept)"),
            ("verify", "verify-only (skip prompts, just run the verification step)"),
            ("cancel", "cancel"),
        ],
        default="cancel",
    )


def _build_ui(args: argparse.Namespace) -> UI:
    if args.non_interactive:
        if not args.config or not args.config.exists():
            print("--non-interactive requires --config <existing-path>", file=sys.stderr)
            sys.exit(2)
        try:
            answers = json.loads(args.config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"could not parse {args.config}: {e}", file=sys.stderr)
            sys.exit(2)
        if not isinstance(answers, dict):
            print(f"{args.config}: expected a JSON object at the top level", file=sys.stderr)
            sys.exit(2)
        return NonInteractiveUI(answers)
    return UI()


def _apply_minimal_defaults(state: Dict[str, Any]) -> None:
    state.setdefault("image_gen_mode", "none")
    state.setdefault("bluesky", False)
    state.setdefault("vision", False)
    state.setdefault("daily_posts", False)


def _select_step_ids(args: argparse.Namespace, state: Dict[str, Any]) -> List[str]:
    """Return the ordered list of step IDs we'll actually run this session."""
    step_ids = [sid for sid, _label, _fn in STEPS]
    if args.minimal:
        step_ids = [sid for sid in step_ids if sid != "optional"]
    return step_ids


def _is_on(value: Any) -> bool:
    return str(value or "").strip().lower() == "true"


def _answers_from_existing(*, for_verify: bool) -> Dict[str, Any]:
    """What an existing `.env-stable` already answers, as wizard state.

    Verify-only takes the file as it stands. A reconfigure uses these as
    prompt defaults, so it leaves out any value still equal to the
    template's: those are placeholders (`your_discord_bot_token_here`,
    `http://your-sd-host:8000/`) or defaults nobody chose.
    """
    existing = parse_existing(TARGET_PATH)
    template = {} if for_verify else parse_existing(EXAMPLE_PATH)
    answers: Dict[str, Any] = {}
    for key in WIZARD_KEYS:
        value = existing.get(key)
        # PLACEHOLDER in the file itself is the mark of the old --resume bug.
        if not value or value == template.get(key) or value == state_mod.PLACEHOLDER:
            continue
        if key in GOAL_SWITCHES and not for_verify:
            continue
        answers[key] = value
    answers["vision"] = _is_on(existing.get("ENABLE_VISION"))
    answers["daily_posts"] = _is_on(existing.get("DAILY_POST_ENABLED"))
    answers["bluesky"] = bool(answers.get("BLUESKY_HANDLE"))
    if _is_on(existing.get("FLUX_ENABLED")):
        answers["image_gen_mode"] = "local_flux"
    return answers


def _restore_secrets(state: Dict[str, Any]) -> List[str]:
    """Put scrubbed secrets back into a resumed state.

    Returns the IDs of the steps that have to run again because their secret
    could not be recovered from an already-written `.env-stable`.
    """
    scrubbed = [key for key in SECRET_STEPS if state.get(key) == state_mod.PLACEHOLDER]
    if not scrubbed:
        return []
    written = parse_existing(TARGET_PATH) if state.get("__config_written__") else {}
    rerun: List[str] = []
    for key in scrubbed:
        if written.get(key) and written[key] != state_mod.PLACEHOLDER:
            state[key] = written[key]
            continue
        # Drop it so no later step can take the placeholder for an answer.
        del state[key]
        rerun.append(SECRET_STEPS[key])
    return rerun


def _maybe_skip_completed(args: argparse.Namespace, state: Dict[str, Any], step_ids: List[str], ui: UI) -> List[str]:
    if not args.resume:
        return step_ids
    completed = set(state.get("__completed_steps__", []))
    pending = [sid for sid in step_ids if sid not in completed]
    # Only the write and verify steps read the secrets.
    if not {"write_config", "verify"} & set(pending):
        return pending
    rerun = _restore_secrets(state)
    for sid, label, _fn in STEPS:
        if sid in rerun and sid in completed and sid in step_ids:
            ui.info(f"step {label!r} runs again: it collects a secret, and secrets are never saved to disk")
    return [sid for sid in step_ids if sid in pending or sid in rerun]


def _run_steps(
    step_ids: List[str],
    state: Dict[str, Any],
    ui: UI,
    *,
    persist: bool = True,
) -> int:
    """Run the given steps in order. `persist=False` keeps the state file
    untouched, for runs that aren't part of a resumable install."""

    def save() -> None:
        if persist:
            state_mod.save(STATE_PATH, state)

    def resume_hint() -> None:
        if persist:
            ui.info(f"pick up where you left off with:  {wizard_command('--resume')}")

    completed: List[str] = list(state.get("__completed_steps__", []))
    for sid, label, fn in STEPS:
        if sid not in step_ids:
            continue
        try:
            updates = fn(state, ui) or {}
        except SystemExit:
            raise
        except WizardAbort:
            ui.fail(f"aborted at step: {label}")
            save()
            resume_hint()
            return 1
        except KeyboardInterrupt:
            ui.warn(f"interrupted at step: {label}")
            save()
            resume_hint()
            return 130
        except Exception as e:  # noqa: BLE001 — surface any step crash
            ui.fail(f"step {label!r} crashed: {e}")
            save()
            raise
        state.update(updates)
        if sid not in completed:
            completed.append(sid)
        state["__completed_steps__"] = completed
        # Persist after each step so a crash mid-flow is resumable.
        save()
    return 0


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if sys.version_info < MIN_PYTHON:
        # Say so before asking anything; step 2 used to be the first to notice.
        print(
            f"Soupy needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer; this is "
            f"{sys.version_info.major}.{sys.version_info.minor} ({sys.executable}). "
            "Run the wizard with a newer one, e.g. `python3.12 install.py`.",
            file=sys.stderr,
        )
        return 2
    ui = _build_ui(args)

    if args.reset:
        state_mod.clear(STATE_PATH)
        ui.info(f"cleared {STATE_PATH.name}")

    state: Dict[str, Any] = state_mod.load(STATE_PATH) if args.resume else {}
    if state.pop("__dry_run__", False):
        # Left behind by a dry run from before those stopped saving state.
        # Its answers still make good defaults; none of its steps did anything.
        state.pop("__completed_steps__", None)
    if args.dry_run:
        state["__dry_run__"] = True
    if args.locked:
        state["__locked__"] = True
    if args.minimal:
        _apply_minimal_defaults(state)

    ui.header("Soupy installer")
    if args.dry_run:
        ui.info("(dry-run: nothing on disk will change)")
    if args.resume:
        completed = state.get("__completed_steps__", [])
        if completed:
            ui.info(f"resuming — {len(completed)} step(s) already completed: {', '.join(completed)}")

    # Existing install: reconfigure / verify / cancel.
    if TARGET_PATH.exists() and not args.resume and not args.non_interactive:
        action = _existing_install_action(ui)
        if action == "cancel":
            ui.info("nothing changed.")
            return 0
        if action == "verify":
            # Run only the verify step, with the existing .env-stable already in place.
            state.update(_answers_from_existing(for_verify=True))
            return _run_steps(["verify"], state, ui, persist=False)
        # Reconfigure: what the file already says becomes each prompt's default.
        state.update(_answers_from_existing(for_verify=False))

    step_ids = _select_step_ids(args, state)
    step_ids = _maybe_skip_completed(args, state, step_ids, ui)
    if not step_ids:
        ui.info("nothing to do — all steps already completed.")
        ui.info(f"to go through the wizard again:  {wizard_command()}")
        return 0

    # A dry run promises to leave the disk alone, and its "completed" steps
    # did nothing a later --resume could build on.
    return _run_steps(step_ids, state, ui, persist=not args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
