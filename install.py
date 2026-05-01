#!/usr/bin/env python3
"""Soupy interactive installer.

Run `python install.py` from a fresh clone. The wizard validates inputs
as it goes, generates `.env-stable`, and offers to hand off to
`run_all.py`. Importable on a clean machine — only stdlib at module
level (plus optional `colorama` for colour, with a graceful fallback).

Flags:
  --dry-run         walk the prompts without creating venvs / installing
                    packages / writing files.
  --resume          skip steps already completed in `.install-state.json`.
  --minimal         skip step 5 entirely (no Bluesky / SD / daily posts);
                    still prompts for Discord and LM Studio.
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
from installer.ui import NonInteractiveUI, UI, WizardAbort
from installer.steps import STEPS


REPO_ROOT = Path(__file__).resolve().parent
STATE_PATH = REPO_ROOT / ".install-state.json"
EXAMPLE_PATH = REPO_ROOT / ".env-stable.example"
TARGET_PATH = REPO_ROOT / ".env-stable"


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
            ("reconfigure", "reconfigure (re-run the wizard, overwrite .env-stable)"),
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


def _maybe_skip_completed(
    args: argparse.Namespace, state: Dict[str, Any], step_ids: List[str]
) -> List[str]:
    if not args.resume:
        return step_ids
    completed = set(state.get("__completed_steps__", []))
    out = [sid for sid in step_ids if sid not in completed]
    return out


def _run_steps(
    step_ids: List[str],
    state: Dict[str, Any],
    ui: UI,
) -> int:
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
            state_mod.save(STATE_PATH, state)
            return 1
        except KeyboardInterrupt:
            ui.warn(f"interrupted at step: {label}")
            state_mod.save(STATE_PATH, state)
            return 130
        except Exception as e:  # noqa: BLE001 — surface any step crash
            ui.fail(f"step {label!r} crashed: {e}")
            state_mod.save(STATE_PATH, state)
            raise
        state.update(updates)
        if sid not in completed:
            completed.append(sid)
        state["__completed_steps__"] = completed
        # Persist after each step so a crash mid-flow is resumable.
        state_mod.save(STATE_PATH, state)
    return 0


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    ui = _build_ui(args)

    if args.reset:
        state_mod.clear(STATE_PATH)
        ui.info(f"cleared {STATE_PATH.name}")

    state: Dict[str, Any] = state_mod.load(STATE_PATH) if args.resume else {}
    if args.dry_run:
        state["__dry_run__"] = True
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
    if (
        TARGET_PATH.exists()
        and not args.resume
        and not args.non_interactive
        and not state.get("__completed_steps__")
    ):
        action = _existing_install_action(ui)
        if action == "cancel":
            ui.info("nothing changed.")
            return 0
        if action == "verify":
            # Run only the verify step, with the existing .env-stable already in place.
            try:
                from installer.env_writer import parse_existing
            except Exception:  # noqa: BLE001
                parse_existing = None  # type: ignore[assignment]
            if parse_existing is not None:
                state.update(parse_existing(TARGET_PATH))
            return _run_steps(["verify"], state, ui)

    step_ids = _select_step_ids(args, state)
    step_ids = _maybe_skip_completed(args, state, step_ids)
    if not step_ids:
        ui.info("nothing to do — all steps already completed.")
        return 0

    return _run_steps(step_ids, state, ui)


if __name__ == "__main__":
    raise SystemExit(main())
