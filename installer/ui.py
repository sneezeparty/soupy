"""Terminal UI helpers.

The interactive `UI` class wraps `print()` / `input()` with colour and
prompting helpers. A `NonInteractiveUI` subclass reads canned answers from
a dict so step functions can be tested or driven from a JSON config without
a TTY.

`colorama` is the one third-party package allowed at install time, but we
fall back gracefully if it isn't present yet (the wizard is supposed to
run on a fresh clone before `pip install` has happened).
"""

from __future__ import annotations

import getpass
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover — colour is cosmetic
    from colorama import Fore, Style, init as _colorama_init

    _colorama_init()
    _COLOUR = True
except Exception:  # noqa: BLE001 — any failure means no colour
    _COLOUR = False

    class _Stub:
        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = _Stub()  # type: ignore[assignment]
    Style = _Stub()  # type: ignore[assignment]


def _c(colour: str, text: str) -> str:
    if not _COLOUR:
        return text
    return f"{colour}{text}{Style.RESET_ALL}"


class WizardAbort(Exception):
    """Raised when the user picks "abort" at a retry/skip/abort prompt."""


class UI:
    """Default interactive UI. Reads from stdin, writes to stdout/stderr."""

    def __init__(self, *, no_colour: bool = False) -> None:
        self._colour = _COLOUR and not no_colour

    # ----- output helpers -------------------------------------------------

    def header(self, title: str) -> None:
        bar = "=" * max(40, len(title) + 4)
        print()
        print(_c(Fore.CYAN, bar))
        print(_c(Fore.CYAN + Style.BRIGHT, f"  {title}"))
        print(_c(Fore.CYAN, bar))

    def info(self, msg: str) -> None:
        print(msg)

    def step(self, msg: str) -> None:
        print(_c(Fore.BLUE, f"-> {msg}"))

    def ok(self, msg: str) -> None:
        print(_c(Fore.GREEN, f"[ok] {msg}"))

    def warn(self, msg: str) -> None:
        print(_c(Fore.YELLOW, f"[warn] {msg}"))

    def fail(self, msg: str) -> None:
        print(_c(Fore.RED, f"[fail] {msg}"), file=sys.stderr)

    def hr(self) -> None:
        print(_c(Fore.WHITE + Style.DIM, "-" * 60))

    # ----- input helpers --------------------------------------------------

    def prompt(
        self,
        label: str,
        *,
        default: Optional[str] = None,
        secret: bool = False,
        allow_blank: bool = False,
    ) -> str:
        """Free-form text prompt. Loops until non-blank unless allow_blank."""
        suffix = f" [{default}]" if default is not None and not secret else ""
        while True:
            try:
                if secret:
                    value = getpass.getpass(f"{label}{suffix}: ")
                else:
                    value = input(f"{label}{suffix}: ")
            except EOFError:
                # Treat EOF (e.g. piped stdin exhausted) as accept-default
                value = ""
            value = value.strip()
            if not value and default is not None:
                return default
            if not value and allow_blank:
                return ""
            if not value:
                self.warn("value required")
                continue
            return value

    def prompt_choice(
        self,
        label: str,
        options: Sequence[Tuple[str, str]],
        *,
        default: Optional[str] = None,
    ) -> str:
        """Numbered single-select.

        `options` is a list of (value, description) tuples. Returns the
        chosen value.
        """
        if not options:
            raise ValueError("prompt_choice requires at least one option")
        print(label)
        default_index: Optional[int] = None
        for i, (value, desc) in enumerate(options, start=1):
            marker = ""
            if default is not None and value == default:
                default_index = i
                marker = " (default)"
            print(f"  {i}. {desc}{marker}")
        suffix = f" [{default_index}]" if default_index else ""
        while True:
            try:
                raw = input(f"choose 1-{len(options)}{suffix}: ").strip()
            except EOFError:
                raw = ""
            if not raw and default_index:
                return options[default_index - 1][0]
            if not raw:
                # No default and no input — bail loudly rather than loop forever
                raise WizardAbort(f"no answer for: {label}")
            try:
                idx = int(raw)
            except ValueError:
                self.warn("enter a number")
                continue
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
            self.warn(f"out of range; pick 1-{len(options)}")

    def confirm(self, label: str, *, default: bool = False) -> bool:
        suffix = " [Y/n]" if default else " [y/N]"
        while True:
            try:
                raw = input(f"{label}{suffix}: ").strip().lower()
            except EOFError:
                raw = ""
            if not raw:
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            self.warn("answer y or n")

    def retry_skip_abort(self, label: str) -> str:
        """Standard prompt after a validation failure. Returns 'retry', 'skip', or 'abort'."""
        return self.prompt_choice(
            label,
            [
                ("retry", "retry"),
                ("skip", "skip and continue"),
                ("abort", "abort"),
            ],
            default="retry",
        )


class NonInteractiveUI(UI):
    """Reads answers from a dict instead of stdin.

    Each call to `prompt` / `prompt_choice` / `confirm` looks up the next
    answer keyed on `label` (or its lowercased prefix). Missing answers
    raise `KeyError` so a misconfigured non-interactive run fails loudly
    rather than hanging on stdin.
    """

    def __init__(self, answers: Dict[str, Any]) -> None:
        super().__init__(no_colour=True)
        self._answers = answers
        self._calls: List[str] = []

    def _lookup(self, label: str) -> Any:
        key = label.strip().rstrip(":?").strip().lower()
        # Exact match wins; otherwise try prefix on the label
        if key in self._answers:
            return self._answers[key]
        for k, v in self._answers.items():
            if key.startswith(k.lower()):
                return v
        raise KeyError(f"non-interactive answers missing key for prompt: {label!r}")

    def prompt(  # type: ignore[override]
        self,
        label: str,
        *,
        default: Optional[str] = None,
        secret: bool = False,
        allow_blank: bool = False,
    ) -> str:
        self._calls.append(label)
        try:
            value = str(self._lookup(label))
        except KeyError:
            if default is not None:
                return default
            if allow_blank:
                return ""
            raise
        return value

    def prompt_choice(  # type: ignore[override]
        self,
        label: str,
        options: Sequence[Tuple[str, str]],
        *,
        default: Optional[str] = None,
    ) -> str:
        self._calls.append(label)
        try:
            value = str(self._lookup(label))
        except KeyError:
            if default is not None:
                return default
            raise
        valid = {v for v, _ in options}
        if value not in valid:
            raise ValueError(
                f"non-interactive answer for {label!r} = {value!r} not in {sorted(valid)}"
            )
        return value

    def confirm(self, label: str, *, default: bool = False) -> bool:  # type: ignore[override]
        self._calls.append(label)
        try:
            value = self._lookup(label)
        except KeyError:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("y", "yes", "true", "1")

    def retry_skip_abort(self, label: str) -> str:  # type: ignore[override]
        # Non-interactive runs never retry — fail loud.
        return "abort"
