"""Step 2 — Environment & dependencies.

- Verify Python ≥ 3.10 (abort otherwise).
- Verify `git` on PATH (informational).
- Create `.venv` if missing and install bot deps from `requirements.txt`.
- If image_gen_mode is local_cuda / local_mps, also build `.venv-sd` with
  the right PyTorch flavour and the SD requirements.

We invoke pip via subprocess and stream stdout line-by-line so the user
sees progress. Sentinel-package checks (`discord.py` for the bot venv,
`diffusers` for the SD venv) let us short-circuit when deps are already
installed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BOT_VENV = REPO_ROOT / ".venv"
SD_VENV = REPO_ROOT / ".venv-sd"

BOT_REQS = REPO_ROOT / "requirements.txt"
SD_REQS = REPO_ROOT / "sd-api" / "requirements.txt"
SD_REQS_MAC = REPO_ROOT / "sd-api" / "requirements-m1-mac.txt"

PYTORCH_CUDA_CMD = [
    "torch==2.4.0+cu118",
    "torchvision==0.19.0+cu118",
    "torchaudio==2.4.0+cu118",
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu118",
]
PYTORCH_MPS_CMD = ["torch", "torchvision", "torchaudio"]


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":  # pragma: no cover — wizard targets POSIX in tests
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_pip(venv: Path) -> Path:
    if sys.platform == "win32":  # pragma: no cover
        return venv / "Scripts" / "pip.exe"
    return venv / "bin" / "pip"


def _python_version_ok() -> bool:
    return sys.version_info >= (3, 10)


def _has_sentinel(venv: Path, sentinel: str) -> bool:
    pip = _venv_pip(venv)
    if not pip.exists():
        return False
    try:
        result = subprocess.run(
            [str(pip), "freeze"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    needle = sentinel.lower()
    for line in result.stdout.splitlines():
        if line.lower().startswith(needle + "==") or line.lower().startswith(needle + " @"):
            return True
    return False


def _create_venv(venv: Path, ui) -> None:
    if venv.exists():
        return
    ui.step(f"creating {venv.name}")
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
    )


def _stream_pip(cmd: List[str], ui) -> int:
    """Run pip with line-by-line output. Returns exit code."""
    ui.info(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        # Trim noisy pip lines but show progress and errors.
        lower = line.lower()
        if "error" in lower or "warning" in lower or line.startswith(("Successfully", "Installing", "Collecting", "Downloading", "Building", "Preparing")):
            ui.info(f"    {line}")
    proc.wait()
    return proc.returncode


def _install(venv: Path, args: Iterable[str], ui) -> None:
    pip = _venv_pip(venv)
    if not pip.exists():
        raise RuntimeError(f"missing pip in {venv}")
    cmd = [str(pip), "install", *args]
    rc = _stream_pip(cmd, ui)
    if rc != 0:
        raise RuntimeError(f"pip install failed (exit {rc}): {' '.join(args)}")


def _setup_bot_venv(ui, dry_run: bool) -> None:
    if _has_sentinel(BOT_VENV, "discord.py"):
        ui.ok(f"{BOT_VENV.name} already has bot deps installed")
        return
    if dry_run:
        ui.info(f"[dry-run] would create {BOT_VENV.name} and install {BOT_REQS}")
        return
    _create_venv(BOT_VENV, ui)
    _install(BOT_VENV, ["-r", str(BOT_REQS)], ui)
    ui.ok("bot deps installed")


def _setup_sd_venv(mode: str, ui, dry_run: bool) -> None:
    if mode not in ("local_cuda", "local_mps"):
        return
    if _has_sentinel(SD_VENV, "diffusers"):
        ui.ok(f"{SD_VENV.name} already has SD deps installed")
        return
    if dry_run:
        ui.info(f"[dry-run] would create {SD_VENV.name} for {mode}")
        return
    _create_venv(SD_VENV, ui)
    if mode == "local_cuda":
        ui.step("installing PyTorch (CUDA 11.8)")
        _install(SD_VENV, PYTORCH_CUDA_CMD, ui)
        _install(SD_VENV, ["-r", str(SD_REQS)], ui)
    else:  # local_mps
        ui.step("installing PyTorch (MPS)")
        _install(SD_VENV, PYTORCH_MPS_CMD, ui)
        _install(SD_VENV, ["-r", str(SD_REQS_MAC)], ui)
    ui.ok("SD deps installed")


def run(state: Dict, ui) -> Dict:
    ui.header("Step 2 — Environment & dependencies")

    if not _python_version_ok():
        ui.fail(
            f"Python 3.10+ required (this is {sys.version_info.major}.{sys.version_info.minor})."
        )
        raise SystemExit(2)
    ui.ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    if shutil.which("git") is None:
        ui.warn("`git` not found on PATH — fine for the install itself, but updating later will be manual")
    else:
        ui.ok("git on PATH")

    dry_run = bool(state.get("__dry_run__"))

    _setup_bot_venv(ui, dry_run)
    _setup_sd_venv(state.get("image_gen_mode", "none"), ui, dry_run)

    return {"venv_ready": True}
