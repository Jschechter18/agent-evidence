from __future__ import annotations

from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Literal


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def build_run_id(timestamp: datetime, git_commit: str) -> str:
    """Build a run identifier from a UTC timestamp and Git commit."""
    normalized_commit = git_commit.strip()
    short_commit = (
        normalized_commit[:12]
        if normalized_commit and normalized_commit != "unknown"
        else "unknown"
    )

    return f"{timestamp:%Y%m%dT%H%M%S%fZ}-{short_commit}"


def get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_package_version(package: str) -> str:
    """Return the installed version of a package when available."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def get_run_command() -> str:
    """Return a shell-safe representation of the current Python command."""
    return shlex.join([sys.executable, *sys.argv])


def ensure_output_available(output_dir: str | Path) -> None:
    """Prevent an existing output path from being overwritten."""
    output_path = Path(output_dir)

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists at {output_path}. "
            "Remove it or choose a different output path."
        )


def _write_json(path: Path, data: dict[str, object]) -> None:
    temporary_path = path.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")

    temporary_path.replace(path)


def build_manifest(
    run_id: str,
    run_name: str,
    layer: int,
    timestamp: datetime,
    git_commit: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "run_name": run_name,
        "layer": layer,
        "created_at": timestamp.isoformat(),
        "git_commit": git_commit,
        "command": get_run_command(),
        "status": "running",
        "error_message": "",
    }


def create_sae_run_directory(
    run_name: str,
    layer: int,
    results_root: Path = Path("results"),
    subdirectories: tuple[str, ...] = (),
) -> Path:
    """Create one experiment run directory and its initial manifest."""
    timestamp = utc_now()
    git_commit = get_git_commit()
    run_id = build_run_id(timestamp, git_commit)
    run_directory = (results_root / "runs" / f"layer_{layer:02d}" / run_id)

    run_directory.mkdir(parents=True)
    for subdirectory in subdirectories:
        (run_directory / subdirectory).mkdir()

    manifest = build_manifest(run_id, run_name, layer, timestamp, git_commit)
    _write_json(run_directory / "manifest.json", manifest)

    return run_directory


def write_run_config(
    run_directory: Path,
    config: dict[str, object],
) -> None:
    """Write the resolved training configuration for a run."""
    _write_json(run_directory / "config.json", config)


def update_run_manifest(
    run_directory: Path,
    status: Literal["running", "completed", "failed"],
    error_message: str = "",
) -> None:
    """Update the lifecycle status of an existing run manifest."""
    manifest_path = run_directory / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    manifest["status"] = status
    manifest["updated_at"] = utc_now().isoformat()
    manifest["error_message"] = error_message

    _write_json(manifest_path, manifest)


def write_run_history(
    run_directory: Path,
    history: list[dict[str, object]],
    test_loss: float | None = None
) -> None:
    """Write the training history for a run."""
    _write_json(run_directory / "history.json", {"history": history, "test_loss": test_loss})