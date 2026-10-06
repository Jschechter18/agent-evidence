from __future__ import annotations

from datetime import datetime, timezone
import csv
import fcntl
import hashlib
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


def get_git_diff() -> bytes:
    """Staged and unstaged tracked changes relative to HEAD, excluding untracked files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not read tracked git changes.") from error
    return result.stdout


def get_git_diff_sha256() -> str:
    """Hash tracked changes; raise if Git is unavailable instead of guessing."""
    return hashlib.sha256(get_git_diff()).hexdigest()


def get_untracked_sha256(*paths: str) -> str:
    """SHA-256 of untracked, non-ignored files under ``paths``: names and bytes.

    Complements ``get_git_diff_sha256``, which cannot see new files that are
    not yet committed.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *paths],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not list untracked files.") from error
    digest = hashlib.sha256()
    for name in sorted(filter(None, result.stdout.split(b"\0"))):
        digest.update(name + b"\0" + hashlib.sha256(Path(name.decode()).read_bytes()).digest())
    return digest.hexdigest()


def sha256_text(text: object) -> str:
    """SHA-256 of a string encoded as UTF-8."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_json(value: object) -> str:
    """SHA-256 of a JSON-serializable value in canonical form."""
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's exact bytes."""
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def get_git_provenance() -> dict[str, object]:
    """Snapshot the working tree before creating any run artifacts."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
        diff = get_git_diff()
        return {
            "git_dirty": bool(status.strip()),
            "git_status": status,
            "git_diff": diff.decode("utf-8", errors="replace"),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "git_diff_scope": "Staged and unstaged tracked changes relative to HEAD; untracked contents excluded.",
        }
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        return {"git_dirty": None}


def write_sae_provenance(
    run_directory: Path,
    activation_files: dict[str, Path],
    git_provenance: dict[str, object],
) -> None:
    """Record input identities and the previously captured working tree."""
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(git_provenance)
    manifest["activation_files"] = {
        split: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for split, path in activation_files.items()
    }
    write_json_atomic(manifest_path, manifest)


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


def write_json_atomic(path: Path, data: dict[str, object]) -> None:
    """Write JSON through a temporary file and rename it into place."""
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
    write_json_atomic(run_directory / "manifest.json", manifest)

    return run_directory


def write_run_config(
    run_directory: Path,
    config: dict[str, object],
) -> None:
    """Write the resolved training configuration for a run."""
    write_json_atomic(run_directory / "config.json", config)


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

    write_json_atomic(manifest_path, manifest)


def write_run_history(
    run_directory: Path,
    history: list[dict[str, object]],
    test_loss: float | None = None,
    summary: dict | None = None,
) -> None:
    """Write the training history for a run."""
    write_json_atomic(run_directory / "history.json", {"history": history, "test_loss": test_loss, "summary": summary})


def append_sae_version(
    run_directory: Path,
    *,
    project_root: Path,
    best_val_score: float,
    test_score: float | None = None,
) -> Path:
    """Append a completed run once to its layer's CSV (scores are SAE losses).

    Paths are relative to project_root. An existing run_id is left unchanged;
    later evaluation of that same version must explicitly update its row.
    """
    manifest = json.loads((run_directory / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed SAE runs can be appended.")
    checkpoint = run_directory / "checkpoints" / "best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = json.loads((run_directory / "config.json").read_text())
    fields = ["run_id", "activation_run_name", "checkpoint_path", "commit", "test_score", "best_val_score"]
    row = {
        "run_id": manifest["run_id"],
        "activation_run_name": config["activation_run_name"],
        "checkpoint_path": checkpoint.resolve().relative_to(project_root.resolve()).as_posix(),
        "commit": manifest["git_commit"],
        "test_score": "null" if test_score is None else test_score,
        "best_val_score": best_val_score,
    }
    csv_path = run_directory.parent / "sae_versions.csv"
    with csv_path.open("a+", newline="", encoding="utf-8") as file:
        # Serialize the duplicate check and append for concurrent training runs.
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.seek(0)
        reader = csv.DictReader(file)
        if reader.fieldnames is not None and reader.fieldnames != fields:
            raise ValueError(f"Unexpected SAE version CSV columns: {csv_path}")
        if any(existing["run_id"] == row["run_id"] for existing in reader):
            return csv_path
        file.seek(0, 2)
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
    return csv_path
