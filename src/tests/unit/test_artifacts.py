import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mas_sae.experiments import artifacts


def test_build_run_id_uses_timestamp_and_short_commit() -> None:
    timestamp = datetime(2026, 9, 24, 15, 30, 45, 123456, tzinfo=timezone.utc)

    run_id = artifacts.build_run_id(
        timestamp,
        "abc123def4567890abc123def4567890abc123de",
    )

    assert run_id == "20260924T153045123456Z-abc123def456"


def test_build_run_id_handles_unknown_commit() -> None:
    timestamp = datetime(2026, 9, 24, 15, 30, 45, 123456, tzinfo=timezone.utc)

    run_id = artifacts.build_run_id(timestamp, "unknown")

    assert run_id == "20260924T153045123456Z-unknown"


def test_create_run_directory_uses_reusable_run_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    git_commit = "abc123def456"
    timestamp = datetime(2026, 9, 16, 15, 30, 45, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{git_commit}\n"),
    )
    monkeypatch.setattr(artifacts, "utc_now", lambda: timestamp)
    monkeypatch.setattr(
        artifacts,
        "get_run_command",
        lambda: "python scripts/train_sae.py",
    )

    run_directory = artifacts.create_sae_run_directory(
        run_name="sae-l64",
        layer=8,
        results_root=tmp_path,
        subdirectories=("checkpoints", "plots"),
    )

    manifest = json.loads((run_directory / "manifest.json").read_text())

    assert run_directory == (
        tmp_path
        / "runs"
        / "layer_08"
        / "20260916T153045123456Z-abc123def456"
    )
    assert manifest["git_commit"] == git_commit
    assert manifest["run_id"] == run_directory.name
    assert manifest["run_name"] == "sae-l64"
    assert manifest["layer"] == 8
    assert manifest["created_at"] == timestamp.isoformat()
    assert manifest["command"] == "python scripts/train_sae.py"
    assert manifest["status"] == "running"
    assert manifest["error_message"] == ""
    assert (run_directory / "checkpoints").is_dir()
    assert (run_directory / "plots").is_dir()


def test_get_run_command_quotes_arguments(monkeypatch) -> None:
    monkeypatch.setattr(artifacts.sys, "executable", "/usr/bin/python")
    monkeypatch.setattr(
        artifacts.sys,
        "argv",
        ["scripts/train_sae.py", "--run-name", "test run"],
    )

    command = artifacts.get_run_command()

    assert command == "/usr/bin/python scripts/train_sae.py --run-name 'test run'"


def test_write_run_config(tmp_path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }

    artifacts.write_run_config(run_directory, config)

    saved_config = json.loads((run_directory / "config.json").read_text())
    assert saved_config == config


def test_update_run_manifest_records_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )
    run_directory = artifacts.create_sae_run_directory(
        run_name="sae-l64",
        layer=8,
        results_root=tmp_path,
    )

    artifacts.update_run_manifest(run_directory, status="completed")

    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["error_message"] == ""


def test_update_run_manifest_records_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )
    run_directory = artifacts.create_sae_run_directory(
        run_name="sae-l64",
        layer=8,
        results_root=tmp_path,
    )

    artifacts.update_run_manifest(
        run_directory,
        status="failed",
        error_message="RuntimeError: training failed",
    )

    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_message"] == "RuntimeError: training failed"


def test_ensure_output_available_rejects_existing_path(tmp_path) -> None:
    output_dir = tmp_path / "test_run" / "train"

    artifacts.ensure_output_available(output_dir)

    output_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Output already exists"):
        artifacts.ensure_output_available(output_dir)


def test_get_package_version(monkeypatch) -> None:
    monkeypatch.setattr(
        artifacts,
        "version",
        lambda package: f"{package}-version",
    )

    assert artifacts.get_package_version("torch") == "torch-version"


def test_get_package_version_returns_unknown_when_missing(
    monkeypatch,
) -> None:
    def raise_missing(package: str) -> None:
        raise artifacts.PackageNotFoundError(package)

    monkeypatch.setattr(artifacts, "version", raise_missing)

    assert artifacts.get_package_version("missing-package") == "unknown"
