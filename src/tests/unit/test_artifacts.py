import csv
import json
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mas_sae.experiments import artifacts


@pytest.mark.parametrize("dirty", [False, True])
def test_sae_provenance_preserves_status_before_run_creation(tmp_path, monkeypatch, dirty):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    activation_path = tmp_path / "train.pt"
    activation_path.write_bytes(b"original activations")
    subprocess.run(["git", "add", "train.pt"], check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "Initial inputs"],
        check=True,
    )
    if dirty:
        activation_path.write_bytes(b"changed activations")

    snapshot = artifacts.get_git_provenance()
    run = artifacts.create_sae_run_directory("test", 33, results_root=tmp_path / "results")
    # The unignored run files dirty the repository, but must not alter its saved snapshot.
    assert artifacts.get_git_provenance()["git_dirty"] is True
    artifacts.write_sae_provenance(run, {"train": activation_path}, snapshot)

    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["git_dirty"] is dirty
    assert "results/" not in manifest["git_status"]
    assert manifest["git_diff"] == snapshot["git_diff"]
    assert bool(manifest["git_diff"]) is dirty
    assert manifest["activation_files"]["train"]["sha256"] == artifacts.sha256_file(activation_path)


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


def test_sha256_helpers_are_canonical(tmp_path) -> None:
    assert artifacts.sha256_json({"a": 1, "b": [2]}) == artifacts.sha256_json({"b": [2], "a": 1})
    assert artifacts.sha256_json({"a": 1}) != artifacts.sha256_json({"a": 2})
    path = tmp_path / "file.txt"
    path.write_bytes("é".encode("utf-8"))
    assert artifacts.sha256_file(path) == artifacts.sha256_text("é")


def make_completed_sae_run(root, run_id, layer=33):
    run = root / "runs" / f"layer_{layer:02d}" / run_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best_checkpoint.pt").touch()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "git_commit": "abc123", "status": "completed",
    }))
    return run


def test_append_sae_versions_creates_appends_and_deduplicates(tmp_path):
    first = make_completed_sae_run(tmp_path, "first")
    second = make_completed_sae_run(tmp_path, "second")
    path = artifacts.append_sae_version(first, project_root=tmp_path, best_val_score=2.5)
    artifacts.append_sae_version(second, project_root=tmp_path, best_val_score=1.5, test_score=1.8)
    before = path.read_bytes()
    artifacts.append_sae_version(first, project_root=tmp_path, best_val_score=2.5)
    assert path.read_bytes() == before
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {"run_id": "first", "checkpoint_path": "runs/layer_33/first/checkpoints/best_checkpoint.pt",
         "commit": "abc123", "test_score": "null", "best_val_score": "2.5"},
        {"run_id": "second", "checkpoint_path": "runs/layer_33/second/checkpoints/best_checkpoint.pt",
         "commit": "abc123", "test_score": "1.8", "best_val_score": "1.5"},
    ]
    other = make_completed_sae_run(tmp_path, "third", layer=17)
    other_path = artifacts.append_sae_version(other, project_root=tmp_path, best_val_score=3.0)
    assert other_path.parent.name == "layer_17"
    assert path.read_bytes() == before


@pytest.mark.parametrize("invalid", ["failed", "running", "missing_checkpoint"])
def test_append_sae_version_rejects_unavailable_versions(tmp_path, invalid):
    run = make_completed_sae_run(tmp_path, "first")
    if invalid == "missing_checkpoint":
        (run / "checkpoints" / "best_checkpoint.pt").unlink()
        error = FileNotFoundError
    else:
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["status"] = invalid
        (run / "manifest.json").write_text(json.dumps(manifest))
        error = ValueError
    with pytest.raises(error):
        artifacts.append_sae_version(run, project_root=tmp_path, best_val_score=2.5)
    assert not (run.parent / "sae_versions.csv").exists()
