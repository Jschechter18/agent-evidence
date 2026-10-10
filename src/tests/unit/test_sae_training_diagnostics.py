import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from mas_sae.sae.hyperparamters import Hyperparameters
from mas_sae.sae.dataloader import create_sae_dataloader


@pytest.mark.parametrize("mode", ["l1", "topk"])
def test_training_writes_diagnostics_and_frozen_feature_artifact(tmp_path, monkeypatch, mode):
    script = Path(__file__).resolve().parents[3] / "scripts" / "train_sae.py"
    spec = importlib.util.spec_from_file_location("train_sae_diagnostics", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    activation_dir = tmp_path / "data" / "activations" / "fixture" / "layer_33"
    activation_dir.mkdir(parents=True)
    torch.manual_seed(4)
    for split, n in (("train", 7), ("validation", 4), ("test", 3)):
        torch.save(torch.randn(n, 4), activation_dir / f"{split}.pt")
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(module, "HP", lambda: Hyperparameters(
        epochs=2, batch_size=3, latent_dim=6, top_k=2, inactivity_window_examples=8,
    ))
    monkeypatch.setattr(module, "create_sae_run_directory", lambda **kwargs: run)
    monkeypatch.setattr(module, "write_sae_provenance", lambda *args: None)
    monkeypatch.setattr(module, "update_run_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "append_sae_version", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "get_git_provenance", lambda: {})
    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return create_sae_dataloader(*args, **kwargs)
    monkeypatch.setattr(module, "create_sae_dataloader", loader)
    monkeypatch.setattr(sys, "argv", [str(script), "--run-name", "fixture", "--layer", "33",
                                      "--sparsity-mode", mode, "--top-k", "2"])
    module.main()
    history = json.loads((run / "history.json").read_text())
    assert history["history"][0]["train_persistent_inactive_feature_fraction"] is None
    assert history["history"][1]["train_persistent_inactive_feature_fraction"] is not None
    assert "val_variance_explained" in history["history"][0]
    frozen = history["summary"]["frozen_checkpoint_metrics"]
    artifact = json.loads((run / "feature_frequencies.json").read_text())
    for split in ("train", "validation", "test"):
        frequencies = artifact["frequencies"][split]
        assert len(frequencies) == 6
        assert sum(frequencies) == pytest.approx(frozen[split]["mean_active_features"])
        assert "variance_explained" in frozen[split]
        if mode == "topk":
            assert frozen[split]["loss"] == frozen[split]["rec_loss"]
            assert frozen[split]["mean_active_features"] <= 2
    assert history["test_loss"] == frozen["test"]["loss"]
