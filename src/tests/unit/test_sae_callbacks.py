from pathlib import Path

import pytest
import torch
from torch import nn

from mas_sae.sae.callbacks.checkpointing import CheckpointEvaluatorCallback, load_sae_checkpoint
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder
from mas_sae.sae.callbacks.early_stopping import EarlyStoppingCallback


@pytest.mark.parametrize("mode,k", [("l1", None), ("topk", 2)])
def test_sae_checkpoint_restores_architecture_and_outputs(tmp_path, mode, k):
    model = SparseAutoencoder(4, 8, 6, mode, k)
    model.input_scale.fill_(7.0)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    callback = CheckpointEvaluatorCallback(tmp_path)
    callback.on_validation_end({"loss": 1.0}, {"loss": 0.5}, 0, model, optimizer, scheduler)
    restored = load_sae_checkpoint(tmp_path / "best_checkpoint.pt")
    assert restored.sparsity_mode == mode
    assert restored.top_k == k
    assert not restored.training
    x = torch.randn(3, 4)
    for expected, actual in zip(model(x), restored(x)):
        assert torch.equal(expected, actual)


def test_legacy_checkpoint_uses_run_config(tmp_path):
    import json
    model = SparseAutoencoder(4, 8, 6)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    path = checkpoint_dir / "best_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    with pytest.raises(ValueError, match="lacks model_config"):
        load_sae_checkpoint(path)
    (tmp_path / "config.json").write_text(json.dumps({
        "input_dim": 4, "hidden_dim": 8, "latent_dim": 6,
    }))
    restored = load_sae_checkpoint(path)
    assert restored.sparsity_mode == "l1"
    x = torch.randn(3, 4)
    assert torch.equal(restored.encoder(x), model.encoder(x))


@pytest.fixture
def training_components():
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.1,
    )

    # Populate Adam's state so the checkpoint verifies more than an empty dict.
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

    return model, optimizer, scheduler


def test_checkpoint_evaluator_saves_complete_loadable_checkpoint(
    tmp_path: Path,
    training_components,
) -> None:
    model, optimizer, scheduler = training_components
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    evaluator = CheckpointEvaluatorCallback(checkpoint_directory)

    evaluator.on_validation_end(
        train_metrics={"loss": 0.6},
        val_metrics={"loss": 0.5},
        epoch=2,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    checkpoint = torch.load(
        checkpoint_directory / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["epoch"] == 2
    assert checkpoint["train_loss"] == pytest.approx(0.6)
    assert checkpoint["val_loss"] == pytest.approx(0.5)
    assert checkpoint["best_loss"] == pytest.approx(0.5)

    restored_model = nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        restored_optimizer,
        patience=2,
        factor=0.1,
    )
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restored_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    for expected, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, restored)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.state_dict()["state"]


def test_checkpoint_evaluator_does_not_save_without_improvement(
    tmp_path: Path,
    training_components,
) -> None:
    model, optimizer, scheduler = training_components
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    evaluator = CheckpointEvaluatorCallback(checkpoint_directory)
    evaluator.best_loss = 0.4

    evaluator.on_validation_end(
        train_metrics={"loss": 0.6},
        val_metrics={"loss": 0.5},
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert not (checkpoint_directory / "best_checkpoint.pt").exists()
    assert evaluator.best_loss == pytest.approx(0.4)


@pytest.mark.parametrize("next_val_loss", [0.5, 0.6])
def test_checkpoint_evaluator_preserves_best_checkpoint_without_improvement(
    tmp_path: Path,
    training_components,
    next_val_loss: float,
) -> None:
    model, optimizer, scheduler = training_components
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    evaluator = CheckpointEvaluatorCallback(checkpoint_directory)

    evaluator.on_validation_end(
        train_metrics={"loss": 0.6},
        val_metrics={"loss": 0.5},
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    evaluator.on_validation_end(
        train_metrics={"loss": 0.7},
        val_metrics={"loss": next_val_loss},
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    checkpoint = torch.load(
        checkpoint_directory / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["epoch"] == 0
    assert checkpoint["val_loss"] == pytest.approx(0.5)


def test_early_stopping_stops_after_patience_consecutive_failures() -> None:
    early_stopping = EarlyStoppingCallback(patience=2)

    assert early_stopping.on_validation_end(1.0) is False
    assert early_stopping.on_validation_end(1.1) is False
    assert early_stopping.on_validation_end(1.2) is True


def test_early_stopping_improvement_resets_failure_counter() -> None:
    early_stopping = EarlyStoppingCallback(patience=2)

    assert early_stopping.on_validation_end(1.0) is False
    assert early_stopping.on_validation_end(1.1) is False
    assert early_stopping.counter == 1

    assert early_stopping.on_validation_end(0.9) is False
    assert early_stopping.counter == 0
    assert early_stopping.best_loss == pytest.approx(0.9)


def test_early_stopping_requires_minimum_improvement() -> None:
    early_stopping = EarlyStoppingCallback(patience=2, min_delta=0.1)

    assert early_stopping.on_validation_end(1.0) is False
    assert early_stopping.on_validation_end(0.95) is False
    assert early_stopping.best_loss == pytest.approx(1.0)
    assert early_stopping.counter == 1

    assert early_stopping.on_validation_end(0.89) is False
    assert early_stopping.best_loss == pytest.approx(0.89)
    assert early_stopping.counter == 0
