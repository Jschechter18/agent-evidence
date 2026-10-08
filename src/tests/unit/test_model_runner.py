import json
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from mas_sae.sae import model_runner as model_runner_module
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


@pytest.mark.parametrize("runner_method", ["train_epoch", "val_epoch", "test"])
def test_topk_metrics_record_zero_l1_and_reconstruction_objective(runner_method):
    model = SparseAutoencoder(4, 8, 6, "topk", 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    runner = ModelRunner(model, sparsity_coefficient=100.0, optimizer=optimizer)
    metrics = getattr(runner, runner_method)(DataLoader(torch.randn(5, 4), batch_size=2))
    assert metrics["weighted_sparsity_loss"] == 0.0
    assert metrics["loss"] == metrics["rec_loss"]
    assert metrics["mean_active_features"] <= 2
    assert json.loads(json.dumps(metrics))["weighted_sparsity_loss"] == 0.0


def test_diagnostics_use_training_mean_and_exact_example_window():
    model = TrackingModel()
    runner = ModelRunner(model, 0.0, torch.optim.SGD(model.parameters(), lr=0.0),
                         training_mean=torch.tensor([1.0, 1.0]), inactivity_window_examples=4)
    first = DataLoader(torch.tensor([[2., 0.], [0., 0.]]), batch_size=1)
    metrics = runner.train_epoch(first)
    assert metrics["persistent_inactive_feature_fraction"] is None
    assert metrics["variance_explained"] == pytest.approx(0.75)
    assert runner.last_feature_frequencies.tolist() == [0.5, 0.0]
    assert metrics["firing_frequency_p50"] == pytest.approx(0.25)
    runner.val_epoch(DataLoader(torch.tensor([[0., 2.]]), batch_size=1))
    assert runner.training_examples_seen == 2
    assert runner.last_fired_example.tolist() == [1, 0]
    metrics = runner.train_epoch(first)
    assert metrics["persistent_inactive_feature_fraction"] == 0.5
    assert runner.last_fired_example.tolist() == [3, 0]


def test_variance_explained_undefined_for_zero_baseline():
    model = TrackingModel()
    runner = ModelRunner(model, 0.0, Mock(), training_mean=torch.ones(2))
    metrics = runner.val_epoch(DataLoader(torch.ones(2, 2), batch_size=1))
    assert metrics["mean_baseline_rec_loss"] == 0.0
    assert metrics["variance_explained"] is None
    assert json.loads(json.dumps(metrics))["variance_explained"] is None


class TrackingModel(nn.Module):
    """Small deterministic model that records the context of each forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.latent_dim = 2
        self.sparsity_mode = "l1"
        self.normalization_calls = 0
        self.training_states: list[bool] = []
        self.grad_states: list[bool] = []

    def normalize_decoder_weights(self) -> None:
        self.normalization_calls += 1

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.training_states.append(self.training)
        self.grad_states.append(torch.is_grad_enabled())
        scaled_activations = activations * self.scale
        return torch.relu(scaled_activations), scaled_activations


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters: object, lr: float) -> None:
        super().__init__(parameters, lr=lr)
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args: object, **kwargs: object) -> None:
        self.zero_grad_calls += 1
        super().zero_grad(*args, **kwargs)

    def step(self, *args: object, **kwargs: object) -> object:
        self.step_calls += 1
        return super().step(*args, **kwargs)


@pytest.fixture(autouse=True)
def disable_progress_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_runner_module,
        "tqdm",
        lambda iterable, **_: iterable,
    )


def test_loss_fn_combines_reconstruction_and_weighted_sparsity_losses() -> None:
    runner = ModelRunner(
        model=Mock(input_scale=torch.tensor(2.0), sparsity_mode="l1"),
        sparsity_coefficient=0.25,
        optimizer=Mock(),
    )
    batch = torch.tensor([[1.0, -1.0], [3.0, 1.0]])
    reconstructed = torch.tensor([[0.0, -2.0], [1.0, 1.0]])
    sparse_features = torch.tensor([[-2.0, 0.0], [1.0, 5.0]])

    losses = runner._loss_fn(reconstructed, batch, sparse_features)

    # Squared errors total 6; divide by 4 elements and scale squared (4).
    assert losses["rec_loss"].item() == pytest.approx(0.375)
    assert losses["weighted_sparsity_loss"].item() == pytest.approx(0.5)
    assert losses["loss"].item() == pytest.approx(0.875)


def test_common_passes_batch_to_model_and_returns_outputs() -> None:
    batch = torch.randn(3, 4)
    sparse_features = torch.randn(3, 6)
    reconstructed = torch.randn(3, 4)
    model = Mock(return_value=(sparse_features, reconstructed), input_scale=torch.tensor(1.0), sparsity_mode="l1")
    model.parameters.return_value = iter([nn.Parameter(torch.zeros(1))])
    runner = ModelRunner(model=model, sparsity_coefficient=0.1, optimizer=Mock())

    outputs = runner._common(batch)

    model.assert_called_once_with(batch)
    assert outputs["reconstructed"] is reconstructed
    assert outputs["sparse_features"] is sparse_features
    expected_loss = F.mse_loss(reconstructed, batch)
    expected_loss += 0.1 * sparse_features.abs().mean()
    assert torch.equal(outputs["loss"], expected_loss)


def test_train_epoch_enables_training_and_updates_model_for_every_batch() -> None:
    model = TrackingModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    runner = ModelRunner(model, sparsity_coefficient=0.05, optimizer=optimizer)
    dataloader = DataLoader(
        torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
        batch_size=2,
    )
    initial_scale = model.scale.detach().clone()

    metrics = runner.train_epoch(dataloader)

    assert model.training is True
    assert model.training_states == [True, True]
    assert model.grad_states == [True, True]
    assert optimizer.zero_grad_calls == len(dataloader)
    assert optimizer.step_calls == len(dataloader)
    assert not torch.equal(model.scale.detach(), initial_scale)
    assert all(value is None or isinstance(value, (float, int)) for value in metrics.values())
    assert metrics["loss"] >= 0
    assert model.normalization_calls == len(dataloader)
    assert metrics["loss"] == pytest.approx(metrics["rec_loss"] + metrics["weighted_sparsity_loss"])
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("runner_method", ["val_epoch", "test"])
def test_evaluation_pipeline_disables_gradients_and_does_not_update_model(
    runner_method: str,
) -> None:
    model = TrackingModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    sparsity_coefficient = 0.2
    runner = ModelRunner(model, sparsity_coefficient, optimizer)
    activations = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    dataloader = DataLoader(activations, batch_size=2)
    initial_scale = model.scale.detach().clone()
    reconstructed = activations * initial_scale
    expected_rec_loss = F.mse_loss(reconstructed, activations).item()
    expected_sparsity_loss = sparsity_coefficient * reconstructed.abs().mean().item()

    metrics = getattr(runner, runner_method)(dataloader)

    assert model.training is False
    assert model.training_states == [False, False]
    assert model.grad_states == [False, False]
    assert optimizer.zero_grad_calls == 0
    assert optimizer.step_calls == 0
    assert torch.equal(model.scale.detach(), initial_scale)
    assert metrics["rec_loss"] == pytest.approx(expected_rec_loss)
    assert metrics["weighted_sparsity_loss"] == pytest.approx(expected_sparsity_loss)
    assert metrics["loss"] == pytest.approx(expected_rec_loss + expected_sparsity_loss)
    assert model.normalization_calls == 0


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("runner_method", ["train_epoch", "val_epoch", "test"])
def test_cpu_batches_run_on_model_device(device: str, runner_method: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    model = SparseAutoencoder(input_dim=4, hidden_dim=8, latent_dim=6).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = ModelRunner(model, sparsity_coefficient=0.05, optimizer=optimizer)
    activations = torch.randn(6, 4)
    dataloader = DataLoader(activations, batch_size=2)
    initial_weight = next(model.parameters()).detach().clone()

    metrics = getattr(runner, runner_method)(dataloader)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values() if value is not None)
    assert torch.allclose(
        model.decoder_layer[0].weight.norm(dim=0),
        torch.ones(model.latent_dim, device=device),
        atol=1e-5,
    )
    assert activations.device.type == "cpu"
    weight_changed = not torch.equal(initial_weight, next(model.parameters()).detach())
    assert weight_changed == (runner_method == "train_epoch")


@pytest.mark.parametrize("runner_method", ["train_epoch", "val_epoch", "test"])
def test_feature_metrics_track_whole_epoch_and_reset(runner_method: str) -> None:
    model = TrackingModel()
    runner = ModelRunner(model, 0.1, torch.optim.SGD(model.parameters(), lr=0.0))
    # Disjoint features fire in the first and final (partial) batches.
    activations = torch.tensor([[1., 0.], [0., 0.], [0., 2.]])
    metrics = getattr(runner, runner_method)(DataLoader(activations, batch_size=2))
    assert metrics["mean_active_features"] == pytest.approx(2 / 3)
    assert metrics["inactive_feature_fraction"] == 0.0

    for activations, expected_l0, expected_inactive in [
        (torch.zeros(3, 2), 0., 1.),
        (torch.tensor([[1., 0.]] * 3), 1., 0.5),
    ]:
        metrics = getattr(runner, runner_method)(DataLoader(activations, batch_size=2))
        assert metrics["mean_active_features"] == expected_l0
        assert metrics["inactive_feature_fraction"] == expected_inactive


@pytest.mark.parametrize("runner_method", ["train_epoch", "val_epoch", "test"])
def test_empty_dataset_raises(runner_method: str) -> None:
    model = TrackingModel()
    runner = ModelRunner(model, 0.1, torch.optim.SGD(model.parameters(), lr=0.1))
    with pytest.raises(ValueError, match="empty dataset"):
        getattr(runner, runner_method)(DataLoader(torch.empty(0, 2), batch_size=2))


def test_zero_features_do_not_erase_reconstruction_loss() -> None:
    model = Mock(input_scale=torch.tensor(1.0), sparsity_mode="l1")
    runner = ModelRunner(model, 0.1, Mock())
    losses = runner._loss_fn(torch.zeros(2, 2), torch.ones(2, 2), torch.zeros(2, 3))
    assert losses["weighted_sparsity_loss"].item() == 0.0
    assert losses["loss"].item() == 1.0
