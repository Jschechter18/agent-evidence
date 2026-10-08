import pytest
import torch

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


INPUT_DIM = 8
HIDDEN_DIM = 6
LATENT_DIM = 12


@pytest.fixture
def model() -> SparseAutoencoder:
    torch.manual_seed(0)
    return SparseAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )


@pytest.mark.parametrize("input_shape", [(4, INPUT_DIM), (2, 5, INPUT_DIM)])
def test_encoder_preserves_leading_dimensions_and_uses_latent_dimension(
    model: SparseAutoencoder,
    input_shape: tuple[int, ...],
) -> None:
    activations = torch.randn(input_shape)

    sparse_features = model.encoder(activations)

    assert sparse_features.shape == (*input_shape[:-1], LATENT_DIM)


def test_encoder_features_are_nonnegative(model: SparseAutoencoder) -> None:
    activations = torch.randn(4, INPUT_DIM)

    sparse_features = model.encoder(activations)

    assert torch.all(sparse_features >= 0)


def test_decoder_projects_features_back_to_input_dimension(
    model: SparseAutoencoder,
) -> None:
    sparse_features = torch.randn(4, LATENT_DIM)

    reconstructed_activations = model.decoder(sparse_features)

    assert reconstructed_activations.shape == (4, INPUT_DIM)


def test_forward_matches_separate_encode_and_decode_calls(
    model: SparseAutoencoder,
) -> None:
    activations = torch.randn(4, INPUT_DIM)

    sparse_features, reconstructed_activations = model(activations)
    expected_features = model.encoder(activations)
    expected_reconstruction = model.decoder(expected_features)

    assert torch.equal(sparse_features, expected_features)
    assert torch.equal(reconstructed_activations, expected_reconstruction)


def test_forward_supports_backpropagation(model: SparseAutoencoder) -> None:
    activations = torch.randn(4, INPUT_DIM)
    sparse_features, reconstructed_activations = model(activations)
    loss = torch.nn.functional.mse_loss(reconstructed_activations, activations)
    loss = loss + sparse_features.abs().mean()

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_input_scale_preserves_features_and_restores_output_units(model: SparseAutoencoder) -> None:
    activations = torch.randn(4, INPUT_DIM)
    features, reconstruction = model(activations)
    model.input_scale.fill_(3.0)

    scaled_features, scaled_reconstruction = model(activations * 3.0)

    assert torch.allclose(scaled_features, features, atol=1e-6)
    assert torch.allclose(scaled_reconstruction, reconstruction * 3.0, atol=1e-6)


def test_checkpoint_restores_scale_and_outputs(model: SparseAutoencoder, tmp_path) -> None:
    model.input_scale.fill_(7.0)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = SparseAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    restored.load_state_dict(torch.load(path, weights_only=True))
    activations = torch.randn(4, INPUT_DIM)

    assert restored.input_scale.item() == 7.0
    assert "input_scale" not in dict(restored.named_parameters())
    for expected, actual in zip(model(activations), restored(activations)):
        assert torch.equal(actual, expected)


def test_decoder_normalization_constrains_columns_and_preserves_bias(model: SparseAutoencoder) -> None:
    weight = model.decoder_layer[0].weight
    assert torch.allclose(weight.norm(dim=0), torch.ones(LATENT_DIM), atol=1e-6)
    bias = model.decoder_layer[0].bias.detach().clone()
    with torch.no_grad():
        weight.mul_(torch.arange(1, LATENT_DIM + 1))

    model.normalize_decoder_weights()

    assert model.decoder_layer[0].weight is weight
    assert torch.allclose(weight.norm(dim=0), torch.ones(LATENT_DIM), atol=1e-6)
    assert torch.equal(model.decoder_layer[0].bias, bias)
