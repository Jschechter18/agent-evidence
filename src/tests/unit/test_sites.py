from types import SimpleNamespace

import pytest

from mas_sae.activations.sites import resolve_solver_sites


@pytest.mark.parametrize("loader,prefix", [
    ("gemma3", "model.language_model.layers"), ("qwen3", "model.layers")])
def test_solver_site_resolution(loader, prefix):
    model = SimpleNamespace(config=SimpleNamespace(model_type=loader),
        named_modules=lambda: [(f"{prefix}.0", object()), (f"{prefix}.2", object())])
    assert resolve_solver_sites(model, loader, [2, 0]) == [f"{prefix}.2", f"{prefix}.0"]
    with pytest.raises(ValueError, match="not found"):
        resolve_solver_sites(model, loader, [1])


@pytest.mark.parametrize("loader,model_type", [("unsupported", "unsupported"), ("gemma3", "qwen3"), ("qwen3", None)])
def test_solver_site_resolution_rejects_unknown_or_mismatched_architecture(loader, model_type):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    with pytest.raises(ValueError, match="Unsupported|requires architecture"):
        resolve_solver_sites(model, loader, [0])
