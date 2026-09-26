"""Resolve and validate Solver capture sites before any generation."""
from typing import Any


# Model-family architecture adapters. These module paths describe where
# supported Hugging Face architectures expose transformer blocks; they are
# not experiment/run-specific configuration. resolve_solver_sites validates
# each resolved path against the loaded model before generation.
_SUPPORTED_SOLVER_LAYER_PATHS = {
    "gemma3": ("gemma3", "model.language_model.layers"),
    "qwen3": ("qwen3", "model.layers"),
}


def resolve_solver_sites(model: Any, loader: str, layers: list[int]) -> list[str]:
    """Require a supported architecture and existing transformer modules.

    The caller supplies the Solver model, never a Critic or Validator. This
    validates names only; hooks are installed by the Solver generation pipeline.
    """
    if loader not in _SUPPORTED_SOLVER_LAYER_PATHS:
        raise ValueError(f"Unsupported Solver loader: {loader!r}")
    expected_type, prefix = _SUPPORTED_SOLVER_LAYER_PATHS[loader]
    actual_type = getattr(getattr(model, "config", None), "model_type", None)
    if actual_type != expected_type:
        raise ValueError(
            f"Solver loader {loader!r} requires architecture {expected_type!r}; "
            f"loaded {actual_type!r}"
        )
    if (not layers or any(type(layer) is not int or layer < 0 for layer in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("Solver layers must be distinct nonnegative integers")
    sites = [f"{prefix}.{layer}" for layer in layers]
    modules = dict(model.named_modules())
    missing = [site for site in sites if site not in modules]
    if missing:
        raise ValueError(f"Solver activation modules not found: {missing}")
    return sites
