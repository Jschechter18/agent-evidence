from typing import Any

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def load_gemma(
    model_id: str = "google/gemma-3-4b-it",
    cache_dir: str = "checkpoints/huggingface",
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
    device_map: Any = "auto",
) -> tuple[Gemma3ForConditionalGeneration, Any]:
    """Load the Gemma model and processor used by the agent pipeline."""
    revision_kwargs = {} if revision is None else {"revision": revision}
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        **revision_kwargs,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        **revision_kwargs,
        dtype=dtype,
        device_map=device_map,
    )

    model.eval()

    return model, processor
