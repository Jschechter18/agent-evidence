from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

import torch


# Chat-template arguments the agent owns. Callers may pass any other template
# option (for example a template's own switches) through ``chat_template_kwargs``
# but never these, so message shape and tensor handling stay predictable.
RESERVED_CHAT_TEMPLATE_KWARGS = frozenset({
    "conversation", "messages", "add_generation_prompt", "tokenize",
    "return_dict", "return_tensors", "return_assistant_tokens_mask",
    "continue_final_message", "chat_template", "padding", "truncation",
    "max_length", "tools", "documents",
})


def validate_chat_template_kwargs(kwargs: Any) -> dict[str, Any]:
    """Return a copy of optional chat-template options after validation.

    Options must be a mapping of non-empty string keys to JSON scalars
    (``bool``, ``int``, ``float``, ``str`` or ``None``) so they can be recorded
    in provenance unchanged. Reserved structural arguments are rejected.
    """
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise ValueError("chat_template_kwargs must be a mapping")
    validated: dict[str, Any] = {}
    for key, value in kwargs.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("chat_template_kwargs keys must be non-empty strings")
        if key in RESERVED_CHAT_TEMPLATE_KWARGS:
            raise ValueError(f"chat_template_kwargs may not override {key!r}")
        if value is not None and type(value) not in (bool, int, float, str):
            raise ValueError(f"chat_template_kwargs[{key!r}] must be a JSON scalar")
        validated[key] = value
    return validated


def _token_ids_sha256(input_ids: Any) -> str | None:
    """SHA-256 of the exact prompt token ids, or None when unavailable."""
    try:
        ids = input_ids.tolist()
        return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001 - telemetry must never break generation
        return None


class ProcessorProtocol(Protocol):
    """Processor interface required by generation agents."""

    def apply_chat_template(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        ...


class Agent:
    """Shared generation behavior for Solver, Critic, and Validator."""

    def __init__(
        self,
        model: Any,
        processor: ProcessorProtocol,
        *,
        max_new_tokens: int = 32,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.generation_settings = {"do_sample": False}
        self.generation_history = []
        self.last_generation = None
        self.text_only = False
        # Optional template options forwarded to ``apply_chat_template``.
        # Empty by default, which leaves the template call unchanged.
        self.chat_template_kwargs: dict[str, Any] = {}
        # Digest of the exact tokenized prompt of the last generation.
        self.last_prompt: dict[str, Any] | None = None

    def _generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        if self.text_only:
            messages[0]["content"] = prompt

        template_kwargs = validate_chat_template_kwargs(self.chat_template_kwargs)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self.model.device)

        prompt_tokens = inputs["input_ids"].shape[-1]
        self.last_prompt = {
            "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            "input_ids_sha256": _token_ids_sha256(inputs["input_ids"]),
            "chat_template_kwargs": dict(template_kwargs),
        }

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                **self.generation_settings,
            )

        count = int(output.shape[-1] - prompt_tokens)
        self.last_generation = {
            "generated_tokens": count,
            "max_new_tokens": self.max_new_tokens,
            "reached_token_budget": count >= self.max_new_tokens,
            "finish_reason": None,
        }
        self.generation_history.append(dict(self.last_generation))
        return self.processor.decode(
            output[0][prompt_tokens:],
            skip_special_tokens=True,
        ).strip()
