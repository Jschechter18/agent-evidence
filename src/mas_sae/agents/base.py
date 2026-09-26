from __future__ import annotations

from typing import Any, Protocol

import torch


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

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_tokens = inputs["input_ids"].shape[-1]

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
