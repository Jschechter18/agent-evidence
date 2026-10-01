"""Generic chat-template option pass-through; no model-specific expectations."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mas_sae.agents.base import (
    RESERVED_CHAT_TEMPLATE_KWARGS,
    validate_chat_template_kwargs,
)
from mas_sae.agents.solver import Solver
from mas_sae.experiments.provenance import role_metadata
from mas_sae.models import roles


class Inputs(dict):
    def to(self, device):
        return self


def make_solver():
    model = Mock(device="cpu")
    model.generate.return_value = torch.tensor([[10, 11, 12]])
    processor = Mock()
    processor.apply_chat_template.return_value = Inputs(input_ids=torch.tensor([[10, 11]]))
    processor.decode.return_value = "answer"
    return Solver(model, processor), processor


def spec(**overrides):
    base = {"id": "any/model", "loader": "qwen3", "revision": "rev", "dtype": "bfloat16",
            "device": "cpu"}
    return {**base, **overrides}


def test_omitted_options_leave_template_call_unchanged():
    solver, processor = make_solver()
    solver._generate("prompt")
    assert processor.apply_chat_template.call_args.kwargs == {
        "add_generation_prompt": True, "tokenize": True, "return_dict": True, "return_tensors": "pt"}
    assert solver.last_prompt["chat_template_kwargs"] == {}
    assert solver.last_prompt["prompt_tokens"] == 2
    assert len(solver.last_prompt["input_ids_sha256"]) == 64


def test_options_reach_apply_chat_template_and_are_recorded():
    solver, processor = make_solver()
    solver.chat_template_kwargs = {"any_switch": False, "style": "terse"}
    solver._generate("prompt")
    kwargs = processor.apply_chat_template.call_args.kwargs
    assert kwargs["any_switch"] is False and kwargs["style"] == "terse"
    assert kwargs["tokenize"] is True and kwargs["return_tensors"] == "pt"
    assert solver.last_prompt["chat_template_kwargs"] == {"any_switch": False, "style": "terse"}
    assert solver.last_generation == {"generated_tokens": 1, "max_new_tokens": 32,
                                      "reached_token_budget": False, "finish_reason": None}


@pytest.mark.parametrize("key", sorted(RESERVED_CHAT_TEMPLATE_KWARGS))
def test_structural_arguments_cannot_be_overridden(key):
    with pytest.raises(ValueError, match="may not override"):
        validate_chat_template_kwargs({key: False})
    solver, processor = make_solver()
    solver.chat_template_kwargs = {key: False}
    with pytest.raises(ValueError):
        solver._generate("prompt")
    processor.apply_chat_template.assert_not_called()


@pytest.mark.parametrize("value", [[True], {"a": 1}, object(), ("x",)])
def test_non_scalar_options_are_rejected(value):
    with pytest.raises(ValueError, match="JSON scalar"):
        validate_chat_template_kwargs({"switch": value})
    with pytest.raises(ValueError, match="mapping"):
        validate_chat_template_kwargs(["switch"])
    assert validate_chat_template_kwargs(None) == {}


def test_role_spec_validates_and_configures_options():
    resolved = roles.resolve_role_spec("critic", spec(chat_template_kwargs={"switch": False}),
                                       default_max_new_tokens=8)
    assert resolved["chat_template_kwargs"] == {"switch": False}
    agent = Solver(object(), object())
    roles.configure_agent(agent, resolved)
    assert agent.chat_template_kwargs == {"switch": False}
    plain = roles.resolve_role_spec("critic", spec(), default_max_new_tokens=8)
    assert "chat_template_kwargs" not in plain
    roles.configure_agent(agent, plain)
    assert agent.chat_template_kwargs == {}
    with pytest.raises(ValueError, match="critic.chat_template_kwargs may not override"):
        roles.resolve_role_spec("critic", spec(chat_template_kwargs={"tokenize": False}),
                                default_max_new_tokens=8)


def test_production_roles_still_reject_unknown_options_and_keep_defaults():
    config = {"roles": {role: spec() for role in ("solver", "critic", "validator")}}
    resolved = roles.resolve_roles(config)
    assert [resolved[r]["generation"]["max_new_tokens"] for r in ("solver", "critic", "validator")] == [32, 128, 4]
    config["roles"]["critic"]["unexpected"] = 1
    with pytest.raises(ValueError, match="Unsupported critic options"):
        roles.resolve_roles(config)


def test_template_options_do_not_duplicate_identical_weights(monkeypatch):
    load = Mock(side_effect=lambda spec: (object(), object()))
    monkeypatch.setattr(roles, "load_spec", load)
    specs = {
        "solver": roles.resolve_role_spec("solver", spec(), default_max_new_tokens=32),
        "same": roles.resolve_role_spec("same", spec(chat_template_kwargs={"switch": False},
                                                     generation={"max_new_tokens": 256}),
                                        default_max_new_tokens=32),
        "other": roles.resolve_role_spec("other", spec(revision="other-rev"), default_max_new_tokens=32),
    }
    loaded = roles.load_role_models(specs)
    assert loaded["solver"] is loaded["same"]
    assert loaded["solver"] is not loaded["other"]
    assert load.call_count == 2
    assert roles.weight_identity(specs["solver"]) == roles.weight_identity(specs["same"])


def test_provenance_records_template_options():
    resolved = roles.resolve_role_spec("critic", spec(chat_template_kwargs={"switch": False}),
                                       default_max_new_tokens=8)
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="rev", architectures=["X"]),
                            dtype=torch.bfloat16, device=torch.device("cpu"))
    agent = Solver(model, object())
    roles.configure_agent(agent, resolved)
    metadata = role_metadata(agent)
    assert metadata["chat_template_kwargs"] == {"switch": False}
    assert metadata["resolved_revision"] == "rev"
    assert role_metadata(Solver(model, object()))["chat_template_kwargs"] == {}
