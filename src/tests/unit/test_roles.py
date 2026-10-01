"""Explicit role specs: validation, loader dispatch, weight sharing, agent setup."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.models import roles


def specs():
    def spec(model, loader, revision):
        return {"id": model, "loader": loader, "revision": revision,
                "dtype": "bfloat16", "device": "cpu"}
    return roles.resolve_roles({"roles": {
        "solver": spec("google/gemma-3-4b-it", "gemma3", "solver-revision"),
        "critic": spec("Qwen/Qwen3-4B-Instruct-2507", "qwen3", "critic-revision"),
        "validator": spec("google/gemma-3-4b-it", "gemma3", "solver-revision"),
    }})


def test_independent_models_and_shared_identical_weights(monkeypatch):
    load = Mock(side_effect=lambda spec: (object(), object()))
    monkeypatch.setattr(roles, "load_spec", load)
    loaded = roles.load_role_models(specs())
    assert loaded["solver"] is loaded["validator"]
    assert loaded["solver"] is not loaded["critic"]
    assert load.call_count == 2
    assert [call.args[0]["revision"] for call in load.call_args_list] == ["solver-revision", "critic-revision"]


def test_loader_dispatch_and_revision(monkeypatch):
    gemma = Mock(return_value=(object(), object()))
    tokenizer = Mock(return_value=object())
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"), eval=Mock())
    qwen = Mock(return_value=model)
    monkeypatch.setattr(roles, "load_gemma", gemma)
    monkeypatch.setattr(roles.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(roles.AutoModelForCausalLM, "from_pretrained", qwen)
    roles.load_spec(specs()["solver"])
    roles.load_spec(specs()["critic"])
    assert gemma.call_args.kwargs["revision"] == "solver-revision"
    assert qwen.call_args.kwargs["revision"] == "critic-revision"
    assert tokenizer.call_args.kwargs["revision"] == "critic-revision"
    assert qwen.call_args.kwargs["device_map"] == {"": "cpu"}


@pytest.mark.parametrize("count,budget,reached", [(2, 2, True), (1, 2, False)])
def test_generation_telemetry_and_text_only_messages(count, budget, reached):
    class Inputs(dict):
        def to(self, device):
            return self
    model = Mock(device="cpu")
    model.generate.return_value = torch.tensor([[10, 11] + list(range(count))])
    processor = Mock()
    processor.apply_chat_template.return_value = Inputs(input_ids=torch.tensor([[10, 11]]))
    processor.decode.return_value = "answer"
    agent = Solver(model, processor, max_new_tokens=budget)
    spec = specs()["critic"]
    spec["generation"] = {"max_new_tokens": budget, "do_sample": True, "temperature": 0.5, "top_p": 0.9}
    roles.configure_agent(agent, spec)
    assert agent._generate("prompt") == "answer"
    assert processor.apply_chat_template.call_args.args[0][0]["content"] == "prompt"
    assert agent.last_generation == {"generated_tokens": count, "max_new_tokens": budget,
                                     "reached_token_budget": reached, "finish_reason": None}
    assert model.generate.call_args.kwargs["temperature"] == 0.5
    assert model.generate.call_args.kwargs["top_p"] == 0.9


def test_shared_gemma_agents_keep_mutable_configuration_isolated(monkeypatch):
    solver_spec = specs()["solver"]
    critic_spec = roles.resolve_role_spec(
        "same_model_critic", {**solver_spec, "generation": {"max_new_tokens": 256}},
        default_max_new_tokens=256,
    )
    load = Mock(side_effect=lambda spec: (object(), object()))
    monkeypatch.setattr(roles, "load_spec", load)
    loaded = roles.load_role_models({"solver": solver_spec, "same_model_critic": critic_spec})
    solver = Solver(*loaded["solver"])
    critic = Critic(*loaded["same_model_critic"], blind_then_compare=True)
    roles.configure_agent(solver, solver_spec)
    roles.configure_agent(critic, critic_spec)

    assert solver.model is critic.model
    assert solver.processor is critic.processor
    assert solver.generation_settings is not critic.generation_settings
    assert solver.generation_settings == critic.generation_settings == {"do_sample": False}
    critic.generation_settings["do_sample"] = True
    assert solver.generation_settings == {"do_sample": False}

    assert solver.chat_template_kwargs is not critic.chat_template_kwargs
    assert solver.chat_template_kwargs == critic.chat_template_kwargs == {}
    critic.chat_template_kwargs["style"] = "critic"
    assert solver.chat_template_kwargs == {}
    solver.chat_template_kwargs["style"] = "solver"
    assert critic.chat_template_kwargs == {"style": "critic"}

    assert solver.max_new_tokens == 32
    assert critic.max_new_tokens == 256
    load.assert_called_once_with(solver_spec)
