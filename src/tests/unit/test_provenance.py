import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from mas_sae.agents.solver import Solver
from mas_sae.experiments import artifacts, provenance
from mas_sae.models import roles


def test_build_resolved_config_adds_provenance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "get_git_commit",
        lambda: "abc123",
    )
    monkeypatch.setattr(
        artifacts,
        "get_package_version",
        lambda package: f"{package}-version",
    )

    config = {
        "model": {"id": "google/gemma-3-4b-it"},
        "dataset": {"source_split": "train", "num_questions": 2},
        "collection": {"layers": [8, 17], "seed": 42},
        "output": {"run_name": "test_run"},
    }

    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="model-revision")
    )
    solver = SimpleNamespace(max_new_tokens=32)
    critic = SimpleNamespace(max_new_tokens=128)
    validator = SimpleNamespace(max_new_tokens=4)

    resolved = provenance.build_resolved_config(
        config,
        model,
        solver,
        critic,
        validator,
    )

    result = resolved["provenance"]

    assert resolved["model"] == config["model"]
    assert result["git_commit"] == "abc123"
    assert result["model_revision"] == "model-revision"
    assert result["dataset_id"] == "dgslibisey/MuSiQue"
    assert result["package_versions"]["torch"] == "torch-version"
    assert result["generation"]["do_sample"] is False
    assert result["generation"]["solver_max_new_tokens"] == 32
    assert result["generation"]["critic_max_new_tokens"] == 128
    assert result["generation"]["validator_max_new_tokens"] == 4


def _resolve(monkeypatch, critic, **kwargs):
    monkeypatch.setattr(artifacts, "get_git_commit", lambda: "abc123")
    monkeypatch.setattr(
        artifacts, "get_package_version", lambda package: "x"
    )

    return provenance.build_resolved_config(
        {"output": {"run_name": "r"}},
        SimpleNamespace(config=SimpleNamespace(_commit_hash="rev")),
        SimpleNamespace(max_new_tokens=32),
        critic,
        SimpleNamespace(max_new_tokens=4),
        **kwargs,
    )["provenance"]


@pytest.mark.parametrize(
    (
        "critic",
        "type_checked_target",
        "expected_natural",
        "expected_controlled",
    ),
    [
        (
            SimpleNamespace(max_new_tokens=128),
            False,
            "NATURAL_PROMPT_V1",
            "CONTROLLED_PROMPT_V1",
        ),
        (
            SimpleNamespace(
                max_new_tokens=128,
                blind_then_compare=False,
                controlled_as_own_conclusion=False,
            ),
            False,
            "NATURAL_PROMPT_V1",
            "CONTROLLED_PROMPT_V1",
        ),
        (
            SimpleNamespace(
                max_new_tokens=128,
                blind_then_compare=True,
                controlled_as_own_conclusion=True,
            ),
            True,
            "NATURAL_BLIND_PROMPT+NATURAL_COMPARE_PROMPT",
            "CONTROLLED_OWN_CONCLUSION_PROMPT",
        ),
        (
            SimpleNamespace(
                max_new_tokens=128,
                blind_then_compare=True,
                controlled_as_own_conclusion=False,
            ),
            False,
            "NATURAL_BLIND_PROMPT+NATURAL_COMPARE_PROMPT",
            "CONTROLLED_PROMPT_V1",
        ),
    ],
    ids=["no-attributes", "all-off", "all-on", "blind-only"],
)
def test_build_resolved_config_records_actual_capabilities(
    monkeypatch,
    critic,
    type_checked_target,
    expected_natural,
    expected_controlled,
) -> None:
    result = _resolve(
        monkeypatch, critic, type_checked_target=type_checked_target
    )
    prompts = result["prompt_versions"]

    assert result["capabilities"] == {
        "blind_then_compare": getattr(critic, "blind_then_compare", False),
        "controlled_as_own_conclusion": getattr(
            critic, "controlled_as_own_conclusion", False
        ),
        "type_checked_target": type_checked_target,
    }
    assert prompts["critic_natural"] == expected_natural
    assert prompts["critic_controlled"] == expected_controlled
    assert prompts["solver_solve"] == "SOLVE_PROMPT_V1"
    assert ("critic_distractor" in prompts) == type_checked_target


def test_build_resolved_config_echoes_protocol_label_without_branching(
    monkeypatch,
) -> None:
    critic = SimpleNamespace(max_new_tokens=128)

    # the label is recorded as given and does not change the behaviour
    # actually recorded
    labelled = _resolve(monkeypatch, critic, protocol_version="v9")
    assert labelled["protocol_version"] == "v9"
    assert labelled["capabilities"]["blind_then_compare"] is False
    assert labelled["prompt_versions"]["critic_natural"] == "NATURAL_PROMPT_V1"

    # no label supplied: no label key, everything else unchanged
    unlabelled = _resolve(monkeypatch, critic)
    assert "protocol_version" not in unlabelled
    assert unlabelled["capabilities"] == labelled["capabilities"]
    assert unlabelled["prompt_versions"] == labelled["prompt_versions"]


def test_provenance_is_role_specific_and_serializable():
    def spec(model, loader, revision):
        return {"id": model, "loader": loader, "revision": revision,
                "dtype": "bfloat16", "device": "cpu"}
    specs = roles.resolve_roles({"roles": {
        "solver": spec("google/gemma-3-4b-it", "gemma3", "solver-revision"),
        "critic": spec("Qwen/Qwen3-4B-Instruct-2507", "qwen3", "critic-revision"),
        "validator": spec("google/gemma-3-4b-it", "gemma3", "solver-revision"),
    }})
    configured = {}
    for role, role_spec in specs.items():
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=role_spec["revision"], architectures=[role_spec["loader"]]), dtype=torch.bfloat16, device=torch.device("cpu"))
        agent = Solver(model, object())
        roles.configure_agent(agent, role_spec)
        configured[role] = agent
    result = provenance.build_resolved_config(
        {"roles": specs, "collection": {"active_conditions": ["natural"]}},
        configured["solver"].model, **configured, protocol_version="v2")
    prov = result["provenance"]
    assert prov["roles"]["solver"]["resolved_revision"] == "solver-revision"
    assert prov["roles"]["critic"]["resolved_revision"] == "critic-revision"
    assert prov["active_conditions"] == ["natural"]
    yaml.safe_dump(result)
    json.dumps(result)


def test_environment_metadata_without_git(monkeypatch):
    monkeypatch.setattr(provenance.subprocess, "run", Mock(side_effect=OSError("no git")))
    assert provenance.environment_metadata()["git_dirty"] is None


def test_role_metadata_records_runtime_chat_template_kwargs():
    spec = roles.resolve_role_spec("critic", {"id": "m", "loader": "qwen3", "revision": "r",
                                              "dtype": "bfloat16", "device": "cpu",
                                              "chat_template_kwargs": {"enable_thinking": False}},
                                   default_max_new_tokens=8)
    agent = Solver(SimpleNamespace(config=SimpleNamespace()), object())
    assert provenance.role_metadata(agent)["chat_template_kwargs"] == {}
    roles.configure_agent(agent, spec)
    assert provenance.role_metadata(agent)["chat_template_kwargs"] == {"enable_thinking": False}
    # the value actually applied at generation time is what gets recorded
    agent.chat_template_kwargs = {}
    assert provenance.role_metadata(agent)["chat_template_kwargs"] == {}
