"""Natural-only production protocol, end to end through ``run_question``,
``collect_examples`` and ``save_collection_artifacts``.

All generations and activation hooks are fake. Per-module behaviour
(behavior labels, roles, provenance, sites, config) is tested beside the
module it belongs to; this file covers the protocol invariants that span
modules.
"""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mas_sae.agents.critic import Critic, CriticBlindAnswerError
from mas_sae.experiments import pipeline, collection, collection_artifacts

SITE = "model.language_model.layers.8"


class Capture:
    seen = []
    def __init__(self, model, sites):
        self.seen.append(model)
        self.activations = {site: torch.ones(1, 4) for site in sites}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def agents():
    prompts = []
    critic = Critic(object(), object(), blind_then_compare=True)
    def generate(prompt):
        prompts.append(prompt)
        if "independently" in prompt:
            return '{"answer": "Blind position"}'
        return '{"verdict": "agree", "advocated_answer": "First position", "explanation": "Reconsidered"}'
    critic._generate = generate
    solver = SimpleNamespace(solve=Mock(return_value="First position"), revise=Mock(return_value="First position"))
    validator = SimpleNamespace(validate=Mock(return_value=SimpleNamespace(is_correct=False, raw_output="NO")))
    return solver, critic, validator, prompts


def run(monkeypatch, conditions, typed=True):
    monkeypatch.setattr(pipeline, "MultiSiteCapture", Capture)
    solver, critic, validator, prompts = agents()
    model = object()
    result = pipeline.run_question(
        question_id="q", question="Question?", paragraphs=[],
        gold="GOLD_SENTINEL", aliases=["ALIAS_SENTINEL"], model=model,
        solver=solver, critic=critic, validator=validator,
        candidate_sites=[SITE], seed=42, active_conditions=conditions,
        type_checked_target=typed,
    )
    return result, solver, critic, prompts, model


def test_natural_only_has_no_control_work_and_no_gold_leak(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Dormant target generation"))
    monkeypatch.setattr(pipeline, "choose_type_checked_incorrect_target", forbidden)
    monkeypatch.setattr(pipeline, "choose_heuristic_incorrect_answer", forbidden)
    Capture.seen = []
    result, solver, critic, prompts, model = run(monkeypatch, ["natural"])
    assert len(result["episodes"]) == 1
    row = result["episodes"][0]["record"]
    assert row["critic_condition"] == "natural"
    assert row["critic_blind_answer"] == "Blind position"
    assert row["critic_advocated_answer"] == "First position"  # revision is allowed
    solver.solve.assert_called_once()
    solver.revise.assert_called_once()
    forbidden.assert_not_called()
    assert len(prompts) == 2
    assert "First position" not in prompts[0]
    assert "Proposed answer: First position" in prompts[1]
    for prompt in prompts:
        for secret in ("GOLD_SENTINEL", "ALIAS_SENTINEL", "controlled", "correctness_label"):
            assert secret not in prompt
    assert Capture.seen == [model, model]
    assert row["solver_behavior"] == "no_conflict"


@pytest.mark.parametrize("conditions", [["controlled_correct"], ["controlled_incorrect"],
    ["natural", "controlled_correct", "controlled_incorrect"]])
def test_controls_and_shared_a1(monkeypatch, conditions):
    result, solver, critic, prompts, model = run(monkeypatch, conditions, typed=False)
    solver.solve.assert_called_once()
    assert solver.revise.call_count == len(conditions)
    assert [ep["record"]["critic_condition"] for ep in result["episodes"]] == conditions
    assert all(ep["record"]["solver_attempt_1"] == "First position" for ep in result["episodes"])
    if "natural" not in conditions:
        assert not any("independently" in p for p in prompts)
    for ep in result["episodes"]:
        row = ep["record"]
        if row["critic_condition"] == "controlled_correct":
            assert row["critic_advocated_answer"] == "GOLD_SENTINEL"
        if row["critic_condition"] == "controlled_incorrect":
            assert row["critic_advocated_answer"] == "Not GOLD_SENTINEL"


def test_typed_control_can_run_alone_without_blind_call(monkeypatch):
    # Use the actual issue #54 target generator on a supported intermediate hop.
    monkeypatch.setattr(pipeline, "MultiSiteCapture", Capture)
    solver, critic, validator, prompts = agents()
    result = pipeline.run_question(question_id="q", question="Who is the spouse?",
        paragraphs=[], gold="Jane Doe", aliases=[], model=object(), solver=solver,
        critic=critic, validator=validator, candidate_sites=[SITE], seed=42,
        decomposition=[{"question": "Who performed?", "answer": "John Smith"},
                       {"question": "Spouse of #1", "answer": "Jane Doe"}],
        type_checked_target=True, active_conditions=["controlled_incorrect"])
    row = result["episodes"][0]["record"]
    assert row["critic_advocated_answer"] == "John Smith"
    assert row["controlled_target_source"] == "hop_intermediate"
    assert not any("independently" in p for p in prompts)


@pytest.mark.parametrize("all_failed", [False, True])
def test_exclusions_preserve_manifest_and_tensor_alignment(monkeypatch, tmp_path, all_failed):
    monkeypatch.setattr(pipeline, "MultiSiteCapture", Capture)
    solver, critic, validator, prompts = agents()
    real_blind = critic.answer_blind
    def blind(question, paragraphs):
        if question == "bad" or all_failed:
            raise CriticBlindAnswerError("missing answer")
        return real_blind(question, paragraphs)
    critic.answer_blind = blind
    examples = [{"id": q, "question": q, "paragraphs": [], "answer": "gold"} for q in ("good", "bad")]
    result = collection.collect_examples(examples=examples, source_split="train", model=object(),
        solver=solver, critic=critic, validator=validator, candidate_sites=[SITE], base_seed=42,
        active_conditions=["natural"])
    manifest = [{"question_id": q, "hop_group": "2hop"} for q in ("good", "bad")]
    summary = collection_artifacts.save_collection_artifacts(
        activation_root=tmp_path / "activations", result_root=tmp_path / "results",
        run_name="test", source_split="train", candidate_sites=[SITE],
        resolved_config={"collection": {"active_conditions": ["natural"]}},
        sampled_questions=manifest, **result)
    out = tmp_path / "results/test/train"
    assert json.loads((out / "sampled_questions.json").read_text()) == manifest
    assert summary["num_questions"] == (0 if all_failed else 1)
    assert summary["num_episodes"] == (0 if all_failed else 1)
    assert summary["num_excluded"] == (2 if all_failed else 1)
    failures = [json.loads(line) for line in (out / "exclusions.jsonl").read_text().splitlines()]
    assert all(f["stage"] == "critic_blind" and f["critic_exists"] for f in failures)
    assert all(f["a1_exists"] and not f["a2_exists"] for f in failures)
    collected = json.loads((out / "collected_question_ids.json").read_text())
    assert collected == ([] if all_failed else ["good"])
    if not all_failed:
        saved = json.loads((out / "interactions.jsonl").read_text())
        assert saved["attempt1_activation_index"] == saved["attempt2_activation_index"] == 0
        assert saved["sae_attempt2_index"] == 1
    else:
        assert not (tmp_path / "activations").exists()
        assert (out / "interactions.jsonl").read_text() == ""


def test_blind_raw_generation_survives_artifact_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "MultiSiteCapture", Capture)
    solver, critic, validator, prompts = agents()
    blind_raw = 'Preamble\n{"answer": "Blind position", "explanation": "Full blind reasoning"}\nTail'
    final_raw = '{"verdict": "agree", "advocated_answer": "First position", "explanation": "Review"}'
    critic._generate = Mock(side_effect=[blind_raw, final_raw])
    result = collection.collect_examples(
        examples=[{"id": "q", "question": "Question?", "paragraphs": [], "answer": "gold"}],
        source_split="train", model=object(), solver=solver, critic=critic, validator=validator,
        candidate_sites=[SITE], base_seed=42, active_conditions=["natural"])
    collection_artifacts.save_collection_artifacts(
        activation_root=tmp_path / "activations", result_root=tmp_path / "results",
        run_name="raw", source_split="train", candidate_sites=[SITE],
        resolved_config={"collection": {"active_conditions": ["natural"]}}, **result)
    row = json.loads((tmp_path / "results/raw/train/interactions.jsonl").read_text())
    assert row["critic_blind_answer"] == "Blind position"
    assert row["critic_blind_raw_output"] == blind_raw
    assert row["critic_raw_output"] == final_raw
    assert row["critic_advocated_answer"] == "First position"
    assert row["solver_attempt_1"] == row["solver_attempt_2"] == "First position"


def test_collection_script_rejects_bad_solver_sites_before_generation(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("collect_script", Path("scripts/collect_activations.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    def role(model, loader):
        return {"id": model, "loader": loader, "revision": None, "dtype": "bfloat16", "device": "cpu"}
    config = {"roles": {"solver": role("google/gemma-3-4b-it", "gemma3"),
                        "critic": role("Qwen/Qwen3-4B-Instruct-2507", "qwen3"),
                        "validator": role("google/gemma-3-4b-it", "gemma3")},
              "dataset": {"source_split": "train", "num_questions": 1},
              "collection": {"layers": [0], "seed": 42, "active_conditions": ["natural"]},
              "output": {"run_name": "unused"}}
    model = SimpleNamespace(config=SimpleNamespace(model_type="unsupported"))
    monkeypatch.setattr(script, "parse_args", lambda: SimpleNamespace(config="unused"))
    monkeypatch.setattr(script, "load_collection_config", lambda path: config)
    monkeypatch.setattr(script, "ensure_output_available", lambda path: None)
    monkeypatch.setattr(script, "load_role_models", lambda specs: {role: (model, object()) for role in specs})
    collect = Mock(side_effect=AssertionError("Must not generate"))
    monkeypatch.setattr(script, "collect_examples", collect)
    with pytest.raises(ValueError, match="requires architecture"):
        script.main()
    collect.assert_not_called()
