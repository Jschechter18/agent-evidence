"""Scientific protocol tests; all generations and activation hooks are fake."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from mas_sae.agents.critic import Critic, CriticBlindAnswerError
from mas_sae.agents.solver import Solver
from mas_sae.experiments import pipeline, collection, collection_artifacts
from mas_sae.experiments.conditions import resolve_conditions
from mas_sae.models import roles
from mas_sae.evaluation.behavior import classify_behavior
from mas_sae.experiments.collection_config import load_collection_config

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
    monkeypatch.setattr(pipeline, "choose_incorrect_answer", forbidden)
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


@pytest.mark.parametrize("value", [[], ["natural", "natural"], ["unknown"], "natural"])
def test_invalid_conditions(value):
    with pytest.raises(ValueError):
        resolve_conditions(value)


@pytest.mark.parametrize("a1,a2,adv,relation,outcome", [
    ("Paris", "Paris", "Paris", "same_position", "no_conflict"),
    ("Paris", "Paris", "London", "different_nonrefusal_candidate", "retained_solver_a1"),
    ("Paris", "London", "London", "different_nonrefusal_candidate", "adopted_critic"),
    ("Paris", "Rome", "London", "different_nonrefusal_candidate", "third_answer_revision"),
    ("Paris", "Rome", None, "ambiguous_or_unresolved", "ambiguous"),
    ("Paris", "Rome", "I cannot determine the answer", "refusal_or_nonanswer", "ambiguous"),
    ("Paris", "Rome", "N/A", "refusal_or_nonanswer", "ambiguous"),
    ("Paris", "Rome", "Paris, France", "ambiguous_or_unresolved", "ambiguous"),
    ("Paris", "Maybe London", "London", "different_nonrefusal_candidate", "ambiguous"),
])
def test_behavior(a1, a2, adv, relation, outcome):
    labels = classify_behavior(a1, a2, adv)
    assert labels["critic_position_relation"] == relation
    assert labels["solver_behavior"] == outcome
    if outcome in {"ambiguous", "no_conflict"}:
        assert labels["direct_critic_adoption"] is None


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


def test_provenance_is_role_specific_and_serializable():
    configured = {}
    for role, spec in specs().items():
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=spec["revision"], architectures=[spec["loader"]]), dtype=torch.bfloat16, device=torch.device("cpu"))
        agent = Solver(model, object())
        roles.configure_agent(agent, spec)
        configured[role] = agent
    result = collection_artifacts.build_resolved_config(
        {"roles": specs(), "collection": {"active_conditions": ["natural"]}},
        configured["solver"].model, **configured, protocol_version="v2")
    prov = result["provenance"]
    assert prov["roles"]["solver"]["resolved_revision"] == "solver-revision"
    assert prov["roles"]["critic"]["resolved_revision"] == "critic-revision"
    assert prov["active_conditions"] == ["natural"]
    yaml.safe_dump(result)
    json.dumps(result)


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
    assert all(f["a1_exists"] and not f["a2_exists"] for f in failures)
    if not all_failed:
        saved = json.loads((out / "interactions.jsonl").read_text())
        assert saved["attempt1_activation_index"] == saved["attempt2_activation_index"] == 0
        assert saved["sae_attempt2_index"] == 1
    else:
        assert not (tmp_path / "activations").exists()


def test_production_template_requires_team_decisions():
    with pytest.raises(ValueError):
        load_collection_config("configs/collection/v2/natural_production.yaml")


def test_malformed_feedback_stays_ambiguous():
    assert classify_behavior("Paris", "London", "London", usable=False)["solver_behavior"] == "ambiguous"


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


def test_explicit_role_config_roundtrip(tmp_path):
    config = {"roles": specs(), "dataset": {"source_split": "train", "num_questions": 2},
              "collection": {"layers": [8], "seed": 42, "protocol_version": "v2", "active_conditions": ["natural"]},
              "output": {"run_name": "unit"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_collection_config(path) == config


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


def test_integrity_rejects_wrong_indices_before_writes(tmp_path):
    with pytest.raises(ValueError, match="indices"):
        collection_artifacts.save_collection_artifacts(
            activation_root=tmp_path / "activations", result_root=tmp_path / "results",
            run_name="bad", source_split="train", candidate_sites=[SITE],
            attempt1_by_site={SITE: [torch.ones(1, 4)]},
            attempt2_by_site={SITE: [torch.ones(1, 4)]},
            records=[{"question_id": "q", "attempt1_activation_index": 9, "attempt2_activation_index": 0}],
            resolved_config={})
    assert not (tmp_path / "activations").exists()


@pytest.mark.parametrize("value", [None, [], "natural", {}, ["bad"], ["natural", "natural"]])
def test_explicit_invalid_condition_config_fails(tmp_path, value):
    config = yaml.safe_load(open("configs/collection/v2/smoke_train.yaml"))
    config["collection"]["active_conditions"] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_collection_config(path)
    with pytest.raises(ValueError):
        resolve_conditions(value)


@pytest.mark.parametrize("value", [["natural"], ["controlled_correct"], ["controlled_incorrect"]])
def test_explicit_valid_condition_config(tmp_path, value):
    config = yaml.safe_load(open("configs/collection/v2/smoke_train.yaml"))
    config["collection"]["active_conditions"] = value
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_collection_config(path)
    assert [c.value for c in resolve_conditions(loaded["collection"]["active_conditions"])] == value


def test_omitted_conditions_preserve_legacy():
    config = load_collection_config("configs/collection/v2/smoke_train.yaml")
    assert "active_conditions" not in config["collection"]
    assert [c.value for c in resolve_conditions()] == ["natural", "controlled_correct", "controlled_incorrect"]


@pytest.mark.parametrize("refusal", [
    "I do not have enough information to answer.",
    "I'm sorry, I can't answer that.",
    "I’m sorry, I can’t answer that.",
    "The context does not identify the person.",
    "There is no evidence in the passage.",
])
def test_refusal_copying_is_not_candidate_adoption(refusal):
    labels = classify_behavior("Paris", refusal, refusal)
    assert labels["critic_position_relation"] == "refusal_or_nonanswer"
    assert labels["solver_behavior"] == "ambiguous"
    assert labels["direct_critic_adoption"] is None
    assert labels["solver_copied_nonanswer"] is True
    assert labels["critic_textual_relation"] == "different"


def test_lexical_difference_does_not_claim_semantic_disagreement():
    labels = classify_behavior("United States", "USA", "USA")
    assert labels["critic_position_relation"] == "different_nonrefusal_candidate"
    assert labels["behavior_schema_version"] == "lexical_v2"
    assert labels["behavior_matching_basis"] == "normalized_text_not_semantic"


@pytest.mark.parametrize("loader,prefix", [
    ("gemma3", "model.language_model.layers"), ("qwen3", "model.layers")])
def test_solver_site_resolution(loader, prefix):
    from mas_sae.activations.sites import resolve_solver_sites
    model = SimpleNamespace(config=SimpleNamespace(model_type=loader),
        named_modules=lambda: [(f"{prefix}.0", object()), (f"{prefix}.2", object())])
    assert resolve_solver_sites(model, loader, [2, 0]) == [f"{prefix}.2", f"{prefix}.0"]
    with pytest.raises(ValueError, match="not found"):
        resolve_solver_sites(model, loader, [1])


@pytest.mark.parametrize("loader,model_type", [("unsupported", "unsupported"), ("gemma3", "qwen3"), ("qwen3", None)])
def test_solver_site_resolution_rejects_unknown_or_mismatched_architecture(loader, model_type):
    from mas_sae.activations.sites import resolve_solver_sites
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    with pytest.raises(ValueError, match="Unsupported|requires architecture"):
        resolve_solver_sites(model, loader, [0])


def test_collection_script_rejects_bad_solver_sites_before_generation(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("collect_script", Path("scripts/collect_activations.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = {"roles": specs(), "dataset": {"source_split": "train", "num_questions": 1},
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


def test_blind_raw_output_does_not_leak_between_questions():
    _, critic, _, _ = agents()
    critic._generate = Mock(side_effect=['{"answer": "First"}', '{"answer": "Second"}', 'broken'])
    assert critic.answer_blind("Q1", []) == "First"
    assert critic.last_blind_raw_output == '{"answer": "First"}'
    assert critic.answer_blind("Q2", []) == "Second"
    assert critic.last_blind_raw_output == '{"answer": "Second"}'
    with pytest.raises(CriticBlindAnswerError):
        critic.answer_blind("Q3", [])
    assert critic.last_blind_raw_output == 'broken'


def test_production_summary_does_not_turn_invalid_feedback_into_rejection():
    row = {"solver_accepted_feedback": False,
           **classify_behavior("Paris", "Paris", "London", usable=False)}
    summary = collection_artifacts.behavioral_summary([row], production=True)
    assert summary["solver_behavior_counts"] == {"ambiguous": 1}
    assert summary["critic_position_counts"] == {"ambiguous_or_unresolved": 1}
    assert "rejected" not in summary
    assert summary["legacy_acceptance_counts"]["rejected"] == 1
    assert row["direct_critic_adoption"] is None
    assert collection_artifacts.behavioral_summary([row], production=False)["rejected"] == 1
