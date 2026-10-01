"""Natural-critic validation: frozen inputs, shared Attempt 1, no gold leakage.

Real Solver/Critic prompt code runs against a scripted fake backend; no
model is loaded and nothing is downloaded.
"""
import copy
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml
from transformers.utils.chat_template_utils import render_jinja_template

from mas_sae.agents.critic import NATURAL_BLIND_PROMPT, NATURAL_COMPARE_PROMPT, Critic
from mas_sae.agents.solver import Solver
from mas_sae.experiments import natural_validation as nv
from mas_sae.experiments.artifacts import sha256_text
from mas_sae.models import roles

REPO = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO / "configs/validation/natural_critics_gemma12b_qwen14b.yaml"
FIXTURE = REPO / "src/tests/fixtures/qwen3_14b_chat_template.json"


# ---------------------------------------------------------------------------
# Scripted backend: real prompts in, scripted text out, every call recorded
# ---------------------------------------------------------------------------

class Inputs(dict):
    def to(self, device):
        return self


def prompt_text(messages):
    content = messages[0]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def stage_of(prompt):
    if "Previous answer:" in prompt:
        return "revise"
    if "Your own answer:" in prompt:
        return "compare"
    if "answer the question independently" in prompt:
        return "blind"
    return "solve"


class Backend:
    """One processor + model pair; ``script(stage, question)`` returns text."""

    def __init__(self, name, script, calls):
        self.name, self.script, self.calls = name, script, calls
        self.model = Mock(device="cpu")
        self.model.generate.side_effect = self._generate
        self.pending = None

    def apply_chat_template(self, messages, **kwargs):
        prompt = prompt_text(messages)
        question = re.search(r"Question: (.*)\n", prompt).group(1)
        stage = stage_of(prompt)
        self.pending = self.script(stage, question)
        self.calls.append({"backend": self.name, "stage": stage, "question": question,
                           "prompt": prompt, "template_kwargs": kwargs})
        return Inputs(input_ids=torch.tensor([[1, 2, 3]]))

    def _generate(self, **kwargs):
        self.calls[-1]["generate_kwargs"] = {k: v for k, v in kwargs.items() if k != "input_ids"}
        return torch.tensor([[1, 2, 3, 4]])

    def decode(self, tokens, **kwargs):
        return self.pending


QUESTIONS = ["Who built bridge zero?", "Who built bridge one?", "Who built bridge two?"]
IDS = ["2hop__10_20", "3hop1__11_21_31", "4hop2__12_22_32_42"]


def make_examples():
    return [{"id": qid, "question": q, "answer": f"GOLDSENTINEL{i}", "answer_aliases": [f"ALIASSENTINEL{i}"],
             "answerable": True,
             "paragraphs": [{"idx": 0, "title": f"Bridge {i}", "paragraph_text": f"Bridge {i} was built long ago."}]}
            for i, (qid, q) in enumerate(zip(IDS, QUESTIONS))]


# Solver A1 per question; arm scripts (blind, advocated); Solver A2 per arm.
A1 = {QUESTIONS[0]: "Alpha", QUESTIONS[1]: "Bravo", QUESTIONS[2]: "Charlie"}
SAME = {QUESTIONS[0]: ("Alpha", "Alpha"), QUESTIONS[1]: ("Delta", "Delta"), QUESTIONS[2]: (None, None)}
CROSS = {QUESTIONS[0]: ("Echo", "Alpha"), QUESTIONS[1]: ("Foxtrot", "Foxtrot"), QUESTIONS[2]: ("Golf", "Hotel")}


def gemma_script(stage, question):
    if stage == "solve":
        return A1[question]
    if stage == "revise":
        return "Omega"  # a third answer, never the critic's
    blind, advocated = SAME[question]
    if stage == "blind":
        return "no json here" if blind is None else json.dumps({"answer": blind})
    return json.dumps({"verdict": "disagree", "advocated_answer": advocated, "explanation": "x"})


def qwen_script(stage, question):
    blind, advocated = CROSS[question]
    if stage == "blind":
        return json.dumps({"answer": blind})
    return json.dumps({"verdict": "disagree", "advocated_answer": advocated, "explanation": "x"})


def load_config(tmp_path, mutate=None):
    """The committed config, optionally mutated, loaded through the real validator."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["manifest"]["path"] = str(REPO / raw["manifest"]["path"])
    if mutate:
        mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return nv.load_validation_config(path)


def build_agents(config, calls):
    gemma = Backend("gemma", gemma_script, calls)
    qwen = Backend("qwen", qwen_script, calls)
    specs = config["roles"]
    solver = Solver(gemma.model, gemma)
    critics = {"same_model": Critic(gemma.model, gemma, blind_then_compare=True),
               "cross_model": Critic(qwen.model, qwen, blind_then_compare=True)}
    agents = {"solver": solver, "same_model_critic": critics["same_model"],
              "cross_model_critic": critics["cross_model"]}
    for role, agent in agents.items():
        roles.configure_agent(agent, specs[role])
    return solver, critics, agents


def run(tmp_path, config, calls, *, resume=False, out="run"):
    solver, critics, agents = build_agents(config, calls)
    examples = make_examples()
    manifest = [{"position": i, "question_id": qid} for i, qid in enumerate(IDS)]
    return nv.run_natural_validation(config=config, output_dir=tmp_path / out, solver=solver, critics=critics,
                                     agents=agents, examples=examples, manifest=manifest,
                                     manifest_sha256="0" * 64, resume=resume)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------

def test_committed_config_pins_models_dataset_manifest_and_qwen_non_thinking(tmp_path):
    config = load_config(tmp_path)
    r = config["roles"]
    assert config["dataset"]["revision"] == "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
    assert (r["solver"]["id"], r["solver"]["revision"]) == (
        "google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80")
    assert (r["cross_model_critic"]["id"], r["cross_model_critic"]["revision"]) == (
        "Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18")
    assert all(spec["generation"]["do_sample"] is False for spec in r.values())
    assert roles.weight_identity(r["same_model_critic"]) == roles.weight_identity(r["solver"])
    assert r["cross_model_critic"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in r["solver"] and "chat_template_kwargs" not in r["same_model_critic"]
    manifest, digest = nv.load_manifest(config["manifest"]["path"], config["manifest"]["sha256"])
    assert digest == "9d99d2e70791dc39280c2c45488077baf9fd79b78c46783eefdb6baf7fe4f4dd"
    assert len(manifest) == 100


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c["roles"]["cross_model_critic"].update(revision=None), "immutable revision"),
    (lambda c: c["dataset"].update(repo="other/dataset"), "dataset.repo"),
    (lambda c: c["roles"]["same_model_critic"].update(revision="a" * 40), "Solver's checkpoint"),
    (lambda c: c["roles"]["same_model_critic"].update(dtype="float16"), "Solver's checkpoint"),
    (lambda c: c["roles"]["same_model_critic"].update(device="cuda:1"), "Solver's checkpoint"),
    (lambda c: c["roles"].pop("cross_model_critic"), "roles must specify exactly"),
    (lambda c: c["roles"].update(validator=copy.deepcopy(c["roles"]["solver"])), "roles must specify exactly"),
    (lambda c: c["validation"].update(protocol="controlled"), "validation.protocol"),
])
def test_config_rejects_unpinned_or_mismatched_settings(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        load_config(tmp_path, mutate)


def test_manifest_must_match_hash_and_be_well_formed(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([{"position": 0, "question_id": "a"}, {"position": 1, "question_id": "a"}]))
    with pytest.raises(ValueError, match="sha256"):
        nv.load_manifest(path, "0" * 64)
    with pytest.raises(ValueError, match="unique"):
        nv.load_manifest(path, nv.sha256_file(path))
    path.write_text(json.dumps([{"position": 1, "question_id": "a"}]))
    with pytest.raises(ValueError, match="positions"):
        nv.load_manifest(path, nv.sha256_file(path))


# ---------------------------------------------------------------------------
# Shared weights, isolated agent settings, Qwen non-thinking template
# ---------------------------------------------------------------------------

def test_same_model_critic_shares_solver_weights_without_changing_solver_settings(tmp_path, monkeypatch):
    config = load_config(tmp_path)
    monkeypatch.setattr(roles, "load_spec", Mock(side_effect=lambda spec: (object(), object())))
    loaded = roles.load_role_models(config["roles"])
    assert loaded["same_model_critic"] is loaded["solver"]
    assert loaded["cross_model_critic"] is not loaded["solver"]

    calls = []
    run(tmp_path, config, calls)
    for call in calls:
        is_solver = call["stage"] in {"solve", "revise"}
        assert call["generate_kwargs"]["max_new_tokens"] == (32 if is_solver else 256)
        assert call["generate_kwargs"]["do_sample"] is False
        expected = {"enable_thinking": False} if call["backend"] == "qwen" else {}
        extra = {k: v for k, v in call["template_kwargs"].items()
                 if k not in {"add_generation_prompt", "tokenize", "return_dict", "return_tensors"}}
        assert extra == expected


def test_qwen_template_honours_enable_thinking_false():
    template = json.loads(FIXTURE.read_text(encoding="utf-8"))["chat_template"]
    conversation = [[{"role": "user", "content": "Question?"}]]

    def render(**kwargs):
        return render_jinja_template(conversation, chat_template=template, add_generation_prompt=True,
                                     **kwargs)[0][0]

    empty_think_at_end = re.compile(r"<think>\s*</think>\s*$")
    assert empty_think_at_end.search(render(enable_thinking=False))
    assert not empty_think_at_end.search(render())
    assert not empty_think_at_end.search(render(enable_thinking=True))


# ---------------------------------------------------------------------------
# End-to-end run with the scripted backend
# ---------------------------------------------------------------------------

def test_attempt1_generated_once_and_reused_by_both_arms(tmp_path):
    config, calls = load_config(tmp_path), []
    run(tmp_path, config, calls)
    out = tmp_path / "run"

    solves = [c for c in calls if c["stage"] == "solve"]
    assert [c["question"] for c in solves] == QUESTIONS
    attempt1 = {row["question_id"]: row for row in read_rows(out / "attempt1.jsonl")}
    for arm in ("same_model", "cross_model"):
        for row in read_rows(out / f"arm_{arm}.jsonl"):
            saved = attempt1[row["question_id"]]
            assert row["solver_attempt_1"] == saved["solver_attempt_1"]
            assert row["attempt1_sha256"] == saved["attempt1_sha256"] == sha256_text(saved["solver_attempt_1"])
    for call in calls:
        if call["stage"] in {"compare", "revise"}:
            shown = re.search(r"(?:Proposed|Previous) answer: (.*)\n", call["prompt"]).group(1)
            assert shown == A1[call["question"]]

    # Resume reuses everything: no generation at all.
    resumed = []
    run(tmp_path, config, resumed, resume=True)
    assert resumed == []


def test_natural_critic_never_sees_gold_aliases_or_labels(tmp_path):
    config, calls = load_config(tmp_path), []
    run(tmp_path, config, calls)
    examples = {e["question"]: e for e in make_examples()}
    for call in calls:
        assert "SENTINEL" not in call["prompt"]
        if call["stage"] not in {"blind", "compare"}:
            continue
        example = examples[call["question"]]
        paragraphs = Solver.format_paragraphs(example["paragraphs"])
        if call["stage"] == "blind":
            expected = NATURAL_BLIND_PROMPT.format(paragraphs=paragraphs, question=example["question"])
        else:
            blind_answer = (SAME if call["backend"] == "gemma" else CROSS)[call["question"]][0]
            expected = NATURAL_COMPARE_PROMPT.format(paragraphs=paragraphs, question=example["question"],
                                                     blind_answer=blind_answer,
                                                     solver_answer=A1[call["question"]])
        assert call["prompt"] == expected


def test_blind_failure_is_recorded_and_other_arm_continues(tmp_path):
    config, calls = load_config(tmp_path), []
    summary = run(tmp_path, config, calls)
    same = {row["question_id"]: row for row in read_rows(tmp_path / "run/arm_same_model.jsonl")}
    cross = {row["question_id"]: row for row in read_rows(tmp_path / "run/arm_cross_model.jsonl")}
    assert same[IDS[2]]["status"] == "failed" and same[IDS[2]]["failed_stage"] == "blind"
    assert same[IDS[2]]["critic_blind_raw_output"] == "no json here"
    assert cross[IDS[2]]["status"] == "completed"
    assert (summary["arms"]["same_model"]["n_failed"], summary["arms"]["cross_model"]["n_failed"]) == (1, 0)


def test_summary_is_descriptive_with_denominators_and_separate_labels(tmp_path):
    config, calls = load_config(tmp_path), []
    summary = run(tmp_path, config, calls)
    text = json.dumps(summary)
    assert "p_value" not in text and "ci95" not in text and "mcnemar" not in text

    same, cross = summary["arms"]["same_model"], summary["arms"]["cross_model"]
    assert same["blind_kind"]["counts"]["stage_failed"] == 1 and same["blind_kind"]["n"] == 3
    assert same["blind_vs_solver_a1"]["counts"] == {"same": 1, "containment": 0, "different": 1,
                                                    "solver_a1_not_answer_like": 0}
    assert same["disagreement_outcomes"]["n"] == 1
    assert same["disagreement_outcomes"]["counts"]["retained_own_position"] == 1
    assert cross["disagreement_outcomes"]["n"] == 3
    assert cross["disagreement_outcomes"]["counts"] == {"retained_own_position": 1, "moved_to_solver_a1": 1,
                                                        "third_position": 1, "unclear": 0}
    for block in ("blind_kind", "blind_vs_solver_a1", "disagreement_outcomes"):
        assert same[block]["denominator"] and cross[block]["denominator"]

    paired = summary["paired_secondary"]
    assert paired["n_jointly_eligible"] == 1
    assert paired["n_eligible_by_arm"] == {"same_model": 1, "cross_model": 3}
    assert paired["counts"]["both_retained"] == 1

    # A changed Solver answer is not relabelled as accepting the critic.
    assert cross["solver_behavior_counts"].get("adopted_critic", 0) == 0
    assert cross["solver_behavior_counts"]["third_answer_revision"] >= 1
    assert "critic_position_relation_counts" in cross
    assert summary["n_solver_attempt_1_correct_post_hoc"] == 0


def test_resume_refuses_changed_configuration(tmp_path):
    config, calls = load_config(tmp_path), []
    run(tmp_path, config, calls)
    with pytest.raises(FileExistsError):
        run(tmp_path, config, [])
    for change in (lambda c: c["roles"]["cross_model_critic"].update(chat_template_kwargs={"enable_thinking": True}),
                   lambda c: c["roles"]["solver"]["generation"].update(max_new_tokens=64),
                   lambda c: c["validation"].update(seed=7),
                   lambda c: c["dataset"].update(revision="other")):
        changed = copy.deepcopy(config)
        change(changed)
        with pytest.raises(RuntimeError, match="different configuration"):
            run(tmp_path, changed, [], resume=True)
    # The output section alone may differ.
    renamed = {**config, "output": {"run_name": "other"}}
    run(tmp_path, renamed, [], resume=True)


def test_persisted_attempt1_mismatch_or_stray_rows_fail_loudly(tmp_path):
    config, calls = load_config(tmp_path), []
    solver, _, _ = build_agents(config, calls)
    examples = make_examples()
    path = tmp_path / "a1.jsonl"
    nv.generate_attempt1(solver=solver, examples=examples, base_seed=42, path=path)
    with pytest.raises(RuntimeError, match="does not match"):
        nv.generate_attempt1(solver=solver, examples=examples, base_seed=7, path=path)
    with pytest.raises(RuntimeError, match="outside the manifest"):
        nv.generate_attempt1(solver=solver, examples=examples[:2], base_seed=42, path=path)


@pytest.mark.parametrize("target", ["dataset", *nv.DEFAULT_TOKENS])
@pytest.mark.parametrize("revision", [None, "main", "refs/heads/main", "v1.0", "abcdef0", "g" * 40])
def test_config_requires_full_commit_revisions(tmp_path, target, revision):
    def mutate(config):
        section = config["dataset"] if target == "dataset" else config["roles"][target]
        section["revision"] = revision

    with pytest.raises(ValueError, match=rf"{target}\.revision.*immutable revision"):
        load_config(tmp_path, mutate)


@pytest.mark.parametrize("prompt", list(nv.PROMPT_VERSIONS))
def test_resume_refuses_changed_prompt_templates(tmp_path, monkeypatch, prompt):
    config = load_config(tmp_path)
    run(tmp_path, config, [])
    out = tmp_path / "run"
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    name, template = nv.PROMPT_VERSIONS[prompt]
    monkeypatch.setitem(nv.PROMPT_VERSIONS, prompt, (name, template + " Changed protocol."))
    calls = []
    with pytest.raises(RuntimeError, match="prompt templates"):
        run(tmp_path, config, calls, resume=True)
    assert calls == []
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_resume_refuses_missing_prompt_hashes(tmp_path):
    config = load_config(tmp_path)
    run(tmp_path, config, [])
    path = tmp_path / "run" / nv.PROVENANCE_FILE
    provenance = json.loads(path.read_text())
    del provenance["prompt_template_sha256"]
    path.write_text(json.dumps(provenance))
    calls = []
    with pytest.raises(RuntimeError, match="prompt templates"):
        run(tmp_path, config, calls, resume=True)
    assert calls == []


@pytest.mark.parametrize("filename", [nv.ATTEMPT1_FILE, *(nv.arm_file(arm) for arm in nv.ARM_ROLES)])
@pytest.mark.parametrize("contents", ["", '{"old": "row"}\n'])
def test_resume_refuses_jsonl_without_provenance(tmp_path, filename, contents):
    config = load_config(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    path = out / filename
    path.write_text(contents)
    calls = []
    with pytest.raises(RuntimeError, match="refusing to resume without provenance"):
        run(tmp_path, config, calls, resume=True)
    assert calls == []
    assert list(out.iterdir()) == [path]
    assert path.read_text() == contents


def test_resume_can_initialize_empty_directory(tmp_path):
    config = load_config(tmp_path)
    (tmp_path / "run").mkdir()
    calls = []
    run(tmp_path, config, calls, resume=True)
    assert (tmp_path / "run" / nv.PROVENANCE_FILE).exists()
    assert len([call for call in calls if call["stage"] == "solve"]) == len(QUESTIONS)


@pytest.mark.parametrize("arm", list(nv.ARM_ROLES))
@pytest.mark.parametrize("field, value", [
    ("solver_attempt_1", "Tampered answer"),
    ("solver_attempt_1", None),
    ("input_sha256", "0" * 64),
    ("context_sha256", "0" * 64),
    ("attempt1_sha256", "0" * 64),
    ("arm", "other"),
    ("protocol", "other"),
    ("position", 99),
    ("seed", -1),
])
def test_resume_refuses_tampered_arm_base_identity(tmp_path, arm, field, value):
    config = load_config(tmp_path)
    run(tmp_path, config, [])
    out = tmp_path / "run"
    path = out / nv.arm_file(arm)
    rows = read_rows(path)
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = {file.name: file.read_bytes() for file in out.iterdir()}
    calls = []
    with pytest.raises(RuntimeError, match="belongs to different inputs"):
        run(tmp_path, config, calls, resume=True)
    assert calls == []
    assert {file.name: file.read_bytes() for file in out.iterdir()} == before


@pytest.mark.parametrize("stage", ["attempt1", *nv.ARM_ROLES])
@pytest.mark.parametrize("invalid_id", ["duplicate", "stale", "missing", "", None, "   "])
@pytest.mark.parametrize("entrypoint", ["run", "stage"])
def test_invalid_persisted_ids_fail_before_generation(tmp_path, stage, invalid_id, entrypoint):
    config = load_config(tmp_path)
    run(tmp_path, config, [])
    out = tmp_path / "run"
    saved_attempt1 = read_rows(out / nv.ATTEMPT1_FILE)
    # Leave incomplete stages so a late validation would generate and append rows.
    for path in out.glob("*.jsonl"):
        path.write_text(json.dumps(read_rows(path)[0]) + "\n", encoding="utf-8")
    path = out / (nv.ATTEMPT1_FILE if stage == "attempt1" else nv.arm_file(stage))
    rows = read_rows(path)
    if invalid_id == "duplicate":
        rows.append(dict(rows[0]))
        match = "duplicate persisted question_id"
    elif invalid_id == "stale":
        rows[0]["question_id"] = "not-in-manifest"
        match = "outside the manifest"
    else:
        if invalid_id == "missing":
            del rows[0]["question_id"]
        else:
            rows[0]["question_id"] = invalid_id
        match = "non-empty question_id"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = {file.name: file.read_bytes() for file in out.iterdir()}
    calls = []
    with pytest.raises(RuntimeError, match=match):
        if entrypoint == "run":
            run(tmp_path, config, calls, resume=True)
        else:
            solver, critics, _ = build_agents(config, calls)
            kwargs = dict(solver=solver, examples=make_examples(),
                          base_seed=config["validation"]["seed"], path=path)
            if stage == "attempt1":
                nv.generate_attempt1(**kwargs)
            else:
                nv.run_critic_arm(arm=stage, critic=critics[stage],
                                  attempt1_rows=saved_attempt1, **kwargs)
    assert calls == []
    assert {file.name: file.read_bytes() for file in out.iterdir()} == before
