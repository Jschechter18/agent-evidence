"""Chunked, resumable collection: prefix rule, identity checks, durability
order, and equivalence of chunked and uninterrupted final artifacts.

``run_question`` is faked; everything from ``collect_examples`` through
``save_collection_artifacts`` runs for real on temporary directories.
"""
import json
from pathlib import Path

import pytest
import torch
import yaml

from mas_sae.agents.critic import CriticBlindAnswerError
from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments import collection, collection_progress, provenance
from mas_sae.experiments.collection_artifacts import save_collection_artifacts
from mas_sae.experiments.collection_config import load_collection_config
from mas_sae.experiments.records import read_jsonl
from mas_sae.experiments.reproducibility import config_sha256
from mas_sae.sae.dataloader import create_sae_dataloader

SITE = "model.language_model.layers.8"
IDS = [f"q{index}" for index in range(5)]
EXCLUDED = {"q2"}


def fake_run_question(**kwargs):
    question_id = kwargs["question_id"]
    if question_id in EXCLUDED:
        raise CriticBlindAnswerError("no usable blind answer")
    value = float(kwargs["seed"])
    return {
        "attempt1_activations": {SITE: torch.full((1, 3), value)},
        "episodes": [{
            "record": {"episode_id": f"{question_id}__natural", "question_id": question_id,
                       "critic_condition": "natural", "seed": kwargs["seed"],
                       "solver_accepted_feedback": True},
            "attempt2_activations": {SITE: torch.full((1, 3), value + 0.5)},
        }],
    }


def examples(ids=IDS):
    return [{"id": qid, "question": qid, "paragraphs": [], "answer": "A"} for qid in ids]


def collect(ids, offset=0):
    return collection.collect_examples(
        examples=examples(ids), source_split="train", model=object(), solver=object(),
        critic=object(), validator=object(), candidate_sites=[SITE], base_seed=42,
        question_offset=offset,
    )


def identity(**changes):
    return {
        "created_at_utc": "now", "config_sha256": "c", "manifest_sha256": "m",
        "dataset_revision": "d", "model_revisions": {"solver": {"resolved_revision": "r"}},
        "git_commit": "g", "git_diff_sha256": "diff", "untracked_code_sha256": "u",
        "package_versions": {"torch": "t"},
        "gpus": ["A10G"], **changes,
    }


def run_chunked(progress, chunk_size, *, stop_after=None):
    """Drive the chunk loop the way ``collect_activations.main`` does."""
    start_position = collection_progress.resume_position(progress, IDS)
    for committed, start in enumerate(range(start_position, len(IDS), chunk_size)):
        if stop_after is not None and committed == stop_after:
            return
        end = min(start + chunk_size, len(IDS))
        collection_progress.write_chunk(
            progress, start=start, question_ids=IDS[start:end],
            candidate_sites=[SITE], result=collect(IDS[start:end], offset=start),
        )


@pytest.fixture(autouse=True)
def fake_generation(monkeypatch):
    monkeypatch.setattr(collection, "run_question", fake_run_question)


def save(result, tmp_path, name):
    save_collection_artifacts(
        activation_root=tmp_path / "activations", result_root=tmp_path / "results",
        run_name=name, source_split="train", candidate_sites=[SITE],
        attempt1_by_site=result["attempt1_by_site"], attempt2_by_site=result["attempt2_by_site"],
        records=result["records"], resolved_config={}, exclusions=result["exclusions"],
    )


def test_offset_keeps_global_seed_and_exclusion_index():
    result = collect(IDS[2:4], offset=2)
    assert result["exclusions"][0]["question_index"] == 2
    assert result["exclusions"][0]["seed"] == 44
    assert result["records"][0]["seed"] == 45
    assert result["records"][0]["attempt1_activation_index"] == 0


def test_interrupted_resumed_run_matches_uninterrupted_artifacts(tmp_path):
    save(collect(IDS), tmp_path, "uninterrupted")

    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    run_chunked(progress, chunk_size=2, stop_after=1)
    assert collection_progress.resume_position(progress, IDS) == 2
    # Resume with a different chunk size: positions, not K, define chunks.
    run_chunked(progress, chunk_size=3)
    save(collection_progress.merge_chunks(progress, question_ids=IDS, candidate_sites=[SITE]),
         tmp_path, "chunked")

    for name in ("interactions.jsonl", "exclusions.jsonl"):
        assert read_jsonl(tmp_path / "results/uninterrupted/train" / name) \
            == read_jsonl(tmp_path / "results/chunked/train" / name)
    indices = [(row["question_id"], row["attempt1_activation_index"])
               for row in read_jsonl(tmp_path / "results/chunked/train/interactions.jsonl")]
    assert indices == [("q0", 0), ("q1", 1), ("q3", 2), ("q4", 3)]
    one = ActivationStore(tmp_path / "activations/uninterrupted/layer_08")
    two = ActivationStore(tmp_path / "activations/chunked/layer_08")
    for split in ("train_attempt1", "train_attempt2", "train"):
        assert torch.equal(one.load_activations(split), two.load_activations(split))


def test_all_excluded_chunk_merges(tmp_path):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    for start, end in ((0, 2), (2, 3), (3, 5)):
        collection_progress.write_chunk(progress, start=start, question_ids=IDS[start:end],
                                        candidate_sites=[SITE], result=collect(IDS[start:end], start))
    merged = collection_progress.merge_chunks(progress, question_ids=IDS, candidate_sites=[SITE])
    assert [row["question_id"] for row in merged["exclusions"]] == ["q2"]
    assert torch.cat(merged["attempt1_by_site"][SITE]).shape == (4, 3)


def test_stale_tmp_is_ignored_and_replaced(tmp_path):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    (progress / "chunk_00000000.tmp").mkdir()
    (progress / "chunk_00000000.tmp" / "junk").write_text("partial")
    assert collection_progress.resume_position(progress, IDS) == 0
    run_chunked(progress, chunk_size=5)
    assert not (progress / "chunk_00000000.tmp").exists()
    assert collection_progress.resume_position(progress, IDS) == 5


@pytest.mark.parametrize("chunks, message", [
    ([(1, IDS[1:2])], "exact prefix"),
    ([(0, ["other"])], "does not match"),
])
def test_prefix_violations_refuse(tmp_path, chunks, message):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    for start, ids in chunks:
        chunk_dir = progress / f"chunk_{start:08d}"
        chunk_dir.mkdir()
        (chunk_dir / "metadata.json").write_text(
            json.dumps({"start": start, "end": start + len(ids), "question_ids": ids}))
    with pytest.raises(RuntimeError, match=message):
        collection_progress.resume_position(progress, IDS)


def test_resume_position_reads_metadata_only(tmp_path, monkeypatch):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    run_chunked(progress, chunk_size=2, stop_after=2)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("tensors loaded"))
    assert collection_progress.resume_position(progress, IDS) == 4


def test_merge_refuses_incomplete_run(tmp_path):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    run_chunked(progress, chunk_size=2, stop_after=1)
    with pytest.raises(RuntimeError, match="2/5"):
        collection_progress.merge_chunks(progress, question_ids=IDS, candidate_sites=[SITE])


def test_chunk_is_renamed_only_after_its_files_are_synced(tmp_path, monkeypatch):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    events = []
    monkeypatch.setattr(collection_progress, "_fsync", lambda path: events.append(("fsync", path.name)))
    original_replace = Path.replace
    def replace(self, target):
        events.append(("rename", self.name))
        return original_replace(self, target)
    monkeypatch.setattr(Path, "replace", replace)

    collection_progress.write_chunk(progress, start=0, question_ids=IDS[:2],
                                    candidate_sites=[SITE], result=collect(IDS[:2]))

    rename = events.index(("rename", "chunk_00000000.tmp"))
    synced_before = {name for kind, name in events[:rename] if kind == "fsync"}
    assert {"metadata.json", "records.jsonl", "exclusions.jsonl", "activations.pt",
            "chunk_00000000.tmp"} <= synced_before
    assert ("fsync", "_progress") in events[rename:]


@pytest.mark.parametrize("key", collection_progress.IDENTITY_KEYS)
def test_identity_change_refuses_resume(tmp_path, key):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    with pytest.raises(RuntimeError, match=key):
        collection_progress.check_identity(progress, identity(**{key: "changed"}))


def test_identity_ignores_creation_time(tmp_path):
    progress = tmp_path / "_progress"
    collection_progress.initialize(progress, identity())
    collection_progress.check_identity(progress, identity(created_at_utc="later"))


def test_start_guards(tmp_path):
    final, progress = tmp_path / "results/run/train", tmp_path / "activations/run/_progress"
    with pytest.raises(FileNotFoundError, match="requires existing provenance"):
        collection_progress.ensure_can_start(resume=True, final_output_dir=final, progress=progress)

    collection_progress.initialize(progress, identity())
    with pytest.raises(FileExistsError, match="pass --resume"):
        collection_progress.ensure_can_start(resume=False, final_output_dir=final, progress=progress)
    collection_progress.ensure_can_start(resume=True, final_output_dir=final, progress=progress)

    final.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing to resume"):
        collection_progress.ensure_can_start(resume=True, final_output_dir=final, progress=progress)
    with pytest.raises(FileExistsError):
        collection_progress.ensure_can_start(resume=False, final_output_dir=final, progress=progress)


def test_progress_provenance_fields(monkeypatch):
    monkeypatch.setattr(provenance.artifacts, "get_git_diff_sha256", lambda: "diff")
    monkeypatch.setattr(provenance.artifacts, "get_untracked_sha256", lambda *paths: "u")
    config = {"dataset": {"revision": "d"}, "collection": {"seed": 42},
              "output": {"run_name": "a", "chunk_size": 3}}
    resolved = {"provenance": {
        "git_commit": "g", "package_versions": {"torch": "t"}, "environment": {"gpus": ["A10G"]},
        "roles": {"solver": {"requested_revision": "r1", "resolved_revision": "r2", "dtype": "x"}},
    }}
    built = provenance.build_collection_progress_provenance(config, resolved, manifest_sha256="m")
    assert set(collection_progress.IDENTITY_KEYS) <= set(built)
    assert built["config_sha256"] == config_sha256(config)
    assert built["model_revisions"] == {"solver": {"requested_revision": "r1", "resolved_revision": "r2"}}
    assert (built["git_diff_sha256"], built["gpus"]) == ("diff", ["A10G"])


def test_config_hash_ignores_output_section():
    base = {"dataset": {"revision": "d"}, "collection": {"seed": 42}}
    assert config_sha256({**base, "output": {"run_name": "a", "chunk_size": 3}}) \
        == config_sha256({**base, "output": {"run_name": "b", "chunk_size": 250}})


def test_manifest_hash_sees_split_reassignment():
    entry = {"question_id": "q0", "hop_group": "2hop", "experiment_split": "discovery"}
    assert collection_progress.manifest_sha256([entry]) \
        != collection_progress.manifest_sha256([{**entry, "experiment_split": "validation"}])


@pytest.mark.parametrize("chunk_size, valid", [(3, True), (0, False), (True, False), ("3", False)])
def test_chunk_size_validation(tmp_path, chunk_size, valid):
    config = yaml.safe_load(open("configs/collection/v2/smoke_train.yaml"))
    config["output"]["chunk_size"] = chunk_size
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    if valid:
        assert load_collection_config(path)["output"]["chunk_size"] == chunk_size
    else:
        with pytest.raises(ValueError, match="chunk_size"):
            load_collection_config(path)


def test_untracked_hash_sees_new_and_edited_files(tmp_path, monkeypatch):
    import subprocess
    from mas_sae.experiments import artifacts
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    empty = artifacts.get_untracked_sha256("src")
    (tmp_path / "src" / "new.py").write_text("a = 1\n")
    first = artifacts.get_untracked_sha256("src")
    (tmp_path / "src" / "new.py").write_text("a = 2\n")
    edited = artifacts.get_untracked_sha256("src")
    assert len({empty, first, edited}) == 3
    (tmp_path / "results.txt").write_text("outside src")
    assert artifacts.get_untracked_sha256("src") == edited


def test_source_splits_resume_independently_and_share_sae_directory(tmp_path):
    activation_root, result_root = tmp_path / "activations", tmp_path / "results"
    run_name = "shared"
    ids_by_split = {"train": ["train0", "train1"], "validation": ["val0", "val1"]}
    progress_by_split = {}
    identities = {}

    def write(split, start):
        ids = ids_by_split[split][start:start + 1]
        result = collection.collect_examples(
            examples=examples(ids), source_split=split, model=object(), solver=object(),
            critic=object(), validator=object(), candidate_sites=[SITE], base_seed=42,
            question_offset=start,
        )
        collection_progress.write_chunk(progress_by_split[split], start=start,
                                        question_ids=ids, candidate_sites=[SITE], result=result)

    for split, ids in ids_by_split.items():
        progress = collection_progress.progress_dir(activation_root, run_name, split)
        assert progress == activation_root / run_name / "_progress" / split
        progress_by_split[split] = progress
        identities[split] = identity(config_sha256=config_sha256({"dataset": {"source_split": split}}))
        collection_progress.ensure_can_start(
            resume=False, final_output_dir=result_root / run_name / split, progress=progress)
        collection_progress.initialize(progress, identities[split])
        write(split, 0)
        assert collection_progress.resume_position(progress, ids) == 1

    for split, ids in ids_by_split.items():
        progress = progress_by_split[split]
        collection_progress.ensure_can_start(
            resume=True, final_output_dir=result_root / run_name / split, progress=progress)
        collection_progress.check_identity(progress, identities[split])
        other = "validation" if split == "train" else "train"
        with pytest.raises(RuntimeError, match="config_sha256"):
            collection_progress.check_identity(progress, identities[other])
        write(split, 1)
        assert collection_progress.resume_position(progress, ids) == 2
        merged = collection_progress.merge_chunks(progress, question_ids=ids, candidate_sites=[SITE])
        save_collection_artifacts(
            activation_root=activation_root, result_root=result_root, run_name=run_name,
            source_split=split, candidate_sites=[SITE],
            attempt1_by_site=merged["attempt1_by_site"], attempt2_by_site=merged["attempt2_by_site"],
            records=merged["records"], resolved_config={}, exclusions=merged["exclusions"],
        )
        if split == "train":
            assert collection_progress.resume_position(progress_by_split["validation"], ids_by_split["validation"]) == 1

    location = activation_root / run_name / "layer_08"
    assert {p.name for p in location.glob("*.pt")} == {
        f"{split}{suffix}.pt" for split in ids_by_split for suffix in ("", "_attempt1", "_attempt2")}
    for split, ids in ids_by_split.items():
        loader = create_sae_dataloader(
            batch_size=4, split=split, num_workers=0, location=location, shuffle=False,
        )
        assert torch.equal(next(iter(loader)), torch.tensor([[42.] * 3, [43.] * 3, [42.5] * 3, [43.5] * 3]))
        records = read_jsonl(result_root / run_name / split / "interactions.jsonl")
        assert [r["question_id"] for r in records] == ids
        # Layer-selection contract (origin/issue-38 load_real_layer_split): rows of
        # <split>_attempt2.pt are indexed by attempt2_activation_index; A1 likewise.
        store = ActivationStore(location)
        attempt1, attempt2 = (store.load_activations(f"{split}_attempt{n}") for n in (1, 2))
        for record in records:
            assert torch.equal(attempt1[record["attempt1_activation_index"]],
                               torch.full((3,), float(record["seed"])))
            assert torch.equal(attempt2[record["attempt2_activation_index"]],
                               torch.full((3,), record["seed"] + 0.5))
        with pytest.raises(FileExistsError):
            collection_progress.ensure_can_start(
                resume=True, final_output_dir=result_root / run_name / split, progress=progress_by_split[split])
