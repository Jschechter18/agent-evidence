"""Durable, resumable state of a chunked collection run.

A long collection commits its questions in chunks under
``<activation_root>/<run_name>/_progress/<source_split>/``. Each chunk is written to
``chunk_<start>.tmp/`` and renamed once every file is on disk, so a
committed chunk is never partial and a ``.tmp`` directory is never read.

A chunk holds ``metadata.json`` (start, end, ordered question ids),
``records.jsonl`` and ``exclusions.jsonl`` without activation indices, and
``activations.pt`` with tensors only. Resume reads metadata alone and accepts
only chunks covering an exact prefix of the ordered question list.
``merge_chunks`` rebuilds one ``CollectionResult`` with global activation
indices, which ``collection_artifacts.save_collection_artifacts`` then
validates and writes as the final run outputs.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from mas_sae.experiments.artifacts import (
    ensure_output_available,
    sha256_json,
    write_json_atomic,
)
from mas_sae.experiments.collection import CollectionResult
from mas_sae.experiments.records import read_jsonl, write_jsonl


PROVENANCE_FILE = "provenance.json"
# Fields of ``provenance.build_collection_progress_provenance`` that must be
# unchanged for a resume; ``created_at_utc`` and ``hash_basis`` are not identity.
IDENTITY_KEYS = (
    "config_sha256",
    "manifest_sha256",
    "dataset_revision",
    "model_revisions",
    "git_commit",
    "git_diff_sha256",
    "untracked_code_sha256",
    "package_versions",
    "gpus",
)
_INDEX_FIELDS = (
    "attempt1_activation_index",
    "attempt2_activation_index",
    "sae_attempt1_index",
    "sae_attempt2_index",
)


def progress_dir(activation_root: str | Path, run_name: str, source_split: str) -> Path:
    """Directory holding one source split's provenance and committed chunks."""
    return Path(activation_root) / run_name / "_progress" / source_split


def manifest_sha256(sampled_questions: list[dict[str, Any]]) -> str:
    """Digest of the ordered question list, including split assignment."""
    return sha256_json([
        {
            "question_id": str(entry["question_id"]),
            "experiment_split": entry.get("experiment_split"),
            "hop_group": entry["hop_group"],
        }
        for entry in sampled_questions
    ])


def ensure_can_start(*, resume: bool, final_output_dir: Path, progress: Path) -> None:
    """Refuse an unsafe fresh start or resume before any model is loaded.

    A fresh run needs neither final outputs nor progress. A resume needs
    progress with provenance and no final outputs, so finalised artifacts
    are never rewritten.
    """
    if not resume:
        ensure_output_available(final_output_dir)
        if progress.exists():
            raise FileExistsError(
                f"Collection progress already exists at {progress}; "
                "pass --resume to continue it."
            )
        return

    if final_output_dir.exists():
        raise FileExistsError(
            f"Final outputs already exist at {final_output_dir}; refusing to "
            "resume over them. Inspect and remove them manually if "
            "finalisation was interrupted."
        )
    if not (progress / PROVENANCE_FILE).is_file():
        raise FileNotFoundError(
            f"--resume requires existing provenance at {progress / PROVENANCE_FILE}."
        )


def _fsync(path: Path) -> None:
    """Flush a file or directory entry to disk."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def initialize(progress: Path, provenance: dict[str, Any]) -> None:
    """Create a fresh progress directory with its durable provenance."""
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.mkdir()
    write_json_atomic(progress / PROVENANCE_FILE, provenance)
    _fsync(progress / PROVENANCE_FILE)
    _fsync(progress)
    _fsync(progress.parent)


def check_identity(progress: Path, expected: dict[str, Any]) -> None:
    """Refuse a resume whose run identity differs from the saved one."""
    path = progress / PROVENANCE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Resume requires existing provenance at {path}.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [key for key in IDENTITY_KEYS if saved.get(key) != expected.get(key)]
    if mismatched:
        raise RuntimeError(
            "Collection resume refused; run identity changed for: "
            f"{', '.join(mismatched)}. Start a new run instead."
        )


def _successful_ids(
    question_ids: list[str],
    records: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> list[str]:
    """Check records and exclusions partition the chunk in order."""
    successful = list(dict.fromkeys(str(row["question_id"]) for row in records))
    excluded = [str(row["question_id"]) for row in exclusions]
    if len(set(excluded)) != len(excluded) or set(successful) & set(excluded):
        raise ValueError("Chunk successful and excluded ids must be disjoint and unique.")
    if successful != [qid for qid in question_ids if qid not in set(excluded)] \
            or not set(excluded) <= set(question_ids):
        raise ValueError("Chunk records and exclusions do not cover its questions in order.")
    return successful


def _stack(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(rows, dim=0) if rows else torch.empty((0, 0))


def write_chunk(
    progress: Path,
    *,
    start: int,
    question_ids: list[str],
    candidate_sites: list[str],
    result: CollectionResult,
) -> Path:
    """Commit one chunk atomically: files, fsync, rename, fsync the parent."""
    records = [
        {key: value for key, value in row.items() if key not in _INDEX_FIELDS}
        for row in result["records"]
    ]
    exclusions = list(result["exclusions"])
    successful = _successful_ids(question_ids, records, exclusions)
    tensors = {
        name: {site: _stack(result[f"{name}_by_site"][site]) for site in candidate_sites}
        for name in ("attempt1", "attempt2")
    }
    for site in candidate_sites:
        if tensors["attempt1"][site].shape[0] != len(successful) \
                or tensors["attempt2"][site].shape[0] != len(records):
            raise ValueError(f"Chunk activation rows do not align with records at {site!r}.")

    final = progress / f"chunk_{start:08d}"
    temporary = progress / f"{final.name}.tmp"
    if final.exists():
        raise FileExistsError(f"Chunk already committed at {final}.")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()

    metadata = {"start": start, "end": start + len(question_ids), "question_ids": list(question_ids)}
    write_json_atomic(temporary / "metadata.json", metadata)
    write_jsonl(temporary / "records.jsonl", records)
    write_jsonl(temporary / "exclusions.jsonl", exclusions)
    torch.save(tensors, temporary / "activations.pt")
    for name in ("metadata.json", "records.jsonl", "exclusions.jsonl", "activations.pt"):
        _fsync(temporary / name)
    _fsync(temporary)
    temporary.replace(final)
    _fsync(progress)
    return final


def _committed_prefix(progress: Path, question_ids: list[str]) -> tuple[list[Path], int]:
    """Committed chunk directories in order, and the first uncovered position.

    Reads only ``metadata.json``. Chunks must tile positions ``0..end`` with
    no gap or overlap and carry exactly the manifest's ids at those positions.
    """
    chunks = []
    for path in progress.glob("chunk_*"):
        if path.is_dir() and not path.name.endswith(".tmp"):
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            chunks.append((metadata["start"], metadata["end"], metadata["question_ids"], path))

    position = 0
    ordered = []
    for start, end, ids, path in sorted(chunks, key=lambda chunk: chunk[0]):
        if start != position:
            raise RuntimeError(
                f"Committed chunks are not an exact prefix: expected start {position}, found {start}."
            )
        if end > len(question_ids) or ids != question_ids[start:end]:
            raise RuntimeError(
                f"Chunk {path.name} does not match the question list at positions {start}:{end}."
            )
        ordered.append(path)
        position = end
    return ordered, position


def resume_position(progress: Path, question_ids: list[str]) -> int:
    """First question position not covered by committed chunks."""
    return _committed_prefix(progress, question_ids)[1]


def merge_chunks(
    progress: Path,
    *,
    question_ids: list[str],
    candidate_sites: list[str],
) -> CollectionResult:
    """Merge a complete run's chunks and assign global activation indices."""
    paths, position = _committed_prefix(progress, question_ids)
    if position != len(question_ids):
        raise RuntimeError(
            f"Cannot finalise: committed chunks cover {position}/{len(question_ids)} questions."
        )

    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    attempt1: dict[str, list[torch.Tensor]] = {site: [] for site in candidate_sites}
    attempt2: dict[str, list[torch.Tensor]] = {site: [] for site in candidate_sites}
    num_attempt1 = 0

    for path in paths:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        chunk_records = read_jsonl(path / "records.jsonl")
        chunk_exclusions = read_jsonl(path / "exclusions.jsonl")
        successful = _successful_ids(metadata["question_ids"], chunk_records, chunk_exclusions)
        tensors = torch.load(path / "activations.pt", map_location="cpu", weights_only=True)

        attempt1_index = {qid: num_attempt1 + offset for offset, qid in enumerate(successful)}
        for row in chunk_records:
            records.append({
                **row,
                "attempt1_activation_index": attempt1_index[str(row["question_id"])],
                "attempt2_activation_index": len(records),
            })
        num_attempt1 += len(successful)
        exclusions.extend(chunk_exclusions)

        for site in candidate_sites:
            a1, a2 = tensors["attempt1"][site], tensors["attempt2"][site]
            if a1.shape[0] != len(successful) or a2.shape[0] != len(chunk_records):
                raise RuntimeError(f"Chunk {path.name} activation rows do not align at {site!r}.")
            if a1.shape[0]:
                attempt1[site].append(a1)
            if a2.shape[0]:
                attempt2[site].append(a2)

    return {
        "records": records,
        "exclusions": exclusions,
        "attempt1_by_site": attempt1,
        "attempt2_by_site": attempt2,
    }
