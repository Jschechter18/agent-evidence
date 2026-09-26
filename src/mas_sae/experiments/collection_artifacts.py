"""Validate and persist one collection run's tensors, records and summary.

Reproducibility metadata is built by ``provenance.build_resolved_config``;
this module only checks integrity and writes what it is given.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.collection import stack_site_activations
from mas_sae.experiments.records import write_jsonl
from mas_sae.experiments.conditions import OMITTED, resolve_conditions


def behavioral_summary(records: list[dict[str, Any]], *, production: bool) -> dict[str, Any]:
    """Keep legacy acceptance counts separate from lexical production outcomes."""
    legacy = {
        "accepted": sum(row["solver_accepted_feedback"] is True for row in records),
        "rejected": sum(row["solver_accepted_feedback"] is False for row in records),
        "unlabeled_noncommittal": sum(row["solver_accepted_feedback"] is None for row in records),
    }
    if not production:
        return legacy
    return {
        "behavior_schema_versions": sorted({row["behavior_schema_version"] for row in records}),
        "behavior_matching_basis": "normalized_text_not_semantic",
        "solver_behavior_counts": _count_by_key(records, "solver_behavior"),
        "critic_position_counts": _count_by_key(records, "critic_position_relation"),
        "legacy_acceptance_counts": legacy,
    }


def site_slug(module_name: str) -> str:
    """Convert a model layer module name to its storage directory name."""
    layer = module_name.rsplit(".", 1)[-1]

    try:
        return f"layer_{int(layer):02d}"
    except ValueError as error:
        raise ValueError(
            f"Could not determine layer number from {module_name!r}."
        ) from error


def _save_and_verify(
    store: ActivationStore,
    split: str,
    tensor: torch.Tensor,
) -> None:
    """Save one activation tensor and verify its local round trip."""
    store.save_activations(split, tensor)
    loaded = store.load_activations(split)

    if not torch.equal(tensor.detach().cpu(), loaded):
        raise RuntimeError(
            f"Activation round trip failed for split {split!r}."
        )


def _add_sae_indices(
    records: list[dict[str, Any]],
    num_attempt1: int,
) -> list[dict[str, Any]]:
    """Add row indices for the combined SAE activation tensor."""
    enriched = []

    for record in records:
        row = dict(record)
        row["sae_attempt1_index"] = int(row["attempt1_activation_index"])
        row["sae_attempt2_index"] = (
            num_attempt1 + int(row["attempt2_activation_index"])
        )
        enriched.append(row)

    return enriched


def _count_by_key(
    entries: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    """Count manifest entries by one key, in sorted key order."""
    return dict(sorted(Counter(str(entry[key]) for entry in entries).items()))


def _validate_alignment(
    *,
    candidate_sites: list[str],
    attempt1: dict[str, torch.Tensor],
    attempt2: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    successful_ids: list[str],
    resolved_config: dict[str, Any],
) -> tuple[int, int]:
    """Check records, tensors and active conditions agree; return row counts."""
    expected_sites = set(candidate_sites)

    if set(attempt1) != expected_sites:
        raise ValueError(
            "Attempt 1 activation sites do not match candidate sites."
        )
    if set(attempt2) != expected_sites:
        raise ValueError(
            "Attempt 2 activation sites do not match candidate sites."
        )

    first_site = candidate_sites[0]
    n1, n2 = attempt1[first_site].shape[0], attempt2[first_site].shape[0]
    if len(successful_ids) != n1 or len(records) != n2:
        raise ValueError("Record/question counts do not align with activation rows")
    id_to_index = {qid: i for i, qid in enumerate(successful_ids)}
    for i, row in enumerate(records):
        if (
            row["attempt1_activation_index"] != id_to_index[row["question_id"]]
            or row["attempt2_activation_index"] != i
        ):
            raise ValueError("Record activation indices do not align")
    active = resolved_config.get("collection", {}).get("active_conditions", OMITTED)
    if active is not OMITTED:
        expected = {c.value for c in resolve_conditions(active)}
        for qid in successful_ids:
            conditions = [row["critic_condition"] for row in records if row["question_id"] == qid]
            if len(conditions) != len(expected) or set(conditions) != expected:
                raise ValueError("Episode conditions do not match active_conditions")
        episode_ids = [row.get("episode_id") for row in records]
        if None in episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Episode IDs must be present and unique")
    for site in candidate_sites:
        a1, a2 = attempt1[site], attempt2[site]
        if (
            a1.ndim != 2
            or a2.ndim != 2
            or a1.shape[0] != n1
            or a2.shape[0] != n2
            or a1.shape[1] != a2.shape[1]
        ):
            raise ValueError("Activation shapes do not align across sites")
    return n1, n2


def save_collection_artifacts(
    *,
    activation_root: str | Path,
    result_root: str | Path,
    run_name: str,
    source_split: str,
    candidate_sites: list[str],
    attempt1_by_site: dict[str, list[torch.Tensor]],
    attempt2_by_site: dict[str, list[torch.Tensor]],
    records: list[dict[str, Any]],
    resolved_config: dict[str, Any],
    sampled_questions: list[dict[str, Any]] | None = None,
    exclusions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save activation tensors, metadata, resolved config, and summary.

    When ``sampled_questions`` is given (the ordered manifest built by
    ``describe_sampled_questions``), it is written to
    ``sampled_questions.json`` so the exact question set can be reproduced,
    and the summary gains question-level hop-group and experiment-split
    counts. When ``exclusions`` is given, ``exclusions.jsonl`` and
    ``collected_question_ids.json`` are written as well, and a run whose
    every question was excluded writes metadata but no tensors. Callers
    that pass neither get the original outputs unchanged.

    Every integrity check runs before anything is written.
    """
    if not candidate_sites:
        raise ValueError("candidate_sites must not be empty.")
    production = "active_conditions" in resolved_config.get("collection", {})
    successful_ids = list(dict.fromkeys(row["question_id"] for row in records))
    excluded_ids = [row["question_id"] for row in (exclusions or [])]
    if len(set(excluded_ids)) != len(excluded_ids) or set(successful_ids) & set(excluded_ids):
        raise ValueError("Successful and excluded question IDs must be disjoint and unique")
    collected_manifest = sampled_questions
    if sampled_questions is not None and exclusions is not None:
        requested_ids = [row["question_id"] for row in sampled_questions]
        if len(set(requested_ids)) != len(requested_ids) or set(requested_ids) != set(successful_ids) | set(excluded_ids):
            raise ValueError("Requested manifest must match successful plus excluded IDs")
        collected_manifest = [row for row in sampled_questions if row["question_id"] in successful_ids]

    if not records and exclusions:
        if any(attempt1_by_site.values()) or any(attempt2_by_site.values()):
            raise ValueError("Failed-only collection must not contain activation rows")
        attempt1 = attempt2 = {}
        num_attempt1 = num_attempt2 = 0
    else:
        attempt1 = stack_site_activations(attempt1_by_site)
        attempt2 = stack_site_activations(attempt2_by_site)
        num_attempt1, num_attempt2 = _validate_alignment(
            candidate_sites=candidate_sites,
            attempt1=attempt1,
            attempt2=attempt2,
            records=records,
            successful_ids=successful_ids,
            resolved_config=resolved_config,
        )

    if sampled_questions is not None and len(collected_manifest) != num_attempt1:
        raise ValueError(
            "sampled_questions length does not match the number of "
            f"collected questions ({len(collected_manifest)} vs "
            f"{num_attempt1})."
        )

    behavior_counts = behavioral_summary(records, production=production)
    site_shapes: dict[str, dict[str, list[int]]] = {}

    for site in (candidate_sites if records else []):
        attempt1_tensor = attempt1[site]
        attempt2_tensor = attempt2[site]
        sae_tensor = torch.cat([attempt1_tensor, attempt2_tensor], dim=0)
        slug = site_slug(site)
        store = ActivationStore(Path(activation_root) / run_name / slug)

        _save_and_verify(
            store, f"{source_split}_attempt1", attempt1_tensor
        )
        _save_and_verify(
            store, f"{source_split}_attempt2", attempt2_tensor
        )
        _save_and_verify(store, source_split, sae_tensor)

        site_shapes[slug] = {
            "attempt1": list(attempt1_tensor.shape),
            "attempt2": list(attempt2_tensor.shape),
            "sae": list(sae_tensor.shape),
        }

    enriched_records = _add_sae_indices(records, num_attempt1)

    output_dir = Path(result_root) / run_name / source_split
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "interactions.jsonl", enriched_records)
    if exclusions is not None:
        write_jsonl(output_dir / "exclusions.jsonl", exclusions)
        (output_dir / "collected_question_ids.json").write_text(json.dumps(successful_ids, indent=2))

    with (output_dir / "resolved_config.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(resolved_config, file, sort_keys=False)

    summary = {
        "run_name": run_name,
        "source_split": source_split,
        "num_questions": num_attempt1,
        "num_episodes": num_attempt2,
        "num_excluded": len(exclusions or []),
        "sae_rows_per_layer": num_attempt1 + num_attempt2,
        **behavior_counts,
        "site_shapes": site_shapes,
    }

    if sampled_questions is not None:
        with (output_dir / "sampled_questions.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(sampled_questions, file, indent=2)
            file.write("\n")

        summary["hop_group_counts"] = _count_by_key(
            collected_manifest, "hop_group"
        )

        if all("experiment_split" in entry for entry in sampled_questions):
            summary["experiment_split_counts"] = _count_by_key(
                collected_manifest, "experiment_split"
            )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    return summary
