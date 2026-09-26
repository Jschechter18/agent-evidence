from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from mas_sae.data.activation_store import ActivationStore
from mas_sae.data.musique import MUSIQUE_DATASET_ID
from mas_sae.experiments import artifacts
from mas_sae.experiments.collection import stack_site_activations
from mas_sae.experiments.records import write_jsonl


def _critic_prompt_names(
    critic: Any,
    type_checked_target: bool,
) -> dict[str, str]:
    """Names of the critic prompts a run actually used, by capability."""
    names = {
        "critic_natural": (
            "NATURAL_BLIND_PROMPT+NATURAL_COMPARE_PROMPT"
            if getattr(critic, "blind_then_compare", False)
            else "NATURAL_PROMPT_V1"
        ),
        "critic_controlled": (
            "CONTROLLED_OWN_CONCLUSION_PROMPT"
            if getattr(critic, "controlled_as_own_conclusion", False)
            else "CONTROLLED_PROMPT_V1"
        ),
    }

    if type_checked_target:
        names["critic_distractor"] = "DISTRACTOR_PROMPT"

    return names


def build_resolved_config(
    config: dict[str, Any],
    model: Any,
    solver: Any,
    critic: Any,
    validator: Any,
    *,
    protocol_version: str | None = None,
    type_checked_target: bool = False,
) -> dict[str, Any]:
    """Add minimal reproducibility provenance to the run config.

    ``protocol_version`` is the caller's opaque run label, recorded as
    given when supplied. The prompt names and ``capabilities`` block are
    derived from the critic's capability flags and ``type_checked_target``,
    so a run records the behaviour that actually produced its feedback.
    """
    capabilities = {
        "blind_then_compare": bool(
            getattr(critic, "blind_then_compare", False)
        ),
        "controlled_as_own_conclusion": bool(
            getattr(critic, "controlled_as_own_conclusion", False)
        ),
        "type_checked_target": bool(type_checked_target),
    }
    label = (
        {} if protocol_version is None
        else {"protocol_version": protocol_version}
    )

    return {
        **config,
        "provenance": {
            "created_at_utc": artifacts.utc_now().isoformat(),
            "git_commit": artifacts.get_git_commit(),
            "model_revision": (
                getattr(model.config, "_commit_hash", None) or "unknown"
            ),
            "dataset_id": MUSIQUE_DATASET_ID,
            "package_versions": {
                "torch": artifacts.get_package_version("torch"),
                "transformers": artifacts.get_package_version("transformers"),
                "datasets": artifacts.get_package_version("datasets"),
            },
            **label,
            "capabilities": capabilities,
            "prompt_versions": {
                "solver_solve": "SOLVE_PROMPT_V1",
                "solver_revise": "REVISE_PROMPT_V1",
                **_critic_prompt_names(critic, type_checked_target),
                "validator": "VALIDATE_PROMPT",
            },
            "generation": {
                "do_sample": False,
                "solver_max_new_tokens": solver.max_new_tokens,
                "critic_max_new_tokens": critic.max_new_tokens,
                "validator_max_new_tokens": validator.max_new_tokens,
            },
        },
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
) -> dict[str, Any]:
    """Save activation tensors, metadata, resolved config, and summary.

    When ``sampled_questions`` is given (the ordered manifest built by
    ``describe_sampled_questions``), it is written to
    ``sampled_questions.json`` so the exact question set can be reproduced,
    and the summary gains question-level hop-group and experiment-split
    counts. Callers that pass nothing get the original outputs unchanged.
    """
    attempt1 = stack_site_activations(attempt1_by_site)
    attempt2 = stack_site_activations(attempt2_by_site)
    expected_sites = set(candidate_sites)

    if set(attempt1) != expected_sites:
        raise ValueError(
            "Attempt 1 activation sites do not match candidate sites."
        )
    if set(attempt2) != expected_sites:
        raise ValueError(
            "Attempt 2 activation sites do not match candidate sites."
        )

    num_attempt1: int | None = None
    num_attempt2: int | None = None
    site_shapes: dict[str, dict[str, list[int]]] = {}

    for site in candidate_sites:
        attempt1_tensor = attempt1[site]
        attempt2_tensor = attempt2[site]

        if attempt1_tensor.shape[1] != attempt2_tensor.shape[1]:
            raise ValueError(f"Hidden dimensions do not match for {site!r}.")

        if num_attempt1 is None:
            num_attempt1 = attempt1_tensor.shape[0]
            num_attempt2 = attempt2_tensor.shape[0]
        elif (
            attempt1_tensor.shape[0] != num_attempt1
            or attempt2_tensor.shape[0] != num_attempt2
        ):
            raise ValueError(
                "Activation row counts differ across candidate sites."
            )

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

    if num_attempt1 is None or num_attempt2 is None:
        raise ValueError("No activation sites were saved.")

    enriched_records = _add_sae_indices(records, num_attempt1)
    accepted = sum(
        row["solver_accepted_feedback"] is True for row in records
    )
    rejected = sum(
        row["solver_accepted_feedback"] is False for row in records
    )
    noncommittal = sum(
        row["solver_accepted_feedback"] is None for row in records
    )

    output_dir = Path(result_root) / run_name / source_split
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "interactions.jsonl", enriched_records)

    with (output_dir / "resolved_config.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(resolved_config, file, sort_keys=False)

    summary = {
        "run_name": run_name,
        "source_split": source_split,
        "num_questions": num_attempt1,
        "num_episodes": num_attempt2,
        "sae_rows_per_layer": num_attempt1 + num_attempt2,
        "accepted": accepted,
        "rejected": rejected,
        "unlabeled_noncommittal": noncommittal,
        "site_shapes": site_shapes,
    }

    if sampled_questions is not None:
        if len(sampled_questions) != num_attempt1:
            raise ValueError(
                "sampled_questions length does not match the number of "
                f"collected questions ({len(sampled_questions)} vs "
                f"{num_attempt1})."
            )

        with (output_dir / "sampled_questions.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(sampled_questions, file, indent=2)
            file.write("\n")

        summary["hop_group_counts"] = _count_by_key(
            sampled_questions, "hop_group"
        )

        if all("experiment_split" in entry for entry in sampled_questions):
            summary["experiment_split_counts"] = _count_by_key(
                sampled_questions, "experiment_split"
            )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    return summary
