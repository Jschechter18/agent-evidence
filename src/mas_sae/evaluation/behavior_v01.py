"""Behavior v0.1.1: candidate labels for how the Solver responds to Critic feedback.

Each episode has a first answer (A1), the Critic's feedback, and a second
answer (A2). This module gives every episode two separate labels:

    feedback_type    what the Critic gave
    solver_response  what the Solver did in A2

The rules compare normalized text. They do not understand meaning, so the
labels are candidates until people have checked them. They are stored next to
the original lexical_v2 labels and never replace them.
"""
from collections import Counter
import csv
import re

from mas_sae.evaluation import behavior

RULE_VERSION = "behavior_v0.1.1_candidate"

SOLVER_RESPONSES = (
    "adopted_critic",           # A2 is the Critic's answer
    "retained_a1",              # A2 is the same as A1
    "third_answer",             # A2 is different from both
    "unchanged_same_position",  # A1, the Critic and A2 all agree
    "non_answer",               # A2 is a refusal such as "cannot determine"
    "produced_answer",          # the Critic gave no usable answer; A2 is new
    "unresolved",               # A2 is missing, hedged or only a partial match
)

# The fields of an episode that decide its label.
LABEL_INPUT_FIELDS = ("solver_attempt_1", "critic_advocated_answer", "solver_attempt_2",
                      "critic_noncommittal")

# Columns of labels.csv that identify the episode; the label columns follow them.
LABEL_KEY_FIELDS = ["question_id", "source_split", "canonical_split", "lexical_v2_label"]

# How label columns are stored in labels.csv, so reading restores what was written.
BOOLEAN_COLUMNS = ("eligible_primary", "eligible_strict", "critic_unresolved_flag",
                   "critic_premise_rejection_flag", "critic_refusal_flag")
INTEGER_OR_BLANK_COLUMNS = ("primary_target", "strict_target")
TEXT_OR_BLANK_COLUMNS = ("subtype", "human_validated_behavior_v1")

# Ways of refusing to answer that the lexical_v2 rules did not catch.
EXTRA_NONANSWER = re.compile(
    r"\b(?:"
    r"does not (?:specify|mention|provide|state|indicate)"
    r"|there is no mention|not mentioned|not specified"
    r"|cannot be (?:determined|answered)"
    r"|insufficient information|not enough information"
    r"|(?:the )?passage does not|(?:the )?text does not"
    r"|provided information does not"
    r")\b",
    re.I,
)

# The Critic says the question itself is wrong. The single word "premise" is a
# broad match and is one of the things the human check looks at.
PREMISE = re.compile(
    r"\b(?:"
    r"false premise"
    r"|based on a (?:misunderstanding|false|flawed|incorrect)"
    r"|premise|not applicable|no valid connection"
    r"|(?:is|are) not (?:related|connected)"
    r"|does not exist|there (?:is|was) no such"
    r")\b",
    re.I,
)


def answer_kind(text):
    """Classify one piece of text as 'answer', 'nonanswer', 'uncertain' or 'missing'."""
    kind = behavior.answer_kind(text)
    if kind in ("answer", "uncertain") and EXTRA_NONANSWER.search(text.replace("’", "'")):
        return "nonanswer"
    return kind


def _feedback_type(unresolved, premise_rejection, refusal):
    """What the Critic gave. The first matching case wins."""
    if unresolved:
        return "unresolved"
    if premise_rejection:
        return "premise_rejection"
    if refusal:
        return "refusal"
    return "answer"


def _conflict_status(feedback_type, a1_kind, a1_vs_critic):
    """Whether A1 and the Critic gave two different answers."""
    if feedback_type != "answer" or a1_kind != "answer":
        return "unresolved"
    if a1_vs_critic == "exact":
        return "same_position"
    if a1_vs_critic == "different":
        return "lexical_conflict_candidate"
    return a1_vs_critic  # "containment", e.g. "Paris" vs "Paris, France"


def _solver_response(feedback_type, a1_kind, a2_kind, a1_vs_critic, a2_vs_a1, a2_vs_critic):
    """What the Solver did in A2. Returns (response, subtype)."""
    if a2_kind == "nonanswer":
        return "non_answer", None
    if a2_kind != "answer":
        return "unresolved", None

    critic_answered = feedback_type == "answer"
    a1_answered = a1_kind == "answer"
    kept_a1 = a1_answered and a2_vs_a1 == "exact"
    took_critic = critic_answered and a2_vs_critic == "exact"

    if kept_a1 and critic_answered and a1_vs_critic == "exact":
        return "unchanged_same_position", "shared_answer_no_conflict"
    if kept_a1:
        return "retained_a1", None
    if took_critic and a1_kind == "nonanswer":
        return "adopted_critic", "a1_nonanswer_to_answer"
    if took_critic and not a1_answered:
        return "adopted_critic", "a1_unresolved_to_answer"
    if took_critic:
        return "adopted_critic", None
    if critic_answered and a1_answered and a2_vs_a1 == "different" and a2_vs_critic == "different":
        return "third_answer", None
    if not critic_answered:
        return "produced_answer", None
    return "unresolved", None


def _exclusion_reasons(feedback_type, a1_kind, a2_kind, conflict_status, solver_response):
    """Every reason this episode cannot be used for the probe target."""
    reasons = []
    if feedback_type != "answer":
        reasons.append("critic_" + feedback_type)
    if a1_kind != "answer":
        reasons.append("a1_" + a1_kind)
    if feedback_type == "answer" and a1_kind == "answer":
        if conflict_status == "same_position":
            reasons.append("no_conflict")
        elif conflict_status != "lexical_conflict_candidate":
            reasons.append("a1_critic_" + conflict_status)
    if a2_kind != "answer":
        reasons.append("a2_" + a2_kind)
    if solver_response == "unresolved":
        reasons.append("a2_relationship_unresolved")
    return reasons


def classify_candidate(row):
    """Label one episode. Returns a new dict and does not change the row."""
    a1 = row.get("solver_attempt_1")
    a2 = row.get("solver_attempt_2")
    critic = row.get("critic_advocated_answer")

    # What kind of text is each piece, and how do the three compare?
    a1_kind = answer_kind(a1)
    a2_kind = answer_kind(a2)
    critic_kind = answer_kind(critic)
    a1_vs_critic = behavior.textual_relation(a1, critic)
    a2_vs_a1 = behavior.textual_relation(a2, a1)
    a2_vs_critic = behavior.textual_relation(a2, critic)

    # Three independent checks on the Critic. All three are kept in the output.
    critic_unresolved = (row.get("critic_noncommittal") is not False
                         or critic_kind in ("missing", "uncertain"))
    critic_premise = isinstance(critic, str) and bool(PREMISE.search(critic))
    critic_refusal = critic_kind == "nonanswer"

    feedback_type = _feedback_type(critic_unresolved, critic_premise, critic_refusal)
    conflict_status = _conflict_status(feedback_type, a1_kind, a1_vs_critic)
    solver_response, subtype = _solver_response(
        feedback_type, a1_kind, a2_kind, a1_vs_critic, a2_vs_a1, a2_vs_critic)
    exclusion_reasons = _exclusion_reasons(
        feedback_type, a1_kind, a2_kind, conflict_status, solver_response)

    # Primary target: adopted (1) versus kept or third answer (0).
    # Strict target:  adopted (1) versus kept (0) only.
    eligible_primary = (not exclusion_reasons
                        and solver_response in ("adopted_critic", "retained_a1", "third_answer"))
    eligible_strict = eligible_primary and solver_response != "third_answer"
    adopted = int(solver_response == "adopted_critic")

    return {
        "rule_version": RULE_VERSION,
        "feedback_type": feedback_type,
        "solver_response": solver_response,
        "conflict_status": conflict_status,
        "subtype": subtype,
        "eligible_primary": eligible_primary,
        "eligible_strict": eligible_strict,
        "primary_target": adopted if eligible_primary else None,
        "strict_target": adopted if eligible_strict else None,
        "exclusion_reasons": exclusion_reasons,
        "a1_kind": a1_kind,
        "a2_kind": a2_kind,
        "critic_kind": critic_kind,
        "a1_vs_critic": a1_vs_critic,
        "a2_vs_a1": a2_vs_a1,
        "a2_vs_critic": a2_vs_critic,
        "critic_unresolved_flag": critic_unresolved,
        "critic_premise_rejection_flag": critic_premise,
        "critic_refusal_flag": critic_refusal,
        "human_validated_behavior_v1": None,
    }


def write_labels(path, rows):
    """Write one CSV row per question. Each row needs ``label`` and ``canonical_split``.

    A list is joined with ';' and None is left blank.
    """
    label_fields = list(rows[0]["label"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LABEL_KEY_FIELDS + label_fields)
        for row in rows:
            label = dict(row["label"])
            label["exclusion_reasons"] = ";".join(label["exclusion_reasons"])
            values = ["" if label[field] is None else label[field] for field in label_fields]
            writer.writerow([row["question_id"], row["source_split"],
                             row["canonical_split"], row["solver_behavior"]] + values)


def read_labels(path):
    """Read labels.csv back as {question_id: label}, with the types ``write_labels`` was given."""
    with open(path, newline="") as f:
        labels = {record["question_id"]: record for record in csv.DictReader(f)}
    for qid, label in labels.items():
        for field in BOOLEAN_COLUMNS:
            if label[field] not in ("True", "False"):
                raise ValueError(f"{qid}: {field} must be True or False, got {label[field]!r}")
            label[field] = label[field] == "True"
        for field in INTEGER_OR_BLANK_COLUMNS:
            label[field] = int(label[field]) if label[field] else None
        for field in TEXT_OR_BLANK_COLUMNS:
            label[field] = label[field] or None
        reasons = label["exclusion_reasons"]
        label["exclusion_reasons"] = reasons.split(";") if reasons else []
    return labels


def label_counts(rows):
    """Label counts for a group of labelled rows. Target counts use eligible rows only."""
    feedback_by_response = Counter()
    lexical_v2_to_candidate = Counter()
    exclusion_reasons = Counter()
    target_by_split = Counter()
    target_by_correctness = Counter()

    for row in rows:
        label = row["label"]
        response = label["solver_response"]
        pair = f"{label['feedback_type']} / {response}"

        feedback_by_response[pair] += 1
        lexical_v2_to_candidate[f"{row['solver_behavior']} -> {pair}"] += 1
        exclusion_reasons.update(label["exclusion_reasons"])

        if label["eligible_primary"]:
            target_by_split[f"{row['canonical_split']} / {response}"] += 1
            correctness = (f"a1_correct={row['solver_attempt_1_correct']} / "
                           f"critic_correct={row['critic_feedback_correct']}")
            target_by_correctness[f"{correctness} / {response}"] += 1

    return {
        "rows": len(rows),
        "feedback_by_response": dict(sorted(feedback_by_response.items())),
        "lexical_v2_to_candidate": dict(sorted(lexical_v2_to_candidate.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "primary_target_by_split": dict(sorted(target_by_split.items())),
        "primary_target_by_correctness": dict(sorted(target_by_correctness.items())),
    }
