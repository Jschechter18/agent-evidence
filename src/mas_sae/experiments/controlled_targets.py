"""Type-checked controlled-incorrect target generation.

Chooses a plausible, type-compatible wrong answer for the
controlled-incorrect critic condition. Candidates are tried in a fixed
order and every candidate must pass the same compatibility check:

1. ``hop_intermediate`` - the answer of a non-final hop in the MuSiQue
   decomposition;
2. ``context_span`` - a year or number extracted from a non-supporting
   paragraph (year and number questions only; paragraph titles are never
   used because their semantic subtype cannot be verified);
3. ``llm_distractor`` - a type-constrained proposal from a caller-supplied
   callback;
4. otherwise ``ControlledTargetError`` is raised. No nonsense fallback.

This module is model-independent: it imports no agent or model code, and
the LLM step is an injected callback.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mas_sae.evaluation.scoring import answer_matches


class AnswerType(StrEnum):
    """Surface type of an answer string."""

    YEAR = "year"
    NUMBER = "number"
    DATE = "date"
    ENTITY = "entity"
    PHRASE = "phrase"


class EntitySubtype(StrEnum):
    """Coarse semantic subtype of an entity answer."""

    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    UNKNOWN = "unknown"


class TargetSource(StrEnum):
    HOP_INTERMEDIATE = "hop_intermediate"
    CONTEXT_SPAN = "context_span"
    LLM_DISTRACTOR = "llm_distractor"


TYPE_CHECK_PASS = "pass"
TYPE_CHECK_UNCHECKED = "unchecked"

MAX_TARGET_WORDS = 8


@dataclass(frozen=True)
class ControlledTarget:
    answer: str
    source: TargetSource
    type_check: str


@dataclass(frozen=True)
class DistractorRequest:
    """Input to the injected LLM distractor callback."""

    question: str
    paragraphs: list[dict[str, Any]]
    type_description: str
    excluded: tuple[str, ...]


DistractorGenerator = Callable[[DistractorRequest], str]


class ControlledTargetError(ValueError):
    """No valid controlled-incorrect target could be constructed."""

    def __init__(self, question_id: str, reasons: Sequence[str]) -> None:
        self.question_id = question_id
        self.reasons = tuple(reasons)
        super().__init__(
            "No valid controlled-incorrect target for question "
            f"{question_id!r}: " + "; ".join(self.reasons)
        )


# ---------------------------------------------------------------------
# Surface type inference
# ---------------------------------------------------------------------

_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
_MONTH_PATTERN = re.compile(
    r"\b(?:" + "|".join(_MONTHS) + r")\b", flags=re.IGNORECASE
)
_YEAR_PATTERN = re.compile(r"^\d{3,4}$")
_NUMBER_PATTERN = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?$")
_YEAR_IN_TEXT = re.compile(r"\b(?:1\d{3}|20\d{2})\b")
_NUMBER_IN_TEXT = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def infer_answer_type(text: str) -> AnswerType:
    """Classify an answer string by its surface form."""
    value = str(text).strip()

    if _YEAR_PATTERN.match(value):
        return AnswerType.YEAR

    if _NUMBER_PATTERN.match(value):
        return AnswerType.NUMBER

    if _MONTH_PATTERN.search(value) and any(c.isdigit() for c in value):
        return AnswerType.DATE

    if value[:1].isalpha() and value[:1].isupper():
        return AnswerType.ENTITY

    return AnswerType.PHRASE


# ---------------------------------------------------------------------
# Entity subtype inference from decomposition relations and the question
# ---------------------------------------------------------------------

# Keyword cues, matched on word boundaries, case-insensitive. Deliberately
# small and editable; extend after team review rather than growing a model.
ORGANIZATION_CUES: tuple[str, ...] = (
    "company", "corporation", "organization", "organisation",
    "institution", "record label", "label", "band", "team", "club",
    "university", "college", "school", "employer", "manufacturer",
    "publisher", "developer", "network", "airline", "agency", "party",
    "league", "member of", "owned by", "operator", "studio", "franchise",
)
LOCATION_CUES: tuple[str, ...] = (
    "where", "city", "town", "village", "country", "state", "province",
    "county", "capital", "located", "location", "place of birth",
    "place of death", "birthplace", "headquarters", "continent", "region",
    "river", "island", "mountain", "territory", "administrative",
    "born in", "died in", "citizenship", "nationality", "origin",
)
PERSON_CUES: tuple[str, ...] = (
    "who", "whose", "whom", "spouse", "husband", "wife", "father",
    "mother", "child", "son", "daughter", "sibling", "brother", "sister",
    "performer", "singer", "author", "writer", "composer", "director",
    "producer", "founder", "founded by", "creator", "inventor",
    "president", "leader", "head of", "actor", "actress", "artist",
    "person", "player", "coach", "screenwriter", "lyricist", "cast member",
)

_SUBTYPE_CUES: tuple[tuple[EntitySubtype, tuple[str, ...]], ...] = (
    (EntitySubtype.ORGANIZATION, ORGANIZATION_CUES),
    (EntitySubtype.LOCATION, LOCATION_CUES),
    (EntitySubtype.PERSON, PERSON_CUES),
)
_WHO_PREFIX = re.compile(r"^\s*(?:who|whose|whom)\b", flags=re.IGNORECASE)


def _cue_position(text: str, cue: str) -> int | None:
    match = re.search(rf"\b{re.escape(cue)}\b", text, flags=re.IGNORECASE)
    return None if match is None else match.start()


def classify_relation(text: str) -> EntitySubtype:
    """Infer an entity subtype from a hop relation or question text.

    A leading ``who``/``whose``/``whom`` wins. Otherwise the cue that
    occurs earliest in the text wins, so the head of a relation
    ("head of state", "country of origin") or of a question ("Which city
    is the birthplace of the author of ...") decides, not a later
    modifier. Ties fall to the table order organization, location, person.
    """
    if _WHO_PREFIX.match(text):
        return EntitySubtype.PERSON

    best: tuple[int, int] | None = None
    best_subtype = EntitySubtype.UNKNOWN

    for order, (subtype, cues) in enumerate(_SUBTYPE_CUES):
        for cue in cues:
            position = _cue_position(text, cue)

            if position is not None and (
                best is None or (position, order) < best
            ):
                best = (position, order)
                best_subtype = subtype

    return best_subtype


def hop_relation(hop: dict[str, Any]) -> str:
    """The relation part of a MuSiQue hop (``"<entity> >> <relation>"``).

    Falls back to the whole hop question when it is not in that form.
    """
    text = str(hop.get("question", ""))
    _, separator, relation = text.rpartition(">>")

    return relation.strip() if separator else text.strip()


def infer_entity_subtype(
    question: str,
    decomposition: Sequence[dict[str, Any]],
) -> EntitySubtype:
    """Subtype of the final answer: final-hop relation, then the question."""
    if decomposition:
        subtype = classify_relation(hop_relation(decomposition[-1]))

        if subtype is not EntitySubtype.UNKNOWN:
            return subtype

    return classify_relation(question)


def type_description(
    answer_type: AnswerType,
    subtype: EntitySubtype,
) -> str:
    """Human-readable constraint for the LLM distractor prompt."""
    if answer_type is AnswerType.YEAR:
        return "a four-digit year"
    if answer_type is AnswerType.NUMBER:
        return "a number"
    if answer_type is AnswerType.DATE:
        return "a calendar date"
    if answer_type is AnswerType.PHRASE:
        return "a short phrase"

    return {
        EntitySubtype.PERSON: "the name of a person",
        EntitySubtype.LOCATION: "the name of a place",
        EntitySubtype.ORGANIZATION: "the name of an organization",
        EntitySubtype.UNKNOWN: "a proper name of the same kind as the answer",
    }[subtype]


# ---------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------


def _reject_reason(
    candidate: str,
    *,
    gold: str,
    aliases: tuple[str, ...],
    attempt1: str,
) -> str | None:
    """Why a candidate string is unusable regardless of its type."""
    value = candidate.strip()

    if not value:
        return "empty"

    if "\n" in value or len(value.split()) > MAX_TARGET_WORDS:
        return "not a short answer"

    if answer_matches(value, gold, aliases):
        return "matches gold or alias"

    if answer_matches(value, attempt1):
        return "matches solver attempt 1"

    return None


def check_compatibility(
    candidate: str,
    *,
    gold_type: AnswerType,
    gold_subtype: EntitySubtype,
    candidate_subtype: EntitySubtype,
    allow_unchecked: bool,
) -> str | None:
    """Return ``"pass"``, ``"unchecked"``, or ``None`` when incompatible.

    Surface types must match. For entity answers the subtypes must match
    when both are known. When compatibility cannot be verified (phrase
    gold, or an unknown subtype on either side) the result is
    ``"unchecked"`` if ``allow_unchecked`` else ``None``.
    """
    candidate_type = infer_answer_type(candidate)

    if gold_type is AnswerType.PHRASE:
        if candidate_type in (AnswerType.YEAR, AnswerType.NUMBER):
            return None
        return TYPE_CHECK_UNCHECKED if allow_unchecked else None

    if candidate_type is not gold_type:
        return None

    if gold_type is not AnswerType.ENTITY:
        return TYPE_CHECK_PASS

    if EntitySubtype.UNKNOWN in (gold_subtype, candidate_subtype):
        return TYPE_CHECK_UNCHECKED if allow_unchecked else None

    return TYPE_CHECK_PASS if gold_subtype is candidate_subtype else None


# ---------------------------------------------------------------------
# Candidate sources
# ---------------------------------------------------------------------


def intermediate_hop_candidates(
    decomposition: Sequence[dict[str, Any]],
) -> list[tuple[str, EntitySubtype]]:
    """Answers of all non-final hops with their relation-derived subtype."""
    return [
        (str(hop.get("answer", "")).strip(), classify_relation(hop_relation(hop)))
        for hop in decomposition[:-1]
    ]


def context_span_candidates(
    paragraphs: Iterable[dict[str, Any]],
    gold_type: AnswerType,
) -> list[str]:
    """Years or numbers from non-supporting paragraphs, in text order."""
    if gold_type is AnswerType.YEAR:
        pattern = _YEAR_IN_TEXT
    elif gold_type is AnswerType.NUMBER:
        pattern = _NUMBER_IN_TEXT
    else:
        return []

    seen: dict[str, None] = {}

    for paragraph in paragraphs:
        if paragraph.get("is_supporting", False):
            continue

        for match in pattern.findall(str(paragraph.get("paragraph_text", ""))):
            seen.setdefault(match, None)

    return list(seen)


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------


def choose_controlled_incorrect_target(
    *,
    question_id: str,
    question: str,
    paragraphs: list[dict[str, Any]],
    gold: str,
    aliases: Iterable[str],
    attempt1: str,
    decomposition: Sequence[dict[str, Any]] = (),
    distractor_generator: DistractorGenerator | None = None,
) -> ControlledTarget:
    """Choose a plausible, type-compatible wrong answer, or raise.

    Every candidate must differ from gold, its aliases, and Solver
    Attempt 1, and pass ``check_compatibility``. ``"unchecked"`` is only
    possible for the LLM step, when the gold type is a phrase or its
    entity subtype is unknown.
    """
    aliases = tuple(aliases)
    gold_type = infer_answer_type(gold)
    gold_subtype = (
        infer_entity_subtype(question, decomposition)
        if gold_type is AnswerType.ENTITY
        else EntitySubtype.UNKNOWN
    )
    reasons: list[str] = []

    def evaluate(
        candidate: str,
        source: TargetSource,
        candidate_subtype: EntitySubtype,
        allow_unchecked: bool,
    ) -> ControlledTarget | None:
        reason = _reject_reason(
            candidate, gold=gold, aliases=aliases, attempt1=attempt1
        )

        if reason is None:
            type_check = check_compatibility(
                candidate,
                gold_type=gold_type,
                gold_subtype=gold_subtype,
                candidate_subtype=candidate_subtype,
                allow_unchecked=allow_unchecked,
            )

            if type_check is not None:
                return ControlledTarget(
                    answer=candidate.strip(),
                    source=source,
                    type_check=type_check,
                )

            reason = "type incompatible"

        reasons.append(f"{source.value} {candidate.strip()!r}: {reason}")
        return None

    # 1. Intermediate hop answers.
    for candidate, subtype in intermediate_hop_candidates(decomposition):
        target = evaluate(
            candidate, TargetSource.HOP_INTERMEDIATE, subtype, False
        )
        if target is not None:
            return target

    # 2. Years / numbers from non-supporting paragraphs.
    for candidate in context_span_candidates(paragraphs, gold_type):
        target = evaluate(
            candidate, TargetSource.CONTEXT_SPAN, EntitySubtype.UNKNOWN, False
        )
        if target is not None:
            return target

    # 3. Type-constrained LLM distractor. The requested subtype is imposed
    #    through the prompt, so the candidate carries the gold subtype.
    if distractor_generator is not None:
        request = DistractorRequest(
            question=question,
            paragraphs=paragraphs,
            type_description=type_description(gold_type, gold_subtype),
            excluded=tuple(dict.fromkeys((gold, *aliases, attempt1))),
        )
        proposal = str(distractor_generator(request) or "")
        target = evaluate(
            proposal, TargetSource.LLM_DISTRACTOR, gold_subtype, True
        )
        if target is not None:
            return target
    else:
        reasons.append("llm_distractor: no generator supplied")

    raise ControlledTargetError(question_id, reasons)
