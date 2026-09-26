import ast
from pathlib import Path

import pytest

from mas_sae.experiments import controlled_targets as ct
from mas_sae.experiments.controlled_targets import (
    AnswerType,
    ControlledTargetError,
    DistractorRequest,
    EntitySubtype,
    TargetSource,
    choose_controlled_incorrect_target,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

PERSON_DECOMPOSITION = [
    {"question": "Green >> performer", "answer": "Steve Hillage"},
    {"question": "#1 >> spouse", "answer": "Miquette Giraudy"},
]

YEAR_DECOMPOSITION = [
    {"question": "Green >> performer", "answer": "Steve Hillage"},
    {"question": "In which year was #1 born?", "answer": "1951"},
]


def paragraphs(*texts, supporting=()):
    return [
        {
            "idx": index,
            "title": f"Title {index}",
            "paragraph_text": text,
            "is_supporting": index in supporting,
        }
        for index, text in enumerate(texts)
    ]


def choose(**overrides):
    kwargs = {
        "question_id": "2hop__1_2",
        "question": "Who is the spouse of the performer of Green?",
        "paragraphs": paragraphs(
            "Steve Hillage released Green in 1978.",
            "Paris is the capital of France, population 2,100,000.",
            supporting=(0,),
        ),
        "gold": "Miquette Giraudy",
        "aliases": (),
        "attempt1": "Miquette Giraudy",
        "decomposition": PERSON_DECOMPOSITION,
        "distractor_generator": None,
    }
    kwargs.update(overrides)
    return choose_controlled_incorrect_target(**kwargs)


# ---------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1951", AnswerType.YEAR),
        ("842", AnswerType.YEAR),
        ("2,100,000", AnswerType.NUMBER),
        ("3.5", AnswerType.NUMBER),
        ("March 4, 1990", AnswerType.DATE),
        ("4 March 1990", AnswerType.DATE),
        ("Barack Obama", AnswerType.ENTITY),
        ("France", AnswerType.ENTITY),
        ("bass guitar", AnswerType.PHRASE),
        ("the Netherlands", AnswerType.PHRASE),
        ("", AnswerType.PHRASE),
    ],
)
def test_infer_answer_type(text, expected):
    assert ct.infer_answer_type(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("spouse", EntitySubtype.PERSON),
        ("performer", EntitySubtype.PERSON),
        ("Who founded the company?", EntitySubtype.PERSON),
        ("country", EntitySubtype.LOCATION),
        ("place of birth", EntitySubtype.LOCATION),
        ("In which city was #1 born?", EntitySubtype.LOCATION),
        ("record label", EntitySubtype.ORGANIZATION),
        ("member of sports team", EntitySubtype.ORGANIZATION),
        ("What company did #1 found?", EntitySubtype.ORGANIZATION),
        # earliest cue wins: the head word, not a later modifier
        ("head of state", EntitySubtype.PERSON),
        ("country of origin", EntitySubtype.LOCATION),
        (
            "Which city is the birthplace of the author of Green?",
            EntitySubtype.LOCATION,
        ),
        ("Where was the founder of the label born?", EntitySubtype.LOCATION),
        ("genre", EntitySubtype.UNKNOWN),
        ("", EntitySubtype.UNKNOWN),
    ],
)
def test_classify_relation(text, expected):
    assert ct.classify_relation(text) is expected


def test_hop_relation_extracts_text_after_separator():
    assert ct.hop_relation({"question": "Green >> performer"}) == "performer"
    assert ct.hop_relation({"question": "Who sang Green?"}) == "Who sang Green?"


def test_infer_entity_subtype_prefers_final_hop_then_question():
    assert ct.infer_entity_subtype("Q?", PERSON_DECOMPOSITION) is (
        EntitySubtype.PERSON
    )
    assert ct.infer_entity_subtype(
        "Where was the performer of Green born?",
        [{"question": "Green >> performer"}, {"question": "#1 >> genre"}],
    ) is EntitySubtype.LOCATION
    assert ct.infer_entity_subtype("What genre is Green?", []) is (
        EntitySubtype.UNKNOWN
    )


# ---------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------


def compat(candidate, gold_type, gold_subtype=EntitySubtype.UNKNOWN,
           candidate_subtype=EntitySubtype.UNKNOWN, allow_unchecked=False):
    return ct.check_compatibility(
        candidate,
        gold_type=gold_type,
        gold_subtype=gold_subtype,
        candidate_subtype=candidate_subtype,
        allow_unchecked=allow_unchecked,
    )


def test_check_compatibility_surface_types():
    assert compat("1960", AnswerType.YEAR) == "pass"
    assert compat("Steve Hillage", AnswerType.YEAR) is None
    assert compat("42", AnswerType.NUMBER) == "pass"
    assert compat("1 June 2001", AnswerType.DATE) == "pass"


def test_check_compatibility_entity_subtypes():
    person, location = EntitySubtype.PERSON, EntitySubtype.LOCATION
    assert compat("Steve Hillage", AnswerType.ENTITY, person, person) == "pass"
    assert compat("France", AnswerType.ENTITY, person, location) is None
    # unknown on either side: only "unchecked" when allowed
    assert compat("France", AnswerType.ENTITY, person) is None
    assert compat("France", AnswerType.ENTITY, person, allow_unchecked=True) == (
        "unchecked"
    )


def test_check_compatibility_phrase_gold():
    assert compat("lead guitar", AnswerType.PHRASE) is None
    assert compat("lead guitar", AnswerType.PHRASE, allow_unchecked=True) == (
        "unchecked"
    )
    assert compat("1990", AnswerType.PHRASE, allow_unchecked=True) is None


# ---------------------------------------------------------------------
# Candidate sources
# ---------------------------------------------------------------------


def test_context_span_candidates_years_from_non_supporting_only():
    docs = paragraphs(
        "Born in 1951, he recorded Green in 1978.",
        "The city was founded in 1200 and rebuilt in 1951.",
        supporting=(0,),
    )

    assert ct.context_span_candidates(docs, AnswerType.YEAR) == ["1200", "1951"]
    assert ct.context_span_candidates(docs, AnswerType.ENTITY) == []


def test_intermediate_hop_candidates_exclude_final_hop():
    assert ct.intermediate_hop_candidates(PERSON_DECOMPOSITION) == [
        ("Steve Hillage", EntitySubtype.PERSON)
    ]
    assert ct.intermediate_hop_candidates([]) == []


# ---------------------------------------------------------------------
# Selection order and rejection rules
# ---------------------------------------------------------------------


def test_hop_intermediate_person_for_person_question_passes():
    target = choose()

    assert target.answer == "Steve Hillage"
    assert target.source is TargetSource.HOP_INTERMEDIATE
    assert target.type_check == "pass"


def test_hop_intermediate_rejected_for_year_question_then_context_span():
    target = choose(
        question="In which year was the performer of Green born?",
        gold="1951",
        attempt1="1951",
        decomposition=YEAR_DECOMPOSITION,
        paragraphs=paragraphs(
            "Born in 1951, he recorded Green in 1978.",
            "The city was founded in 1200.",
            supporting=(0,),
        ),
    )

    assert target.answer == "1200"
    assert target.source is TargetSource.CONTEXT_SPAN
    assert target.type_check == "pass"


def test_context_span_skips_gold_and_attempt1_years():
    target = choose(
        question="In which year was the performer of Green born?",
        gold="1951",
        attempt1="1200",
        decomposition=YEAR_DECOMPOSITION,
        paragraphs=paragraphs(
            "Born in 1951.",
            "Founded in 1200, expanded in 1951, closed in 1999.",
            supporting=(0,),
        ),
    )

    assert target.answer == "1999"


def test_hop_intermediate_rejected_when_subtype_mismatches():
    decomposition = [
        {"question": "Green >> country of origin", "answer": "France"},
        {"question": "#1 >> head of state", "answer": "Emmanuel Macron"},
    ]

    with pytest.raises(ControlledTargetError, match="type incompatible"):
        choose(
            question="Who is the head of state of the country Green is from?",
            gold="Emmanuel Macron",
            attempt1="Emmanuel Macron",
            decomposition=decomposition,
        )


@pytest.mark.parametrize(
    ("gold", "aliases", "attempt1", "reason"),
    [
        ("Steve Hillage", (), "Other", "matches gold or alias"),
        ("Other", ("steve hillage",), "Other", "matches gold or alias"),
        ("Other", (), "Steve Hillage", "matches solver attempt 1"),
    ],
)
def test_hop_intermediate_rejected_when_equal_to_gold_alias_or_attempt1(
    gold, aliases, attempt1, reason
):
    with pytest.raises(ControlledTargetError, match=reason):
        choose(gold=gold, aliases=aliases, attempt1=attempt1)


def test_paragraph_titles_are_never_used_for_entity_gold():
    with pytest.raises(ControlledTargetError) as info:
        choose(decomposition=[], attempt1="Someone Else")

    assert "Title" not in str(info.value)
    assert "no generator supplied" in str(info.value)


def test_llm_distractor_is_constrained_validated_and_marked_pass():
    requests = []

    def generator(request: DistractorRequest) -> str:
        requests.append(request)
        return "Daevid Allen"

    target = choose(decomposition=[], distractor_generator=generator)

    assert target.answer == "Daevid Allen"
    assert target.source is TargetSource.LLM_DISTRACTOR
    assert target.type_check == "pass"
    assert requests[0].type_description == "the name of a person"
    assert requests[0].excluded == ("Miquette Giraudy",)


def test_llm_distractor_unchecked_only_when_unverifiable():
    unknown = choose(
        question="What genre is Green?",
        gold="Progressive Rock",
        attempt1="Progressive Rock",
        decomposition=[],
        distractor_generator=lambda request: "Jazz Fusion",
    )
    assert unknown.type_check == "unchecked"

    phrase = choose(
        question="What instrument did the performer of Green play?",
        gold="lead guitar",
        attempt1="lead guitar",
        decomposition=[],
        distractor_generator=lambda request: "bass guitar",
    )
    assert phrase.type_check == "unchecked"


@pytest.mark.parametrize(
    "proposal",
    [
        "Miquette Giraudy",           # gold
        "miquette giraudy",           # gold, different case
        "",                           # empty
        "1978",                       # wrong surface type
        "She was married to a musician from Gong for many years.",
    ],
)
def test_llm_distractor_rejected_when_invalid(proposal):
    with pytest.raises(ControlledTargetError, match="llm_distractor"):
        choose(decomposition=[], distractor_generator=lambda request: proposal)


def test_error_lists_every_rejection_reason():
    with pytest.raises(ControlledTargetError) as info:
        choose(
            attempt1="Steve Hillage",
            distractor_generator=lambda request: "Miquette Giraudy",
        )

    message = str(info.value)
    assert info.value.question_id == "2hop__1_2"
    assert "hop_intermediate 'Steve Hillage': matches solver attempt 1" in message
    assert "llm_distractor 'Miquette Giraudy': matches gold or alias" in message


def test_selection_is_deterministic():
    assert choose() == choose()


def test_module_is_model_independent():
    source = Path(ct.__file__).read_text(encoding="utf-8")
    imported = {
        getattr(node, "module", None) or alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(
        name.startswith(("torch", "mas_sae.agents", "mas_sae.models"))
        for name in imported
    )
