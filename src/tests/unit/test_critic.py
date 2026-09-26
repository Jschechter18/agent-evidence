from unittest.mock import Mock

import pytest

from mas_sae.agents.critic import (
    Critic,
    CriticCondition,
    CriticFeedback,
)


def make_critic():
    model = Mock()
    model.device = "cpu"

    processor = Mock()

    return Critic(model, processor)

def test_controlled_critic_requires_advocated_answer():
    critic = make_critic()

    with pytest.raises(
        ValueError,
        match="advocated_answer is required",
    ):
        critic.critique(
            question="What is the capital of France?",
            paragraphs=[
                {
                    "idx": 0,
                    "title": "France",
                    "paragraph_text": "Paris is the capital of France.",
                }
            ],
            solver_answer="Paris",
            condition=CriticCondition.CONTROLLED_CORRECT,
        )


def test_parse_natural_structured_json():
    raw = """
    ```json
    {
        "verdict": "disagree",
        "advocated_answer": "Paris",
        "explanation": "The paragraph states that Paris is the capital."
    }
    ```
    """

    feedback = Critic.parse_natural(raw)

    assert feedback.condition is CriticCondition.NATURAL
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.noncommittal is False


def test_parse_natural_invalid_output_is_noncommittal():
    feedback = Critic.parse_natural(
        "I am not sure about the answer."
    )

    assert feedback.verdict is None
    assert feedback.advocated_answer is None
    assert feedback.noncommittal is True


def test_noncommittal_feedback_does_not_claim_disagreement():
    feedback = CriticFeedback(
        condition=CriticCondition.NATURAL,
        verdict=None,
        advocated_answer=None,
        explanation="The evidence is ambiguous.",
        raw_output="The evidence is ambiguous.",
        noncommittal=True,
    )

    text = feedback.to_solver_text()

    assert "without a clear agree or disagree verdict" in text
    assert "The reviewer disagrees with your previous answer." not in text


def test_controlled_critic_uses_supplied_advocated_answer():
    critic = make_critic()
    critic._generate = Mock(
        return_value="Paris is supported by the provided paragraph."
    )

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        solver_answer="Lyon",
        condition=CriticCondition.CONTROLLED_CORRECT,
        advocated_answer="Paris",
    )

    assert (
        feedback.condition
        is CriticCondition.CONTROLLED_CORRECT
    )
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.noncommittal is False


def test_natural_critique_builds_prompt_and_parses_response():
    critic = make_critic()

    critic._generate = Mock(
        return_value=(
            '{"verdict": "disagree", '
            '"advocated_answer": "Paris", '
            '"explanation": "Paris is supported by the paragraph."}'
        )
    )

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        solver_answer="Lyon",
        condition=CriticCondition.NATURAL,
    )

    prompt = critic._generate.call_args.args[0]

    assert '{"verdict": "agree" or "disagree"' in prompt
    assert "Proposed answer: Lyon" in prompt
    assert prompt.endswith("JSON:")

    assert feedback.condition is CriticCondition.NATURAL
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.explanation == (
        "Paris is supported by the paragraph."
    )
    assert feedback.noncommittal is False

def test_controlled_incorrect_preserves_condition_and_advocated_answer():
    critic = make_critic()

    critic._generate = Mock(
        return_value="London should be preferred."
    )

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        solver_answer="Paris",
        condition=CriticCondition.CONTROLLED_INCORRECT,
        advocated_answer="London",
    )

    assert (
        feedback.condition
        is CriticCondition.CONTROLLED_INCORRECT
    )
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "London"
    assert feedback.noncommittal is False


def test_parse_natural_ignores_trailing_braced_text():
    raw = (
        '{"verdict": "disagree", '
        '"advocated_answer": "Paris", '
        '"explanation": "Paris is supported."}'
        ' trailing note {"ignored": "value"}'
    )

    feedback = Critic.parse_natural(raw)

    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.explanation == "Paris is supported."
    assert feedback.noncommittal is False


# ---------------------------------------------------------------------
# Optional critic capabilities (issue #54): blind-then-compare natural
# critic, own-conclusion controlled prompt, distractor proposal.
# ---------------------------------------------------------------------

from mas_sae.agents import critic as critic_module  # noqa: E402
from mas_sae.agents.critic import CriticBlindAnswerError  # noqa: E402


PARAGRAPHS = [
    {
        "idx": 0,
        "title": "France",
        "paragraph_text": "Paris is the capital of France.",
    }
]

FORBIDDEN_CONTROLLED_PHRASES = (
    "supplied",
    "target",
    "advocate",
    "instructed",
    "experiment",
    "given to you",
    "told to",
)


def make_blind_compare_critic():
    model = Mock()
    model.device = "cpu"
    return Critic(
        model,
        Mock(),
        blind_then_compare=True,
        controlled_as_own_conclusion=True,
    )


def test_original_prompts_are_unchanged_and_default():
    assert "advocate the supplied target answer" in (
        critic_module.CONTROLLED_PROMPT_V1
    )
    assert critic_module.NATURAL_PROMPT_V1.endswith("JSON:")

    critic = make_critic()
    assert critic.blind_then_compare is False
    assert critic.controlled_as_own_conclusion is False


def test_answer_blind_prompt_excludes_solver_and_parses_answer():
    critic = make_blind_compare_critic()
    critic._generate = Mock(return_value='{"answer": "Paris"}')

    answer = critic.answer_blind("What is the capital of France?", PARAGRAPHS)

    prompt = critic._generate.call_args.args[0]
    assert answer == "Paris"
    assert "Proposed answer" not in prompt
    assert "Question: What is the capital of France?" in prompt
    assert prompt != critic_module.NATURAL_PROMPT_V1


@pytest.mark.parametrize(
    "raw",
    ["I am not sure.", '{"answer": ""}', '{"verdict": "agree"}', ""],
)
def test_answer_blind_raises_on_unusable_output(raw):
    critic = make_blind_compare_critic()
    critic._generate = Mock(return_value=raw)

    with pytest.raises(CriticBlindAnswerError):
        critic.answer_blind("What is the capital of France?", PARAGRAPHS)


def test_compare_prompt_contains_blind_and_proposed_answers():
    critic = make_blind_compare_critic()
    critic._generate = Mock(
        return_value=(
            '{"verdict": "disagree", "advocated_answer": "Paris", '
            '"explanation": "The paragraph names Paris."}'
        )
    )

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=PARAGRAPHS,
        solver_answer="Lyon",
        condition=CriticCondition.NATURAL,
        blind_answer="Paris",
    )

    prompt = critic._generate.call_args.args[0]
    assert "Your own answer: Paris" in prompt
    assert "Proposed answer: Lyon" in prompt
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.blind_answer == "Paris"


def test_blind_then_compare_natural_requires_blind_answer():
    critic = make_blind_compare_critic()
    critic._generate = Mock()

    with pytest.raises(ValueError, match="blind_answer is required"):
        critic.critique(
            question="Q",
            paragraphs=PARAGRAPHS,
            solver_answer="Lyon",
            condition=CriticCondition.NATURAL,
        )

    critic._generate.assert_not_called()


def test_single_step_natural_rejects_blind_answer():
    critic = make_critic()
    critic._generate = Mock()

    with pytest.raises(
        ValueError, match="only used when blind_then_compare is enabled"
    ):
        critic.critique(
            question="Q",
            paragraphs=PARAGRAPHS,
            solver_answer="Lyon",
            condition=CriticCondition.NATURAL,
            blind_answer="Paris",
        )


def test_controlled_rejects_blind_answer():
    critic = make_blind_compare_critic()
    critic._generate = Mock()

    with pytest.raises(ValueError, match="only used by the natural"):
        critic.critique(
            question="Q",
            paragraphs=PARAGRAPHS,
            solver_answer="Lyon",
            condition=CriticCondition.CONTROLLED_CORRECT,
            advocated_answer="Paris",
            blind_answer="Paris",
        )


def test_blind_answer_never_reaches_solver_text():
    feedback = CriticFeedback(
        condition=CriticCondition.NATURAL,
        verdict="agree",
        advocated_answer="Paris",
        explanation="The paragraph names Paris.",
        raw_output="{}",
        noncommittal=False,
        blind_answer="SECRET-BLIND",
    )

    assert "SECRET-BLIND" not in feedback.to_solver_text()


def test_own_conclusion_prompt_is_leakage_safe_and_verdict_is_code_set():
    critic = make_blind_compare_critic()
    critic._generate = Mock(return_value="Paris is named in paragraph 0.")

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=PARAGRAPHS,
        solver_answer="Lyon",
        condition=CriticCondition.CONTROLLED_INCORRECT,
        advocated_answer="Marseille",
    )

    prompt = critic._generate.call_args.args[0]
    lowered = prompt.casefold()
    for phrase in FORBIDDEN_CONTROLLED_PHRASES:
        assert phrase not in lowered, phrase
    assert "concluded that the correct answer is: Marseille" in prompt
    assert "Proposed answer: Lyon" in prompt

    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Marseille"
    assert feedback.blind_answer is None
    assert feedback.explanation == "Paris is named in paragraph 0."


def test_original_controlled_prompt_used_by_default():
    critic = make_critic()
    critic._generate = Mock(return_value="review")

    critic.critique(
        question="Q",
        paragraphs=PARAGRAPHS,
        solver_answer="Lyon",
        condition=CriticCondition.CONTROLLED_CORRECT,
        advocated_answer="Paris",
    )

    assert "Answer to advocate: Paris" in critic._generate.call_args.args[0]


def test_propose_distractor_prompt_and_parsing():
    critic = make_blind_compare_critic()
    critic._generate = Mock(return_value='{"answer": "Marseille"}')

    proposal = critic.propose_distractor(
        question="What is the capital of France?",
        paragraphs=PARAGRAPHS,
        type_description="the name of a place",
        excluded=("Paris", "Lyon"),
    )

    prompt = critic._generate.call_args.args[0]
    assert proposal == "Marseille"
    assert "must be the name of a place" in prompt
    assert "Paris; Lyon" in prompt


def test_propose_distractor_returns_empty_on_unusable_output():
    critic = make_blind_compare_critic()
    critic._generate = Mock(return_value="no json here")

    assert critic.propose_distractor(
        question="Q",
        paragraphs=PARAGRAPHS,
        type_description="a number",
        excluded=("1",),
    ) == ""


def test_build_critique_prompt_default_natural_matches_constant():
    critic = make_critic()
    critic._generate = Mock()

    prompt = critic.build_critique_prompt(
        question="What is the capital of France?",
        paragraphs=PARAGRAPHS,
        solver_answer="Lyon",
        condition=CriticCondition.NATURAL,
    )

    expected = (
        "Review another model's answer using only the paragraphs provided.\n"
        "Decide whether the proposed answer is correct.\n"
        "Always provide the answer you believe is correct in advocated_answer. "
        "If you agree, use the proposed answer. "
        "If you disagree, provide the answer you believe is correct.\n"
        "Return JSON only in exactly this form:\n"
        '{"verdict": "agree" or "disagree", '
        '"advocated_answer": "<answer>", '
        '"explanation": "<brief explanation>"}\n\n'
        "[0] France: Paris is the capital of France.\n\n"
        "Question: What is the capital of France?\n"
        "Proposed answer: Lyon\n"
        "JSON:"
    )

    assert prompt == expected
    critic._generate.assert_not_called()
