from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from mas_sae.agents.base import Agent
from mas_sae.agents.solver import Solver
from mas_sae.evaluation.scoring import answer_matches


class CriticCondition(StrEnum):
    NATURAL = "natural"
    CONTROLLED_CORRECT = "controlled_correct"
    CONTROLLED_INCORRECT = "controlled_incorrect"


class CriticBlindAnswerError(RuntimeError):
    """The critic failed to produce a usable blind answer.

    Raised instead of falling back to a compare step without a blind
    answer, which would silently change the natural-critic treatment.
    Retry and whole-question exclusion are handled by later QC work.
    """


@dataclass(frozen=True)
class CriticFeedback:
    condition: CriticCondition
    verdict: str | None
    advocated_answer: str | None
    explanation: str
    raw_output: str
    noncommittal: bool
    # Natural condition with ``blind_then_compare`` only: the answer the
    # critic gave before seeing the Solver's attempt. Never included in
    # ``to_solver_text``.
    blind_answer: str | None = None

    def to_solver_text(self) -> str:
        if self.verdict == "agree":
            verdict_text = "The reviewer agrees with your previous answer."
        elif self.verdict == "disagree":
            verdict_text = "The reviewer disagrees with your previous answer."
        else:
            verdict_text = (
                "The reviewer provided feedback without a clear "
                "agree or disagree verdict."
            )

        parts = [verdict_text]

        if self.advocated_answer:
            parts.append(
                f"The reviewer advocates this answer: {self.advocated_answer}"
            )

        if self.explanation:
            parts.append(
                f"Reviewer explanation: {self.explanation}"
            )

        return " ".join(parts)


NATURAL_PROMPT_V1 = (
    "Review another model's answer using only the paragraphs provided.\n"
    "Decide whether the proposed answer is correct.\n"
    "Always provide the answer you believe is correct in advocated_answer. "
    "If you agree, use the proposed answer. "
    "If you disagree, provide the answer you believe is correct.\n"
    "Return JSON only in exactly this form:\n"
    '{{"verdict": "agree" or "disagree", '
    '"advocated_answer": "<answer>", '
    '"explanation": "<brief explanation>"}}\n\n'
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "Proposed answer: {solver_answer}\n"
    "JSON:"
)


CONTROLLED_PROMPT_V1 = (
    "Review another model's answer using only the paragraphs provided.\n"
    "For this review, advocate the supplied target answer.\n"
    "Explain briefly why that answer should be preferred.\n\n"
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "Proposed answer: {solver_answer}\n"
    "Answer to advocate: {advocated_answer}\n"
    "Review:"
)


# ---------------------------------------------------------------------
# Optional critic capabilities. The prompts above are frozen; the ones
# below are selected per Critic instance through ``blind_then_compare``
# and ``controlled_as_own_conclusion``.
# ---------------------------------------------------------------------

# Step 1 of the blind-then-compare natural critic: an independent answer
# formed without seeing the Solver's attempt. Deliberately worded and
# formatted differently from the Solver's SOLVE_PROMPT_V1 so greedy
# decoding does not simply replay Attempt 1.
NATURAL_BLIND_PROMPT = (
    "Using only the paragraphs provided, answer the question "
    "independently.\n"
    "Return JSON only, in exactly this form:\n"
    '{{"answer": "<short answer>"}}\n\n'
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "JSON:"
)


# Step 2 of the blind-then-compare natural critic: compare the blind
# answer with the Solver's attempt and return the same JSON schema as
# NATURAL_PROMPT_V1.
NATURAL_COMPARE_PROMPT = (
    "Review another model's answer using only the paragraphs provided.\n"
    "You already answered this question yourself; your own answer is "
    "shown below.\n"
    "Decide whether the proposed answer is correct.\n"
    "Always provide the answer you believe is correct in advocated_answer. "
    "If you agree, use the proposed answer. "
    "If you disagree, provide the answer you believe is correct.\n"
    "Return JSON only in exactly this form:\n"
    '{{"verdict": "agree" or "disagree", '
    '"advocated_answer": "<answer>", '
    '"explanation": "<brief explanation>"}}\n\n'
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "Your own answer: {blind_answer}\n"
    "Proposed answer: {solver_answer}\n"
    "JSON:"
)


# Controlled feedback (both controlled conditions) with
# ``controlled_as_own_conclusion``. The target is presented as the
# reviewer's own conclusion; no experimental vocabulary ("supplied",
# "target", "advocate", "instructed") that could be echoed to the Solver.
CONTROLLED_OWN_CONCLUSION_PROMPT = (
    "Review another model's answer to the question using only the "
    "paragraphs provided.\n"
    "You have read the paragraphs and concluded that the correct answer "
    "is: {advocated_answer}\n"
    "Write a brief review for the other model. Explain, citing only the "
    "paragraphs, why {advocated_answer} answers the question, and address "
    "the proposed answer if it differs.\n"
    "Write in your own voice as the reviewer.\n\n"
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "Proposed answer: {solver_answer}\n"
    "Review:"
)


# Type-constrained wrong-answer proposal used as the last resort of the
# type-checked controlled-incorrect target generator. Its output is only
# ever used as a target string; it is never shown to the Solver.
DISTRACTOR_PROMPT = (
    "Using only the paragraphs provided, propose one plausible but "
    "incorrect answer to the question.\n"
    "The proposed answer must be {type_description}.\n"
    "It must not be any of the following: {excluded}\n"
    "Return JSON only, in exactly this form:\n"
    '{{"answer": "<short answer>"}}\n\n'
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "JSON:"
)


_JSON_DECODER = json.JSONDecoder()


def _extract_first_json_object(
    raw_output: str,
) -> dict | None:
    for index, char in enumerate(raw_output):
        if char != "{":
            continue

        try:
            parsed, _ = _JSON_DECODER.raw_decode(
                raw_output[index:]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    return None


def _parse_answer_json(raw_output: str) -> str | None:
    """Return the non-empty ``answer`` of the first JSON object, if any."""
    parsed = _extract_first_json_object(raw_output)

    if parsed is None:
        return None

    answer = parsed.get("answer")

    if answer is None:
        return None

    answer = str(answer).strip()

    return answer or None


class Critic(Agent):
    def __init__(
        self,
        model,
        processor,
        max_new_tokens: int = 128,
        *,
        blind_then_compare: bool = False,
        controlled_as_own_conclusion: bool = False,
    ) -> None:
        """Critic agent with two optional prompt capabilities.

        ``blind_then_compare``: the natural condition first answers the
        question without seeing the Solver (``answer_blind``), then
        compares that blind answer with the Solver's attempt. Otherwise
        the natural condition uses ``NATURAL_PROMPT_V1`` in one step.

        ``controlled_as_own_conclusion``: controlled feedback presents
        the advocated answer as the reviewer's own conclusion
        (``CONTROLLED_OWN_CONCLUSION_PROMPT``). Otherwise
        ``CONTROLLED_PROMPT_V1`` is used.
        """
        super().__init__(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )
        self.blind_then_compare = blind_then_compare
        self.controlled_as_own_conclusion = controlled_as_own_conclusion

    def answer_blind(
        self,
        question: str,
        paragraphs: list[dict],
    ) -> str:
        """Answer the question without seeing the Solver (blind step).

        Raises ``CriticBlindAnswerError`` when the output carries no
        usable answer, so a run never continues on a compare step that
        lacks the blind answer.
        """
        prompt = NATURAL_BLIND_PROMPT.format(
            paragraphs=Solver.format_paragraphs(paragraphs),
            question=question,
        )

        raw_output = self._generate(prompt)
        answer = _parse_answer_json(raw_output)

        if answer is None:
            raise CriticBlindAnswerError(
                "Critic blind answer was missing or unparseable: "
                f"{raw_output!r}"
            )

        return answer

    def propose_distractor(
        self,
        *,
        question: str,
        paragraphs: list[dict],
        type_description: str,
        excluded: Sequence[str],
    ) -> str:
        """Propose one type-constrained wrong answer (target last resort).

        Returns an empty string when the output carries no answer; the
        caller validates the proposal like any other candidate.
        """
        prompt = DISTRACTOR_PROMPT.format(
            paragraphs=Solver.format_paragraphs(paragraphs),
            question=question,
            type_description=type_description,
            excluded="; ".join(excluded),
        )

        raw_output = self._generate(prompt)

        return _parse_answer_json(raw_output) or ""

    @staticmethod
    def parse_natural(raw_output: str) -> CriticFeedback:
        parsed = _extract_first_json_object(raw_output)

        if parsed is None:
            return CriticFeedback(
                condition=CriticCondition.NATURAL,
                verdict=None,
                advocated_answer=None,
                explanation="",
                raw_output=raw_output,
                noncommittal=True,
            )

        verdict = parsed.get("verdict")

        if isinstance(verdict, str):
            verdict = verdict.strip().casefold()

        if verdict not in {"agree", "disagree"}:
            verdict = None

        advocated_answer = parsed.get("advocated_answer")

        if advocated_answer is not None:
            advocated_answer = str(advocated_answer).strip()

            if not advocated_answer:
                advocated_answer = None

        explanation = str(
            parsed.get("explanation", "")
        ).strip()

        noncommittal = (
            verdict is None
            or advocated_answer is None
        )

        return CriticFeedback(
            condition=CriticCondition.NATURAL,
            verdict=verdict,
            advocated_answer=advocated_answer,
            explanation=explanation,
            raw_output=raw_output,
            noncommittal=noncommittal,
        )

    def build_critique_prompt(
        self,
        *,
        question: str,
        paragraphs: list[dict],
        solver_answer: str,
        condition: CriticCondition,
        advocated_answer: str | None = None,
        blind_answer: str | None = None,
    ) -> str:
        """Build the critique prompt for one condition.

        Validates the argument combination and selects the prompt; no
        model call is made. ``blind_answer`` is required for the natural
        condition when ``blind_then_compare`` is enabled (the output of
        ``answer_blind``) and must be omitted otherwise.
        ``advocated_answer`` is required for the controlled conditions.
        """
        formatted_paragraphs = Solver.format_paragraphs(
            paragraphs
        )

        if condition is CriticCondition.NATURAL:
            if self.blind_then_compare:
                if not blind_answer:
                    raise ValueError(
                        "blind_answer is required for the natural "
                        "condition when blind_then_compare is enabled; "
                        "call answer_blind first."
                    )

                return NATURAL_COMPARE_PROMPT.format(
                    paragraphs=formatted_paragraphs,
                    question=question,
                    blind_answer=blind_answer,
                    solver_answer=solver_answer,
                )

            if blind_answer is not None:
                raise ValueError(
                    "blind_answer is only used when blind_then_compare "
                    "is enabled."
                )

            return NATURAL_PROMPT_V1.format(
                paragraphs=formatted_paragraphs,
                question=question,
                solver_answer=solver_answer,
            )

        if blind_answer is not None:
            raise ValueError(
                "blind_answer is only used by the natural condition."
            )

        if advocated_answer is None:
            raise ValueError(
                "advocated_answer is required for controlled conditions"
            )

        controlled_prompt = (
            CONTROLLED_OWN_CONCLUSION_PROMPT
            if self.controlled_as_own_conclusion
            else CONTROLLED_PROMPT_V1
        )

        return controlled_prompt.format(
            paragraphs=formatted_paragraphs,
            question=question,
            solver_answer=solver_answer,
            advocated_answer=advocated_answer,
        )

    def critique(
        self,
        *,
        question: str,
        paragraphs: list[dict],
        solver_answer: str,
        condition: CriticCondition,
        advocated_answer: str | None = None,
        blind_answer: str | None = None,
    ) -> CriticFeedback:
        """Generate feedback for one critic condition.

        ``blind_answer`` is required for the natural condition when
        ``blind_then_compare`` is enabled (the output of ``answer_blind``)
        and must be omitted otherwise. It is recorded on the natural
        feedback and never reaches ``to_solver_text``.
        """
        prompt = self.build_critique_prompt(
            question=question,
            paragraphs=paragraphs,
            solver_answer=solver_answer,
            condition=condition,
            advocated_answer=advocated_answer,
            blind_answer=blind_answer,
        )

        raw_output = self._generate(prompt)

        if condition is CriticCondition.NATURAL:
            return replace(
                self.parse_natural(raw_output),
                blind_answer=blind_answer,
            )

        verdict = (
            "agree"
            if answer_matches(
                solver_answer,
                advocated_answer,
            )
            else "disagree"
        )

        return CriticFeedback(
            condition=condition,
            verdict=verdict,
            advocated_answer=advocated_answer,
            explanation=raw_output,
            raw_output=raw_output,
            noncommittal=False,
        )
