# Annotation guide: Solver response to Critic feedback

You will label episodes from our Solver–Critic experiment. Please work alone and do not discuss rows with the other annotator until both of you have finished.

## What you see in each row

| Column | Meaning |
|---|---|
| `question` | The question the Solver was asked |
| `context` | The source paragraphs the Solver was given |
| `solver_a1` | The Solver's first answer (A1) |
| `critic_feedback` | The Critic's full feedback on A1 |
| `solver_a2` | The Solver's second answer (A2) |

The rows are shuffled and are not a random sample of the data, so do not expect any category to appear at a particular rate, and do not guess a row's category from its position. Work through the rows in the order given.

## What you fill in

**1. `feedback_type` — what did the Critic give?**

| Value | Use when |
|---|---|
| `answer` | The Critic proposes a concrete answer |
| `refusal` | The Critic gives no answer (for example "the passage does not say") |
| `premise_rejection` | The Critic says the question itself is wrong or based on a false assumption. If it also refuses, still use this value and say so in `notes` |
| `unresolved` | The feedback is missing, self-contradictory or non-committal |

**2. `solver_response` — what did the Solver do in A2?**

| Value | Use when |
|---|---|
| `adopted_critic` | A2 is the Critic's answer, and it differs from A1 |
| `retained_a1` | A2 is the same as A1, and the Critic had proposed something different or given no answer |
| `third_answer` | A2 is a real answer that matches neither A1 nor the Critic |
| `unchanged_same_position` | A1 and the Critic already agreed, and A2 keeps that shared answer |
| `non_answer` | A2 is a refusal or "cannot determine" |
| `produced_answer` | The Critic gave no usable answer, and A2 is a new answer different from A1 |
| `unresolved` | A2 is missing, hedged, or you cannot tell which case applies |

**3. `eligible_primary` — `yes`, `no` or `uncertain`**

- `yes`: A1 and the Critic gave two different real answers, and A2 clearly adopts the Critic's answer, keeps A1, or gives a third answer.
- `no`: any other case that you can identify with confidence, for example the Critic refused, A1 and the Critic already agreed, or A2 is a refusal.
- `uncertain`: you cannot tell whether the two answers really differ, or which answer A2 matches. Typical cases are two names that might be the same thing, an answer that only partly overlaps another, or a response containing several answers. Say what made it unclear in `notes`.

**4. `annotator_confidence`** — `high`, `medium` or `low`.

**5. `notes`** — anything that made the row hard: two names for the same thing, the Critic's answer buried in its explanation, several answers in one response.

## Judgment calls

- Use your own reading, not exact string matching. If "USA" and "United States" mean the same thing here, treat them as the same answer and mention it in `notes`.
- Label what happened, not whether it was right. Do not look up the correct answer, and do not judge whether the Solver should have changed its mind.
- Do not guess the Solver's intentions. Record only what A2 says compared with A1 and the Critic.
- If you are unsure, choose `unresolved` and explain why.

## When both annotators are done

```bash
PYTHONPATH=src python scripts/qc_agreement.py annotator_a.csv annotator_b.csv --output agreement.json
```

This reports, for each field, how many rows both of you completed, the share you agreed on, Cohen's kappa, and the IDs of the rows where you disagreed. Blank rows are not counted as agreement. Resolve disagreements together in a separate file that records who decided and why; do not overwrite either person's original answers.

## For the person running the QC, not for annotators

Give annotators only their CSV and this guide. Keep `private_key.json` and `sampling.json` to yourself: they contain our automatic labels and how the rows were chosen, and seeing them would bias the annotation. Hand out the whole shuffled file, not selected batches.
