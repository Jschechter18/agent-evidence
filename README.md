# Capstone Project

Master's Data Science Capstone project.

## Getting Started

### 1. Clone the Repository

```bash
git clone <git@github.com:Jschechter18/fall-2026-group3.git>
cd fall-2026-group3
```

### 2. Install Conda

This project uses Conda to manage the Python environment and dependencies.

If Conda is not already installed, install either **Miniconda** or **Anaconda** before continuing.

### 3. Create the Environment

The project's dependencies are defined in `environment.yml`.

From the root directory of the repository, run:

```bash
conda env create -f environment.yml
```

This will create the `capstone` environment with the appropriate Python version and project dependencies.

### 4. Activate the Environment

```bash
conda activate capstone
```

### 5. Install the project

```bash
python -m pip install -e .
```

Verify that the environment is using Python 3.11:

```bash
python --version
```

You should see:

```text
Python 3.11.x
```

### Updating the Environment

If `environment.yml` changes after you have already created the environment, update your local environment with:

```bash
conda env update -f environment.yml --prune
```

This installs new dependencies and removes dependencies that are no longer specified in the environment file.

## Project Structure

```text
.
├── cookbooks/                    # Jupyter notebooks and tutorials
├── data/                         # Downloaded project datasets
├── demo/                         # Demonstrations and demo figures
├── documents/                    # Supporting project documents and references
├── presentation/                 # Presentation materials
├── reports/                      # Project and progress reports
├── research_paper/               # Research paper source and materials
├── results/                      # Experimental outputs and results
├── scripts/                      # Executable Python entry-point scripts
├── src/
│   ├── mas_sae/                 # Reusable Python package
│   │   ├── agents/              # Solver and critic agent logic
│   │   ├── data/                # Dataset loading and processing
│   │   ├── evaluation/          # Evaluation functions and metrics
│   │   ├── models/              # Language-model loading and interaction
│   │   └── sae/                 # Sparse-autoencoder functionality
│   └── tests/                   # Unit and integration tests
├── environment.yml               # Conda environment and dependencies
├── pyproject.toml                # Python package configuration
├── pytest.ini                    # Pytest configuration
└── README.md                     # Setup and project documentation
```

The project structure may change as development progresses.

## Development

When adding a new dependency, add it to `environment.yml` so that all team members use a consistent environment.

After modifying `environment.yml`, update your environment:

```bash
conda env update -f environment.yml --prune
```

## Jupyter Notebooks

After environment is set up, when using a jupyter notebook, make sure to run the following command to ensure you can select the capstone environment inside the kernel:

```bash
python -m ipykernel install --user \
  --name capstone \
  --display-name "Python 3.11 (capstone)"
```

## Data Collection

After activating the Conda environment and installing the project, run the following command from the repository root:

```bash
python scripts/get_musique_dataset.py
```

## Activation Collection

Activation collection is config-driven. Run from the repository root:

    python scripts/collect_activations.py --config configs/collection/v1/smoke_train.yaml

Collection run definitions are versioned under `configs/collection/`. Once a
config has produced a recorded experiment run, keep that version unchanged and
create a new version directory for protocol changes.

Use the corresponding train or validation config for larger runs.

Activation tensors are saved under:

    data/activations/<run_name>/layer_<N>/

Each source split produces:
- <split>_attempt1.pt
- <split>_attempt2.pt
- <split>.pt

<split>.pt is the SAE-facing tensor with Attempt 1 rows followed by Attempt 2 rows.

Run metadata and reproducibility information are saved under:

    results/collection/<run_name>/<split>/

This directory contains interactions.jsonl, resolved_config.yaml, and summary.json.

## Testing

After activating the Conda environment and installing the project, run:

```bash
pytest
```

## Gemma Model Access

This project uses the Hugging Face model:

`google/gemma-3-4b-it`

Because Gemma is a gated model, each team member must request/accept access on Hugging Face before running the Solver-Critic pipeline.

### 1. Create or log in to Hugging Face

Go to:

https://huggingface.co/

### 2. Request/accept access to Gemma

Open:

https://huggingface.co/google/gemma-3-4b-it

Accept the Gemma license/access requirements shown on the model page.

### 3. Create a Hugging Face access token

Go to:

https://huggingface.co/settings/tokens

Create a token with read access.

Do not commit or share your Hugging Face token.

### 4. Authenticate on the machine running the project

Activate the project environment:

```bash
conda activate capstone
```

Then log in to Hugging Face:

```
hf auth login
```

Paste your Hugging Face token when prompted.

Verify that authentication worked:

```
hf auth whoami
```

### 5. Run the project

Once Gemma access and authentication are complete, the model will download automatically when the project calls load_gemma().

The model is cached under:

checkpoints/huggingface/


## Collection condition policy

Natural, Controlled Correct, and Controlled Incorrect remain supported.
The main scaled experiment activates **Natural only**. The controlled conditions
remain implemented and tested, but inactive in the production configuration.
`collection.active_conditions` accepts a nonempty list of `natural`,
`controlled_correct`, and `controlled_incorrect`. Omitting it preserves V1's
three-condition behavior. Explicit null and empty lists are errors. Disabled
controls generate no targets or episodes.
V2 Natural answers independently before reviewing Solver A1, without gold labels
or a requirement to retain its initial position. Only Solver activations are captured.

`configs/collection/v2/natural_production.yaml` records the condition policy but
intentionally fails validation until model identities, revisions, precision,
placement, layers, and dataset scope are approved. Token budgets are provisional.
Larger-model confirmation is required; final larger-model identities and whether
that Solver is the final mechanistic target remain unresolved.

Explicit `roles.solver`, `roles.critic`, and `roles.validator` replace the legacy
`model` section. Each specifies `id`, `revision`, `loader` (`gemma3` or `qwen3`),
`dtype`, `device` or `device_map`, and `generation`. Identical loading specifications
share weights; different role specifications load independently. There is no
automatic quantization, dtype fallback, or placement change. Solver activation
modules are resolved and checked for Gemma3/Qwen3 before any generation.

New explicit-condition runs add `behavior_schema_version: lexical_v2` and preserve
all historical labels. `critic_textual_relation` records exact/containment/different
or unresolved text relationships. `critic_position_relation` distinguishes same
position, different nonrefusal candidate, refusal/nonanswer, and unresolved output.
These are conservative lexical heuristics, not validated semantic judgments;
unrecognized paraphrases and refusals require review before scientific use.
`solver_behavior` distinguishes no conflict, retained A1, adopted Critic, third
answer, and ambiguity. `direct_critic_adoption` is null outside resolved conflict.
The legacy `solver_accepted_feedback` is not direct adoption of competing feedback.
Production summaries use `solver_behavior_counts` and `critic_position_counts`;
old acceptance counts are isolated under `legacy_acceptance_counts`. Lexical
adoption means matching an answer-like candidate, not proven semantic uptake.
Recognized refusal copying is recorded separately as `solver_copied_nonanswer`.
Natural records preserve `critic_blind_raw_output` alongside the parsed blind
answer and existing final review text so later parsers can revisit both steps.

Generation telemetry records token counts and whether the configured budget was
reached; no definitive truncation/finish reason is inferred. Expected blind/target
failures are written to `exclusions.jsonl`; the requested sample manifest is
preserved separately from successful question IDs and activation row indices.

Full-run checkpoint/resume is deferred. The minimum design is an atomic per-question
checkpoint containing its records, Solver tensors, or exclusion, tied to the run
SHA/config and ordered sample manifest. Resume must validate these identities and
rebuild indices from completed checkpoints before collecting remaining questions.
