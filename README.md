# Capstone Project

Master's Data Science Capstone project.

## Messages for Professor/AI Reviewer:

# Instructor Review 1 Response:

- src contains a module that we are using to create reusable code. We are doing so and using scripts to actually run our pipeline
- we will have shell scripts that run separately in src/shellscripts
- Scripts is used to actually run our code. We package up whatever we can make reusable in the module, and call it in individual scripts files. Think of scripts as something as a frontend for our module. (we obviously can make a real UI later for demo purposes if necessary)
-

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

- <split>\_attempt1.pt
- <split>\_attempt2.pt
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

## Model Access

Models used by the Solver-Critic pipeline (all from Hugging Face):

- **Baseline / current Solver (all checked-in runnable configs):**
  `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`
  (the validated snapshot recovered during the team's provenance review).
- **Cross-model Natural Critic used in protocol validation:**
  `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- **Final production roles (`configs/collection/v2/natural_production.yaml`):**
  intentionally `null` until the larger-model validation is complete. No
  larger model has been selected in this repository yet.

Setup (once per machine):

1. Create or log in to a Hugging Face account: https://huggingface.co/
2. Accept the Gemma license at https://huggingface.co/google/gemma-3-4b-it
   (Gemma is gated; Qwen is not).
3. Create a **read** token at https://huggingface.co/settings/tokens.
   Never commit or share the token.
4. In the `capstone` environment run `hf auth login` and paste the token,
   then confirm with `hf auth whoami`.
5. Weights download on first use and are cached under `checkpoints/huggingface/`.

Rules so runs stay comparable across machines:

- Put the exact model `id` **and** `revision` in the experiment config
  (`roles.solver`, `roles.critic`, `roles.validator`); the legacy `model.id`
  form is still accepted for V1 configs.
- Do not silently substitute a newer `main` revision; a run's
  `resolved_config.yaml` records the resolved revision of every role.
- Explicit `dtype` and `device`/`device_map` are required; there is no
  automatic quantization, dtype fallback, or placement change.

## Collection condition policy

- Natural, Controlled Correct (B) and Controlled Incorrect (C) remain
  supported; `collection.active_conditions` selects a nonempty subset.
  Omitting it keeps V1's three-condition behaviour; `null` or `[]` is an error.
- The current production policy activates **Natural only**. B and C stay
  implemented and tested but inactive; disabled conditions generate no
  targets and no episodes.
- Natural is blind-first: the Critic answers independently, then reviews
  Solver A1. It never sees gold answers or aliases.
- Solver A1 is generated once per question and reused by every enabled condition.
- Only Solver activations are captured; Critic and Validator activations never are.
- Solver, Critic and Validator are independent role specs; identical specs
  share weights, different specs load separately.
- Raw Critic outputs are kept (`critic_blind_raw_output`, `critic_raw_output`)
  so both steps can be re-parsed later.
- Behaviour labels (`behavior_schema_version: lexical_v2`) are conservative
  lexical heuristics, not semantic judgments: `critic_position_relation`,
  `solver_behavior`, `direct_critic_adoption`, `solver_copied_nonanswer`.
  Refusals, malformed output and unresolved cases are never counted as
  rejection, and a lexical difference is not proven disagreement. Legacy
  `solver_accepted_feedback` counts are reported separately under
  `legacy_acceptance_counts`.
- Questions whose blind answer or controlled target cannot be constructed are
  skipped and written to `exclusions.jsonl`; `collected_question_ids.json`
  lists the questions that produced activation rows.
- Generation telemetry records token counts and whether the budget was reached.
- Still unresolved for production: final role models and revisions, layers,
  dataset scope, and token budgets. Full-run checkpoint/resume is deferred.
