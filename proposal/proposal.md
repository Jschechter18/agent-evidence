# Capstone Proposal

## SAE-MAS: Causal Analysis of Feedback Uptake in Solver-Critic Multi-Agent LLM Systems

### Proposed by: Josh Schechter, Israel Omoniyi , Raye Oji

#### Advisor: Dr. Amir Jafari

#### The George Washington University, Washington DC

#### Data Science Program

---

## 1 Objective:

Large language model systems increasingly use multiple agents that critique, revise, or verify one another's outputs. A common architecture consists of a **Solver** that produces an initial answer and a **Critic** that evaluates the answer and provides feedback. The Solver can then revise its answer based on the Critic's response.

Although these systems are designed around the assumption that feedback can improve reasoning, considerably less is understood about **what changes internally within the Solver when it receives feedback and decides whether to accept or reject it**.

Sparse Autoencoders (SAEs) provide a potential mechanism for studying this process. SAEs decompose dense LLM activations into high-dimensional sparse feature representations. These sparse representations can be analyzed to identify features associated with particular model behaviors and, importantly, can be manipulated and decoded back into the LLM activation space to test whether those features are causally involved in the behavior.

The primary research question of this project is:

> **What internal representations change when an LLM Solver receives critic feedback, which sparse features predict whether the Solver accepts or rejects that feedback, and are those features causally involved in feedback uptake?**

The project will study this question using a controlled Solver-Critic system. The same initial Solver response will, where possible, be exposed to controlled correct and incorrect critic feedback. Solver activations will be collected before and after feedback, represented using Sparse Autoencoders, and analyzed using interpretable linear probe models.

Candidate SAE features identified by the predictive analysis will then be directly manipulated during subsequent Solver forward passes. If manipulating a feature systematically changes whether the Solver accepts or rejects critic feedback, this provides evidence that the feature is not merely correlated with feedback uptake but is causally involved in the mechanism.

A secondary objective is to compare a **general pretrained SAE** against an **interaction-specific SAE trained on activations collected during Solver-Critic interactions**. This tests whether an SAE trained specifically on the activation distribution produced during feedback processing better isolates features associated with feedback uptake. 

![SAE-MAS Experimental Pipeline](revised_capstone_diagram.drawio.svg)

### Key Objectives:

1. **Build a controlled Solver-Critic interaction pipeline and activation dataset.**
   - Run benchmark questions through a Solver-Critic-Solver interaction.
   - Cache selected Solver activations before and after critic feedback.
   - Generate controlled correct and incorrect critic feedback using known benchmark ground truth.
   - Record behavioral outcomes including feedback acceptance, answer changes, and final correctness.
   - Record the primary binary feedback-acceptance target `solver_accepted_feedback ∈ {0,1}` and record Critic-feedback correctness separately. Retain the four combinations of Critic correctness × Solver acceptance as a 2×2 behavioral outcome matrix for descriptive and causal analysis.

2. **Represent Solver activations using Sparse Autoencoders.**
   <!-- - Load a compatible pretrained SAE for the selected LLM and activation site. -->
   - Train an interaction-specific SAE on cached Solver activations from the Solver-Critic dataset.
   - Evaluate SAE reconstruction quality and sparsity before using the representation for downstream analysis.

3. **Identify sparse features predictive of feedback uptake.**
   - Encode Solver activations into SAE latent feature vectors.
   - Train regularized linear probe models using the latent feature vectors to classify which classes (described in 1) the Solver will yield (basically only cares about accepting or rejecting).
   - Use probe coefficients and held-out validation to select a small set of candidate features associated with feedback uptake (how the solver incorporates the critics feedback).
   - Compare predictive structure discovered by the pretrained and interaction-trained SAEs.

4. **Test whether selected sparse features are causally involved in feedback uptake.**
   - Suppress or amplify selected SAE features during the Solver's processing of critic feedback.
   - Decode the modified SAE representation back into the LLM activation space.
   - Continue the Solver forward pass using the modified activation.
   - Measure whether the intervention changes the Solver's feedback-acceptance behavior or acceptance rate.

5. **Characterize whether feedback-related internal mechanisms support appropriate feedback uptake.**
   - Compare interventions under correct and incorrect critic feedback.
   - Determine whether selected features correspond to useful correction uptake, general willingness to revise, resistance to feedback, susceptibility to incorrect feedback, or other feedback-related behavior.
   - Package the interaction pipeline, activation collection, SAE training/loading, probe analysis, and causal intervention experiments into a reproducible repository.

---

## 2 Dataset:

The project requires two related forms of data:

1. an existing benchmark containing questions with objectively known answers, and
2. a generated Solver-Critic interaction dataset containing behavioral outcomes and Solver activations.

The interaction dataset will be generated by the project rather than obtained from an existing multi-agent benchmark.

### TIER 1 -- BASE REASONING DATASET:

**MuSiQue**

MuSiQue is a multi-hop question-answering benchmark that requires models to combine information across multiple reasoning steps.

Dataset:

https://huggingface.co/datasets/dgslibisey/MuSiQue

MuSiQue provides the underlying questions and known answers required to objectively evaluate the Solver's initial and revised responses.

Additional objectively scored reasoning datasets may be considered if pilot experiments show that MuSiQue does not produce sufficient variation in Solver correctness or feedback acceptance. The final benchmark set should be frozen after the initial pilot.

### TIER 2 -- GENERATED SOLVER-CRITIC INTERACTION DATASET:

Each benchmark question will first be passed through the Solver to generate an initial answer. The same initial Solver response will then be reused across multiple Critic conditions where possible. This creates matched interactions in which the question, initial answer, and initial Solver reasoning remain fixed while the feedback provided to the Solver changes.

The dataset will contain one natural Solver-Critic condition and two controlled feedback conditions.

#### Condition A -- Natural Critic Feedback

The Critic independently evaluates the Solver's initial response and provides whatever feedback it determines is appropriate. The Critic is not instructed to agree or disagree with the Solver and is not given a predetermined answer to advocate.

This condition represents the normal Solver-Critic interaction and will serve as the project's primary observational dataset.

After the interaction, the known benchmark ground truth will be used to determine whether the Critic's feedback was correct/helpful or incorrect/harmful. The Solver's second response will then be evaluated to determine whether it accepted or rejected the Critic's feedback.

The Natural Critic will return a structured response containing at minimum a `verdict` (`agree` / `disagree`), an `advocated_answer`, and an explanation. This structured output allows Critic correctness to be scored against MuSiQue gold answers and aliases. Non-committal feedback without a clear advocated answer will be flagged separately.

This produces naturally occurring examples of:

1. Correct/helpful feedback + Solver accepts
2. Correct/helpful feedback + Solver rejects
3. Incorrect/harmful feedback + Solver accepts
4. Incorrect/harmful feedback + Solver rejects

These interactions will be used to investigate which internal Solver representations are associated with feedback acceptance and rejection under normal multi-agent behavior.

#### Condition B -- Controlled Correct Feedback

The same initial Solver response will be provided to a Critic instructed to advocate for the known ground-truth answer.

This creates a controlled condition in which the feedback presented to the Solver is known to support the correct answer.

This condition measures how the Solver responds when exposed to useful feedback and provides a controlled comparison against the natural Critic condition.

#### Condition C -- Controlled Incorrect Feedback

The same initial Solver response will be provided to a Critic instructed to advocate for a deliberately incorrect answer.

This creates a controlled condition in which the feedback presented to the Solver is known to support an incorrect answer.

This condition measures the Solver's susceptibility to misleading feedback and provides a matched comparison with the controlled correct-feedback condition.

### Paired Experimental Design

Where technically feasible, all three Critic conditions will branch from the same initial Solver response:

`Solver Attempt 1 → Natural Critic → Solver Attempt 2A`

`Solver Attempt 1 → Controlled Correct Critic → Solver Attempt 2B`

`Solver Attempt 1 → Controlled Incorrect Critic → Solver Attempt 2C`

This design holds the original question and Solver Attempt 1 constant while changing the feedback presented to the Solver.

The natural condition provides the primary dataset for studying feedback behavior as it occurs organically within the Solver-Critic system. The controlled conditions provide experimental comparisons that allow the project to determine how feedback correctness affects the Solver's behavior and internal representations.

### Interaction Record

Each feedback event will contain at minimum:

- `question_id`
- `question`
- `ground_truth`
- `solver_attempt_1`
- `solver_attempt_1_correct`
- `critic_condition`
- `critic_feedback`
- `critic_feedback_correct`
- `solver_attempt_2`
- `solver_attempt_2_correct`
- `solver_changed_answer`
- `solver_accepted_feedback`
- `activation_attempt_1`
- `activation_attempt_2`

The primary behavioral target will be:

`solver_accepted_feedback ∈ {0,1}`

Critic correctness will be recorded separately rather than being combined with Solver acceptance into the primary target. This allows the project to distinguish **whether the Solver accepts feedback** from **whether accepting that feedback was appropriate**.

_Important_: The four combinations (2 x 2 table) of feedback correctness and Solver acceptance will still be retained for descriptive analysis:

1. Correct feedback + Solver accepts
2. Correct feedback + Solver rejects
3. Incorrect feedback + Solver accepts
4. Incorrect feedback + Solver rejects

### TIER 3 -- ACTIVATION DATASET:

Selected internal Solver activations will be cached during:

1. **Solver Attempt 1:** before critic feedback
2. **Solver Attempt 2:** while the Solver processes critic feedback and produces its revised answer

The exact transformer layer and activation site will be selected during the initial technical feasibility stage.

The project will initially target **one activation site at one layer**. A second layer may be added only if compute and schedule permit.

These cached activations serve two purposes:

- training the interaction-specific SAE;
- generating SAE latent representations for predictive analysis.

Because both pre-feedback and post-feedback activations are collected, the project can additionally investigate changes in sparse representation such as:

`Δz = z_after_feedback - z_before_feedback`

This provides a direct representation of how the Solver's internal state changes after exposure to critic feedback.

### DATA SPLITS:

Questions must be divided into separate experimental splits before feature selection:

- **Discovery / training split:** SAE training and probe-based feature discovery
- **Validation split:** confirmation that selected features predict feedback uptake on unseen interactions
- **Intervention / test split:** causal intervention experiments

Questions from the intervention/test split must not be used to select candidate SAE features.

This separation prevents the causal experiment from being evaluated on the same interactions used to discover the candidate features.

---

## 3 Rationale:

Multi-agent LLM systems commonly rely on interaction patterns such as critique, debate, verification, and revision. These architectures assume that exposing one model to another agent's feedback can improve the final response.

However, observing that a Solver changes its answer after criticism does not explain **how the feedback is internally processed**.

The Solver may:

- recognize that the critic has identified an error;
- detect disagreement and reconsider its previous reasoning;
- defer to another agent regardless of whether that agent is correct;
- resist external feedback;
- or respond through a distributed mechanism that cannot be represented by a small number of interpretable features.

Behavioral evaluation alone cannot distinguish these possibilities.

Sparse Autoencoders provide a useful framework for investigating this question because they transform dense transformer activations into sparse latent feature representations. Instead of attempting to interpret thousands of dense activation dimensions directly, the project can analyze which sparse features change during feedback processing and which features are predictive of subsequent Solver behavior.

However, **predictive association is not sufficient to establish mechanism**.

A sparse feature may reliably predict feedback acceptance without causing it. The feature could be downstream of the actual mechanism or correlated with another internal process.

The central methodological contribution of this project is therefore the transition from:

> **feature detection:** "This sparse feature is predictive of feedback acceptance."

to:

> **causal intervention:** "Manipulating this sparse feature changes the Solver's probability of accepting feedback."

This distinction is critical for mechanistic interpretability.

The controlled correct-versus-incorrect feedback design further allows the project to distinguish different possible interpretations of a candidate feature.

For example, suppose amplifying a feature increases acceptance of **both correct and incorrect feedback**. The feature may represent general deference or willingness to revise rather than the ability to recognize useful feedback.

Conversely, if manipulating a feature selectively changes acceptance of correct criticism while having little effect on false criticism, the feature may be associated with a more discriminative feedback-evaluation mechanism.

### PRETRAINED SAE VS. INTERACTION-SPECIFIC SAE:

A secondary research question concerns whether general-purpose SAE representations are sufficient for studying multi-agent interactions.

A pretrained SAE has been trained on a broad activation distribution and may already contain features relevant to disagreement, revision, confidence, or correction.

The interaction-specific SAE will instead be trained directly on Solver activations generated during the controlled feedback experiment.

Importantly, this SAE remains **unsupervised**. It is not explicitly trained to identify "feedback acceptance." Rather, it learns a sparse dictionary over the activation distribution produced while the Solver participates in Solver-Critic interactions.

The comparison therefore asks:

> **Does an interaction-specific SAE produce sparse representations that better isolate feedback-related behavioral structure than a general pretrained SAE?**

The same downstream probe and causal intervention protocol can be applied to both representations.

### EXPECTED CONTRIBUTION:

The project aims to provide:

1. An analysis of sparse internal features associated with feedback acceptance and rejection.
2. A comparison between general pretrained and interaction-specific SAE representations.
3. Controlled causal evidence testing whether selected SAE features actually influence feedback uptake.
4. A reproducible framework for studying internal mechanisms of agent-to-agent feedback in multi-agent LLM systems.

A negative result remains scientifically meaningful.

---

## 4 Approach:

### Model and SAE Configuration

### Model and SAE Configuration

We plan to use **Gemma 3 IT** for both the Solver and Critic, with different role prompts. We chose this model family because Gemma Scope 2 provides pretrained SAEs that are compatible with the instruction-tuned Gemma 3 models.

If we use the proposed AWS `g5.xlarge` setup with one A10G GPU, **Gemma 3 4B IT** is the most practical starting point. The 12B model would need more GPU memory once activation extraction, SAE processing, and generation are included. If we are given access to a larger GPU, we can test 12B using the same pipeline.

We will make the final model choice after a small pilot confirms that the model performs well enough on MuSiQue, the Solver and Critic behave as intended, activation extraction and SAE reconstruction work properly, and the compute requirements are manageable.

### PHASE 1: PROJECT CONCEPTUALIZATION & EXPLORATORY ANALYSIS

#### [Weeks 1-2: Research Design, Literature Review, and Dataset Exploration]

- Finalize the primary research question and experimental objectives.
- Review relevant literature on:
  - Sparse Autoencoders and mechanistic interpretability;
  - SAE feature intervention and causal validation;
  - Solver-Critic and multi-agent LLM systems;
  - LLM response to external feedback and critique.
- Define the Solver-Critic architecture and interaction protocol at a conceptual level.
- Define the primary behavioral outcome: whether the Solver accepts or rejects Critic feedback.
- Define the 2 × 2 behavioral outcome matrix:

|                               | Solver Accepts Feedback | Solver Rejects Feedback |
| ----------------------------- | ----------------------- | ----------------------- |
| **Correct Critic Feedback**   | Correct + Accept        | Correct + Reject        |
| **Incorrect Critic Feedback** | Incorrect + Accept      | Incorrect + Reject      |

- Define the three primary Critic conditions:
  - natural/uncontrolled Critic feedback;
  - controlled correct feedback;
  - controlled incorrect feedback.
- Perform exploratory analysis of MuSiQue and determine:
  - dataset size and structure;
  - available ground-truth fields;
  - question and answer formats;
  - whether answers can be scored reliably;
  - whether the dataset is appropriate for generating controlled incorrect answers;
  - preprocessing or filtering requirements.
- Identify candidate open-weight LLMs based on:
  - instruction-following capability;
  - availability of compatible pretrained SAEs;
  - ability to extract and intervene on internal activations;
  - computational requirements.
- Identify candidate pretrained SAEs and compatible activation sites.
- Define the initial experimental controls, evaluation metrics, and dataset-splitting strategy.
- Produce the finalized experimental design and initial system architecture.

**Milestone:** By the end of Week 2, the research question, experimental design, dataset, behavioral outcomes, candidate model/SAE configuration, and evaluation strategy should be sufficiently defined to begin implementation.

---

### PHASE 2: PARALLEL SYSTEM DEVELOPMENT

#### [Weeks 3-5: Solver-Critic and SAE Development]

Following completion of the experimental design, development will proceed through two parallel workstreams.

#### Workstream A -- Solver-Critic System and Activation Collection

Two team members will focus on implementing the multi-agent experimental system and the pipeline required to collect the internal activations that will eventually be used to train and evaluate the SAEs.

- Implement reusable Solver and Critic components.
- Implement the basic interaction:

  `Question → Solver Attempt 1 → Critic Feedback → Solver Attempt 2`

- Implement the three Critic conditions:
  - natural/uncontrolled Critic feedback;
  - controlled correct feedback;
  - controlled incorrect feedback.
- Implement benchmark loading, preprocessing, and answer scoring.
- Implement interaction logging and behavioral labeling.
- Implement forward hooks for extracting Solver activations from the selected model layer and activation site.
- Verify activation collection during Solver Attempt 1 and Solver Attempt 2.
- Develop the pipeline for storing activation tensors separately from interaction metadata.
- Integrate the selected pretrained SAE.
- Verify that Solver activations can be encoded and reconstructed using the pretrained SAE.
- Measure reconstruction error.
- Develop the mechanism required to inject reconstructed activations back into the Solver.
- Perform an initial SAE intervention proof of concept:

  `Solver Activation → SAE Encode → Modify Latent → SAE Decode → Inject Activation → Continue Generation`

#### Workstream B -- Interaction-Specific SAE Development

One team member will simultaneously develop the custom SAE and its training infrastructure.

- Implement the selected SAE architecture.
- Implement:
  - SAE encoding;
  - SAE decoding;
  - reconstruction loss;
  - sparsity regularization;
  - training and validation loops;
  - checkpointing;
  - experiment configuration;
  - training diagnostics.
- Develop metrics for monitoring:
  - reconstruction error;
  - sparsity;
  - dead-feature rate;
  - feature activation frequency;
  - training stability.
- Test the SAE pipeline using temporary or sample activation tensors with the same expected dimensionality as the final Solver activations.
- Establish initial SAE hyperparameters and training configuration.
- Implement efficient loading of activation vectors generated by the Solver-Critic pipeline.
- Prepare the SAE training pipeline so that training can begin once sufficient real Solver activations have been collected.
- If there are available activations, we can start training immediately.

The custom SAE will remain an **unsupervised representation-learning model**. Feedback-acceptance labels will not be used during SAE training.

#### Integration

As the two workstreams mature, the custom SAE pipeline will be tested on a small sample of actual Solver activations.

This integration will verify:

- activation dimensionality and formatting;
- SAE input compatibility;
- reconstruction behavior;
- activation storage and loading;
- end-to-end SAE encoding and decoding.

**Milestone:** By the end of Week 5, the Solver-Critic activation pipeline and custom SAE training pipeline should both be operational and compatible. The team should also have demonstrated that SAE-based activation intervention is technically feasible using the selected model and activation site.

---

### PHASE 3: FULL INTERACTION & ACTIVATION DATASET GENERATION

#### [Week 6: Integration Full Forward-Pass Data Collection]

Run the finalized Solver-Critic experiment across the selected benchmark questions and Critic conditions.

For each benchmark question:

1. Generate Solver Attempt 1.
2. Evaluate the initial Solver answer against ground truth.
3. Cache the selected Solver activations.
4. Run the natural/uncontrolled Critic condition.
5. Generate and evaluate the corresponding Solver Attempt 2.
6. Run the controlled correct-feedback condition using the same initial Solver response.
7. Generate and evaluate the corresponding Solver Attempt 2.
8. Run the controlled incorrect-feedback condition using the same initial Solver response.
9. Generate and evaluate the corresponding Solver Attempt 2.
10. Cache the required post-feedback Solver activations.
11. Record Critic correctness, feedback acceptance/rejection, answer changes, and final correctness.
12. Assign each interaction to the appropriate cell of the 2 × 2 behavioral outcome matrix.

Behavioral metadata will be stored separately from high-dimensional activation tensors and linked through stable episode identifiers.

The resulting activation data will provide the inputs required for both the pretrained SAE analysis and interaction-specific SAE training.

Only activation data belonging to the **discovery/training split** will be used to train the custom SAE.

---

### PHASE 4: INTERACTION-SPECIFIC SAE TRAINING & REPRESENTATION GENERATION

#### [Weeks 7-10: SAE Training and Validation]

Once sufficient Solver activations have been collected, training of the interaction-specific SAE will begin using the training infrastructure developed during Weeks 3-5.

#### Interaction-Specific SAE

- Train the SAE on Solver activation vectors from the discovery/training split.
- Monitor:
  - training and validation reconstruction loss;
  - sparsity;
  - dead-feature rate;
  - feature activation frequency;
  - training stability.
- Tune SAE hyperparameters where necessary.
- Save intermediate checkpoints and training diagnostics.
- Evaluate reconstruction quality on held-out activation data.
- Generate sparse representations for the collected Solver-Critic interactions.

The experimental sequence will remain explicitly separated:

`Unsupervised SAE Training → Supervised Feature Discovery → Causal Intervention`

**Milestone:** By the end of Week 10, the team should have a validated interaction-specific SAE and corresponding sparse representations from both the pretrained and interaction-specific SAEs. This concludes the SAE modeling component.

---

### PHASE 5: PREDICTIVE FEATURE DISCOVERY

#### [Weeks 10-11: Predictive Modeling and Feature Selection]

Use the sparse SAE representations to identify internal features associated with whether the Solver accepts or rejects Critic feedback.

The primary behavioral target will be:

`solver_accepted_feedback ∈ {0,1}`

Critic feedback correctness will remain a separate experimental variable rather than being combined with acceptance into a four-class prediction target.

- Train regularized linear classifiers, such as L1 or elastic-net logistic regression, using SAE latent activations as input features.
- Evaluate predictive performance using:
  - AUROC;
  - F1 score;
  - balanced accuracy.
- Analyze predictive performance:
  - across all interactions;
  - under correct feedback;
  - under incorrect feedback;
  - under natural Critic feedback.
- Compare predictive performance between:
  - pretrained SAE features;
  - interaction-specific SAE features.
- Identify candidate features using:
  - coefficient magnitude;
  - coefficient direction;
  - activation frequency;
  - stability across data subsets or random seeds.
- Analyze changes in SAE feature activation before and after Critic feedback where appropriate.

For selected feature \(i\), this may include:

`Δz_i = z_i(after feedback) - z_i(before feedback)`

where \(z_i\) represents the activation of SAE feature \(i\).

#### Held-Out Validation

Before causal intervention:

- Evaluate candidate predictive features on the held-out validation split.
- Determine whether associations between candidate features and feedback acceptance generalize beyond the discovery data.
- Select a small final set of features for causal testing.

Features will not be selected using the intervention/test split.

**Milestone:** By the end of Week 11, the team should have a small validated set of SAE features associated with feedback acceptance or rejection and ready for causal testing.

---

### PHASE 6: CAUSAL SAE INTERVENTION

#### [Weeks 12-13: Feature Suppression and Amplification]

This is the final step of our experiment. The primary causal experiment will test whether features identified during predictive analysis actually influence how the Solver responds to Critic feedback.

For each selected feature:

1. Run the Solver to the selected intervention point.
2. Capture the relevant internal activation.
3. Encode the activation using the SAE.
4. Modify the selected SAE latent.
5. Decode the modified sparse representation.
6. Inject the reconstructed activation back into the Solver.
7. Continue generation.
8. Measure whether Solver feedback acceptance changes.

#### Intervention Conditions

Selected features will be evaluated using:

- **Target-feature suppression:** decrease or remove activation of the selected feature.
- **Target-feature amplification:** increase activation of the selected feature.
- **Intervention magnitude sweep:** evaluate multiple intervention strengths where feasible.

#### Required Controls

The causal experiment will include:

1. **No-intervention baseline**  
   Run the Solver normally without SAE reconstruction or feature modification.

2. **SAE reconstruction-only control**  
   Encode and decode the activation without deliberately modifying a feature. This controls for behavioral changes caused by SAE reconstruction error.

3. **Random-feature intervention control**  
   Apply matched interventions to unrelated SAE features to test whether observed effects are specific to the selected feature rather than a generic consequence of perturbing the SAE representation.

Where feasible, random features will be matched on characteristics such as activation frequency or typical activation magnitude.

#### Primary Outcome

The primary causal outcome will be the change in Solver feedback acceptance under intervention relative to the appropriate control condition.

#### Secondary Outcomes

Secondary outcomes will include:

- final-answer correctness;
- answer-change rate;
- acceptance of correct feedback;
- acceptance of incorrect feedback;
- rejection of correct feedback;
- resistance to incorrect feedback;
- generation coherence and output quality.

Intervention effects will be analyzed separately under correct and incorrect Critic feedback.

For example:

- if amplifying a feature increases acceptance of both correct and incorrect feedback, the feature may represent general deference or willingness to revise;
- if amplification selectively increases acceptance of correct feedback, the feature may be involved in evaluating feedback quality;
- if suppressing a feature reproducibly decreases acceptance relative to matched controls, this provides evidence that the feature is causally involved in feedback acceptance rather than merely correlated with it.

---

### PHASE 7: ANALYSIS & ROBUSTNESS

#### [Week 14: Final Statistical Analysis and Interpretation]

- NOTE: this really should be more about interpretation and analysis. If we did a good job implementing, when we run the experiment the tests will populate csv's with the data we want to see.

- Compare behavioral outcomes across:
  - natural Critic interactions;
  - controlled correct feedback;
  - controlled incorrect feedback.
- Compare pre-feedback and post-feedback sparse representations.
- Compare accepted and rejected feedback interactions.
- Compare predictive feature strength with measured causal intervention effects.
- Compare pretrained and interaction-specific SAE representations.
- Quantify intervention effects relative to:
  - no-intervention behavior;
  - SAE reconstruction-only controls;
  - random-feature controls.
- Evaluate whether intervention effects remain consistent across intervention strengths and data subsets.
- Analyze whether identified features appear related to:
  - general feedback acceptance;
  - resistance to feedback;
  - sensitivity to feedback correctness;
  - or broader behavioral changes unrelated to feedback.
- Report null or inconsistent intervention results explicitly rather than interpreting predictive association alone as evidence of mechanism.

The central distinction will be between:

**Observation:** a sparse feature is associated with Solver feedback acceptance.

and

**Causal evidence:** manipulating that feature produces a reproducible change in Solver feedback acceptance relative to appropriate controls.

**Visualization prep** by this week, we should basically be done. We should be focused on ensuring all of our visualizations for the paper are ready.

---

### PHASE 8: FINAL PAPER & REPRODUCIBILITY

#### [Week 15: Final Documentation, Paper, and Presentation]

- Finalize statistical analyses and figures.
- Document:
  - model configuration;
  - Solver and Critic prompts;
  - Critic conditions;
  - benchmark preprocessing;
  - activation site and layer;
  - pretrained SAE configuration;
  - custom SAE architecture and training procedure;
  - predictive models;
  - feature-selection procedure;
  - intervention magnitudes;
  - random seeds;
  - experimental controls.
- Prepare final tables and visualizations showing:
  - the 2 × 2 behavioral outcome distribution;
  - predictive feature performance;
  - pretrained versus interaction-specific SAE results;
  - causal intervention effects;
  - correct versus incorrect feedback behavior.
- Final organization checks for the codebase into reproducible components for:
  - Solver-Critic orchestration;
  - benchmark processing;
  - interaction generation;
  - activation caching;
  - SAE loading and training;
  - predictive modeling;
  - feature intervention;
  - evaluation and statistical analysis.
- Complete the final capstone paper.
- Prepare the final presentation and project demonstration.

## 5 Timeline:

**Weeks 1-2:** Proposal development, research conceptualization, literature review, dataset selection, EDA, and experimental design.

**Week 3:** Begin parallel development:

- Solver-Critic system, benchmark pipeline, and activation-extraction infrastructure;
- interaction-specific SAE architecture and training infrastructure.

**Week 4:** Continue parallel development. Integrate Solver/Critic conditions, behavioral scoring, activation extraction and storage, SAE training components, and pretrained SAE support. Begin testing the custom SAE on sample or available real activations.

**Week 5:** Complete core Solver-Critic and SAE infrastructure. Perform end-to-end integration and validation of interaction generation, activation extraction, SAE encoding/reconstruction, and activation intervention. Begin preliminary SAE training if sufficient real activations are available.

**Week 6:** Run the finalized Solver-Critic experiment and generate the full behavioral and activation datasets. Finalize discovery/training, validation, and intervention/test splits. Prepare finalized training activations for the interaction-specific SAE.

**Week 7:** Begin full interaction-specific SAE training and hyperparameter evaluation. In parallel, continue development of downstream predictive modeling, feature-selection, causal-intervention, control, and evaluation pipelines.

**Week 8:** Continue interaction-specific SAE training and evaluate reconstruction quality, sparsity, dead-feature rate, and training stability. Generate pretrained SAE representations in parallel. Continue development and testing of downstream experimental infrastructure.

**Week 9:** Finalize and validate the interaction-specific SAE. Generate sparse representations of the Solver-Critic interactions using both pretrained and interaction-specific SAEs. Begin predictive feature analysis as validated representations become available.

**Week 10:** Train regularized predictive models, analyze feedback-related SAE features, compare pretrained and interaction-specific SAE representations, and identify candidate features associated with feedback acceptance and rejection.

**Week 11:** Perform held-out validation of candidate SAE features and select the final feature set for causal testing. Finalize causal intervention conditions, reconstruction-only controls, random-feature controls, and intervention-strength configurations.

**Week 12:** Run target-feature suppression and amplification experiments across correct and incorrect Critic-feedback conditions.

**Week 13:** Complete causal intervention experiments and associated controls. Compare intervention effects across feedback conditions, intervention strengths, and SAE representations. Begin final interpretation and visualization of experimental results.

**Week 14:** Complete statistical analysis and robustness checks. Focus on interpretation of predictive and causal results and finalize tables, figures, and visualizations for the paper.

**Week 15:** Complete the research paper, repository documentation, reproducibility checks, final presentation, and project demonstration.

**TOTAL: 15 weeks**

### KEY MILESTONES:

- **Week 2:** Research question, dataset, eda experimental design, candidate model/SAE configuration, and evaluation strategy finalized.
- **Week 5:** Solver-Critic system, activation pipeline, SAE training pipeline, and end-to-end SAE intervention mechanism operational.
- **Week 6:** Full behavioral and activation datasets generated and experimental splits finalized.
- **Week 9:** Interaction-specific SAE trained and validated; pretrained and interaction-specific sparse representations available for analysis.
- **Week 11:** Candidate feedback-related SAE features selected using held-out validation.
- **Week 13:** Causal feature-intervention experiments and controls complete.
- **Week 14:** Primary statistical analyses, interpretation, and research figures complete.
- **Week 15:** Final paper, reproducible codebase, and presentation complete.

### DELIVERABLES BY WEEK 15:

- Reproducible Solver-Critic experimental framework.
- Behavioral dataset containing natural, controlled-correct, and controlled-incorrect Critic interactions.
- Solver activation dataset collected before and after Critic feedback.
- Pretrained SAE representations of Solver-Critic activations.
- Interaction-specific SAE trained on Solver-Critic activations.
- Predictive feature-selection analysis of feedback acceptance and rejection.
- Held-out evaluation of feedback-related sparse features.
- Controlled SAE feature-intervention experiments, including reconstruction-only and random-feature controls.
- Quantitative analysis of whether selected SAE features are causally involved in feedback acceptance.
- Comparison of pretrained and interaction-specific SAE representations.
- Final research paper/report.
- Documented and reproducible GitHub repository.
- Final project presentation and demonstration.

## 6 Expected Number Students:

**RECOMMENDED: 3 STUDENTS**

### ISRAEL OMONIYI -- SOLVER-CRITIC SYSTEM, DATA PIPELINE, AND EXPERIMENT INFRASTRUCTURE

Primary responsibility:

- Design and implement the modular Solver-Critic pipeline.
- Implement benchmark ingestion and evaluation.
- Implement controlled correct/incorrect critic-feedback generation.
- Implement interaction logging and behavioral-label generation.
- Build the activation-caching infrastructure.
- Maintain experiment configuration and reproducibility.
- Support the causal-intervention pipeline by ensuring the forward-pass infrastructure can be reused during Objective 4.

This role owns the foundation of the project. The remaining experiments depend on the interaction and activation datasets produced by this pipeline.

All team members should be involved in designing and validating the controlled interaction protocol because errors in this stage would affect the validity of every downstream experiment.

---

### JOSHUA SCHECHTER -- SPARSE AUTOENCODER TRAINING AND REPRESENTATION ANALYSIS

Primary responsibility:

- Integrate the pretrained SAE.
- Determine compatible activation sites and layers with the team.
- Implement the interaction-specific SAE.
- Train the SAE on cached Solver activations.
- Evaluate reconstruction loss, sparsity, dead features, and training stability.
- Produce SAE latent representations for downstream predictive modeling.
- Compare pretrained and interaction-specific SAE representations.
- Support intervention experiments requiring SAE encoding, latent manipulation, and decoding.

This is expected to be one of the most technically difficult components because poor SAE reconstruction or unstable sparse representations could compromise downstream feature discovery and intervention.

---

### RAYE OJI  -- PREDICTIVE MODELING, FEATURE SELECTION, AND CAUSAL INTERVENTION

Primary responsibility:

- Build the linear-probe modeling pipeline.
- Train regularized logistic regression models predicting feedback acceptance.
- Evaluate AUROC, F1, and balanced accuracy.
- Analyze probe coefficients and feature stability.
- Select candidate SAE features using the discovery split.
- Validate candidate features on held-out interactions.
- Implement target-feature suppression and amplification experiments.
- Implement random-feature and reconstruction-only controls.
- Analyze causal effects on feedback acceptance and final correctness.

This role owns the transition from correlational feature discovery to causal testing.

---

### SHARED RESPONSIBILITIES:

All three students will contribute to:

- experimental design;
- controlled-feedback protocol;
- selection of activation sites;
- evaluation methodology;
- statistical analysis;
- interpretation of results;
- robustness checks;
- paper writing;
- final presentation;
- repository documentation.

The project should be treated as a single experimental pipeline rather than three independent subprojects. In particular, the behavioral labels created by Student 1, SAE representations produced by Student 2, and candidate features identified by Student 3 must remain aligned through stable episode identifiers and frozen experimental splits.

---

## 7 Possible Issues:

### TECHNICAL CHALLENGES AND SOLUTIONS:

The project has a few technical and experimental risks that we need to address before running the full study. Most of these decisions will be tested during a small pilot so that we do not lock in assumptions too early.

### 1. Model and pretrained-SAE compatibility

**Issue:**  
We plan to use the Gemma 3 instruction-tuned model family with Gemma Scope 2 SAEs. However, pretrained SAEs are tied to specific model versions, layers, and activation sites, so we still need to confirm that the exact combination we choose works correctly for our Solver setup.

**Plan:**  
During the technical pilot, we will test the selected Gemma 3 model with the corresponding Gemma Scope 2 SAE and check activation extraction, SAE encoding/decoding, reconstruction quality, memory use, and runtime. The pretrained SAE gives us a lower-risk starting point, while the interaction-specific SAE will be trained as a separate comparison.

---

### 2. Choosing the activation layer, site, and token position

**Issue:**  
A transformer produces activations at every layer and for every token. This means that saying we will compare "before feedback" and "after feedback" is not enough unless we also define which activation we are comparing.

**Plan:**  
The pilot will be used to test reasonable layer, activation-site, and token-position or aggregation choices. Once we find a setup that is technically stable and useful for the analysis, we will freeze it before the full experiment. Token-level activations may still be collected separately for training the interaction-specific SAE.

---

### 3. Amount of activation data needed for the interaction-specific SAE

**Issue:**  
The custom SAE will need more activation data than the linear probe because it is learning a sparse representation directly from the Solver's internal activations. A small number of pooled interaction vectors may not be enough.

**Plan:**  
We will collect token-level activations from the discovery/training split and use the pilot to estimate how much data and storage are practical. We will monitor reconstruction loss, sparsity, dead-feature rate, feature activation frequency, and training stability before deciding whether the amount of activation data is sufficient.

---

### 4. Interaction-specific SAE quality

**Issue:**  
The interaction-specific SAE may not train well enough to produce useful sparse features. Poor reconstruction or unstable features would make downstream analysis difficult.

**Plan:**  
We will evaluate the custom SAE using reconstruction quality, sparsity, dead-feature rate, feature frequency, and training stability. If the custom SAE performs poorly, the compatible pretrained SAE will still allow the main probe and intervention analyses to continue. The custom-SAE result would then be reported as part of the comparison rather than treated as a failed project.

---

### 5. Defining feedback acceptance consistently

**Issue:**  
Most interactions should be easy to describe as the Solver accepting or rejecting the Critic's feedback, but some responses may be ambiguous.

**Plan:**  
We will test the feedback-acceptance labeling approach on pilot examples and make sure the team can apply it consistently. The final rule will be fixed before the full dataset is generated. More detailed edge-case handling will be documented in the Objective 1 pipeline specification.

---

### 6. Scoring the Natural Critic's feedback

**Issue:**  
The Natural Critic is allowed to respond freely. If its response does not clearly state which answer it supports, it may be difficult to decide whether the feedback was correct.

**Plan:**  
The Natural Critic will return a structured response that includes a verdict, an advocated answer, and an explanation. This allows the advocated answer to be compared with the MuSiQue ground truth while still allowing the Critic to behave naturally.

---

### 7. Imbalanced behavioral outcomes

**Issue:**  
We can control whether the Critic receives a correct or deliberately incorrect target, but we cannot control whether the Solver accepts or rejects that feedback. Some correctness × acceptance outcome groups may therefore be much smaller than others.

**Plan:**  
The pilot will measure how often the Solver accepts and rejects correct and incorrect feedback. We will report the full 2 × 2 outcome distribution and investigate any missing or very small groups before the full run. The main requirement is enough behavioral variation to support the probe and causal analyses, not simply forcing the data to look balanced.

---

### 8. Probe stability and interpretation

**Issue:**  
SAE features may be correlated with one another, and probe coefficients can change depending on feature scale, regularization, or the particular data sample. A large coefficient alone does not prove that a feature is important.

**Plan:**  
We will standardize the probe inputs, use regularized models such as L1 or elastic-net logistic regression, and check feature stability across seeds or data subsets. Candidate features will also be evaluated on held-out data before they are used in the causal experiment.

A probe using the original model activations will be used as a predictive baseline. SAE features are not required to outperform raw activations because the value of the SAE also comes from producing sparse, interpretable, and directly manipulable features.

---

### 9. Feature-selection leakage

**Issue:**  
If we select candidate SAE features using the same interactions later used for the causal experiment, we could overestimate how meaningful those features are.

**Plan:**  
The discovery/training, validation, and intervention/test splits will remain separate. Candidate features will be identified using discovery data and checked on the validation split before the intervention/test data is used.

---

### 10. Separating a real feature effect from a general intervention effect

**Issue:**  
Changing an internal activation can affect the Solver even if the selected SAE feature is not actually responsible for feedback uptake. SAE reconstruction itself can also slightly change the activation.

**Plan:**  
The causal experiment will compare the selected-feature intervention against three controls:

1. **No-intervention replay**
2. **SAE reconstruction-only**
3. **Matched random-feature intervention**

We will also test more than one intervention strength where practical. The exact point in the Solver's processing where the intervention is applied will be chosen during the pilot and fixed before the causal experiment.

---

### 11. Compute, memory, and experiment size

**Issue:**  
The full study includes three Critic conditions, activation caching, SAE processing, probe analysis, causal controls, and intervention-strength tests. This can increase GPU memory use, storage requirements, and runtime quickly.

**Plan:**  
The team is currently considering an AWS `g5.xlarge` setup with one A10G GPU as the starting configuration. Under that setup, Gemma 3 4B IT is the practical model to pilot first. We will measure actual GPU memory, runtime, prompt length, and activation-storage requirements instead of relying only on estimates.

If larger-memory GPU access is available, we can test Gemma 3 12B IT using the same pipeline. The full run size will only be decided after the pilot shows what the available compute can handle reliably.

---

### 12. Replay stability during causal intervention

**Issue:**  
Objective 4 requires us to rerun Solver Attempt 2 and then change an internal feature. If an unmodified replay already produces a different answer from the stored interaction, it becomes harder to say that a later difference was caused by our intervention.

**Plan:**  
We will use fixed generation settings, fixed seeds where applicable, and pinned model/library versions. An unmodified replay is expected to reproduce the stored behavior under the frozen setup. Any replay mismatch will be flagged and investigated rather than automatically treated as a valid intervention result.

---

### 13. Version and configuration drift

**Issue:**  
Model checkpoints, SAE releases, Python libraries, prompts, and dataset versions can change during the semester. Small changes could make later experiments difficult to compare with earlier ones.

**Plan:**  
Once the technical pilot is complete, we will pin the tested software versions and record the model revision, SAE configuration, dataset version, prompts, generation settings, random seeds, and code version used for each experiment.

---

### Open Questions to Resolve During the Pilot

The following decisions are intentionally being left open until we have tested the system:

1. Which GPU resources will be available for the main experiment?
2. Which Gemma 3 model size will be used for the final experiment?
3. Which Gemma Scope 2 SAE release and configuration will be used?
4. Which model layer and activation site will be analyzed?
5. Which token position or aggregation rule will be used for aligned activation comparisons?
6. Which framework will be used for activation extraction and intervention?
7. What final rule will be used to label feedback acceptance in ambiguous cases?
8. What exact structured output will be required from the Critic?
9. How will plausible incorrect answers be generated for the Controlled Incorrect condition?
10. How will MuSiQue questions be grouped or filtered when creating the experimental splits?
11. How much activation data and storage will be needed for the interaction-specific SAE?
12. At what point in Solver Attempt 2 will the causal intervention be applied, and what intervention strengths will be tested?

---

### Risk Mitigation Timeline

- **Technical pilot:** Test the proposed Gemma 3 / Gemma Scope 2 setup, measure compute use, verify activation extraction, check Natural-Critic behavior, and test the initial feedback-labeling approach.
- **System integration:** Verify activation storage, SAE reconstruction, Solver replay, and a small end-to-end activation-intervention test.
- **Before full data generation:** Freeze the model configuration, prompts, Critic conditions, data splits, activation policy, and experiment run plan.
- **SAE and probe stages:** Monitor SAE quality, probe stability, held-out performance, and feature-selection leakage.
- **Causal stage:** Run the selected-feature interventions together with the no-intervention, reconstruction-only, and random-feature controls.
- **Final analysis:** Report positive, negative, inconsistent, and inconclusive results, and document the final configuration needed to reproduce the experiments.
