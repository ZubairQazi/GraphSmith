# GraphSmith — Graph Learning Research Agent

Version: 0.1 proposal · 15 September 2026

## 1. Purpose and scope

Build a research assistant that takes a graph-learning question and a dataset, proposes falsifiable hypotheses, runs controlled experiments, challenges its findings, and produces an auditable research report. GraphSmith is a working name; name availability has not been checked.

The initial product is an experimental research system. It does not claim mathematical proof, autonomous scientific novelty, or guaranteed improvements. Inspiration comes from CausalSmith's discovery-and-verification loop; empirical replication replaces formal verification in the initial version. A future Lean component would constitute a separate capability.

Primary user: a graph ML researcher who can define the task and review scientific conclusions. Initial interface: command line plus Markdown reports. A dashboard is optional after the research loop works.

## 2. Supported tasks and release order

| Task | Prediction unit and example | Proposed release | Additional requirements |
|---|---|---|---|
| Node classification | One node; document category or account class | MVP | Node masks; transductive versus inductive protocol |
| Edge/pair classification | Observed or queried pair; synergy versus antagonism, transaction category | MVP | Separate query pairs from message-passing edges; explicit direction and context |
| Edge/pair regression | Pair and optional context; continuous synergy score | Next | Regression loss, scale handling, grouped splits |
| Link prediction | Candidate pair; whether a relationship exists | Next | Candidate universe, negative sampling, ranking protocol |
| Graph classification/regression | Entire graph; molecular property | Next | Graph batching, pooling, graph-level splits |
| Node/edge anomaly detection | Anomaly score for a node or interaction | Later | Label availability, contamination assumptions, imbalance-aware evaluation |
| Temporal forecasting | Future node/edge values or events | Later | Time cutoffs, rolling evaluation, temporal features |
| Knowledge-graph completion | Missing entity in a typed triple | Later | Relation-aware scoring and filtered ranking |
| Clustering/community discovery | Node grouping | Deferred | Independent definition of quality; unstable objectives |
| Graph generation/optimization | A new graph or combinatorial solution | Deferred | Different action spaces and validity/objective checks |

Heterogeneous, directed, weighted, and temporal graphs describe the input representation; they are not prediction tasks. Heterophily is a research property, not a task. Keep these axes independent in configuration.

MVP accepts static homogeneous attributed graphs for both tasks. Its pair interface includes optional context features. Typed heterogeneous message passing is a subsequent extension; do not imply full heterogeneous support simply because an input contains node-type metadata.

## 3. Edge classification is a first-class use case

For every supervised example, represent a query as (source, destination, optional context, target). The message-passing graph is a separate object and may contain different relationships from the query pairs.

Example: drug A, drug B, and cell-line features predict synergy versus antagonism. A missing measurement is unknown, not an antagonistic example. Any neutral band, threshold, aggregation of replicates, and treatment of ambiguous measurements must be frozen in the dataset manifest before search.

The first baseline is a feature-only pair MLP. For unordered drug pairs, use a symmetric representation such as concatenated embedding sum, absolute difference, and elementwise product, followed by context features. GNN variants replace the endpoint feature encoder while keeping the decoder fixed. Directed tasks retain ordered endpoint information.

Split modes must be named and interpreted separately:

- Pair-held-out: a canonical unordered pair belongs to exactly one partition; by default all its contexts stay together. Measures generalization to new combinations of possibly known entities.
- One-endpoint-unseen: evaluation pairs have exactly one endpoint absent from training queries. Define whether its features/background topology are available at inference.
- Both-endpoints-unseen: neither evaluation endpoint occurs in training queries.
- Context-held-out: entire contexts are excluded from training. May reuse pairs and measures a different generalization question.
- Temporal: partitions follow information availability time.

Random example splits are allowed only when explicitly justified. Reverse copies and replicated measurements must not straddle partitions. Context-held-out and pair-held-out are separate experiments unless a combined constraint is requested.

If query-edge existence is genuinely available at prediction time, it may be visible under an explicit edge-classification protocol. If the graph was built from outcomes, held-out target edges and reverse copies must be excluded. Labels must never be encoded in visible edge types, weights, or features. The system must refuse an unspecified topology-visibility policy.

## 4. User workflow

1. Supply a dataset manifest, task, research question, and budget.
2. Validate data and freeze the evaluation contract.
3. Produce a graph profile and run required baselines.
4. Propose a hypothesis with mechanism, prediction, competing explanation, and falsification experiment.
5. Implement one controlled intervention and execute experiments.
6. Critique results; run targeted ablations or reject the hypothesis.
7. Freeze the selected method and evaluate through an isolated final-test service.
8. Export the complete evidence record, including failures and inconclusive findings.

Example research question: Does keeping endpoint features separate from aggregated neighbor features improve pair classification when neighborhood information is misleading?

## 5. System components

| Component | Responsibility | Output |
|---|---|---|
| Orchestrator | Deterministic state transitions, budgets, retries, resumability | Append-only event log |
| Dataset adapter | Parse inputs, align IDs, construct splits and permitted topology | Versioned dataset bundle |
| Profiler | Degree/connectivity, feature quality, class balance, task-specific diagnostics | Profile with provenance |
| Planner | Propose hypotheses and discriminating experiments | Structured hypothesis and experiment plan |
| Implementer | Modify allowed model modules in isolated working directories | Code diff and smoke-test result |
| Runner | Train permitted jobs with resource limits | Checkpoints, predictions, runtime and metrics |
| Evaluator | Apply frozen metrics and enforce split isolation | Evaluation records |
| Critic | Check confounds, propose counterexamples and ablations | Supported/rejected/inconclusive assessment |
| Evidence store | Index previous runs and their applicability | Searchable hypothesis history |
| Reporter | Summarize only traceable evidence | Markdown report and machine-readable results |

Planner, implementer, and critic are logical roles; MVP may call one model sequentially. Multiple agents are not a prerequisite. Model access sits behind a provider adapter; confirm supported authentication and unattended execution behavior during implementation. Training and evaluation must run without an LLM dependency.

## 6. Data and component contracts

DatasetManifest: dataset_id, version/hash, source/license, node and edge schema, feature provenance, target definition, missing-label policy, directedness, context schema, split policy, topology visibility, preprocessing fit scope.

TaskSpec: task_type, target unit, output dimension, symmetry policy, loss, primary metric, secondary metrics, positive class, threshold-selection rule, evaluation unit and generalization claim.

DatasetBundle: node_features, message_edges, optional edge_features, query_index or node_target_index, optional context_features, targets, split_ids, stable source IDs. Store withheld targets outside the agent-visible bundle.

TaskAdapter interface: validate(bundle, spec), build_model(config), make_batches(split), compute_loss(prediction, target), evaluate(prediction, target). Evaluation is implemented in protected infrastructure, not generated model code.

Hypothesis: id, statement, mechanism, expected measurable outcome, competing explanation, planned intervention, falsification test, applicable regimes, parent evidence IDs.

Experiment: id, hypothesis_id, dataset_hash, split_hash, code_hash, environment_lock_hash, model config, seeds, resource caps, metric policy, status, artifact paths.

Evidence: experiment IDs, observed effect, variability, limitations, verdict, next experiment, references. Store failed proposals as well as successes. Retrieval must filter by task, graph regime and split compatibility.

## 7. Experiment policy

Required node baselines: majority predictor, feature-only MLP, GCN, GraphSAGE, and one selected heterophily-oriented model after checking implementation compatibility.

Required pair baselines: class-prior predictor, symmetric or ordered pair MLP, GCN encoder with the same decoder, and GraphSAGE encoder with the same decoder. Context-only and endpoint-only baselines are required when context is present.

Allowed initial interventions: aggregation choice; separate ego/neighbor paths; depth; residual connections; endpoint decoder; restricted feature transformations. Change one conceptual factor at a time. Compare encoder and decoder changes separately before combining them.

Use equal declared search budgets and comparable early stopping. Record parameter count, training time, epochs and peak memory. Do not describe equal parameter count alone as a fair comparison.

Synthetic graphs vary mixing structure, degree, feature informativeness and noise. For pair tasks, define a separate known target-generation rule, such as an endpoint/context interaction. Node homophily is not automatically an explanation for edge-label performance. Compute label-dependent diagnostics from training labels only.

Screening may use one seed and must be marked exploratory. Confirmation uses at least five paired seeds on the same splits, plus additional independent splits or graph draws when making broader claims. Seed variability is not evidence of population-level uncertainty. Report dependence limitations for examples sharing endpoints.

Classification: macro-F1 or average precision as the predeclared primary metric, depending on the task; report ROC-AUC and class prevalence where appropriate. Thresholds are selected on validation data and frozen. Regression: predeclare MAE or RMSE; correlations are secondary. Link prediction uses an explicit candidate set and benchmark-compatible ranking evaluator when applicable.

## 8. Evaluation isolation and integrity

Training labels support fitting and diagnostics. Development validation supports adaptive search; repeated access is logged. A separate confirmation split may be used for shortlisted hypotheses. The final test is inaccessible to planner, implementer, runner and critic until selection is frozen.

Protected evaluator runs separately from generated code and validates prediction IDs, shapes and finite values. The agent may not modify splits, metrics, evaluator source or test files. Merely hiding filenames is insufficient isolation; enforce filesystem/process permissions or a separate evaluation service.

Fit learned preprocessing on the permitted training scope. Unsupervised transductive preprocessing using all allowed features/topology requires an explicit protocol declaration. Hash caches with split and topology policy to prevent cross-split reuse.

Test exposure closes the campaign. Follow-up revisions require a new untouched evaluation set or must be labeled exploratory with test reuse disclosed. No automated retries to improve a released test result.

Audit duplicate IDs, reverse-edge leakage, outcome-derived features, illegal future information, cross-partition groups and checkpoint selection. Benchmark-specific protocols override generic assumptions and must be recorded.

## 9. Execution and budget controls

Proposed stack: Python, PyTorch/PyG, NetworkX for small-graph diagnostics, YAML configuration, SQLite metadata, JSONL event logs, local artifact directories, and Markdown reporting. Pin exact compatible versions when implementing. Start with one local worker; add a Slurm execution adapter afterward.

Example initial campaign budget: eight candidate configurations, one-seed screening, two finalists with five-seed confirmation, per-job timeout, maximum epochs, GPU-hour ceiling and LLM-call ceiling. Count baseline work and failed jobs against the budget. These are configurable planning defaults, not runtime estimates.

State machine: created → validated → baselined → proposed → executing → critiqued. Critique can return to proposal, proceed to frozen → final_evaluated → reported, or terminate as inconclusive/budget_exhausted. Failed jobs retain diagnostics; repair attempts are capped. Persist state after each transition and resume without duplicating completed runs.

Suggested CLI contract: graphsmith validate CONFIG; graphsmith profile CONFIG; graphsmith research CONFIG; graphsmith resume RUN_ID; graphsmith finalize RUN_ID; graphsmith report RUN_ID.

## 10. Deliverables per campaign

- Dataset and task manifests, frozen splits, dependency lock and source revision.
- Graph profile with the data scope of every statistic.
- Hypothesis ledger and every attempted configuration.
- Predictions, checkpoints where practical, seeds, logs, compute usage and failures.
- Baseline comparison, ablations, effect variability and failure cases.
- Report distinguishing exploratory patterns, confirmed empirical findings and untested explanations.
- Reproduction command; no invented citations or unsupported novelty claims.

“Self-improvement” initially means reusing evidence to improve later experiment selection. It does not mean updating model weights or allowing the agent to alter evaluation rules.

## 11. Milestones and acceptance criteria

M1 — trustworthy experiment engine: node and pair adapters; deterministic toy examples; ID/split validation; baseline training; budget enforcement; reproducible records.

M2 — bounded research loop: structured hypothesis generation; allowed code changes; critic; resume; evidence-backed report. Demonstrate both a rejected and an inconclusive hypothesis, not only a success.

M3 — generalization: evaluate on several graph regimes; add regression and link prediction; then typed heterogeneous encoders.

M4 — research-system evaluation: compare fixed-budget GraphSmith against random search, a conventional tuning procedure, and a generic coding agent. Measure reproducibility, invalid-run rate, compute cost, and independently confirmed hypothesis quality. Evaluate on datasets withheld from system development. No publication novelty claim until a dedicated related-work review.

Required correctness checks: unordered pair predictions are invariant to endpoint swap; directed pairs may differ; changing final-test labels cannot affect search artifacts; intentionally injected duplicate/reverse-edge leakage is rejected; permutation of node IDs preserves aligned predictions within numerical tolerance; interrupted runs resume without duplicate billing; all report numbers trace to recorded outputs; budget caps terminate work.

A successful MVP completes one node-classification and one pair-classification campaign within budget, with valid baselines, one controlled ablation, full traceability, and enforced test isolation. Beating a baseline is not required for correctness.

## 12. Optional formal extension

A later module could search for counterexamples on small graphs, formulate a precise claim under explicit assumptions, and attempt Lean proofs of selected aggregation or symmetry properties. A numerical test is not a proof. Formal verification proves the encoded statement under its assumptions, not its empirical usefulness or novelty.

## References and design basis

- CausalSmith: https://github.com/Jiyuan-Tan/CausalSmith — inspiration for discovery, checking and accumulated results; the proposed system above is an independent design.
- PyTorch Geometric introduction: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html — graph data conventions, node/edge features and graph batching.
- Open Graph Benchmark link tasks: https://ogb.stanford.edu/docs/linkprop/ — task-specific splits and evaluators for link property prediction.

The architecture, priorities, defaults and acceptance criteria in this document are proposed engineering decisions, not claims made by those sources.
