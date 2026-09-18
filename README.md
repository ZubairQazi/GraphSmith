# GraphSmith

An auditable experiment engine for graph-learning research. It runs **synthetic unordered pair classification** on a CPU, freezes structured hypotheses, and produces traceable development evidence plus optional isolated final-test scores. The full [proposal](docs/proposal.md) is the roadmap, not a list of completed capabilities.

For real drug-synergy research, begin with the [whole-project evidence review](docs/dpsyn.md), which joins historical result families with data audits and pipeline context before a new campaign.

## Quick start

From `~/Documents/GraphSmith`:

```sh
source .venv/bin/activate
python -m graphsmith validate examples/synthetic-pairs.json
python -m graphsmith profile examples/synthetic-pairs.json
python -m graphsmith research examples/synthetic-pairs.json
```

The last command prints the campaign directory and report path. It runs 25 jobs: five paired seeds for a class-prior baseline, feature-only pair MLP, one-layer GCN, one-layer mean GraphSAGE, and a GCN with an additional raw endpoint feature path. Every neural encoder uses the same symmetric pair-decoder architecture.

```sh
python -m graphsmith resume runs/RUN_ID
python -m graphsmith report runs/RUN_ID
python -m unittest discover -s tests -v
```

A new environment can be installed with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

The lock captures the tested Python 3.14 environment. The development engine is tested on macOS ARM and Linux; sealed workers currently require macOS. The module CLI works directly from the repository even without an editable installation.

## Implemented

- Deterministic synthetic graph and query generation, with a known target rule.
- A manifest declaring topology visibility, feature/target provenance, split rules, and fixed metric/threshold policy.
- Unique unordered pairs, explicit query IDs, separate background topology, and train/validation partitions.
- Strict shape, endpoint, duplicate/reverse-pair, label, and finite-value validation.
- Structured hypothesis planning through an offline provider, ChatGPT-authenticated Codex CLI, or OpenAI Responses API adapter. No LLM account is needed for offline runs.
- Evidence-backed critique with supported, rejected, and inconclusive exploratory outcomes.
- CPU training with early stopping, selected checkpoints, prediction files, training curves, and validation-access counts.
- Single-worker orchestration with per-job process timeouts, job-count and cumulative worker-time caps.
- Atomic job reservations and state files, an append-only event log, and a campaign process lock.
- Source/config/data/environment fingerprints; resume rejects changed inputs or runtime.
- A Markdown report and JSON summary, recalculated from predictions and checked against recorded evidence hashes.

## Optional sealed campaign

Start the local Docker evaluator using [the isolation guide](docs/isolated-evaluation.md), then:

```sh
python -m graphsmith research examples/sealed-pairs.json --evaluator http://127.0.0.1:8765
python -m graphsmith freeze runs/RUN_ID
python -m graphsmith finalize runs/RUN_ID
```

Labels stay in the evaluator's private volume. Research workers have no networking or access to that volume. The selected ensemble and predictions are frozen before the one-shot score is released. The 26-job example reserves one job for final inference. Read the guide for the trusted-owner boundary and platform limits.

## Plan with your Codex sign-in

```sh
python -m graphsmith research examples/codex-pairs.json
```

This uses your existing ChatGPT-authenticated Codex CLI, with no API key. See [provider setup and critic policy](docs/planning.md) for details.

## Scientific scope

The latent binary node group is observed through a noisy feature. A query is positive when its endpoints share a group, subject to independent label noise. Background edges are sampled before query outcomes and favor same-group or different-group endpoints according to `neighbor_agreement`. That parameter controls relative edge probabilities; it is not an exact realized graph statistic. Query existence and labels do not construct topology.

All permitted node features and background edges are visible under an explicit transductive protocol. Unique query pairs are split 70/30 into training and development validation. Missing queries are unknown, never negative examples. No learned preprocessing is applied. Node labels used by the generator are not included in the bundle.

The shared decoder uses endpoint embedding sum, absolute difference, and product. The GCN ablation concatenates raw node features with normalized GCN aggregation before encoding; it adds parameters, which the report explicitly identifies as a confound. GraphSAGE uses raw features plus mean neighbor features. Encoders are deliberately small and dense; they are not tuned benchmark implementations.

Validation selects checkpoints and is reused for comparisons. Development comparisons and hypothesis verdicts are **exploratory**, even with five seeds. Optional finalization reports one separately held-out ensemble score. Five seeds on one graph/split do not provide independent confirmation or population uncertainty. No novelty claim is made. Sealed campaigns use 60/20/20 splits and a private service seed, as described in the isolation guide.

## Failure and budget semantics

Each worker is reserved in durable state before launching. Failed, timed-out, and interrupted jobs count toward `max_jobs`. Completed jobs are never relaunched by resume. An interrupted reservation is marked interrupted and conservatively charged its full reserved wall time; it is not automatically retried. Resume continues only pending jobs within the original frozen budget. Start a new campaign for a deliberate rerun.

A worker has its own POSIX alarm as well as the orchestrator timeout, so it remains time-bounded if the orchestrator is killed. The campaign seconds limit covers cumulative launched-worker wall time (including startup), not report generation or idle time. Termination/scheduling overhead may slightly exceed a boundary. Memory limits and peak-memory measurement are not implemented. Event logs provide an operational trail; atomic state files are the authoritative job ledger. Local files are not tamper-proof.

## Real drug-synergy data

The `dpsyn-profile` adapter profiles actual prepared drug/protein data through an
existing dpsyn checkout, audits split coverage and joins, inventories historical
results, and emits evidence-linked candidate experiments. It runs in the dpsyn
environment on a compute node. See [setup and scope](docs/dpsyn.md). This is a
reconnaissance workflow, separate from synthetic `research` campaigns. The
`dpsyn-campaign` backend additionally executes frozen two-arm validation-only
comparisons and exports subgroup evidence. `dpsyn-diagnose` audits feature
failures and selective-synergy feasibility from real measurements and validation
predictions. New campaigns use standardization; whitening is reserved for
historical dpsyn reproduction. Adaptive hypothesis planning remains
unimplemented.

## Deliberately unfinished

The node task adapter, native real-data training beyond the dpsyn backend, context features in the native trainer, independent multi-graph confirmation, adaptive replanning, generated-code sandbox, cross-campaign evidence retrieval, and broader graph tasks are future work. Sealed worker isolation currently supports macOS only. This release does not complete the full M1/MVP.

See [next milestones](docs/roadmap.md).
