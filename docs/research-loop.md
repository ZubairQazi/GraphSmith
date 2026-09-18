# Bounded DrugComb research loop

GraphSmith can now plan, execute, verify and critique up to two DrugComb research
cycles, carrying the whole-project evidence and earlier cycle results into each
model call. This backend is separate from the original synthetic research engine.
It chooses among implemented scientific controls; it does not generate training
code or recursively delegate work.

## Limits agreed for the first version

| Resource | Session limit |
|---|---:|
| Research cycles | 2 |
| Model calls | 4: one planning and one critique turn per cycle |
| Reported model tokens | 50,000 stopping budget |
| New fits | 336 |
| Concurrent GPUs | 3 |
| Maximum epochs per fit | 400 |
| Worker training alarm | 900 seconds per fit |
| Reserved training GPU time | 302,400 seconds (84 hours), a ceiling, not an estimate |

Planning and critique use ChatGPT-authenticated Codex CLI. Input plus output usage
is charged after each call; cached input tokens are not counted twice. Codex does
not expose a hard token limit for this individual CLI turn, so one call can exceed
the remaining 50,000-token allowance. The controller then stops; the overshoot is
recorded. Four is a hard ceiling on CLI turns, not on internal model requests or
transport retries. Prompts are compact (65 KB maximum project context) and ask
for concise responses. No automatic model-call retries occur.

A planning turn needs room for a later critique call; after planning, at least
5,000 tokens of headroom must remain before training is submitted. That reserve
is an admission heuristic, not a guarantee of the next turn's usage. Missing
usage or an interrupted external action stops the session for reconciliation.
Unknown usage is never treated as free. Failed calls/experiments remain recorded.

The training alarm covers the native fit. Slurm also bounds each GPU allocation
at 20 minutes; startup and verification mean total allocated GPU time can exceed
the reserved fit-time ceiling. CPU verification and scheduler waits are separate.
No test scores are produced. Validation reuse and retrospective historical test
information mean these are exploratory decisions, not untouched confirmation.

## Project evidence and decisions

A session freezes the independently verified project review and feature audit,
plus a compact project synthesis. Every model call sees all reviewed comparison
IDs across settings, current verified prerequisite results, available recipes,
budgets, and prior cycle plans/results/critiques. It does not retrieve only the
latest local branch. Narrative notes are bounded; full evidence remains on disk.

The planner must state a hypothesis, mechanism, competing explanation,
falsification condition, scope limitations, practical effect and supporting **and
conflicting** evidence IDs. Unknown evidence IDs, recipes, extra fields and invalid
budgets are rejected. Model text never becomes shell commands or training code.
The critic interprets numerically verified results and chooses continue/stop.
Measured outputs remain authoritative; narrative claims are not independent proof.

The initial trusted recipe catalog is:

- `pd_relationships`: DD+PPD with real PD versus degree-preserving rewired PD.
  This tests PD relationships, with DD/PPI/root paths held fixed. Duplicate
  rewired edges and degree multiplicity remain caveats.
- `protein_semantics`: DD+PPD with real versus random protein features.
  Width, missingness, graph and drug inputs are matched. Distribution/effective
  rank differences remain a limitation.

Each selected recipe covers all seven settings, four folds and three seeds:
168 fits. B3A uses the both-cold validation subset; its checkpoint selection still
uses the native B3-style validation partition. It is not the canonical B3A test
score or independent replication of B3. Recipes cannot be repeated in a session.
The planner can stop without launching either recipe.

This is bounded autonomy within the catalog, not open-ended discovery of new
architectures. New interventions, dataset changes and study protocols require
new reviewed implementations. The controller does not edit papers, code or Git.

## Run

The controller needs Python 3.10+ and access to the independent
[exptrack](https://github.com/ZubairQazi/exptrack) checkout. Model workers use the
existing dpsyn environment (Python 3.9 is supported by these adapters). Set both
interpreter paths explicitly; do not install the synthetic dependency lock into
the training environment.

Edit `examples/drugcomb-research.json` for your paths and allocation. It contains
local project paths, not bundled data. `provider.model` empty means the Codex CLI
default; an explicit model is preserved. Login is checked using `codex login status`.
No API key is needed. Credentials are not copied into session artifacts.

In a CPU allocation with access to the account's Codex executable/login:

```bash
export PYTHONPATH=/path/to/GraphSmith
export PYTHONDONTWRITEBYTECODE=1
python -m graphsmith.dpsyn_research dpsyn-research init \
  /path/to/GraphSmith/examples/drugcomb-research.json --output /private/research-001
python -m graphsmith.dpsyn_research dpsyn-research run \
  /private/research-001 --poll-seconds 30 --max-wait-seconds 3600
python -m graphsmith.dpsyn_research dpsyn-research status /private/research-001
```

The regular `python -m graphsmith dpsyn-research ...` entry point is also available
in an environment with the native CLI's dependencies. `step` performs one durable
state transition; `run` repeatedly advances and polls until stopped or its wall
window expires. Run the same session again to continue an ordinary waiting state.
There is no background service after this foreground process exits. Choose a long
enough CPU allocation or resume it to let GPU results trigger the next critique.
Polling and unmet prerequisites consume no model turns.

`init` and `status` do not spend model tokens or submit training. **`step` and `run`
authorize the bounded model calls and catalog training described in the frozen
configuration.** The example waits for the current benchmark's verified completion
before any planning. GPU work is submitted through exptrack; Slurm completion
alone does not count as accepted scientific evidence.

## Outputs and failure handling

- `config.json`, `context.json`, `review.json`, `feature_audit.json`, `frozen.json`:
  frozen inputs and source hashes; source edits invalidate resume.
- `state.json`: authoritative reservations, calls, reported tokens and phases.
- `call-N-input.json`, `call-N-output.json`: exact context, schema, decisions and
  metering receipts. No hidden chain-of-thought is requested or persisted.
- `cycle-N/`: frozen native configurations, exptrack manifest/registry, attempts,
  accepted-artifact inventory, paired results and numerical evidence.
- `project_understanding.json`: cumulative verified evidence and model critiques,
  fed into subsequent decisions. It does not overwrite the curated project notes.

Before accepting a cycle, workers verify data/source hashes, exact feature and
cell-scaling contracts, and split identity. The summarizer checks accepted artifact
hashes, parameter counts, matched rows, and independently recomputes validation
metrics from predictions. Full-setting means, descriptive SDs, missing subgroup
metrics, training warnings and unflagged sensitivity remain explicit.

A lock prevents two controllers from spending the same reservation. Intent is
written before a model call or submission. A crash during those actions produces
an `uncertain` session rather than automatically repeating an expensive request.
Use exptrack's reconciliation tools when needed; this first loop has no automatic
retry or manual override command. Preserve uncertain evidence and create a new
bounded session only after checking what actually ran.

Tests use fixture model replies and tracker state; real-data manifest validation
performs no model calls and submits no training. Live Codex authentication/network
and a complete autonomous two-cycle scientific run are separate validation steps.
