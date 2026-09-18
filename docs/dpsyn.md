# Whole-project review comes first

For dpsyn research, start with `dpsyn-review` before choosing a training branch.
The earlier `dpsyn-profile` is a structural audit, not a complete literature or
experiment review. Its historical filename filter misses important experiments.

```bash
python -m graphsmith dpsyn-review /path/to/graphsmith_project_review.json \
  --output /path/to/new/project-review
```

Run this read-only scan in a CPU allocation. The manifest specifies result roots,
explicit comparison recipes, expected fold/seed keys, and context files. The
analysis checkout provides `scripts/analysis/graphsmith_project_review.json` and
`run_graphsmith_project_review.sh` for the current project. Update the manifest
when new results arrive; each review writes a new immutable snapshot directory.

The review discovers CSV runs by schema, including `e400_*`, molecular shards and
nested diagnostic baselines. It preserves all original row fields in runs.jsonl,
groups within source/configuration, and only pools explicitly enumerated recipe
shards. Duplicate keys, recorded split mismatches, missing metrics and hard
failures cannot enter paired contrasts. Training warnings have a separate
sensitivity summary. Missing diagnostics remain unknown. Other CSV schemas are
cataloged, not interpreted as run evidence. Non-CSV legacy experiment directories
still require a separate review; a successful scan does not certify them.

Read `report.md`, `review.json`, and the copied `context/` evidence together.
Historical test scores are retrospective development evidence. Current source
snapshots document current mechanisms; they do not prove historical code identity.
The review reuses identified real-data audits and split contracts; it does not
silently rerun or replace them. `complete.json` records source/output hashes.

Every new research proposal should reference a completed project review, identify
supporting and conflicting evidence IDs, name the affected settings and task,
state competing explanations, define matched controls, and state what result
would change the project decision. Compare feature utility, graph utility and
dataset limitations together. Selective synergy is one branch of this agenda.
Update the project synthesis after each branch; a local success is not a global
conclusion. Existing campaign commands remain available for reproduction; this
review-first workflow is a documented requirement, not a runtime launch gate.

This is evidence infrastructure plus an assistant-authored synthesis. The native
synthetic planner has not been upgraded into an autonomous dpsyn research planner.

---

# Real-data reconnaissance for drug synergy

`dpsyn-profile` is a real-data adapter, separate from the synthetic campaign
engine. It uses the specified dpsyn checkout's loader and exact split functions.
It does not train models or invoke an LLM. Its hypothesis candidates are explicit
templates populated with observed training evidence; they are not discoveries.

Use the existing dpsyn environment (on the original cluster, `gnn_het`). The
adapter was verified with that environment's Python 3.9; the native synthetic
worker still requires GraphSmith's supported Python >=3.11 environment. Do not
install GraphSmith's synthetic-engine dependency lock into that environment.
Expose both checkouts using `PYTHONPATH`; the module CLI needs no installation.
Run profiling in a CPU compute allocation, not a login node. It loads graph and
feature artifacts, audits joins, and hashes inputs twice to detect changes.

```bash
export DPSYN_DATA_ROOT=/nfs/turbo/coe-dkoutra/zqazi/drugprot_data
export DPSYN_WORK_ROOT=/nfs/turbo/coe-dkoutra/zqazi/multigraph_gnn_results/graphsmith
export PYTHONPATH=/home/zqazi/GraphSmith:/home/zqazi/Multigraph-GNN-Training
export PYTHONDONTWRITEBYTECODE=1
python -m graphsmith dpsyn-profile \
  --dpsyn-repo /home/zqazi/Multigraph-GNN-Training \
  --output "$DPSYN_WORK_ROOT/reconnaissance-001" \
  --results-root /nfs/turbo/coe-dkoutra/zqazi/multigraph_gnn_results/dpsyn \
  --setups B1 LCOW B2 B2P B3 B3A B4 --num-folds 4 --seeds 0
```

The output directory must not already exist. A failed run can leave partial
outputs; `complete.json` is written only after all outputs finish. Preserve failed
evidence and use a new output directory to retry. Shared input artifacts and dpsyn
caches are never written. This adapter loads locally trusted PyTorch/NumPy files;
it is not a safe importer for arbitrary untrusted serialized objects.

## Outputs and scientific scope

- `profile.json`: schema/duplicate/join checks, feature alignment, PD/PPI edge
  diagnostics, split counts, coverage, missing expression, and training-label
  summaries. Stored all-label DD edges are ignored, as they are in dpsyn.
- `hypotheses.json`: untested candidates with exact profile pointers, observed
  values, comparisons, competing explanations/controls and falsification criteria.
  The initial candidates use the first requested setup/fold/seed's training rows;
  examine the other profiles before freezing an experiment.
- `historical_results.json`: optional retrospective inventory of recursive
  `results.json` and top-level `sweep_*.csv`. Means stay within each source and
  recorded configuration, with available fold/seed keys and exclusions. Missing
  metadata stays unknown. Missing planned runs require the original planned grid.
  Duplicate fold/seed groups are excluded from aggregation; failed leakage checks,
  run errors, degeneracy and training flags also exclude rows. Absent historical
  diagnostics do not establish validity. No cross-source pooling occurs.
- `historical_results.csv`: browsable projection of those groups and metrics.
  Split flags such as LOW_TRAIN_FRAC cause conservative exclusion, even where
  expected under entity splits; flagged does not automatically mean invalid.
- `provenance.json`: artifact hashes, dpsyn source hashes, adapter hash and runtime
  versions. Historical artifact hashes are included in the results inventory.
- `report.md`: readable structural profile and candidate experiment descriptions.
- `complete.json`: output checksums and completion marker.

Held-out partitions contribute structural counts, not label summaries. Splitting
itself can use labels (e.g. B1 stratification), and different folds reuse rows in
training. This is exploratory cross-validation reconnaissance, not sealed test
isolation. Historical test metrics are kept outside candidate generation; using
them manually to select hypotheses also makes the follow-up exploratory. Future
confirmation needs a separately frozen evaluation protocol.

## Independent feature-control audit

Before a new attribution campaign, run `dpsyn-feature-audit` in a CPU allocation:

```bash
python -m graphsmith dpsyn-feature-audit \
  --dpsyn-repo /path/to/Multigraph-GNN-Training \
  --review /path/to/completed/project-review \
  --reference-contract /path/to/alignment/contract.json \
  --output /path/to/new/feature-audit
```

It requires an independently verified project review and the previously frozen
B3 alignment contract. It reads actual canonical inputs, checks 60 combinations
of independent drug/protein treatments over three seeds, and freezes all 84 paper
split assignments. It compares B3 assignments exactly with the earlier contract
and checks covered-training-cell statistics using the native encoder. It does not
train a predictive model, score test labels or claim improved accuracy.

New attribution runs must explicitly set `feature_protocol=independent_v1` and
an explicit `prot_feat_mode` in the dpsyn configuration. This fixes input widths,
missingness and random-stream independence; it intentionally changes the baseline
transform too. Compare all arms under that protocol. Existing historical campaigns
retain legacy settings. Cell interventions are not implemented by this audit;
it verifies that node treatments leave raw cell data unchanged and that existing
cell scaling uses covered training cells. Tensor hashes alone do not prove useful
features; the subsequent matched models still need to test that hypothesis.

## Validation-only campaigns

`dpsyn-campaign` freezes a two-arm manifest and runs each planned dpsyn job in a
bounded subprocess. Use an allocated GPU for real training. The launcher
`examples/dpsyn-campaign.slurm` checks GPU availability; configure the same
checkout, data, work-root and Python variables as the profiling launcher.

```bash
python -m graphsmith dpsyn-campaign examples/dpsyn-target-coverage.json \
  --dpsyn-repo /home/zqazi/Multigraph-GNN-Training \
  --output /path/to/new/campaign
python -m graphsmith dpsyn-campaign-report /path/to/new/campaign
```

The example is specific to the original workspace and its recorded profile;
change the evidence path for another installation. `common` and each arm's
`config` accept dpsyn RunConfig fields. Seeds, fold indices, output paths,
validation-only mode and empty floors are campaign-owned. The example compares
DD to DD+PPD on B3 fold 0 seed 0, with standardization and a 400-epoch maximum.
All resolved defaults, the hypothesis, evidence pointer/value, input and source
hashes, runtime versions and job budgets are frozen in `plan.json` before workers
start. The worker saves only validation predictions and metrics, never test scores.

`state.json` records reservations, failures, elapsed worker time and evidence
hashes. Failed attempts count against the frozen job budget; no retry or resume
is implemented. An interrupted campaign may retain a running reservation; do
not treat it as completed. Workers also set an alarm to bound orphan execution.
Source/input changes invalidate the campaign. Reports recompute validation metrics
from hashed predictions and require matching splits, triplet keys and labels.

An optional `split_contract` manifest path additionally pins shared configuration,
resolved input artifact fingerprints, historical references and exact split
hashes. Each worker checks its actual contextual-identity assignments before
model construction. The contract's split-array artifact records train/validation/
test row indices without labels. The analysis checkout's
`scripts/analysis/prepare_graphsmith_alignment.py` prepares the contract against
the standardized molecular series (DD+PPD fingerprint arm, B3, four folds, seeds
0/1/2). Historical CSVs omit assignment arrays and some hyperparameters: matching
their recorded settings/counts is not proof of identical historical membership.
The generated contract states these limitations rather than filling gaps silently.

For multiple fold/seed pairs, the report includes an equal-weight mean of paired
wq-MRR differences, descriptive sample SD, observed/expected pair counts and
parameter-count agreement. This does not make correlated folds independent.

`summary.json` and `report.md` compare the two arms overall and within zero-, one-,
and two-target-covered endpoint groups. Subgroup ranking rebuilds queries inside
each group, so candidate pools and uninformative MRR floors differ. The exported
training frequencies enable later matching, but this pilot does not adjust for
frequency or parameter count. Validation also selects checkpoints. No random-row
control is run. The verdict remains exploratory/inconclusive, regardless of the
observed difference; additional matched controls and independent confirmation
are needed for attribution.

Native synthetic `research`, `resume` and `finalize` do not accept these output
directories. Automated hypothesis planning and adaptive follow-up remain future
work; this backend executes an explicitly frozen, user-reviewable comparison.

## Feature failures and selective-synergy feasibility

`dpsyn-diagnose` audits the recorded standardized/whitened DD+PPD comparison,
analyzes a completed aligned campaign's validation predictions, and reconstructs
the original measured panels before extmap's neutral-band exclusion. New
GraphSmith campaigns reject `feat_transform=whiten`; standardization is the active
choice. The native dpsyn option and pinned historical grids remain available for
reproduction. The whitening audit records paired EARLY_BEST warnings and a
sensitivity summary excluding them instead of confusing warnings with failures.

Run in a CPU allocation, using the dpsyn environment:

```bash
python -m graphsmith dpsyn-diagnose \
  --dpsyn-repo /home/zqazi/Multigraph-GNN-Training \
  --results-root /nfs/turbo/coe-dkoutra/zqazi/multigraph_gnn_results/dpsyn \
  --campaign /path/to/completed/aligned-campaign \
  --output /path/to/new/diagnosis
```

The launcher is `examples/dpsyn-diagnose.slurm`. The current adapter expects the
original workspace's historical comparison files and a complete 12-pair B3
campaign with a split contract; it is not a generic arbitrary-dataset importer.

Panel reconstruction uses the original canonical build metadata's drug-ID map,
which differs from today's larger model graph. It must reproduce every canonical
triplet identity, label and mean ZIP and the recorded pre-band row count before
reporting. Replicates are averaged before thresholding, as in the original
builder. Source outlier/mapping/cell filters are preserved; source data is never
rewritten. The stored fold column is not used for runtime evaluation splits.

Outputs include `whitening_audit.json`, `feature_failure_summary.csv`,
`cell_ranking_summary.csv`, `observed_pair_panels.csv`,
`selective_observed_pairs.csv`, `selectivity_feasibility.json`, `hypotheses.json`,
`report.md` and completion checksums. Full per-run diagnostics retain undefined
metrics and subgroup sample sizes; summaries expose valid metric counts.

Observed selectivity means at least one positive mean ZIP >10, but positives in
at most 10% of measured cells, with at least 10 measured cells. Neutral responses
remain in the denominator; unmeasured cells remain unknown. Coverage sensitivity
at 20/30/50 cells and a Wilson-upper-bound criterion are reported. Binomial bounds
do not remedy nonrandom panel selection or dependent cell lines. This is a
retrospective feasibility census, not a validated selectivity predictor or a
normal-cell safety assay.

The cell-ranking diagnostic fixes each drug pair and ranks its measured
validation cells against a training-only cell-rate baseline and exact random-tie
expectations. It uses at least three cells and both classes, still on the original
extreme-label task. These metrics differ from the original fixed-anchor/cell
partner ranking. A new selectivity predictor needs a separately frozen,
pair-grouped or drug-held-out evaluation retaining neutral measurements.


## Neutral-inclusive selective-synergy pilot

`dpsyn-selectivity` runs a bounded feature-model campaign on actual DrugComb
measurements before the neutral-band filter. It requires the prior exact B3
split contract and checks every original row assignment before fitting models.
The original drug universe and four-fold/three-seed allocation are retained;
additional measured cells are loaded through dpsyn, and neutral-only drugs
outside the original split universe are excluded with recorded counts.

```bash
python -m graphsmith dpsyn-selectivity \
  --dpsyn-repo /path/to/Multigraph-GNN-Training \
  --contract /path/to/alignment/contract.json \
  --output /path/to/new-selectivity-run
```

Run in a GPU allocation. The fixed budget is 72 fits, each at most 120 epochs.
Five molecular/cell feature models compare full binary labels, extreme-only
training, continuous ZIP, permuted cell context, and a within-pair ranking loss.
A sixth predicts selective-pair status directly. Training-only cell response,
constant, drug-rate, and measured-panel-design controls accompany the models.
These controls use actual AtomPair and CCLE features with training-entity-fitted
standardization; they do not run the DD+PPD GNN or use whitening.

A selective panel has at least one positive mean ZIP >10 but positives in at most
10% of measured cells. Minimum coverage is 10, with >=50-cell sensitivity.
Neutral responses count; missing cells are unknown. Positive-cell localization
and pair-level selective-status AP are separate endpoints. Triplet-to-panel
probability aggregation assumes conditionally independent cell responses.

Outputs include a frozen plan, source snapshots, identities, expanded split
arrays, histories, validation-only predictions, metrics, summary and checksummed
completion manifest. Checkpoint selection reuses validation data; source loaders
can access all labels. This is exploratory, not a sealed test evaluation. A fixed
cell permutation retains arbitrary cell identity, so its comparison tests feature
alignment. Cancer-panel selectivity does not establish normal-cell safety.

Analysis-side launcher and independent report:
`Drug_Protein_Heterophily/scripts/run_graphsmith_selectivity.sh` and
`scripts/analysis/report_graphsmith_selectivity.py`. Detailed protocol and results:
`Drug_Protein_Heterophily/notes/graphsmith-selective-synergy-pilot.md`.


## Native channel controls

`dpsyn-channels prepare` freezes the previously verified selective-synergy dataset
and its original B3 assignments; `dpsyn-channels worker --index 0..11` runs one
fold/seed comparison. Supply `--dpsyn-repo` to both stages, `--parent` and a new
`--output` to preparation, and `--run` to workers. Use compute allocations.

The fixed campaign compares native ego, DD, PPD and mean-fused DD+PPD, corresponding
empty-edge controls, and DD+PPD with rewired DD or PD. Both localization and direct
selective-pair models are trained: 216 fits across four folds and three seeds.
Each fit is bounded at 160 epochs. It uses the actual dpsyn encoder and decoder,
with native standardization, training-only positive DD edges, and no whitening.

Controls match initial weights and registered parameters within a channel set;
empty channels can disconnect parameters or use native self-feature fallbacks.
Rewiring diagnostics, active parameter counts, stopping flags and source hashes
are recorded. This is validation development on B3, not an independent test claim.
Analysis entry points and protocol are in the analysis repository:
`scripts/run_graphsmith_channels.sh`,
`scripts/analysis/report_graphsmith_channels.py`, and
`notes/graphsmith-channel-selectivity.md`.
