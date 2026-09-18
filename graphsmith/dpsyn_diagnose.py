"""Dataset/feature failure diagnostics and observed-panel selectivity feasibility.

Uses real validation predictions and reconstructs source panels before the
extmap neutral-band exclusion. This is exploratory analysis, not final testing.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from .dpsyn import fingerprint, table, triplet_key
from .io import read, write


def upper_wilson(positive, total):
    if not total:
        return None
    z = 1.959963984540054
    p = positive / total
    return (p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def panel_summary(scores):
    """Missing cells are absent, not negatives; all-zero panels are not selective."""
    n = len(scores)
    positive = sum(score > 10 for score in scores)
    return {'measured_cells': n, 'positive_cells': positive,
            'positive_fraction': positive/n if n else None,
            'wilson95_upper': upper_wilson(positive, n),
            'selective_observed_10pct': n >= 10 and 0 < positive <= .1*n,
            'selective_wilson_upper_10pct': n >= 10 and positive > 0 and upper_wilson(positive, n) <= .1}


def reconstruct(root, graph):
    source = root / 'drugcomb/drugcombs_scored.csv'
    mapping = root / 'drugcomb/drugcomb_name_to_dbid_mapping_extended.csv'
    cellmap = root / 'ccle/cell_line_ccle_map.csv'
    canonical = root / 'unified/canonical_folds_extmap.csv'
    metadata = root / 'unified/canonical_folds_extmap_meta.json'
    meta = read(metadata)
    # The canonical task was filtered against the builder's older drug universe,
    # not every drug in today's larger model graph. Preserve that universe.
    drugmap = Path(meta['drug_id_map_csv'])
    source_files = (source, mapping, cellmap, drugmap, canonical, metadata)
    sources = [fingerprint(p) for p in source_files]
    names, cells = {}, {}
    alias_conflicts = 0
    for row in table(mapping):
        db = row['drugbank_id'].strip()
        if not db or db.lower() == 'nan':
            continue
        for col in ('name', 'name_norm'):
            name = row.get(col, '').strip()
            for key in (name, name.lower()):
                if key:
                    alias_conflicts += int(key in names and names[key] != db)
                    names[key] = db  # Same ordered last-write behavior as canonical builder.
    for row in table(cellmap):
        name, mid = row['drugcomb_name'].strip(), row.get('ccle_modelid', '').strip()
        if name and mid and mid.lower() != 'nan':
            cells[name] = cells[name.lower()] = mid
    allowed = {r['drugbank_id'] for r in table(drugmap)}
    aggregates, counts = {}, Counter()
    with source.open() as stream:
        for row in csv.DictReader(stream):
            counts['source_rows'] += 1
            try:
                score = float(row['ZIP'])
            except (ValueError, TypeError):
                counts['missing_score'] += 1
                continue
            if not math.isfinite(score) or abs(score) > 200:
                counts['nonfinite_or_outlier'] += 1
                continue
            a, b = (names.get(row[k].strip()) or names.get(row[k].strip().lower()) for k in ('Drug1', 'Drug2'))
            if a not in allowed or b not in allowed:
                counts['unmapped_drug'] += 1
                continue
            if a == b:
                counts['self_pair'] += 1
                continue
            cell = row['Cell line'].strip()
            if not (cells.get(cell) or cells.get(cell.lower())):
                counts['unmapped_cell'] += 1
                continue
            key = tuple(sorted((a, b))) + (cell,)
            rec = aggregates.setdefault(key, [0., 0, 0, 0])
            rec[0] += score
            rec[1] += 1
            rec[2] += int(score > 10)
            rec[3] += int(score < -10)
    measured = {}
    for (a, b, cell), rec in aggregates.items():
        key = (a, b, cell.replace('\\', '').strip().upper())
        if key in measured:
            raise ValueError('Cell normalization merged distinct source triplets')
        measured[key] = {'mean_zip': rec[0]/rec[1], 'replicates': rec[1],
                         'replicate_sign_conflict': bool(rec[2] and rec[3])}
    canonical_rows = table(canonical)
    canonical_map = {triplet_key(r): r for r in canonical_rows}
    if len(canonical_map) != len(canonical_rows):
        raise ValueError('Duplicate canonical keys')
    extremes = {k for k, v in measured.items() if abs(v['mean_zip']) > 10}
    if extremes != set(canonical_map):
        raise ValueError(f'Raw reconstruction differs from extmap: missing={len(set(canonical_map)-extremes)}, extra={len(extremes-set(canonical_map))}')
    for key, row in canonical_map.items():
        value = measured[key]['mean_zip']
        if int(value > 10) != int(row['label']) or abs(value-float(row['mean_zip'])) > 1e-5:
            raise ValueError('Reconstructed labels or ZIP means disagree with canonical data')
    if len(measured) != meta['dedup_stats']['n_triplets_pre_band']:
        raise ValueError('Reconstructed measured universe differs from canonical build metadata')
    if sources != [fingerprint(p) for p in source_files]:
        raise ValueError('Source inputs changed during reconstruction')
    counts.update(measured_triplets=len(measured), canonical_triplets=len(canonical_rows),
                  excluded_neutral_triplets=len(measured)-len(canonical_rows),
                  triplets_with_replicates=sum(v['replicates'] > 1 for v in measured.values()),
                  triplets_with_opposite_extreme_replicates=sum(v['replicate_sign_conflict'] for v in measured.values()),
                  alias_conflicts=alias_conflicts)
    return measured, canonical_rows, dict(counts), sources


def whitening_audit(root, config_fields):
    records, sources = [], []
    for i in range(12):
        paths = [root / f'sweep_higsyn_molecular_60800849_{i}.csv',
                 root / f'standardized_molecular_60828758_{i}.csv']
        groups = []
        for path in paths:
            sources.append(fingerprint(path))
            rows = [r for r in table(path) if r['arm'] == 'ddppd|none']
            keyed = {(r['setup'], r['fold'], r['seed']): r for r in rows}
            if len(rows) != 3 or len(keyed) != 3:
                raise ValueError('Missing/duplicate preprocessing comparison rows')
            groups.append(keyed)
        a, b = groups
        if a.keys() != b.keys():
            raise ValueError('Preprocessing comparison splits do not match')
        for key in a:
            x, y = a[key], b[key]
            for field in set(config_fields) & x.keys() & y.keys() - {'feat_transform', 'tag', 'output_dir'}:
                if x[field] != y[field]:
                    raise ValueError(f'Confounded whitening comparison: {key} {field}')
            for field in ('n_rows', 'n_train', 'n_val', 'n_test'):
                if x[field] != y[field]:
                    raise ValueError('Preprocessing comparison split counts differ')
            if x['feat_transform'] != 'whiten' or y['feat_transform'] != 'std':
                raise ValueError('Incorrect preprocessing arms')
            if any(r.get('error') or r.get('leak_check') != 'PASS' or
                   any(flag in (r.get('train_flags') or '') for flag in ('NEVER_SELECTED', 'DEGENERATE_SCORES'))
                   for r in (x, y)):
                raise ValueError('Flagged/error run in preprocessing comparison')
            records.append({'setup': key[0], 'fold': int(key[1]), 'seed': int(key[2]),
                'whiten_wq_mrr': float(x['wq_mrr']), 'std_wq_mrr': float(y['wq_mrr']),
                'whiten_training_flags': x.get('train_flags', ''), 'std_training_flags': y.get('train_flags', ''),
                'difference': float(x['wq_mrr'])-float(y['wq_mrr'])})
    summary = []
    for setup in sorted({r['setup'] for r in records}):
        rows = [r for r in records if r['setup'] == setup]
        unflagged = [r for r in rows if not r['whiten_training_flags'] and not r['std_training_flags']]
        summary.append({'setup': setup, 'n': len(rows),
                        **{f'mean_{key}': statistics.mean(r[key] for r in rows) for key in ('whiten_wq_mrr', 'std_wq_mrr', 'difference')},
                        'paired_sd': statistics.stdev(r['difference'] for r in rows),
                        'unflagged_pairs': len(unflagged),
                        'mean_difference_unflagged': statistics.mean(r['difference'] for r in unflagged) if unflagged else None})
    return {'records': records, 'summary': summary, 'sources': sources,
            'scope': 'Retrospective historical test scores, separate from validation diagnosis. Equal-weight means over 4 folds x 3 seeds; SD is descriptive. Historical code/input hashes unavailable.',
            'decision': 'No consistent benefit in this matched series. Use std for active GraphSmith campaigns; retain historical whitening reproduction.'}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        if rows:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def diagnosis(campaign, canonical, measured):
    import numpy as np
    from dpsyn import metrics as M
    from dpsyn.development import clean
    state, plan = read(campaign/'state.json'), read(campaign/'plan.json')
    from .data import digest
    if digest(plan) != state['plan_hash']:
        raise ValueError('Campaign plan changed')
    contract_path = plan['manifest']['split_contract']
    recorded_contract = next(x for x in plan['inputs'] if x['path'] == contract_path)
    if fingerprint(contract_path) != recorded_contract:
        raise ValueError('Split contract changed')
    contract = read(plan['manifest']['split_contract'])
    for source in contract['inputs']:
        if fingerprint(source['path']) != source:
            raise ValueError('Current dataset/features differ from prediction campaign inputs')
    arrays = np.load(contract['split_arrays']['path'])
    if fingerprint(contract['split_arrays']['path']) != contract['split_arrays']:
        raise ValueError('Frozen arrays changed')
    output = []
    for job in state['jobs']:
        if job['config'].get('rewire'):
            continue
        if job['status'] != 'completed':
            raise ValueError('Missing real-PD model run')
        for field in ('result', 'predictions'):
            if fingerprint(job[field]['path']) != job[field]:
                raise ValueError('Prediction/result evidence changed')
        result = read(job['result']['path'])
        if result['predictions_sha256'] != job['predictions']['sha256']:
            raise ValueError('Prediction hash does not match result')
        if result['training']['flags']:
            raise ValueError('Training flags require review before diagnosis')
        rows = table(job['predictions']['path'])
        key = f"B3_f{job['fold']}_s{job['seed']}"
        ids = [int(r['row_id']) for r in rows]
        if len(set(ids)) != len(ids) or set(ids) != set(arrays[key+'_validation']):
            raise ValueError('Validation membership changed')
        cell_y = defaultdict(list)
        tr = arrays[key+'_train']
        for i in tr:
            r = canonical[int(i)]
            cell_y[triplet_key(r)[2]].append(int(r['label']))
        prior = statistics.mean(int(canonical[int(i)]['label']) for i in tr)
        for row, i in zip(rows, ids):
            rowkey = tuple(sorted((row['drug1'], row['drug2']))) + (row['cell'],)
            if rowkey != triplet_key(canonical[i]) or int(row['label']) != int(canonical[i]['label']):
                raise ValueError('Prediction identity/label mismatch')
            labels = cell_y[row['cell']]
            row['cell_prior'] = (sum(labels)+10*prior)/(len(labels)+10)
            row['source_mean_zip'] = measured[rowkey]['mean_zip']
            row['replicate_conflict'] = measured[rowkey]['replicate_sign_conflict']
        masks = {'all': lambda r: True,
                 'no_ccle': lambda r: r['cell_has_ccle'] == 'False',
                 'has_ccle': lambda r: r['cell_has_ccle'] == 'True',
                 'no_targets': lambda r: r['covered_endpoints'] == '0',
                 'one_targeted_drug': lambda r: r['covered_endpoints'] == '1',
                 'both_targeted': lambda r: r['covered_endpoints'] == '2',
                 'near_label_boundary': lambda r: abs(r['source_mean_zip']) < 15,
                 'far_label_boundary': lambda r: abs(r['source_mean_zip']) >= 15,
                 'replicate_conflict': lambda r: r['replicate_conflict'],
                 'no_replicate_conflict': lambda r: not r['replicate_conflict']}
        for name, mask in masks.items():
            selected = [r for r in rows if mask(r)]
            if not selected:
                continue
            y = np.array([int(r['label']) for r in selected]); p = np.array([float(r['probability']) for r in selected])
            met = clean(M.basic(y, p))
            output.append({'fold': job['fold'], 'seed': job['seed'], 'stratum': name, 'rows': len(y),
                'prevalence': float(y.mean()), 'auroc': met['auroc'], 'ap': met['ap'],
                'brier': float(np.mean((p-y)**2)), 'mean_score': float(p.mean())})
        # Reverse the question: fix a drug pair and rank measured validation cells.
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[tuple(sorted((row['drug1'], row['drug2'])))].append(i)
        queries = [ix for ix in groups.values() if len(ix) >= 3]
        y = np.array([int(r['label']) for r in rows])
        scores = {'model': np.array([float(r['probability']) for r in rows]),
                  'train_cell_rate': np.array([r['cell_prior'] for r in rows]),
                  'constant_tie': np.full(len(rows), prior)}
        for method, score in scores.items():
            output.append({'fold': job['fold'], 'seed': job['seed'], 'stratum': 'pair_conditioned_cell_ranking',
                           'method': method, **clean(M.within_query(queries, y, score))})
    if len({(r['fold'], r['seed']) for r in output}) != 12:
        raise ValueError('Expected all 12 matched validation runs')
    return output


def run(args):
    import sys
    sys.path.insert(0, str(args.dpsyn_repo.resolve()))
    from dpsyn import config as C
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    graph = C.GRAPHS[C.DEFAULT_GRAPH]
    measured, canonical, counts, sources = reconstruct(C.DATA_ROOT, graph)
    whitening = whitening_audit(args.results_root, C.RunConfig.__dataclass_fields__)
    write(out/'whitening_audit.json', whitening)
    write_csv(out/'whitening_paired.csv', whitening['records'])
    panels = defaultdict(list)
    retained = defaultdict(list)
    for key, rec in measured.items():
        panels[key[:2]].append(rec['mean_zip'])
        if abs(rec['mean_zip']) > 10:
            retained[key[:2]].append(rec['mean_zip'])
    panel_rows = []
    for pair, scores in sorted(panels.items()):
        row = {'drug1': pair[0], 'drug2': pair[1], **panel_summary(scores)}
        extreme = retained[pair]
        row['retained_extmap_cells'] = len(extreme)
        row['extmap_positive_fraction'] = sum(v > 10 for v in extreme)/len(extreme) if extreme else None
        panel_rows.append(row)
    write_csv(out/'observed_pair_panels.csv', panel_rows)
    write_csv(out/'selective_observed_pairs.csv', [r for r in panel_rows if r['selective_observed_10pct']])
    feasibility = {'scope': 'Retrospective full measured-panel census, not a held-out predictive result. Missing cells are unknown; no normal-cell safety inference.',
        'definition': 'At least one mean ZIP > 10, in <=10% of measured cells; minimum 10 measured cells. Neutral measured responses remain in the denominator.',
        'counts': counts, 'pairs': len(panel_rows),
        'coverage': {str(n): sum(r['measured_cells'] >= n for r in panel_rows) for n in (2, 5, 10, 20, 30, 50)},
        'observed_selective': {str(n): sum(r['selective_observed_10pct'] and r['measured_cells'] >= n for r in panel_rows) for n in (10, 20, 30, 50)},
        'selective_with_wilson_upper_le_10pct': sum(r['selective_wilson_upper_10pct'] for r in panel_rows),
        'uncertainty_note': 'Wilson bounds assume binomial sampling; nonrandom cell panels and dependent cell lines limit that interpretation.',
        'all_zero_panels': sum(r['positive_cells'] == 0 for r in panel_rows),
        'sources': sources}
    write(out/'selectivity_feasibility.json', feasibility)
    diagnostics = diagnosis(args.campaign, canonical, measured)
    write(out/'validation_diagnostics.json', diagnostics)
    summarized = []
    for name in sorted({r['stratum'] for r in diagnostics if r['stratum'] != 'pair_conditioned_cell_ranking'}):
        rows = [r for r in diagnostics if r['stratum'] == name]
        summarized.append({'stratum': name, 'runs': len(rows), 'mean_rows_per_run': statistics.mean(r['rows'] for r in rows),
                           'runs_with_defined_auroc': sum(r['auroc'] is not None for r in rows),
                           'runs_with_defined_ap': sum(r['ap'] is not None for r in rows),
                           **{f'mean_{k}': statistics.mean(v) if (v := [r[k] for r in rows if r[k] is not None]) else None for k in ('prevalence','auroc','ap','brier')}})
    write_csv(out/'feature_failure_summary.csv', summarized)
    ranking = []
    for method in ('model', 'train_cell_rate', 'constant_tie'):
        rows = [r for r in diagnostics if r.get('method') == method]
        ranking.append({'method': method, 'runs': len(rows), 'mean_eligible_pairs': statistics.mean(r['n_queries'] for r in rows),
                        **{f'mean_{k}': statistics.mean(v) if (v := [r[k] for r in rows if r[k] is not None]) else None for k in ('wq_mrr','wq_auroc','wq_hits1')}})
    write_csv(out/'cell_ranking_summary.csv', ranking)
    hypotheses = {'provider': 'explicit_evidence_templates', 'status': 'untested_followups', 'candidates': [
        {'question': 'Can the model distinguish cell-specific synergy beyond cell response propensity?',
         'evidence': ranking, 'experiment': 'Compare true and train-permuted cell expression with matched architecture; include a training-cell-rate baseline and evaluate pair-conditioned cell ranking.'},
        {'question': 'Can selectively synergistic pairs be identified on sufficiently measured panels?',
         'evidence': feasibility['observed_selective'], 'experiment': 'Use all measured mean-ZIP rows including neutrals. Hold out whole pairs; fit models only on training-pair rows and evaluate both rare-pair PR-AUC versus prevalence and within-pair cell ranking. Require >=10 cells and report sensitivity at 20/30/50. Freeze panel inclusion and split before fitting.',
         'controls': 'Compare pair-level base rate, cell response propensity and constant/tied rankings. Do not treat unmeasured cells as negative or normal-cell toxicity as observed.'},
        {'question': 'Are feature gaps and unstable labels associated with errors?',
         'evidence': summarized, 'experiment': 'Within matched splits, compare missingness-aware expression handling and replicate-weighted or soft targets. Match label prevalence, training frequency and panel coverage before attributing subgroup differences.'}]}
    write(out/'hypotheses.json', hypotheses)
    lines = ['# GraphSmith dataset and feature diagnosis', '', '## Whitening evidence', '',
             '| Setting | Runs | Whiten wq-MRR | Std wq-MRR | Whiten − std |', '|---|---:|---:|---:|---:|']
    for r in whitening['summary']:
        lines.append(f"| {r['setup']} | {r['n']} | {r['mean_whiten_wq_mrr']:.6f} | {r['mean_std_wq_mrr']:.6f} | {r['mean_difference']:+.6f} |")
    lines += ['', whitening['decision'], '', 'These are retrospective historical test results; failure diagnostics below use validation predictions.',
              'EARLY_BEST warnings are retained in the primary comparison and excluded in an additional sensitivity summary in whitening_audit.json; failed or degenerate runs are rejected.',
              '', '## Selective-synergy feasibility', '', feasibility['definition'], '',
              f"Reconstructed {counts['measured_triplets']:,} measured triplets; the canonical task retains {counts['canonical_triplets']:,} and excludes {counts['excluded_neutral_triplets']:,} neutral triplets.",
              f"Pairs with >=10 measured cells: {feasibility['coverage']['10']:,}; observed selective pairs: {feasibility['observed_selective']['10']:,}; Wilson-upper-qualified pairs: {feasibility['selective_with_wilson_upper_le_10pct']:,}.",
              '', feasibility['scope'], feasibility['uncertainty_note'], '', '## Validation feature diagnostics', '',
              '| Stratum | Runs | Mean rows/run | Mean AUROC | Mean Brier |', '|---|---:|---:|---:|---:|']
    fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
    for r in summarized:
        lines.append(f"| {r['stratum']} | {r['runs']} | {r['mean_rows_per_run']:.1f} | {fmt(r['mean_auroc'])} | {fmt(r['mean_brier'])} |")
    lines += ['', 'Equal weight per available fold/seed statistic; missing or single-class strata have undefined AUROC/AP. Different subgroup prevalence/composition confounds attribution.',
              '', '## Ranking cells for a fixed pair', '', '| Method | Runs | Mean eligible pairs | Mean cell-ranking MRR |', '|---|---:|---:|---:|']
    for r in ranking:
        lines.append(f"| {r['method']} | {r['runs']} | {r['mean_eligible_pairs']:.1f} | {fmt(r['mean_wq_mrr'])} |")
    lines += ['', 'These queries fix the drug pair and rank >=3 measured validation cells with both classes. They differ from the original partner-ranking metric and still exclude neutral rows. They are not selective-synergy prediction results.',
              'Cell-rate baseline uses only training labels with a fixed prior weight of 10. Constant ties use the metric’s exact random-tie expectation; random performance is not a fixed 0.5.',
              '', '## Candidate follow-ups', '']
    for h in hypotheses['candidates']:
        lines += [h['question'], '', h['experiment'], '']
    lines += ['Validation was already used for checkpoint selection. All findings are exploratory, with no new independent test scores. Full-panel selectivity labels are retrospective and require a new frozen evaluation design before predictive claims.', '']
    (out/'report.md').write_text('\n'.join(lines))
    write(out/'complete.json', {'outputs': [fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()],
                              'campaign_evidence': [fingerprint(args.campaign/name) for name in ('plan.json', 'state.json')],
                              'adapter': fingerprint(__file__), 'campaign': str(args.campaign.resolve())})
    return out/'report.md'


def add_parser(sub):
    p = sub.add_parser('dpsyn-diagnose', help='Diagnose real-data features and selective-synergy feasibility')
    p.add_argument('--dpsyn-repo', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
