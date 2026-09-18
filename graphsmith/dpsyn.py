"""Read-only real-data reconnaissance using the project's own loader and splits.

This is separate from GraphSmith's synthetic campaign engine. No training, search,
test scoring, or claims of evaluator isolation are made by this adapter.
"""
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
from collections import Counter, defaultdict

from .data import digest


def fingerprint(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def triplet_key(row):
    a, b = sorted((row['drug1_dbid'].strip(), row['drug2_dbid'].strip()))
    return a, b, row['cell_name'].replace('\\', '').strip().upper()


def audit_inputs(paths):
    """Reject ambiguous joins before dpsyn's dictionary-based alignment."""
    import numpy as np
    audit = {}
    for name, index, identifier in [('drug_id_map', 'drug_idx', 'drugbank_id'),
                                    ('prot_id_map', 'protein_idx', 'entrez_id')]:
        rows = table(paths[name])
        indices = [int(r[index]) for r in rows]
        ids = [r[identifier].strip() for r in rows]
        if sorted(indices) != list(range(len(rows))) or len(set(ids)) != len(ids) or '' in ids:
            raise ValueError(f'{name}: invalid indices or duplicate/empty node identifiers')
        audit[name] = {'rows': len(rows), 'duplicate_ids': 0}
    for name in ('fold_csv', 'targets_csv'):
        if not paths[name].exists():
            audit[name] = {'missing': True}
            continue
        rows = table(paths[name])
        keys = [triplet_key(r) for r in rows]
        duplicates = len(keys) - len(set(keys))
        if duplicates:
            raise ValueError(f'{name}: {duplicates} duplicate normalized unordered triplet keys')
        if name == 'fold_csv':
            if not rows or any(float(r['label']) not in (0, 1) for r in rows):
                raise ValueError('fold_csv: empty data or nonbinary labels')
            fold_keys = set(keys)
        audit[name] = {'rows': len(rows), 'duplicate_triplets': 0}
        if name == 'targets_csv':
            audit[name]['fold_keys_matched'] = len(fold_keys & set(keys))
            audit[name]['fold_keys_unmatched'] = len(fold_keys - set(keys))
    for name in ('drug_feat_npz', 'prot_feat_npz'):
        with np.load(paths[name], allow_pickle=True) as z:
            key = next(k for k in ('ids', 'drug_ids', 'protein_ids', 'entrez_ids') if k in z.files)
            ids = [str(x).strip() for x in z[key]]
            if len(ids) != len(set(ids)) or '' in ids:
                raise ValueError(f'{name}: duplicate/empty feature identifiers')
            audit[name] = {'rows': len(ids), 'duplicate_ids': 0}
    return audit


def distribution(values):
    import numpy as np
    a = np.asarray(values)
    if not a.size:
        return {'count': 0}
    return {'count': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)),
            'p90': float(np.quantile(a, .9)), 'max': float(a.max()), 'mean': float(a.mean())}


def validate_dataset(ds):
    import numpy as np
    n = ds.n
    if not n or any(len(getattr(ds, k)) != n for k in ('d1', 'd2', 'cell', 'y')):
        raise ValueError('Empty or misaligned triplet arrays')
    keys = [(min(int(a), int(b)), max(int(a), int(b)), int(c))
            for a, b, c in zip(ds.d1, ds.d2, ds.cell)]
    if len(set(keys)) != n or np.any(ds.d1 == ds.d2):
        raise ValueError('Duplicate/reversed triplets or self pairs')
    if not np.isin(ds.y, [0, 1]).all():
        raise ValueError('Nonbinary targets')
    for array, limit in ((ds.d1, ds.n_drug), (ds.d2, ds.n_drug), (ds.cell, ds.n_cell)):
        if np.any(array < 0) or np.any(array >= limit):
            raise ValueError('Triplet endpoint out of bounds')
    for name, count in [('drug_x', ds.n_drug), ('prot_x', ds.n_prot), ('cell_x', ds.n_cell)]:
        x = getattr(ds, name).numpy()
        if x.ndim != 2 or x.shape[0] != count or not np.isfinite(x).all():
            raise ValueError(f'{name}: nonfinite or misaligned feature matrix')
    diagnostics = {}
    for name, limits in [('e_pd', (ds.n_prot, ds.n_drug)), ('e_pp', (ds.n_prot, ds.n_prot))]:
        edges = getattr(ds, name).numpy()
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(f'{name}: invalid edge shape')
        for side, limit in enumerate(limits):
            if np.any(edges[side] < 0) or np.any(edges[side] >= limit):
                raise ValueError(f'{name}: endpoint out of bounds')
        diagnostics[name] = {'stored_edges': int(edges.shape[1]),
                             'duplicate_directed_edges': int(edges.shape[1] - len(np.unique(edges.T, axis=0)))}
    return diagnostics


def profile_split(ds, train, validation, test, degree):
    import numpy as np
    parts = [np.asarray(x, dtype=np.int64) for x in (train, validation, test)]
    joined = np.concatenate(parts)
    if any(len(p) == 0 for p in parts) or len(set(joined.tolist())) != len(joined):
        raise ValueError('Empty, duplicate, or overlapping split partitions')
    if np.any(joined < 0) or np.any(joined >= ds.n):
        raise ValueError('Split index out of bounds')
    train_drugs = set(ds.d1[train]) | set(ds.d2[train])
    train_cells = set(ds.cell[train])
    out = {'split_hash': digest([p.tolist() for p in parts]),
           'dropped_rows': int(ds.n - len(joined)), 'partitions': {}}
    for name, idx in zip(('train', 'validation', 'test'), parts):
        drugs = np.unique(np.concatenate((ds.d1[idx], ds.d2[idx])))
        cells = np.unique(ds.cell[idx])
        covered = (degree[ds.d1[idx]] > 0).astype(int) + (degree[ds.d2[idx]] > 0)
        cold = np.array([int(a not in train_drugs) + int(b not in train_drugs)
                         for a, b in zip(ds.d1[idx], ds.d2[idx])])
        info = {'rows': len(idx), 'drugs': len(drugs), 'cells': len(cells),
                'pd_covered_drugs': int((degree[drugs] > 0).sum()),
                'rows_by_pd_covered_endpoints': {str(k): int((covered == k).sum()) for k in range(3)},
                'rows_by_training_unseen_endpoints': {str(k): int((cold == k).sum()) for k in range(3)},
                'training_unseen_cells': sum(int(c not in train_cells) for c in cells),
                'rows_without_ccle': int((~ds.cell_has_ccle.numpy()[ds.cell[idx]]).sum()),
                'drugs_without_features': int(ds.drug_feat_missing.numpy()[drugs].sum()),
                'pd_degree': distribution(degree[drugs])}
        if name == 'train':
            labels = ds.y[idx]
            pair_labels, queries = defaultdict(set), defaultdict(list)
            for a, b, c, y in zip(ds.d1[idx], ds.d2[idx], ds.cell[idx], labels):
                pair_labels[tuple(sorted((int(a), int(b))))].add(int(y))
                queries[(int(a), int(c))].append(int(y))
                queries[(int(b), int(c))].append(int(y))
            info['labels'] = {'positive': int(labels.sum()), 'prevalence': float(labels.mean()),
                              'unique_pairs': len(pair_labels),
                              'pairs_with_opposite_labels_across_cells': sum(len(v) > 1 for v in pair_labels.values()),
                              'anchor_cell_query_sizes': distribution([len(v) for v in queries.values()]),
                              'queries_with_both_classes_and_at_least_3_candidates': sum(len(v) >= 3 and 0 < sum(v) < len(v) for v in queries.values())}
        out['partitions'][name] = info
    return out


def hypotheses(profile):
    """Deterministic candidate generation; evidence motivates, never confirms."""
    candidates = []
    first = next(iter(profile['splits']))
    tr = profile['splits'][first]['partitions']['train']
    evidence_base = f'/splits/{first}/partitions/train'
    specs = [
        ('target_coverage', tr['rows_by_pd_covered_endpoints'], evidence_base + '/rows_by_pd_covered_endpoints',
         'The value of protein messages may depend on the number of target-covered drugs in a pair.',
         'Compare DD and DD+PPD on identical splits; stratify validation predictions by zero, one, or two covered drugs.',
         'Match or adjust for training frequency and degree; report every stratum size. Missing targets are not evidence of no biological target.',
         'A coverage-dependent gain that vanishes after frequency matching weakens the target-information explanation.'),
        ('cell_context', tr['labels']['pairs_with_opposite_labels_across_cells'], evidence_base + '/labels/pairs_with_opposite_labels_across_cells',
         'Cell context may resolve pair outcomes that differ across cell lines.',
         'Compare real, training-permuted, and absent cell context, keeping the decoder capacity and all other inputs matched.',
         'Evaluate matched validation pairs and distinguish missing expression from true zero expression; label disagreement may reflect assay noise.',
         'No advantage over permuted context weakens the claim that biological cell context explains the improvement.'),
        ('neighborhood_weighting', tr['pd_degree'], evidence_base + '/pd_degree',
         'Unequal neighborhood sizes may make a fixed protein contribution unsuitable for some drugs.',
         'Compare mean fusion with coverage/degree-conditioned fusion and a parameter-matched gate without those covariates.',
         'Stratify by training frequency and degree; include shuffled protein features. More parameters or identity proxies can explain apparent gains.',
         'An equal gain from a covariate-free matched gate weakens the neighborhood-specific explanation.'),
    ]
    for ident, value, pointer, statement, experiment, control, falsification in specs:
        candidates.append({'id': ident, 'status': 'untested_candidate', 'statement': statement,
                           'evidence': {'profile_pointer': pointer, 'observed': value},
                           'experiment': experiment, 'control': control, 'falsification': falsification,
                           'primary_comparison': 'paired validation wq_mrr; also AUROC/AP and subgroup counts',
                           'selection_policy': 'Preserve dpsyn validation-AUROC checkpoint selection; freeze arms before running.',
                           'limitation': 'Training associations motivate this candidate; they do not establish a mechanism or predictive gain.'})
    return {'provider': 'deterministic_real_data_templates', 'profile_hash': digest(profile),
            'historical_test_metrics_used': False, 'candidates': candidates}


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def inventory(root, config_fields):
    """Retrospective summaries stay within source file and recorded configuration.

    Omitted configuration is unknown, never filled with today's defaults. Means
    describe available unflagged rows, not a verified complete experiment grid.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Results root does not exist: {root}')
    groups, artifacts, errors = [], [], []
    required = ('folds', 'graph', 'drug_feat', 'prot_feat', 'feat_transform', 'channels', 'setup', 'num_folds')
    for path in sorted(set(root.rglob('results.json')) | set(root.glob('sweep_*.csv'))):
        try:
            source = fingerprint(path)
            if path.suffix == '.json':
                raw = json.loads(path.read_text())
                if not isinstance(raw.get('config'), dict) or not isinstance(raw.get('test_metrics'), dict):
                    raise ValueError('Unrecognized results.json schema')
                cfg = raw['config']
                records = [{**cfg, **raw['test_metrics'], 'fold': cfg.get('fold_index'),
                            'leak_check': raw.get('leak_check'), 'error': raw.get('error'),
                            'train_flags': raw.get('training', {}).get('flags'),
                            'degenerate': raw.get('split', {}).get('degenerate')}]
            else:
                records = table(path)
                if records and not {'setup', 'seed', 'fold', 'wq_mrr'} <= records[0].keys():
                    raise ValueError('Unrecognized sweep schema')
            buckets = defaultdict(list)
            for row in records:
                config = {k: row[k] for k in sorted(set(config_fields) | {'arm', 'grid'})
                          if k in row and k not in {'seed', 'fold_index', 'tag', 'output_dir'}}
                buckets[json.dumps(config, sort_keys=True)].append(row)
            for key, rows in buckets.items():
                config = json.loads(key)
                keys = [(str(r.get('fold')), str(r.get('seed'))) for r in rows]
                duplicates = len(keys) - len(set(keys))
                reasons = Counter()
                eligible = []
                for row in rows:
                    issues = []
                    if row.get('error'):
                        issues.append('error')
                    if row.get('leak_check') != 'PASS':
                        issues.append('leak_check_missing_or_failed')
                    if row.get('train_flags') not in (None, '', [], '[]'):
                        issues.append('training_flags')
                    if row.get('degenerate') not in (None, '', False, 'False', 'false', 0, '0'):
                        issues.append('split_degeneracy')
                    if row.get('fold') in (None, '') or row.get('seed') in (None, ''):
                        issues.append('missing_fold_or_seed')
                    if duplicates:
                        issues.append('duplicate_fold_seed_in_group')
                    reasons.update(issues)
                    if not issues:
                        eligible.append(row)
                metrics = {}
                for metric in ('wq_mrr', 'auroc', 'ap'):
                    values = [v for r in eligible if (v := finite(r.get(metric))) is not None]
                    metrics[metric] = {'n': len(values), 'missing_or_nonfinite': len(eligible) - len(values),
                                       'mean': statistics.mean(values) if values else None,
                                       'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
                groups.append({'artifact': str(path), 'config': config, 'rows': len(rows),
                               'observed_fold_seed_keys': [list(k) for k in sorted(set(keys))],
                               'duplicate_keys': duplicates, 'excluded_reason_counts': dict(reasons),
                               'eligible_rows': len(eligible), 'metrics': metrics,
                               'rows_missing_training_diagnostics': sum('train_flags' not in r or r['train_flags'] is None for r in rows),
                               'missing_config_fields': [k for k in required if config.get(k) in (None, '')],
                               'missing_runs': 'Unknown: no original planned grid supplied',
                               'aggregation': 'Equal-weight arithmetic mean over available eligible fold/seed rows within this source and recorded configuration; SD is descriptive, not uncertainty.'})
            artifacts.append({**source, 'rows': len(records)})
        except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    return {'scope': 'retrospective_test_results; excluded from hypothesis generator',
            'root': str(root), 'artifacts': artifacts, 'groups': groups, 'errors': errors,
            'limitations': ['No cross-file pooling or automatic model ranking.',
                           'Old code, graph and feature provenance may be incomplete; values are not revalidated predictions.',
                           'The same experiment may occur in several artifacts; source files are kept separate.',
                           'A PASS leakage check does not establish absence of all leakage.',
                           'Only recursive results.json and top-level sweep_*.csv are recognized.']}


def history_table(path, history):
    fields = ['artifact', 'arm', 'setup', 'folds', 'graph', 'drug_feat', 'prot_feat',
              'feat_transform', 'channels', 'fuse', 'node_feat_mode', 'prot_feat_mode',
              'rows', 'eligible_rows', 'observed_fold_seed_keys', 'excluded_reason_counts',
              'missing_config_fields', 'rows_missing_training_diagnostics',
              'wq_mrr_n', 'wq_mrr_mean', 'wq_mrr_sample_sd', 'auroc_mean', 'ap_mean']
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in history['groups']:
            row = {key: group['config'].get(key) for key in fields if key in group['config']}
            row.update({key: group[key] for key in fields if key in group})
            row.update(wq_mrr_n=group['metrics']['wq_mrr']['n'],
                       wq_mrr_mean=group['metrics']['wq_mrr']['mean'],
                       wq_mrr_sample_sd=group['metrics']['wq_mrr']['sample_sd'],
                       auroc_mean=group['metrics']['auroc']['mean'], ap_mean=group['metrics']['ap']['mean'])
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run(args):
    import numpy as np
    repo = args.dpsyn_repo.resolve()
    if not (repo / 'dpsyn' / 'data.py').is_file():
        raise ValueError(f'Not a dpsyn checkout: {repo}')
    sys.path.insert(0, str(repo))
    C = importlib.import_module('dpsyn.config')
    D = importlib.import_module('dpsyn.data')
    S = importlib.import_module('dpsyn.splits')
    if Path(D.__file__).resolve().parent.parent != repo:
        raise ValueError('A different dpsyn checkout is already imported')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or len(set(args.setups)) != len(args.setups):
        raise ValueError('Seeds and setups must be nonempty and unique')
    if args.num_folds < 3 or any(s not in C.PAPER_SETUPS for s in args.setups):
        raise ValueError('Use at least three folds and supported paper setups')
    cfg = C.RunConfig(folds=args.folds, graph=args.graph or C.DEFAULT_GRAPH,
                      drug_feat=args.drug_feat, prot_feat=args.prot_feat)
    paths = C.resolve(cfg)
    # Refuse before reading data if output already exists; never overwrite evidence.
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {k: fingerprint(p) for k, p in paths.items() if p.exists()}
    code_sources = [fingerprint(p) for p in sorted((repo / 'dpsyn').glob('*.py'))]
    audit = audit_inputs(paths)
    ds = D.build(cfg, verbose=True)  # Avoid stale cache reuse and any writes to shared inputs.
    graph_checks = validate_dataset(ds)
    for name, count in [('drug_id_map', ds.n_drug), ('prot_id_map', ds.n_prot)]:
        if audit[name]['rows'] != count:
            raise ValueError(f'{name}: map size differs from graph node count')
    degree = np.bincount(ds.e_pd[1].numpy(), minlength=ds.n_drug)
    used = np.unique(np.concatenate((ds.d1, ds.d2)))
    profile = {'schema_version': 1, 'task': 'unordered_drug_pair_with_cell_context',
               'configuration': {'folds': cfg.folds, 'graph': cfg.graph, 'drug_feat': cfg.drug_feat,
                                 'prot_feat': cfg.prot_feat, 'num_folds': args.num_folds, 'seeds': args.seeds},
               'scope': {'label_statistics': 'training rows only, separately per split',
                         'topology': 'Full-universe transductive PD/PPI; artifact DD edges are ignored.',
                         'features': 'Raw aligned features before transformations; no fitted preprocessing performed.',
                         'held_out': 'Structural counts only. dpsyn stratified splits may use labels for allocation.',
                         'isolation': 'Read-only analysis convention, not sealed test isolation.',
                         'cross_split': 'Rows can be training in one fold and test in another; this is exploratory reconnaissance, not an untouched confirmation set.'},
               'input_audit': audit, 'graph_checks': graph_checks,
               'universe': {'triplets': ds.n, 'drug_nodes': ds.n_drug, 'protein_nodes': ds.n_prot,
                            'task_drugs': len(used), 'cells': ds.n_cell,
                            'pd_covered_task_drugs': int((degree[used] > 0).sum()),
                            'cells_with_ccle': int(ds.cell_has_ccle.sum()),
                            'task_drugs_without_features': int(ds.drug_feat_missing.numpy()[used].sum()),
                            'proteins_without_features': int(ds.prot_feat_missing.sum()),
                            'rows_with_study_metadata': int((ds.study != '').sum()),
                            'feature_dimensions': {'drug': ds.drug_x.shape[1], 'protein': ds.prot_x.shape[1], 'cell': ds.cell_x.shape[1]},
                            'fold_rows_not_loaded': audit['fold_csv']['rows'] - ds.n},
               'splits': {}}
    for setup in args.setups:
        for seed in args.seeds:
            for fold in range(args.num_folds):
                parts = S.make_split(setup, cid1=ds.d1, cid2=ds.d2, cell=ds.cell, y=ds.y,
                                     num_folds=args.num_folds, fold_index=fold, seed=seed)
                profile['splits'][f'{setup}_f{fold}_s{seed}'] = profile_split(ds, *parts, degree)
    if sources != {k: fingerprint(paths[k]) for k in sources}:
        raise ValueError('Input artifacts changed during profiling')
    if code_sources != [fingerprint(p) for p in sorted((repo / 'dpsyn').glob('*.py'))]:
        raise ValueError('dpsyn sources changed during profiling')
    provenance = {'inputs': sources, 'dpsyn_sources': code_sources,
                  'graphsmith_adapter': fingerprint(__file__), 'python': sys.version,
                  'numpy': np.__version__, 'torch': importlib.import_module('torch').__version__,
                  'sklearn': importlib.import_module('sklearn').__version__}
    save(out / 'profile.json', profile)
    save(out / 'provenance.json', provenance)
    plan = hypotheses(profile)
    save(out / 'hypotheses.json', plan)
    history = inventory(args.results_root, C.RunConfig.__dataclass_fields__) if args.results_root else None
    if history is not None:
        save(out / 'historical_results.json', history)
        history_table(out / 'historical_results.csv', history)
    lines = ['# GraphSmith real-data reconnaissance', '',
             f"Dataset: {cfg.folds}; graph: {cfg.graph}; features: {cfg.drug_feat} / {cfg.prot_feat}.",
             f"{ds.n:,} loaded triplets; {len(used):,} task drugs; {ds.n_cell} cells; {int((degree[used] > 0).sum()):,} task drugs with protein targets.", '',
             'Training-label summaries use the actual dpsyn splits. Held-out partitions have structural counts only.',
             'This is exploratory profiling across folds, not sealed evaluation or independent confirmation.', '',
             '| Split | Train | Validation | Test | Dropped | Train positive fraction |',
             '|---|---:|---:|---:|---:|---:|']
    for key, split in profile['splits'].items():
        p = split['partitions']
        lines.append(f"| {key} | {p['train']['rows']} | {p['validation']['rows']} | {p['test']['rows']} | {split['dropped_rows']} | {p['train']['labels']['prevalence']:.4f} |")
    lines += ['', '## Data-grounded candidates', '', 'Generated by explicit templates, not an LLM or an adaptive search. No candidate is confirmed.', '']
    for h in plan['candidates']:
        lines += [f"### {h['id']}", '', h['statement'], '',
                  f"Observed training evidence: `{h['evidence']['profile_pointer']}` = `{json.dumps(h['evidence']['observed'])}`.", '',
                  h['experiment'], '', f"Control: {h['control']}", '', f"Falsification: {h['falsification']}", '']
    if history is not None:
        excluded = Counter()
        for group in history['groups']:
            excluded.update(group['excluded_reason_counts'])
        lines += ['## Historical results', '',
                  f"Inventoried {len(history['artifacts'])} artifacts and {len(history['groups'])} recorded-configuration groups; {len(history['errors'])} unreadable/unrecognized artifacts.",
                  f"Artifact rows: {sum(a['rows'] for a in history['artifacts']):,}; eligible by available diagnostics: {sum(g['eligible_rows'] for g in history['groups']):,}. These are artifact rows, not deduplicated experiments across files.",
                  f"Exclusion counts (can overlap): `{json.dumps(dict(excluded), sort_keys=True)}`.",
                  'Split flags include LOW_TRAIN_FRAC and DROPPED_ROWS, which can be expected for entity splits. Exclusion is conservative and does not imply that a flagged experiment is scientifically invalid.',
                  'See historical_results.csv for a browsable table and historical_results.json for complete recorded configurations and source hashes.',
                  'Missing planned runs cannot be inferred without the original grid. Historical test metrics were not supplied to candidate generation.', '']
    lines += ['Full evidence: profile.json, hypotheses.json, provenance.json. No training was launched by this command.', '']
    (out / 'report.md').write_text('\n'.join(lines))
    save(out / 'complete.json', {'profile_hash': digest(profile), 'hypotheses_hash': digest(plan),
                               'outputs': [fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()]})
    return out / 'report.md'


def add_parser(sub):
    p = sub.add_parser('dpsyn-profile', help='Profile real drug-synergy data and inventory historical results')
    p.add_argument('--dpsyn-repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--results-root', type=Path)
    p.add_argument('--folds', default='extmap')
    p.add_argument('--graph')
    p.add_argument('--drug-feat', default='atompair_full')
    p.add_argument('--prot-feat', default='esm2_full')
    p.add_argument('--setups', nargs='+', default=['B1', 'LCOW', 'B2', 'B2P', 'B3', 'B3A', 'B4'])
    p.add_argument('--num-folds', type=int, default=4)
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
