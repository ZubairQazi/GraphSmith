"""Real-data, validation-only selective-synergy feature campaign.

Expands neutral rows while preserving frozen B3 drug folds; never reuses integer
row indices on the expanded universe. Explicit bounded hypotheses, not LLM search.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import platform
from .dpsyn import fingerprint, triplet_key
from .dpsyn_diagnose import reconstruct, write_csv
from .io import read, write

ARMS = ('full_binary', 'extreme_binary', 'zip_regression', 'permuted_cell', 'within_pair_ranking')


def expanded_assignment(original_drugs, d1, d2, fold, seed):
    import numpy as np
    from dpsyn.splits import _entity_folds
    drugs = np.unique(original_drugs)
    if not np.isin(np.r_[d1, d2], drugs).all():
        raise ValueError('Expanded rows contain drugs outside frozen split universe')
    folds = _entity_folds(len(drugs), 4, seed)
    test = np.isin(d1, drugs[folds[fold]]) | np.isin(d2, drugs[folds[fold]])
    val = (np.isin(d1, drugs[folds[(fold+1)%4]]) | np.isin(d2, drugs[folds[(fold+1)%4]])) & ~test
    return np.where(test, 2, np.where(val, 1, 0)).astype(np.int8)


def selective_probability(probabilities):
    import numpy as np
    """P(1 <= positive count <= floor(n/10)), conditional independence baseline."""
    n = len(probabilities)
    limit = n//10
    if not limit:
        return 0.
    mass = np.zeros(limit+1); mass[0] = 1.
    for p in probabilities:
        previous = mass.copy()
        mass *= 1-p
        mass[1:] += previous[:-1]*p
    return float(mass[1:].sum())


def binary_metrics(y, p):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return {'n': 0, 'prevalence': None, 'ap': None, 'auroc': None, 'brier': None}
    return {'n': len(y), 'prevalence': float(y.mean()),
        'ap': float(average_precision_score(y, p)) if y.sum() else None,
        'auroc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'brier': float(np.mean((y-p)**2))}


def expected_recall(y, scores, k):
    import numpy as np
    """Expected recall under random tie breaking at the selection boundary."""
    y, scores = np.asarray(y), np.asarray(scores)
    if not y.sum():
        return None
    threshold = np.sort(scores)[-k]
    above, tied = scores > threshold, scores == threshold
    hits = y[above].sum() + (k-above.sum())*y[tied].mean()
    return float(hits/y.sum())


def evaluate(rows, y, probability, min_cells):
    import numpy as np
    from sklearn.metrics import average_precision_score
    from dpsyn.metrics import within_query
    from dpsyn.development import clean
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(row[:2])].append(i)
    panels = [np.array(ids) for ids in groups.values() if len(ids) >= min_cells]
    selective = [ids for ids in panels if 0 < y[ids].sum() <= .1*len(ids)]
    pair_y = np.array([0 < y[ix].sum() <= .1*len(ix) for ix in panels], int)
    pair_p = np.array([selective_probability(probability[ix]) for ix in panels])
    recalls = [expected_recall(y[ix], probability[ix], int(np.ceil(.1*len(ix)))) for ix in selective]
    aps = [average_precision_score(y[ix], probability[ix]) for ix in selective]
    return {'triplets': binary_metrics(y, probability), 'pair_selectivity': binary_metrics(pair_y, pair_p),
        'selective_cell_ranking': clean(within_query(selective, y, probability)),
        'selective_pairs': len(selective), 'selective_macro_ap': float(np.mean(aps)) if aps else None,
        'selective_recall_at_10pct': float(np.mean(recalls)) if recalls else None}


def run(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import sklearn
    sys.path.insert(0, str(args.dpsyn_repo.resolve()))
    from dpsyn import config as C, data as D, splits as S, selectivity as F
    import torch
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    contract = read(args.contract)
    if fingerprint(contract['split_arrays']['path']) != contract['split_arrays']:
        raise ValueError('Frozen split arrays changed')
    for item in contract['inputs']:
        if fingerprint(item['path']) != item:
            raise ValueError('Inputs changed since frozen B3 campaign')
    cfg = C.RunConfig(**contract['shared_config'])
    if cfg.feat_transform != 'std':
        raise ValueError('This campaign requires standardization')
    ds = D.build(cfg, verbose=False)
    measured, canonical, counts, sources = reconstruct(C.DATA_ROOT, C.GRAPHS[cfg.graph])
    original_keys = [triplet_key(r) for r in canonical]
    if original_keys != list(zip(ds.d1_dbid, ds.d2_dbid, ds.cell_name)):
        raise ValueError('Canonical-to-model row order mismatch')
    drugs = dict(zip(ds.d1_dbid, ds.d1)); drugs.update(zip(ds.d2_dbid, ds.d2))
    cell_names = ds.cell_names + sorted({k[2] for k in measured}-set(ds.cell_names))
    cells = {name: i for i, name in enumerate(cell_names)}
    cell_values, cell_has, tissue_ids, tissue_names = D._load_cells(cell_names, C.resolve(cfg))
    if not np.array_equal(cell_values[:ds.n_cell], ds.cell_x.numpy()) or not np.array_equal(cell_has[:ds.n_cell], ds.cell_has_ccle.numpy()):
        raise ValueError('Original cell features changed during panel expansion')
    n_cell = len(cell_names)
    excluded_drugs = sorted({d for key in measured for d in key[:2] if d not in drugs})
    keys = sorted(k for k in measured if k[0] in drugs and k[1] in drugs and k[2] in cells)
    # Exclude whole pairs with any unsupported cell; never silently shorten a panel denominator.
    excluded_cell_pairs = {k[:2] for k in measured if k[2] not in cells}
    keys = [k for k in keys if k[:2] not in excluded_cell_pairs]
    lookup = {key: i for i, key in enumerate(keys)}
    if not set(original_keys) <= set(keys):
        raise ValueError('Expanded task lost canonical rows')
    original_positions = np.array([lookup[k] for k in original_keys])
    rows = np.array([(drugs[a], drugs[b], cells[c]) for a, b, c in keys], np.int64)
    zip_scores = np.array([measured[k]['mean_zip'] for k in keys])
    y = (zip_scores > 10).astype(int)
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key[:2]].append(i)
    pair_keys = sorted(groups)
    pair_indices = [np.array(groups[k]) for k in pair_keys]
    panel_n = np.array([len(ix) for ix in pair_indices])
    pair_y = np.array([0 < y[ix].sum() <= .1*len(ix) for ix in pair_indices], int)
    pair_rows = np.array([rows[ix[0]] for ix in pair_indices]); pair_rows[:, 2] = 0
    panel_design = np.zeros((len(pair_keys), n_cell+1), np.float32)
    for i, ix in enumerate(pair_indices):
        panel_design[i, rows[ix, 2]] = 1
        panel_design[i, -1] = len(ix)/100
    original_drugs = np.unique(np.r_[ds.d1, ds.d2])
    frozen = np.load(contract['split_arrays']['path'])
    splits, split_checks = {}, []
    for seed in range(3):
        for fold in range(4):
            key = f'B3_f{fold}_s{seed}'
            assignment = expanded_assignment(original_drugs, rows[:, 0], rows[:, 1], fold, seed)
            old = np.full(ds.n, -1)
            for partition, name in enumerate(('train', 'validation', 'test')):
                old[frozen[key+'_'+name]] = partition
                splits[key+'_'+name] = np.flatnonzero(assignment == partition)
            if not np.array_equal(assignment[original_positions], old) or (old < 0).any():
                raise ValueError('Original contextual rows changed partition')
            if any(len(np.unique(assignment[ix])) != 1 for ix in pair_indices):
                raise ValueError('Pair spans partitions')
            native = S.split_B3(ds.d1, ds.d2, 4, fold, seed)
            for part, name in zip(native, ('train', 'validation', 'test')):
                if not np.array_equal(part, frozen[key+'_'+name]):
                    raise ValueError('Current native split differs from frozen arrays')
            split_checks.append({'fold': fold, 'seed': seed, 'original_rows_preserved': ds.n,
                'train_rows': int((assignment == 0).sum()), 'validation_rows': int((assignment == 1).sum()),
                'test_rows_reserved': int((assignment == 2).sum()), 'pair_overlap': 0})
    np.savez_compressed(out/'split_indices.npz', **splits)
    write_csv(out/'row_identities.csv', [dict(row_id=i, drug1=k[0], drug2=k[1], cell=k[2]) for i, k in enumerate(keys)])
    write_csv(out/'pair_identities.csv', [dict(pair_id=i, drug1=k[0], drug2=k[1], measured_cells=int(panel_n[i])) for i, k in enumerate(pair_keys)])
    source_paths = [Path(__file__), Path(F.__file__), Path(S.__file__), Path(D.__file__), Path(C.__file__), args.contract,
                    Path(sys.modules[reconstruct.__module__].__file__),
                    Path(D.__file__).with_name('metrics.py'), Path(D.__file__).with_name('development.py'),
                    Path(__file__).with_name('dpsyn.py'), Path(__file__).with_name('io.py')]
    source_records = [fingerprint(p) for p in source_paths]
    snapshot = out/'source'; snapshot.mkdir()
    for p in source_paths:
        (snapshot/p.name).write_bytes(p.read_bytes())
    plan = {'name': 'selective-synergy-feature-pilot', 'hypotheses': [
        'Restoring neutral measurements improves selective-cell localization.',
        'Real cell features help beyond cell propensity and arbitrary cell identity.',
        'Continuous ZIP or within-pair ranking supervision improves localization.',
        'Molecular pair features predict observed selectivity beyond panel design and drug label rates.'],
        'data': counts, 'expanded_rows': len(rows), 'excluded_rows': len(measured)-len(rows),
        'excluded_new_drugs': excluded_drugs, 'cells': n_cell, 'original_cells': ds.n_cell,
        'cells_with_expression': int(cell_has.sum()), 'excluded_unsupported_cell_pairs': len(excluded_cell_pairs),
        'panel_counts': {str(n): {'pairs': int((panel_n >= n).sum()), 'selective': int(pair_y[panel_n >= n].sum())} for n in (10, 50)},
        'arms': list(ARMS)+['pair_molecular'], 'seeds': [0,1,2], 'folds': [0,1,2,3],
        'max_fits': 72, 'max_epochs': 120, 'patience': 20, 'warmup': 10,
        'optimizer': 'AdamW lr=.001 weight_decay=.0001 gradient_clip=5',
        'features': 'Actual AtomPair-full and CCLE expression+tissue; train-entity-fitted std; no whitening or graph message passing.',
        'threshold': 'mean ZIP > 10; selective iff 1..floor(.1*n) positives with n>=10 measured cells',
        'primary': 'Equal-weight fold/seed mean selective-pair macro AP for responsive-cell localization; direct pair-level AP separately.',
        'secondary': 'Within-pair MRR/AUROC; recall at ceil(.1*n); pair-level AP/AUROC/Brier; >=50-cell sensitivity.',
        'split_checks': split_checks, 'sources': sources+contract['inputs']+source_records,
        'split_arrays': fingerprint(out/'split_indices.npz'), 'row_identities': fingerprint(out/'row_identities.csv'),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__, 'device': args.device},
        'limitations': ['Exploratory validation selection and evaluation; no test scoring, loader can read all source labels.',
            'New neutral-inclusive task; prior benchmark scores are not directly comparable.',
            'Cell permutation is fixed across train/validation within a run, retaining arbitrary identity; it tests biological alignment, not absence of cell identity.',
            'Aggregated triplet probabilities assume conditional independence; direct pair classifier is a separate endpoint.',
            'Expanded drugs outside the original task universe are excluded to preserve B3 assignments.',
            'Cancer cell panels are nonrandom; observed selectivity does not establish normal-cell safety.']}
    write(out/'plan.json', plan)
    print(f'Frozen {len(rows)} rows; all 12 original B3 assignments preserved; starting 72 bounded fits', flush=True)
    drug_x = ds.drug_x.numpy()
    tissue = np.eye(len(tissue_names), dtype=np.float32)[tissue_ids]
    cell_x = np.c_[cell_values, tissue, cell_has.astype(np.float32)]
    if not np.isfinite(drug_x).all() or not np.isfinite(cell_x).all():
        raise ValueError('Nonfinite input features')
    results, state = [], {'completed': [], 'expected_fits': 72, 'plan': fingerprint(out/'plan.json')}
    write(out/'state.json', state)
    for seed in range(3):
        for fold in range(4):
            key = f'B3_f{fold}_s{seed}'
            tr, va = splits[key+'_train'], splits[key+'_validation']
            tr_set, va_set = set(tr), set(va)
            ptr = np.array([i for i, ix in enumerate(pair_indices) if ix[0] in tr_set and len(ix) >= 10])
            pva = np.array([i for i, ix in enumerate(pair_indices) if ix[0] in va_set and len(ix) >= 10])
            if len(np.unique(y[tr])) < 2 or len(np.unique(pair_y[ptr])) < 2:
                raise ValueError('Training task has only one class')
            prior = y[tr].mean()
            n = np.bincount(rows[tr, 2], minlength=n_cell)
            positive = np.bincount(rows[tr, 2], weights=y[tr], minlength=n_cell)
            cell_rate = (positive+10*prior)/(n+10)
            scores = {'train_cell_rate': cell_rate[rows[va, 2]], 'constant': np.full(len(va), prior)}
            for arm in ARMS:
                train = tr[np.abs(zip_scores[tr]) > 10] if arm == 'extreme_binary' else tr
                kind = 'regression' if arm == 'zip_regression' else 'ranking' if arm == 'within_pair_ranking' else 'binary'
                context = cell_x[np.random.RandomState(seed).permutation(n_cell)] if arm == 'permuted_cell' else cell_x
                target = zip_scores/10 if kind == 'regression' else y
                state['active'] = {'fold': fold, 'seed': seed, 'arm': arm}; write(out/'state.json', state)
                probability, raw, training = F.fit(drug_x, context, rows[train], target[train], rows[va], target[va], seed=seed, kind=kind, device=args.device)
                scores[arm] = probability
                name = key+'_'+arm
                np.savez_compressed(out/(name+'.npz'), row_id=va, probability=probability, raw=raw)
                training.update(train_rows=len(train), validation_rows=len(va))
                write(out/(name+'.json'), training)
                state['completed'].append(name); write(out/'state.json', state)
                print(f'{name}: selected {training["best_epoch"]}/{training["epochs"]}', flush=True)
            for arm, probability in scores.items():
                if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                    raise ValueError('Invalid probabilities')
                for minimum in (10, 50):
                    results.append({'fold': fold, 'seed': seed, 'arm': arm, 'min_cells': minimum,
                                    **evaluate(rows[va], y[va], probability, minimum)})
            np.savez_compressed(out/(key+'_controls.npz'), row_id=va, train_cell_rate=scores['train_cell_rate'], constant=scores['constant'])
            state['active'] = {'fold': fold, 'seed': seed, 'arm': 'pair_molecular'}; write(out/'state.json', state)
            pp, raw, training = F.fit(drug_x, np.zeros((1, 1)), pair_rows[ptr], pair_y[ptr], pair_rows[pva], pair_y[pva], seed=seed, pair_only=True, device=args.device)
            name = key+'_pair_molecular'
            write(out/(name+'.json'), training)
            pair_prior = pair_y[ptr].mean()
            rate_n, rate_y = defaultdict(int), defaultdict(int)
            for i in ptr:
                for d in pair_rows[i, :2]:
                    rate_n[int(d)] += 1; rate_y[int(d)] += int(pair_y[i])
            identity = np.array([np.mean([(rate_y[int(d)]+10*pair_prior)/(rate_n[int(d)]+10) for d in pair_rows[i, :2]]) for i in pva])
            design = LogisticRegression(C=1., max_iter=1000, random_state=seed).fit(panel_design[ptr], pair_y[ptr])
            pair_scores = {'pair_molecular': pp, 'pair_constant': np.full(len(pva), pair_prior),
                'pair_drug_rate': identity, 'pair_panel_design': design.predict_proba(panel_design[pva])[:, 1]}
            np.savez_compressed(out/(name+'.npz'), pair_id=pva, **pair_scores)
            for arm, score in pair_scores.items():
                for minimum in (10, 50):
                    mask = panel_n[pva] >= minimum
                    results.append({'fold': fold, 'seed': seed, 'arm': arm, 'min_cells': minimum,
                        'pair_selectivity': binary_metrics(pair_y[pva][mask], score[mask])})
            state['completed'].append(name); state['active'] = None; write(out/'state.json', state)
            write(out/'metrics.json', results)
            print(f'{key}: validation complete ({len(ptr)} train / {len(pva)} validation eligible pairs)', flush=True)
    if len(state['completed']) != 72 or len(set(state['completed'])) != 72:
        raise ValueError('Incomplete/duplicate fits')
    for source in plan['sources']:
        if fingerprint(source['path']) != source:
            raise ValueError('An input/source changed during the campaign')
    summaries = []
    for arm in sorted({r['arm'] for r in results}):
        for minimum in (10, 50):
            records = [r for r in results if r['arm'] == arm and r['min_cells'] == minimum]
            if len(records) != 12:
                raise ValueError('Missing runs in aggregation')
            record = {'arm': arm, 'min_cells': minimum, 'runs': len(records)}
            for metric in ('ap', 'auroc', 'brier', 'prevalence'):
                values = [r['pair_selectivity'][metric] for r in records]
                if any(v is None for v in values):
                    raise ValueError('Undefined pair-level metric')
                record['pair_'+metric] = float(np.mean(values))
            if 'selective_macro_ap' in records[0]:
                for metric in ('selective_macro_ap', 'selective_recall_at_10pct', 'selective_pairs'):
                    record[metric] = float(np.mean([r[metric] for r in records]))
            summaries.append(record)
    write(out/'summary.json', summaries)
    write(out/'complete.json', {'fits': 72, 'test_scoring': False,
        'files': [fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()]})
    return str(out)


def add_parser(sub):
    p = sub.add_parser('dpsyn-selectivity', help='Frozen B3 neutral-inclusive selective-synergy feature pilot')
    p.add_argument('--dpsyn-repo', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
