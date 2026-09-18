"""Frozen, bounded real-data development campaigns; no final-test evaluation."""
import csv
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from .data import digest
from .dpsyn import fingerprint
from .engine import locked
from .io import read, write


def use_repo(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    C = importlib.import_module('dpsyn.config')
    if Path(C.__file__).resolve().parent.parent != repo:
        raise ValueError('A different dpsyn checkout was imported')
    return C


def freeze(raw, repo, output):
    C = use_repo(repo)
    required = {'name', 'hypothesis', 'minimum_effect', 'common', 'arms', 'seeds',
                'fold_indices', 'max_jobs', 'job_timeout_seconds', 'evidence_profile', 'evidence_pointer'}
    if set(raw) - {'split_contract'} != required:
        raise ValueError(f'Manifest fields must be exactly {sorted(required)}')
    if not isinstance(raw['hypothesis'], str) or not raw['hypothesis'].strip():
        raise ValueError('A hypothesis is required')
    if not isinstance(raw['minimum_effect'], (int, float)) or not 0 < raw['minimum_effect'] < 1:
        raise ValueError('minimum_effect must be in (0, 1)')
    if len(raw['arms']) != 2 or len({a['name'] for a in raw['arms']}) != 2:
        raise ValueError('This pilot backend supports exactly two named comparison arms')
    for values in (raw['seeds'], raw['fold_indices']):
        if not values or len(set(values)) != len(values) or any(type(x) is not int or x < 0 for x in values):
            raise ValueError('Seeds and fold indices must be unique nonnegative integers')
    count = 2 * len(raw['seeds']) * len(raw['fold_indices'])
    if type(raw['max_jobs']) is not int or count != raw['max_jobs'] or count > 24:
        raise ValueError('max_jobs must equal the frozen grid size (at most 24)')
    if type(raw['job_timeout_seconds']) is not int or not 1 <= raw['job_timeout_seconds'] <= 7200:
        raise ValueError('Worker timeout must be 1..7200 seconds')
    evidence = read(raw['evidence_profile'])
    value = evidence
    for key in raw['evidence_pointer'].strip('/').split('/'):
        value = value[key]
    contract = read(raw['split_contract']) if raw.get('split_contract') else None
    jobs, inputs = [], {}
    for fold in raw['fold_indices']:
        for seed in raw['seeds']:
            for arm in raw['arms']:
                if set(arm) != {'name', 'config'} or not arm['name'].replace('_', '').isalnum():
                    raise ValueError('Invalid arm name or fields')
                changes = {**raw['common'], **arm['config']}
                if set(changes) & {'seed', 'fold_index', 'output_dir', 'validation_only', 'floors', 'expected_split_hash'}:
                    raise ValueError('Seeds, folds, output and evaluation policy are campaign-owned')
                cfg = C.RunConfig(**changes, seed=seed, fold_index=fold, validation_only=True, floors=())
                if cfg.feat_transform == 'whiten':
                    raise ValueError('Whitening is disabled for new GraphSmith campaigns; use std. Historical dpsyn reproduction remains available.')
                if cfg.target != 'label' or cfg.setup not in C.PAPER_SETUPS or cfg.counterfactual_oldstop:
                    raise ValueError('Only paper setups, label targets and validation-only selection are supported')
                if not 3 <= cfg.num_folds or fold >= cfg.num_folds or not 1 <= cfg.max_epochs <= 400:
                    raise ValueError('Invalid folds or epoch budget (maximum 400)')
                if contract:
                    for field, expected in contract['shared_config'].items():
                        if cfg.to_dict()[field] != expected:
                            raise ValueError(f'Configuration differs from reference contract: {field}')
                    key = f'{cfg.setup}_f{fold}_s{seed}'
                    cfg.expected_split_hash = contract['splits'][key]['split_hash']
                for path in C.resolve(cfg).values():
                    if path.exists():
                        inputs[str(path.resolve())] = fingerprint(path)
                jobs.append({'id': f"{arm['name']}_f{fold}_s{seed}", 'arm': arm['name'],
                             'fold': fold, 'seed': seed, 'config': cfg.to_dict(), 'status': 'pending'})
    if contract:
        if inputs != {item['path']: item for item in contract['inputs']}:
            raise ValueError('Resolved data artifacts differ from the split contract')
        for item in contract['inputs'] + contract['references'] + [contract['split_arrays']]:
            if fingerprint(item['path']) != item:
                raise ValueError(f"Reference contract evidence changed: {item['path']}")
            inputs[item['path']] = item
        item = fingerprint(raw['split_contract'])
        inputs[item['path']] = item
    sources = [fingerprint(p) for parent in (Path(repo) / 'dpsyn', Path(__file__).parent)
               for p in sorted(parent.glob('*.py'))]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'manifest': raw, 'dpsyn_repo': str(Path(repo).resolve()), 'jobs': jobs,
            'evidence': {'file': fingerprint(raw['evidence_profile']), 'pointer': raw['evidence_pointer'], 'value': value},
            'inputs': list(inputs.values()), 'sources': sources, 'python': sys.version,
            'versions': {name: importlib.import_module(name).__version__ for name in ('numpy', 'torch', 'sklearn')},
            'scope': 'Exploratory validation, not sealed isolation; no historical test results used.',
            'budget': {'max_jobs': count, 'max_worker_seconds': count * raw['job_timeout_seconds']}}
    write(output / 'plan.json', plan)
    write(output / 'state.json', {'plan_hash': digest(plan), 'jobs': jobs, 'status': 'frozen'})
    return output


def verify(root):
    plan, state = read(root / 'plan.json'), read(root / 'state.json')
    if digest(plan) != state['plan_hash']:
        raise ValueError('Frozen plan changed')
    for item in plan['inputs'] + plan['sources'] + [plan['evidence']['file']]:
        if fingerprint(item['path']) != item:
            raise ValueError(f"Frozen input/source changed: {item['path']}")
    return plan, state


def worker(root, job_id):
    plan, _ = verify(root)
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    C = use_repo(plan['dpsyn_repo'])
    cfg = C.RunConfig(**{**job['config'], 'output_dir': str(root / 'jobs' / job_id)})
    from dpsyn.run import run
    result = run(cfg)
    if 'NEVER_SELECTED' in result['training']['flags']:
        raise ValueError('No checkpoint selected')


def execute(root):
    with locked(root):
        plan, state = verify(root)
        if state['status'] != 'frozen':
            raise ValueError('Campaign already attempted; no implicit retry or resume')
        timeout = plan['manifest']['job_timeout_seconds']
        state['status'] = 'running'
        write(root / 'state.json', state)
        for job in state['jobs']:
            verify(root)
            folder = root / 'jobs' / job['id']
            folder.mkdir(parents=True)
            job.update(status='running', reserved_seconds=timeout)
            write(root / 'state.json', state)
            start = time.monotonic()
            try:
                with (folder / 'worker.log').open('w') as log:
                    proc = subprocess.Popen([sys.executable, '-m', 'graphsmith.dpsyn_campaign',
                                             'worker', str(root), job['id']], stdout=log, stderr=subprocess.STDOUT,
                                            cwd=Path(__file__).parent.parent, start_new_session=True)
                    try:
                        code = proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        raise ValueError('Worker timed out')
                if code:
                    raise ValueError(f'Worker exited {code}; inspect worker.log')
                result = read(folder / 'validation_results.json')
                if result.get('evaluation_split') != 'validation' or 'test_metrics' in result:
                    raise ValueError('Worker returned an invalid evaluation scope')
                job.update(status='completed', result=fingerprint(folder / 'validation_results.json'),
                           predictions=fingerprint(folder / 'validation_predictions.csv'))
            except (ValueError, OSError) as exc:
                job.update(status='failed', error=str(exc))
            job['elapsed_seconds'] = time.monotonic() - start
            write(root / 'state.json', state)
        verify(root)
        state['status'] = 'completed' if all(j['status'] == 'completed' for j in state['jobs']) else 'incomplete'
        write(root / 'state.json', state)
        return report(root)


def prediction_rows(root, job):
    for key in ('result', 'predictions'):
        if fingerprint(job[key]['path']) != job[key]:
            raise ValueError('Recorded evidence changed')
    result = read(job['result']['path'])
    if result['predictions_sha256'] != job['predictions']['sha256']:
        raise ValueError('Prediction hash disagrees with result')
    with Path(job['predictions']['path']).open() as stream:
        rows = list(csv.DictReader(stream))
    keys = [tuple(sorted((r['drug1'], r['drug2']))) + (r['cell'],) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate contextual pair in predictions')
    return result, {k: row for k, row in zip(keys, rows)}


def compare_rows(a, b, min_candidates):
    import numpy as np
    from dpsyn import metrics as M
    from dpsyn.development import clean
    if set(a) != set(b):
        raise ValueError('Comparison predictions do not cover identical contextual pairs')
    keys = sorted(a)
    for key in keys:
        if any(a[key][field] != b[key][field] for field in ('label', 'covered_endpoints',
               'drug1_pd_degree', 'drug2_pd_degree', 'drug1_train_frequency', 'drug2_train_frequency')):
            raise ValueError('Comparison labels or structural covariates disagree')
    summaries = []
    drugs = {name: i for i, name in enumerate(sorted({x for k in keys for x in k[:2]}))}
    cells = {name: i for i, name in enumerate(sorted({k[2] for k in keys}))}
    for group in ('all', '0', '1', '2'):
        selected = [k for k in keys if group == 'all' or a[k]['covered_endpoints'] == group]
        entry = {'coverage_group': group, 'rows': len(selected)}
        for name, rows in [('baseline', a), ('intervention', b)]:
            if not selected:
                entry[name] = {}
                continue
            y = np.array([int(rows[k]['label']) for k in selected])
            scores = np.array([float(rows[k]['probability']) for k in selected])
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError('Invalid probability')
            entry[name] = clean(M.evaluate(y, scores, test_idx=np.arange(len(y)),
                cell=np.array([cells[k[2]] for k in selected]), d1=np.array([drugs[k[0]] for k in selected]),
                d2=np.array([drugs[k[1]] for k in selected]), min_candidates=min_candidates, prevalence=float(y.mean())))
        entry['delta_wq_mrr'] = (entry['intervention'].get('wq_mrr') - entry['baseline'].get('wq_mrr')
            if entry['baseline'].get('wq_mrr') is not None and entry['intervention'].get('wq_mrr') is not None else None)
        summaries.append(entry)
    return summaries


def report(root):
    plan, state = verify(root)
    use_repo(plan['dpsyn_repo'])
    comparisons = []
    names = [a['name'] for a in plan['manifest']['arms']]
    for fold in plan['manifest']['fold_indices']:
        for seed in plan['manifest']['seeds']:
            jobs = [next(j for j in state['jobs'] if j['fold'] == fold and j['seed'] == seed and j['arm'] == a) for a in names]
            if any(j['status'] != 'completed' for j in jobs):
                continue
            ra, a = prediction_rows(root, jobs[0])
            rb, b = prediction_rows(root, jobs[1])
            if ra['split']['split_hash'] != rb['split']['split_hash']:
                raise ValueError('Comparison uses different splits')
            groups = compare_rows(a, b, ra['config']['min_candidates'])
            for result, name in ((ra, 'baseline'), (rb, 'intervention')):
                for metric in ('wq_mrr', 'auroc', 'ap'):
                    expected, actual = result['validation_metrics'][metric], groups[0][name][metric]
                    if expected != actual and (expected is None or actual is None or abs(expected - actual) > 1e-10):
                        raise ValueError(f'Metric {metric} does not reproduce from predictions')
            comparisons.append({'fold': fold, 'seed': seed, 'groups': groups,
                'baseline_parameters': ra['training']['parameter_count'],
                'intervention_parameters': rb['training']['parameter_count'],
                'baseline_training': ra['training'], 'intervention_training': rb['training'],
                'split': ra['split']})
    differences = [p['groups'][0]['delta_wq_mrr'] for p in comparisons
                   if p['groups'][0]['delta_wq_mrr'] is not None]
    aggregate = {'observed_pairs': len(differences),
                 'expected_pairs': len(plan['manifest']['fold_indices']) * len(plan['manifest']['seeds']),
                 'mean_delta_wq_mrr': statistics.mean(differences) if differences else None,
                 'sample_sd_delta_wq_mrr': statistics.stdev(differences) if len(differences) > 1 else None,
                 'all_parameter_counts_equal': bool(comparisons) and all(p['baseline_parameters'] == p['intervention_parameters'] for p in comparisons),
                 'aggregation': 'Equal weight per matched fold/seed pair; descriptive SD, not independent uncertainty.'}
    summary = {'aggregate': aggregate, 'hypothesis': plan['manifest']['hypothesis'], 'minimum_effect': plan['manifest']['minimum_effect'],
               'status': state['status'], 'comparisons': comparisons,
               'verdict': 'inconclusive_exploratory_pilot',
               'limitations': ['Validation selected checkpoints and is reused for comparison; no test scores produced.',
                   'Inspect parameter counts: unequal counts confound attribution; equal counts alone do not establish a biological mechanism.',
                   'Coverage subgroups rebuild ranking queries within each subgroup; candidate pools and MRR floors differ.',
                   'Training-frequency covariates are exported but this pilot does not adjust or match on them.',
                   'No random-row leakage control or tabular baseline was run; structural split checks alone are insufficient.',
                   'Different folds are not independent graph draws. No confirmatory claim is made.']}
    write(root / 'summary.json', summary)
    lines = ['# GraphSmith validation pilot', '', summary['hypothesis'], '',
             f"Status: {state['status']}; verdict: {summary['verdict']}.", '',
             f"Paired summary: {aggregate}", '',
             '| Fold | Seed | Coverage | Rows | Baseline wq-MRR | Intervention wq-MRR | Difference |',
             '|---|---|---|---:|---:|---:|---:|']
    fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
    for index, pair in enumerate(comparisons):
        if index:
            lines += ['| Fold | Seed | Coverage | Rows | Baseline wq-MRR | Intervention wq-MRR | Difference |', '|---|---|---|---:|---:|---:|---:|']
        for g in pair['groups']:
            lines.append(f"| {pair['fold']} | {pair['seed']} | {g['coverage_group']} | {g['rows']} | {fmt(g['baseline'].get('wq_mrr'))} | {fmt(g['intervention'].get('wq_mrr'))} | {fmt(g['delta_wq_mrr'])} |")
        lines += ['', f"Parameters: {pair['baseline_parameters']:,} baseline; {pair['intervention_parameters']:,} intervention.",
                  f"Training flags: {pair['baseline_training']['flags']} baseline; {pair['intervention_training']['flags']} intervention.", '']
    lines += ['', *summary['limitations'], '', 'Source manifest, input/code fingerprints and observed evidence are in plan.json; job outcomes are in state.json.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return root / 'report.md'


def add_parser(sub):
    p = sub.add_parser('dpsyn-campaign', help='Run a frozen validation-only real-data comparison')
    p.add_argument('manifest', type=Path)
    p.add_argument('--dpsyn-repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('dpsyn-campaign-report', help='Recompute a validation campaign report from predictions')
    p.add_argument('run', type=Path)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != 'worker':
        raise SystemExit('Internal entry: worker CAMPAIGN JOB_ID')
    root = Path(sys.argv[2]).resolve()
    plan = read(root / 'plan.json')
    signal.alarm(plan['manifest']['job_timeout_seconds'])
    worker(root, sys.argv[3])
