"""Persistent project evidence, separate from training and the synthetic planner.

Standard-library only. Historical scores are retrospective development evidence.
Never infers equivalence of protocols from filenames or fills missing metadata.
"""
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
from collections import Counter, defaultdict

CONFIG = ('grid target folds graph dd_key dd_attr dd_sign channels node_feat_mode '
          'drug_feat prot_feat missing_feat feat_transform feat_dim max_epochs '
          'pair_pool cell_cond side_blocks side_mode fuse conv conv_root cell_feat '
          'cell_transform rewire drop_edges prot_feat_mode random_dim min_epochs '
          'cp_topk cp_zero_cells num_folds molecular_encoder feature_protocol').split()
METRICS = ('auroc', 'ap', 'wq_mrr', 'wq_hits1')
SETUPS = ('B1', 'LCOW', 'B2', 'B2P', 'B3', 'B3A', 'B4')


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return {'path': str(Path(path).resolve()), 'sha256': h.hexdigest(),
            'bytes': Path(path).stat().st_size}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def hard_failure(row):
    reasons = []
    for key in ('error',):
        if str(row.get(key, '')).strip().lower() not in ('', '0', 'false', 'none'):
            reasons.append(key)
    split_flags = re.sub(r'(LOW_TRAIN_FRAC|DROPPED_ROWS|TINY_TRAIN)\([^)]*\)', '', str(row.get('degenerate', ''))).strip()
    if split_flags.lower() not in ('', '0', 'false', 'none'):
        reasons.append('degenerate')
    if row.get('leak_check', '').strip().lower() not in ('', 'pass', 'true'):
        reasons.append('leak_check')
    if row.get('group_disjoint', '').strip().lower() not in ('', 'pass', 'true'):
        reasons.append('group_disjoint')
    return reasons


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sd': statistics.stdev(values) if len(values) > 1 else None}


def key(row):
    return row.get('fold', ''), row.get('seed', '')


def paired(rows, left, right, setup, metric='wq_mrr', right_metric=None,
           expected_keys=None):
    """Pair by fold/seed, reject duplicate keys and recorded split mismatches."""
    a, b = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get('setup') != setup:
            continue
        if r.get('arm') == left:
            a[key(r)].append(r)
        if r.get('arm') == right:
            b[key(r)].append(r)
    deltas, clean, exclusions, records = [], [], [], []
    differences = defaultdict(set)
    for k in sorted(set(a) | set(b)):
        reason = None
        if len(a[k]) != 1 or len(b[k]) != 1:
            reason = 'missing_or_duplicate_key'
        else:
            x, y = a[k][0], b[k][0]
            vx, vy = number(x.get(metric)), number(y.get(right_metric or metric))
            if hard_failure(x) or hard_failure(y):
                reason = 'hard_failure'
            elif vx is None or vy is None:
                reason = 'missing_metric'
            elif any(x.get(f) and y.get(f) and x[f] != y[f]
                     for f in ('split_sha256', 'n_train', 'n_val', 'n_test')):
                reason = 'recorded_split_mismatch'
            else:
                for f in CONFIG:
                    if x.get(f) != y.get(f):
                        differences[f].add((x.get(f), y.get(f)))
                delta = vx-vy
                deltas.append(delta)
                flags = [r.get('train_flags', '') for r in (x, y) if r.get('train_flags')]
                if not flags:
                    clean.append(delta)
                records.append({'fold': k[0], 'seed': k[1], 'left': vx, 'right': vy,
                                'delta': delta, 'train_flags': flags,
                                'split_warnings': [z.get('degenerate', '') for z in (x,y)],
                                'split_hash_verified': bool(x.get('split_sha256')),
                                'source_rows': [{'path': z.get('_source'), 'line': z['_source_row']} for z in (x,y)]})
        if reason:
            exclusions.append({'fold': k[0], 'seed': k[1], 'reason': reason})
    observed = set(a) | set(b)
    return {'setup': setup, 'left': left, 'right': right, 'metric': metric,
            'right_metric': right_metric or metric, 'delta': stats(deltas),
            'unflagged_delta': stats(clean), 'positive': sum(d > 0 for d in deltas),
            'expected_keys': expected_keys,
            'absent_from_both': [list(k) for k in (expected_keys or []) if tuple(k) not in observed],
            'unexpected_keys': [list(k) for k in observed if expected_keys is not None and list(k) not in expected_keys],
            'config_differences': {f: sorted(v, key=str) for f, v in differences.items()},
            'exclusions': exclusions, 'pairs': records,
            'interpretation': 'Descriptive paired association; inspect configuration differences and mechanism confounds.'}


def inventory(roots, output):
    catalog, groups, by_source = [], [], {}
    seen = set()
    with (output/'runs.jsonl').open('w') as raw:
        for root in roots:
            for path in sorted(Path(root).rglob('*.csv')):
                path = path.resolve()
                if path in seen or output == path or output in path.parents:
                    continue
                seen.add(path)
                before = fingerprint(path)
                entry = dict(before)
                try:
                    with path.open(newline='') as f:
                        reader = csv.DictReader(f)
                        fields = reader.fieldnames or []
                        entry['columns'] = fields
                        recognized = {'arm', 'fold', 'seed'}.issubset(fields) and ('setup' in fields or 'protocol' in fields) and any(m in fields for m in METRICS)
                        if not recognized:
                            entry['status'] = 'catalog_only_unsupported_schema'
                            catalog.append(entry)
                            continue
                        rows = []
                        for i, row in enumerate(reader, 2):
                            if 'setup' not in row:
                                row['setup'] = 'strict:' + row['protocol']
                            row['_source'] = str(path)
                            row['_source_row'] = i
                            raw.write(json.dumps(row, sort_keys=True)+'\n')
                            rows.append(row)
                    if fingerprint(path) != before:
                        raise ValueError('Source changed during read: '+str(path))
                    entry.update(status='indexed_run_rows', rows=len(rows))
                    by_source[str(path)] = rows
                    grouped = defaultdict(list)
                    for r in rows:
                        cfg = {f: r.get(f) for f in CONFIG}
                        grouped[(r['setup'], r['arm'], json.dumps(cfg, sort_keys=True))].append(r)
                    for (setup, arm, cfg), rr in grouped.items():
                        counts = Counter(key(r) for r in rr)
                        valid = [r for r in rr if counts[key(r)] == 1 and not hard_failure(r)]
                        groups.append({'source': str(path), 'setup': setup, 'arm': arm,
                            'config': json.loads(cfg), 'rows': len(rr),
                            'duplicate_keys': [list(k) for k, n in counts.items() if n > 1],
                            'hard_failure_rows': sum(bool(hard_failure(r)) for r in rr),
                            'split_warning_rows': sum(bool(r.get('degenerate')) for r in rr),
                            'flagged_rows': sum(bool(r.get('train_flags')) for r in rr),
                            'missing_diagnostics': [f for f in ('leak_check', 'train_flags', 'split_sha256') if f not in fields],
                            'observed_keys': [list(k) for k in sorted(counts)],
                            'metrics': {m: stats([number(r.get(m)) for r in valid if number(r.get(m)) is not None]) for m in METRICS},
                            'unflagged_metrics': {m: stats([number(r.get(m)) for r in valid if not r.get('train_flags') and number(r.get(m)) is not None]) for m in METRICS}})
                except (UnicodeError, csv.Error) as exc:
                    entry.update(status='unreadable', error=str(exc))
                catalog.append(entry)
    return catalog, groups, by_source


def run(args):
    config = json.loads(args.manifest.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    catalog, groups, sources = inventory(config['results_roots'], out)
    comparisons = []
    for recipe in config.get('comparisons', []):
        paths = [str(Path(p).resolve()) for p in recipe['sources']]
        missing = [p for p in paths if p not in sources]
        rows = [r for p in paths for r in sources.get(p, [])]
        for setup in recipe.get('setups', SETUPS):
            result = paired(rows, recipe['left'], recipe['right'], setup,
                            recipe.get('metric', 'wq_mrr'), recipe.get('right_metric'),
                            recipe.get('expected_keys'))
            result.update(id=recipe['id']+':'+setup, sources=paths,
                          missing_sources=missing, caveats=recipe.get('caveats', []))
            comparisons.append(result)
    context = []
    context_dir = out/'context'
    context_dir.mkdir()
    for i, item in enumerate(config.get('context', [])):
        p = Path(item['path'])
        if not p.is_file():
            context.append(dict(item, status='missing'))
            continue
        info = fingerprint(p)
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError('Context changed during read: '+str(p))
        dest = context_dir/('%03d_' % i+p.name)
        dest.write_bytes(data)
        context.append(dict(item, **{k:v for k,v in info.items() if k != 'path'}, snapshot=str(dest.relative_to(out)), status='snapshotted'))
    review = {'schema_version': 1, 'scope': config['scope'],
              'limitations': ['Retrospective historical test metrics are development evidence, not untouched confirmation.',
                'Missing metadata remains unknown; identical counts do not prove identical split membership.',
                'Sources are never pooled except explicitly enumerated recipe shards.',
                'SD across overlapping folds is descriptive, not an independent-sample uncertainty estimate.',
                'Unsupported CSVs and non-CSV legacy artifacts are not numerically synthesized.',
                'LOW_TRAIN_FRAC, DROPPED_ROWS and TINY_TRAIN are retained as split warnings, not treated as leakage; unflagged means no training warnings.',
                'Existing profile and audit snapshots are reused; this command does not rerun data audits.',
                'Evidence aggregation is deterministic. Research synthesis is assistant-authored, not an autonomous planner.'],
              'catalog': catalog, 'groups': groups, 'comparisons': comparisons,
              'context': context, 'research_questions': config['research_questions']}
    save(out/'review.json', review)
    save(out/'manifest.json', config)
    lines = ['# Whole-project evidence review', '', config['scope'], '',
             '## Coverage', '', '%d CSV artifacts; %d indexed rows; %d source/config/arm/setting groups.' %
             (len(catalog), sum(c.get('rows', 0) for c in catalog), len(groups)), '',
             'Context snapshots include data audits, model code, previous campaigns and active research notes.', '',
             '## Paired historical contrasts', '',
             '| Evidence ID | Δ wq-MRR (left − right) | Paired n | Unflagged Δ (n) |',
             '|---|---:|---:|---:|']
    for c in comparisons:
        d, u = c['delta'], c['unflagged_delta']
        fmt = lambda x: 'unknown' if x is None else '%+.5f' % x
        lines.append('| %s | %s | %d | %s (%d) |' % (c['id'], fmt(d['mean']), d['n'], fmt(u['mean']), u['n']))
    lines += ['', '## Questions to resolve across the project', '']
    lines += ['- '+q for q in config['research_questions']]
    lines += ['', '## Interpretation limits', '']+[ '- '+x for x in review['limitations']]
    lines += ['', 'See review.json for exact recipes, configuration differences, exclusions, missing runs, sources and row-level paired evidence. Read the context snapshots before choosing a follow-up.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    complete = {'status': 'complete', 'adapter': fingerprint(__file__),
                'manifest': fingerprint(args.manifest),
                'outputs': [fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()]}
    save(out/'complete.json', complete)
    return out/'report.md'


def add_parser(sub):
    p = sub.add_parser('dpsyn-review', help='Review whole-project evidence before designing another campaign; no training')
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
