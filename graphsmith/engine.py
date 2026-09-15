"""Single-worker, fixed-plan development campaigns with durable job reservations."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

from .data import Config, digest, generate, profile, validate
from .io import read, write
from .metrics import evaluate


def now():
    return datetime.now(timezone.utc).isoformat()


def code_hash():
    root = Path(__file__).parent
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))})


def environment():
    return {"python": sys.version, "packages": dict(sorted(
        (d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))}


def event(root, kind, **details):
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({"time": now(), "event": kind, **details}, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def locked(root):
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Campaign is already running in another process') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def create(config, output, evaluator_url=None):
    config.validate()
    if evaluator_url:
        from .finalization import service_call
        from .isolation import command
        command([sys.executable, '-V'], Path(output))  # Fail closed on unsupported platforms.
        bundle = service_call(evaluator_url, 'initialize', asdict(config))
        validate(bundle)
    else:
        bundle = generate(config)
    root = Path(output).resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    write(root / 'config.json', asdict(config))
    write(root / 'dataset.json', bundle)
    write(root / 'profile.json', profile(bundle))
    from .planning import OfflinePlanner, OpenAIPlanner, validate_plan
    planning = {'status': 'reserved', 'llm_calls': 1 if config.planner == 'openai' else 0}
    write(root / 'planning.json', planning)
    context = {'profile': profile(bundle), 'allowed_intervention': 'gcn → ego_gcn'}
    write(root / 'planner-input.json', context)
    try:
        provider = OpenAIPlanner(config.planner_model) if config.planner == 'openai' else OfflinePlanner()
        plan, receipt = provider.propose(context)
        validate_plan(plan)
        write(root / 'hypothesis.json', plan)
        planning.update(status='completed', receipt=receipt, hypothesis_hash=digest(plan))
    except (ValueError, OSError) as exc:
        planning.update(status='failed', error=str(exc))
        write(root / 'planning.json', planning)
        raise ValueError(f'Planning failed; attempt retained at {root}: {exc}') from None
    write(root / 'planning.json', planning)
    if evaluator_url:
        write(root / 'service.json', {'url': evaluator_url, 'boundary': 'private Docker volume; sandboxed research workers'})
    env = environment()
    write(root / 'environment.json', env)
    write(root / 'provenance.json', {
        **validate(bundle), "config_hash": digest(asdict(config)), "code_hash": code_hash(),
        "environment_hash": digest(env), "hypothesis_hash": digest(read(root / 'hypothesis.json')),
        "created": now(), "evaluation": "development_validation_only",
        "evidence_version": 2,
        "service_hash": digest(read(root / 'service.json')) if evaluator_url else None,
    })
    jobs = [{"id": f'{model}-{seed}', "model": model, "seed": seed, "status": "pending", "charged_seconds": 0}
            for seed in config.seeds for model in config.models]
    write(root / 'state.json', {"status": "validated", "jobs": jobs, "created": now()})
    event(root, 'validated', dataset_hash=digest(bundle))
    return root


def verify(root, check_runtime=False):
    provenance = read(root / 'provenance.json')
    bundle = read(root / 'dataset.json')
    if validate(bundle)['dataset_hash'] != provenance['dataset_hash']:
        raise ValueError('Dataset changed since campaign creation')
    if digest(read(root / 'config.json')) != provenance['config_hash']:
        raise ValueError('Frozen configuration changed')
    if digest(read(root / 'hypothesis.json')) != provenance['hypothesis_hash']:
        raise ValueError('Frozen hypothesis changed')
    if check_runtime and (code_hash() != provenance['code_hash'] or digest(environment()) != provenance['environment_hash']):
        raise ValueError('Code or environment changed; start a new campaign')
    if provenance.get('service_hash') is not None:
        if digest(read(root / 'service.json')) != provenance['service_hash']:
            raise ValueError('Evaluator connection changed')
    return bundle


def collect(root, job, bundle):
    path = root / 'jobs' / job['id']
    result = read(path / 'result.json')
    predictions = read(path / 'predictions.json')
    expected = [q for q in bundle['queries'] if q['split'] == 'validation']
    if result['job_id'] != job['id'] or result['model'] != job['model'] or result['seed'] != job['seed']:
        raise ValueError('Result identity mismatch')
    if evaluate(expected, predictions) != result['metrics']:
        raise ValueError('Result metrics do not match saved predictions')
    from .finalization import artifact_hash
    checkpoint = path / ('prior.json' if job['model'] == 'prior' else 'checkpoint.pt')
    recorded = {"result": result, "predictions": predictions}
    if read(root / 'provenance.json').get('evidence_version', 1) >= 2:
        recorded['checkpoint_hash'] = artifact_hash(checkpoint)
    evidence_hash = digest(recorded)
    if job.get('evidence_hash') and job['evidence_hash'] != evidence_hash:
        raise ValueError('Recorded evidence changed')
    return result, evidence_hash


def execute(root):
    root = Path(root).resolve()
    with locked(root):
        bundle = verify(root, check_runtime=True)
        config = Config.parse(read(root / 'config.json'))
        state = read(root / 'state.json')
        if state['status'] in {'frozen', 'final_submitting', 'final_evaluated'}:
            raise ValueError('Search is closed after method freeze; use finalize or report')
        # A lost orchestrator never causes the same reservation to be billed/launched again.
        for job in state['jobs']:
            if job['status'] == 'running':
                job['status'] = 'interrupted'
                job['error'] = 'Orchestrator interrupted; full reserved time remains charged. No automatic retry.'
                event(root, 'interrupted', job_id=job['id'])
            elif job['status'] == 'completed':
                collect(root, job, bundle)
        write(root / 'state.json', state)
        for job in state['jobs']:
            if job['status'] != 'pending':
                continue
            attempted = sum(j['status'] != 'pending' for j in state['jobs'])
            remaining = config.campaign_seconds - sum(j['charged_seconds'] for j in state['jobs'])
            if attempted >= config.max_jobs or remaining < .1:
                break
            timeout = min(config.job_timeout_seconds, remaining)
            job.update(status='running', charged_seconds=timeout, started=now())
            state['status'] = 'executing'
            write(root / 'state.json', state)
            job_dir = root / 'jobs' / job['id']
            write(job_dir / 'request.json', {"model": job['model'], "seed": job['seed']})
            event(root, 'job_reserved', job_id=job['id'], seconds=timeout)
            print(f"  {job['id']}", flush=True)
            start = time.monotonic()
            process = None
            interrupted = False
            try:
                with (job_dir / 'worker.log').open('w') as log:
                    arguments = [sys.executable, '-m', 'graphsmith.worker', str(root), job['id'], str(timeout)]
                    if (root / 'service.json').exists():
                        from .isolation import command
                        arguments = command(arguments, root)
                    from .isolation import worker_environment
                    process = subprocess.Popen(arguments,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                               env=worker_environment(root),
                                               cwd=Path(__file__).resolve().parent.parent)
                    return_code = process.wait(timeout=timeout)
                if return_code == -signal.SIGALRM:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                if return_code:
                    raise ValueError(f'Worker exited {return_code}; see worker.log')
                _, evidence_hash = collect(root, job, bundle)
                job.update(status='completed', evidence_hash=evidence_hash)
            except subprocess.TimeoutExpired:
                job.update(status='timed_out', error='Worker wall-time budget exhausted')
            except KeyboardInterrupt:
                job.update(status='interrupted', error='User interrupted execution')
                interrupted = True
            except (ValueError, OSError, KeyError) as exc:
                job.update(status='failed', error=str(exc))
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                job['charged_seconds'] = max(0.001, time.monotonic() - start)
                job['finished'] = now()
                write(root / 'state.json', state)
                event(root, 'job_finished', job_id=job['id'], status=job['status'], charged_seconds=job['charged_seconds'])
            if interrupted:
                state['status'] = 'interrupted'
                write(root / 'state.json', state)
                break
        if state['status'] != 'interrupted':
            statuses = {j['status'] for j in state['jobs']}
            state['status'] = ('budget_exhausted' if 'pending' in statuses else
                               'completed' if statuses == {'completed'} else 'completed_with_failures')
            write(root / 'state.json', state)
        event(root, 'campaign_stopped', status=state['status'])
        return report(root)


def report(root):
    root = Path(root).resolve()
    bundle = verify(root)
    state = read(root / 'state.json')
    grouped = {}
    rows = []
    for job in state['jobs']:
        if job['status'] == 'completed':
            result, _ = collect(root, job, bundle)
            grouped.setdefault(job['model'], []).append(result)
            rows.append(f"| {job['model']} | {job['seed']} | {result['metrics']['macro_f1']:.4f} | {result['selected_epoch']} | {result['parameter_count']} | [evidence](jobs/{job['id']}/result.json) |")
    summary = []
    for model, results in grouped.items():
        values = [r['metrics']['macro_f1'] for r in results]
        summary.append({"model": model, "n": len(values), "mean_macro_f1": statistics.mean(values),
                        "seed_sd": statistics.stdev(values) if len(values) > 1 else None})
    base = {r['seed']: r['metrics']['macro_f1'] for r in grouped.get('gcn', [])}
    altered = {r['seed']: r['metrics']['macro_f1'] for r in grouped.get('ego_gcn', [])}
    paired = [{"seed": s, "difference": altered[s] - base[s]} for s in sorted(base.keys() & altered.keys())]
    delta = statistics.mean(p['difference'] for p in paired) if paired else None
    verdict = ('inconclusive: no paired evidence' if delta is None else
               'directional prediction observed on development data' if delta > 0 else
               'directional prediction not observed on development data')
    evidence = {"status": state['status'], "evaluation": "development_validation_only", "summary": summary,
                "paired_ego_minus_gcn": paired, "mean_difference": delta, "assessment": verdict,
                "charged_worker_seconds": sum(j['charged_seconds'] for j in state['jobs']),
                "attempts": sum(j['status'] != 'pending' for j in state['jobs'])}
    from .planning import critique
    hypothesis = read(root / 'hypothesis.json')
    if 'minimum_effect' in hypothesis:
        assessment = critique(hypothesis, paired, read(root / 'config.json')['seeds'])
        write(root / 'critique.json', assessment)
        evidence['critique'] = assessment
    final_inference = state.get('final_inference', {})
    evidence['charged_worker_seconds'] += final_inference.get('charged_seconds', 0)
    evidence['attempts'] += int(bool(final_inference))
    write(root / 'summary.json', evidence)
    lines = ['# GraphSmith experiment report', '', '**Exploratory development results — no final test or independent confirmation.**', '',
             f"Status: `{state['status']}`. Dataset: `{read(root / 'provenance.json')['dataset_hash']}`.", '',
             '## Research question', '', read(root / 'hypothesis.json')['statement'], '',
             '## Baseline comparison', '', '| Model | Seeds | Mean validation macro-F1 | Seed SD |', '|---|---:|---:|---:|']
    for row in summary:
        sd = f"{row['seed_sd']:.4f}" if row['seed_sd'] is not None else 'n/a'
        lines.append(f"| {row['model']} | {row['n']} | {row['mean_macro_f1']:.4f} | {sd} |")
    lines += ['', '## Controlled intervention', '', f"Assessment: **{verdict}**.", '',
              f"Paired ego-path minus GCN mean macro-F1: {f'{delta:+.4f}' if delta is not None else 'n/a'} ({len(paired)} paired seeds).", '',
              'The decoder architecture and GCN aggregation are fixed. The ego path adds encoder parameters, so capacity remains a competing explanation.', '',
              '## Individual evidence', '', '| Model | Seed | Validation macro-F1 | Selected epoch | Parameters | Source |',
              '|---|---:|---:|---:|---:|---|', *rows, '', '## Budget and failures', '',
              f"Attempts: {evidence['attempts']}; charged worker wall time: {evidence['charged_worker_seconds']:.2f} seconds.", '']
    for job in state['jobs']:
        if job['status'] != 'completed':
            lines.append(f"- `{job['id']}`: {job['status']}. {job.get('error', '')}")
    missing = sorted({'prior', 'mlp', 'gcn', 'sage'} - grouped.keys())
    if missing:
        lines += ['', f"Incomplete baseline coverage: {', '.join(missing)}."]
    lines += ['', '## Interpretation limits', '',
              '- Validation selects checkpoints and is reused for comparisons. These are exploratory findings.',
              '- All seeds share one graph and split. Seed SD measures training variability, not population uncertainty.',
              '- Queries share endpoints. Pair-held-out results do not establish unseen-entity generalization.',
              '- One-layer dense CPU encoders are intended for small synthetic graphs. No context features in this slice.',
              '- Threshold 0.5 was fixed before training. No threshold optimization was performed.',
              '- Peak memory is not instrumented. Failures and baseline jobs count against the budget.',
              '- Next: match encoder capacity, vary neighborhood agreement, and repeat on independent graph draws.', '',
              '## Reproduce and inspect', '',
              'Use the pinned environment and unchanged source recorded in [provenance](provenance.json).', '',
              '```sh', f'python -m graphsmith research "{root / "config.json"}"', '```', '',
              '[Frozen configuration](config.json) · [Dataset](dataset.json) · [Profile](profile.json) · [Machine-readable summary](summary.json) · [Event log](events.jsonl)', '']
    if 'critique' in evidence:
        lines += ['## Structured critique', '',
                  f"**{evidence['critique']['verdict']}**: {evidence['critique']['reason']}", '',
                  '[Frozen hypothesis](hypothesis.json) · [Critique evidence](critique.json) · [Planner receipt](planning.json)', '']
    if state['status'] == 'final_evaluated':
        final = read(root / 'final-result.json')
        if digest(final) != state.get('final_result_hash'):
            raise ValueError('Final evaluation record changed')
        lines[2] = '**Development findings are exploratory. A separately held-out final score is reported below.**'
        lines += ['## Frozen final-test evaluation', '',
                  f"Selected ensemble: `{read(root / 'selection.json')['model']}`.", '',
                  f"Held-out macro-F1: **{final['metrics']['macro_f1']:.4f}** over {final['metrics']['count']} pairs.", '',
                  'This campaign is closed. One test set on one synthetic graph does not establish broader generalization.', '',
                  '[Frozen selection](selection.json) · [Final predictions](test-predictions.json) · [Evaluator receipt](final-result.json)', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return root / 'report.md'
