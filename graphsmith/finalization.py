"""Freeze a development-selected ensemble and submit it once to a private evaluator."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib import request, error, parse

from .data import digest
from .io import read, write
from .isolation import command, worker_environment


def service_call(url, action, payload=None):
    parsed = parse.urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path not in ('', '/'):
        raise ValueError('Local evaluator URL must be http://127.0.0.1:PORT')
    # Local evaluation must never traverse an environment-configured HTTP proxy.
    opener = request.build_opener(request.ProxyHandler({}))
    req = request.Request(url.rstrip('/') + '/' + action, data=json.dumps(payload or {}).encode(),
                          headers={'Content-Type': 'application/json'})
    try:
        with opener.open(req, timeout=30) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = json.loads(exc.read()).get('error', 'Evaluator rejected request')
        raise ValueError(detail) from None
    except (error.URLError, TimeoutError):
        raise ValueError('Evaluator unavailable; retry the same frozen submission when it is available') from None


def artifact_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root):
    from .engine import verify, locked, event, collect, report
    root = Path(root).resolve()
    with locked(root):
        bundle = verify(root, check_runtime=True)
        state = read(root / 'state.json')
        if not (root / 'service.json').exists():
            raise ValueError('No isolated test service: start a new campaign with --evaluator')
        if (root / 'selection.json').exists():
            selection = read(root / 'selection.json')
            if digest(selection) != state.get('selection_hash'):
                raise ValueError('Frozen local selection changed')
            return root / 'selection.json'
        if state['status'] != 'completed':
            raise ValueError('Every planned job must complete before method selection is frozen')
        config = read(root / 'config.json')
        if not {'prior', 'mlp', 'gcn', 'sage'} <= set(config['models']):
            raise ValueError('Required baseline coverage is incomplete')
        scores = {}
        for job in state['jobs']:
            result, _ = collect(root, job, bundle)
            scores.setdefault(job['model'], []).append(result['metrics']['macro_f1'])
        model = sorted(scores, key=lambda m: (-sum(scores[m])/len(scores[m]), m))[0]
        checkpoints = {}
        for seed in config['seeds']:
            name = 'prior.json' if model == 'prior' else 'checkpoint.pt'
            checkpoints[str(seed)] = artifact_hash(root / 'jobs' / f'{model}-{seed}' / name)
        selection = {'dataset_hash': digest(bundle), 'code_hash': read(root / 'provenance.json')['code_hash'],
                     'model': model, 'seeds': config['seeds'], 'checkpoint_hashes': checkpoints,
                     'threshold': .5, 'selection_policy': 'maximum mean validation macro-F1 across all predeclared seeds; model name breaks ties; mean-probability ensemble'}
        write(root / 'selection.json', selection)
        state.update(status='frozen', selection_hash=digest(selection))
        write(root / 'state.json', state)
        event(root, 'method_frozen', model=model, selection_hash=digest(selection))
        report(root)
        return root / 'selection.json'


def finalize(root):
    from .engine import verify, locked, event, report
    root = Path(root).resolve()
    if not (root / 'selection.json').exists():
        freeze(root)
    with locked(root):
        verify(root, check_runtime=True)
        state = read(root / 'state.json')
        selection = read(root / 'selection.json')
        if digest(selection) != state.get('selection_hash'):
            raise ValueError('Frozen local selection changed')
        if state['status'] == 'final_evaluated':
            report(root)
            return root / 'report.md'
        config = read(root / 'config.json')
        url = read(root / 'service.json')['url']
        for seed, expected in selection['checkpoint_hashes'].items():
            name = 'prior.json' if selection['model'] == 'prior' else 'checkpoint.pt'
            if artifact_hash(root / 'jobs' / f"{selection['model']}-{seed}" / name) != expected:
                raise ValueError('Selected checkpoint changed')
        prediction_path = root / 'test-predictions.json'
        inference = state.get('final_inference')
        if inference is None:
            remaining = config['campaign_seconds'] - sum(j['charged_seconds'] for j in state['jobs'])
            attempts = sum(j['status'] != 'pending' for j in state['jobs'])
            if attempts >= config['max_jobs'] or remaining < .1:
                raise ValueError('No budget for final inference; reserve one extra job before starting the campaign')
            timeout = min(config['job_timeout_seconds'], remaining)
            state['final_inference'] = {'status': 'running', 'charged_seconds': timeout}
            write(root / 'state.json', state)
            event(root, 'final_inference_reserved', seconds=timeout)
            started = time.monotonic()
            try:
                with (root / 'final-inference.log').open('w') as log:
                    subprocess.run(command([sys.executable, '-m', 'graphsmith.inference', str(root), str(timeout)], root),
                                   check=True, timeout=timeout + 1, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=Path(__file__).resolve().parent.parent,
                                   env=worker_environment(root))
                predictions = read(prediction_path)
                state['final_inference'].update(status='completed', predictions_hash=digest(predictions))
            except (subprocess.SubprocessError, OSError, ValueError):
                state['final_inference']['status'] = 'failed'
                raise ValueError('Final inference failed; no automatic rerun. See final-inference.log') from None
            finally:
                state['final_inference']['charged_seconds'] = time.monotonic() - started
                write(root / 'state.json', state)
        elif inference['status'] != 'completed':
            raise ValueError('Final inference was interrupted or failed; no automatic rerun')
        predictions = read(prediction_path)
        if digest(predictions) != state['final_inference']['predictions_hash']:
            raise ValueError('Final predictions changed')
        contract = {**selection, 'predictions_hash': digest(predictions)}
        # Persist exact request before any test exposure; only exact transport retries are permitted.
        submission = {'selection_hash': digest(contract), 'predictions': predictions}
        if (root / 'submission.json').exists() and read(root / 'submission.json') != submission:
            raise ValueError('Frozen submission changed')
        write(root / 'submission.json', submission)
        state['status'] = 'final_submitting'
        write(root / 'state.json', state)
        service_call(url, 'freeze', contract)
        result = service_call(url, 'evaluate', submission)
        if result.get('selection_hash') != digest(contract) or result.get('predictions_hash') != digest(predictions) or result.get('closed') is not True:
            raise ValueError('Evaluator response does not match frozen submission')
        write(root / 'final-result.json', result)
        state.update(status='final_evaluated', final_result_hash=digest(result))
        write(root / 'state.json', state)
        event(root, 'final_evaluated', selection_hash=digest(contract))
        return report(root)
