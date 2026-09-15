"""Private evaluator service. Only public data, status, and one frozen score leave it."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys

from .data import Config, generate, digest, validate
from .metrics import evaluate


class Vault:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS campaign (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def call(self, action, payload=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT body FROM campaign WHERE id=1').fetchone()
            state = json.loads(row[0]) if row else None
            if action == 'initialize':
                if state is not None:
                    if state['selection'] is not None:
                        raise ValueError('Campaign frozen or closed; new search requires a fresh evaluator volume')
                    if digest(payload) != state['initial_config_hash']:
                        raise ValueError('Evaluator already initialized with a different contract')
                    return state['public']
                config = Config.parse(payload)
                # This private seed never appears in the agent-visible config or profile.
                secret_seed = secrets.randbits(128)
                bundle = generate(replace(config, data_seed=secret_seed))
                queries = bundle['queries']
                train_end, dev_end = int(len(queries) * .6), int(len(queries) * .8)
                public_queries = []
                for i, q in enumerate(queries[:dev_end]):
                    public_queries.append({**q, 'split': 'train' if i < train_end else 'validation'})
                private_targets = queries[dev_end:]
                bundle['queries'] = public_queries
                bundle['test_queries'] = [{k: q[k] for k in ('id', 'source', 'destination')} for q in private_targets]
                bundle['manifest']['generator'].pop('data_seed')
                bundle['manifest']['generator']['seed_policy'] = 'private service seed; public data_seed ignored for sealed campaigns'
                bundle['manifest']['final_test'] = 'labels retained only in evaluator volume'
                validate(bundle)
                if {q['target'] for q in private_targets} != {0, 1}:
                    raise ValueError('Generated test partition lacks both classes; use a larger dataset')
                state = {'public': bundle, 'targets': private_targets, 'seed': secret_seed,
                         'initial_config_hash': digest(payload), 'config': payload, 'selection': None, 'result': None,
                         'evaluation_request_hash': None}
                db.execute('INSERT INTO campaign VALUES (1, ?)', (json.dumps(state),))
                return bundle
            if state is None:
                raise ValueError('Evaluator is not initialized')
            if action == 'bundle':
                return state['public']
            if action == 'status':
                return {'initialized': True, 'frozen': state['selection'] is not None,
                        'closed': state['result'] is not None}
            if action == 'freeze':
                required = {'dataset_hash', 'code_hash', 'model', 'seeds', 'checkpoint_hashes',
                            'predictions_hash', 'threshold', 'selection_policy'}
                if not isinstance(payload, dict) or set(payload) != required:
                    raise ValueError('Invalid selection contract')
                if payload['dataset_hash'] != digest(state['public']) or payload['threshold'] != .5:
                    raise ValueError('Selection differs from the frozen evaluation contract')
                if payload['model'] not in {'prior', 'mlp', 'gcn', 'sage', 'ego_gcn'}:
                    raise ValueError('Unapproved model')
                if payload['seeds'] != list(Config.parse(state['config']).seeds):
                    raise ValueError('Selected seeds differ from the predeclared contract')
                if set(payload['checkpoint_hashes']) != {str(seed) for seed in payload['seeds']}:
                    raise ValueError('Checkpoint set differs from selected seeds')
                import re
                hashes = [payload['code_hash'], payload['predictions_hash'], *payload['checkpoint_hashes'].values()]
                if any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes):
                    raise ValueError('Malformed artifact hash')
                if state['selection'] is not None and state['selection'] != payload:
                    raise ValueError('Method already frozen; a new prediction set is forbidden')
                state['selection'] = payload
                response = {'frozen': True, 'selection_hash': digest(payload)}
            elif action == 'evaluate':
                if state['selection'] is None:
                    raise ValueError('Freeze selection before evaluation')
                request_hash = digest(payload)
                if state['result'] is not None:
                    if request_hash != state['evaluation_request_hash']:
                        raise ValueError('Campaign closed; test reuse is forbidden')
                    return state['result']  # Exact transport retry, never a second evaluation.
                if not isinstance(payload, dict) or set(payload) != {'selection_hash', 'predictions'}:
                    raise ValueError('Invalid evaluation request')
                if payload['selection_hash'] != digest(state['selection']):
                    raise ValueError('Selection hash mismatch')
                if digest(payload['predictions']) != state['selection']['predictions_hash']:
                    raise ValueError('Predictions differ from the frozen submission')
                metrics = evaluate(state['targets'], payload['predictions'])
                response = {'metrics': metrics, 'selection_hash': digest(state['selection']),
                            'predictions_hash': state['selection']['predictions_hash'],
                            'closed': True, 'evaluation': 'single_held_out_test',
                            'generalization': 'new pairs of potentially known endpoints on one synthetic graph'}
                state['result'], state['evaluation_request_hash'] = response, request_hash
            else:
                raise ValueError('Unsupported evaluator action')
            db.execute('UPDATE campaign SET body=? WHERE id=1', (json.dumps(state),))
            return response


def serve(path, port=8080):
    vault = Vault(path)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4_000_000:
                    raise ValueError('Request body size is invalid')
                body = json.loads(self.rfile.read(size))
                result = vault.call(self.path.lstrip('/'), body)
                code = 200
            except (ValueError, KeyError, TypeError) as exc:
                result, code = {'error': str(exc)}, 409
            raw = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *_):
            pass  # No label-bearing requests or credentials in logs.
    HTTPServer(('0.0.0.0', port), Handler).serve_forever()


if __name__ == '__main__':
    serve(sys.argv[1] if len(sys.argv) > 1 else '/vault/evaluator.sqlite')
