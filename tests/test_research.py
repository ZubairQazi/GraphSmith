import copy
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from graphsmith.data import Config, digest, validate
from graphsmith.engine import create, execute
from graphsmith.evaluator import Vault
from graphsmith.io import read, write
from graphsmith.planning import OfflinePlanner, OpenAIPlanner, critique, validate_plan


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.plan, _ = OfflinePlanner().propose({})

    def test_unapproved_interventions_and_metric_changes_rejected(self):
        for update in ({'intervention': 'shell command'}, {'metric': 'accuracy'}, {'minimum_effect': float('nan')}):
            with self.assertRaises(ValueError):
                validate_plan({**self.plan, **update})

    def test_rejected_and_inconclusive_outcomes(self):
        seeds = list(range(5))
        below = [{'seed': s, 'difference': -.01} for s in seeds]
        self.assertEqual(critique(self.plan, below, seeds)['verdict'], 'rejected_exploratory')
        self.assertEqual(critique(self.plan, below[:1], seeds)['verdict'], 'inconclusive')
        mixed = [dict(row) for row in below]
        mixed[0]['difference'] = .1
        self.assertEqual(critique(self.plan, mixed, seeds)['verdict'], 'inconclusive')

    def test_responses_adapter_schema_usage_and_no_tools(self):
        captured = {}
        def transport(req, timeout):
            captured.update(json.loads(req.data))
            return io.BytesIO(json.dumps({'id': 'fixture-response', 'status': 'completed', 'usage': {'total_tokens': 40},
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(self.plan)}]}]}).encode())
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-fixture-key'}):
            plan, receipt = OpenAIPlanner('test-model', transport=transport).propose({'profile': {}})
        self.assertEqual(plan, self.plan)
        self.assertEqual(receipt['usage']['total_tokens'], 40)
        self.assertNotIn('tools', captured)
        self.assertFalse(captured['store'])
        self.assertEqual(captured['max_output_tokens'], 2000)
        self.assertTrue(captured['text']['format']['strict'])

    def test_provider_refusal_is_not_a_plan(self):
        def transport(*args, **kwargs):
            return io.BytesIO(json.dumps({'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'refusal'}]}]}).encode())
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-fixture-key'}):
            with self.assertRaisesRegex(ValueError, 'declined'):
                OpenAIPlanner('test-model', transport=transport).propose({})

    def test_failed_planner_call_is_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(Config(), planner='openai', planner_model='test-model', max_llm_calls=1)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-fixture-key'}), patch.object(OpenAIPlanner, 'propose', side_effect=ValueError('offline')) as call:
                with self.assertRaisesRegex(ValueError, 'attempt retained'):
                    create(config, directory)
                self.assertEqual(call.call_count, 1)
            ledger = read(next(Path(directory).glob('*/planning.json')))
            self.assertEqual(ledger['llm_calls'], 1)
            self.assertEqual(ledger['status'], 'failed')


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vault = Vault(Path(self.temp.name) / 'private.sqlite')
        self.config = asdict(Config())
        self.public = self.vault.call('initialize', self.config)
        self.predictions = [{'id': q['id'], 'probability': .7} for q in self.public['test_queries']]
        self.selection = {'dataset_hash': digest(self.public), 'code_hash': 'a'*64,
            'model': 'mlp', 'seeds': list(Config().seeds),
            'checkpoint_hashes': {str(s): 'b'*64 for s in Config().seeds},
            'predictions_hash': digest(self.predictions), 'threshold': .5, 'selection_policy': 'fixed-test-fixture'}

    def test_public_bundle_has_no_private_seed_or_targets(self):
        validate(self.public)
        self.assertNotIn('data_seed', self.public['manifest']['generator'])
        self.assertTrue(all(set(q) == {'id', 'source', 'destination'} for q in self.public['test_queries']))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.vault.call('targets')

    def test_freeze_required_and_test_reuse_rejected(self):
        request = {'selection_hash': digest(self.selection), 'predictions': self.predictions}
        with self.assertRaisesRegex(ValueError, 'Freeze'):
            self.vault.call('evaluate', request)
        self.vault.call('freeze', self.selection)
        result = self.vault.call('evaluate', request)
        self.assertTrue(result['closed'])
        # Same request survives service restart, without another score calculation.
        restarted = Vault(self.vault.path)
        with patch('graphsmith.evaluator.evaluate', side_effect=AssertionError('must not recompute')):
            self.assertEqual(restarted.call('evaluate', request), result)
        changed = copy.deepcopy(request)
        changed['predictions'][0]['probability'] = .1
        with self.assertRaisesRegex(ValueError, 'reuse'):
            restarted.call('evaluate', changed)
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            restarted.call('freeze', {**self.selection, 'model': 'gcn'})
        with self.assertRaisesRegex(ValueError, 'frozen or closed'):
            restarted.call('initialize', self.config)

    def test_changed_predictions_rejected_before_exposure(self):
        self.vault.call('freeze', self.selection)
        with self.assertRaisesRegex(ValueError, 'Predictions differ'):
            self.vault.call('evaluate', {'selection_hash': digest(self.selection), 'predictions': self.predictions[:-1]})
        self.assertFalse(self.vault.call('status')['closed'])

    def test_label_mutation_cannot_change_search_artifacts(self):
        def public_artifacts():
            from graphsmith.data import profile
            from graphsmith.planning import OfflinePlanner
            data = self.vault.call('bundle')
            context = {'profile': profile(data)}
            plan, _ = OfflinePlanner().propose(context)
            return digest({'data': data, 'planner_input': context, 'hypothesis': plan})
        before = public_artifacts()
        with self.vault.connect() as db:
            state = json.loads(db.execute('SELECT body FROM campaign WHERE id=1').fetchone()[0])
            for row in state['targets']:
                row['target'] = 1 - row['target']
            db.execute('UPDATE campaign SET body=? WHERE id=1', (json.dumps(state),))
        self.assertEqual(before, public_artifacts())

    def test_private_label_changes_leave_training_predictions_identical(self):
        config = replace(Config(), seeds=(7,), models=('mlp',), epochs=3, max_jobs=1)
        output = Path(self.temp.name) / 'runs'
        with patch('graphsmith.engine.generate', return_value=self.vault.call('bundle')):
            first = create(config, output)
        execute(first)
        with self.vault.connect() as db:
            state = json.loads(db.execute('SELECT body FROM campaign WHERE id=1').fetchone()[0])
            for row in state['targets']:
                row['target'] = 1 - row['target']
            db.execute('UPDATE campaign SET body=? WHERE id=1', (json.dumps(state),))
        with patch('graphsmith.engine.generate', return_value=self.vault.call('bundle')):
            second = create(config, output)
        execute(second)
        for artifact in ('dataset.json', 'planner-input.json', 'hypothesis.json', 'jobs/mlp-7/predictions.json'):
            self.assertEqual(read(first / artifact), read(second / artifact))

    def test_test_pair_overlap_or_target_injection_is_rejected(self):
        bad = copy.deepcopy(self.public)
        bad['test_queries'][0]['target'] = 1
        with self.assertRaisesRegex(ValueError, 'forbids labels'):
            validate(bad)
        bad = copy.deepcopy(self.public)
        bad['test_queries'][0] = {k: bad['queries'][0][k] for k in ('id', 'source', 'destination')}
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate(bad)


@unittest.skipUnless(sys.platform == 'darwin', 'macOS isolation backend')
class IsolationTests(unittest.TestCase):
    def test_training_worker_runs_inside_sandbox(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(Config(), seeds=(7,), models=('mlp',), epochs=3, max_jobs=1)
            root = create(config, directory)
            # Test-only registration of a service boundary; no HTTP call is needed for training.
            service = {'url': 'http://127.0.0.1:1'}
            write(root / 'service.json', service)
            provenance = read(root / 'provenance.json')
            provenance['service_hash'] = digest(service)
            write(root / 'provenance.json', provenance)
            execute(root)
            self.assertEqual(read(root / 'state.json')['status'], 'completed',
                             (root / 'jobs/mlp-7/worker.log').read_text())

    def test_file_read_and_network_are_denied_in_worker(self):
        from graphsmith.isolation import command, probe
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'public'
            root.mkdir()
            secret = Path(directory) / 'withheld.json'
            secret.write_text('test-only-secret')
            self.assertTrue(probe(root, secret))
            script = ('import socket,sys\n'
                      'try: socket.create_connection(("127.0.0.1", 80), timeout=1)\n'
                      'except PermissionError: sys.exit(0)\n'
                      'except OSError: sys.exit(2)\n'
                      'sys.exit(1)')
            result = subprocess.run(command([sys.executable, '-c', script], root), capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
