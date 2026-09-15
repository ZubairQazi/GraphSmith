import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch

from graphsmith.data import Config, generate, validate, digest
from graphsmith.engine import create, execute, report
from graphsmith.io import read, write
from graphsmith.metrics import evaluate
from graphsmith.models import PairModel, operators


class DataTests(unittest.TestCase):
    def setUp(self):
        self.bundle = generate(Config())

    def test_deterministic_generation(self):
        self.assertEqual(digest(self.bundle), digest(generate(Config())))

    def test_reverse_copy_across_splits_rejected(self):
        duplicate = copy.deepcopy(self.bundle['queries'][0])
        duplicate['source'], duplicate['destination'] = duplicate['destination'], duplicate['source']
        duplicate.update(id='reverse-copy', split='validation')
        self.bundle['queries'].append(duplicate)
        with self.assertRaisesRegex(ValueError, 'Duplicate/reversed'):
            validate(self.bundle)

    def test_duplicate_node_ids_rejected(self):
        self.bundle['node_ids'][0] = self.bundle['node_ids'][1]
        with self.assertRaisesRegex(ValueError, 'Node IDs'):
            validate(self.bundle)

    def test_nonfinite_features_rejected(self):
        self.bundle['node_features'][0][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate(self.bundle)

    def test_topology_policy_required(self):
        with self.assertRaisesRegex(ValueError, 'explicit topology'):
            Config.parse({})

    def test_no_test_targets_accepted(self):
        self.bundle['queries'][0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'train/validation'):
            validate(self.bundle)


class ModelTests(unittest.TestCase):
    def test_swap_invariance_and_node_permutation(self):
        torch.manual_seed(4)
        x = torch.randn(6, 3)
        edges = [[0, 1], [1, 0], [1, 3], [3, 1], [2, 4], [4, 2]]
        pairs = torch.tensor([[0, 4], [1, 2], [3, 5]])
        norm, mean = operators(x, edges)
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        inverse = torch.argsort(permutation)
        renamed_edges = [[inverse[a].item(), inverse[b].item()] for a, b in edges]
        perm_norm, perm_mean = operators(x[permutation], renamed_edges)
        for kind in ('mlp', 'gcn', 'sage', 'ego_gcn'):
            with self.subTest(kind=kind):
                model = PairModel(kind, 3, 8).eval()
                expected = model(x, norm, mean, pairs)
                torch.testing.assert_close(expected, model(x, norm, mean, pairs.flip(1)))
                torch.testing.assert_close(expected, model(x[permutation], perm_norm, perm_mean, inverse[pairs]), atol=1e-6, rtol=1e-5)


class MetricTests(unittest.TestCase):
    def test_id_alignment_and_known_f1(self):
        queries = [{'id': str(i), 'target': y} for i, y in enumerate([0, 0, 1, 1])]
        predictions = [{'id': str(i), 'probability': p} for i, p in enumerate([.1, .9, .8, .7])]
        self.assertAlmostEqual(evaluate(queries, predictions[::-1])['macro_f1'], (2/3 + 4/5)/2)
        for bad in (predictions[:-1], predictions + predictions[:1], [{'id': str(i), 'probability': float('nan')} for i in range(4)]):
            with self.assertRaises(ValueError):
                evaluate(queries, bad)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = replace(Config(), nodes=20, pairs=60, seeds=(7,), models=('prior',), epochs=3, max_jobs=1)

    def test_resume_preserves_completed_jobs_and_report_matches_predictions(self):
        root = create(self.config, self.temp.name)
        execute(root)
        before = read(root / 'state.json')['jobs']
        execute(root)
        self.assertEqual(before, read(root / 'state.json')['jobs'])
        self.assertEqual(read(root / 'summary.json')['attempts'], 1)
        predictions = read(root / 'jobs/prior-7/predictions.json')
        predictions[0]['probability'] = 0.123
        write(root / 'jobs/prior-7/predictions.json', predictions)
        with self.assertRaisesRegex(ValueError, 'evidence changed|metrics do not match'):
            report(root)

    def test_job_cap_counts_baselines(self):
        root = create(replace(self.config, models=('prior', 'mlp')), self.temp.name)
        execute(root)
        state = read(root / 'state.json')
        self.assertEqual(state['status'], 'budget_exhausted')
        self.assertEqual([j['status'] for j in state['jobs']], ['completed', 'pending'])
        execute(root)
        self.assertEqual(state['jobs'], read(root / 'state.json')['jobs'])

    def test_interrupted_reservation_is_not_relaunched(self):
        root = create(self.config, self.temp.name)
        state = read(root / 'state.json')
        state['jobs'][0].update(status='running', charged_seconds=60)
        write(root / 'state.json', state)
        execute(root)
        self.assertEqual(read(root / 'state.json')['jobs'][0]['status'], 'interrupted')
        self.assertEqual(read(root / 'summary.json')['charged_worker_seconds'], 60)
        self.assertFalse((root / 'jobs/prior-7/result.json').exists())

    def test_timeout_counts_against_job_budget(self):
        config = replace(self.config, models=('mlp', 'prior'), epochs=100000,
                         patience=100000, job_timeout_seconds=1)
        root = create(config, self.temp.name)
        execute(root)
        jobs = read(root / 'state.json')['jobs']
        self.assertEqual(jobs[0]['status'], 'timed_out')
        self.assertEqual(jobs[1]['status'], 'pending')
        self.assertEqual(read(root / 'summary.json')['attempts'], 1)

    def test_changed_dataset_rejected(self):
        root = create(self.config, self.temp.name)
        data = read(root / 'dataset.json')
        data['node_features'][0][0] += .1
        write(root / 'dataset.json', data)
        with self.assertRaisesRegex(ValueError, 'Dataset changed'):
            execute(root)

    def test_frozen_configuration_enforced(self):
        root = create(self.config, self.temp.name)
        config = read(root / 'config.json')
        config['epochs'] += 1
        write(root / 'config.json', config)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            execute(root)

    def test_seed_reproduces_training_predictions(self):
        config = replace(self.config, models=('mlp',))
        first, second = create(config, self.temp.name), create(config, self.temp.name)
        execute(first)
        execute(second)
        self.assertEqual(read(first / 'jobs/mlp-7/predictions.json'), read(second / 'jobs/mlp-7/predictions.json'))


if __name__ == '__main__':
    unittest.main()
