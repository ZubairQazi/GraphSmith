import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from graphsmith.dpsyn import inventory, profile_split, validate_dataset, hypotheses, triplet_key, history_table


class RealDataTests(unittest.TestCase):
    def setUp(self):
        self.ds = SimpleNamespace(n=6, n_drug=4, n_prot=2, n_cell=2,
            d1=np.array([0, 0, 1, 0, 1, 2]), d2=np.array([1, 1, 2, 3, 3, 3]),
            cell=np.array([0, 1, 0, 1, 1, 0]), y=np.array([0, 1, 1, 0, 1, 0]),
            drug_x=torch.ones(4, 3), prot_x=torch.ones(2, 2), cell_x=torch.ones(2, 3),
            e_pd=torch.tensor([[0, 1], [0, 1]]), e_pp=torch.tensor([[0, 1], [1, 0]]),
            drug_feat_missing=torch.tensor([False, False, True, False]),
            cell_has_ccle=torch.tensor([True, False]))
        self.parts = (np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
        self.degree = np.array([1, 1, 0, 0])

    def test_same_pair_in_different_cells_is_valid(self):
        validate_dataset(self.ds)
        self.ds.cell[1] = 0
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_dataset(self.ds)

    def test_nonfinite_features_and_invalid_edges_rejected(self):
        self.ds.drug_x[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_dataset(self.ds)
        self.ds.drug_x[0, 0] = 1
        self.ds.e_pd[1, 0] = 4
        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            validate_dataset(self.ds)

    def test_heldout_labels_do_not_change_profile_or_candidates_for_fixed_split(self):
        before = profile_split(self.ds, *self.parts, self.degree)
        self.ds.y[2:] = 1 - self.ds.y[2:]
        after = profile_split(self.ds, *self.parts, self.degree)
        self.assertEqual(before, after)
        self.assertEqual(hypotheses({'splits': {'B3_f0_s0': before}}),
                         hypotheses({'splits': {'B3_f0_s0': after}}))
        self.assertNotIn('labels', after['partitions']['test'])
        self.assertEqual(after['partitions']['train']['labels']['pairs_with_opposite_labels_across_cells'], 1)

    def test_overlapping_split_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            profile_split(self.ds, self.parts[0], self.parts[0], self.parts[2], self.degree)

    def test_normalized_unordered_join_key(self):
        self.assertEqual(triplet_key({'drug1_dbid': ' B ', 'drug2_dbid': 'A', 'cell_name': ' nci\\/res '}),
                         ('A', 'B', 'NCI/RES'))

    def test_inventory_never_pools_sources_and_excludes_failed_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = ['setup', 'fold', 'seed', 'graph', 'wq_mrr', 'auroc', 'leak_check', 'error', 'train_flags']
            rows = [dict(setup='B3', fold=i, seed=0, graph='g', wq_mrr=m,
                         auroc=.7, leak_check='PASS', error='', train_flags='')
                    for i, m in enumerate([.2, .4, .9])]
            rows[2]['error'] = 'failed'
            for filename in ['sweep_a.csv', 'sweep_b.csv']:
                with (root / filename).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            result = inventory(root, ['setup', 'graph'])
            self.assertEqual(len(result['groups']), 2)
            history_table(root / 'summary.csv', result)
            with (root / 'summary.csv').open() as stream:
                projected = list(csv.DictReader(stream))
            self.assertEqual(len(projected), 2)
            self.assertAlmostEqual(float(projected[0]['wq_mrr_mean']), .3)
            for group in result['groups']:
                self.assertEqual(group['eligible_rows'], 2)
                self.assertAlmostEqual(group['metrics']['wq_mrr']['mean'], .3)
                self.assertIn('drug_feat', group['missing_config_fields'])
                self.assertIn('Unknown', group['missing_runs'])

    def test_duplicate_fold_seed_prevents_aggregation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'sweep_duplicates.csv'
            p.write_text('setup,fold,seed,wq_mrr,leak_check\nB1,0,0,0.9,PASS\nB1,0,0,0.1,PASS\n')
            group = inventory(Path(temp), ['setup'])['groups'][0]
            self.assertEqual(group['duplicate_keys'], 1)
            self.assertIsNone(group['metrics']['wq_mrr']['mean'])

    def test_json_metrics_and_flagged_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'results.json').write_text(json.dumps({'config': {'setup': 'B3', 'fold_index': 0, 'seed': 0},
                'test_metrics': {'wq_mrr': .8}, 'training': {'flags': ['NEVER_SELECTED']}, 'leak_check': 'PASS'}))
            group = inventory(root, ['setup', 'seed', 'fold_index'])['groups'][0]
            self.assertEqual(group['eligible_rows'], 0)
            self.assertEqual(group['excluded_reason_counts'], {'training_flags': 1})


if __name__ == '__main__':
    unittest.main()
