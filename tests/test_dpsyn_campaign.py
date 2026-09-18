import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from graphsmith.data import digest
from graphsmith.dpsyn import fingerprint
from graphsmith.dpsyn_campaign import compare_rows, verify
from graphsmith.io import write


@unittest.skipUnless(importlib.util.find_spec('dpsyn'), 'Requires the dpsyn backend checkout')
class CampaignTests(unittest.TestCase):
    def rows(self):
        return {('D0', f'D{i+1}', 'C0'): {'label': str(int(i == 0)), 'probability': str(p),
            'covered_endpoints': '2', 'drug1_pd_degree': '1', 'drug2_pd_degree': '1',
            'drug1_train_frequency': '10', 'drug2_train_frequency': '2'}
            for i, p in enumerate([.2, .8, .7, .6])}

    def test_known_ranking_change_reproduces_from_contextual_predictions(self):
        a = self.rows()
        b = copy.deepcopy(a)
        b[('D0', 'D1', 'C0')]['probability'] = '.9'
        result = compare_rows(a, b, 3)
        self.assertEqual(result[0]['baseline']['wq_mrr'], .25)
        self.assertEqual(result[0]['intervention']['wq_mrr'], 1.)
        self.assertEqual(result[0]['delta_wq_mrr'], .75)
        self.assertEqual(result[1]['rows'], 0)
        self.assertIsNone(result[1]['delta_wq_mrr'])

    def test_unmatched_rows_and_changed_labels_rejected(self):
        a = self.rows()
        b = copy.deepcopy(a)
        b.pop(next(iter(b)))
        with self.assertRaisesRegex(ValueError, 'identical'):
            compare_rows(a, b, 3)
        b = copy.deepcopy(a)
        b[next(iter(b))]['label'] = '0'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            compare_rows(a, b, 3)

    def test_frozen_plan_and_inputs_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.txt'
            source.write_text('original')
            plan = {'inputs': [], 'sources': [], 'evidence': {'file': fingerprint(source)}}
            write(root / 'plan.json', plan)
            write(root / 'state.json', {'plan_hash': digest(plan)})
            verify(root)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify(root)
            source.write_text('original')
            write(root / 'plan.json', {**plan, 'new_field': True})
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
