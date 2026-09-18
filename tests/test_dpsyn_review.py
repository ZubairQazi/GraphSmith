import importlib.util
import tempfile
import unittest
from pathlib import Path
from graphsmith import dpsyn_review as r

class ReviewTests(unittest.TestCase):
    def row(self, arm, **kw):
        return dict(arm=arm, setup='B3', fold='0', seed='0', wq_mrr='.7', _source_row=2, **kw)

    def test_duplicates_excluded(self):
        x = r.paired([self.row('a'), self.row('a'), self.row('b')], 'a', 'b', 'B3')
        self.assertEqual(x['delta']['n'], 0)
        self.assertEqual(x['exclusions'][0]['reason'], 'missing_or_duplicate_key')

    def test_split_mismatch(self):
        x = r.paired([self.row('a', split_sha256='a'), self.row('b', split_sha256='b')], 'a', 'b', 'B3')
        self.assertEqual(x['exclusions'][0]['reason'], 'recorded_split_mismatch')

    def test_warning_sensitivity_and_absent_keys(self):
        x = r.paired([self.row('a', train_flags='EARLY_BEST'), self.row('b')], 'a', 'b', 'B3', expected_keys=[['0','0'],['1','0']])
        self.assertEqual(x['delta']['n'], 1)
        self.assertEqual(x['unflagged_delta']['n'], 0)
        self.assertEqual(x['absent_from_both'], [['1','0']])
        self.assertFalse(x['pairs'][0]['split_hash_verified'])

    def test_schema_discovery_unknown_config_no_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'out'
            out.mkdir()
            for name in ('e400_old.csv', 'unusual.csv'):
                (root/name).write_text('arm,setup,fold,seed,wq_mrr\na,LCO,0,0,.8\n')
            (root/'aggregate.csv').write_text('arm,mean\na,.8\n')
            catalog, groups, sources = r.inventory([root], out)
            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0]['config']['graph'], None)
            self.assertEqual(groups[0]['setup'], 'LCO')
            self.assertEqual(sum(c['status']=='catalog_only_unsupported_schema' for c in catalog), 1)

    def test_expected_split_warnings_are_not_leakage(self):
        good = self.row('a', degenerate='LOW_TRAIN_FRAC(25%) DROPPED_ROWS(63%)')
        self.assertEqual(r.hard_failure(good), [])
        self.assertEqual(r.hard_failure(self.row('a', degenerate='DRUG_LEAK(4)')), ['degenerate'])
        self.assertEqual(r.hard_failure(self.row('a', group_disjoint='FAIL')), ['group_disjoint'])

    def test_floor_paired_once(self):
        x = r.paired([self.row('a', floor='.6')], 'a', 'a', 'B3', right_metric='floor')
        self.assertEqual(x['delta']['n'], 1)
        self.assertAlmostEqual(x['delta']['mean'], .1)

if __name__ == '__main__':
    unittest.main()
