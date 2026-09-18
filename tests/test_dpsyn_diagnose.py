import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest

from graphsmith.dpsyn_diagnose import panel_summary, upper_wilson, reconstruct, whitening_audit


class SelectivityTests(unittest.TestCase):
    def test_whitening_warning_sensitivity_preserves_matched_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(12):
                setup = ['B1', 'B3', 'B3A'][i//4]
                for template, transform, score in [('sweep_higsyn_molecular_60800849', 'whiten', .7),
                                                    ('standardized_molecular_60828758', 'std', .8)]:
                    records = [dict(arm='ddppd|none', setup=setup, fold=i%4, seed=seed,
                        n_rows=100, n_train=25, n_val=25, n_test=50,
                        feat_transform=transform, wq_mrr=score, error='', leak_check='PASS',
                        train_flags='EARLY_BEST' if i == 6 and seed == 0 else '') for seed in range(3)]
                    with (root/f'{template}_{i}.csv').open('w', newline='') as stream:
                        w = csv.DictWriter(stream, fieldnames=list(records[0]))
                        w.writeheader(); w.writerows(records)
            result = whitening_audit(root, [])
            b3 = next(r for r in result['summary'] if r['setup'] == 'B3')
            self.assertEqual(b3['n'], 12)
            self.assertEqual(b3['unflagged_pairs'], 11)
            self.assertAlmostEqual(b3['mean_difference_unflagged'], -.1)
            path = root/'sweep_higsyn_molecular_60800849_6.csv'
            path.write_text(path.read_text().replace('EARLY_BEST', 'NEVER_SELECTED'))
            with self.assertRaisesRegex(ValueError, 'Flagged'):
                whitening_audit(root, [])

    def test_reconstruction_uses_original_builder_drug_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('drugcomb', 'ccle', 'unified', 'larger_graph'):
                (root/name).mkdir()
            (root/'drugcomb/drugcomb_name_to_dbid_mapping_extended.csv').write_text(
                'name,name_norm,drugbank_id\na,a,D1\nb,b,D2\nc,c,D3\n')
            (root/'drugcomb/drugcombs_scored.csv').write_text(
                'Drug1,Drug2,Cell line,ZIP\na,b,C0,20\na,b,C1,0\na,c,C0,30\n')
            (root/'ccle/cell_line_ccle_map.csv').write_text('drugcomb_name,ccle_modelid\nC0,M0\nC1,M1\n')
            original = root/'original-map.csv'
            original.write_text('drugbank_id\nD1\nD2\n')
            (root/'larger_graph/drug_id_map.csv').write_text('drugbank_id\nD1\nD2\nD3\n')
            (root/'unified/canonical_folds_extmap.csv').write_text(
                'drug1_dbid,drug2_dbid,cell_name,label,mean_zip\nD1,D2,C0,1,20\n')
            (root/'unified/canonical_folds_extmap_meta.json').write_text(json.dumps({
                'drug_id_map_csv': str(original), 'dedup_stats': {'n_triplets_pre_band':2}}))
            measured, canonical, counts, sources = reconstruct(root, root/'larger_graph')
            self.assertEqual(len(measured), 2)
            self.assertEqual(len(canonical), 1)
            self.assertEqual(counts['unmapped_drug'], 1)

    def test_neutral_measured_cells_are_in_the_denominator(self):
        result = panel_summary([20] + [0]*9)
        self.assertTrue(result['selective_observed_10pct'])
        self.assertEqual(result['positive_fraction'], .1)
        self.assertFalse(panel_summary([20])['selective_observed_10pct'])

    def test_no_synergy_and_insufficient_coverage_are_not_selective(self):
        self.assertFalse(panel_summary([-20]*100)['selective_observed_10pct'])
        self.assertFalse(panel_summary([20]+[-20]*8)['selective_observed_10pct'])
        self.assertIsNone(panel_summary([])['positive_fraction'])

    def test_uncertainty_distinguishes_small_and_large_panels(self):
        self.assertGreater(upper_wilson(1,10),.1)
        self.assertLess(upper_wilson(1,100),.1)
        self.assertTrue(panel_summary([20]+[0]*99)['selective_wilson_upper_10pct'])
        self.assertFalse(panel_summary([10]*100)['selective_observed_10pct'])

    @unittest.skipUnless(importlib.util.find_spec('dpsyn'), 'Requires dpsyn checkout')
    def test_new_campaign_rejects_whitening_before_loading_data(self):
        from dpsyn import config as C
        from graphsmith.dpsyn_campaign import freeze
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); evidence = root/'evidence.json'
            evidence.write_text(json.dumps({'observed':1}))
            raw = {'name':'test', 'hypothesis':'test', 'minimum_effect':.01,
                   'common': {'feat_transform':'whiten'},
                   'arms':[{'name':'a','config':{}},{'name':'b','config':{}}],
                   'seeds':[0], 'fold_indices':[0], 'max_jobs':2, 'job_timeout_seconds':1,
                   'evidence_profile':str(evidence), 'evidence_pointer':'/observed'}
            with self.assertRaisesRegex(ValueError,'Whitening is disabled'):
                freeze(raw,Path(C.__file__).parent.parent,root/'output')
            self.assertFalse((root/'output').exists())
