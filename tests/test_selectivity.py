import unittest
import numpy as np
from dpsyn.selectivity import standardize, fit, ranking_samples
from graphsmith.dpsyn_selectivity import expanded_assignment, selective_probability, expected_recall
from dpsyn.splits import split_B3


class SelectivityTests(unittest.TestCase):
    def test_expansion_preserves_original_drug_split_and_groups_pairs(self):
        a = np.array([0, 1, 2, 3, 4, 5, 6, 7]); b = np.roll(a, 1)
        for seed in range(3):
            for fold in range(4):
                parts = split_B3(a, b, 4, fold, seed)
                assignment = expanded_assignment(np.r_[a,b], np.repeat(a,3), np.repeat(b,3), fold, seed)
                self.assertTrue(np.all(assignment.reshape(-1,3) == assignment.reshape(-1,3)[:, :1]))
                for i, rows in enumerate(parts):
                    self.assertTrue(np.all(assignment[rows*3] == i))
        with self.assertRaises(ValueError):
            expanded_assignment(a, np.array([0]), np.array([99]), 0, 0)

    def test_scaling_does_not_fit_heldout_entities(self):
        x = np.array([[1,2],[3,2],[100,100]])
        z = standardize(x, np.array([0,1]))
        np.testing.assert_allclose(z[:2], [[-1,0],[1,0]])
        x[-1] = [-900,-900]
        np.testing.assert_allclose(standardize(x,[0,1])[:2], z[:2])

    def test_selectivity_probability_excludes_zero_positives(self):
        self.assertEqual(selective_probability(np.zeros(20)), 0)
        self.assertEqual(selective_probability(np.ones(20)), 0)
        self.assertEqual(selective_probability(np.r_[1.,np.zeros(9)]), 1)
        self.assertAlmostEqual(selective_probability(np.full(10,.1)), 10*.1*.9**9)

    def test_recall_ties_have_expected_random_floor(self):
        self.assertAlmostEqual(expected_recall([1]+[0]*9, np.zeros(10), 1), .1)
        self.assertEqual(expected_recall([1,0,0], [3,2,1], 1), 1)
        self.assertIsNone(expected_recall([0,0], [1,1], 1))

    def test_ranking_samples_share_pair_and_opposite_labels(self):
        rows = np.array([[0,1,0],[0,1,1],[2,3,0],[2,3,1]])
        groups = ranking_samples(rows, np.array([1,0,0,0]))
        self.assertEqual(len(groups),1)
        np.testing.assert_array_equal(groups[0][0],[0])
        np.testing.assert_array_equal(groups[0][1],[1])

    def test_fitted_predictor_is_symmetric_under_drug_pair_exchange(self):
        # A complete tiny CPU fit consumes only explicit train/validation arrays.
        rng = np.random.RandomState(9)
        drug, cell = rng.normal(size=(8,3)), rng.normal(size=(4,2))
        tr = np.array([[0,1,0],[0,1,1],[2,3,2],[2,3,3]])
        va = np.array([[4,0,0],[4,0,1],[5,1,2],[5,1,3]])
        y = np.array([1,0,1,0])
        p1, _, info = fit(drug,cell,tr,y,va,y,seed=0,max_epochs=11,device='cpu')
        exchanged = va.copy()
        exchanged[:, :2] = exchanged[:, [1,0]]
        p2, _, _ = fit(drug,cell,tr,y,exchanged,y,seed=0,max_epochs=11,device='cpu')
        np.testing.assert_array_equal(p1,p2)
        self.assertTrue(np.isfinite(p1).all())
        self.assertEqual(info['epochs'],11)


if __name__ == '__main__':
    unittest.main()
