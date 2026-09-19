import importlib.util
import unittest
from types import SimpleNamespace
import numpy as np
import torch
HAS_DPSYN = importlib.util.find_spec("dpsyn") is not None
if HAS_DPSYN:
    from dpsyn.channel_selectivity import graph_inputs, controlled_edges, fit


def dataset():
    return SimpleNamespace(d1=np.array([0,0,1,1,2,3]),d2=np.array([1,1,2,2,4,5]),
        cell=np.array([0,1,0,1,0,1]),y=np.array([1,0,1,0,1,0]),
        tissue_of_cell=torch.tensor([0,1]),cell_has_ccle=torch.tensor([True,True]),
        cell_x=torch.tensor([[1.,2.,3.],[3.,1.,2.]]),n_cell=2,n_drug=6,n_prot=4,tissue_names=['A','B'],
        e_pd=torch.tensor([[0,1,2,3,0,1],[0,1,2,3,4,5]]),e_pp=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]]))


@unittest.skipUnless(HAS_DPSYN, "Requires the dpsyn backend checkout")
class ChannelTests(unittest.TestCase):
    def test_heldout_labels_cannot_change_dd_edges(self):
        ds=dataset();a=graph_inputs(ds,np.arange(4),seed=0)
        ds.y[4:]=1-ds.y[4:];b=graph_inputs(ds,np.arange(4),seed=0)
        self.assertTrue(torch.equal(a.edge_index,b.edge_index))
        self.assertEqual({tuple(sorted(x)) for x in a.edge_index.T.tolist()},{(0,1),(1,2)})

    def test_pd_rewire_preserves_both_degrees_and_dd(self):
        ds=dataset();dd=graph_inputs(ds,np.arange(4),seed=0)
        changed,pd,pp=controlled_edges(dd,ds.e_pd,ds.e_pp,'rewired_pd',2)
        self.assertTrue(torch.equal(changed.edge_index,dd.edge_index))
        self.assertTrue(torch.equal(pp,ds.e_pp))
        for axis in (0,1):
            np.testing.assert_array_equal(np.sort(pd[axis].numpy()),np.sort(ds.e_pd[axis].numpy()))

    def test_dd_rewiring_does_not_introduce_heldout_drugs(self):
        ds=dataset();dd=graph_inputs(ds,np.arange(4),seed=0)
        d,_,_=controlled_edges(dd,ds.e_pd,ds.e_pp,'rewired_dd',1)
        self.assertTrue(set(d.edge_index.ravel().tolist())<=set(dd.edge_index.ravel().tolist()))
        edges={tuple(x) for x in d.edge_index.T.tolist()}
        self.assertTrue(all((b,a) in edges for a,b in edges))

    def test_empty_controls_share_initialization_and_parameters(self):
        ds=dataset();dd=graph_inputs(ds,np.arange(4),seed=0)
        torch.manual_seed(4);dx=torch.randn(6,4);px=torch.randn(4,5)
        rows=np.c_[ds.d1,ds.d2,ds.cell]
        for real,empty in [('dd','dd_empty'),('ppd','ppd_empty'),('ddppd','ddppd_empty')]:
            results=[]
            for arm in (real,empty):
                score,info=fit(ds,dx,px,dd,rows[:4],ds.y[:4],rows[4:],ds.y[4:],
                    arm=arm,task='localization',seed=1,device='cpu',max_epochs=2,min_epochs=1,patience=1)
                self.assertTrue(np.isfinite(score).all());results.append(info)
            self.assertEqual(results[0]['parameters'],results[1]['parameters'])
            self.assertEqual(results[0]['initial_parameters_sha256'],results[1]['initial_parameters_sha256'])
            self.assertEqual(results[1]['edges']['dd_edges'],0)
            self.assertEqual(results[1]['edges']['pd_edges'],0)

if __name__=='__main__':unittest.main()
