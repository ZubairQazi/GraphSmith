"""Audit independent controls against actual dpsyn inputs and all paper splits.

No fitting of predictive models and no test scoring. Existing audits are evidence,
not a substitute for checking the current inputs and tensors.
"""
import json
from pathlib import Path
import sys

from .dpsyn import fingerprint, audit_inputs, validate_dataset, save


def run(args):
    sys.path.insert(0, str(args.dpsyn_repo.resolve()))
    import numpy as np
    import torch
    from dpsyn import config as C, data as D, splits as S, features as F
    from dpsyn import feature_contract as FC
    from dpsyn.development import split_diagnostics
    from dpsyn.model import Encoder
    if Path(C.__file__).resolve().parent.parent != args.dpsyn_repo.resolve():
        raise ValueError('Wrong dpsyn checkout')
    review = json.loads((args.review/'review.json').read_text())
    verification = json.loads((args.review/'verification.json').read_text())
    if verification['status'] != 'PASS' or verification['review_sha256'] != fingerprint(args.review/'review.json')['sha256']:
        raise ValueError('Project review has not passed independent verification')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = C.RunConfig(folds='extmap', graph='dp_drugs_pd_drugbank_bindingdb_kikd',
        drug_feat='atompair_full', prot_feat='esm2_full', feat_transform='std',
        node_feat_mode='real_std', prot_feat_mode='real_std', feature_protocol='independent_v1',
        channels=('dd','ppd'), dd_attr='none', num_folds=4, validation_only=True, floors=())
    base_config = cfg.to_dict()
    paths = C.resolve(cfg)
    inputs = [fingerprint(p) for p in paths.values() if p.is_file()]
    sources = [fingerprint(p) for p in sorted(Path(C.__file__).parent.glob('*.py'))]+[fingerprint(__file__)]
    ds = D.build(cfg, verbose=False)
    data_audit, graph_audit = audit_inputs(paths), validate_dataset(ds)
    if ds.n != 42874 or ds.n != data_audit['fold_csv']['rows']:
        raise ValueError('Unexpected canonical universe')
    original = {name: FC.tensor_record(getattr(ds,name)) for name in
                ('drug_x','prot_x','cell_x','drug_feat_missing','prot_feat_missing','cell_has_ccle')}
    records, legacy = [], []
    for seed in (0,1,2):
        reference_d, reference_p = {}, {}
        missing_d, missing_p = None, None
        for dm in FC.MODES:
            for pm in ('real_std','random','shuffled','constant'):
                rng = torch.get_rng_state().clone()
                dx,px,keep = F.prepare_node_features(ds,mode=dm,prot_mode=pm,seed=seed,
                    protocol='independent_v1',transform='std',missing='random',verbose=False)
                if not torch.equal(rng,torch.get_rng_state()) or not keep.all():
                    raise ValueError('RNG or population changed')
                d,p = FC.tensor_record(dx),FC.tensor_record(px)
                if list(dx.shape)!=list(ds.drug_x.shape) or list(px.shape)!=list(ds.prot_x.shape):
                    raise ValueError('Width changed')
                if reference_d.setdefault(dm,d)!=d or reference_p.setdefault(pm,p)!=p:
                    raise ValueError('A treatment changed the other block')
                md,mp = FC.tensor_record(dx[ds.drug_feat_missing]),FC.tensor_record(px[ds.prot_feat_missing])
                if missing_d is None: missing_d,missing_p=md,mp
                if md!=missing_d or mp!=missing_p: raise ValueError('Missing treatment drift')
                records.append(dict(seed=seed,drug_mode=dm,protein_mode=pm,drug=d,protein=p,
                    drug_missing=md,protein_missing=mp,global_rng_unchanged=True,all_rows_retained=True))
        for dm in ('real_std','random','constant'):
            dx,px,_ = F.prepare_node_features(ds,mode=dm,prot_mode='real_std',seed=seed,
                protocol='legacy',transform='std',missing='random',verbose=False)
            legacy.append(dict(seed=seed,drug_mode=dm,protein_mode='real_std',
                               drug=FC.tensor_record(dx),protein=FC.tensor_record(px)))
    for name,rec in original.items():
        if FC.tensor_record(getattr(ds,name))!=rec: raise ValueError('Dataset mutation: '+name)
    contract = json.loads(args.reference_contract.read_text())
    historical_arrays = np.load(contract['split_arrays']['path'], allow_pickle=False)
    if fingerprint(contract['split_arrays']['path']) != contract['split_arrays']:
        raise ValueError('Previously frozen split arrays changed')
    for item in contract['inputs']:
        if fingerprint(item['path']) != item: raise ValueError('Historical alignment input changed')
    splits,arrays={},{}
    with torch.random.fork_rng(devices=[]):
        enc = Encoder(ds.drug_x.shape[1],ds.prot_x.shape[1],ds.cell_x.shape[1],
                      len(ds.tissue_names),channels=('ego',),hidden=8,out=4,edge_dim=4)
    for setup in C.PAPER_SETUPS:
        for fold in range(4):
            for seed in (0,1,2):
                cfg.setup,cfg.fold_index,cfg.seed=setup,fold,seed
                tr,va,te=S.make_split(setup,cid1=ds.d1,cid2=ds.d2,cell=ds.cell,y=ds.y,
                                     num_folds=4,fold_index=fold,seed=seed)
                summary=split_diagnostics(cfg,ds,tr,va,te)
                key=f'{setup}_f{fold}_s{seed}'
                if setup=='B3':
                    if summary['split_hash']!=contract['splits'][key]['split_hash']:
                        raise ValueError('Existing B3 split hash mismatch')
                    for label,idx in zip(('train','validation','test'),(tr,va,te)):
                        if not np.array_equal(idx,historical_arrays[key+'_'+label]):
                            raise ValueError('Existing B3 row assignments differ')
                for label,idx in zip(('train','validation','test'),(tr,va,te)):
                    arrays[key+'_'+label]=idx
                train_cells=np.zeros(ds.n_cell,dtype=bool)
                train_cells[np.unique(ds.cell[tr])]=True
                fit=torch.as_tensor(train_cells)&ds.cell_has_ccle
                # Reset buffers explicitly because some splits may have fewer than two fit cells.
                enc.cell_mu.zero_();enc.cell_sd.fill_(1)
                enc.set_cell_stats(ds.cell_x,train_cells,ds.cell_has_ccle)
                if int(fit.sum())>=2:
                    raw=ds.cell_x[fit]
                    sd=raw.std(0)
                    expected_sd=torch.where(sd>1e-6,sd,torch.ones_like(sd))
                    if not torch.equal(enc.cell_mu,raw.mean(0)) or not torch.equal(enc.cell_sd,expected_sd):
                        raise ValueError('Cell scaling not fitted on covered training cells')
                summary['cell_scaling']={'covered_train_cells':int(fit.sum()),
                    'mu':FC.tensor_record(enc.cell_mu),'sd':FC.tensor_record(enc.cell_sd),
                    'keep':FC.tensor_record(enc.cell_keep),'train_cells':FC.tensor_record(train_cells)}
                splits[key]=summary
    if inputs != [fingerprint(p) for p in paths.values() if p.is_file()]:
        raise ValueError('Inputs changed during audit')
    if any(fingerprint(x['path'])!=x for x in sources):
        raise ValueError('Source changed during audit')
    np.savez_compressed(out/'split_indices.npz',**arrays)
    result={'status':'PASS','protocol':FC.PROTOCOL,'base_config':base_config,
            'review':fingerprint(args.review/'review.json'),'reference_contract':fingerprint(args.reference_contract),
            'input_audit':data_audit,'graph_audit':graph_audit,'original_tensors':original,
            'treatments':records,'legacy_treatments':legacy,'splits':splits,
            'split_arrays':fingerprint(out/'split_indices.npz'), 'inputs':inputs,'sources':sources,
            'limitations':['No model training or test scoring; this proves control invariants, not improved prediction.',
              'Node standardization uses all covered graph nodes; cell scaling uses covered training cells.',
              'New output-space imputation, retained columns and population-SD scaling intentionally differ from legacy.',
              'Constant means covered-node constant; random missing-node vectors are retained identically across arms.',
              'Cell interventions are not implemented here; existing raw cell data and fitted scaling are audited.',
              'All seven native split grids are frozen; only B3 also has a previously frozen exact-array reference.',
              'Historical CSV count/config agreement cannot prove unrecorded historical membership.']}
    save(out/'audit.json',result)
    (out/'report.md').write_text('# Independent feature-control audit\n\nPASS: 60 drug/protein treatment combinations across three seeds; fixed widths, independent blocks, identical missing-node vectors, unchanged raw cell data and no global RNG mutation.\n\n84 native paper splits audited and frozen; all 12 B3 assignments match the earlier contract exactly. Covered-training-cell scaling checked against the native encoder. No training or test scoring.\n\n'+ '\n'.join('- '+s for s in result['limitations'])+'\n')
    save(out/'complete.json',{'status':'complete','outputs':[fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()]})
    return out/'report.md'


def add_parser(sub):
    p=sub.add_parser('dpsyn-feature-audit',help='Verify independent feature controls on real inputs and all paper splits; no training')
    p.add_argument('--dpsyn-repo',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--reference-contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
