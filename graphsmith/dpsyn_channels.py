"""Frozen native-channel experiments on the preceding real-data selectivity task."""
from pathlib import Path
import sys
from .io import read, write
from .dpsyn import fingerprint, table, triplet_key


def prepare(args):
    import dataclasses
    import numpy as np
    import torch
    import sklearn
    import platform
    from .dpsyn_diagnose import reconstruct
    from dpsyn import config as C, data as D, channel_selectivity as F
    parent=args.parent.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    prior=read(parent/'plan.json'); complete=read(parent/'complete.json')
    if complete['fits']!=72 or not (parent/'verification.json').exists():
        raise ValueError('Requires completed and independently verified feature campaign')
    for item in complete['files']+prior['sources']:
        if fingerprint(item['path'])!=item:
            raise ValueError('Prior campaign evidence changed')
    contract_path=next(p['path'] for p in prior['sources'] if p['path'].endswith('/contract.json'))
    contract=read(contract_path); cfg=C.RunConfig(**contract['shared_config'])
    ds=D.build(cfg,verbose=False)
    measured,canonical,counts,sources=reconstruct(C.DATA_ROOT,C.GRAPHS[cfg.graph])
    identity=table(parent/'row_identities.csv')
    keys=[(r['drug1'],r['drug2'],r['cell']) for r in identity]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate expanded identities')
    drug=dict(zip(ds.d1_dbid,ds.d1));drug.update(zip(ds.d2_dbid,ds.d2))
    names=ds.cell_names+sorted({k[2] for k in measured}-set(ds.cell_names))
    cells={c:i for i,c in enumerate(names)}
    cx,ch,tis,tn=D._load_cells(names,C.resolve(cfg))
    rows=np.array([(drug[a],drug[b],cells[c]) for a,b,c in keys],np.int64)
    y=np.array([measured[k]['mean_zip']>10 for k in keys],np.int64)
    ds=dataclasses.replace(ds,d1=rows[:,0],d2=rows[:,1],cell=rows[:,2],y=y,
        cell_names=names,cell_x=torch.from_numpy(cx),cell_has_ccle=torch.from_numpy(ch),
        tissue_of_cell=torch.from_numpy(tis),tissue_names=tn,
        d1_dbid=np.array([k[0] for k in keys]),d2_dbid=np.array([k[1] for k in keys]),cell_name=np.array([k[2] for k in keys]),
        fold_col=np.zeros(len(rows),np.int64),study=np.full(len(rows),'',dtype=object),targets={})
    from collections import defaultdict
    groups=defaultdict(list)
    for i,key in enumerate(keys): groups[key[:2]].append(i)
    pair_keys=sorted(groups); indices=[np.array(groups[k]) for k in pair_keys]
    pair_rows=np.array([rows[ix[0]] for ix in indices])
    pair_y=np.array([0<y[ix].sum()<=.1*len(ix) for ix in indices],np.int64)
    panel_n=np.array([len(ix) for ix in indices])
    splits=dict(np.load(parent/'split_indices.npz'))
    original=dict(np.load(contract['split_arrays']['path']))
    lookup={k:i for i,k in enumerate(keys)}
    old_pos=np.array([lookup[triplet_key(r)] for r in canonical])
    checks=[]
    for seed in range(3):
        for fold in range(4):
            key=f'B3_f{fold}_s{seed}'
            assignment=np.full(len(rows),-1)
            for part,name in enumerate(('train','validation','test')):
                ix=splits[key+'_'+name]
                if np.any(assignment[ix]!=-1):raise ValueError('Split overlap')
                assignment[ix]=part
                if not set(old_pos[original[key+'_'+name]])<=set(ix):raise ValueError('Original assignments changed')
            if (assignment<0).any() or any(len(np.unique(assignment[ix]))!=1 for ix in indices):
                raise ValueError('Incomplete partition or pair leakage')
            checks.append({'fold':fold,'seed':seed,'preserved_original_rows':len(canonical),'pair_overlap':0})
    torch.save({'ds':ds,'rows':rows,'pair_rows':pair_rows,'pair_y':pair_y,'panel_n':panel_n,
        'pair_indices':indices,'splits':splits,'config':contract['shared_config']},out/'data.pt')
    source_files=[Path(__file__),Path(F.__file__),Path(D.__file__),Path(C.__file__),
        Path(F.__file__).with_name('model.py'),Path(F.__file__).with_name('graph.py'),
        Path(F.__file__).with_name('features.py'),Path(F.__file__).with_name('metrics.py'),
        Path(__file__).with_name('dpsyn_selectivity.py'),Path(__file__).with_name('dpsyn_diagnose.py')]
    snapshot=out/'source';snapshot.mkdir()
    for path in source_files:(snapshot/path.name).write_bytes(path.read_bytes())
    plan={'parent':str(parent),'parent_complete':fingerprint(parent/'complete.json'),
        'parent_verification':fingerprint(parent/'verification.json'),'payload':fingerprint(out/'data.pt'),
        'sources':[fingerprint(p) for p in source_files]+sources+contract['inputs'],
        'arms':{k:{'channels':list(v[0]),'control':v[1]} for k,v in F.ARMS.items()},
        'tasks':['localization','pair_selectivity'],'folds':[0,1,2,3],'seeds':[0,1,2],
        'fits':216,'max_epochs':160,'patience':25,'min_epochs':20,
        'hidden':256,'out':128,'edge_dim':64,'dropout':.3,'conv':'sage','fuse':'mean',
        'optimizer':'AdamW lr=.001 weight_decay=.0001 clip=5',
        'selection':'validation AP for the fitted endpoint',
        'node_features':'Native dpsyn AtomPair-full/ESM2-full, real_std/std, full-universe transductive scaling; missing=random, seeded.',
        'cell_features':'Native expression+tissue; covered training cells fit scaling; unmatched expression zeroed.',
        'dd':'Symmetric training-positive pair graph only, dd_attr=none. For pair task, all training triplet synergy labels may supply DD edges, including small panels.',
        'rows':len(rows),'split_checks':checks,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__},
        'limitations':['Validation-only, reused for selection/reporting; source dataset exposes all labels.',
            'Native channel models differ from the earlier feature pilot; use the newly fitted ego and edge controls for causal comparisons.',
            'No-edge controls retain parameterization/initialization but disconnect parameters and use native empty-PPD fallback; active parameter counts are reported.',
            'PD rewiring preserves degree multiplicities, can add duplicate edges. DD rewiring preserves symmetry, approximately preserves degrees and can remove self-loops.',
            'B3 only; no inference across all split settings. Two endpoint models, not end-to-end joint evaluation.',
            'Pair-level DD models have auxiliary training synergy labels through the graph, beyond pair-status labels. Rewiring tests whether exact associations matter.',
            'Seeded GPU scatter operations are not guaranteed bitwise deterministic.']}
    write(out/'plan.json',plan)
    print(f'Prepared {len(rows)} frozen rows, 216 fits; original assignments preserved',flush=True)
    return str(out)


def worker(args):
    import numpy as np
    import torch
    from dpsyn import features as NF, channel_selectivity as F
    from .dpsyn_selectivity import evaluate,binary_metrics
    root=args.run.resolve(); plan=read(root/'plan.json')
    for item in [plan['payload']]+plan['sources']:
        if fingerprint(item['path'])!=item:raise ValueError('Frozen input/source changed')
    data=torch.load(root/'data.pt',map_location='cpu',weights_only=False)
    ds,rows=data['ds'],data['rows']; y=ds.y
    seed,fold=divmod(args.index,4); key=f'B3_f{fold}_s{seed}'
    if seed not in range(3):raise ValueError('Worker index must be 0..11')
    out=root/key;out.mkdir(exist_ok=False)
    tr,va=[data['splits'][key+'_'+p] for p in ('train','validation')]
    assignment=np.full(len(rows),2);assignment[tr]=0;assignment[va]=1
    pair_part=np.array([assignment[ix[0]] for ix in data['pair_indices']])
    ptr=np.flatnonzero((pair_part==0)&(data['panel_n']>=10))
    pva=np.flatnonzero((pair_part==1)&(data['panel_n']>=10))
    dx,px,keep=NF.prepare_node_features(ds,mode='real_std',missing='random',seed=seed,
        verbose=False,transform='std',feat_dim=256,prot_mode='real_std')
    if not keep.all():raise ValueError('Feature treatment dropped rows')
    dd=F.graph_inputs(ds,tr,seed=seed)
    seen=np.unique(rows[tr,:2]);cold=2-np.isin(rows[:,:2],seen).sum(1)
    degree=np.bincount(ds.e_pd[1].numpy(),minlength=ds.n_drug)
    covered=(degree[rows[:,:2]]>0).sum(1)
    # Pair-uniform strata preserve complete measured-panel denominators.
    strata={'all':np.ones(len(rows),bool),'one_cold':cold==1,'two_cold':cold==2,
            'no_targets':covered==0,'one_target':covered==1,'both_targeted':covered==2}
    records=[];state={'plan':fingerprint(root/'plan.json'),'completed':[],'expected_fits':18}
    write(out/'state.json',state)
    for task in plan['tasks']:
        if task=='localization':
            train_rows,val_rows,train_y,val_y=rows[tr],rows[va],y[tr],y[va]
            ids=va
        else:
            train_rows,val_rows=data['pair_rows'][ptr],data['pair_rows'][pva]
            train_y,val_y=data['pair_y'][ptr],data['pair_y'][pva];ids=pva
        for arm in plan['arms']:
            name=task+'_'+arm;state['active']=name;write(out/'state.json',state)
            score,training=F.fit(ds,dx,px,dd,train_rows,train_y,val_rows,val_y,
                arm=arm,task=task,seed=seed,device='cuda',max_epochs=plan['max_epochs'],patience=plan['patience'],min_epochs=plan['min_epochs'])
            np.savez_compressed(out/(name+'.npz'),ids=ids,probability=score)
            write(out/(name+'.json'),training)
            for minimum in (10,50):
                for stratum,mask in strata.items():
                    if task=='localization':
                        selected=mask[va]
                        if selected.any():
                            metrics=evaluate(rows[va][selected],y[va][selected],score[selected],minimum)
                        else:continue
                    else:
                        mask_pair=np.array([mask[ix[0]] for ix in data['pair_indices']])
                        selected=mask_pair[pva]&(data['panel_n'][pva]>=minimum)
                        metrics={'pair_selectivity':binary_metrics(val_y[selected],score[selected])}
                    records.append({'task':task,'arm':arm,'fold':fold,'seed':seed,'min_cells':minimum,'stratum':stratum,**metrics})
            state['completed'].append(name);state['active']=None
            write(out/'state.json',state);write(out/'metrics.json',records)
            print(f'{key} {name}: selected {training["best_epoch"]}/{training["epochs"]} flags={training["flags"]}',flush=True)
    # All controls for a channel set must start from exactly the same weights.
    for task in plan['tasks']:
        for group in [('dd','dd_empty'),('ppd','ppd_empty'),('ddppd','ddppd_empty','ddppd_rewired_pd','ddppd_rewired_dd')]:
            infos=[read(out/(task+'_'+a+'.json')) for a in group]
            if len({r['initial_parameters_sha256'] for r in infos})!=1 or len({r['parameters'] for r in infos})!=1:
                raise ValueError('Control parameterization/initialization mismatch')
    for item in plan['sources']:
        if fingerprint(item['path'])!=item:raise ValueError('Source changed during worker')
    write(out/'complete.json',{'fits':18,'test_scoring':False,
        'files':[fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()]})
    return str(out)


def run(args):
    sys.path.insert(0,str(args.dpsyn_repo.resolve()))
    return prepare(args) if args.stage=='prepare' else worker(args)


def add_parser(sub):
    p=sub.add_parser('dpsyn-channels',help='Native channel and edge controls on frozen selective-synergy data')
    p.add_argument('stage',choices=['prepare','worker'])
    p.add_argument('--dpsyn-repo',type=Path,required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--run',type=Path)
    p.add_argument('--index',type=int)
