"""Frozen full-setting native architecture benchmark using verified GraphSmith evidence."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import statistics
import sys

from .dpsyn import fingerprint, save
from .dpsyn_campaign import use_repo, prediction_rows, compare_rows
from .data import digest

ARMS={'ego':['ego'],'dd':['dd'],'ppd':['ppd'],'ddppd':['dd','ppd']}
SETUPS=('B1','LCOW','B2','B2P','B3','B3A','B4')


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    assert digest(plan)==(root/'plan.sha256').read_text().strip(), 'Frozen plan changed'
    for item in plan['evidence']+plan['inputs']+plan['sources']:
        if fingerprint(item['path'])!=item: raise ValueError('Frozen artifact changed: '+item['path'])
    return plan


def freeze(args):
    C=use_repo(args.repo)
    audit=json.loads((args.audit/'audit.json').read_text())
    check=json.loads((args.audit/'verification.json').read_text())
    if check['status']!='PASS' or check['audit_sha256']!=fingerprint(args.audit/'audit.json')['sha256']:
        raise ValueError('Feature audit verification does not match')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    base=dict(audit['base_config'])
    base.update(max_epochs=400,patience=40,min_epochs=20,lr=.001,weight_decay=.0001,
        hidden_dim=256,out_dim=128,edge_dim=64,dropout=.3,conv='sage',fuse='mean',conv_root=True,
        dd_key='pair',dd_attr='none',dd_sign='pos',pair_pool='sym',cell_cond='feat',
        cell_feat='ccle_tissue',cell_transform='std',side_blocks=[],molecular_mode='none',
        validation_only=True,floors=[],feature_protocol='independent_v1',
        prot_feat_mode='real_std',node_feat_mode='real_std',feat_transform='std',missing_feat='random')
    tasks=[]
    for setup in SETUPS:
        for fold in range(4):
            for seed in range(3):
                key=f'{setup}_f{fold}_s{seed}'
                expected=next(r for r in audit['treatments'] if r['seed']==seed and r['drug_mode']=='real_std' and r['protein_mode']=='real_std')
                jobs=[]
                for arm,channels in ARMS.items():
                    cfg=C.RunConfig(**{**base,'setup':setup,'fold_index':fold,'seed':seed,'channels':channels,
                        'expected_split_hash':audit['splits'][key]['split_hash']})
                    jobs.append({'id':key+'_'+arm,'arm':arm,'config':cfg.to_dict()})
                tasks.append({'key':key,'setup':setup,'fold':fold,'seed':seed,'jobs':jobs,
                    'expected_features':{b:expected[b] for b in ('drug','protein')},
                    'expected_cell':audit['splits'][key]['cell_scaling']})
    evidence=[fingerprint(args.audit/name) for name in ('audit.json','verification.json','split_indices.npz')]
    evidence+=[audit['review']]
    sources=[fingerprint(p) for p in sorted((Path(C.__file__).parent).glob('*.py'))]
    sources += [fingerprint(p) for p in (Path(__file__),Path(__file__).with_name('dpsyn_campaign.py'))]
    plan={'tasks':tasks,'dpsyn_repo':str(args.repo.resolve()),'evidence':evidence,
        'inputs':audit['inputs'],'sources':sources,'budget':{'fits':336,'max_epochs':400,'seconds_per_fit':1500,'max_concurrent_gpus':3},
        'scope':'Retrospective evidence motivated this exploratory validation benchmark. No test scoring. Validation selects checkpoints.',
        'project_questions':['Where do graph channels add beyond native ego across the seven settings?', 'Do gains persist without training-warning rows and across target coverage?', 'Which settings justify a later matched topology or stronger feature-learner comparison?'],
        'supporting_and_conflicting_evidence':['augmented_graph_vs_mlp','channels_dd+ppd_real_std','std_graph_vs_ego','separate_vs_mixed','morgan_gbt_vs_atompair_gbt'],
        'primary':'Paired validation wq-MRR, equal-weight folds/seeds; B3A uses two-unseen-drug validation subset.',
        'caveats':['Architectures differ in parameter count; this alone does not isolate exact-edge value.',
        'No new non-neural baseline or rewiring arm; existing baseline evidence remains context, not a matched result.',
        'B3A validation training/checkpoint selection is shared in definition with B3; primary report filters both-cold validation rows.',
        'Node preprocessing transductive; cell scaling fitted on covered training cells.']}
    save(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(plan)+'\n')
    verify(root)
    return root


def worker(args):
    import torch
    if not torch.cuda.is_available(): raise ValueError('GPU allocation required')
    plan=verify(args.root)
    use_repo(plan['dpsyn_repo'])
    from dpsyn.config import RunConfig
    from dpsyn.run import run
    task=plan['tasks'][args.index]
    parent=args.root/('smoke' if args.smoke else 'jobs')
    folder=parent/task['key'];folder.mkdir(parents=True,exist_ok=False)
    results=[]
    for job in task['jobs']:
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Fit time budget exceeded')))
        signal.alarm(plan['budget']['seconds_per_fit'])
        cfg=RunConfig(**{**job['config'],'output_dir':str(folder/job['arm'])})
        if args.smoke: cfg.max_epochs,cfg.min_epochs,cfg.patience=2,1,2
        result=run(cfg,verbose=False)
        signal.alarm(0)
        if result.get('evaluation_split')!='validation' or 'test_metrics' in result:
            raise ValueError('Unexpected evaluation scope')
        if 'NEVER_SELECTED' in result['training']['flags']:raise ValueError('No checkpoint')
        contract=result['feature_contract']
        for b in ('drug','protein'):
            if contract[b]!=task['expected_features'][b]:raise ValueError('Feature tensor differs from audit')
        for b in ('mu','sd','keep','train_cells'):
            if contract['cell_scaling'][b]!=task['expected_cell'][b]:raise ValueError('Cell scaling differs from audit')
        if result['split']['split_hash']!=job['config']['expected_split_hash']:raise ValueError('Split mismatch')
        path=folder/job['arm']
        results.append({'id':job['id'],'arm':job['arm'],'result':fingerprint(path/'validation_results.json'),
                        'predictions':fingerprint(path/'validation_predictions.csv')})
    verify(args.root)
    save(folder/'complete.json',{'smoke':args.smoke,'task':task['key'],'jobs':results})
    return folder


def report(args):
    plan=verify(args.root);use_repo(plan['dpsyn_repo'])
    pairs=[];missing=[]
    for task in plan['tasks']:
        folder=args.root/'jobs'/task['key']
        if not (folder/'complete.json').exists():missing.append(task['key']);continue
        complete=json.loads((folder/'complete.json').read_text())
        if complete['smoke']:raise ValueError('Smoke cannot enter benchmark report')
        loaded={j['arm']:prediction_rows(args.root,j) for j in complete['jobs']}
        if set(loaded)!=set(ARMS):raise ValueError('Missing or duplicated arms')
        a,ra=loaded['ego']
        for arm in ('dd','ppd','ddppd'):
            b,rb=loaded[arm]
            groups=compare_rows(ra,rb,a['config']['min_candidates'])
            for result,side in ((a,'baseline'),(b,'intervention')):
                for metric in ('wq_mrr','auroc','ap'):
                    expected,actual=result['validation_metrics'][metric],groups[0][side][metric]
                    if expected!=actual and (expected is None or actual is None or abs(expected-actual)>1e-10):
                        raise ValueError('Prediction metric mismatch')
            if a['feature_contract']!=b['feature_contract'] or a['split']!=b['split']:
                raise ValueError('Arms do not share features and splits')
            if task['setup']=='B3A':
                cold=lambda rows:{k:v for k,v in rows.items() if int(v['drug1_train_frequency'])==0 and int(v['drug2_train_frequency'])==0}
                primary=compare_rows(cold(ra),cold(rb),a['config']['min_candidates'])[0]
            else: primary=groups[0]
            flags=sorted(set(a['training']['flags']+b['training']['flags']))
            if any(x['training']['epochs_ran']>=400 for x in (a,b)):flags.append('EPOCH_CAP')
            pairs.append({'setup':task['setup'],'fold':task['fold'],'seed':task['seed'],'arm':arm,
                'primary':primary,'coverage_groups':groups,'flags':flags,
                'parameters':[a['training']['parameter_count'],b['training']['parameter_count']]})
    aggregates=[]
    for setup in SETUPS:
        for arm in ('dd','ppd','ddppd'):
            rows=[r for r in pairs if r['setup']==setup and r['arm']==arm]
            vals=[r['primary']['delta_wq_mrr'] for r in rows if r['primary']['delta_wq_mrr'] is not None]
            clean=[r['primary']['delta_wq_mrr'] for r in rows if not r['flags'] and r['primary']['delta_wq_mrr'] is not None]
            aggregates.append({'setup':setup,'arm':arm,'n':len(vals),'expected_n':12,
                'mean_delta':statistics.mean(vals) if vals else None,'sd':statistics.stdev(vals) if len(vals)>1 else None,
                'unflagged_n':len(clean),'unflagged_mean':statistics.mean(clean) if clean else None})
    save(args.root/'summary.json',{'complete':not missing,'missing_tasks':missing,'pairs':pairs,'aggregates':aggregates})
    lines=['# Full-setting architecture comparison','','Validation only; Δ wq-MRR relative to ego. B3A uses the both-cold validation subset. No causal topology claim.','',
        '| Setting | Arm | Δ wq-MRR | n/12 | Unflagged Δ |','|---|---|---:|---:|---:|']
    fmt=lambda x:'undefined' if x is None else f'{x:+.5f}'
    for r in aggregates:lines.append(f"| {r['setup']} | {r['arm']} | {fmt(r['mean_delta'])} | {r['n']}/12 | {fmt(r['unflagged_mean'])} ({r['unflagged_n']}) |")
    lines+=['',f'Missing tasks: {len(missing)}. See summary.json for exclusions, per-pair metrics and coverage subgroups.']
    (args.root/'report.md').write_text('\n'.join(lines)+'\n')
    if missing:raise ValueError('Incomplete benchmark; partial report saved with missing tasks')
    save(args.root/'complete.json',{'status':'complete','fits':336,'report':fingerprint(args.root/'report.md'),'summary':fingerprint(args.root/'summary.json')})
    return args.root/'report.md'


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','worker','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path)
    p.add_argument('--audit',type=Path);p.add_argument('--index',type=int);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();print({'freeze':freeze,'worker':worker,'report':report}[a.stage](a))

if __name__=='__main__':main()
