"""Trusted dpsyn recipes executed through exptrack, with independent prediction review."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from .dpsyn_research import RECIPES,SETUPS,fingerprint
from .io import read,write


class ExperimentBackend:
    def __init__(self,config,root):
        self.config,self.root=config,Path(root)
        self.runtime=config['runtime']

    def tracker(self,*args):
        repo=Path(self.runtime['tracker_repo'])
        entry=repo/'src/exptrack/__main__.py'
        if not entry.is_file():entry=repo/'exptrack/__main__.py'
        cmd=[self.runtime['tracker_python'],str(entry),*map(str,args)]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        if p.returncode:raise ValueError('exptrack failed: '+p.stderr[-2000:])
        return json.loads(p.stdout)

    def prepare(self,cycle):
        cfg=self.config;rt=self.runtime;limits=cfg['limits'];audit=read(self.root/'feature_audit.json')
        directory=self.root/f"cycle-{cycle['number']}"
        directory.mkdir(exist_ok=False)
        recipe=RECIPES[cycle['recipe_id']]
        sources=[fingerprint(p) for p in sorted((Path(rt['training_repo'])/'dpsyn').glob('*.py'))]
        sources += [fingerprint(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
        tracker_root=Path(rt['tracker_repo'])
        tracker_sources=tracker_root/'src/exptrack'
        if not tracker_sources.is_dir():tracker_sources=tracker_root/'exptrack'
        sources += [fingerprint(p) for p in sorted(tracker_sources.glob('*.py'))]
        # Expensive input hashing is checked on workers; controller stores audited hashes.
        inputs=[{'path':x['path'],'sha256':x['sha256']} for x in audit['inputs']]+sources
        jobs=[]
        for setup in SETUPS:
            for fold in range(4):
                for seed in range(3):
                    split_key=f'{setup}_f{fold}_s{seed}'
                    for arm,changes in recipe['arms'].items():
                        config={**audit['base_config'],'setup':setup,'fold_index':fold,'seed':seed,
                          'channels':['dd','ppd'],'feature_protocol':'independent_v1','node_feat_mode':'real_std',
                          'prot_feat_mode':'real_std','feat_transform':'std','missing_feat':'random','rewire':[],
                          'dd_key':'pair','dd_attr':'none','dd_sign':'pos','fuse':'mean','conv':'sage','conv_root':True,
                          'hidden_dim':256,'out_dim':128,'edge_dim':64,'dropout':.3,'max_epochs':limits['epochs'],
                          'min_epochs':min(20,limits['epochs']),'patience':40,'validation_only':True,'floors':[],
                          'expected_split_hash':audit['splits'][split_key]['split_hash'],**changes}
                        expected=next(r for r in audit['treatments'] if r['seed']==seed and r['drug_mode']==config['node_feat_mode'] and r['protein_mode']==config['prot_feat_mode'])
                        jobs.append({'id':split_key+'_'+arm,'setup':setup,'fold':fold,'seed':seed,'arm':arm,
                          'config':config,'expected_features':{b:expected[b] for b in ('drug','protein')},
                          'expected_cell':audit['splits'][split_key]['cell_scaling']})
        plan={'recipe_id':cycle['recipe_id'],'cycle':cycle['number'],'hypothesis':cycle['plan'],
              'jobs':jobs,'sources':sources,'inputs':inputs,'runtime':rt,'fit_seconds':limits['fit_seconds']}
        write(directory/'plan.json',plan)
        plan_file=fingerprint(directory/'plan.json');inputs=inputs+[plan_file]
        tasks=[]
        for job in jobs:
            identity={k:job[k] for k in ('id','setup','fold','seed','arm')}
            tasks.append({'id':job['id'],'identity':identity,'cwd':rt['training_repo'],
              'command':[rt['training_python'],'-m','graphsmith.research_experiments','worker',
                         '--plan',str(directory/'plan.json'),'--task',job['id'],'--output','{output_dir}'],
              'environment':{'PYTHONPATH':str(Path(__file__).parent.parent)+os.pathsep+rt['training_repo'],
                'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                'DPSYN_DATA_ROOT':rt['data_root'],'DPSYN_WORK_ROOT':'{attempt_dir}/work'},
              'inputs':inputs,'outputs':[
                {'path':'validation_results.json','kind':'json',
                 'equals':{'evaluation_split':'validation','split.split_hash':job['config']['expected_split_hash'],
                           'config.feature_protocol':'independent_v1'},
                 'finite':['validation_metrics.wq_mrr','validation_metrics.auroc']},
                {'path':'validation_predictions.csv','kind':'file','sha256_from':{'path':'validation_results.json','field':'predictions_sha256'}},
                {'path':'contract_verified.json','kind':'json','equals':{'status':'PASS','job':job['id']}}]})
        manifest={'version':1,'provenance':{'research_session':str(self.root),'plan':plan_file},
                  'slurm':{'account':rt['account'],'partition':rt['partition'],'gres':'gpu:1',
                           'cpus-per-task':2,'mem':'16G','time':'00:20:00'},'tasks':tasks}
        write(directory/'manifest.json',manifest)
        study=directory/'tracker';self.tracker('init',directory/'manifest.json',study)
        return str(study)

    def submit(self,study):
        return self.tracker('submit',study,'--execute','--concurrency',self.config['limits']['concurrency'])

    def poll(self,study):
        view=self.tracker('status',study)
        directory=Path(study).parent
        write(directory/'inventory.json',view)
        statuses={r['status'] for r in view['tasks']}
        if statuses & {'FAILED','INVALID','MISSING','SUBMISSION_UNKNOWN'}:
            return {'status':'failed','reason':'Experiment acceptance failure; no automatic retries: '+str(view['counts'])}
        if view['warnings'] or statuses!={'COMPLETE'}:return {'status':'pending'}
        output=directory/'evidence.json'
        env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).parent.parent)+os.pathsep+self.runtime['training_repo']
        env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
        p=subprocess.run([self.runtime['training_python'],'-m','graphsmith.research_experiments','summarize',
                          '--plan',str(directory/'plan.json'),'--inventory',str(directory/'inventory.json'),
                          '--output',str(output)],env=env,capture_output=True,text=True,timeout=1800)
        if p.returncode:raise ValueError('Scientific verification failed: '+p.stderr[-2000:])
        return {'status':'complete','result':read(output)}


def worker(args):
    import torch
    plan=read(args.plan)
    if not torch.cuda.is_available():raise ValueError('GPU required')
    sys.path.insert(0,plan['runtime']['training_repo'])
    from dpsyn.config import RunConfig
    from dpsyn.run import run
    job=next(j for j in plan['jobs'] if j['id']==args.task)
    for item in plan['inputs']:
        if fingerprint(item['path'])!=item:raise ValueError('Frozen input/source changed: '+item['path'])
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Fit budget exceeded')))
    signal.alarm(plan['fit_seconds'])
    result=run(RunConfig(**{**job['config'],'output_dir':str(args.output)}),verbose=False)
    signal.alarm(0)
    if result.get('evaluation_split')!='validation' or 'test_metrics' in result:raise ValueError('Wrong evaluation scope')
    if 'NEVER_SELECTED' in result['training']['flags']:raise ValueError('No selected checkpoint')
    for b in ('drug','protein'):
        if result['feature_contract'][b]!=job['expected_features'][b]:raise ValueError('Feature contract mismatch')
    for b in ('mu','sd','keep','train_cells'):
        if result['feature_contract']['cell_scaling'][b]!=job['expected_cell'][b]:raise ValueError('Cell scaling mismatch')
    if result['split']['split_hash']!=job['config']['expected_split_hash']:raise ValueError('Split hash mismatch')
    for item in plan['inputs']:
        if fingerprint(item['path'])!=item:raise ValueError('Input/source changed during run')
    write(args.output/'contract_verified.json',{'status':'PASS','job':job['id']})


def summarize(args):
    plan=read(args.plan);view=read(args.inventory)
    sys.path.insert(0,plan['runtime']['training_repo'])
    from .dpsyn_campaign import prediction_rows,compare_rows
    from .dpsyn import fingerprint as full_fingerprint
    expected={j['id'] for j in plan['jobs']}
    if len(view['tasks'])!=168 or {t['id'] for t in view['tasks']}!=expected or any(t['status']!='COMPLETE' for t in view['tasks']):
        raise ValueError('Incomplete or duplicate task inventory')
    tasks={t['id']:t for t in view['tasks']};arms=list(RECIPES[plan['recipe_id']]['arms'])
    pairs=[]
    for setup in SETUPS:
        for fold in range(4):
            for seed in range(3):
                loaded=[]
                for arm in arms:
                    key=f'{setup}_f{fold}_s{seed}_{arm}';path=Path(tasks[key]['directory'])/'outputs'
                    for item in tasks[key]['evidence']['artifacts']:
                        if fingerprint(item['path'])['sha256']!=item['sha256']:raise ValueError('Accepted artifact changed')
                    result,rows=prediction_rows(args.plan.parent,{'result':full_fingerprint(path/'validation_results.json'),
                                                              'predictions':full_fingerprint(path/'validation_predictions.csv')})
                    job=next(j for j in plan['jobs'] if j['id']==key)
                    if result['split']['split_hash']!=job['config']['expected_split_hash']:raise ValueError('Wrong split')
                    for b in ('drug','protein'):
                        if result['feature_contract'][b]!=job['expected_features'][b]:raise ValueError('Changed feature contract')
                    loaded.append((result,rows))
                (a,ra),(b,rb)=loaded
                if a['split']!=b['split']:raise ValueError('Unmatched splits')
                if a['training']['parameter_count']!=b['training']['parameter_count']:raise ValueError('Control parameter counts differ')
                groups=compare_rows(ra,rb,3)
                for result,side in ((a,'baseline'),(b,'intervention')):
                    for metric in ('wq_mrr','auroc','ap'):
                        x,y=result['validation_metrics'][metric],groups[0][side][metric]
                        if x!=y and (x is None or y is None or abs(x-y)>1e-10):raise ValueError('Metric does not reproduce')
                if setup=='B3A':
                    cold=lambda rows:{k:v for k,v in rows.items() if int(v['drug1_train_frequency'])==0 and int(v['drug2_train_frequency'])==0}
                    primary=compare_rows(cold(ra),cold(rb),3)[0]
                else:primary=groups[0]
                delta=primary['delta_wq_mrr']
                # Real minus control: opposite orientation to generic baseline/intervention helper.
                delta=-delta if delta is not None else None
                flags=sorted(set(a['training']['flags']+b['training']['flags']))
                if any(x['training']['epochs_ran']>=x['config']['max_epochs'] for x in (a,b)):flags.append('EPOCH_CAP')
                pairs.append({'setup':setup,'fold':fold,'seed':seed,'delta':delta,'flags':flags,
                              'primary_rows':primary['rows'],'primary':primary,'coverage_groups':groups})
    settings=[]
    for setup in SETUPS:
        group=[r for r in pairs if r['setup']==setup]
        values=[r['delta'] for r in group if r['delta'] is not None]
        clean=[r['delta'] for r in group if r['delta'] is not None and not r['flags']]
        settings.append({'id':f"cycle:{plan['cycle']}:{setup}",'setup':setup,'n':len(values),'expected_n':12,
            'delta_real_minus_control':statistics.mean(values) if values else None,
            'sd':statistics.stdev(values) if len(values)>1 else None,
            'positive_pairs':sum(v>0 for v in values),'unflagged_n':len(clean),
            'unflagged_delta':statistics.mean(clean) if clean else None})
    write(args.output.with_name('paired_results.json'),pairs)
    write(args.output,{'status':'verified','recipe_id':plan['recipe_id'],'all_conditions_present':True,
        'fits':168,'metric':'validation wq-MRR; B3A both-cold validation subset',
        'direction':'real minus control','settings':settings,'pairs':fingerprint(args.output.with_name('paired_results.json')),
        'caveats':RECIPES[plan['recipe_id']]['caveats']+['Checkpoint selection reuses validation; exploratory evidence only.',
        'Fold/seed SD is descriptive; correlated folds are not independent replicates.']})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','summarize'])
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--task');p.add_argument('--inventory',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    (worker if args.stage=='worker' else summarize)(args)

if __name__=='__main__':main()
