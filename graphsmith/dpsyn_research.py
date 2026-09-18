"""Persistent, budgeted DrugComb plan/execute/verify/critique loop.

Scientific experiments are a trusted catalog, never model-generated commands.
The model token ceiling is a stopping budget: one Codex turn can overshoot it.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import re
from pathlib import Path
import time

from .io import read, write

RECIPES={
 'pd_relationships':{'question':'Do exact PD relationships add beyond degree-preserving PD rewiring?',
  'arms':{'real':{},'rewired':{'rewire':['pd']}},
  'caveats':['Rewiring preserves degree multiplicity but can introduce duplicates.',
             'PPI, DD and root paths remain; this tests PD relationships, not every graph component.']},
 'protein_semantics':{'question':'Do real protein features add beyond random protein identities?',
  'arms':{'real':{},'random_protein':{'prot_feat_mode':'random'}},
  'caveats':['Random and real distributions/effective ranks differ despite matched width and missingness.',
             'This tests representation information within DD+PPD, not graph versus feature-only utility.']},
}
SETUPS=['B1','LCOW','B2','B2P','B3','B3A','B4']
DEFAULT_LIMITS={'cycles':2,'model_calls':4,'tokens':50000,'fits':336,
                'gpu_seconds':302400,'concurrency':3,'fit_seconds':900,'epochs':400}
TERMINAL={'stopped','failed','uncertain'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return {'path':str(path),'sha256':h.hexdigest()}


@contextmanager
def locked(root):
    with (root/'.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another research controller holds the lock') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def compact_review(review):
    records=[];definitions={}
    for c in review['comparisons']:
        definitions.setdefault(c['id'].split(':',1)[0],{'left':c['left'],'right':c['right'],
            'metric':c.get('metric','wq_mrr'),'caveats':c.get('caveats',[]),
            'sources':[Path(p).name for p in c.get('sources',[])]})
        records.append({'id':c['id'],'setup':c['setup'],'left':c['left'],'right':c['right'],
                        'delta':c['delta'],'clean':c['unflagged_delta'],
                        'missing':len(c.get('missing_sources',[]))+len(c.get('absent_from_both',[])),
                        'exclusions':len(c.get('exclusions',[]))})
    return {'scope':review['scope'],'limitations':review['limitations'],
            'questions':review['research_questions'],'contrast_definitions':definitions,'comparisons':records}


def initialize(config,root):
    config=read(config) if isinstance(config,(str,Path)) else config
    expected={'review','feature_audit','project_notes','prerequisites','provider','limits','runtime','recipes'}
    if set(config)!=expected:raise ValueError('Unexpected/missing research configuration fields')
    limits=config['limits']
    if set(limits)!=set(DEFAULT_LIMITS):raise ValueError('Specify every budget field')
    for key,ceiling in DEFAULT_LIMITS.items():
        if type(limits[key]) is not int or not 0<limits[key]<=ceiling:raise ValueError('Invalid budget: '+key)
    if limits['model_calls']>2*limits['cycles']:raise ValueError('At most one planner and one critic call per cycle')
    if config['provider'].get('name')!='codex' or set(config['provider'])!={'name','model','timeout_seconds'}:
        raise ValueError('This loop uses explicit ChatGPT/Codex configuration')
    if not 1<=config['provider']['timeout_seconds']<=180:raise ValueError('Provider timeout exceeds limit')
    if not config['recipes'] or len(set(config['recipes']))!=len(config['recipes']) or set(config['recipes'])-set(RECIPES):
        raise ValueError('Unknown or repeated experiment recipe')
    runtime=config['runtime']
    if set(runtime)!={'training_repo','training_python','tracker_repo','tracker_python','data_root','account','partition'}:
        raise ValueError('Invalid runtime configuration')
    for key in ('training_repo','training_python','tracker_repo','tracker_python','data_root'):
        if not Path(runtime[key]).is_absolute() or not Path(runtime[key]).exists():raise ValueError('Missing absolute runtime path: '+key)
    review_root=Path(config['review']).resolve();review=read(review_root/'review.json')
    check=read(review_root/'verification.json')
    if check['status']!='PASS' or fingerprint(review_root/'review.json')['sha256']!=check['review_sha256']:
        raise ValueError('Whole-project review verification mismatch')
    audit_root=Path(config['feature_audit']).resolve();audit=read(audit_root/'audit.json');check=read(audit_root/'verification.json')
    if check['status']!='PASS' or fingerprint(audit_root/'audit.json')['sha256']!=check['audit_sha256']:
        raise ValueError('Feature/split audit verification mismatch')
    context=compact_review(review)
    notes=[]
    for path in config['project_notes']:
        p=Path(path).resolve();text=p.read_text()
        sections=[x for x in re.split(r'(?m)(?=^#{1,3} )',text) if x.strip()]
        limit=max(100,5000//max(1,len(sections)))
        excerpt='\n'.join(x[:limit] for x in sections) if len(text)>5000 else text
        notes.append({'id':'note:'+p.name,'text':excerpt[:5500],'truncated':len(text)>5000,
                      'excerpt_policy':'bounded excerpts across all sections, not just the first branch','source':fingerprint(p)})
    context['project_notes']=notes
    # Keep the full source review locally; every model call gets a compact whole-project map.
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    write(root/'config.json',config);write(root/'context.json',context);write(root/'review.json',review)
    write(root/'feature_audit.json',audit)
    files=[fingerprint(root/name) for name in ('config.json','context.json','review.json','feature_audit.json')]
    for name in ('dpsyn_research.py','research_provider.py','research_experiments.py'):
        p=Path(__file__).with_name(name)
        if p.exists():files.append(fingerprint(p))
    write(root/'frozen.json',{'files':files})
    state={'phase':'planning','cycles':[],'calls':[],'tokens':0,'fits_reserved':0,
           'gpu_seconds_reserved':0,'prerequisites':None,'reason':None}
    write(root/'state.json',state)
    return root


def verify(root):
    for item in read(root/'frozen.json')['files']:
        if fingerprint(item['path'])!=item:raise ValueError('Frozen research input/source changed: '+item['path'])
    return read(root/'config.json'),read(root/'state.json')


def prerequisites(config,state):
    if state['prerequisites'] is not None:return True
    results=[]
    for directory in config['prerequisites']:
        root=Path(directory)
        if not (root/'complete.json').exists():return False
        complete=read(root/'complete.json')
        if complete.get('status')!='complete':raise ValueError('Invalid prerequisite completion marker')
        for field in ('summary','report'):
            item=complete[field]
            if fingerprint(item['path'])['sha256']!=item['sha256']:raise ValueError('Changed prerequisite '+field)
        summary=read(complete['summary']['path'])
        if not summary.get('complete') or summary.get('missing_tasks'):raise ValueError('Incomplete prerequisite benchmark')
        results.append({'id':'prerequisite:'+root.name,'aggregates':summary['aggregates'],
                        'completion':fingerprint(root/'complete.json'),
                        'scope':'Exploratory validation; B3A both-cold subgroup, not canonical test score.'})
    state['prerequisites']=results
    return True


def schema(phase,recipe_ids):
    string={'type':'string'};strings={'type':'array','items':string}
    if phase=='planning':
        properties={'action':{'type':'string','enum':['run','stop']},
            'recipe_id':{'type':'string','enum':recipe_ids+['none']},
            **{k:string for k in ('hypothesis','mechanism','falsification','scope_limitations')},
            'alternatives':strings,'supporting_evidence':strings,'conflicting_evidence':strings,
            'minimum_effect':{'type':'number'}}
    else:
        properties={'decision':{'type':'string','enum':['continue','stop']},
                    'interpretation':string,'limitations':strings,'next_question':string,'evidence_ids':strings}
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def evidence_context(root,config,state):
    base=read(root/'context.json')
    # Compact table keeps every reviewed setting/comparison rather than retrieving only one branch.
    table=base.pop('comparisons')
    base['comparison_columns']=['id','delta_mean','delta_sd','n','unflagged_mean','unflagged_n','missing','exclusions']
    base['comparisons']=[[r['id'],r['delta']['mean'],r['delta']['sd'],r['delta']['n'],
                          r['clean']['mean'],r['clean']['n'],r['missing'],r['exclusions']] for r in table]
    attempted={c['recipe_id'] for c in state['cycles']}
    base['available_recipes']={k:v for k,v in RECIPES.items() if k in config['recipes'] and k not in attempted}
    base['history']=[{k:c[k] for k in ('recipe_id','plan','result','critique') if k in c} for c in state['cycles']]
    base['prerequisites']=state['prerequisites'] or []; base['budgets']={'limits':config['limits'],'used_tokens':state['tokens'],
        'used_calls':len(state['calls']),'reserved_fits':state['fits_reserved']}
    base['scope']='Each experiment covers all seven settings, four folds and three seeds; 168 fits, fixed features/splits.'
    if len(json.dumps(base).encode())>65000:
        # Preserve quantitative evidence and all branches; bound only narrative context.
        for note in base['project_notes']:
            lines=note['text'].splitlines();note['text']='\n'.join(x[:150] for x in lines)[:3000];note['truncated']=True
    if len(json.dumps(base).encode())>65000:raise ValueError('Project context exceeds 65 KB; curate it before a model call')
    ids={r['id'] for r in table}|{r['id'] for r in base['project_notes']}|{r['id'] for r in base['prerequisites']}
    ids|={r['id'] for c in state['cycles'] for r in c.get('result',{}).get('settings',[])}
    return base,ids


def validate_output(value,phase,available,ids):
    if not isinstance(value,dict) or set(value)!=set(schema(phase,available)['required']):raise ValueError('Invalid model output fields')
    spec=schema(phase,available)['properties']
    for k,v in value.items():
        kind=spec[k]['type']
        if kind=='string' and not isinstance(v,str):raise ValueError('Expected text: '+k)
        if kind=='array' and not isinstance(v,list):raise ValueError('Expected list: '+k)
        if isinstance(v,str) and (not v.strip() or len(v)>4000):raise ValueError('Invalid narrative field: '+k)
        if isinstance(v,list) and (not 1<=len(v)<=12 or any(not isinstance(x,str) or not x.strip() or len(x)>2000 for x in v)):
            raise ValueError('Invalid list field: '+k)
    if phase=='planning':
        if value['action'] not in ('run','stop'):raise ValueError('Invalid planning action')
        if value['action']=='run' and value['recipe_id'] not in available:raise ValueError('Unavailable experiment')
        if value['action']=='stop' and value['recipe_id']!='none':raise ValueError('Stop must name no recipe')
        effect=value['minimum_effect']
        if type(effect) not in (int,float) or not 0<effect<=.1:raise ValueError('Invalid practical effect')
        for k in ('supporting_evidence','conflicting_evidence'):
            if not isinstance(value[k],list) or not value[k] or set(value[k])-ids:raise ValueError('Unrecognized evidence citation')
    else:
        if value['decision'] not in ('continue','stop'):raise ValueError('Invalid critique decision')
        if not isinstance(value['evidence_ids'],list) or not value['evidence_ids'] or set(value['evidence_ids'])-ids:
            raise ValueError('Unrecognized critique evidence')
    return value


def model_call(root,config,state,provider,phase):
    limits=config['limits']
    if len(state['calls'])>=limits['model_calls'] or state['tokens']>=limits['tokens']:
        state.update(phase='stopped',reason='model_budget');return None
    context,ids=evidence_context(root,config,state)
    available=list(context['available_recipes'])
    number=len(state['calls'])+1
    request={'phase':phase,'context':context,'schema':schema(phase,available)}
    write(root/f'call-{number}-input.json',request)
    record={'phase':phase,'status':'reserved','input_hash':digest(request)}
    state['calls'].append(record);state['phase']='calling'
    write(root/'state.json',state)  # durable reservation before any billable request
    try:
        value,receipt=provider.call(phase,context,request['schema'])
        write(root/f'call-{number}-output.json',{'value':value,'receipt':receipt})
        usage=receipt.get('usage') or {}
        if any(type(usage.get(k)) is not int or usage[k]<0 for k in ('input_tokens','output_tokens')):
            raise ValueError('Provider usage missing; no automatic retry')
        used=usage['input_tokens']+usage['output_tokens']  # cached input is already included
        state['tokens']+=used;record.update(status='returned',tokens=used,receipt=receipt)
        if not receipt.get('completed') or receipt.get('returncode') or receipt.get('disallowed_tool_events'):
            raise ValueError('Provider did not complete an allowed tool-free turn')
        value=validate_output(value,phase,available,ids)
        record['status']='accepted'
        return value
    except Exception as exc:
        record.update(status='failed',error=type(exc).__name__+': '+str(exc))
        state.update(phase='uncertain' if 'tokens' not in record else 'failed',reason=record['error'])
        write(root/'state.json',state)
        return None


def advance(root,provider=None,backend=None):
    root=Path(root).resolve()
    with locked(root):
        config,state=verify(root)
        if state['phase'] in TERMINAL:return state
        if state['phase'] in ('calling','submitting'):
            state.update(phase='uncertain',reason='Interrupted external action; reconcile before a new session')
            write(root/'state.json',state);return state
        if provider is None:
            from .research_provider import CodexResearchProvider
            provider=CodexResearchProvider(**{k:v for k,v in config['provider'].items() if k!='name'})
        if backend is None:
            from .research_experiments import ExperimentBackend
            backend=ExperimentBackend(config,root)
        try:
            if not prerequisites(config,state):
                state['reason']='waiting_for_verified_prerequisite';write(root/'state.json',state);return state
            state['reason']=None
            if state['phase']=='planning':
                if len(state['cycles'])>=config['limits']['cycles']:
                    state.update(phase='stopped',reason='depth_limit')
                elif len(state['calls'])+2>config['limits']['model_calls'] or state['tokens']>=config['limits']['tokens']:
                    state.update(phase='stopped',reason='insufficient_plan_and_critique_budget')
                elif len(state['cycles'])==len(config['recipes']):
                    state.update(phase='stopped',reason='catalog_exhausted')
                else:
                    proposal=model_call(root,config,state,provider,'planning')
                    if proposal is not None:
                        if proposal['action']=='stop':state.update(phase='stopped',reason='planner_stop',stop_proposal=proposal)
                        elif state['tokens']+5000>config['limits']['tokens']:
                            state.update(phase='stopped',reason='insufficient_critique_token_headroom',unexecuted_proposal=proposal)
                        else:
                            fits=168;seconds=fits*config['limits']['fit_seconds']
                            if state['fits_reserved']+fits>config['limits']['fits'] or state['gpu_seconds_reserved']+seconds>config['limits']['gpu_seconds']:
                                state.update(phase='stopped',reason='compute_budget',unexecuted_proposal=proposal)
                            else:
                                cycle={'number':len(state['cycles'])+1,'recipe_id':proposal['recipe_id'],'plan':proposal}
                                state['cycles'].append(cycle);state['fits_reserved']+=fits;state['gpu_seconds_reserved']+=seconds
                                state['phase']='preparing';write(root/'state.json',state)
            elif state['phase']=='preparing':
                cycle=state['cycles'][-1]
                cycle['study']=backend.prepare(cycle)
                state['phase']='submitting';write(root/'state.json',state)
                cycle['submission']=backend.submit(cycle['study'])
                state['phase']='waiting'
            elif state['phase']=='waiting':
                cycle=state['cycles'][-1];outcome=backend.poll(cycle['study'])
                cycle['last_status']=outcome['status']
                if outcome['status']=='failed':state.update(phase='failed',reason=outcome.get('reason','experiment_failed'))
                elif outcome['status']=='complete':
                    result=outcome['result']
                    if result.get('status')!='verified' or not result.get('all_conditions_present') or result.get('fits')!=168:
                        raise ValueError('Experiment lacks complete scientific verification')
                    if len(result.get('settings',[]))!=7 or {r['setup'] for r in result['settings']}!=set(SETUPS):
                        raise ValueError('Experiment does not cover all seven settings')
                    cycle['result']=result;write(root/f"cycle-{cycle['number']}-evidence.json",cycle['result'])
                    state['phase']='critiquing'
            elif state['phase']=='critiquing':
                cycle=state['cycles'][-1]
                critique=model_call(root,config,state,provider,'critiquing')
                if critique is not None:
                    cycle['critique']=critique
                    if state['tokens']>=config['limits']['tokens']:state.update(phase='stopped',reason='token_budget')
                    elif critique['decision']=='stop':state.update(phase='stopped',reason='critic_stop')
                    elif len(state['cycles'])>=config['limits']['cycles']:state.update(phase='stopped',reason='depth_limit')
                    else:state['phase']='planning'
        except Exception as exc:
            state.update(phase='uncertain' if state['phase']=='submitting' else 'failed',reason=type(exc).__name__+': '+str(exc))
        write(root/'state.json',state)
        write(root/'project_understanding.json',{'phase':state['phase'],'cycles':state['cycles'],
            'tokens':state['tokens'],'calls':len(state['calls']),'reason':state['reason'],
            'scope':'Model interpretations are exploratory; verified numerical evidence remains authoritative.'})
        return state


def run_loop(root,poll_seconds=30,max_wait_seconds=3600):
    if not 1<=poll_seconds<=60 or not 1<=max_wait_seconds<=172800:raise ValueError('Invalid polling/wall-time limit')
    deadline=time.monotonic()+max_wait_seconds
    while time.monotonic()<deadline:
        state=advance(root)
        print(json.dumps({'phase':state['phase'],'cycles':len(state['cycles']),'calls':len(state['calls']),'tokens':state['tokens'],'reason':state['reason']}),flush=True)
        if state['phase'] in TERMINAL:return state
        if state['phase']=='waiting' or state['reason']=='waiting_for_verified_prerequisite':time.sleep(min(poll_seconds,max(0,deadline-time.monotonic())))
    return read(Path(root)/'state.json')


def add_parser(sub):
    p=sub.add_parser('dpsyn-research',help='Bounded project-wide DrugComb plan/experiment/critique loop')
    p.add_argument('stage',choices=['init','step','run','status'])
    p.add_argument('path',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--poll-seconds',type=int,default=30);p.add_argument('--max-wait-seconds',type=int,default=3600)


def run(args):
    if args.stage=='init':
        if args.output is None:raise ValueError('--output is required for init')
        return str(initialize(args.path,args.output))
    if args.stage=='step':return advance(args.path)
    if args.stage=='run':return run_loop(args.path,args.poll_seconds,args.max_wait_seconds)
    return read(args.path/'state.json')


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);add_parser(sub)
    print(json.dumps(run(p.parse_args()),indent=2))

if __name__=='__main__':main()
