import json
from pathlib import Path
import tempfile
import unittest

from graphsmith import dpsyn_research as R
from graphsmith.io import read,write


class Provider:
    def __init__(self,usage=100,missing_usage=False):
        self.contexts=[];self.usage=usage;self.missing_usage=missing_usage
    def call(self,phase,context,schema):
        self.contexts.append((phase,context))
        if phase=='planning':
            recipe=next(iter(context['available_recipes']))
            value={'action':'run','recipe_id':recipe,'hypothesis':'Test information transfer.',
                'mechanism':'Exact relationships may help.','alternatives':['Capacity or prevalence.'],
                'supporting_evidence':['support:B3'],'conflicting_evidence':['conflict:B1'],
                'falsification':'Small paired differences.','scope_limitations':'Validation reused.', 'minimum_effect':.005}
        else:
            value={'decision':'continue','interpretation':'Mixed evidence, exploratory.',
                   'limitations':['Correlated folds.'],'next_question':'Test another explanation.',
                   'evidence_ids':[context['history'][-1]['result']['settings'][0]['id']]}
        receipt={'usage':None if self.missing_usage else {'input_tokens':self.usage,'output_tokens':10},
                 'completed':True,'returncode':0,'disallowed_tool_events':[]}
        return value,receipt


class Backend:
    def __init__(self):self.submissions=0;self.polls=0
    def prepare(self,cycle):return str(cycle['number'])
    def submit(self,study):self.submissions+=1;return {'job_id':study}
    def poll(self,study):
        self.polls+=1
        return {'status':'complete','result':{'status':'verified','all_conditions_present':True,'fits':168,
            'settings':[{'id':f'cycle:{study}:{s}','setup':s,'n':12,'delta_real_minus_control':.002} for s in R.SETUPS]}}


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        review=self.base/'review';review.mkdir();audit=self.base/'audit';audit.mkdir()
        comparisons=[]
        for id in ('support:B3','conflict:B1'):
            comparisons.append({'id':id,'setup':id.split(':')[1],'left':'a','right':'b',
                'delta':{'mean':.01,'sd':.02,'n':12},'unflagged_delta':{'mean':.01,'n':10}})
        write(review/'review.json',{'scope':'Whole project','limitations':['Retrospective.'],
            'research_questions':['Features, graphs, dataset?'],'comparisons':comparisons})
        write(review/'verification.json',{'status':'PASS','review_sha256':R.fingerprint(review/'review.json')['sha256']})
        write(audit/'audit.json',{})
        write(audit/'verification.json',{'status':'PASS','audit_sha256':R.fingerprint(audit/'audit.json')['sha256']})
        self.config={'review':str(review),'feature_audit':str(audit),'project_notes':[],
            'prerequisites':[],'provider':{'name':'codex','model':'','timeout_seconds':180},
            'limits':dict(R.DEFAULT_LIMITS),'recipes':list(R.RECIPES),
            'runtime':{**{k:str(self.base) for k in ('training_repo','training_python','tracker_repo','tracker_python','data_root')},
                       'account':'test','partition':'gpu'}}
        self.root=self.base/'session';R.initialize(self.config,self.root)
    def tearDown(self):self.tmp.cleanup()
    def drive(self,provider=None,backend=None):
        provider=provider or Provider();backend=backend or Backend()
        for _ in range(12):
            state=R.advance(self.root,provider,backend)
            if state['phase'] in R.TERMINAL:break
        return state,provider,backend
    def test_two_adaptive_cycles_four_calls_and_persistent_context(self):
        state,p,b=self.drive()
        self.assertEqual(state['phase'],'stopped');self.assertEqual(state['reason'],'depth_limit')
        self.assertEqual(len(state['calls']),4);self.assertEqual(len(state['cycles']),2)
        self.assertEqual(b.submissions,2);self.assertEqual(state['fits_reserved'],336)
        second=p.contexts[2][1]
        self.assertEqual(len(second['comparisons']),2)
        self.assertIn('critique',second['history'][0])
        self.assertNotIn(state['cycles'][0]['recipe_id'],second['available_recipes'])
        R.advance(self.root,p,b);self.assertEqual(len(p.contexts),4)
    def test_token_overshoot_stops_before_compute(self):
        s,p,b=self.drive(Provider(55000))
        self.assertEqual(len(s['calls']),1);self.assertEqual(b.submissions,0)
        self.assertEqual(s['tokens'],55010);self.assertEqual(s['phase'],'stopped')
    def test_unknown_usage_never_retried(self):
        s,p,b=self.drive(Provider(missing_usage=True))
        self.assertEqual(s['phase'],'uncertain');self.assertEqual(b.submissions,0)
        R.advance(self.root,p,b);self.assertEqual(len(p.contexts),1)
    def test_interrupted_call_is_not_reissued(self):
        state=read(self.root/'state.json');state['phase']='calling';write(self.root/'state.json',state)
        s,p,b=self.drive();self.assertEqual(s['phase'],'uncertain');self.assertFalse(p.contexts)
    def test_unverified_prerequisite_spends_no_calls(self):
        other=self.base/'second';config={**self.config,'prerequisites':[str(self.base/'pending')]}
        R.initialize(config,other);p=Provider();s=R.advance(other,p,Backend())
        self.assertEqual(s['reason'],'waiting_for_verified_prerequisite');self.assertFalse(p.contexts)
    def test_frozen_context_tamper_rejected(self):
        write(self.root/'context.json',{'scope':'changed'})
        with self.assertRaisesRegex(ValueError,'Frozen'):R.advance(self.root,Provider(),Backend())
    def test_unknown_recipe_evidence_and_wrong_types_rejected(self):
        ctx,_=R.evidence_context(self.root,self.config,read(self.root/'state.json'))
        p,_=Provider().call('planning',ctx,{})
        ids={'support:B3','conflict:B1'}
        for field,value in [('recipe_id','arbitrary_shell'),('supporting_evidence',['invented']),('alternatives','wrong type')]:
            with self.assertRaises(ValueError):R.validate_output({**p,field:value},'planning',list(R.RECIPES),ids)
    def test_incomplete_experiment_cannot_reach_critic(self):
        class Bad(Backend):
            def poll(self,study):return {'status':'complete','result':{'status':'verified','fits':1}}
        s,p,b=self.drive(backend=Bad());self.assertEqual(s['phase'],'failed');self.assertEqual(len(p.contexts),1)
    def test_compute_budget_prevents_submission(self):
        root=self.base/'small';config={**self.config,'limits':{**self.config['limits'],'fits':1}}
        R.initialize(config,root);p=Provider();b=Backend();s=R.advance(root,p,b)
        self.assertEqual(s['reason'],'compute_budget');self.assertEqual(b.submissions,0)
    def test_failed_submission_cannot_be_repeated(self):
        class Bad(Backend):
            def submit(self,study):self.submissions+=1;raise TimeoutError('unknown submission')
        s,p,b=self.drive(backend=Bad());self.assertEqual(s['phase'],'uncertain')
        R.advance(self.root,p,b);self.assertEqual(b.submissions,1)

if __name__=='__main__':unittest.main()
