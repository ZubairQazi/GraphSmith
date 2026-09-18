import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from graphsmith.research_provider import CodexResearchProvider

class ProviderTests(unittest.TestCase):
    def test_tool_free_turn_and_usage_receipt(self):
        calls=[]
        def runner(args,**kw):
            calls.append((args,kw))
            if args[1:3]==['login','status']:
                return SimpleNamespace(returncode=0,stdout='Logged in using ChatGPT',stderr='')
            Path(args[args.index('--output-last-message')+1]).write_text('{"decision":"stop"}')
            kw['stdout'].write(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}})+'\n')
            return SimpleNamespace(returncode=0)
        with patch('shutil.which',return_value='/fake/codex'):
            provider=CodexResearchProvider(runner=runner)
            value,receipt=provider.call('critiquing',{'scope':'all seven'}, {'type':'object'})
        self.assertEqual(value,{'decision':'stop'});self.assertEqual(receipt['usage']['input_tokens'],100)
        args,kw=calls[-1]
        self.assertIn('--ephemeral',args);self.assertIn('multi_agent',args)
        self.assertIn('Use ONLY the supplied project-wide evidence',kw['input'])
        self.assertNotIn('OPENAI_API_KEY',kw['env']);self.assertNotIn('CODEX_API_KEY',kw['env'])
    def test_tool_event_preserves_metering_for_controller_rejection(self):
        def runner(args,**kw):
            if args[1:3]==['login','status']:return SimpleNamespace(returncode=0,stdout='ChatGPT',stderr='')
            kw['stdout'].write(json.dumps({'type':'item.completed','item':{'type':'command_execution'}})+'\n')
            kw['stdout'].write(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':2}})+'\n')
            return SimpleNamespace(returncode=0)
        with patch('shutil.which',return_value='/fake/codex'):
            value,receipt=CodexResearchProvider(runner=runner).call('planning',{}, {})
        self.assertIn('command_execution',receipt['disallowed_tool_events']);self.assertIsNone(value)
        self.assertEqual(receipt['usage']['output_tokens'],2)

if __name__=='__main__':unittest.main()
