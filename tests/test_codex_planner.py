import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from graphsmith.planning import CodexPlanner, OfflinePlanner
from graphsmith.data import Config


class CodexPlannerTests(unittest.TestCase):
    def setUp(self):
        self.plan, _ = OfflinePlanner().propose({})
        self.calls = []

    def runner(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[1:] == ['login', 'status']:
            return subprocess.CompletedProcess(args, 0, '', 'Logged in using ChatGPT')
        path = Path(args[args.index('--output-last-message') + 1])
        path.write_text(json.dumps(self.plan))
        for event in ({'type': 'thread.started', 'thread_id': 'fixture-thread'},
                      {'type': 'item.completed', 'item': {'type': 'agent_message'}},
                      {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 20}}):
            kwargs['stdout'].write(json.dumps(event) + '\n')
        return subprocess.CompletedProcess(args, 0)

    def test_chatgpt_login_schema_output_and_tool_restrictions(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture-secret', 'CODEX_API_KEY': 'fixture-secret'}):
            plan, receipt = CodexPlanner(executable='codex', runner=self.runner).propose({'profile': {}})
        self.assertEqual(plan, self.plan)
        self.assertEqual(receipt['authentication'], 'chatgpt')
        self.assertEqual(receipt['usage']['output_tokens'], 20)
        args, kwargs = self.calls[1]
        self.assertIn('--ignore-user-config', args)
        self.assertIn('forced_login_method="chatgpt"', args)
        self.assertIn('read-only', args)
        self.assertIn('suppress_unstable_features_warning=true', args)
        self.assertNotIn('--model', args)
        self.assertNotIn('OPENAI_API_KEY', kwargs['env'])
        self.assertNotIn('CODEX_API_KEY', kwargs['env'])
        for feature in CodexPlanner.disabled_features:
            self.assertIn(feature, args)
        self.assertFalse(Path(kwargs['cwd']).exists())

    def test_explicit_model_is_preserved(self):
        CodexPlanner(model='user-chosen-model', executable='codex', runner=self.runner).propose({})
        args = self.calls[1][0]
        self.assertEqual(args[args.index('--model') + 1], 'user-chosen-model')

    def test_api_login_is_rejected_before_planning(self):
        def runner(args, **kwargs):
            self.calls.append(args)
            return subprocess.CompletedProcess(args, 0, '', 'Logged in using an API key')
        with self.assertRaisesRegex(ValueError, 'ChatGPT sign-in'):
            CodexPlanner(executable='codex', runner=runner).propose({})
        self.assertEqual(len(self.calls), 1)

    def test_timeout_is_not_retried(self):
        def runner(args, **kwargs):
            if args[1] == 'login':
                return self.runner(args, **kwargs)
            self.calls.append((args, kwargs))
            raise subprocess.TimeoutExpired(args, 120)
        with self.assertRaisesRegex(ValueError, 'timed out'):
            CodexPlanner(executable='codex', runner=runner).propose({})
        self.assertEqual(len(self.calls), 2)

    def test_tool_event_is_rejected(self):
        def runner(args, **kwargs):
            result = self.runner(args, **kwargs)
            if args[1] == 'exec':
                kwargs['stdout'].write(json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution'}}) + '\n')
            return result
        with self.assertRaisesRegex(ValueError, 'disallowed item'):
            CodexPlanner(executable='codex', runner=runner).propose({})

    def test_nonfatal_diagnostic_is_retained_with_completed_turn(self):
        def runner(args, **kwargs):
            result = self.runner(args, **kwargs)
            if args[1] == 'exec':
                kwargs['stdout'].write(json.dumps({'type': 'item.completed', 'item': {'type': 'error', 'message': 'Code Mode disabled'}}) + '\n')
            return result
        plan, receipt = CodexPlanner(executable='codex', runner=runner).propose({})
        self.assertEqual(plan, self.plan)
        self.assertEqual(receipt['diagnostics'], ['Code Mode disabled'])

    def test_turn_failure_is_rejected_even_with_output_file(self):
        def runner(args, **kwargs):
            result = self.runner(args, **kwargs)
            if args[1] == 'exec':
                kwargs['stdout'].write(json.dumps({'type': 'turn.failed'}) + '\n')
            return result
        with self.assertRaisesRegex(ValueError, 'turn failed'):
            CodexPlanner(executable='codex', runner=runner).propose({})

    def test_codex_requires_a_planning_turn_budget(self):
        with self.assertRaisesRegex(ValueError, 'declared planning turn'):
            Config.parse({'topology_visibility': 'independent_background_transductive', 'planner': 'codex'})
        config = Config.parse({'topology_visibility': 'independent_background_transductive', 'planner': 'codex', 'max_llm_calls': 1})
        self.assertEqual(config.planner_model, '')
