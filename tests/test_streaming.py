import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault('requests', types.ModuleType('requests'))
spec = importlib.util.spec_from_file_location('inference', Path(__file__).parents[1] / 'app/inference.py')
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)

class Response:
    def __init__(self, events): self.events = events; self.closed = False
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def raise_for_status(self): pass
    def iter_lines(self):
        for event in self.events:
            yield json.dumps(event, ensure_ascii=False).encode()

class StreamingTests(unittest.TestCase):
    def run_stream(self, events):
        response = Response(events)
        return response, patch.object(inference.requests, 'post', return_value=response, create=True)

    def test_unicode_tokens_and_final_token(self):
        response, mock = self.run_stream([
            {'message': {'content': 'Ναι, '}},
            {'message': {'content': 'ισχύει.'}, 'done': True}])
        with mock as post:
            self.assertEqual(''.join(inference.stream_answer('q', 'c')), 'Ναι, ισχύει.')
            self.assertTrue(post.call_args.kwargs['stream'])
            self.assertTrue(post.call_args.kwargs['json']['stream'])
        self.assertTrue(response.closed)

    def test_error_and_premature_end_close_response(self):
        for events in ([{'error': 'model unavailable'}], [{'message': {'content': 'partial'}}]):
            response, mock = self.run_stream(events)
            with mock, self.assertRaises(RuntimeError):
                list(inference.stream_answer('q', 'c'))
            self.assertTrue(response.closed)

    def test_consumer_cancellation_closes_response(self):
        response, mock = self.run_stream([{'message': {'content': 'a'}}, {'done': True}])
        with mock:
            stream = inference.stream_answer('q', 'c')
            self.assertEqual(next(stream), 'a')
            stream.close()
        self.assertTrue(response.closed)

    def test_completion_callback_reports_output_limit(self):
        response, mock = self.run_stream([{'message': {'content': 'partial'}, 'done': True,
                                          'done_reason': 'length', 'eval_count': 500,
                                          'prompt_eval_count': 3500}])
        diagnostics = {}
        with mock:
            self.assertEqual(''.join(inference.stream_answer('q', 'c', diagnostics.update)), 'partial')
        self.assertTrue(diagnostics['output_limit_reached'])
        self.assertEqual(diagnostics['prompt_eval_count'], 3500)
        self.assertEqual(diagnostics['prompt_truncation'], 'unknown')

    def test_normal_stop_and_missing_reason_at_limit(self):
        with self.assertLogs('inference', level='WARNING'):
            normal = inference.completion_diagnostics({'done_reason': 'stop', 'eval_count': 20}, 'q', 'c')
            limited = inference.completion_diagnostics({'eval_count': 500}, 'q', 'c')
        self.assertFalse(normal['output_limit_reached'])
        self.assertTrue(limited['output_limit_reached'])

    def test_output_override_changes_only_generation_limit(self):
        baseline = inference._payload('q', 'context')
        extended = inference._payload('q', 'context', num_predict=1000)
        self.assertEqual(baseline['messages'], extended['messages'])
        self.assertEqual(extended['options']['num_predict'], 1000)
        response, mock = self.run_stream([{'done': True, 'done_reason': 'stop', 'eval_count': 600}])
        diagnostics = {}
        with mock as post:
            list(inference.stream_answer('q', 'context', diagnostics.update, num_predict=1000))
            self.assertEqual(post.call_args.kwargs['json']['options']['num_predict'], 1000)
        self.assertEqual(diagnostics['num_predict'], 1000)
        self.assertFalse(diagnostics['output_limit_reached'])

    def test_same_prompt_and_parameters_for_both_modes(self):
        normal = inference._payload('q', 'c')
        streaming = inference._payload('q', 'c', True)
        streaming['stream'] = False
        self.assertEqual(normal, streaming)

if __name__ == '__main__': unittest.main()
