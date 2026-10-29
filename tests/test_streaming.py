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

    def test_same_prompt_and_parameters_for_both_modes(self):
        normal = inference._payload('q', 'c')
        streaming = inference._payload('q', 'c', True)
        streaming['stream'] = False
        self.assertEqual(normal, streaming)

if __name__ == '__main__': unittest.main()
