import sys
import types
sys.modules.setdefault("requests", types.ModuleType("requests"))
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))
import prompt_context as pc


def source(text):
    return dict(doc_key='TEST', page=1, page_end=2, breadcrumb='§1', text=text)

class PackingTests(unittest.TestCase):
    def test_reserves_output_and_keeps_whole_sources(self):
        sources = [source('α' * 2000), source('β' * 5000), source('γ' * 100)]
        selected, context = pc.pack_sources('Ερώτηση;', sources)
        self.assertIn(sources[0], selected)
        self.assertNotIn(sources[1], selected)
        self.assertIn(sources[2], selected)
        self.assertLessEqual(pc.estimated_prompt_tokens('Ερώτηση;', context),
                             pc.CHAT_NUM_CTX - pc.CHAT_NUM_PREDICT - pc.PROMPT_MARGIN)
        self.assertIn('[ΠΗΓΗ 2]', context)
        self.assertNotIn('[ΠΗΓΗ 3]', context)

    def test_oversize_source_fails_explicitly(self):
        with self.assertRaises(ValueError): pc.pack_sources('q', [source('x' * 100000)])

    def test_empty_sources(self):
        self.assertEqual(pc.pack_sources('q', []), ([], ''))

    def test_headers_are_counted(self):
        small = source('a')
        self.assertGreater(pc.estimated_prompt_tokens('q', pc.format_context([small])),
                           pc.estimated_prompt_tokens('q', 'a'))
