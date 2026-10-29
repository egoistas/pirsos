import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))
from context_expansion import expand_group


def row(n, text, subsection='', page=56):
    return dict(ordinal=n, body=text, subsection=subsection, page_start=page,
                page_end=page, breadcrumb='document > Τμήμα 12 > §2 Ενδείξεις > '+subsection)

class ExpansionTests(unittest.TestCase):
    def test_heading_gets_all_children_and_actual_page_span(self):
        rows = [row(0, 'Ενδείξεις προτεραιότητας')]
        names = ['Αστραπιαίο', 'Άμεσο', 'Κατεπείγον', 'Επείγον', 'Κοινό']
        rows += [row(n, name, str(n), 57) for n, name in enumerate(names, 1)]
        expanded = expand_group(rows[0], rows)
        for name in names: self.assertIn(name, expanded['text'])
        self.assertEqual((expanded['page'], expanded['page_end']), (56, 57))
        self.assertTrue(expanded['context_complete'])
        self.assertEqual(expanded['breadcrumb'], 'document > Τμήμα 12 > §2 Ενδείξεις')

    def test_budget_keeps_selected_unit_without_character_truncation(self):
        rows = [row(0, 'header'), row(1, 'x' * 100, 'α'), row(2, 'selected', 'β')]
        expanded = expand_group(rows[2], rows, max_chars=30)
        self.assertEqual(expanded['text'], 'header\n\nselected')
        self.assertFalse(expanded['context_complete'])
        self.assertLessEqual(len(expanded['text']), 30)

    def test_oversize_selected_is_preserved(self):
        selected = row(2, 'x' * 100, 'β')
        self.assertEqual(expand_group(selected, [selected], 10)['text'], selected['body'])

if __name__ == '__main__': unittest.main()
