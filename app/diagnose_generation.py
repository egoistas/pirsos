"""Read-only comparison: identical question/settings, focused vs retrieved context.

Run inside the API container. No requests are sent outside configured Ollama.
"""
import argparse
import json
from db import connect
from inference import EMBED_MODEL, stream_answer
from retrieval import retrieve


def format_context(sources):
    return '\n\n'.join(
        f"[ΠΗΓΗ {i}] {s['doc_key']} | PDF σελ. {s['page']}–{s['page_end']} | {s['breadcrumb']}\n{s['text']}"
        for i, s in enumerate(sources, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--doc-key', required=True)
    parser.add_argument('--section', required=True)
    parser.add_argument('--paragraph', required=True)
    parser.add_argument('--subsection', required=True)
    args = parser.parse_args()
    full = retrieve(args.question)
    with connect() as conn:
        rows = conn.execute('''SELECT c.*,d.doc_key FROM chunks c
            JOIN documents d ON d.id=c.document_id
            WHERE d.doc_key=%s AND d.is_active=true AND d.status='ready'
              AND c.embed_model=%s AND c.section=%s AND c.paragraph=%s AND c.subsection=%s
            ORDER BY c.ordinal''', (args.doc_key, EMBED_MODEL, args.section,
                                    args.paragraph, args.subsection)).fetchall()
    if not rows:
        parser.error('No matching indexed provision; check the document key and structural fields.')
    focused = [dict(doc_key=r['doc_key'], page=r['page_start'], page_end=r['page_end'],
                    breadcrumb=r['breadcrumb'], text=r['body']) for r in rows]
    for label, sources in [('focused', focused), ('retrieved', full)]:
        if not sources:
            print(json.dumps({'mode': label, 'error': 'No retrieved sources'}, ensure_ascii=False))
            continue
        context = format_context(sources)
        diagnostics = {}
        answer = ''.join(stream_answer(args.question, context, on_complete=diagnostics.update))
        print(json.dumps({'mode': label, 'answer': answer, 'diagnostics': diagnostics,
                          'sources': sources}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
