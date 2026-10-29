"""Read-only comparison: identical question/settings, focused vs retrieved context.

Run inside the API container. No requests are sent outside configured Ollama.
"""
import hashlib
import argparse
import json
from db import connect
from inference import EMBED_MODEL, stream_answer, SIMPLE_PROMPT, CHAT_NUM_CTX, CHAT_NUM_PREDICT
from prompt_context import pack_sources
from retrieval import retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-limits', nargs='+', type=int,
                        help='Compare output limits with sources packed once using CHAT_NUM_PREDICT')
    parser.add_argument('--compare-prompts', action='store_true')
    parser.add_argument('--question', required=True)
    parser.add_argument('--doc-key', required=True)
    parser.add_argument('--section', required=True)
    parser.add_argument('--paragraph', required=True)
    parser.add_argument('--subsection', required=True)
    args = parser.parse_args()
    limits = args.output_limits or [CHAT_NUM_PREDICT]
    if any(n < 1 or n >= CHAT_NUM_CTX for n in limits):
        parser.error('Output limits must be positive and smaller than CHAT_NUM_CTX')
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
        sources, context = pack_sources(args.question, sources)
        prompts = [('current', None)]
        if args.compare_prompts:
            prompts.append(('simple', SIMPLE_PROMPT))
        context_hash = hashlib.sha256(context.encode('utf-8')).hexdigest()
        for prompt_name, prompt in prompts:
            for output_limit in limits:
                diagnostics = {}
                try:
                    answer = ''.join(stream_answer(args.question, context, on_complete=diagnostics.update,
                                                  system_prompt=prompt, num_predict=output_limit))
                    result = {'answer': answer, 'diagnostics': diagnostics}
                except Exception as exc:
                    # Keep the comparison running if one provider request fails.
                    response = getattr(exc, 'response', None)
                    detail = response.text[:2000] if response is not None else str(exc)
                    result = {'error': detail}
                print(json.dumps({'mode': label, 'prompt': prompt_name,
                                  'output_limit': output_limit, 'context_sha256': context_hash,
                                  'sources': sources, **result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
