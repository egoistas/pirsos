"""Whole-source prompt packing with an explicit output reserve.

Uses a configurable heuristic, NOT a model tokenizer or a truncation guarantee.
Ollama prompt_eval_count must be monitored on the target model/language.
"""
import math
import os
from inference import _payload, CHAT_NUM_CTX, CHAT_NUM_PREDICT, PROMPT_MARGIN

CHARS_PER_TOKEN = float(os.environ.get('PROMPT_CHARS_PER_TOKEN', '2'))
if not math.isfinite(CHARS_PER_TOKEN) or CHARS_PER_TOKEN <= 0:
    raise ValueError('PROMPT_CHARS_PER_TOKEN must be positive and finite')


def format_context(sources):
    return '\n\n'.join(
        f"[ΠΗΓΗ {i}] {s['doc_key']} | PDF σελ. {s['page']}–{s['page_end']} | {s['breadcrumb']}\n{s['text']}"
        for i, s in enumerate(sources, 1))


def estimated_prompt_tokens(question, context):
    messages = _payload(question, context)['messages']
    return math.ceil(sum(len(m['content']) for m in messages) / CHARS_PER_TOKEN)


def pack_sources(question, sources):
    limit = CHAT_NUM_CTX - CHAT_NUM_PREDICT - PROMPT_MARGIN
    selected = []
    for source in sources:
        candidate = selected + [source]
        if estimated_prompt_tokens(question, format_context(candidate)) <= limit:
            selected = candidate
    if sources and not selected:
        raise ValueError('No complete source fits the prompt budget; increase CHAT_NUM_CTX or reduce context expansion')
    return selected, format_context(selected)
