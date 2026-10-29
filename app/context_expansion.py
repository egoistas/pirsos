"""Bounded paragraph reconstruction using existing indexed chunks."""


def expand_group(selected, siblings, max_chars=4500):
    """Prefer the full paragraph; otherwise keep the hit, opener and nearby units.

    Never cut a provision mid-sentence or claim the selected child's breadcrumb
    describes the whole reconstructed paragraph.
    """
    rows = sorted(siblings, key=lambda r: r['ordinal'])
    by_ordinal = {r['ordinal']: r for r in rows}
    by_ordinal[selected['ordinal']] = selected
    rows = sorted(by_ordinal.values(), key=lambda r: r['ordinal'])
    if sum(len(r['body']) for r in rows) + 2 * (len(rows) - 1) <= max_chars:
        chosen = rows
    else:
        chosen = [selected]
        used = len(selected['body'])
        opener = next((r for r in rows if not r.get('subsection')), None)
        candidates = sorted(rows, key=lambda r: (0 if r is opener else 1,
                                                abs(r['ordinal'] - selected['ordinal'])))
        for row in candidates:
            if row['ordinal'] == selected['ordinal']:
                continue
            size = len(row['body']) + 2
            if used + size <= max_chars:
                chosen.append(row)
                used += size
        chosen.sort(key=lambda r: r['ordinal'])
    if len(chosen) == 1:
        breadcrumb = selected['breadcrumb']
    else:
        # The first two levels identify the document and section; the next is §.
        breadcrumb = ' > '.join(selected['breadcrumb'].split(' > ')[:3])
    return {
        'text': '\n\n'.join(r['body'] for r in chosen),
        'page': min(r['page_start'] for r in chosen),
        'page_end': max(r['page_end'] for r in chosen),
        'breadcrumb': breadcrumb,
        'context_complete': len(chosen) == len(rows),
    }
