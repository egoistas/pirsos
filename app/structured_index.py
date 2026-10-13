"""Structure-aware extraction of Greek regulation PDFs (proof-of-concept).

Keeps page spans while avoiding splitting every passage at PDF page boundaries.
Segments at ΤΜΗΜΑ / numbered paragraphs / alphabetic subdivisions / (n) items.
Falls back to page-level paragraphs for differently structured documents.
"""
from __future__ import annotations
import re
from pathlib import Path
import fitz

PAGE_NO = re.compile(r'^\s*-\s*\d{1,4}\s*-\s*$')
SECTION = re.compile(r'^\s*ΤΜΗΜΑ\s+(\d{1,3})\s*$', re.I)
MAIN = re.compile(r'^\s*(\d{1,3})\.\s+(.+)$')
ALPHA = re.compile(r'^\s*([α-ω])\.\s+(.+)$', re.I)
ITEM = re.compile(r'^\s*\((\d{1,3})\)\s+(.+)$')
TITLE = re.compile(r'^[Α-ΩΆΈΉΊΌΎΏΪΫ\s\-–(),/]+$')


def _append(acc: list[str], line: str) -> None:
    line = line.strip()
    if not line:
        return
    if acc and acc[-1].endswith(('\u00ad', '-')) and re.match(r'^[α-ωάέήίόύώϊϋΐΰ]', line, re.I):
        acc[-1] = acc[-1].rstrip('\u00ad-') + line
    else:
        acc.append(line)


def extract_units(pdf_path: Path) -> list[dict]:
    """Create atomic logical units with structural metadata and source page spans."""
    pdf_path = Path(pdf_path)
    pdf = fitz.open(pdf_path)
    pages = [p.get_text('text', sort=True) for p in pdf]
    body_start = next((i for i, t in enumerate(pages)
                       if re.search(r'^\s*ΤΜΗΜΑ\s+1\s*$', t, re.M)
                       and len(t) > 600), 0)
    examples_start = next((i for i in range(body_start + 1, len(pages))
                           if re.search(r'^\s*ΥΠΟΔΕΙΓΜΑΤΑ\s*$', pages[i], re.M)), None)

    units = []
    section = ''
    section_title = ''
    paragraph = ''
    paragraph_title = ''
    alpha = ''
    item = ''
    unit_text = []
    start_page = None
    end_page = None
    kind = 'body'
    pending_section_title = False

    def flush():
        nonlocal unit_text, start_page, end_page
        if not unit_text:
            return
        text = re.sub(r'\s+', ' ', ' '.join(unit_text)).replace('\u00ad', '').strip()
        unit_text = []
        if len(text) < 25:
            return
        # IDs keep the parent paragraph together despite page breaks.
        section_id = section or 'unstructured'
        paragraph_id = paragraph or f'p{start_page}'
        group = f'{pdf_path.name}::{section_id}::{paragraph_id}'
        breadcrumb = ' > '.join(x for x in [
            pdf_path.stem,
            f'Τμήμα {section}' + (f' — {section_title}' if section_title else '') if section else '',
            f'§{paragraph} {paragraph_title}' if paragraph else '',
            f'{alpha}.' if alpha else '',
            f'({item})' if item else ''
        ] if x)
        units.append(dict(file=pdf_path.name, text=text,
                          breadcrumb=breadcrumb, section=section,
                          section_title=section_title, paragraph=paragraph,
                          paragraph_title=paragraph_title, alpha=alpha, item=item,
                          group=group, kind=kind,
                          page=start_page, page_end=end_page,
                          ordinal=len(units)))

    for page_idx, page in enumerate(pages):
        page_n = page_idx + 1
        # Non-normative material is split by page rather than treated as one giant passage.
        if page_idx < body_start or (examples_start is not None and page_idx >= examples_start):
            flush()
        if page_idx < body_start:
            page_kind = 'front_matter'
        elif examples_start is not None and page_idx >= examples_start:
            page_kind = 'example'
        else:
            page_kind = 'body'
        for line in page.splitlines():
            if not line.strip() or PAGE_NO.match(line):
                continue
            m = SECTION.match(line)
            if m and page_kind == 'body':
                flush()
                section, section_title = m.group(1), ''
                paragraph = paragraph_title = alpha = item = ''
                pending_section_title = True
                continue
            if pending_section_title and page_kind == 'body' and TITLE.match(line.strip()):
                section_title = (section_title + ' ' + line.strip()).strip()
                continue
            pending_section_title = False
            m = MAIN.match(line)
            if m and page_kind == 'body':
                flush()
                paragraph, paragraph_title = m.group(1), m.group(2).strip()[:100]
                alpha = item = ''
            else:
                m = ALPHA.match(line) if page_kind == 'body' else None
                if m:
                    flush()
                    alpha, item = m.group(1), ''
                else:
                    m = ITEM.match(line) if page_kind == 'body' else None
                    if m:
                        flush()
                        item = m.group(1)
            if kind != page_kind:
                flush()
                kind = page_kind
            if start_page is None or not unit_text:
                start_page = page_n
            end_page = page_n
            _append(unit_text, line)
    flush()
    pdf.close()
    # Very long units split at sentence breaks, retaining structural breadcrumbs.
    result = []
    for u in units:
        if len(u['text']) <= 1800:
            result.append(u)
            continue
        pieces = re.split(r'(?<=[.;])\s+(?=[Α-ΩΆΈΉΊΌΎΏΪΫα-ωάέήίόύώ\d(])', u['text'])
        partial = ''
        for piece in pieces:
            if partial and len(partial) + len(piece) > 1700:
                v = dict(u, text=partial.strip(), ordinal=len(result))
                result.append(v)
                partial = ''
            partial += (' ' if partial else '') + piece
        if partial:
            result.append(dict(u, text=partial.strip(), ordinal=len(result)))
    return result


def expand_context(unit: dict, group_items: list[dict], max_chars: int = 2400) -> dict:
    """Selected child + authoritative parent paragraph opener + siblings in same subsection."""
    same_group = sorted(group_items, key=lambda u: u['ordinal'])
    parent = next((u for u in same_group if not u['alpha'] and not u['item']), None)
    chosen = [unit]
    if parent and parent['ordinal'] != unit['ordinal']:
        chosen.insert(0, parent)
    # Siblings in the same alphabetic subdivision add context without including all unrelated ones.
    for u in same_group:
        if u['ordinal'] in {x['ordinal'] for x in chosen}:
            continue
        if unit['alpha'] and u['alpha'] != unit['alpha']:
            continue
        if sum(len(x['text']) for x in chosen) + len(u['text']) < max_chars:
            chosen.append(u)
    chosen = sorted(chosen, key=lambda u: u['ordinal'])
    # Keep the queried child even if it is longer than the budget.
    text = '\n\n'.join(u['text'] for u in chosen)
    if len(text) > max_chars and len(chosen) > 1:
        chosen = [x for x in chosen if x['ordinal'] == unit['ordinal'] or (parent and x['ordinal'] == parent['ordinal'])]
        text = '\n\n'.join(u['text'] for u in chosen)
    return dict(unit, text=text, page=min(x['page'] for x in chosen),
                page_end=max(x['page_end'] for x in chosen),
                breadcrumb=unit['breadcrumb'])
