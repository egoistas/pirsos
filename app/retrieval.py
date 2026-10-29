import json
from context_expansion import expand_group
from prompt_context import pack_sources
from db import connect
from inference import EMBED_MODEL, embed, answer


def retrieve(question, top_k=5):
    vector = json.dumps(embed([question])[0],separators=(',',':'))
    with connect() as conn:
        # Two separate retrieval lists. RRF avoids adding incomparable raw distances.
        semantic = conn.execute('''SELECT c.*,d.filename,d.doc_key
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE d.status='ready' AND d.is_active=true AND c.embed_model=%s
            ORDER BY c.embedding <=> %s::vector LIMIT 24''',(EMBED_MODEL,vector)).fetchall()
        lexical = conn.execute('''SELECT c.*,d.filename,d.doc_key
            FROM chunks c JOIN documents d ON d.id=c.document_id
            WHERE d.status='ready' AND d.is_active=true AND c.embed_model=%s
            ORDER BY c.search_text <-> %s LIMIT 24''',(EMBED_MODEL,question)).fetchall()
        by_id={}
        scores={}
        for weight, rows in ((1.0,semantic),(0.9,lexical)):
            for rank, row in enumerate(rows,1):
                ident = row['id']
                by_id[ident]=row
                scores[ident]=scores.get(ident,0) + weight/(60 + rank)
        # Reduce non-normative index/table-of-contents distractions.
        illustrative = any(term in question.casefold() for term in ('υποδειγμ','παραδειγμ','προτυπ','πίνακ','πινακ'))
        for ident,row in by_id.items():
            if row['kind']=='front_matter':
                scores[ident]*=0.55
            elif row['kind']=='example' and not illustrative:
                scores[ident]*=0.78
        chosen=[]
        seen=set()
        # avoid context dominated by adjacent sibling chunks
        for ident in sorted(scores,key=lambda i:scores[i],reverse=True):
            row=by_id[ident]
            group=(row['document_id'],row['group_key'])
            if group in seen: continue
            chosen.append(row)
            seen.add(group)
            if len(chosen)>=top_k: break
        # Expand after group deduplication so lists retain their child provisions.
        groups={}
        for row in chosen:
            groups[row['id']]=conn.execute('''SELECT body,ordinal,page_start,page_end,subsection,breadcrumb
                FROM chunks WHERE document_id=%s AND group_key=%s
                ORDER BY ordinal''',(row['document_id'],row['group_key'])).fetchall()
        remaining=7500-sum(len(row['body']) for row in chosen)
        expansions={}
        # Heading hits need their children most. Keep output order/ranking unchanged.
        for row in sorted(chosen,key=lambda r:bool(r.get('subsection'))):
            budget=max(len(row['body']),min(4500,len(row['body'])+remaining))
            expanded=expand_group(row,groups[row['id']],max_chars=budget)
            remaining-=len(expanded['text'])-len(row['body'])
            expansions[row['id']]=expanded
        result=[]
        for row in chosen:
            result.append({'document_id':str(row['document_id']),'filename':row['filename'],
                           'doc_key':row['doc_key'],**expansions[row['id']],
                           'score':round(scores[row['id']],5)})
        return result


def run_rag(question):
    sources,context=pack_sources(question,retrieve(question))
    if not sources:
        return 'Δεν υπάρχουν ενεργοποιημένοι, έτοιμοι κανονισμοί με το τρέχον embedding model.',[]
    result=answer(question,context)
    return result,sources
