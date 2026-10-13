import json
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
        # Group context reconstruction is deliberately capped for 4096 ctx.
        result=[]
        for row in chosen:
            parent=conn.execute('''SELECT body,ordinal,page_start,page_end
                FROM chunks WHERE document_id=%s AND group_key=%s
                ORDER BY ordinal LIMIT 8''',(row['document_id'],row['group_key'])).fetchall()
            text=row['body']
            opener=next((p for p in parent if p['ordinal']<row['ordinal'] and p['body']!=text),None)
            if opener:
                text=(opener['body'][:450]+'\n'+text)[:1200]
            result.append({'document_id':str(row['document_id']),'filename':row['filename'],
                           'doc_key':row['doc_key'],'page':row['page_start'],
                           'page_end':row['page_end'],'breadcrumb':row['breadcrumb'],
                           'text':text,'score':round(scores[row['id']],5)})
        return result


def run_rag(question):
    sources=retrieve(question)
    if not sources:
        return 'Δεν υπάρχουν ενεργοποιημένοι, έτοιμοι κανονισμοί με το τρέχον embedding model.',[]
    passages=[]
    for i,source in enumerate(sources,1):
        passages.append(f"[ΠΗΓΗ {i}] {source['doc_key']} | PDF σελ. {source['page']}–{source['page_end']} | {source['breadcrumb']}\n{source['text']}")
    result=answer(question,'\n\n'.join(passages))
    return result,sources
