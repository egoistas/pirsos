"""Asynchronous ingestion, invoked by Redis/RQ. The PDF is never sent to the LLM."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import traceback

import fitz
from db import connect, audit
from inference import embed, EMBED_MODEL
from structured_index import extract_units


def vec_literal(values):
    return json.dumps([float(v) for v in values], separators=(',', ':'))


def process_document(document_id: str):
    with connect() as conn:
        doc = conn.execute('SELECT * FROM documents WHERE id=%s FOR UPDATE', (document_id,)).fetchone()
        if not doc:
            return
        if doc['status'] == 'ready':
            return
        conn.execute("UPDATE documents SET status='processing', error_message=NULL, progress_done=0 WHERE id=%s",(document_id,))
        audit(conn,document_id,'worker','processing')
    try:
        pdfpath = Path(doc['stored_path'])
        with fitz.open(pdfpath) as pdf:
            total_pages = len(pdf)
        units = extract_units(pdfpath)
        if not units:
            raise ValueError('Δεν εξήχθη αναγνώσιμο κείμενο. Απαιτείται OCR ή έλεγχος του PDF.')
        with connect() as conn:
            conn.execute('DELETE FROM chunks WHERE document_id=%s',(document_id,))
            conn.execute('UPDATE documents SET pages=%s,progress_total=%s WHERE id=%s',
                         (total_pages,len(units),document_id))

        # Small batches reduce Ollama HTTP overhead dramatically for large PDFs.
        # Reusing exact content+model embeddings avoids recomputing unchanged passages.
        batch_size = 8
        for start in range(0,len(units),batch_size):
            batch = units[start:start+batch_size]
            records=[]
            for offset,unit in enumerate(batch):
                searchable=unit['breadcrumb'] + '\n' + unit['text']
                content_hash=hashlib.sha256((EMBED_MODEL+'\n'+searchable).encode()).hexdigest()
                records.append((start+offset,unit,searchable,content_hash))
            with connect() as conn:
                cached={}
                for _,_,_,content_hash in records:
                    if content_hash not in cached:
                        row=conn.execute('SELECT embedding::text AS vec FROM embedding_cache WHERE embed_model=%s AND content_sha256=%s',
                                         (EMBED_MODEL,content_hash)).fetchone()
                        if row: cached[content_hash]=row['vec']
            pending=[]
            pending_hashes=[]
            for _,_,searchable,content_hash in records:
                if content_hash not in cached and content_hash not in pending_hashes:
                    pending_hashes.append(content_hash)
                    pending.append(searchable)
            if pending:
                vectors=embed(pending)
                for content_hash,vec in zip(pending_hashes,vectors):
                    cached[content_hash]=vec_literal(vec)
            with connect() as conn:
                for ordinal,unit,searchable,content_hash in records:
                    vector=cached[content_hash]
                    conn.execute('INSERT INTO embedding_cache (embed_model,content_sha256,embedding) VALUES (%s,%s,%s::vector) ON CONFLICT DO NOTHING',
                                 (EMBED_MODEL,content_hash,vector))
                    conn.execute("""INSERT INTO chunks
                        (document_id,ordinal,group_key,breadcrumb,section,paragraph,subsection,page_start,page_end,kind,body,search_text,content_sha256,embed_model,embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
                        (document_id,ordinal,unit['group'],unit['breadcrumb'],unit['section'],unit['paragraph'],unit['alpha'],
                         unit['page'],unit['page_end'],unit['kind'],unit['text'],searchable,content_hash,EMBED_MODEL,vector))
                conn.execute('UPDATE documents SET progress_done=%s WHERE id=%s',
                             (start+len(batch),document_id))
        with connect() as conn:
            conn.execute("UPDATE documents SET status='ready',chunk_count=%s,error_message=NULL WHERE id=%s",(len(units),document_id))
            audit(conn,document_id,'worker','ready',f'{len(units)} chunks')
    except Exception as exc:
        with connect() as conn:
            conn.execute('UPDATE documents SET status=%s,is_active=false,error_message=%s WHERE id=%s',
                         ('failed',str(exc)[:900],document_id))
            audit(conn,document_id,'worker','failed',str(exc))
        traceback.print_exc()
        raise
