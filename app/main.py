"""Military RAG v0.5 portable Docker pilot. NOT APPROVED for protected/classified networks."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import secrets
import uuid

import fitz
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from redis import Redis
from rq import Queue
from db import connect, audit
from retrieval import run_rag
from inference import EMBED_MODEL

STORAGE=Path('/storage/documents')
STORAGE.mkdir(parents=True,exist_ok=True)
MAX_UPLOAD = int(os.environ.get('MAX_UPLOAD_MB','40'))*1024*1024
ADMIN_USER=os.environ.get('ADMIN_USER','')
ADMIN_PASSWORD=os.environ.get('ADMIN_PASSWORD','')
VIEWER_USER=os.environ.get('VIEWER_USER','')
VIEWER_PASSWORD=os.environ.get('VIEWER_PASSWORD','')
if not all((ADMIN_USER,ADMIN_PASSWORD,VIEWER_USER,VIEWER_PASSWORD)) or any('CHANGE_ME' in v for v in (ADMIN_PASSWORD,VIEWER_PASSWORD)):
    raise RuntimeError('Configure strong ADMIN_PASSWORD and VIEWER_PASSWORD in .env')

security=HTTPBasic()
app=FastAPI(title='Military RAG v0.5',docs_url=None,redoc_url=None,openapi_url=None)
templates=Jinja2Templates(directory='templates')


def matches(credentials: HTTPBasicCredentials,username,password):
    return secrets.compare_digest(credentials.username.encode(),username.encode()) and secrets.compare_digest(credentials.password.encode(),password.encode())

def require_admin(c: HTTPBasicCredentials=Depends(security)):
    if not matches(c,ADMIN_USER,ADMIN_PASSWORD):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,headers={'WWW-Authenticate':'Basic realm="RAG Administration"'})
    return c.username

def require_viewer(c: HTTPBasicCredentials=Depends(security)):
    if matches(c,ADMIN_USER,ADMIN_PASSWORD) or matches(c,VIEWER_USER,VIEWER_PASSWORD):
        return c.username
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,headers={'WWW-Authenticate':'Basic realm="RAG Pilot"'})

def render(request,name,**data):
    return templates.TemplateResponse(request,name,{'org':os.environ.get('ORG_LABEL','RAG Pilot'),**data})

@app.get('/health')
def health():
    return {'status':'ok'}

@app.get('/',response_class=HTMLResponse)
def chat_home(request:Request,username=Depends(require_viewer)):
    return render(request,'chat.html',answer=None,sources=None,question='',username=username)

@app.post('/ask',response_class=HTMLResponse)
def ask(request:Request,question:str=Form(...),username=Depends(require_viewer)):
    question=question.strip()
    if not question or len(question)>1500: raise HTTPException(400,'Question length must be 1-1500 characters')
    try: result,sources=run_rag(question)
    except Exception as exc: result,sources='Σφάλμα επικοινωνίας με την υπηρεσία AI ή τη βάση. Ελέγξτε Ollama και τα logs.',[]
    return render(request,'chat.html',answer=result,sources=sources,question=question,username=username)

@app.get('/admin',response_class=HTMLResponse)
def admin_home(request:Request,username=Depends(require_admin)):
    with connect() as conn:
        docs=conn.execute('SELECT * FROM documents ORDER BY uploaded_at DESC').fetchall()
        totals=conn.execute("SELECT count(*) AS documents,coalesce(sum(chunk_count),0) AS chunks, count(*) FILTER (WHERE status='processing' OR status='queued') AS queued FROM documents").fetchone()
    return render(request,'admin.html',documents=docs,totals=totals,username=username,embed_model=EMBED_MODEL)

@app.post('/admin/upload')
async def upload_pdf(request:Request,code:str=Form(''),file:UploadFile=File(...),username=Depends(require_admin)):
    filename=Path(file.filename or 'unknown.pdf').name
    if Path(filename).suffix.lower()!='.pdf': raise HTTPException(400,'Only PDF allowed')
    doc_key=code.strip().upper() or Path(filename).stem[:120].upper()
    if not doc_key or len(doc_key)>120: raise HTTPException(400,'Invalid code')
    doc_id=uuid.uuid4()
    destination=STORAGE/f'{doc_id}.pdf'
    sha=hashlib.sha256()
    length=0
    try:
        with destination.open('wb') as out:
            while True:
                block=await file.read(1024*1024)
                if not block: break
                length+=len(block)
                if length>MAX_UPLOAD: raise HTTPException(413,'PDF too large')
                sha.update(block);out.write(block)
        if length<8 or destination.open('rb').read(5)!=b'%PDF-':
            raise HTTPException(400,'File is not a PDF')
        # Parse header for basic validity without executing attachments.
        with fitz.open(destination) as pdf:
            if len(pdf)<1: raise HTTPException(400,'Empty PDF')
        with connect() as conn:
            existing=conn.execute('SELECT id FROM documents WHERE sha256=%s',(sha.hexdigest(),)).fetchone()
            if existing:
                destination.unlink(missing_ok=True)
                return RedirectResponse('/admin?duplicate=1',status_code=303)
            # Serialize new-version numbering for the same document code.
            conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(doc_key,))
            version=conn.execute('SELECT coalesce(max(version_number),0)+1 AS v FROM documents WHERE doc_key=%s',(doc_key,)).fetchone()['v']
            conn.execute('''INSERT INTO documents
                (id,doc_key,version_number,filename,sha256,stored_path,embed_model,pipeline_version)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
                (doc_id,doc_key,version,filename,sha.hexdigest(),str(destination),EMBED_MODEL,'structured-v05.1'))
            audit(conn,doc_id,username,'uploaded',filename)
        Queue('ingestion',connection=Redis.from_url(os.environ['REDIS_URL'])).enqueue(
            'worker.process_document',str(doc_id),job_timeout=3600,result_ttl=300)
        return RedirectResponse('/admin?uploaded=1',status_code=303)
    except Exception:
        # Keep files only after a document row was committed.
        with connect() as conn:
            exists=conn.execute('SELECT 1 FROM documents WHERE id=%s',(doc_id,)).fetchone()
        if not exists: destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

@app.get('/admin/document/{doc_id}',response_class=HTMLResponse)
def document_detail(doc_id:uuid.UUID,request:Request,username=Depends(require_admin)):
    with connect() as conn:
        doc=conn.execute('SELECT * FROM documents WHERE id=%s',(doc_id,)).fetchone()
        if not doc: raise HTTPException(404,'Document not found')
        chunks=conn.execute('SELECT id,ordinal,breadcrumb,page_start,page_end,body FROM chunks WHERE document_id=%s ORDER BY ordinal LIMIT 250',
                            (doc_id,)).fetchall()
        history=conn.execute('SELECT * FROM document_events WHERE document_id=%s ORDER BY happened_at DESC LIMIT 40',(doc_id,)).fetchall()
    return render(request,'document.html',doc=doc,chunks=chunks,history=history)

@app.post('/admin/document/{doc_id}/activate')
def activate(doc_id:uuid.UUID,username=Depends(require_admin)):
    with connect() as conn:
        doc=conn.execute('SELECT * FROM documents WHERE id=%s FOR UPDATE',(doc_id,)).fetchone()
        if not doc: raise HTTPException(404,'Not found')
        if doc['status']!='ready': raise HTTPException(409,'Cannot activate until indexing is ready')
        if doc['embed_model']!=EMBED_MODEL: raise HTTPException(409,'Embedding model mismatch')
        conn.execute('UPDATE documents SET is_active=false WHERE doc_key=%s',(doc['doc_key'],))
        conn.execute('UPDATE documents SET is_active=true WHERE id=%s',(doc_id,))
        audit(conn,doc_id,username,'activated')
    return RedirectResponse('/admin',status_code=303)

@app.post('/admin/document/{doc_id}/deactivate')
def deactivate(doc_id:uuid.UUID,username=Depends(require_admin)):
    with connect() as conn:
        conn.execute('UPDATE documents SET is_active=false WHERE id=%s',(doc_id,))
        audit(conn,doc_id,username,'deactivated')
    return RedirectResponse('/admin',status_code=303)

@app.post('/admin/document/{doc_id}/retry')
def retry(doc_id:uuid.UUID,username=Depends(require_admin)):
    with connect() as conn:
        doc=conn.execute('SELECT status FROM documents WHERE id=%s FOR UPDATE',(doc_id,)).fetchone()
        if not doc: raise HTTPException(404,'Not found')
        if doc['status']!='failed': raise HTTPException(409,'Only failed jobs can be retried')
        conn.execute("UPDATE documents SET status='queued',error_message=NULL WHERE id=%s",(doc_id,))
        audit(conn,doc_id,username,'retried')
    Queue('ingestion',connection=Redis.from_url(os.environ['REDIS_URL'])).enqueue(
        'worker.process_document',str(doc_id),job_timeout=3600,result_ttl=300)
    return RedirectResponse('/admin',status_code=303)

@app.get('/files/{doc_id}')
def pdf_file(doc_id:uuid.UUID,username=Depends(require_viewer)):
    with connect() as conn:
        doc=conn.execute('SELECT stored_path,filename,is_active FROM documents WHERE id=%s',(doc_id,)).fetchone()
    if not doc: raise HTTPException(404,'Document not found')
    if not doc['is_active'] and username!=ADMIN_USER: raise HTTPException(403,'Not published')
    # The UUID-backed storage path is from our own DB, not a client-supplied path.
    return FileResponse(doc['stored_path'],media_type='application/pdf',filename=doc['filename'],content_disposition_type='inline')
