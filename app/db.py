import os
import psycopg
from psycopg.rows import dict_row

def connect():
    return psycopg.connect(os.environ['DATABASE_URL'], row_factory=dict_row)

def audit(conn, doc_id, actor, event, detail=''):
    conn.execute('INSERT INTO document_events (document_id,actor,event,detail) VALUES (%s,%s,%s,%s)',
                 (doc_id,actor,event,detail[:1000]))
