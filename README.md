# Military RAG v0.5 — Dockerized database/admin pilot

**Scope:** working pilot for **approved, unclassified** materials in an **isolated test environment**. **NOT** a hardened military production deployment, and **NOT** approved to process classified documents.

## Components
- `db`: PostgreSQL 16 + pgvector + pg_trgm (persistent named volume).
- `redis`: ingestion queue (persistent volume).
- `worker`: PyMuPDF structure-aware parser (adapted from v0.4), SHA-256 cache, embeddings from a local Ollama endpoint.
- `api`: FastAPI web application with GUI for document admin and Greek RAG chat.
- `Ollama`: runs separately on the host or on a dedicated **approved** AI inference node; **not** a Compose service in this pilot, keeping GPU configuration portable.

Each organization should have its **own Docker Compose deployment and its own database/volumes**. Do not share one unsegregated database across organizations. There is no multi-tenant IAM or fine-grained classification/ACL support in this pilot.

## Quick start — Windows PC with Docker Desktop, RTX 3070, Ollama

1. Install Docker Desktop with WSL2 and compatible NVIDIA drivers; keep Ollama running on the Windows host.
2. Download the required models in PowerShell:

   ```powershell
   ollama pull bge-m3
   ollama pull gemma3:4b
   ollama list
   ```

   **Important:** this is a *new* embedding model, `bge-m3` (1024 dimensions). Existing `.npy` embeddings from v0.4 are **not compatible**, and will be regenerated from PDF. Models download over the Internet initially; check software and source policy first.

3. Extract this project outside OneDrive, e.g. `C:\MilitaryRAG-v05`.
4. In PowerShell (project root):

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Replace **all** `CHANGE_ME_...` values with long, distinct passwords. Change `ORG_LABEL`. Default `OLLAMA_URL=http://host.docker.internal:11434` works in many Windows Docker Desktop setups with Ollama on the host. Validate from inside the container if needed.

5. Launch:

   ```powershell
   docker compose up --build -d
   docker compose ps
   ```

6. Admin UI: http://localhost:8080/admin — login with `ADMIN_USER` / `ADMIN_PASSWORD` from `.env`.
7. User chat: http://localhost:8080/ — login with `VIEWER_USER` / `VIEWER_PASSWORD`.
8. Upload a **permitted, unclassified PDF**. Enter a stable document code (e.g. `DK-0-3`), upload, then refresh Admin to monitor the asynchronous worker. The document must show `ready` before a human admin can choose **Activate**. Only the active version is queried by the RAG.
9. To view errors:

   ```powershell
   docker compose logs --tail=100 worker
   docker compose logs --tail=100 api
   ```

10. Stop: `docker compose down` (preserves data). **DO NOT** run `docker compose down -v` unless you intentionally want to remove persistent PostgreSQL, Redis, and PDF volumes.

## How duplicate detection / versions work
- SHA-256 of the **entire PDF** is unique within each organization deployment. Exact duplicate uploads are rejected without additional embedding work.
- The same `doc_key` with a different SHA-256 gets a new immutable `version_number` and new chunks. It does **not** automatically replace an active version.
- Each chunk's embedding is cached by `(embed_model, sha256(model + breadcrumb + text))`; the worker reuses cached vectors for unchanged content.
- A document moves through `queued → processing → ready` (or `failed`). `ready` **is not** an approval of its legal validity; admins manually activate a version.
- Do not change `EMBED_MODEL` or its vector dimensions on an existing database without a deliberate reindex/migration. This schema uses `vector(1024)` and currently expects `bge-m3`.
- OCR for scanned PDFs **is not included**; scanned PDFs that cannot be extracted will fail. Adding secure OCR is a subsequent step.

## Database / container boundaries
All four services are Dockerized; model serving is attached through the `OLLAMA_URL` environment variable and can be swapped later for a suitable inference API adapter. For Linux, networking to host Ollama often requires an approved private interface; avoid exposing Ollama to untrusted networks. Docker Desktop access from containers to `host.docker.internal` should be tested before deployment.

The default exposed web port is deliberately bound to `127.0.0.1:8080` — the pilot is **NOT** reachable from other workstations and should **NOT** be exposed as-is. Do **not** edit the Compose port binding to `0.0.0.0` to enable LAN use. For multi-workstation deployments, first implement: enterprise identity/SSO, TLS reverse proxy, authorization per document, CSRF protection, rate limiting, malware/PDF validation, per-user audit logs, disaster-recovery backups, network/security approval, and capacity testing. Basic Auth is provided **only** for isolated pilot operation.

The application does not establish whether an uploaded legal/military regulation is currently in force. Human validation of version and legal status is required.

## Known limitations / planned v0.6
- No OCR for scanned pages, cross-document version reconciliation, differential changes, advanced review workflow, LDAP/AD/SSO, role-based ACL, hardened browser security, or end-user session isolation.
- Fixed 1024-dimensional vectors and bge-m3. Adapt migration for different embedding provider.
- Initial hybrid retrieval uses cosine vector search + PostgreSQL trigram lexical similarity fused via RRF. Evaluate Greek-language relevance on a gold Q/A set.
- Document-level-only indexing, no automatic multi-document ingestion; can upload sequentially in UI.
- The demo has not undergone real Docker runtime or penetration testing in this build environment; verify on a controlled test PC.

## Useful SQL (run inside DB container)

```powershell
docker compose exec db psql -U rag -d rag -c "SELECT doc_key,version_number,status,is_active,chunk_count FROM documents ORDER BY uploaded_at DESC;"
```

To back up the PostgreSQL database, make a consistent `pg_dump` and independently back up the PDF volume. Test restoration; raw PDFs are stored in the `documents` Docker volume, not inside Postgres.
