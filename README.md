# Engram Platform — Inference-as-a-Service

Connect your document base, onboard it once into resident-KV **cartridges**, then chat
with it or expose it to your agents over MCP. Answers as accurate as search-based AI, at a
fraction of the latency and cost, that don't slow down as your library grows.

This repo is the hosted platform. The serving engine consumes the cartridge IP
(`engram-cartridge`) as a pip dependency — that package lives in the sibling
`Engram-Smart-CAG` repo (`cartridges/`).

> **Build plan and product decisions:** [PLAN.md](PLAN.md).

## Services

| Service | Stack | Port | Role |
|---|---|---|---|
| `ml_service` | FastAPI + `cartridges` (torch) | 8001 | GPU/torch process — onboard (one forward pass) + HF inference |
| `ml_service` (serve) | FastAPI + vLLM ≥0.26 | 8002 | Inference Service — serve resident-KV carts by `doc_id` via the cartridge connector |
| `backend` | FastAPI + SQLAlchemy | 8000 | control plane: auth, tenants, corpora, docs, jobs, chat, MCP-query, retrieval |
| `frontend` | Next.js 14 + Tailwind | 3000 | onboarding wizard, chat, document management, MCP panel |
| `mcp_server` | official `mcp` SDK | stdio | per-corpus tool a tenant's agent (Claude…) calls |

Local defaults: **SQLite** (`.data/platform.db`) + **filesystem storage**
(`.data/storage`). Swap to Postgres/S3/vLLM by env — see `.env.example`.

## Quickstart — native (the tested path, no Docker required)

Prereqs: Python 3.11+, Node 18+, and the cartridge IP installed. Until
`engram-cartridge` is published to our package index, install it editable from the sibling
repo:

```bash
pip install -e ../Engram-Smart-CAG        # provides the `cartridges` package + torch/transformers
pip install -r backend/requirements.txt   # control plane (torch-free)
pip install -r ml_service/requirements.txt # GPU plane (pulls engram-cartridge[s3,build])
```

```bash
# 1. ML service. PRODUCTION default is Qwen3-30B-A3B (needs a big GPU); a laptop can't
#    hold it, so override to a small model for local dev:
CARTRIDGES_MODEL=Qwen/Qwen3-0.6B python -m uvicorn app:app --app-dir ml_service --port 8001

# 2. Control-plane API
python -m uvicorn app.main:app --app-dir backend --port 8000

# 3. Frontend
npm --prefix frontend install && npm --prefix frontend run dev
```

Open **http://localhost:3000** → register → create a corpus → upload docs → onboard → chat.

## Environments (local vs AWS)

Config is loaded with python-dotenv (both `backend/app/config.py` and `ml_service/app.py`
load it at startup):

| File | Committed | Role |
|---|---|---|
| `.env.example` | ✅ | template — copy to `.env.local` (or `.env`) and fill in |
| `.env.local` | ❌ gitignored | your machine-local env (small model, SQLite, filesystem, dev secrets) |
| `.env` | ❌ gitignored | personal overrides — beats `.env.local` |
| `.env.aws.example` | ✅ | production template — Qwen3-30B-A3B, Postgres (RDS), S3, real domains, secrets |

**Precedence (low → high): `.env.local` < `.env` < real process env** (docker-compose /
ECS). `load_dotenv` never overrides an already-set variable, so AWS-injected env always
wins. Secrets are never committed — generate local ones with `openssl rand -hex 32`.

## Consuming the cartridge IP (`engram-cartridge`)

The serving engine imports `cartridges` (the KV cartridge format, the vLLM
`CartridgeKVConnector`, the cart store, onboarding, compat/model-binding gates). It is a
**pip dependency**, pinned in `ml_service/requirements.txt` as
`engram-cartridge[s3,build]`. Local dev installs it editable from `../Engram-Smart-CAG`;
production installs the published/pinned package. See PLAN.md → E5 for the package
build/release step.

## Connect an agent via MCP

The corpus page emits a Claude Desktop `mcpServers` config; the server exposes
`query_corpus(question, k)`, authenticated by a per-corpus token.

## Local vs production — one env switch per concern

| Concern | Local default | Production | Switch |
|---|---|---|---|
| Base model | small (0.6B/1.7B) | **Qwen3-30B-A3B** | `CARTRIDGES_MODEL` |
| Database | SQLite | Postgres (RDS/Aurora) | `DATABASE_URL` |
| Object storage | filesystem | S3 (+ working mirror) | `PLATFORM_STORAGE_BACKEND=s3` |
| Jobs | in-process BackgroundTask | RQ on Redis + `app.worker` | `JOB_BACKEND=rq` |
| Auth | local JWT (PBKDF2) + Google | OIDC/JWKS (enterprise SSO) | `AUTH_BACKEND=oidc` |
| Retrieval | TF-IDF / BM25 | fused (BM25+dense+rerank) | `RETRIEVAL_BACKEND=fused` |
| Inference | HF DynamicCache | vLLM resident-KV carts | `INFERENCE_BACKEND=vllm` |

`/health` is liveness; `/ready` checks DB + ML-service reachability. Tests: `pytest`; lint:
`ruff check`.
