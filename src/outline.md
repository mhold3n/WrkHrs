# Build & Deploy Plan: Laptop → Desktop (GPU) for Your Full Stack

Below is a **concise, high-detail roadmap** you can execute end-to-end. It lets you **develop on a laptop** (CPU-only OK) and **deploy on a desktop** (with GPUs) without breaking your IDE (e.g., Cursor). It includes **middleware + model + orchestration + MCP + RAG + multimodal + plugin auto-discovery**.

---

## 1) Target Architecture (at a glance)

```
[Cursor/IDE] → [gateway-api] → [orchestrator] → (routes to)
                           ├─► [tool-registry] (Pluggy)
                           ├─► [mcp-chem/mech/...]
                           ├─► [rag-api] ↔ [qdrant] (vector DB)
                           ├─► [asr-api] (audio/video → text/segments)
                           └─► [llm-runner] (Ollama or vLLM)
                                           ▲
                                 (GPU on desktop; CPU on laptop)
```

* **gateway-api**: FastAPI service; non-generative conditioning (weights/constraints/units), auth, rate limits.
* **orchestrator**: LangGraph graph; decides which tools to call, in what order.
* **tool-registry**: Pluggy-based dynamic tool discovery (CLI/SDK tools auto-register).
* **mcp-\***: Multi-Context Protocol servers per domain (chem, mech, materials).
* **rag-api**: Haystack pipeline front-end; **qdrant** as vector store.
* **asr-api**: Whisper/faster-whisper wrapper for audio/video ingestion.
* **llm-runner**: Ollama (simple) or vLLM (high-throughput/OpenAI-compatible).

---

## 2) Repository Scaffold (single mono-repo)

```
ai-stack/
  docker/
    gateway/Dockerfile
    orchestrator/Dockerfile
    mcp/Dockerfile
    rag/Dockerfile
    asr/Dockerfile
    tool-registry/Dockerfile
  services/
    gateway/app.py                  # FastAPI; domain-weighting, constraints, SI units
    orchestrator/app.py             # LangGraph graph definition
    mcp/                            # MCP servers (config-driven: chem/mech/materials)
    rag/app.py                      # Haystack REST; connects to Qdrant
    asr/app.py                      # faster-whisper/ffmpeg wrapper
    tool_registry/app.py            # Pluggy plugin server (auto-discovers tools)
    plugins/                        # drop-in CLI/tool plugins (auto-registered)
  models/                           # (optional) local models/weights mount
  compose/
    docker-compose.base.yml
    docker-compose.dev.yml
    docker-compose.prod.yml
  .env.example
  Makefile
  README.md
```

---

## 3) Environment & Networking

* **Single user-defined network**: `llm_net`
* **Shared volumes** for persistence:

  * `qdrant_data`, `rag_cache`, `models`, `plugins`, `mcp_data`, `logs`, `gateway_state`
* **.env** controls ports, model names, and GPU flags.

**`.env.example`**

```
# Ports
GATEWAY_PORT=8080
ORCH_PORT=8081
RAG_PORT=8082
QDRANT_PORT=6333
ASR_PORT=8084
MCP_PORT=8085
TOOLS_PORT=8086

# LLM
LLM_BACKEND=ollama          # or vllm
OLLAMA_MODEL=llama3:8b-instruct
VLLM_MODEL=/models/Mistral-7B-Instruct

# GPU (prod only)
ENABLE_GPU=false

# RAG
QDRANT_URL=http://qdrant:6333
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ASR
ASR_MODEL=medium
```

---

## 4) Base Compose (common to dev & prod)

**`compose/docker-compose.base.yml`**

```yaml
version: "3.9"
name: ai-stack
networks: { llm_net: {} }
volumes:
  qdrant_data: {}
  rag_cache: {}
  models: {}
  plugins: {}
  mcp_data: {}
  logs: {}
  gateway_state: {}

services:
  gateway-api:
    build: { context: .., dockerfile: docker/gateway/Dockerfile }
    env_file: [ ../.env ]
    networks: [ llm_net ]
    ports: [ "${GATEWAY_PORT}:8000" ]
    depends_on: [ orchestrator ]
    volumes: [ "gateway_state:/state", "logs:/logs" ]
    healthcheck: { test: ["CMD","curl","-f","http://localhost:8000/health"], interval: 10s, timeout: 3s, retries: 10 }

  orchestrator:
    build: { context: .., dockerfile: docker/orchestrator/Dockerfile }
    env_file: [ ../.env ]
    networks: [ llm_net ]
    ports: [ "${ORCH_PORT}:8000" ]
    depends_on: [ tool-registry, rag-api, asr-api, llm-runner ]
    healthcheck: { test: ["CMD","curl","-f","http://localhost:8000/health"], interval: 10s, timeout: 3s, retries: 10 }

  tool-registry:
    build: { context: .., dockerfile: docker/tool-registry/Dockerfile }
    env_file: [ ../.env ]
    networks: [ llm_net ]
    ports: [ "${TOOLS_PORT}:8000" ]
    volumes: [ "plugins:/plugins:rw" ]
    healthcheck: { test: ["CMD","curl","-f","http://localhost:8000/health"], interval: 10s, timeout: 3s, retries: 10 }

  mcp:
    build: { context: .., dockerfile: docker/mcp/Dockerfile }
    env_file: [ ../.env ]
    networks: [ llm_net ]
    ports: [ "${MCP_PORT}:8000" ]
    volumes: [ "mcp_data:/data:rw" ]
    healthcheck: { test: ["CMD","curl","-f","http://localhost:8000/health"], interval: 10s, timeout: 3s, retries: 10 }

  qdrant:
    image: qdrant/qdrant:latest
    networks: [ llm_net ]
    ports: [ "${QDRANT_PORT}:6333" ]
    volumes: [ "qdrant_data:/qdrant/storage" ]
    healthcheck: { test: ["CMD","wget","-qO-","http://localhost:6333/ready"], interval: 10s, timeout: 3s, retries: 10 }

  rag-api:
    build: { context: .., dockerfile: docker/rag/Dockerfile }
    env_file: [ ../.env ]
    networks: [ llm_net ]
    ports: [ "${RAG_PORT}:8000" ]
    depends_on: [ qdrant ]
    volumes: [ "rag_cache:/cache" ]
    healthcheck: { test: ["CMD","curl","-f","http://localhost:8000/health"], interval: 10s, timeout: 3s, retries: 10 }

  asr-api:
    build: { context: .., dockerfile: docker/asr/Dockerfile }
    env_file: [ ../.env ]
    networks: [ llm_net ]
    ports: [ "${ASR_PORT}:8000" ]
    healthcheck: { test: ["CMD","curl","-f","http://localhost:8000/health"], interval: 10s, timeout: 3s, retries: 10 }

  llm-runner:
    image: ${LLM_BACKEND:-ollama} == "ollama" ? "ollama/ollama:latest" : "vllm/vllm-openai:latest"
    networks: [ llm_net ]
    # Ports only for OpenAI-compatible APIs (vLLM); Ollama uses 11434
    ports:
      - "${LLM_BACKEND:-ollama} == 'vllm' ? '8001:8000' : '11434:11434'"
    volumes:
      - "models:/models"
    environment:
      - OLLAMA_KEEP_ALIVE=24h
    healthcheck:
      test: ["CMD","sh","-c","(curl -sf http://localhost:11434/api/tags || curl -sf http://localhost:8000/health)"]
      interval: 15s
      timeout: 5s
      retries: 20
```

> **Note:** The ternary in `image/ports` above is illustrative; Compose doesn’t parse ternaries. In practice create **two** service variants (see dev/prod overrides below).

---

## 5) Dev Override (CPU on Laptop)

**`compose/docker-compose.dev.yml`**

```yaml
services:
  llm-runner:
    image: ollama/ollama:latest
    ports: [ "11434:11434" ]
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_KEEP_ALIVE=24h
    # CPU only; no device requests
    command: [ "serve" ]

  rag-api:
    environment:
      - QDRANT_URL=${QDRANT_URL}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL}

  asr-api:
    environment:
      - ASR_MODEL=${ASR_MODEL}
      - ASR_DEVICE=cpu
```

**Dev steps (laptop):**

```bash
cp .env.example .env
# Choose CPU-friendly model in .env (e.g., OLLAMA_MODEL=llama3:8b-instruct-q4)
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.dev.yml build
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.dev.yml up -d
# Load model (once):
curl http://localhost:11434/api/pull -d '{"name":"'"$OLLAMA_MODEL"'"}'
# Sanity checks:
curl http://localhost:${GATEWAY_PORT}/health
curl http://localhost:${ORCH_PORT}/health
```

---

## 6) Prod Override (GPU on Desktop)

**Prereqs (desktop):**

* Latest **NVIDIA driver**
* **NVIDIA Container Toolkit** installed
* Adequate **vRAM** for your chosen model(s)

**`compose/docker-compose.prod.yml`**

```yaml
services:
  llm-runner:
    image: vllm/vllm-openai:latest
    ports: [ "8001:8000" ]  # OpenAI-compatible
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    device_requests:
      - driver: nvidia
        count: all
        capabilities: [gpu]
    volumes:
      - "models:/models"
    command: [
      "python",
      "-m", "vllm.entrypoints.openai.api_server",
      "--model", "${VLLM_MODEL}",
      "--trust-remote-code"
    ]

  asr-api:
    environment:
      - ASR_MODEL=${ASR_MODEL}
      - ASR_DEVICE=cuda
    device_requests:
      - driver: nvidia
        count: 1
        capabilities: [gpu]
```

**Prod steps (desktop):**

```bash
# Copy images from laptop OR pull from your registry (see §7)
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.prod.yml up -d
# Confirm:
curl http://localhost:${GATEWAY_PORT}/health
curl http://localhost:8001/v1/models
```

---

## 7) Moving Images Laptop → Desktop

**Option A: Local Registry (recommended)**

```bash
# On laptop
docker buildx create --use
docker buildx build -f docker/gateway/Dockerfile -t ghcr.io/<you>/gateway:dev --push .
# Repeat for each service...
# On desktop
docker pull ghcr.io/<you>/gateway:dev
```

**Option B: Tarball**

```bash
# On laptop
docker save ghcr.io/<you>/gateway:dev | gzip > gateway_dev.tar.gz
# Transfer file, then on desktop:
gunzip -c gateway_dev.tar.gz | docker load
```

---

## 8) Minimal Dockerfiles (pattern)

**`docker/gateway/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY services/gateway/pyproject.toml services/gateway/poetry.lock* ./
RUN pip install --no-cache-dir fastapi uvicorn pydantic[dotenv]
COPY services/gateway/ ./
EXPOSE 8000
CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker/orchestrator/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir langgraph langchain pydantic requests
COPY services/orchestrator/ ./
EXPOSE 8000
CMD ["uvicorn", "app:api", "--host","0.0.0.0","--port","8000"]
```

**`docker/tool-registry/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pluggy
COPY services/tool_registry/ ./
VOLUME ["/plugins"]
EXPOSE 8000
CMD ["uvicorn","app:api","--host","0.0.0.0","--port","8000"]
```

**`docker/rag/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir farm-haystack[faiss,preprocessing]==1.* qdrant-client sentence-transformers fastapi uvicorn
COPY services/rag/ ./
EXPOSE 8000
CMD ["uvicorn","app:api","--host","0.0.0.0","--port","8000"]
```

**`docker/asr/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir faster-whisper fastapi uvicorn
COPY services/asr/ ./
EXPOSE 8000
CMD ["uvicorn","app:api","--host","0.0.0.0","--port","8000"]
```

**`docker/mcp/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pydantic
COPY services/mcp/ ./
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn","app:api","--host","0.0.0.0","--port","8000"]
```

---

## 9) Non-Generative Conditioning (where it lives)

* Implement in **gateway-api**:

  * Parse prompt → extract entities/units.
  * Score **domain weights** (chem/mech/materials) via an **embedding-based classifier**.
  * Apply weights to:

    * **RAG retrieval** parameters (k, rerank emphasis).
    * **Prompt frame order** (constraints/units first).
  * Forward a **text-identical** prompt + **weighted evidence** to orchestrator.

This keeps the **text unchanged**, but **changes what evidence** and **what structure** the LLM sees.

---

## 10) Tool Auto-Discovery (Pluggy)

* **tool-registry** scans `/plugins` for Python entry points or YAML tool manifests.
* Expose a **/tools** endpoint (JSON schema) consumed by the orchestrator.
* Drop new tools (CLI wrappers, calculators, unit converters) into `services/plugins/` → **no IDE changes, no manual updates**.

---

## 11) MCP & RAG

* **mcp** serves domain corpora via HTTP; keep each domain in its own subpath (`/chem`, `/mech`), plus metadata for **units and safety constraints**.
* **rag-api** stores chunked passages in **qdrant**; supports:

  * **domain weighting** (vector field),
  * **reranking** (BM25 + embedding),
  * **unit/constraint tags** to prioritize “hard facts.”

---

## 12) Multimodal (ASR) Path

* **asr-api** accepts audio/video, extracts:

  * transcripts,
  * **segment tags** (technical vs narrative),
  * timestamps for citation.
* Orchestrator requests **technical segments only** for retrieval to reduce token use.

---

## 13) Dev→Prod Workflow (practical)

1. **Laptop**

   * Implement service stubs; `docker compose ... dev up`.
   * Use CPU quantized model via Ollama; run smoke tests (health, minimal RAG, ASR).
   * Add 2–3 **unit tests per service** (CI optional).

2. **Publish images**

   * Use **buildx** to push to GHCR (or save tarballs).

3. **Desktop**

   * Install NVIDIA toolkit; set `.env` with `ENABLE_GPU=true`, choose `VLLM_MODEL`.
   * `docker compose ... prod up`.
   * Load baseline indexes into **qdrant** (MCP ingesters).

4. **Point Cursor**

   * Set Cursor’s custom model endpoint to **gateway-api** (e.g., `http://DESKTOP:8080`).

---

## 14) Health, Logs, Safety

* Each service has `/health`.
* Centralize logs in `logs/` volume; add **structured logging** (JSON).
* Add **response validators** in gateway:

  * **Units normalization (SI)**,
  * Range checks,
  * Constraint-first templates.

---

## 15) Optional Niceties

* **Reverse proxy** (Caddy/Traefik) for TLS and nice hostnames.
* **Profiles** in Compose to toggle services (`--profile asr`, `--profile mcp`).
* **Tailscale/ZeroTier** to reach desktop from laptop across networks.

---

### Want me to turn this into a ready-to-run **Compose skeleton** (files + minimal service stubs), or generate a **Makefile** with common targets (`build-dev`, `push`, `up-prod`)?
