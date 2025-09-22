# Deployment Guide

This guide explains how to run the AI Stack on low-resource hardware (CPU-only, no LLM) and how to deploy to production with GPU.

## Environments

- Development (CPU, mock LLM): compose/dev + mock backend
- Production (GPU, vLLM): compose/prod + vLLM backend

## Prerequisites

- Docker + Docker Compose
- Make
- Optional: NVIDIA drivers + container toolkit (for GPU production)

## Quickstart (Low-resource hardware)

1) Create environment

```
cp env.example .env
```

Ensure in `.env`:

```
LLM_BACKEND=mock
ASR_DEVICE=cpu
```

2) Initialize folders

```
make setup
```

3) Start services

```
make up-dev
```

4) Verify

```
make health
```

- Gateway: http://localhost:8080/health
- Orchestrator: http://localhost:8081/health

5) Test endpoints

```
make test-chat
```

Mock responses will be returned without requiring an LLM.

## Monitoring (optional)

```
make monitoring-up
# Grafana http://localhost:3000 (admin/admin)
```

## Deploy to Production (GPU)

1) Prepare `.env`:

```
LLM_BACKEND=vllm
ASR_DEVICE=cuda
ENABLE_GPU=true
VLLM_MODEL=/models/<your-model>
```

2) Build & run

```
make build-prod
make up-prod
```

3) Push images to registry

```
make push-images REGISTRY=ghcr.io/<username>
```

## Data Management

- RAG data: `data/rag_cache` and Qdrant storage
- MCP domain data: `data/mcp`
- Logs: `logs/`

Use:

```
make init-data
make backup-data
```

## Troubleshooting

- If orchestrator/LLM is unavailable, the gateway will return a fallback message.
- Use `LLM_BACKEND=mock` to operate without a model.
- Check logs: `make logs`


