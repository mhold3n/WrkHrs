# Usage Guide

## Services

- Gateway API: http://localhost:${GATEWAY_PORT}
- Orchestrator: http://localhost:${ORCH_PORT}
- RAG API: http://localhost:${RAG_PORT}
- MCP: http://localhost:${MCP_PORT}
- Tool Registry: http://localhost:${TOOLS_PORT}
- ASR API: http://localhost:${ASR_PORT}

## Auth

- API Key: set API_KEY_SECRET in .env and pass header x-api-key: <key> (if enabled)
- JWT: obtain via /auth/login

## Chat

POST http://localhost:${GATEWAY_PORT}/v1/chat/completions

```
{
  "messages": [
    {"role": "user", "content": "What is the yield strength of 1018 steel?"}
  ],
  "model": "any",
  "temperature": 0.2
}
```

If LLM is disabled, a mock response is returned.

## RAG

- Add document: POST /documents
- Search: POST /search

## MCP

- Chemistry: /chemistry/molecular_weight
- Mechanical: /mechanical/beam_calculation
- Materials: /materials/properties

## ASR

- File upload: POST /transcribe/file
- URL/Base64: POST /transcribe with { "audio_url": "..." } or { "audio_data": "data:audio/wav;base64,..." }


