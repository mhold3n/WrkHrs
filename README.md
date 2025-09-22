# AI Stack: Laptop → Desktop GPU Deployment

A comprehensive AI stack with microservices architecture designed for **development on laptop** (CPU) and **production deployment on desktop** (GPU). Features non-generative conditioning, domain weighting, RAG, ASR, and plugin auto-discovery.

## 🏗️ Architecture Overview

```
[Cursor/IDE] → [gateway-api] → [orchestrator] → (routes to)
                           ├─► [tool-registry] (Pluggy)
                           ├─► [mcp-chem/mech/materials]
                           ├─► [rag-api] ↔ [qdrant] (vector DB)
                           ├─► [asr-api] (audio/video → text)
                           └─► [llm-runner] (Ollama/vLLM)
                                           ▲
                                 (GPU on desktop; CPU on laptop)
```

### Core Services

- **Gateway API**: FastAPI with non-generative conditioning, domain weighting, SI unit normalization
- **Orchestrator**: LangGraph-based workflow engine coordinating all services
- **Tool Registry**: Pluggy-based auto-discovery for Python/YAML/CLI tools
- **RAG API**: Haystack + Qdrant for domain-weighted document retrieval
- **ASR API**: Whisper-based speech recognition with technical segment extraction
- **MCP Services**: Multi-Context Protocol servers for chemistry, mechanical, materials domains
- **LLM Runner**: Ollama (dev) or vLLM (prod) for language model inference

## 🚀 Quick Start

### Prerequisites

**Development (Laptop):**
- Docker & Docker Compose
- Make
- curl, jq (for testing)

**Production (Desktop):**
- NVIDIA GPU with drivers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Adequate VRAM (8GB+ recommended)

### One-Command Setup

```bash
# Complete development environment setup
make dev-setup
```

This will:
1. Create `.env` from template
2. Build all services
3. Start development environment
4. Pull required models
5. Initialize sample data

## 📋 Manual Setup

### 1. Environment Configuration

```bash
# Create environment file
make setup

# Review and modify .env as needed
vim .env
```

Key configuration options in `.env`:

```bash
# Ports
GATEWAY_PORT=8080
RAG_PORT=8082
QDRANT_PORT=6333

# LLM Backend
LLM_BACKEND=ollama          # dev: ollama, prod: vllm
OLLAMA_MODEL=llama3:8b-instruct
VLLM_MODEL=/models/Mistral-7B-Instruct

# GPU (production)
ENABLE_GPU=false            # set to true for production

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ASR_MODEL=medium
```

### 2. Development Deployment

```bash
# Build and start development environment
make up-dev

# Check service health
make health

# View logs
make logs

# Test the system
make test-chat
```

### 3. Production Deployment

```bash
# Ensure GPU setup is complete
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi

# Deploy to production
make up-prod

# Verify GPU access
make logs-service SERVICE=llm-runner
```

## 🔧 Usage Examples

### Basic Chat Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the tensile strength of steel?"}
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
  }'
```

### Domain-Specific Queries

The gateway automatically detects domain keywords and weights responses:

```bash
# Chemistry query (auto-detected)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the pH of a 0.1M HCl solution?"}
    ]
  }'

# Mechanical engineering query
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Calculate the stress in a beam with 1000N force"}
    ]
  }'
```

### Document Upload to RAG

```bash
# Add document to knowledge base
curl -X POST http://localhost:8082/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Steel has a typical tensile strength of 400-500 MPa...",
    "domain": "materials",
    "source": "materials_handbook"
  }'

# Upload file
curl -X POST http://localhost:8082/documents/upload \
  -F "file=@technical_document.txt" \
  -F "domain=chemistry"
```

### Audio Transcription

```bash
# Transcribe audio file with technical segment extraction
curl -X POST http://localhost:8084/transcribe/file \
  -F "file=@lecture.mp3" \
  -F "extract_technical=true"
```

### Plugin Management

```bash
# List available plugins
make list-plugins

# Refresh plugin discovery
make refresh-plugins
```

## 🔌 Plugin Development

### Python Plugin Example

Create `services/plugins/calculator.py`:

```python
class CalculatorPlugin:
    def get_tool_info(self):
        return {
            "name": "calculator",
            "description": "Basic arithmetic calculator",
            "parameters": {
                "expression": {"type": "string", "description": "Math expression"}
            }
        }
    
    def execute_tool(self, parameters):
        expression = parameters.get("expression", "")
        try:
            result = eval(expression)  # Use safely in production
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
```

### YAML Plugin Example

Create `services/plugins/unit_converter.yaml`:

```yaml
tool:
  name: "unit_converter"
  description: "Convert between units"
  command: "units '{input_unit}' '{output_unit}'"
  parameters:
    input_unit:
      type: string
      description: "Input unit and value"
    output_unit:
      type: string
      description: "Target unit"
```

### CLI Plugin Example

Create `services/plugins/materials_lookup_cli.json`:

```json
{
  "name": "materials_lookup",
  "description": "Look up material properties",
  "command": "python /tools/materials_db.py --material {material} --property {property}",
  "parameters": {
    "material": {
      "type": "string",
      "description": "Material name"
    },
    "property": {
      "type": "string", 
      "description": "Property to lookup"
    }
  }
}
```

## 📊 Monitoring & Management

### Health Checking

```bash
# Check all services
make health

# Check specific service
curl http://localhost:8080/health | jq .
```

### Logging

```bash
# View all logs
make logs

# Service-specific logs
make logs-service SERVICE=gateway-api

# Follow logs in real-time
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.dev.yml logs -f gateway-api
```

### Resource Monitoring

```bash
# Real-time container stats
make monitor

# Container status
make status
```

### Data Management

```bash
# Initialize sample data
make init-data

# Backup Qdrant data
make backup-data

# Get collection info
curl http://localhost:8082/collections/info | jq .
```

## 🏭 Production Deployment

### Image Registry Workflow

```bash
# Build production images
make build-prod

# Push to registry
make push-images REGISTRY=ghcr.io/yourusername

# On production server
make deploy-prod
```

### GPU Configuration

Ensure production `.env` has:

```bash
ENABLE_GPU=true
LLM_BACKEND=vllm
VLLM_MODEL=/models/Mistral-7B-Instruct
ASR_DEVICE=cuda
```

### Resource Requirements

**Development:**
- 8GB+ RAM
- 2+ CPU cores
- 10GB disk space

**Production:**
- 16GB+ RAM
- 8GB+ VRAM (GPU)
- 4+ CPU cores
- 50GB+ disk space

## 🔍 Non-Generative Conditioning

The gateway implements sophisticated conditioning without changing the original prompt:

1. **Domain Detection**: Analyzes keywords to determine chemistry/mechanical/materials relevance
2. **Unit Extraction**: Identifies and normalizes SI units
3. **Constraint Detection**: Finds safety and operational constraints
4. **Evidence Weighting**: Retrieves domain-relevant context from RAG
5. **Enhanced Context**: Adds weighted evidence as system context while preserving user prompt

Example conditioning flow:
```
User: "What's the strength of steel in construction?"
↓
Domain Analysis: materials=0.8, mechanical=0.6, chemistry=0.2
↓
RAG Retrieval: Gets materials science documents
↓
System Context: "Domain weights: materials=0.8... Evidence: Steel strength ranges from 250-1000 MPa..."
↓
LLM receives: [System context] + [Original user prompt]
```

## 🛠️ Development Commands

```bash
# Development workflow
make dev-setup        # Complete setup
make build-dev        # Build development images
make up-dev          # Start development environment
make down            # Stop all services
make restart         # Restart all services

# Service management
make restart-service SERVICE=gateway-api
make shell SERVICE=gateway-api
make logs-service SERVICE=orchestrator

# Testing
make test            # Basic API tests
make test-chat       # Test chat endpoint
make health          # Health check all services

# Data operations
make init-data       # Add sample data
make backup-data     # Backup persistent data
make refresh-plugins # Reload plugins

# Cleanup
make clean          # Remove containers/networks
make clean-all      # Full cleanup including images
```

## 🐛 Troubleshooting

### Common Issues

**Service Won't Start:**
```bash
# Check logs for specific service
make logs-service SERVICE=gateway-api

# Check container status
make status

# Rebuild and restart
make build-service SERVICE=gateway-api
make restart-service SERVICE=gateway-api
```

**Model Loading Issues:**
```bash
# Check Ollama model
curl http://localhost:11434/api/tags

# Pull model manually
curl -X POST http://localhost:11434/api/pull \
  -d '{"name":"llama3:8b-instruct"}'

# Check vLLM (production)
curl http://localhost:8001/v1/models
```

**Health Check Failures:**
```bash
# Detailed health info
curl http://localhost:8080/health | jq .

# Check dependencies
curl http://localhost:6333/ready  # Qdrant
curl http://localhost:8082/health # RAG API
```

**GPU Issues (Production):**
```bash
# Verify NVIDIA setup
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi

# Check container GPU access
make logs-service SERVICE=llm-runner
make logs-service SERVICE=asr-api
```

## 📚 API Documentation

Once running, access interactive API documentation:

- **Gateway API**: http://localhost:8080/docs
- **Orchestrator**: http://localhost:8081/docs  
- **RAG API**: http://localhost:8082/docs
- **ASR API**: http://localhost:8084/docs
- **Tool Registry**: http://localhost:8086/docs
- **MCP API**: http://localhost:8085/docs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and test locally (`make dev-setup && make test`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check `/docs` directory for detailed guides

## 🗺️ Roadmap

- [ ] Web UI for system management
- [ ] Advanced plugin marketplace
- [ ] Multi-tenant support
- [ ] Kubernetes deployment manifests
- [ ] Advanced security features
- [ ] Performance optimization guides
- [ ] Additional domain support (biology, physics, etc.)

---

Built with ❤️ for the AI development community